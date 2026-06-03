"""lensemble.eval.world — the minimal eval-world seam + an ``env_id`` registry (RFC-0005 §3, Open Q).

The harness must resolve ``env_id`` from config rather than hard-code an environment list (RFC-0005 Open
Question: the Stage-B suite is fixed at v0.2). The real environments come from ``stable-worldmodel``, which
is NOT vendored yet (deferred to issue #96); until then a deployment registers a LOCAL env factory. This
module defines the structural :class:`EvalWorld` seam every env satisfies and a process-level registry so
``evaluate`` resolves an id to a concrete world without importing the (absent) suite.

``EvalWorld`` carries only observation *clips*, actions, and a goal — the harness encodes those to latents
internally; nothing here crosses a trust boundary and the seam imposes no tensor on the report sink
(``INV-RESIDENCY`` is enforced structurally by :class:`~lensemble.eval.report.EvalReport`, not here).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

from lensemble.errors import EvaluationError, LensembleErrorCode

if TYPE_CHECKING:
    from torch import Tensor

    from lensemble.config import LensembleConfig
    from lensemble.contracts import ActionSpec

# A factory builds a concrete world from the resolved config (the env reads its own knobs from cfg).
EvalWorldFactory = Callable[["LensembleConfig"], "EvalWorld"]

_STABLE_WORLDMODEL_PREFIX = "stable-worldmodel://"

# The built-in toy env id (#167): a deterministic CPU world that lets `evaluate` produce a real EvalReport
# for a green end-to-end run WITHOUT the unvendored stable-worldmodel suite (#96).
SYNTHETIC_TOY_ENV_ID = "synthetic://toy"


@runtime_checkable
class EvalWorld(Protocol):
    """The structural contract a latent-MPC eval environment satisfies (RFC-0005 §3).

    ``action_spec`` is the embodiment's :class:`~lensemble.contracts.ActionSpec`; ``reset(seed)`` returns
    an observation clip ``(T, C, Hpx, Wpx)`` matching the encoder config; ``goal()`` returns the goal
    observation clip; ``step(action)`` advances and returns the next observation clip; ``succeeded()``
    reports whether the current state is within the goal tolerance. Deterministic given the seed
    (best-effort, seed-pinned; conventions 9).
    """

    action_spec: "ActionSpec"

    def reset(self, seed: int) -> "Tensor":
        """Reset to a seeded initial state; return its observation clip ``(T, C, Hpx, Wpx)``."""
        ...

    def goal(self) -> "Tensor":
        """Return the goal observation clip ``(T, C, Hpx, Wpx)`` (the planner's target latent source)."""
        ...

    def step(self, action: "Tensor") -> "Tensor":
        """Apply one action; return the next observation clip ``(T, C, Hpx, Wpx)``."""
        ...

    def succeeded(self) -> bool:
        """``True`` iff the current state is within the goal tolerance (the per-episode outcome)."""
        ...


# Process-level registry: env_id -> factory. Mutated only via register_env (idempotent re-register).
_REGISTRY: dict[str, EvalWorldFactory] = {}


def register_env(env_id: str, factory: EvalWorldFactory) -> None:
    """Register an eval-world ``factory`` under ``env_id`` (RFC-0005 §3; the local-env seam).

    Lets a deployment (or a test) supply a LOCAL :class:`EvalWorld` without the unvendored
    ``stable-worldmodel`` suite (#96). Re-registering an id replaces its factory (so tests can rebind a
    deterministic stub).
    """
    _REGISTRY[env_id] = factory


def resolve_env(env_id: str, *, cfg: "LensembleConfig") -> EvalWorld:
    """Resolve ``env_id`` to a concrete :class:`EvalWorld`, building it from ``cfg`` (RFC-0005 §3).

    Resolution order:

    1. A registered ``env_id`` -> call its factory with ``cfg``.
    2. A ``stable-worldmodel://`` id -> attempt to import ``stable_worldmodel``; on ``ImportError`` raise
       :class:`~lensemble.errors.EvaluationError` (the suite is not vendored yet, #96).
    3. Otherwise -> raise :class:`~lensemble.errors.EvaluationError` (an unknown id).

    Fail-closed: an unresolvable id never returns a partial/None world.
    """
    factory = _REGISTRY.get(env_id)
    if factory is not None:
        return factory(cfg)
    if env_id.startswith(_STABLE_WORLDMODEL_PREFIX):
        try:
            import stable_worldmodel  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise EvaluationError(
                f"cannot resolve {env_id!r}: stable-worldmodel is not importable",
                code=LensembleErrorCode.EVALUATION_FAILED,
                remediation="stable-worldmodel is not vendored yet (issue #96); register a local env or "
                "vendor the suite",
            ) from exc
        raise EvaluationError(  # pragma: no cover - the suite is unvendored in CI (#96)
            f"stable-worldmodel is importable but {env_id!r} is not wired to a factory yet",
            code=LensembleErrorCode.EVALUATION_FAILED,
            remediation="wire the stable-worldmodel suite to register_env (issue #96)",
        )
    raise EvaluationError(
        f"unknown env_id {env_id!r}",
        code=LensembleErrorCode.EVALUATION_FAILED,
        remediation="register it with register_env, or use a stable-worldmodel:// id",
    )


# --- the built-in synthetic://toy world (#167): a deterministic CPU EvalWorld for a green e2e run ---


class _ToyWorld:
    """A deterministic CPU :class:`EvalWorld` with closed-form clips and a KNOWN, non-trivial success rate.

    The built-in toy world (#167): it lets ``evaluate`` produce a real :class:`~lensemble.eval.report.EvalReport`
    for an end-to-end green run WITHOUT the unvendored ``stable-worldmodel`` suite (#96). It reads its clip
    shape from ``cfg.model`` (``num_frames``, ``in_channels``, ``image_size``) so the clips
    ``(num_frames, in_channels, image_size, image_size)`` match the encoder the harness reconstructs (the
    harness does ``encoder(obs.unsqueeze(0))``, batching one clip). ``reset(seed)``/``goal()``/``step`` are
    seeded so a run is reproducible (conventions 9), and ``action_spec`` is continuous dim 2.

    ``succeeded()`` is rigged by the reset seed's PARITY: an even reset seed succeeds, an odd one does not.
    Over the harness's seed-pinned episodes (seeds ``root_seed + i`` for ``i in range(4)``) at the default
    ``root_seed == 0`` this yields exactly two successes → ``success_rate == 0.5``, a KNOWN non-trivial
    value (NOT always-0 / always-1), so the e2e success-rate assertion is non-vacuous.
    """

    action_spec: "ActionSpec"

    def __init__(self, cfg: "LensembleConfig") -> None:
        from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec

        m = cfg.model
        self._num_frames = int(getattr(m, "num_frames"))
        self._in_channels = int(getattr(m, "in_channels", 3))
        self._image_size = int(getattr(m, "image_size"))
        self.action_spec = ActionSpec(
            embodiment_id="synthetic-toy",
            kind=ActionKind.CONTINUOUS,
            dim=2,
            low=(-1.0, -1.0),
            high=(1.0, 1.0),
            num_classes=None,
            units=("u", "u"),
            wmcp_version=WMCP_VERSION,
        )
        self._seed = 0
        self._steps = 0

    def _clip(self, seed: int) -> "Tensor":
        import torch

        gen = torch.Generator().manual_seed(seed)
        return torch.randn(
            self._num_frames,
            self._in_channels,
            self._image_size,
            self._image_size,
            generator=gen,
        )

    def reset(self, seed: int) -> "Tensor":
        self._seed = seed
        self._steps = 0
        return self._clip(seed)

    def goal(self) -> "Tensor":
        return self._clip(
            7919
        )  # a fixed goal clip (a prime, distinct from the episode seeds)

    def step(self, action: "Tensor") -> "Tensor":
        self._steps += 1
        return self._clip(self._seed + self._steps)

    def succeeded(self) -> bool:
        # Rigged by reset-seed parity → 0.5 over the harness's four consecutive seeds (a KNOWN non-trivial
        # rate; deterministic, never all-0). Independent of the planned actions (the toy world has no real
        # dynamics) so the success rate is stable across runs.
        return self._seed % 2 == 0


def _toy_factory(cfg: "LensembleConfig") -> "_ToyWorld":
    """Build the deterministic ``synthetic://toy`` world from ``cfg`` (the registered factory, #167)."""
    return _ToyWorld(cfg)


# Self-register the built-in toy env at import (mirrors the data adapters' self-registration), so a bare
# `evaluate(..., env_id="synthetic://toy")` resolves without any deployment wiring (#167).
register_env(SYNTHETIC_TOY_ENV_ID, _toy_factory)
