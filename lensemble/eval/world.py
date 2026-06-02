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
