"""lensemble.federation.sweeps — the RFC-0005 §7 sweep DRIVERS over the §6 ladder (#56).

The federated-runtime drivers that run the §7 robustness sweeps (Claim 4: the recipe holds across
heterogeneity and scale) OVER the §6 ablation-ladder rungs, REUSING #55's runner
(:func:`lensemble.federation.ablation.run_ablation_ladder`) and harness
(:func:`lensemble.federation.simulation.run_federated_simulation`). They live in ``federation`` (band L7)
because they drive the runtime; the eval-side pieces they consume — the synthetic non-IID partition
(:func:`lensemble.eval.sweeps.partition_synthetic_noniid`) and the seeded drift-pair sampler
(:func:`lensemble.eval.sweeps.sample_drift_pairs`) — live one band DOWN in :mod:`lensemble.eval.sweeps`
(``eval`` may not depend on ``federation``, RFC-0001 §3), so this module depends downward onto them. This
mirrors the #55 split (:mod:`lensemble.eval.ablation` composes; :mod:`lensemble.federation.ablation` drives).

The three §7 axes:

* **Non-IID severity** (:func:`non_iid_severity_sweep`): partition the synthetic per-silo data by a
  per-silo distribution shift scaled by the severity ``s in [0, 1]`` and run the ladder at each ``s``;
  report the per-rung drift-degradation curve vs severity. The REAL factors-of-variation partition is the
  deferred seam (#96) — a ``factor`` other than ``"synthetic"`` fail-closes (the eval-side partition raises
  :class:`~lensemble.errors.EvaluationError`).
* **Participant count C / inner horizon H** (:func:`participant_horizon_sweep`): vary
  ``federation.participant_count`` and ``federation.inner_horizon`` and run the ladder at each ``(C, H)``;
  a longer ``H`` rotates the per-silo frames further apart before the outer step (RFC-0002 §2.1), so the
  naive drift grows with ``H``.
* **Scale** (:func:`scale_sweep`): repeat the key rungs at increasing ``model.latent_dim`` (each a coherent
  ViT shape via the #166 bridge: ``num_heads`` divides ``latent_dim``; ``num_tokens`` is independent of it)
  to show the recipe holds as the encoder grows.

All three keep the dims/rounds/silos tiny (CPU-fast) and run ONLY the load-bearing ``naive-fedavg`` +
``frame-anchor`` rungs per swept point (:data:`_SWEEP_RUNGS`) — the directional claim each axis asserts is
naive-vs-anchored, so the two bracketing rungs suffice and the many-point sweep stays fast. Temp-dir-safe:
the harness rmtree's each Coordinator's per-run artifacts dir (#55), so a many-point sweep leaks no temp dir.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, TypedDict

from lensemble.eval.sweeps import SYNTHETIC_FACTOR, partition_synthetic_noniid
from lensemble.federation.ablation import run_ablation_ladder
from lensemble.federation.simulation import SiloData

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lensemble.config.schema import LensembleConfig
    from lensemble.contracts import ActionSpec
    from lensemble.eval.ablation import RungReport
    from lensemble.eval.sweeps import SiloPartition


class _ClipShape(TypedDict):
    """The synthetic-partition clip-shape kwargs (so ``**shape`` keeps a precise, non-``str`` type)."""

    num_frames: int
    in_channels: int
    image_size: int
    action_dim: int


__all__ = [
    "non_iid_severity_sweep",
    "participant_horizon_sweep",
    "scale_sweep",
]

# The two load-bearing rungs every sweep point runs: the negative control + the recommended config. The
# directional claim each §7 axis asserts is naive-vs-anchored, so the bracketing rungs suffice and the
# many-point sweep stays CPU-fast (the full five-rung ladder is the §6 central experiment, #55).
_SWEEP_RUNGS: tuple[str, ...] = ("naive-fedavg", "frame-anchor")


def non_iid_severity_sweep(
    base_cfg: "LensembleConfig",
    severities: "Sequence[float]",
    *,
    num_silos: int,
    num_rounds: int,
    seed: int = 0,
    factor: str = SYNTHETIC_FACTOR,
) -> dict[float, dict[str, "RungReport"]]:
    """Run the ladder at each non-IID ``severity``; return the per-rung drift curve vs severity (§7).

    For each ``s`` in ``severities`` (``0`` = near-IID, all silos share the synthetic distribution; ``1`` =
    strongly non-IID, each silo's mean is shifted by its per-silo factor scaled by ``s``), partition the
    synthetic per-silo toy data (:func:`~lensemble.eval.sweeps.partition_synthetic_noniid`) and run the
    bracketing ladder rungs (:func:`~lensemble.federation.ablation.run_ablation_ladder`). Returns
    ``{severity -> {rung_name -> RungReport}}`` — the per-rung drift-degradation curve vs severity. The
    EXPECTED TREND (RFC-0005 §7 / Claim 4): the naive rung's inter-silo frame drift GROWS with severity (the
    per-silo distribution shift pulls the unconstrained frames apart), while the anchored rung stays low
    (the Variant-A anchor pins each frame onto the round-0 reference regardless of the shift).

    The REAL factors-of-variation partition is the deferred seam (#96): a ``factor`` other than
    ``"synthetic"`` fail-closes (the eval-side partition raises :class:`~lensemble.errors.EvaluationError`,
    propagated here). The clip shapes are read from ``base_cfg.model`` so the synthetic windows match the
    encoder; everything is seeded so the swept curve is reproducible.
    """
    shape = _clip_shape(base_cfg)
    curve: dict[float, dict[str, RungReport]] = {}
    for s in severities:
        silos = partition_synthetic_noniid(
            num_silos,
            severity=float(s),
            seed=seed,
            action_spec=_sweep_action_spec(),
            factor=factor,
            **shape,
        )
        cfg = _with_participant_count(base_cfg, num_silos)
        curve[float(s)] = run_ablation_ladder(
            cfg,
            _to_silo_data(silos),
            num_rounds=num_rounds,
            rung_names=_SWEEP_RUNGS,
        )
    return curve


def participant_horizon_sweep(
    base_cfg: "LensembleConfig",
    *,
    counts: "Sequence[int]",
    horizons: "Sequence[int]",
    num_rounds: int,
    seed: int = 0,
) -> dict[tuple[int, int], dict[str, "RungReport"]]:
    """Run the ladder across the ``(C, H)`` grid; return the per-rung drift per point (§7).

    For each ``(C, H)`` in ``counts x horizons``: set ``federation.participant_count = C`` (and the quorum
    to match so the round CLOSES) and ``federation.inner_horizon = H``, partition ``C`` strongly-non-IID
    synthetic silos, and run the bracketing ladder rungs. Returns ``{(C, H) -> {rung_name -> RungReport}}``.
    The EXPECTED TREND (RFC-0002 §2.1 / RFC-0005 §7): a longer inner horizon ``H`` rotates the per-silo
    frames further apart before the outer step, so the NAIVE rung's inter-silo drift grows with ``H``;
    varying ``C`` characterizes the DiLoCo robustness in the JEPA setting. Keep ``C``/``H`` tiny (CPU-fast).
    """
    shape = _clip_shape(base_cfg)
    grid: dict[tuple[int, int], dict[str, RungReport]] = {}
    for c in counts:
        silos = partition_synthetic_noniid(
            int(c),
            severity=1.0,
            seed=seed,
            action_spec=_sweep_action_spec(),
            **shape,
        )
        silo_data = _to_silo_data(silos)
        for h in horizons:
            cfg = _with_participant_count(base_cfg, int(c))
            cfg = dataclasses.replace(
                cfg,
                federation=dataclasses.replace(cfg.federation, inner_horizon=int(h)),
            )
            grid[(int(c), int(h))] = run_ablation_ladder(
                cfg, silo_data, num_rounds=num_rounds, rung_names=_SWEEP_RUNGS
            )
    return grid


def scale_sweep(
    base_cfg: "LensembleConfig",
    *,
    latent_dims: "Sequence[int]",
    num_rounds: int,
    seed: int = 0,
    num_silos: int = 2,
) -> dict[int, dict[str, "RungReport"]]:
    """Repeat the key rungs at increasing ``model.latent_dim``; return the per-rung drift per scale (§7).

    For each ``latent_dim`` in ``latent_dims``: set ``model.latent_dim`` (and the coupled widths
    ``d``/``cond_dim``/``predictor_width`` so the ViT shape stays coherent via the #166 bridge —
    ``num_heads`` already divides each tiny dim and ``num_tokens`` is independent of ``latent_dim``),
    partition the synthetic silos, and run the bracketing ladder rungs. Returns
    ``{latent_dim -> {rung_name -> RungReport}}``. The recipe HOLDS as the encoder grows: at every scale the
    anchored rung's inter-silo drift stays below the naive rung's (the anchor pins the frame regardless of
    the encoder width). Keep the dims small (e.g. ``8 -> 16``) so it is CPU-fast.
    """
    shape = _clip_shape(base_cfg)
    out: dict[int, dict[str, RungReport]] = {}
    for dim in latent_dims:
        cfg = _with_latent_dim(_with_participant_count(base_cfg, num_silos), int(dim))
        silos = partition_synthetic_noniid(
            num_silos,
            severity=1.0,
            seed=seed,
            action_spec=_sweep_action_spec(),
            **shape,
        )
        out[int(dim)] = run_ablation_ladder(
            cfg,
            _to_silo_data(silos),
            num_rounds=num_rounds,
            rung_names=_SWEEP_RUNGS,
        )
    return out


# --- internals: the eval->federation SiloPartition -> SiloData map + the cfg overrides ---


def _to_silo_data(silos: "Sequence[SiloPartition]") -> list[SiloData]:
    """Map each eval-side :class:`~lensemble.eval.sweeps.SiloPartition` to a band-L7 ``SiloData``.

    The compose side returns the layer-neutral :class:`SiloPartition` (it may not name the band-L7
    ``SiloData``); the driver lifts each into the harness's ``SiloData`` here (a structural copy, no data
    transformation — the raw windows still never cross the transport, ``INV-RESIDENCY``).
    """
    return [
        SiloData(
            participant_id=s.participant_id,
            windows=s.windows,
            action_spec=s.action_spec,
            dataset_root=s.dataset_root,
        )
        for s in silos
    ]


def _sweep_action_spec() -> "ActionSpec":
    """The continuous toy :class:`~lensemble.contracts.ActionSpec` the synthetic silos share (dim 2)."""
    from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec

    return ActionSpec(
        embodiment_id="toy",
        kind=ActionKind.CONTINUOUS,
        dim=2,
        low=(-1.0, -1.0),
        high=(1.0, 1.0),
        num_classes=None,
        units=("u", "u"),
        wmcp_version=WMCP_VERSION,
    )


def _clip_shape(cfg: "LensembleConfig") -> _ClipShape:
    """Read the synthetic-partition clip shape (frames/channels/image/action) from ``cfg.model``."""
    m = cfg.model
    return _ClipShape(
        num_frames=int(getattr(m, "num_frames")),
        in_channels=int(getattr(m, "in_channels", 3)),
        image_size=int(getattr(m, "image_size")),
        action_dim=2,
    )


def _with_participant_count(cfg: "LensembleConfig", count: int) -> "LensembleConfig":
    """Set ``federation.participant_count = count`` with the quorum clamped to match (so the round CLOSES).

    The round quorum is ``K = max(fault_tolerance_min_participants, secure_agg_threshold)`` and must be
    ``<= C`` AND match the silo count for ``try_round`` to reach ``CLOSED`` (validate_config / RFC-0013 §3).
    The sweep supplies exactly ``count`` silos, so the quorum is pinned to ``count``.
    """
    federation = dataclasses.replace(
        cfg.federation,
        participant_count=count,
        fault_tolerance_min_participants=count,
        secure_agg_threshold=count,
    )
    return dataclasses.replace(cfg, federation=federation)


def _with_latent_dim(cfg: "LensembleConfig", dim: int) -> "LensembleConfig":
    """Set ``model.latent_dim`` and the coupled widths so the ViT shape stays coherent (the #166 bridge).

    ``d`` / ``cond_dim`` / ``predictor_width`` track ``latent_dim`` (build_encoder/build_predictor read the
    same hidden dim); ``num_heads`` and ``num_tokens`` are left as the base config's (the tiny ``num_heads``
    already divides each swept dim and ``num_tokens`` is a function of the patching geometry, not the hidden
    dim). The result resolves to a distinct, validated :class:`~lensemble.config.schema.LensembleConfig`.
    """
    m = cfg.model
    model = dataclasses.replace(
        m,
        latent_dim=dim,
        d=dim,
        cond_dim=dim,
        predictor_width=dim,
    )
    return dataclasses.replace(cfg, model=model)
