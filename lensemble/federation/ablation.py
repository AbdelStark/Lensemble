"""lensemble.federation.ablation — the ablation-ladder DRIVER (RFC-0005 §6, the core experiment; #55).

The federated-runtime driver that runs each ladder rung through the live multi-round simulation harness
and reduces it to a per-rung record. It lives in ``federation`` (band L7) because it drives the
:class:`~lensemble.federation.coordinator.Coordinator` / :class:`~lensemble.federation.participant.Participant`
runtime; the EVAL-SIDE pieces it consumes — the ordered :data:`~lensemble.eval.ablation.LADDER_RUNGS`
table, the frozen :class:`~lensemble.eval.ablation.RungReport`, the rung-config composition
(:func:`~lensemble.eval.ablation.compose_rung`), and the Layer-4 distillation
(:func:`~lensemble.eval.ablation.run_distillation`) — live one band DOWN in ``lensemble.eval.ablation``
(``eval`` may not depend on ``federation``, RFC-0001 §3), so this module depends downward onto them and the
banded import DAG stays acyclic. This mirrors ``lensemble.eval.baselines`` (compose + reduce) vs the
``Coordinator`` (run): eval composes the rung table, the runtime drives it.

The ladder realizes the RFC-0002 §4 gauge fix additively (one mechanism per rung): ``naive-fedavg``
(negative control) → ``shared-sketch`` (Layer 1) → ``procrustes-backstop`` (Layer 3) → ``frame-anchor``
(Layer 2, the recommended config) → ``distillation`` (Layer 4). Each rung is driven over genuinely
DIFFERENT per-silo data so the naive frames actually drift, and all three metric families are reported at
each rung (frame drift §2, MPC success §3, effective dim §4). The expected qualitative ordering is naive
worst on drift; anchored flat (RFC-0005 §6).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lensemble.eval.ablation import (
    LADDER_RUNGS,
    RungReport,
    cleanup_rung,
    compose_rung,
    run_distillation,
)
from lensemble.federation.simulation import run_federated_simulation

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from lensemble.config.schema import LensembleConfig
    from lensemble.federation.simulation import SiloData

__all__ = ["run_ablation_ladder"]


def run_ablation_ladder(
    base_cfg: "LensembleConfig",
    participants_data: "Sequence[SiloData]",
    *,
    num_rounds: int,
    rung_names: "Sequence[str] | None" = None,
) -> dict[str, RungReport]:
    """Run every ladder rung through the live federated-simulation harness; return per-rung records (#55).

    For each rung in :data:`~lensemble.eval.ablation.LADDER_RUNGS` (the RFC-0002 §4 additive rollout):
    compose its config + pinned probe + backstop reference (:func:`~lensemble.eval.ablation.compose_rung`),
    run :func:`~lensemble.federation.simulation.run_federated_simulation` (with the Layer-3 backstop wired
    per the rung flag), and reduce its per-round metrics to a :class:`~lensemble.eval.ablation.RungReport`
    (frame drift §2, MPC success §3, effective dim §4). The Layer-4 rung additionally runs the
    gauge-invariant function-space distillation consensus on the final-round frames
    (:func:`~lensemble.eval.ablation.run_distillation`) — the top-rung mechanism; the consensus does not
    change the reported drift (the drift is the pre-consensus per-silo signal, RFC-0002 §6).

    ``rung_names`` optionally restricts the run to a SUBSET of the ladder (in ladder order, ignoring any
    unknown name): the RFC-0005 §7 sweeps (#56) run MANY ladder points, so they drive only the load-bearing
    ``naive-fedavg`` + ``frame-anchor`` rungs per point to stay CPU-fast, while the full ladder run (the §6
    central experiment) passes ``None`` to run all five. Composing/running each rung is independent of the
    others, so a subset is a faithful slice of the full ladder.

    The per-rung frame-drift figure is the MEAN inter-silo rotation angle across rounds — the stable,
    non-flaky reduction that smooths the round-to-round variance of the toy outer loop, so the qualitative
    ordering (naive worst, anchored flat) is a robust comparison rather than a single noisy last-round
    sample (success/effective-dim are reported from the last round). Returns ``{rung_name -> RungReport}``
    in ladder order. Each rung pins its own throwaway probe dir (cleaned up after the run) and the harness
    cleans up each Coordinator's ``tempfile.mkdtemp`` artifacts dir, so the multi-rung run leaks no temp dir.
    """
    selected = None if rung_names is None else set(rung_names)
    reports: dict[str, RungReport] = {}
    for name, spec in LADDER_RUNGS:
        if selected is not None and name not in selected:
            continue
        comp = compose_rung(base_cfg, spec)
        try:
            result = run_federated_simulation(
                participants_data,
                cfg=comp.cfg,
                num_rounds=num_rounds,
                backstop=spec.backstop,
                reference_embeddings=comp.reference_frame,
            )
            if spec.distill:
                # Layer 4 (RFC-0002 §6): the gauge-invariant function-space consensus on the final frames.
                run_distillation(result.final_frames)
            angle = _mean(r.frame_drift_angle_deg for r in result.per_round)
            residual = _mean(r.frame_drift_residual for r in result.per_round)
            reports[name] = RungReport(
                frame_drift_residual=residual,
                frame_drift_angle_deg=angle,
                success_rate=result.per_round[-1].success_rate,
                effective_dim=result.per_round[-1].effective_dim,
            )
        finally:
            cleanup_rung(comp)
    return reports


def _mean(values: "Iterable[float]") -> float:
    """The arithmetic mean of an iterable of floats (empty -> 0.0)."""
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
