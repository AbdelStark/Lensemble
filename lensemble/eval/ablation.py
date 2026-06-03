"""lensemble.eval.ablation — the ablation-ladder definition + config composition (RFC-0005 §6).

The ladder *is* the additive rollout of the RFC-0002 §4 gauge fix: rung 1 is the negative control and
each subsequent rung adds exactly ONE mechanism. This module owns the EVAL-SIDE pieces — the ordered
:data:`LADDER_RUNGS` table, the frozen per-rung :class:`RungReport`, the rung-config composition, and the
``lambda_anc`` sweep — exactly as ``lensemble.eval.baselines`` composes the bracketing baselines without
running them. **Driving** each rung through the live federated-simulation harness is the federated runtime's
job (band L7 > L6, RFC-0001 §3): the driver :func:`lensemble.federation.run_ablation_ladder` consumes this
module's rung table + report shape and the harness (``eval`` may not depend on ``federation``, so the
driver lives one band up and depends DOWN onto this module).

The five rungs → RFC-0002 §4 layer mapping (one mechanism added per rung):

============================  ==========================================  ====================
Rung                          Mechanism added                              RFC-0002 layer
============================  ==========================================  ====================
1. ``naive-fedavg``           — (negative control: FedAvg, no gauge ctrl)  none (§2.1 failure)
2. ``shared-sketch``          + shared sketch matrix ``A`` (lambda_sig>0)   Layer 1 (§3)
3. ``procrustes-backstop``    + Procrustes align-then-average backstop      Layer 3 (§5)
4. ``frame-anchor``           + frame-anchor loss (lambda_anc>0) ← REC      Layer 2 (§4)
5. ``distillation``           + function-space distillation                Layer 4 (§6)
============================  ==========================================  ====================

Each rung is a small ``RungSpec`` — config overrides (the ``lambda_sig``/``lambda_anc`` knobs, the
``anchor_variant``) plus the mechanism-enablement flags the runner reads (``backstop`` wires the #18
Layer-3 seam ON in the harness's coordinator; ``distill`` runs the Layer-4 consensus on the final-round
frames). The ladder is an in-code ``LADDER_RUNGS`` table the runner composes — a ``configs/ladder/`` Hydra
group is optional and the in-code table is cleaner for the CPU regression test (RFC-0009 config composition
is still the mechanism: each rung and each ``lambda_anc`` sweep value resolves to a distinct, validated
:class:`~lensemble.config.schema.LensembleConfig`).

The per-rung record (:class:`RungReport`, frozen): ``frame_drift_residual`` / ``frame_drift_angle_deg``
(§2), ``success_rate`` (§3), ``effective_dim`` (§4) — the shape every rung emits.

The lambda_anc sweep (RFC-0002 §7, the central hyperparameter): :func:`lambda_anc_sweep` resolves each
swept value to a DISTINCT valid config-group override. The "pin frame, not content" sweet spot is a SMALL
positive ``lambda_anc``: too high (``>> 1``) clamps the encoder to the reference frame AND its quality;
too low (``-> 0``) lets the frame drift and averaging degrades (RFC-0002 §7). Warm-start + a small
``lambda_anc`` keeps frames pinned cheaply, so Layers 3-4 rarely fire — the hypothesis the sweep
characterizes against the frame-drift diagnostic and MPC success in Stage B.
"""

from __future__ import annotations

import dataclasses
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from torch import Tensor

    from lensemble.config.schema import LensembleConfig
    from lensemble.model.encoder import ReferenceEncoder

__all__ = [
    "LADDER_RUNGS",
    "RungReport",
    "RungSpec",
    "RungComposition",
    "compose_rung",
    "cleanup_rung",
    "run_distillation",
    "lambda_anc_sweep",
]


class RungReport(BaseModel):
    """One ladder rung's per-rung record — the three metric families (RFC-0005 §6, §2-4).

    ``frame_drift_angle_deg`` is the mean inter-silo rotation angle on the probe (the headline figure,
    §2), ``frame_drift_residual`` the mean optimal-Procrustes residual, ``success_rate`` the downstream
    MPC planning success (§3), and ``effective_dim`` the participation-ratio collapse guard over the
    committed global frame (§4). Frozen: a reported rung record is immutable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_drift_residual: float
    frame_drift_angle_deg: float
    success_rate: float
    effective_dim: float


@dataclass(frozen=True)
class RungSpec:
    """A ladder rung: the objective-knob overrides + the mechanism-enablement flags the runner reads.

    ``lambda_sig`` / ``lambda_anc`` set the SIGReg sketch and frame-anchor weights (the composed
    :class:`~lensemble.config.schema.ObjectiveConfig`); ``backstop`` wires the #18 Layer-3 Procrustes seam
    ON in the harness's coordinator; ``distill`` runs the Layer-4 function-space consensus on the
    final-round frames. ``anchored`` records whether the rung pins a real probe (``lambda_anc > 0`` needs a
    pinned probe, RFC-0004 / validate_config).
    """

    name: str
    lambda_sig: float
    lambda_anc: float
    backstop: bool
    distill: bool

    @property
    def anchored(self) -> bool:
        return self.lambda_anc > 0.0


# The ordered five-rung ladder (RFC-0002 §4 additive rollout). ``lambda_sig`` toggles Layer 1 (the shared
# sketch); ``lambda_anc`` toggles Layer 2 (the frame anchor); ``backstop`` toggles Layer 3; ``distill``
# toggles Layer 4. Each rung adds exactly one mechanism over its predecessor.
_LAMBDA_SIG_ON = 0.1  # the shared-sketch SIGReg weight when Layer 1 is on (cfg default)
_LAMBDA_ANC_ON = (
    50.0  # a strong anchor so the frame is visibly pinned on the toy CPU budget (§7)
)

_RUNGS: tuple[RungSpec, ...] = (
    RungSpec(
        "naive-fedavg", lambda_sig=0.0, lambda_anc=0.0, backstop=False, distill=False
    ),
    RungSpec(
        "shared-sketch",
        lambda_sig=_LAMBDA_SIG_ON,
        lambda_anc=0.0,
        backstop=False,
        distill=False,
    ),
    RungSpec(
        "procrustes-backstop",
        lambda_sig=_LAMBDA_SIG_ON,
        lambda_anc=0.0,
        backstop=True,
        distill=False,
    ),
    RungSpec(
        "frame-anchor",
        lambda_sig=_LAMBDA_SIG_ON,
        lambda_anc=_LAMBDA_ANC_ON,
        backstop=True,
        distill=False,
    ),
    RungSpec(
        "distillation",
        lambda_sig=_LAMBDA_SIG_ON,
        lambda_anc=_LAMBDA_ANC_ON,
        backstop=True,
        distill=True,
    ),
)

# The public (name -> spec-as-tuple) view of the ladder: the ordered rung names + their flags, for the
# runner's keys and for callers/tests that introspect the ladder without depending on the private spec
# dataclass shape.
LADDER_RUNGS: tuple[tuple[str, RungSpec], ...] = tuple((r.name, r) for r in _RUNGS)


@dataclass(frozen=True)
class RungComposition:
    """A composed ladder rung ready to drive: its config, the pinned-probe dir, and the backstop ``E_ref``.

    ``cfg`` is the rung's validated :class:`~lensemble.config.schema.LensembleConfig`; ``probe_dir`` is the
    throwaway dir its pinned probe was written to (``None`` when the rung uses the coordinator's
    placeholder-hash probe — rungs that neither anchor nor backstop); ``reference_frame`` is the round-0
    ``E_ref = f_ref(P)`` ``(k*N, d)`` for the Layer-3 backstop (``None`` when the rung does not backstop).
    The federated driver (band L7) runs the harness over this and cleans up ``probe_dir`` afterwards.
    """

    spec: RungSpec
    cfg: "LensembleConfig"
    probe_dir: "Path | None"
    reference_frame: "Tensor | None"


def compose_rung(base_cfg: "LensembleConfig", spec: RungSpec) -> RungComposition:
    """Compose ONE ladder rung — the config + pinned probe + backstop reference (the eval-side step, #55).

    Sets the rung's objective knobs (``lambda_sig`` / ``lambda_anc`` / the Variant-A anchor), pins a real
    ``k >= d`` landmark probe with round-0 ``f_ref`` targets when the rung anchors OR backstops, and builds
    the Layer-3 backstop reference ``E_ref = f_ref(P)`` for the backstop rungs. Returns a
    :class:`RungComposition` the federated driver (:func:`lensemble.federation.run_ablation_ladder`) runs
    through the harness. This is the EVAL-SIDE composition (no ``federation`` dependency, RFC-0001 §3): the
    driver one band up consumes it.
    """
    cfg, probe_dir = _compose_rung_cfg(base_cfg, spec)
    reference = _reference_frame(cfg) if spec.backstop else None
    return RungComposition(
        spec=spec, cfg=cfg, probe_dir=probe_dir, reference_frame=reference
    )


def cleanup_rung(composition: RungComposition) -> None:
    """Remove a composed rung's throwaway pinned-probe dir (the driver calls this after running the rung)."""
    if composition.probe_dir is not None:
        _cleanup_dir(composition.probe_dir)


def lambda_anc_sweep(
    base_cfg: "LensembleConfig", values: "Sequence[float]"
) -> dict[float, "LensembleConfig"]:
    """Resolve each ``lambda_anc`` value to a DISTINCT valid config (the RFC-0002 §7 central sweep).

    Each value is a config-group override: a fresh :class:`~lensemble.config.schema.LensembleConfig` with
    ``objective.lambda_anc`` set to that value (every other field inherited from ``base_cfg``). The sweep
    characterizes the central hyperparameter against the frame-drift diagnostic and MPC success — the "pin
    frame, not content" sweet spot is a SMALL positive value: too high (``>> 1``) clamps the encoder to the
    reference frame AND its quality, too low (``-> 0``) lets the frame drift and averaging degrades
    (RFC-0002 §7).

    Returns ``{lambda_anc_value -> LensembleConfig}``; the configs are distinct objects with distinct
    ``objective.lambda_anc``. The caller drives each through the ladder harness to plot drift / MPC success
    vs ``lambda_anc``. NOTE: an anchored config (``lambda_anc > 0``) in ``coordinator``/``eval`` run mode
    requires a pinned probe path (validate_config); pass a ``base_cfg`` whose run mode / probe pin admits
    the swept values (e.g. ``train_local`` for a pure-knob resolution, or a coordinator cfg with
    ``data.probe_path`` set).
    """
    sweep: dict[float, LensembleConfig] = {}
    for value in values:
        objective = dataclasses.replace(base_cfg.objective, lambda_anc=float(value))
        sweep[float(value)] = dataclasses.replace(base_cfg, objective=objective)
    return sweep


# --- internals: rung composition, the reference frame, the Layer-4 consensus ---


# The pinned-probe landmark count for the ladder rungs. k = 24 >> d gives a denser frame anchor (more
# absolute constraints) than the bare k = d minimum, so the Variant-A anchor pins the toy frame more
# firmly within the inner-loop budget (RFC-0002 §4: k >= d generic landmarks pin all O(d) gauge dofs; a
# larger k tightens the pin under the AdamW inner loop). All clips are tiny so the run stays CPU-fast.
_LADDER_LANDMARK_COUNT = 24


def _compose_rung_cfg(
    base_cfg: "LensembleConfig", spec: RungSpec
) -> tuple["LensembleConfig", Path | None]:
    """Compose a rung's config: set the objective knobs; pin a real probe when the rung needs the probe.

    Returns ``(cfg, probe_dir)`` where ``probe_dir`` is the throwaway dir the pinned probe was written to
    (``None`` for rungs 1-2, which neither anchor nor backstop and so use the coordinator's placeholder-hash
    probe). A rung is pinned a real probe whenever it ANCHORS (``lambda_anc > 0``; required by
    validate_config, RFC-0004) OR runs the BACKSTOP (which aligns to the round-0 ``E_ref`` on the probe) —
    both need a real ``k >= d`` landmark probe with round-0 ``f_ref`` targets that the harness and the
    reference frame share.
    """
    objective = dataclasses.replace(
        base_cfg.objective,
        lambda_sig=spec.lambda_sig,
        lambda_anc=spec.lambda_anc,
        anchor_variant="landmark",
    )
    cfg = dataclasses.replace(base_cfg, objective=objective)
    if not (spec.anchored or spec.backstop):
        return cfg, None

    probe_dir = Path(tempfile.mkdtemp(prefix="lensemble-ladder-probe-"))
    probe_path = _write_probe(cfg, probe_dir)
    data = dataclasses.replace(cfg.data, probe_path=str(probe_path))
    return dataclasses.replace(cfg, data=data), probe_dir


def _write_probe(cfg: "LensembleConfig", out_dir: Path) -> Path:
    """Write a deterministic ``k`` landmark probe (round-0 ``f_ref`` targets); return its path.

    The probe: ``k = _LADDER_LANDMARK_COUNT >= d`` generic landmark clips (enough to pin all O(d) gauge
    dofs, RFC-0002 §4) with targets ``t_i = f_ref(p_i)`` from the round-0 encoder ``f_ref``. ``f_ref`` is
    built under ``torch.manual_seed(root_seed)`` — the SAME seed the :class:`Coordinator` uses to build its
    round-0 warm-start — so the probe (and hence its content hash + targets) is FULLY DETERMINISTIC across
    runs (``build_encoder`` consumes global RNG, so an unseeded build would yield a different ``f_ref`` each
    time and break reproducibility) and the targets pin onto the federation's actual round-0 frame. The
    coordinator broadcasts this probe's content hash, so every participant accepts the round AND the
    anchor's ``INV-PROBE-PIN`` check passes (the participant loads the SAME probe via ``cfg.data.probe_path``).
    """
    from lensemble.data.probe import build_probe, save_probe

    points, f_ref = _probe_points_and_ref(cfg)
    probe = build_probe(
        points, torch.arange(_LADDER_LANDMARK_COUNT), f_ref, probe_version=1
    )
    probe_path = out_dir / "probe.safetensors"
    save_probe(probe, probe_path)
    return probe_path


def _reference_frame(cfg: "LensembleConfig") -> "Tensor":
    """The round-0 reference frame ``E_ref = f_ref(P)`` ``(k*N, d)`` for the Layer-3 backstop (RFC-0002 §5).

    Loads the rung's pinned probe (every backstop rung pins one, see :func:`_compose_rung_cfg`), forwards
    the round-0 ``f_ref`` on the probe landmarks, and flattens to ``(k*N, d)`` — the frame each
    over-threshold participant is Procrustes-aligned onto before the outer step. Matches the shape of the
    per-silo ``f_c(P)`` the harness measures (both ``(k*N, d)``). ``f_ref`` is rebuilt under the same
    ``root_seed`` so ``E_ref`` and the harness's per-silo frames live on one consistent round-0 reference.
    """
    from lensemble.data.probe import load_probe
    from lensemble.model.encoder import build_encoder, snapshot_reference

    probe_path = getattr(cfg.data, "probe_path", None)
    if probe_path is not None:
        probe = load_probe(probe_path)
        points = probe.points[probe.landmark_idx]
    else:  # pragma: no cover — backstop rungs always pin a probe (defensive)
        points, _ = _probe_points_and_ref(cfg)
    torch.manual_seed(int(cfg.determinism.root_seed))
    f_ref = snapshot_reference(build_encoder(cfg))
    with torch.no_grad():
        tokens = f_ref(points).tokens.to(torch.float32)
    return tokens.reshape(-1, tokens.shape[-1])


def _probe_points_and_ref(
    cfg: "LensembleConfig",
) -> "tuple[Tensor, ReferenceEncoder]":
    """The deterministic probe landmark clips + round-0 ``f_ref`` (seeded by ``root_seed``).

    ``points`` are ``k = _LADDER_LANDMARK_COUNT`` fixed-seed clips ``(k, T, C, H, W)``; ``f_ref`` is the
    frozen round-0 encoder snapshot built under ``torch.manual_seed(root_seed)`` (the coordinator's
    warm-start seed), so both are reproducible across runs and across the probe-write / reference-frame
    paths.
    """
    from lensemble.model.encoder import build_encoder, snapshot_reference

    m = cfg.model
    num_frames = int(getattr(m, "num_frames"))
    in_channels = int(getattr(m, "in_channels", 3))
    image_size = int(getattr(m, "image_size"))
    gen = torch.Generator().manual_seed(20240602)
    points = torch.randn(
        _LADDER_LANDMARK_COUNT,
        num_frames,
        in_channels,
        image_size,
        image_size,
        generator=gen,
    )
    torch.manual_seed(int(cfg.determinism.root_seed))
    f_ref = snapshot_reference(build_encoder(cfg))
    return points, f_ref


def run_distillation(final_frames: "dict[str, Tensor]") -> "Tensor | None":
    """Run the Layer-4 function-space distillation consensus on the final per-silo frames (RFC-0002 §6).

    Forms the gauge-invariant align-then-mean consensus over each silo's last-round ``f_c(P)`` (the
    align=True path of :func:`~lensemble.gauge.distill.distill_consensus`) and distills a global student
    against it (:func:`~lensemble.gauge.distill.distill_to_consensus`). This compares *functions on the
    shared probe, never weights*, so it is gauge-invariant by construction (RFC-0002 §6) — it is the
    top-rung mechanism the ladder exercises. Returns the distilled student frame (the consensus the global
    student matched), or ``None`` when fewer than one frame is available. The reported per-rung drift is the
    pre-consensus per-silo signal, unchanged by this pass.
    """
    from lensemble.gauge.distill import distill_consensus, distill_to_consensus

    if not final_frames:
        return None
    consensus = distill_consensus(final_frames, align=True)
    return distill_to_consensus(consensus)


def _cleanup_dir(path: Path) -> None:
    """Best-effort removal of a throwaway probe dir (the anchored-rung probe artifacts)."""
    import shutil
    from contextlib import suppress

    with suppress(Exception):
        shutil.rmtree(path, ignore_errors=True)
