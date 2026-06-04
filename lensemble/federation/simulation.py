"""lensemble.federation.simulation — the live multi-round federated training-sim harness (#55).

The harness the ablation-ladder runner (RFC-0005 §6, the paper's core experiment) drives the five rungs
through. It stands up ONE :class:`~lensemble.federation.transport.InProcessTransport`, ONE
:class:`~lensemble.federation.coordinator.Coordinator`, and one
:class:`~lensemble.federation.participant.Participant` per silo, then runs ``num_rounds`` real DiLoCo outer
rounds: each round every participant runs its inner loop on its OWN local windows
(:meth:`Participant.local_round`), submits the privatized, bound pseudo-gradient through the transport, and
the coordinator drives ``try_round()`` (``OPEN → COLLECTING → AGGREGATING → ALIGNING → COMMITTING →
CLOSED``). Only pseudo-gradients cross the transport seam (``INV-RESIDENCY``); the raw windows never leave
their silo's :class:`Participant`.

The frame-drift signal (RFC-0002 §9 / RFC-0005 §2). Each round the harness also measures the inter-silo
latent frame drift on the SHARED pinned probe: every silo's trained encoder is run forward on the probe
landmarks to give ``f_c(P)``, and :func:`~lensemble.gauge.drift.frame_drift` recovers the inter-frame
rotation angle. For a non-vacuous drift signal the silos MUST hold genuinely different data (different
``SiloData.windows``); identical data shows no drift regardless of the gauge mechanism and makes the
experiment vacuous (the ablation runner supplies per-silo seeds). The probe is shared across silos so
``GlobalState.probe_hash`` matches every participant's pinned-probe hash (``INV-PROBE-PIN``).

The Layer-3 Procrustes backstop (RFC-0002 §5, #18) is wired ON for rung 3+ by passing ``backstop=True``
(and a ``reference_embeddings`` frame ``E_ref``): a :class:`_SimCoordinator` then overrides the
``_probe_embeddings``/``_reference_embeddings`` seam so each over-threshold participant's predictor delta
is conjugated by ``Q_c*`` before the outer step. With ``backstop=False`` the coordinator's ALIGNING is the
byte-identical measured pass-through.

Residency-safe + temp-dir-safe. The harness owns the Coordinator it builds, so it registers a cleanup of
the coordinator's per-run ``tempfile.mkdtemp`` artifacts dir (``Coordinator._artifacts_dir``) on exit — the
ablation runner builds many Coordinators (5 rungs × the sweep), so leaking those temp dirs would fill the
disk. The cleanup is best-effort (``ignore_errors``); the artifacts are throwaway round checkpoints.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lensemble.data.probe import PublicProbe
from lensemble.federation.coordinator import Coordinator
from lensemble.federation.participant import Participant
from lensemble.federation.transport import InProcessTransport
from lensemble.model.action_head import build_action_head
from lensemble.model.encoder import build_encoder

if TYPE_CHECKING:
    from lensemble.config.schema import LensembleConfig
    from lensemble.contracts import ActionSpec
    from lensemble.data.episode import Window
    from lensemble.federation.state import GlobalState

__all__ = [
    "SiloData",
    "RoundMetrics",
    "SimulationResult",
    "run_federated_simulation",
]

_GLOBAL_KEY = (
    "global"  # the reserved frame_drift participant id for the aggregated model
)


@dataclass(frozen=True)
class SiloData:
    """One silo's residency-bound local data for the simulation (the #22 data-layer boundary).

    ``windows`` are the silo's RAW local training windows (they NEVER cross the transport — only the
    released pseudo-gradient does, ``INV-RESIDENCY``); ``action_spec`` is the silo embodiment's
    :class:`~lensemble.contracts.ActionSpec`; ``dataset_root`` is the 32-byte Merkle root ``R_c`` the
    released delta binds to (``INV-COMMIT-BINDING``). For a non-vacuous frame-drift signal the silos must
    carry DIFFERENT ``windows`` (different seeds) so the per-silo frames genuinely diverge under the naive
    rung (RFC-0005 §6).
    """

    participant_id: str
    windows: Sequence["Window"]
    action_spec: "ActionSpec"
    dataset_root: bytes


@dataclass(frozen=True)
class RoundMetrics:
    """The three metric families measured at one outer round (RFC-0005 §2-4).

    ``frame_drift_angle_deg`` is the MEAN inter-silo rotation angle on the probe (the headline figure,
    §2); ``frame_drift_residual`` the mean optimal-Procrustes residual; ``effective_dim`` the
    participation-ratio collapse guard over the committed global frame (§4); ``success_rate`` the
    downstream MPC planning success on the eval world (§3).
    """

    round_index: int
    frame_drift_angle_deg: float
    frame_drift_residual: float
    effective_dim: float
    success_rate: float
    global_hash: str


@dataclass(frozen=True)
class SimulationResult:
    """The per-round metrics of one federated-simulation run (the ablation runner's per-rung input).

    ``final_frames`` carries each silo's last-round trained frame ``f_c(P)`` ``(k*N, d)`` so the Layer-4
    rung can run the gauge-invariant distillation consensus on them (RFC-0002 §6); it is the public-probe
    output only (``INV-RESIDENCY`` not at stake — the probe is public).
    """

    per_round: tuple[RoundMetrics, ...]
    initial_global_hash: str
    final_global_hash: str
    contributing_participants: tuple[str, ...] = field(default_factory=tuple)
    final_frames: dict[str, Tensor] = field(default_factory=dict)


class _SimParticipant(Participant):
    """A :class:`Participant` whose #22 hooks return one silo's fixtures + exposes its trained frame.

    Overrides ``_local_windows`` / ``_pinned_probe`` / ``_dataset_root`` / ``_action_spec`` from the
    silo's :class:`SiloData` and the shared pinned probe (so ``GlobalState.probe_hash`` matches,
    ``INV-PROBE-PIN``). :meth:`probe_embeddings` reproduces the SAME inner-loop training pass the released
    round runs (a fresh encoder loaded from ``θ_t``, the silo's objective + windows, ``H`` AdamW steps) and
    returns the trained encoder's embeddings on the probe landmarks ``f_c(P)`` ``(k*N, d)`` — the frame the
    drift diagnostic measures (RFC-0002 §9). It is a pure read of the trained encoder on the PUBLIC probe;
    nothing private crosses (the probe is public, ``INV-RESIDENCY`` not at stake).
    """

    def __init__(
        self,
        config: "LensembleConfig",
        *,
        participant_id: str,
        transport: InProcessTransport,
        silo: SiloData,
        probe: PublicProbe,
    ) -> None:
        super().__init__(config, participant_id=participant_id, transport=transport)
        self._silo = silo
        self._probe = probe

    def _pinned_probe(self) -> PublicProbe:
        return self._probe

    def _local_windows(self) -> Sequence["Window"]:
        return self._silo.windows

    def _dataset_root(self) -> bytes:
        return self._silo.dataset_root

    def _action_spec(self) -> "ActionSpec":
        return self._silo.action_spec

    def _build_objective(self, cfg, round_seed, encoder, probe):  # type: ignore[no-untyped-def, override]
        """Build the per-round objective with the anchor pinned to the FIXED round-0 reference (RFC-0002 §4).

        The faithful Variant-A anchor (``INV-PROBE-PIN``): the landmark targets derive ONLY from ``f_ref``,
        the round-0 encoder — never a later ``f_t``. The probe carries ``landmark_targets = f_ref(P)`` (set
        at ``build_probe`` time from the round-0 encoder), so anchoring to ``probe.landmark_targets`` pins
        every silo's frame to the SAME fixed reference across ALL rounds. (The base
        :meth:`Participant._build_objective` re-snapshots ``f_ref`` from the CURRENT ``θ_t`` each round — a
        documented per-round approximation; for the multi-round simulation the fixed round-0 reference is
        the faithful, stable pin.) With ``lambda_anc == 0`` the bare LeJEPA objective runs (anchor=None).
        """
        from lensemble.model.objective import Objective
        from lensemble.model.sigreg import build_sketch

        o = cfg.objective
        anchor = None
        if float(o.lambda_anc) > 0.0:
            from lensemble.gauge.anchor import FrameAnchor

            anchor = FrameAnchor(
                probe,
                probe.landmark_targets,  # f_ref(P) — the FIXED round-0 reference (INV-PROBE-PIN)
                variant=o.anchor_variant,
                probe_hash=probe.content_hash.hex(),
            ).loss
        _ = build_sketch(round_seed, encoder.d, int(o.sigreg_sketch_dim))
        return Objective(
            lambda_pred=float(o.lambda_pred),
            lambda_sig=float(o.lambda_sig),
            lambda_anc=float(o.lambda_anc),
            sketch_seed=round_seed,
            sketch_dim=int(o.sigreg_sketch_dim),
            ep_knots=int(o.sigreg_knots),
            anchor=anchor,
        )

    def probe_embeddings(self, global_state: "GlobalState", round_seed: int) -> Tensor:
        """``f_c(P)`` ``(k*N, d)``: the silo's trained-encoder frame on the probe landmarks (RFC-0002 §9).

        Re-runs the exact inner-loop training pass :meth:`Participant.local_round` runs — fetch ``θ_t``
        into a fresh encoder, build the silo objective, run ``H`` inner AdamW steps over the silo windows —
        then runs the trained encoder forward on the probe landmark clips and returns the flattened
        ``(k*N, d)`` token frame. Deterministic given ``(θ_t, round_seed, silo.windows)``; a pure read of
        the trained encoder on the public probe. Used ONLY by the simulation diagnostic (it does not cross
        the transport; the released object is still the pseudo-gradient).
        """
        from lensemble.model.predictor import build_predictor

        cfg = self.config
        probe = self._probe
        theta_weights = self.transport.fetch_params(global_state.theta_ref)
        phi_weights = self.transport.fetch_params(global_state.phi_ref)
        encoder = build_encoder(cfg)
        predictor = build_predictor(cfg)
        encoder.load_state_dict(theta_weights, strict=True)
        predictor.load_state_dict(phi_weights, strict=True)
        action_head = build_action_head(cfg, self._action_spec())
        objective = self._build_objective(cfg, round_seed, encoder, probe)
        self._run_inner_loop(cfg, encoder, predictor, action_head, objective)

        landmarks = probe.points[probe.landmark_idx]
        with torch.no_grad():
            tokens = encoder(landmarks).tokens.to(torch.float32)
        # (k, N, d) -> (k*N, d): the per-landmark token frame the drift diagnostic Procrustes-aligns.
        return tokens.reshape(-1, tokens.shape[-1])


class _SimCoordinator(Coordinator):
    """A :class:`Coordinator` that wires the #18/#22 backstop seam from harness-supplied embeddings.

    The harness measures each silo's ``f_c(P)`` itself (the trained frame) and stages them here before the
    round; overriding ``_probe_embeddings`` / ``_reference_embeddings`` makes ALIGNING fire the Layer-3
    Procrustes backstop (RFC-0002 §5) — each over-threshold participant's predictor delta is conjugated by
    ``Q_c*`` before the outer step. With no embeddings staged (the default) ALIGNING is the byte-identical
    measured pass-through, exactly like the base :class:`Coordinator`.
    """

    def __init__(
        self,
        config: "LensembleConfig",
        *,
        transport: InProcessTransport,
        backstop: bool,
        reference_embeddings: Tensor | None,
    ) -> None:
        super().__init__(config, transport=transport)
        self._backstop_on = backstop
        self._reference = reference_embeddings
        self._staged_embeddings: dict[str, Tensor] = {}

    def stage_embeddings(self, embeddings: dict[str, Tensor]) -> None:
        """Stage the round's per-participant ``f_c(P)`` for the backstop seam (harness-side)."""
        self._staged_embeddings = dict(embeddings)

    def _probe_embeddings(self, t: int) -> "dict[str, Tensor] | None":  # noqa: ARG002
        if not self._backstop_on or not self._staged_embeddings:
            return None
        return dict(self._staged_embeddings)

    def _reference_embeddings(self, t: int) -> "Tensor | None":  # noqa: ARG002
        if not self._backstop_on:
            return None
        return self._reference

    def _measure_drift(self, t: int, embeddings: "dict[str, Tensor] | None") -> None:
        """Degenerate-safe ALIGNING drift measurement (the anchor-pinned frames may coincide).

        The base :meth:`Coordinator._measure_drift` runs :func:`~lensemble.gauge.drift.frame_drift` on the
        staged embeddings; when a STRONG anchor pins two silos onto a near-identical frame the inter-pair
        Procrustes ``M`` becomes rank-deficient and raises
        :class:`~lensemble.errors.DegenerateProcrustes`. Coinciding frames are the GOOD anchored case
        (drift -> 0), so the harness's diagnostic must not abort the round on it: a degenerate inter-pair
        measurement is swallowed here (the report stays at its prior value). The committed reduction is
        untouched — this only guards the DIAGNOSTIC, which the harness recomputes degenerate-safely itself.
        """
        from lensemble.errors import DegenerateProcrustes

        with suppress(DegenerateProcrustes):
            super()._measure_drift(t, embeddings)


def run_federated_simulation(
    participants_data: Sequence[SiloData],
    *,
    cfg: "LensembleConfig",
    num_rounds: int,
    backstop: bool = False,
    reference_embeddings: Tensor | None = None,
    eval_success: bool = True,
) -> SimulationResult:
    """Run ``num_rounds`` live DiLoCo outer rounds over the silos; return the per-round metrics (#55).

    Stands up one :class:`InProcessTransport`, one :class:`_SimCoordinator` (the backstop wired ON when
    ``backstop=True`` + ``reference_embeddings`` supplied), and one :class:`_SimParticipant` per silo.
    Each round: every participant measures its trained frame ``f_c(P)`` on the shared probe (staged for
    the backstop), runs :meth:`Participant.local_round` and submits its pseudo-gradient, then the
    coordinator drives ``try_round()`` to ``CLOSED``. The round's :class:`RoundMetrics` carries the
    inter-silo frame drift (mean rotation angle + residual, §2), the effective dimension of the committed
    global frame measured on the probe (§4), and — when ``eval_success`` — the MPC ``success_rate`` on the
    config's eval world (§3).

    Residency-safe: only pseudo-gradients cross the transport. Temp-dir-safe: the coordinator's per-run
    artifacts dir (``tempfile.mkdtemp``) is removed on exit (the ablation runner builds many coordinators).
    """
    if not participants_data:
        raise ValueError("run_federated_simulation needs at least one silo")
    if num_rounds < 1:
        raise ValueError(f"num_rounds must be >= 1, got {num_rounds}")

    transport = InProcessTransport()
    coordinator = _SimCoordinator(
        cfg,
        transport=transport,
        backstop=backstop,
        reference_embeddings=reference_embeddings,
    )
    artifacts_dir = coordinator._artifacts_dir  # noqa: SLF001 — own the temp dir we built
    participants = [
        _SimParticipant(
            cfg,
            participant_id=silo.participant_id,
            transport=transport,
            silo=silo,
            probe=_shared_probe(cfg, coordinator.global_state()),
        )
        for silo in participants_data
    ]
    contributing = tuple(sorted(p.participant_id for p in participants))
    initial_global_hash = coordinator.global_state_hash()

    try:
        per_round, final_frames = _drive_rounds(
            coordinator,
            transport,
            participants,
            cfg=cfg,
            num_rounds=num_rounds,
            eval_success=eval_success,
        )
        return SimulationResult(
            per_round=tuple(per_round),
            initial_global_hash=initial_global_hash,
            final_global_hash=coordinator.global_state_hash(),
            contributing_participants=contributing,
            final_frames=final_frames,
        )
    finally:
        # Best-effort cleanup of the Coordinator's throwaway per-run artifacts dir (no public hook; we own
        # it because we built the Coordinator). The ablation runner builds many coordinators; leaking these
        # tempfile.mkdtemp dirs would fill the disk.
        with suppress(Exception):
            shutil.rmtree(artifacts_dir, ignore_errors=True)


def _drive_rounds(
    coordinator: _SimCoordinator,
    transport: InProcessTransport,
    participants: list[_SimParticipant],
    *,
    cfg: "LensembleConfig",
    num_rounds: int,
    eval_success: bool,
) -> tuple[list[RoundMetrics], dict[str, Tensor]]:
    """Drive ``num_rounds`` rounds; collect a :class:`RoundMetrics` per round + the final per-silo frames."""
    per_round: list[RoundMetrics] = []
    success = _success_rate_for(cfg) if eval_success else 0.0
    last_embeddings: dict[str, Tensor] = {}

    for _ in range(num_rounds):
        gs = coordinator.global_state()
        round_seed = gs.sketch_seed

        # 1. Each silo measures its trained frame f_c(P) (the drift diagnostic + the backstop input).
        embeddings: dict[str, Tensor] = {
            p.participant_id: p.probe_embeddings(gs, round_seed) for p in participants
        }
        last_embeddings = embeddings
        coordinator.stage_embeddings(embeddings)

        # 2. Each silo runs its local round and submits its pseudo-gradient (only the delta crosses).
        for p in participants:
            pg = p.local_round(gs, round_seed)
            transport.submit_update(
                participant_id=p.participant_id,
                round_index=gs.round_index,
                update=pg,
            )

        # 3. The coordinator drives the outer round to CLOSED.
        from lensemble.federation.round import RoundState

        state = coordinator.try_round()
        if state is not RoundState.CLOSED:
            raise RuntimeError(
                f"simulation round {gs.round_index} did not CLOSE (state={state}); "
                "check the quorum vs the silo count"
            )

        # 4. The three metric families on the round just committed. The inter-silo drift is computed
        # DEGENERATE-SAFELY: a strong anchor can pin two silos onto a near-identical frame, whose Procrustes
        # M is rank-deficient — that is the GOOD anchored case (drift -> 0), not an error, so a degenerate
        # pair contributes ~0deg rather than aborting (INV-PROBE-PIN was already enforced participant-side
        # at local_round; the drift diagnostic only records the broadcast probe_hash).
        angle, residual = _inter_silo_drift(embeddings)
        eff = _effective_dim_of_global(embeddings)
        per_round.append(
            RoundMetrics(
                round_index=gs.round_index,
                frame_drift_angle_deg=angle,
                frame_drift_residual=residual,
                effective_dim=eff,
                success_rate=success,
                global_hash=coordinator.global_state_hash(),
            )
        )
    return per_round, last_embeddings


def _inter_silo_drift(embeddings: dict[str, Tensor]) -> tuple[float, float]:
    """Mean inter-silo rotation angle + residual on the probe, degenerate-safe (RFC-0002 §9 / §2).

    Procrustes-aligns every silo PAIR's probe frames and averages the recovered rotation angle (the
    headline figure) and the alignment residual. A pair whose frames COINCIDE (a strong anchor pinned both
    onto the same reference) yields a rank-deficient ``M`` and raises
    :class:`~lensemble.errors.DegenerateProcrustes`; that is the GOOD anchored case (drift -> 0), so a
    degenerate pair contributes ``0.0`` deg rather than aborting. Returns ``(0.0, 0.0)`` for fewer than two
    silos.
    """
    from lensemble.errors import DegenerateProcrustes
    from lensemble.gauge.drift import _rotation_angle_deg
    from lensemble.gauge.procrustes import procrustes_align

    ids = sorted(pid for pid in embeddings if pid != _GLOBAL_KEY)
    angles: list[float] = []
    residuals: list[float] = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            try:
                rotation, residual = procrustes_align(embeddings[a], embeddings[b])
            except DegenerateProcrustes:
                # Coinciding frames (the anchor pinned them together): treat as ~0deg drift.
                angles.append(0.0)
                residuals.append(0.0)
                continue
            angles.append(_rotation_angle_deg(rotation))
            residuals.append(residual)
    if not angles:
        return 0.0, 0.0
    return sum(angles) / len(angles), sum(residuals) / len(residuals)


def _effective_dim_of_global(embeddings: dict[str, Tensor]) -> float:
    """The participation-ratio collapse guard over the stacked per-silo probe frames (§4).

    The committed global frame's effective dimension: stack every silo's ``f_c(P)`` and report the
    covariance participation ratio. A collapse re-introduced by averaging across mutually-rotated frames
    drives this toward ``1`` (RFC-0002 §2.1).
    """
    from lensemble.eval.metrics import effective_dim

    stacked = torch.cat([emb for emb in embeddings.values()], dim=0)
    return effective_dim(stacked)


def _success_rate_for(cfg: "LensembleConfig") -> float:
    """The MPC planning success on the config's eval world (§3); 0.0 when the world cannot resolve.

    Resolves ``cfg.eval.env_id`` to an :class:`~lensemble.eval.world.EvalWorld` and reports the seed-pinned
    success fraction over a few held-out episodes (the toy world's success is rigged by reset-seed parity).
    The metric is gauge-independent here (the toy world has no real dynamics), so it is reported once per
    run rather than re-planned every round — the ladder's load-bearing signal is the frame drift (§2/§6).
    """
    with suppress(Exception):
        from lensemble.eval.metrics import success_rate
        from lensemble.eval.world import resolve_env

        env_id = cfg.eval.env_id
        world = resolve_env(env_id, cfg=cfg)
        root = int(cfg.determinism.root_seed)
        outcomes = []
        for i in range(4):
            world.reset(root + i)
            outcomes.append(world.succeeded())
        return success_rate(outcomes)
    return 0.0


def _shared_probe(cfg: "LensembleConfig", global_state: "GlobalState") -> PublicProbe:
    """Build the SHARED pinned probe whose content hash equals ``GlobalState.probe_hash`` (``INV-PROBE-PIN``).

    When the coordinator pinned a real probe (``cfg.data.probe_path``) the broadcast ``probe_hash`` is that
    probe's content hash; otherwise it is the 32-byte placeholder. For the anchored rungs the harness needs
    a real probe (``k >= d`` landmark clips + round-0 ``f_ref`` targets) whose hash MATCHES the broadcast,
    so the participants accept the round AND the anchor's ``INV-PROBE-PIN`` check passes. We build a probe
    from a fixed seed and, when the coordinator did not pin one, the participant's anchor uses this probe's
    own hash (the coordinator-side resolution is the #22/#04 boundary; here the harness pins both sides to
    the SAME probe object so the hashes agree).
    """
    from lensemble.data.probe import build_probe, load_probe
    from lensemble.model.encoder import snapshot_reference
    from lensemble.model.numerics import resolve_device

    probe_path = getattr(cfg.data, "probe_path", None)
    if probe_path is not None:
        return load_probe(probe_path)

    # No pinned probe path: build a probe whose content hash IS the broadcast probe_hash. The coordinator
    # uses the 32-byte placeholder when no probe is pinned, so a bare (lambda_anc=0) rung never reads the
    # probe points and any probe with that placeholder hash passes INV-PROBE-PIN. For the anchored rungs the
    # runner pins a real probe via cfg.data.probe_path (the branch above), so this placeholder-hash probe is
    # only used by the non-anchored rungs that never touch the probe.
    m = cfg.model
    num_frames = int(getattr(m, "num_frames"))
    in_channels = int(getattr(m, "in_channels", 3))
    image_size = int(getattr(m, "image_size"))
    d = int(getattr(m, "d", getattr(m, "latent_dim")))
    # Build the probe on the compute device so the f_ref forward (here) and the later
    # `probe_embeddings` forward run device/dtype-consistently with the encoder (CUDA inner loop). The CPU
    # generator keeps the bits identical; `probe_content_hash` canonicalizes via `.cpu()`, so the content
    # hash (INV-PROBE-PIN) is device-invariant and the CPU fallback is unchanged (#182).
    device = resolve_device()
    gen = torch.Generator().manual_seed(20240601)
    points = torch.randn(
        d, num_frames, in_channels, image_size, image_size, generator=gen
    ).to(device)
    f_ref = snapshot_reference(build_encoder(cfg))
    probe = build_probe(points, torch.arange(d), f_ref, probe_version=1)
    # Override the content hash to the broadcast placeholder so INV-PROBE-PIN holds for the bare rungs.
    return PublicProbe(
        points=probe.points,
        landmark_idx=probe.landmark_idx,
        landmark_targets=probe.landmark_targets,
        content_hash=global_state.probe_hash,
        probe_version=probe.probe_version,
    )
