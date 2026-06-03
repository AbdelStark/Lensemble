"""Elasticity, churn, and rejoiner recovery — the runtime fault-tolerance model (#44).

Covers the RFC-0013 §3 (Fault tolerance & elasticity) and §7 (Failure modes) acceptance criteria on a
tiny CPU V-JEPA config, reusing the `_ModelConfig` bridge + `InProcessTransport` seeding + the test
`Participant` subclass patterns from tests/ml/test_coordinator.py and tests/ml/test_participant.py:

- **Dropout above K → elastic completion.** With > K of `participant_count` updates staged (some absent),
  the round COMMITs over the PRESENT set and the `ContributionRecord` records exactly the present
  `participants` / `C_t` (the absent ones are not in it). Run across two participant counts and assert the
  Nesterov outer step stays stable (finite, commits a hash) under the varying count (RFC-0013 §3 / §Testing).
- **Dropout below K → `FaultToleranceExceeded` + retry.** Fewer than K updates → `try_round()` yields
  `RoundState.ABORTED` with the global hash AND round index unchanged; staging enough updates for the SAME
  round `t` and re-attempting COMMITs and advances. `run` surfaces `FaultToleranceExceeded`
  (code `FAULT_TOLERANCE_EXCEEDED`, carrying `contributing`/`quorum`).
- **collect_timeout drop / no stall.** An absent (un-staged) participant is dropped; the round still
  completes over the present set (does not stall) and the dropped participant reconciles next round.
- **Rejoiner recovery.** A test `Participant.join`s, recovers the committed `GlobalState`, validates the
  checkpoint hash (a tampered θ ref → `CheckpointIntegrityError`), and on a clean recovery runs
  `local_round` next round producing a well-formed `PseudoGradient`. A round-0 rejoiner whose fetched
  encoder hash differs from the pinned warm-start raises `GaugeError`.
- **`K = max(min_participants, secure_agg_threshold)` is honored.** `secure_agg_threshold >
  fault_tolerance_min_participants` makes the higher threshold gate.

NOTE: placed in tests/ml (NOT tests/federation): the §8 CI gate scans tests/{unit,property,integration,
ml,e2e,regression}; a tests/federation directory would not run. The coordinator + participant are
model-bearing runtime objects, so tests/ml is their home (mirrors tests/ml/test_coordinator.py).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pytest
import torch
from torch import Tensor

from lensemble.config import LensembleConfig
from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
from lensemble.data.episode import Window
from lensemble.data.probe import PublicProbe, probe_content_hash
from lensemble.errors import (
    CheckpointIntegrityError,
    FaultToleranceExceeded,
    GaugeError,
    LensembleErrorCode,
)
from lensemble.federation import (
    Coordinator,
    GlobalState,
    InProcessTransport,
    ParamRef,
    Participant,
    RoundState,
    build_pseudogradient,
)
from lensemble.federation.pseudogradient import PseudoGradient
from lensemble.federation.transport import weights_content_hash
from lensemble.model import build_encoder, build_predictor

# --- a tiny CPU model config carrying BOTH the real ModelConfig fields AND the V-JEPA shape fields the
# build_* functions read (mirrors tests/ml/test_coordinator.py::_ModelConfig). ---

_D = 8
_NUM_TOKENS = (
    4  # (num_frames//tubelet) * (image_size//patch_size)**2 = (2//2)*(4//2)**2 = 4
)
_T, _C, _H, _W = 2, 3, 4, 4
_ACTION_DIM = 2
_NUM_STEPS = 2
_K_LANDMARKS = 8  # k >= d landmarks (INV-PROBE-PIN coverage)
_ROOT = b"\x2a" * 32  # a fixed 32-byte dataset root R_c (INV-COMMIT-BINDING)


@dataclass(frozen=True)
class _ModelConfig:
    # real ModelConfig fields (keep config_hash / manifest well-formed)
    encoder: str = "vjepa2-vit-l"
    warm_start_release: str = "vjepa2-2.0"
    latent_dim: int = _D
    num_tokens: int = _NUM_TOKENS
    predictor_depth: int = 1
    predictor_width: int = _D
    wmcp_version: str = WMCP_VERSION
    encoder_frozen: bool = False
    # V-JEPA shape fields the build_* functions read
    d: int = _D
    in_channels: int = _C
    num_frames: int = _T
    image_size: int = _H
    patch_size: int = 2
    tubelet: int = 2
    depth: int = 1
    num_heads: int = 2
    cond_dim: int = _D


def _cfg(
    *,
    min_participants: int = 2,
    secure_agg_threshold: int = 2,
    participant_count: int = 4,
    collect_timeout_s: float = 30.0,
    **overrides: object,
) -> LensembleConfig:
    """A coordinator-mode config with a tiny model and the elasticity quorum/timeout knobs set."""
    base = LensembleConfig()
    fed = dataclasses.replace(
        base.federation,
        num_rounds=2,
        outer_lr=0.7,
        outer_nesterov_momentum=0.9,
        participant_count=participant_count,
        fault_tolerance_min_participants=min_participants,
        secure_agg_threshold=secure_agg_threshold,
        collect_timeout_s=collect_timeout_s,
    )
    model = _ModelConfig()  # type: ignore[arg-type]
    cfg = dataclasses.replace(base, model=model, federation=fed, run_mode="coordinator")
    return dataclasses.replace(cfg, **overrides)  # type: ignore[arg-type]


def _toy_update(
    cfg: LensembleConfig,
    *,
    round_index: int,
    seed: int,
    scale: float = 1e-2,
) -> PseudoGradient:
    """A tiny θ⊕φ-sized PseudoGradient in the canonical encoder.*/predictor.* order (mirrors test_coordinator)."""
    torch.manual_seed(seed)
    enc = build_encoder(cfg)
    pred = build_predictor(cfg)
    gen = torch.Generator().manual_seed(seed)
    param_deltas: dict[str, Tensor] = {}
    for name, t in enc.state_dict().items():
        param_deltas[f"encoder.{name}"] = scale * torch.randn(
            t.shape, generator=gen, dtype=torch.float32
        )
    for name, t in pred.state_dict().items():
        param_deltas[f"predictor.{name}"] = scale * torch.randn(
            t.shape, generator=gen, dtype=torch.float32
        )
    return build_pseudogradient(
        param_deltas, dataset_root=_ROOT, round_index=round_index, clipped=True
    )


def _stage(
    transport: InProcessTransport,
    cfg: LensembleConfig,
    *,
    round_index: int,
    participant_ids: list[str],
    seed_base: int = 100,
) -> dict[str, PseudoGradient]:
    """Stage one PseudoGradient per present participant for `round_index`; return the staged updates."""
    staged: dict[str, PseudoGradient] = {}
    for i, pid in enumerate(participant_ids):
        pg = _toy_update(cfg, round_index=round_index, seed=seed_base + i)
        transport.submit_update(participant_id=pid, round_index=round_index, update=pg)
        staged[pid] = pg
    return staged


# --- dropout ABOVE K → elastic completion over the PRESENT set; ContributionRecord records C_t ---


def test_dropout_above_k_completes_elastically_over_present_set() -> None:
    # participant_count = 4, K = max(2, 2) = 2; stage 3 of 4 (one absent) — above K → elastic completion.
    cfg = _cfg(min_participants=2, secure_agg_threshold=2, participant_count=4)
    transport = InProcessTransport()
    coord = Coordinator(cfg, transport=transport)
    h0 = coord.global_state_hash()
    present = ["c0", "c1", "c2"]  # c3 is absent (dropped for the round)
    _stage(transport, cfg, round_index=0, participant_ids=present)

    state = coord.try_round()

    assert state == RoundState.CLOSED  # the round completed elastically
    assert coord.global_state_hash() != h0  # the committed hash advanced
    rec = coord.ledger_records()[-1]
    # The record carries EXACTLY the present set — the absent c3 is not in it (C_t = 3).
    assert rec.participants == ("c0", "c1", "c2")
    assert set(rec.dataset_roots) == {"c0", "c1", "c2"}
    assert "c3" not in rec.participants


@pytest.mark.parametrize("present_count", [2, 3])
def test_nesterov_stable_under_varying_participant_count(present_count: int) -> None:
    # The Nesterov outer step stays stable (finite, commits a hash) for a varying C_t (RFC-0013 §3).
    cfg = _cfg(min_participants=2, secure_agg_threshold=2, participant_count=4)
    transport = InProcessTransport()
    coord = Coordinator(cfg, transport=transport)
    ids = [f"c{i}" for i in range(present_count)]
    _stage(transport, cfg, round_index=0, participant_ids=ids)

    state = coord.try_round()

    assert state == RoundState.CLOSED
    assert len(coord.global_state_hash()) == 64
    params = coord.global_params()
    assert bool(torch.isfinite(params).all())  # the outer step stayed finite
    assert len(coord.ledger_records()[-1].participants) == present_count


# --- dropout BELOW K → FaultToleranceExceeded; ABORTED; hash + round index unchanged; THEN retry round t ---


def test_dropout_below_k_aborts_then_retries_same_round() -> None:
    cfg = _cfg(min_participants=2, secure_agg_threshold=2, participant_count=4)
    transport = InProcessTransport()
    coord = Coordinator(cfg, transport=transport)
    h0 = coord.global_state_hash()
    t0 = coord.global_state().round_index
    # Stage ONE update — below K = 2.
    _stage(transport, cfg, round_index=0, participant_ids=["c0"])

    state = coord.try_round()

    # Below K → ABORTED; the global hash and the round index are unchanged (no partial commit, no advance).
    assert state == RoundState.ABORTED
    assert coord.global_state_hash() == h0
    assert coord.global_state().round_index == t0

    # Stage enough updates for the SAME round t and re-attempt → it COMMITs and advances.
    _stage(transport, cfg, round_index=0, participant_ids=["c1", "c2"])
    state2 = coord.try_round()

    assert state2 == RoundState.CLOSED
    assert coord.global_state_hash() != h0  # advanced now
    rec = coord.ledger_records()[-1]
    assert rec.round_index == 0  # still round t = 0 (it was retried, not skipped)
    assert set(rec.participants) == {"c0", "c1", "c2"}


def test_run_surfaces_fault_tolerance_exceeded_below_k() -> None:
    # `run` surfaces FaultToleranceExceeded (code + carried fields) when a round is below K.
    cfg = _cfg(min_participants=2, secure_agg_threshold=2, participant_count=4)
    transport = InProcessTransport()
    coord = Coordinator(cfg, transport=transport)
    h_before = coord.global_state_hash()
    _stage(transport, cfg, round_index=0, participant_ids=["c0"])  # below K = 2

    with pytest.raises(FaultToleranceExceeded) as exc:
        coord.run(1)
    assert exc.value.code == LensembleErrorCode.FAULT_TOLERANCE_EXCEEDED
    assert exc.value.contributing == 1  # type: ignore[attr-defined]
    assert exc.value.quorum == 2  # type: ignore[attr-defined]
    assert coord.round_state() == RoundState.ABORTED
    assert coord.global_state_hash() == h_before  # unchanged


# --- K = max(min_participants, secure_agg_threshold): the HIGHER threshold gates ---


def test_quorum_is_max_of_min_participants_and_secure_agg_threshold() -> None:
    # min_participants = 2 but secure_agg_threshold = 3 → K = 3; staging 2 must abort below K.
    cfg = _cfg(min_participants=2, secure_agg_threshold=3, participant_count=4)
    transport = InProcessTransport()
    coord = Coordinator(cfg, transport=transport)
    _stage(transport, cfg, round_index=0, participant_ids=["c0", "c1"])  # 2 < K = 3

    with pytest.raises(FaultToleranceExceeded) as exc:
        coord.run(1)
    assert exc.value.quorum == 3  # type: ignore[attr-defined]  # K = max(2, 3)
    assert exc.value.contributing == 2  # type: ignore[attr-defined]
    assert coord.round_state() == RoundState.ABORTED

    # Staging the third update for the SAME round and re-attempting commits (now at K = 3).
    _stage(transport, cfg, round_index=0, participant_ids=["c2"])
    assert coord.try_round() == RoundState.CLOSED


# --- collect_timeout drop / no stall: an absent participant is dropped, the round completes, it reconciles ---


def test_absent_participant_dropped_does_not_stall_and_reconciles_next_round() -> None:
    # participant_count = 3, K = 2; c2 is absent at round 0 (the collect_timeout drop the present set models).
    cfg = _cfg(min_participants=2, secure_agg_threshold=2, participant_count=3)
    transport = InProcessTransport()
    coord = Coordinator(cfg, transport=transport)
    _stage(transport, cfg, round_index=0, participant_ids=["c0", "c1"])  # c2 absent

    state0 = coord.try_round()  # must not stall on the absent c2

    assert state0 == RoundState.CLOSED
    rec0 = coord.ledger_records()[-1]
    assert rec0.round_index == 0
    assert "c2" not in rec0.participants  # the dropped participant is not in round 0

    # The dropped participant reconciles next round t+1 (it contributes its update at round 1).
    assert coord.global_state().round_index == 1  # try_round advanced + opened round 1
    _stage(transport, cfg, round_index=1, participant_ids=["c0", "c1", "c2"])
    state1 = coord.try_round()
    assert state1 == RoundState.CLOSED
    rec1 = coord.ledger_records()[-1]
    assert rec1.round_index == 1
    assert "c2" in rec1.participants  # the previously-dropped participant reconciled


# --- rejoiner recovery: join() recovers the committed GlobalState, validates the checkpoint hash ---


def _action_spec() -> ActionSpec:
    return ActionSpec(
        embodiment_id="toy",
        kind=ActionKind.CONTINUOUS,
        dim=_ACTION_DIM,
        low=(-1.0, -1.0),
        high=(1.0, 1.0),
        num_classes=None,
        units=("u", "u"),
        wmcp_version=WMCP_VERSION,
    )


def _build_probe(cfg: LensembleConfig, seed: int = 0) -> PublicProbe:
    gen = torch.Generator().manual_seed(seed)
    points = torch.randn(_K_LANDMARKS, _T, _C, _H, _W, generator=gen)
    landmark_idx = torch.arange(_K_LANDMARKS)
    enc = build_encoder(cfg)
    targets = enc(points[landmark_idx]).tokens.detach()
    return PublicProbe(
        points=points,
        landmark_idx=landmark_idx,
        landmark_targets=targets,
        content_hash=probe_content_hash(points, landmark_idx),
        probe_version=1,
    )


def _build_windows(seed: int = 1, count: int = 2) -> list[Window]:
    gen = torch.Generator().manual_seed(seed)
    windows: list[Window] = []
    for _ in range(count):
        obs = torch.randn(_NUM_STEPS + 1, _T, _C, _H, _W, generator=gen)
        actions = torch.randn(_NUM_STEPS, _ACTION_DIM, generator=gen)
        windows.append(
            Window(obs=obs, actions=actions, num_steps=_NUM_STEPS, embodiment_id="toy")
        )
    return windows


class _TestParticipant(Participant):
    """A Participant whose #22 data-layer hooks are wired to tiny fixtures (mirrors test_participant)."""

    def __init__(
        self,
        config: LensembleConfig,
        *,
        participant_id: str,
        transport: InProcessTransport,
        probe: PublicProbe,
        windows: list[Window],
        dataset_root: bytes = _ROOT,
        warmstart_hash: str | None = None,
    ) -> None:
        super().__init__(config, participant_id=participant_id, transport=transport)
        self._probe = probe
        self._windows = windows
        self._root = dataset_root
        self._warmstart = warmstart_hash

    def _pinned_probe(self) -> PublicProbe:
        return self._probe

    def _local_windows(self) -> list[Window]:
        return self._windows

    def _dataset_root(self) -> bytes:
        return self._root

    def _action_spec(self) -> ActionSpec:
        return _action_spec()

    def _warmstart_hash(self) -> str | None:
        return self._warmstart


def _participant_cfg() -> LensembleConfig:
    """A participant-mode config (small inner horizon, DP on) for the rejoiner local_round path."""
    base = LensembleConfig()
    fed = dataclasses.replace(
        base.federation, inner_horizon=2, quantize_pseudo_gradient=False
    )
    priv = dataclasses.replace(
        base.privacy, clip_norm=0.1, noise_multiplier=0.5, enabled=True
    )
    model = _ModelConfig()  # type: ignore[arg-type]
    return dataclasses.replace(
        base, model=model, federation=fed, privacy=priv, run_mode="participant"
    )


def _seed_transport(
    cfg: LensembleConfig,
    probe: PublicProbe,
    *,
    round_index: int = 1,
    sketch_seed: int = 7,
) -> tuple[InProcessTransport, GlobalState, dict[str, Tensor], dict[str, Tensor]]:
    """Build a transport committed with a GlobalState whose θ/φ refs resolve to fresh global weights."""
    torch.manual_seed(0)
    encoder = build_encoder(cfg)
    predictor = build_predictor(cfg)
    theta = dict(encoder.state_dict())
    phi = dict(predictor.state_dict())
    theta_ref = ParamRef(
        content_hash=weights_content_hash(theta), locator="mem://theta"
    )
    phi_ref = ParamRef(content_hash=weights_content_hash(phi), locator="mem://phi")
    gs = GlobalState(
        theta_ref=theta_ref,
        phi_ref=phi_ref,
        round_index=round_index,
        sketch_seed=sketch_seed,
        probe_hash=probe.content_hash,
        wmcp_version=WMCP_VERSION,
    )
    transport = InProcessTransport()
    transport.commit(gs, theta_weights=theta, phi_weights=phi)
    return transport, gs, theta, phi


def test_rejoiner_recovers_committed_global_state_and_contributes_next_round() -> None:
    cfg = _participant_cfg()
    probe = _build_probe(cfg)
    transport, gs, _, _ = _seed_transport(cfg, probe, round_index=1, sketch_seed=7)
    p = _TestParticipant(
        cfg,
        participant_id="rejoiner",
        transport=transport,
        probe=probe,
        windows=_build_windows(),
    )

    # join() recovers the committed GlobalState (the authoritative recovered state).
    recovered = p.join("coordinator://local")
    assert recovered == gs
    assert recovered.round_index == 1

    # On a clean recovery the rejoiner runs local_round next round → a well-formed PseudoGradient.
    pg = p.local_round(recovered, round_seed=recovered.sketch_seed)
    assert pg.round_index == 1
    assert pg.clipped is True
    assert pg.dataset_root == _ROOT and len(pg.dataset_root) == 32
    assert bool(torch.isfinite(pg.delta).all())
    assert pg.delta.dtype == torch.float32


def test_rejoiner_rejects_tampered_checkpoint_on_join() -> None:
    cfg = _participant_cfg()
    probe = _build_probe(cfg)
    transport, gs, _, _ = _seed_transport(cfg, probe, round_index=1, sketch_seed=7)
    # Tamper the stored θ weights under the SAME content hash so the recomputed hash diverges.
    transport.corrupt_stored_weights(
        gs.theta_ref.content_hash, {"encoder.bogus": torch.ones(3)}
    )
    p = _TestParticipant(
        cfg,
        participant_id="rejoiner",
        transport=transport,
        probe=probe,
        windows=_build_windows(),
    )
    with pytest.raises(CheckpointIntegrityError) as exc:
        p.join("coordinator://local")
    assert exc.value.code == LensembleErrorCode.CHECKPOINT_INTEGRITY


def test_round0_rejoiner_revalidates_warmstart_and_raises_gauge_error() -> None:
    # A round-0 rejoiner whose fetched encoder hash differs from the pinned warm-start raises GaugeError
    # at join() (INV-WARMSTART-T0 revalidated on the recovery path, RFC-0013 §3).
    cfg = _participant_cfg()
    probe = _build_probe(cfg)
    transport, gs, _, _ = _seed_transport(cfg, probe, round_index=0, sketch_seed=7)
    assert gs.round_index == 0
    p = _TestParticipant(
        cfg,
        participant_id="rejoiner-t0",
        transport=transport,
        probe=probe,
        windows=_build_windows(),
        warmstart_hash="0"
        * 64,  # a pinned hash the recovered round-0 encoder cannot match
    )
    with pytest.raises(GaugeError) as exc:
        p.join("coordinator://local")
    assert exc.value.code == LensembleErrorCode.FRAME_DRIFT_EXCEEDED
    assert exc.value.remediation


def test_round0_rejoiner_with_matching_warmstart_recovers() -> None:
    # The positive INV-WARMSTART-T0 revalidation branch on join(): a matching pinned hash recovers cleanly.
    cfg = _participant_cfg()
    probe = _build_probe(cfg)
    transport, gs, theta, _ = _seed_transport(cfg, probe, round_index=0, sketch_seed=7)
    from lensemble.model.encoder import encoder_content_hash as _ech

    enc = build_encoder(cfg)
    enc.load_state_dict(theta, strict=True)
    pinned = _ech(enc)
    p = _TestParticipant(
        cfg,
        participant_id="rejoiner-t0-ok",
        transport=transport,
        probe=probe,
        windows=_build_windows(),
        warmstart_hash=pinned,
    )
    recovered = p.join("coordinator://local")
    assert recovered == gs
    assert recovered.round_index == 0
