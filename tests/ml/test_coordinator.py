"""Coordinator orchestration — the outer-round loop driven through the RoundState machine (#42).

Covers the RFC-0013 §1/§4 acceptance criteria on a tiny CPU V-JEPA config (mirrors
tests/ml/test_participant.py's `_ModelConfig` bridge + `InProcessTransport` seeding):

- determinism (`INV-AGG-DETERMINISM`): the same contributing set committed twice yields a
  bitwise-identical `(θ_{t+1}, φ_{t+1})` (the committed `global_model_hash` and the flat params agree);
- arrival-order independence: staging the identical updates in two submission orders commits the same
  hash (the reduction is in canonical participant-id-sorted order, not arrival order, RFC-0013 §4);
- corrupt reduction → abort: a non-reproducible reduction raises `NonDeterministicAggregation` and the
  round `ABORTED`s with the committed global hash unchanged (security-critical, never swallowed);
- payload restriction (`INV-ACTIONHEAD-LOCAL`): the broadcast/aggregated flat params length is exactly
  the encoder+predictor param count (no action head), and an `action_head.*` group cannot enter a
  `PseudoGradient` (`ResidencyViolation`);
- below-quorum → `FaultToleranceExceeded` with the round `ABORTED` and the global hash unchanged;
- the `round_state()` / `global_state()` lifecycle hooks (OPEN after broadcast, CLOSED after a round).

NOTE: placed in tests/ml (NOT tests/federation): the §8 CI gate scans tests/{unit,property,integration,
ml,e2e,regression}; a tests/federation directory would not run. The coordinator is a model-bearing
runtime object, so tests/ml is its home (mirrors tests/ml/test_participant.py).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from lensemble.config import LensembleConfig
from lensemble.contracts import WMCP_VERSION
from lensemble.data.probe import PublicProbe, probe_content_hash, save_probe
from lensemble.errors import (
    FaultToleranceExceeded,
    LensembleErrorCode,
    NonDeterministicAggregation,
    ResidencyViolation,
)
from lensemble.federation import (
    Coordinator,
    InProcessTransport,
    RoundState,
    build_pseudogradient,
)
from lensemble.federation.pseudogradient import PseudoGradient
from lensemble.gauge.drift import FrameDriftReport
from lensemble.model import build_encoder, build_predictor

# --- a tiny CPU model config carrying BOTH the real ModelConfig fields AND the V-JEPA shape fields the
# build_* functions read (mirrors tests/ml/test_participant.py::_ModelConfig). ---

_D = 8
_NUM_TOKENS = (
    4  # (num_frames//tubelet) * (image_size//patch_size)**2 = (2//2)*(4//2)**2 = 4
)
_T, _C, _H, _W = 2, 3, 4, 4
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


def _cfg(**overrides: object) -> LensembleConfig:
    """A coordinator-mode config with a tiny model and a low quorum so a 2-3 participant round runs."""
    base = LensembleConfig()
    fed = dataclasses.replace(
        base.federation,
        num_rounds=2,
        outer_lr=0.7,
        outer_nesterov_momentum=0.9,
        fault_tolerance_min_participants=2,
    )
    model = _ModelConfig()  # type: ignore[arg-type]
    cfg = dataclasses.replace(base, model=model, federation=fed, run_mode="coordinator")
    return dataclasses.replace(cfg, **overrides)  # type: ignore[arg-type]


def _global_param_count(cfg: LensembleConfig) -> int:
    """The encoder+predictor flat param count — the exact length of the broadcast/aggregated θ⊕φ."""
    torch.manual_seed(0)
    enc = build_encoder(cfg)
    pred = build_predictor(cfg)
    return sum(t.numel() for t in enc.state_dict().values()) + sum(
        t.numel() for t in pred.state_dict().values()
    )


def _toy_update(
    cfg: LensembleConfig,
    *,
    round_index: int,
    seed: int,
    scale: float = 1e-2,
) -> PseudoGradient:
    """A tiny θ⊕φ-sized PseudoGradient built from per-group toy deltas (canonical encoder.*/predictor.*).

    Reuses `build_pseudogradient` so the flat delta is in the SAME canonical (encoder sorted, then
    predictor sorted) order the coordinator flattens the global params in — element-wise aligned.
    """
    torch.manual_seed(seed)
    enc = build_encoder(cfg)
    pred = build_predictor(cfg)
    gen = torch.Generator().manual_seed(seed)
    param_deltas: dict[str, torch.Tensor] = {}
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
    """Stage one PseudoGradient per participant for `round_index`; return the staged updates."""
    staged: dict[str, PseudoGradient] = {}
    for i, pid in enumerate(participant_ids):
        pg = _toy_update(cfg, round_index=round_index, seed=seed_base + i)
        transport.submit_update(participant_id=pid, round_index=round_index, update=pg)
        staged[pid] = pg
    return staged


# --- the lifecycle hooks reflect a constructed-but-not-run coordinator ---


def test_coordinator_constructs_round_zero_open() -> None:
    cfg = _cfg()
    coord = Coordinator(cfg, transport=InProcessTransport())
    # __init__ commits round 0 and opens it: the driver is OPEN at round_index 0.
    assert coord.round_state() == RoundState.OPEN
    gs = coord.global_state()
    assert gs.round_index == 0
    assert gs.wmcp_version == WMCP_VERSION
    # The broadcast refs are 64-hex content hashes; the probe_hash is the 32-byte pin.
    assert len(gs.theta_ref.content_hash) == 64
    assert len(gs.phi_ref.content_hash) == 64
    assert len(gs.probe_hash) == 32


# --- payload restriction (INV-ACTIONHEAD-LOCAL): the broadcast/aggregated flat covers only θ/φ ---


def test_broadcast_and_aggregation_cover_only_encoder_predictor() -> None:
    cfg = _cfg()
    transport = InProcessTransport()
    coord = Coordinator(cfg, transport=InProcessTransport())
    expected = _global_param_count(cfg)

    # The coordinator's flat global params are exactly the encoder+predictor count (no action head).
    assert coord.global_params().numel() == expected

    # A staged update's delta has the same length (the canonical θ⊕φ flat), so the reduction stays θ/φ.
    coord2 = Coordinator(cfg, transport=transport)
    staged = _stage(transport, cfg, round_index=0, participant_ids=["c0", "c1"])
    for pg in staged.values():
        assert pg.delta.numel() == expected
    coord2.run(1)
    assert coord2.round_state() == RoundState.CLOSED


def test_action_head_group_cannot_enter_a_pseudogradient() -> None:
    # The aggregated payload can never carry an action head: build_pseudogradient fail-closes on it.
    with pytest.raises(ResidencyViolation) as exc:
        build_pseudogradient(
            {"encoder.w": torch.zeros(4), "action_head.toy.weight": torch.zeros(_D)},
            dataset_root=_ROOT,
            round_index=0,
        )
    assert exc.value.code == LensembleErrorCode.RESIDENCY_VIOLATION
    assert exc.value.tensor_role == "action_head"  # type: ignore[attr-defined]


# --- a clean round drives OPEN -> ... -> CLOSED and advances the committed global hash ---


def test_run_one_round_commits_and_closes() -> None:
    cfg = _cfg()
    transport = InProcessTransport()
    coord = Coordinator(cfg, transport=transport)
    h0 = coord.global_state_hash()
    _stage(transport, cfg, round_index=0, participant_ids=["c0", "c1", "c2"])

    coord.run(1)

    assert coord.round_state() == RoundState.CLOSED
    h1 = coord.global_state_hash()
    assert h1 != h0  # the committed global hash advanced (INV-CHECKPOINT-HASH)
    # the contribution ledger recorded the round with the contributing participants
    rec = coord.ledger_records()[-1]
    assert rec.round_index == 0
    assert rec.participants == ("c0", "c1", "c2")
    assert rec.global_model_hash == h1
    assert set(rec.dataset_roots) == {"c0", "c1", "c2"}


def test_run_two_rounds_advances_round_index() -> None:
    cfg = _cfg()
    transport = InProcessTransport()
    coord = Coordinator(cfg, transport=transport)
    _stage(transport, cfg, round_index=0, participant_ids=["c0", "c1"])
    _stage(transport, cfg, round_index=1, participant_ids=["c0", "c1"])

    coord.run(2)

    # After two requested rounds the last round is CLOSED at round_index 1.
    assert coord.round_state() == RoundState.CLOSED
    assert coord.global_state().round_index == 1
    assert len(coord.ledger_records()) == 2
    assert [r.round_index for r in coord.ledger_records()] == [0, 1]


# --- determinism: the same contributing set committed twice gives a bitwise-identical (θ_{t+1}, φ_{t+1}) ---


def test_determinism_same_contributing_set_commits_identical_hash() -> None:
    cfg = _cfg()

    def _commit_once() -> tuple[str, torch.Tensor]:
        transport = InProcessTransport()
        coord = Coordinator(cfg, transport=transport)
        _stage(transport, cfg, round_index=0, participant_ids=["c0", "c1"])
        coord.run(1)
        return coord.global_state_hash(), coord.global_params().clone()

    h_a, params_a = _commit_once()
    h_b, params_b = _commit_once()

    assert h_a == h_b  # bitwise-identical committed (θ_{t+1}, φ_{t+1}) content hash
    assert torch.equal(params_a, params_b)


# --- arrival-order independence: the canonical participant-id-order reduction is the same either way ---


def test_arrival_order_independence_commits_same_hash() -> None:
    cfg = _cfg()
    pg0 = _toy_update(cfg, round_index=0, seed=200)
    pg1 = _toy_update(cfg, round_index=0, seed=201)

    def _commit(order: list[tuple[str, PseudoGradient]]) -> str:
        transport = InProcessTransport()
        coord = Coordinator(cfg, transport=transport)
        for pid, pg in order:
            transport.submit_update(participant_id=pid, round_index=0, update=pg)
        coord.run(1)
        return coord.global_state_hash()

    h_forward = _commit([("c0", pg0), ("c1", pg1)])
    h_reversed = _commit([("c1", pg1), ("c0", pg0)])  # submitted in the opposite order
    assert h_forward == h_reversed


# --- corrupt reduction -> NonDeterministicAggregation AND the round ABORTED, the global hash unchanged ---


def test_corrupt_reduction_aborts_with_global_hash_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg()
    transport = InProcessTransport()
    coord = Coordinator(cfg, transport=transport)
    h_before = coord.global_state_hash()
    _stage(transport, cfg, round_index=0, participant_ids=["c0", "c1"])

    # Force a non-reproducible reduction: average_deltas returns a DIFFERENT tensor on each call, so the
    # determinism self-check's two recomputations disagree (INV-AGG-DETERMINISM).
    from lensemble.federation import outer as outer_mod

    counter = {"n": 0}

    def _nondeterministic_average(
        self: outer_mod.OuterOptimizer, deltas: object
    ) -> torch.Tensor:
        counter["n"] += 1
        return torch.full((coord.global_params().numel(),), float(counter["n"]))

    monkeypatch.setattr(
        outer_mod.OuterOptimizer, "average_deltas", _nondeterministic_average
    )

    with pytest.raises(NonDeterministicAggregation) as exc:
        coord.run(1)
    assert exc.value.code == LensembleErrorCode.AGG_NONDETERMINISTIC
    assert coord.round_state() == RoundState.ABORTED
    assert coord.global_state_hash() == h_before  # no partial commit (hash unchanged)


# --- below-quorum -> FaultToleranceExceeded, round ABORTED, global hash unchanged ---


def test_below_quorum_aborts_with_fault_tolerance_exceeded() -> None:
    cfg = _cfg()  # fault_tolerance_min_participants = 2
    transport = InProcessTransport()
    coord = Coordinator(cfg, transport=transport)
    h_before = coord.global_state_hash()
    # Stage only ONE update — below the quorum of 2.
    _stage(transport, cfg, round_index=0, participant_ids=["c0"])

    with pytest.raises(FaultToleranceExceeded) as exc:
        coord.run(1)
    assert exc.value.code == LensembleErrorCode.FAULT_TOLERANCE_EXCEEDED
    assert coord.round_state() == RoundState.ABORTED
    assert (
        coord.global_state_hash() == h_before
    )  # the round was discarded, hash unchanged


def test_no_updates_aborts_below_quorum() -> None:
    cfg = _cfg()
    transport = InProcessTransport()
    coord = Coordinator(cfg, transport=transport)
    # Stage nothing at all: 0 < quorum.
    with pytest.raises(FaultToleranceExceeded):
        coord.run(1)
    assert coord.round_state() == RoundState.ABORTED


# --- transport fetch round-trip: the broadcast refs resolve through the SAME transport ---


def test_committed_refs_fetch_round_trip_through_transport() -> None:
    cfg = _cfg()
    transport = InProcessTransport()
    coord = Coordinator(cfg, transport=transport)
    gs = coord.global_state()
    # The coordinator seeded the fetch store under each ref's content hash (weights_content_hash), so a
    # participant fetching θ_0/φ_0 round-trips and hash-verifies (INV-CHECKPOINT-HASH).
    theta = transport.fetch_params(gs.theta_ref)
    phi = transport.fetch_params(gs.phi_ref)
    assert all(
        k.startswith(("pos_embed", "patch_embed", "blocks", "norm")) for k in theta
    )
    total = sum(t.numel() for t in theta.values()) + sum(
        t.numel() for t in phi.values()
    )
    assert total == _global_param_count(cfg)

    # After a committed round the NEW refs also round-trip (the store is re-seeded for t+1).
    _stage(transport, cfg, round_index=0, participant_ids=["c0", "c1"])
    coord.run(1)
    gs1 = coord.global_state()
    transport.fetch_params(gs1.theta_ref)
    transport.fetch_params(gs1.phi_ref)


# --- ALIGNING is a MEASURED PASS-THROUGH: with per-participant embeddings wired (the #18 hook), the
# frame-drift report is measured but θ/φ are NOT corrected (the Procrustes fold-in is #18, out of scope) ---


class _DriftMeasuringCoordinator(Coordinator):
    """A Coordinator whose ALIGNING hook returns tiny per-participant probe embeddings (the #18 seam).

    Overriding ``_probe_embeddings`` makes ALIGNING measure a :class:`FrameDriftReport`; it still does
    NOT correct the gauge or mutate θ/φ (the backstop fold-in is #18, out of scope), so the measured
    round commits the same θ_{t+1} as the un-measured one.
    """

    def _probe_embeddings(self, t: int) -> dict[str, torch.Tensor] | None:
        gen = torch.Generator().manual_seed(7)
        d = 4
        return {
            "c0": torch.randn(6, d, generator=gen),
            "c1": torch.randn(6, d, generator=gen),
            "global": torch.randn(6, d, generator=gen),
        }


def test_aligning_measures_drift_but_does_not_alter_commit() -> None:
    cfg = _cfg()
    # Run with the measured-drift coordinator.
    transport_m = InProcessTransport()
    coord_m = _DriftMeasuringCoordinator(cfg, transport=transport_m)
    _stage(transport_m, cfg, round_index=0, participant_ids=["c0", "c1"])
    coord_m.run(1)
    measured_hash = coord_m.global_state_hash()
    report = coord_m.frame_drift_report()
    assert isinstance(report, FrameDriftReport)
    assert report.round_index == 0
    assert {"c0", "c1"} <= set(
        report.drift_from_global
    )  # drift_from_global was measured

    # Run the SAME contributing set through the plain coordinator (ALIGNING is a no-op pass-through).
    transport_p = InProcessTransport()
    coord_p = Coordinator(cfg, transport=transport_p)
    _stage(transport_p, cfg, round_index=0, participant_ids=["c0", "c1"])
    coord_p.run(1)

    # The measured drift did NOT change the committed θ_{t+1} (the gauge is not corrected here, #18).
    assert measured_hash == coord_p.global_state_hash()
    assert (
        coord_p.frame_drift_report() is None
    )  # the plain coordinator measured nothing


# --- the probe_path-set branch: probe_hash is the pinned probe's content hash (#22/#04 boundary) ---


def test_probe_hash_resolved_from_pinned_probe_path(tmp_path: Path) -> None:
    # Build + save a tiny pinned probe, then point cfg.data.probe_path at it.
    gen = torch.Generator().manual_seed(3)
    points = torch.randn(_D, _T, _C, _H, _W, generator=gen)
    landmark_idx = torch.arange(_D)
    targets = torch.randn(_D, _NUM_TOKENS, _D, generator=gen)
    probe = PublicProbe(
        points=points,
        landmark_idx=landmark_idx,
        landmark_targets=targets,
        content_hash=probe_content_hash(points, landmark_idx),
        probe_version=1,
    )
    probe_file = tmp_path / "probe.safetensors"
    save_probe(probe, probe_file)

    cfg = _cfg()
    cfg = dataclasses.replace(
        cfg, data=dataclasses.replace(cfg.data, probe_path=str(probe_file))
    )
    coord = Coordinator(cfg, transport=InProcessTransport())
    # The broadcast probe_hash equals the pinned probe's content hash (not the placeholder).
    assert coord.global_state().probe_hash == probe.content_hash
    assert coord.global_state().probe_hash != b"\x00" * 32
