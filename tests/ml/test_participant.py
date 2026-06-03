"""Participant local round — the H-step inner loop that emits a privatized, bound PseudoGradient (#43).

Covers the RFC-0013 §1 acceptance criteria on a tiny CPU V-JEPA config: INV-PROBE-PIN (a probe-hash
mismatch raises ProbeError), INV-WARMSTART-T0 (a round-0 encoder-hash mismatch raises GaugeError — the
#43 criterion pins GaugeError, vs SPEC 03 §7's CheckpointIntegrityError), INV-SKETCH-CONSISTENCY
(round_seed != sketch_seed raises GaugeError), the released PseudoGradient (encoder/predictor groups only,
one 32-byte dataset_root, l2_norm == ||delta||, clipped, finite, INV-DP-BOUND held), the join() rejoiner
path (committed GlobalState returned; a tampered θ ref raises CheckpointIntegrityError), and INV-RESIDENCY
(delta is the only tensor field reachable on the carrier).

NOTE: placed in tests/ml (NOT tests/federation): the §8 CI gate scans tests/{unit,property,integration,
ml,e2e,regression}; a tests/federation directory would not run. The participant is a model-bearing
runtime object, so tests/ml is the right home (mirrors tests/ml/test_harness.py).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pytest
import torch
from torch import Tensor

from lensemble.config import LensembleConfig
from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
from lensemble.data import guard_egress
from lensemble.data.episode import Window
from lensemble.data.probe import PublicProbe, probe_content_hash
from lensemble.errors import (
    CheckpointIntegrityError,
    GaugeError,
    LensembleErrorCode,
    ProbeError,
    ResidencyViolation,
    RoundError,
)
from lensemble.federation import (
    GlobalState,
    InProcessTransport,
    ParamRef,
    Participant,
    build_pseudogradient,
)
from lensemble.federation.transport import weights_content_hash
from lensemble.model import build_encoder, build_predictor

# --- a tiny CPU model config carrying BOTH the real ModelConfig fields AND the V-JEPA shape fields
# build_encoder/predictor/action_head read (mirrors tests/ml/test_harness.py::_EvalModelConfig — the same
# LensembleConfig->architecture bridge gap the CLI papers over). ---

_D = 8
_NUM_TOKENS = (
    4  # (num_frames//tubelet) * (image_size//patch_size)**2 = (2//2)*(4//2)**2 = 4
)
_T, _C, _H, _W = 2, 3, 4, 4
_ACTION_DIM = 2
_NUM_STEPS = 2  # window obs is (num_steps + 1) clips
_K = 8  # k >= d landmarks (INV-PROBE-PIN coverage)
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


def _cfg(**model_overrides: object) -> LensembleConfig:
    base = LensembleConfig()
    # Small inner horizon + small clip so the inner loop runs fast and DP clipping engages.
    fed = dataclasses.replace(
        base.federation, inner_horizon=2, quantize_pseudo_gradient=False
    )
    priv = dataclasses.replace(
        base.privacy, clip_norm=0.1, noise_multiplier=0.5, enabled=True
    )
    model = _ModelConfig(**model_overrides)  # type: ignore[arg-type]
    return dataclasses.replace(
        base, model=model, federation=fed, privacy=priv, run_mode="participant"
    )


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


def _build_probe(seed: int = 0) -> PublicProbe:
    """A tiny probe with k >= d clip-shaped landmarks; landmark_targets derived from a fresh encoder."""
    gen = torch.Generator().manual_seed(seed)
    points = torch.randn(_K, _T, _C, _H, _W, generator=gen)
    landmark_idx = torch.arange(_K)
    enc = build_encoder(_cfg())
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


def _global_weights(
    cfg: LensembleConfig,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Build a fresh encoder+predictor and return their plain state_dicts (the global θ_t, φ_t)."""
    torch.manual_seed(0)
    encoder = build_encoder(cfg)
    predictor = build_predictor(cfg)
    return dict(encoder.state_dict()), dict(predictor.state_dict())


class _TestParticipant(Participant):
    """A Participant whose #22 data-layer hooks are wired to tiny fixtures (the toy seam)."""

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


def _seed_transport(
    cfg: LensembleConfig,
    probe: PublicProbe,
    *,
    round_index: int = 1,
    sketch_seed: int = 7,
) -> tuple[InProcessTransport, GlobalState, dict[str, Tensor], dict[str, Tensor]]:
    """Build a transport committed with a GlobalState whose θ/φ refs resolve to fresh global weights."""
    theta, phi = _global_weights(cfg)
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


# --- INV-PROBE-PIN: a mismatched probe hash is refused (ProbeError) ---


def test_local_round_rejects_probe_hash_mismatch() -> None:
    cfg = _cfg()
    probe = _build_probe()
    transport, gs, _, _ = _seed_transport(cfg, probe, sketch_seed=7)
    # The participant pins a DIFFERENT probe than the one in GlobalState.probe_hash.
    other_probe = _build_probe(seed=99)
    assert other_probe.content_hash != probe.content_hash
    p = _TestParticipant(
        cfg,
        participant_id="c0",
        transport=transport,
        probe=other_probe,
        windows=_build_windows(),
    )
    with pytest.raises(ProbeError) as exc:
        p.local_round(gs, round_seed=7)
    assert exc.value.code == LensembleErrorCode.PROBE_INVALID
    assert exc.value.remediation


# --- INV-WARMSTART-T0: a round-0 encoder hash that differs from the pinned warm-start raises GaugeError ---


def test_local_round_t0_warmstart_drift_raises_gauge_error() -> None:
    cfg = _cfg()
    probe = _build_probe()
    transport, _, _, _ = _seed_transport(cfg, probe, round_index=0, sketch_seed=7)
    gs = transport.recover_global_state(participant_id="c0")
    assert gs.round_index == 0
    # Pin a warm-start hash the loaded global encoder cannot match (a deliberate drift).
    p = _TestParticipant(
        cfg,
        participant_id="c0",
        transport=transport,
        probe=probe,
        windows=_build_windows(),
        warmstart_hash="0" * 64,
    )
    with pytest.raises(GaugeError) as exc:
        p.local_round(gs, round_seed=7)
    assert exc.value.code == LensembleErrorCode.FRAME_DRIFT_EXCEEDED
    assert exc.value.remediation


def test_local_round_t0_warmstart_match_succeeds() -> None:
    # The positive INV-WARMSTART-T0 branch: when the pinned warm-start hash equals the loaded round-0
    # encoder hash, the round runs and releases a PseudoGradient.
    cfg = _cfg()
    probe = _build_probe()
    transport, _, theta, _ = _seed_transport(cfg, probe, round_index=0, sketch_seed=7)
    gs = transport.recover_global_state(participant_id="c0")
    # The pinned hash IS the loaded global-encoder hash (the canonical safetensors content hash).
    from lensemble.model import build_encoder as _be
    from lensemble.model.encoder import encoder_content_hash as _ech

    enc = _be(cfg)
    enc.load_state_dict(theta, strict=True)
    pinned = _ech(enc)
    p = _TestParticipant(
        cfg,
        participant_id="c0",
        transport=transport,
        probe=probe,
        windows=_build_windows(),
        warmstart_hash=pinned,
    )
    pg = p.local_round(gs, round_seed=7)
    assert pg.clipped is True and pg.round_index == 0


# --- INV-SKETCH-CONSISTENCY: round_seed != sketch_seed raises GaugeError ---


def test_local_round_rejects_sketch_seed_mismatch() -> None:
    cfg = _cfg()
    probe = _build_probe()
    transport, gs, _, _ = _seed_transport(cfg, probe, sketch_seed=7)
    p = _TestParticipant(
        cfg,
        participant_id="c0",
        transport=transport,
        probe=probe,
        windows=_build_windows(),
    )
    with pytest.raises(GaugeError) as exc:
        p.local_round(gs, round_seed=8)  # != gs.sketch_seed (7)
    assert exc.value.code == LensembleErrorCode.GAUGE_FAILED


# --- a successful local_round returns a well-formed PseudoGradient ---


def test_local_round_returns_well_formed_pseudogradient() -> None:
    cfg = _cfg()
    probe = _build_probe()
    transport, gs, theta, phi = _seed_transport(cfg, probe, sketch_seed=7)
    p = _TestParticipant(
        cfg,
        participant_id="c0",
        transport=transport,
        probe=probe,
        windows=_build_windows(),
    )
    pg = p.local_round(gs, round_seed=7)

    # covers ONLY encoder + predictor groups: flat length == total encoder+predictor param count.
    expected_numel = sum(t.numel() for t in theta.values()) + sum(
        t.numel() for t in phi.values()
    )
    assert pg.delta.numel() == expected_numel

    # exactly one 32-byte dataset_root (INV-COMMIT-BINDING)
    assert pg.dataset_root == _ROOT and len(pg.dataset_root) == 32

    # l2_norm == ||delta|| (post_init holds), clipped, correct round, finite delta
    assert pg.l2_norm == pytest.approx(float(pg.delta.norm()))
    assert pg.clipped is True
    assert pg.quantized is False
    assert pg.round_index == gs.round_index
    assert bool(torch.isfinite(pg.delta).all())
    assert pg.delta.dtype == torch.float32


def test_action_head_group_cannot_enter_the_released_delta() -> None:
    # The released delta covers only (θ, φ); injecting an action_head.* group into build_pseudogradient
    # is rejected fail-closed (INV-ACTIONHEAD-LOCAL) — proving the head can never be federated.
    with pytest.raises(ResidencyViolation) as exc:
        build_pseudogradient(
            {
                "encoder.w": torch.zeros(4),
                "action_head.toy.weight": torch.zeros(_D),
            },
            dataset_root=_ROOT,
            round_index=1,
        )
    assert exc.value.code == LensembleErrorCode.RESIDENCY_VIOLATION
    assert exc.value.tensor_role == "action_head"  # type: ignore[attr-defined]


def test_local_round_dp_bound_holds_with_tiny_clip() -> None:
    # A tiny clip_norm forces the post-clip projection; the INV-DP-BOUND assert inside local_round must
    # hold (no AssertionError), and the released delta is finite/fp32 with an honest l2_norm.
    cfg = _cfg()  # clip_norm = 0.1 (tiny)
    probe = _build_probe()
    transport, gs, _, _ = _seed_transport(cfg, probe, sketch_seed=7)
    p = _TestParticipant(
        cfg,
        participant_id="c-tiny",
        transport=transport,
        probe=probe,
        windows=_build_windows(),
    )
    pg = p.local_round(gs, round_seed=7)  # must not raise the INV-DP-BOUND assertion
    assert pg.clipped is True
    assert bool(torch.isfinite(pg.delta).all())


def test_local_round_with_quantization_sets_quantized() -> None:
    cfg = _cfg()
    cfg = dataclasses.replace(
        cfg,
        federation=dataclasses.replace(cfg.federation, quantize_pseudo_gradient=True),
    )
    probe = _build_probe()
    transport, gs, _, _ = _seed_transport(cfg, probe, sketch_seed=7)
    p = _TestParticipant(
        cfg,
        participant_id="c-q",
        transport=transport,
        probe=probe,
        windows=_build_windows(),
    )
    pg = p.local_round(gs, round_seed=7)
    assert pg.quantized is True
    assert pg.l2_norm == pytest.approx(float(pg.delta.norm()))


def test_local_round_runs_with_lambda_anc_zero() -> None:
    # The bare LeJEPA objective (anchor=None) path: lambda_anc == 0 means no FrameAnchor is constructed.
    cfg = _cfg()
    cfg = dataclasses.replace(
        cfg, objective=dataclasses.replace(cfg.objective, lambda_anc=0.0)
    )
    probe = _build_probe()
    transport, gs, _, _ = _seed_transport(cfg, probe, sketch_seed=7)
    p = _TestParticipant(
        cfg,
        participant_id="c-anc0",
        transport=transport,
        probe=probe,
        windows=_build_windows(),
    )
    pg = p.local_round(gs, round_seed=7)
    assert pg.clipped is True


# --- join(): the rejoiner-recovery path ---


def test_join_returns_committed_global_state() -> None:
    cfg = _cfg()
    probe = _build_probe()
    transport, gs, _, _ = _seed_transport(cfg, probe, sketch_seed=7)
    p = _TestParticipant(
        cfg,
        participant_id="joiner",
        transport=transport,
        probe=probe,
        windows=_build_windows(),
    )
    recovered = p.join("coordinator://local")
    assert recovered == gs
    assert recovered.round_index == gs.round_index


def test_join_rejects_a_tampered_theta_ref() -> None:
    cfg = _cfg()
    probe = _build_probe()
    transport, gs, _, _ = _seed_transport(cfg, probe, sketch_seed=7)
    # Tamper the stored θ weights under the SAME content hash so the recomputed hash diverges.
    transport.corrupt_stored_weights(
        gs.theta_ref.content_hash, {"encoder.bogus": torch.ones(3)}
    )
    p = _TestParticipant(
        cfg,
        participant_id="joiner",
        transport=transport,
        probe=probe,
        windows=_build_windows(),
    )
    with pytest.raises(CheckpointIntegrityError) as exc:
        p.join("coordinator://local")
    assert exc.value.code == LensembleErrorCode.CHECKPOINT_INTEGRITY


def test_local_round_rejects_a_tampered_fetch() -> None:
    cfg = _cfg()
    probe = _build_probe()
    transport, gs, _, _ = _seed_transport(cfg, probe, sketch_seed=7)
    transport.corrupt_stored_weights(
        gs.theta_ref.content_hash, {"encoder.bogus": torch.ones(3)}
    )
    p = _TestParticipant(
        cfg,
        participant_id="c0",
        transport=transport,
        probe=probe,
        windows=_build_windows(),
    )
    with pytest.raises(CheckpointIntegrityError):
        p.local_round(gs, round_seed=7)


# --- INV-RESIDENCY: the released carrier exposes only `delta` as a tensor ---


def test_released_pseudogradient_passes_the_egress_guard() -> None:
    cfg = _cfg()
    probe = _build_probe()
    transport, gs, _, _ = _seed_transport(cfg, probe, sketch_seed=7)
    p = _TestParticipant(
        cfg,
        participant_id="c0",
        transport=transport,
        probe=probe,
        windows=_build_windows(),
    )
    pg = p.local_round(gs, round_seed=7)
    # The only tensor field is `delta`; the egress guard permits the carrier and ONLY its delta.
    assert guard_egress(pg) is None
    tensor_fields = [
        name for name, value in vars(pg).items() if isinstance(value, torch.Tensor)
    ]
    assert tensor_fields == ["delta"]


# --- the DP non-private honesty path (enabled=False) ---


def test_local_round_with_dp_disabled_releases_unclipped_norm() -> None:
    cfg = _cfg()
    cfg = dataclasses.replace(
        cfg, privacy=dataclasses.replace(cfg.privacy, enabled=False)
    )
    probe = _build_probe()
    transport, gs, _, _ = _seed_transport(cfg, probe, sketch_seed=7)
    p = _TestParticipant(
        cfg,
        participant_id="c-nodp",
        transport=transport,
        probe=probe,
        windows=_build_windows(),
    )
    pg = p.local_round(
        gs, round_seed=7
    )  # the INV-DP-BOUND assert is skipped when DP is off
    assert pg.clipped is True  # the carrier still records the gauge/clip flag
    assert pg.l2_norm == pytest.approx(float(pg.delta.norm()))


# --- the #22 data-layer-boundary default hooks fail closed (no toy source wired) ---


def test_default_data_hooks_fail_closed() -> None:
    cfg = _cfg()  # cfg.data.probe_path is None by default
    transport = InProcessTransport()
    p = Participant(cfg, participant_id="bare", transport=transport)
    with pytest.raises(RoundError):
        p._pinned_probe()  # no probe_path configured
    with pytest.raises(RoundError):
        p._local_windows()  # no loader wired (#22)
    with pytest.raises(RoundError):
        p._dataset_root()  # no commitment wired (#22)
    with pytest.raises(RoundError):
        p._action_spec()  # no embodiment spec wired (#22)
    assert p._warmstart_hash() is None  # default disables the t=0 check


def test_default_pinned_probe_loads_from_config_path(tmp_path: object) -> None:
    from pathlib import Path

    from lensemble.data.probe import save_probe

    probe = _build_probe()
    probe_file = Path(tmp_path) / "probe.safetensors"  # type: ignore[arg-type]
    save_probe(probe, probe_file)
    cfg = _cfg()
    cfg = dataclasses.replace(
        cfg, data=dataclasses.replace(cfg.data, probe_path=str(probe_file))
    )
    p = Participant(cfg, participant_id="loader", transport=InProcessTransport())
    loaded = p._pinned_probe()
    assert loaded.content_hash == probe.content_hash


def test_inner_loop_with_no_windows_fails_closed() -> None:
    cfg = _cfg()
    probe = _build_probe()
    transport, gs, _, _ = _seed_transport(cfg, probe, sketch_seed=7)
    p = _TestParticipant(
        cfg,
        participant_id="empty",
        transport=transport,
        probe=probe,
        windows=[],  # the inner loop has nothing to run on
    )
    with pytest.raises(RoundError):
        p.local_round(gs, round_seed=7)


def test_train_local_without_a_data_source_fails_closed() -> None:
    # train_local resolves windows through the #22 data layer (cfg.data.data_source); with no source the
    # default _local_windows fails closed citing #22 (#167). The full green path is in tests/e2e.
    from lensemble.federation import train_local

    with pytest.raises(RoundError):
        train_local(_cfg())  # cfg.data.data_source is None by default


# --- coordinator-side transport methods (the #42 surface, smoke-covered here) ---


def test_transport_coordinator_side_round_trip() -> None:
    cfg = _cfg()
    probe = _build_probe()
    transport, gs, _, _ = _seed_transport(cfg, probe, sketch_seed=7)
    # broadcast_round_open republishes the committed state; collect_updates returns submitted updates.
    transport.broadcast_round_open(gs)
    assert transport.recover_global_state(participant_id="any") == gs
    p = _TestParticipant(
        cfg,
        participant_id="c0",
        transport=transport,
        probe=probe,
        windows=_build_windows(),
    )
    pg = p.local_round(gs, round_seed=7)
    transport.submit_update(participant_id="c0", round_index=gs.round_index, update=pg)
    collected = transport.collect_updates(gs.round_index)
    assert set(collected) == {"c0"}
    assert collected["c0"] is pg
    assert transport.collect_updates(999) == {}  # no updates for an unknown round


def test_fetch_params_unknown_ref_fails_closed() -> None:
    transport = InProcessTransport()
    ref = ParamRef(content_hash="f" * 64, locator="mem://missing")
    with pytest.raises(CheckpointIntegrityError):
        transport.fetch_params(ref)


# --- GlobalState / ParamRef validation ---


def test_global_state_and_param_ref_validate() -> None:
    good = ParamRef(content_hash="a" * 64, locator="mem://x")
    assert good.content_hash == "a" * 64
    with pytest.raises(ValueError):
        ParamRef(content_hash="abc", locator="mem://x")  # not 64 hex
    with pytest.raises(ValueError):
        ParamRef(content_hash="A" * 64, locator="mem://x")  # uppercase rejected
    with pytest.raises(ValueError):
        ParamRef(content_hash="a" * 64, locator="")  # empty locator

    theta = ParamRef(content_hash="a" * 64, locator="mem://t")
    phi = ParamRef(content_hash="b" * 64, locator="mem://p")
    with pytest.raises(ValueError):
        GlobalState(
            theta_ref=theta,
            phi_ref=phi,
            round_index=-1,
            sketch_seed=1,
            probe_hash=b"\x00" * 32,
            wmcp_version=WMCP_VERSION,
        )
    with pytest.raises(ValueError):
        GlobalState(
            theta_ref=theta,
            phi_ref=phi,
            round_index=0,
            sketch_seed=1,
            probe_hash=b"\x00" * 16,  # not 32 bytes
            wmcp_version=WMCP_VERSION,
        )
    with pytest.raises(ValueError):
        GlobalState(
            theta_ref=theta,
            phi_ref=phi,
            round_index=0,
            sketch_seed=1,
            probe_hash=b"\x00" * 32,
            wmcp_version="",  # empty
        )


def test_recover_global_state_before_commit_fails_closed() -> None:
    transport = InProcessTransport()
    with pytest.raises(CheckpointIntegrityError):
        transport.recover_global_state(participant_id="c0")
