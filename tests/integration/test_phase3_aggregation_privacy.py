"""Phase 3 secure-aggregation and DP reporting (#226)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pytest
import torch
from torch import Tensor

from lensemble.config import (
    LensembleConfig,
    Phase3ActionContract,
    Phase3ConsortiumManifest,
    Phase3Contact,
    Phase3DataDeclaration,
    Phase3DPPolicy,
    Phase3ModelAgreement,
    Phase3ObservationContract,
    Phase3ParticipantCapabilities,
    Phase3ParticipantDeclaration,
    Phase3PublicProbe,
    Phase3RuntimePolicy,
)
from lensemble.contracts import WMCP_VERSION
from lensemble.errors import SecureAggregationError
from lensemble.federation import (
    InProcessTransport,
    Phase3CoordinatorService,
    PseudoGradient,
    RoundState,
    build_phase3_aggregation_privacy_report,
    build_pseudogradient,
)
from lensemble.model import build_encoder, build_predictor

_D = 8
_NUM_TOKENS = 4
_T, _C, _H, _W = 2, 3, 4, 4
_ACTION_DIM = 2
_PARTICIPANTS = ("agent-a", "agent-b", "agent-c")


@dataclass(frozen=True)
class _ModelConfig:
    encoder: str = "vjepa2-vit-l"
    warm_start_release: str = "vjepa2-2.0"
    latent_dim: int = _D
    num_tokens: int = _NUM_TOKENS
    predictor_depth: int = 1
    predictor_width: int = _D
    wmcp_version: str = WMCP_VERSION
    encoder_frozen: bool = False
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
    backend: Literal["simulated", "masking", "tee"] = "simulated",
    secure_agg_threshold: int = 2,
    noise_multiplier: float = 1.0,
) -> LensembleConfig:
    base = LensembleConfig()
    fed = dataclasses.replace(
        base.federation,
        participant_count=3,
        fault_tolerance_min_participants=2,
        secure_agg_threshold=secure_agg_threshold,
        collect_timeout_s=0.01,
        transport="in_process",
        aggregation_backend=backend,
    )
    objective = dataclasses.replace(
        base.objective, target_stop_gradient=False, lambda_anc=0.0
    )
    privacy = dataclasses.replace(
        base.privacy,
        enabled=True,
        clip_norm=0.5,
        noise_multiplier=noise_multiplier,
        epsilon=8.0,
        delta=1e-5,
        accountant="rdp",
    )
    return dataclasses.replace(
        base,
        model=_ModelConfig(),  # type: ignore[arg-type]
        federation=fed,
        objective=objective,
        privacy=privacy,
        run_mode="coordinator",
    )


def _manifest(
    *, backend: Literal["simulated", "masking", "tee"] = "simulated"
) -> Phase3ConsortiumManifest:
    action = Phase3ActionContract(
        contract_id="toy-agent-action-v1",
        embodiment_id="toy-agent",
        kind="continuous",
        dim=_ACTION_DIM,
        low=(-1.0, -1.0),
        high=(1.0, 1.0),
        num_classes=None,
        units=("u", "u"),
        wmcp_version=WMCP_VERSION,
    )
    observation = Phase3ObservationContract(
        contract_id="toy-agent-window-v1",
        shape=(3, _T, _C, _H, _W),
        dtype="float32",
        frame_skip=1,
        wmcp_version=WMCP_VERSION,
    )
    probe = Phase3PublicProbe(
        probe_id="toy-agent-probe",
        version=1,
        content_hash="a" * 64,
    )
    participants = tuple(
        Phase3ParticipantDeclaration(
            participant_id=pid,
            role="trainer",
            contact=Phase3Contact(owner=pid, contact=f"{pid}@example.invalid"),
            action_contract=action,
            observation_contract=observation,
            accepted_probe_hash=probe.content_hash,
            accepted_probe_version=probe.version,
            capabilities=Phase3ParticipantCapabilities(
                network_transport=False,
                secure_aggregation_backends=("simulated", "masking", "tee"),
                dp_accountants=("rdp",),
                max_model_latent_dim=_D,
                resumable=True,
                private_data_mounts=True,
            ),
            data=Phase3DataDeclaration(
                data_ref=f"local://{pid}",
                format="hdf5",
                smoke_report_uri=f"artifact://{pid}/smoke.json",
                smoke_report_sha256=f"{idx:064x}",
                window_steps=2,
                heldout_policy="last toy episode",
                license="test-only",
                raw_data_crosses_boundary=False,
            ),
        )
        for idx, pid in enumerate(_PARTICIPANTS, start=1)
    )
    return Phase3ConsortiumManifest(
        consortium_id="phase3-privacy-test",
        run_id="phase3-privacy-smoke",
        coordinator_id="coordinator-test",
        created_at=datetime(2026, 6, 5, tzinfo=timezone.utc),
        model=Phase3ModelAgreement(
            model_family="LeWorldModel-claim-mode",
            wmcp_version=WMCP_VERSION,
            latent_dim=_D,
            num_tokens=_NUM_TOKENS,
            objective_target_stop_gradient=False,
            lambda_anc=0.0,
            base_checkpoint_ref=None,
            config_hash=None,
        ),
        public_probe=probe,
        runtime=Phase3RuntimePolicy(
            transport="in_process",
            secure_aggregation_backend=backend,
            secure_aggregation_required=True,
            dp_required=True,
            min_trainers=2,
            dropout_retry_budget=0,
        ),
        dp_policy=Phase3DPPolicy(
            enabled=True,
            clip_norm=0.5,
            noise_multiplier=1.0,
            epsilon=8.0,
            delta=1e-5,
            accountant="rdp",
        ),
        accepted_action_contracts=(action,),
        accepted_observation_contracts=(observation,),
        participants=participants,
        claim_boundary="test-only non-cryptographic aggregation/privacy smoke",
    )


def _root(participant_id: str) -> bytes:
    value = sum(ord(char) for char in participant_id) % 256
    return bytes([value]) * 32


def _toy_update(
    cfg: LensembleConfig,
    *,
    participant_id: str,
    round_index: int,
    seed: int,
) -> PseudoGradient:
    torch.manual_seed(seed)
    enc = build_encoder(cfg)
    pred = build_predictor(cfg)
    gen = torch.Generator().manual_seed(seed)
    param_deltas: dict[str, Tensor] = {}
    for name, tensor in enc.state_dict().items():
        param_deltas[f"encoder.{name}"] = 1e-3 * torch.randn(
            tensor.shape, generator=gen, dtype=torch.float32
        )
    for name, tensor in pred.state_dict().items():
        param_deltas[f"predictor.{name}"] = 1e-3 * torch.randn(
            tensor.shape, generator=gen, dtype=torch.float32
        )
    return build_pseudogradient(
        param_deltas,
        dataset_root=_root(participant_id),
        round_index=round_index,
        clipped=True,
    )


def _updates(cfg: LensembleConfig, *, count: int = 3) -> dict[str, PseudoGradient]:
    return {
        pid: _toy_update(cfg, participant_id=pid, round_index=0, seed=10 + idx)
        for idx, pid in enumerate(_PARTICIPANTS[:count])
    }


def _service(tmp_path: Path) -> Phase3CoordinatorService:
    return Phase3CoordinatorService(
        _cfg(),
        manifest=_manifest(),
        transport=InProcessTransport(),
        artifacts_dir=tmp_path / "artifacts",
        trace_path=tmp_path / "coordinator_trace.jsonl",
    )


def test_service_records_secure_aggregation_and_dp_report(tmp_path: Path) -> None:
    service = _service(tmp_path)
    cfg = _cfg()
    for participant_id in _PARTICIPANTS:
        service.join(
            participant_id=participant_id,
            endpoint=f"in-process://{participant_id}",
        )
    for participant_id in _PARTICIPANTS:
        service.assign_round(participant_id=participant_id)
    for participant_id, update in _updates(cfg, count=2).items():
        service.submit_update(participant_id=participant_id, update=update)

    assert service.close_round() is RoundState.CLOSED

    report = service.aggregation_privacy_report()
    assert report is not None
    assert report.secure_aggregation.secure_sum_consumed is True
    assert report.secure_aggregation.fallback_used is False
    assert report.secure_aggregation.contributing_count == 2
    assert report.dp_accounting.status == "accounted"
    assert report.dp_accounting.effective_dp is True
    assert report.dp_accounting.epsilon_spent is not None
    assert service.report().aggregation_privacy_report == report


def test_masking_backend_records_explicit_fallback_without_secure_sum() -> None:
    cfg = _cfg(backend="masking")
    report = build_phase3_aggregation_privacy_report(
        cfg, _manifest(backend="masking"), _updates(cfg, count=3), round_index=0
    )

    assert report.secure_aggregation.backend == "masking"
    assert report.secure_aggregation.backend_status == "explicit_fallback"
    assert report.secure_aggregation.secure_sum_consumed is False
    assert report.secure_aggregation.fallback_used is True
    assert report.secure_aggregation.fallback_reason


def test_secure_aggregation_report_rejects_below_threshold() -> None:
    cfg = _cfg(secure_agg_threshold=3)

    with pytest.raises(SecureAggregationError):
        build_phase3_aggregation_privacy_report(
            cfg, _manifest(), _updates(cfg, count=2), round_index=0
        )


def test_aggregation_privacy_report_redacts_individual_values() -> None:
    cfg = _cfg()
    report = build_phase3_aggregation_privacy_report(
        cfg, _manifest(), _updates(cfg, count=3), round_index=0
    )
    payload = report.model_dump_json()

    for participant_id in _PARTICIPANTS:
        assert participant_id not in payload
    for forbidden in ("dataset_root", "l2_norm", "delta_sha256", "update_delta"):
        assert forbidden not in payload
