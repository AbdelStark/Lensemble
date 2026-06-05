"""Phase 3 coordinator-service control plane and dropout lifecycle."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from lensemble.data import (
    Phase3DatasetProbeRegistry,
    phase3_registry_from_consortium_manifest,
)
from lensemble.errors import RoundError
from lensemble.federation import (
    InProcessTransport,
    Phase3CoordinatorService,
    PseudoGradient,
    RoundState,
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


def _cfg() -> LensembleConfig:
    base = LensembleConfig()
    fed = dataclasses.replace(
        base.federation,
        participant_count=3,
        fault_tolerance_min_participants=2,
        secure_agg_threshold=2,
        collect_timeout_s=0.01,
        transport="in_process",
        aggregation_backend="simulated",
    )
    objective = dataclasses.replace(
        base.objective, target_stop_gradient=False, lambda_anc=0.0
    )
    privacy = dataclasses.replace(
        base.privacy,
        enabled=True,
        clip_norm=0.1,
        noise_multiplier=0.0,
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


def _manifest(*, retry_budget: int = 1) -> Phase3ConsortiumManifest:
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
                secure_aggregation_backends=("simulated",),
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
        consortium_id="phase3-coordinator-test",
        run_id="phase3-coordinator-smoke",
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
            secure_aggregation_backend="simulated",
            secure_aggregation_required=False,
            dp_required=True,
            min_trainers=2,
            dropout_retry_budget=retry_budget,
        ),
        dp_policy=Phase3DPPolicy(
            enabled=True,
            clip_norm=0.1,
            noise_multiplier=0.0,
            epsilon=8.0,
            delta=1e-5,
            accountant="rdp",
        ),
        accepted_action_contracts=(action,),
        accepted_observation_contracts=(observation,),
        participants=participants,
        claim_boundary="test-only non-cryptographic coordinator-service smoke",
    )


def _service(tmp_path: Path, *, retry_budget: int = 1) -> Phase3CoordinatorService:
    manifest = _manifest(retry_budget=retry_budget)
    return Phase3CoordinatorService(
        _cfg(),
        manifest=manifest,
        transport=InProcessTransport(),
        artifacts_dir=tmp_path / "artifacts",
        trace_path=tmp_path / "coordinator_trace.jsonl",
    )


def _service_with_registry(
    tmp_path: Path, registry: Phase3DatasetProbeRegistry
) -> Phase3CoordinatorService:
    manifest = _manifest()
    return Phase3CoordinatorService(
        _cfg(),
        manifest=manifest,
        registry=registry,
        transport=InProcessTransport(),
        artifacts_dir=tmp_path / "artifacts",
        trace_path=tmp_path / "coordinator_trace.jsonl",
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
        param_deltas[f"encoder.{name}"] = 1e-2 * torch.randn(
            tensor.shape, generator=gen, dtype=torch.float32
        )
    for name, tensor in pred.state_dict().items():
        param_deltas[f"predictor.{name}"] = 1e-2 * torch.randn(
            tensor.shape, generator=gen, dtype=torch.float32
        )
    return build_pseudogradient(
        param_deltas,
        dataset_root=_root(participant_id),
        round_index=round_index,
        clipped=True,
    )


def _join_all(service: Phase3CoordinatorService) -> None:
    for participant_id in _PARTICIPANTS:
        service.join(
            participant_id=participant_id,
            endpoint=f"in-process://{participant_id}",
        )
        service.heartbeat(participant_id=participant_id)


def _assign_all(service: Phase3CoordinatorService) -> None:
    for participant_id in _PARTICIPANTS:
        gs = service.assign_round(participant_id=participant_id)
        assert gs.round_index == 0


def _events(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_service_completes_three_participant_smoke_with_one_dropout(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    _join_all(service)
    _assign_all(service)

    for seed, participant_id in enumerate(("agent-a", "agent-b"), start=10):
        service.submit_update(
            participant_id=participant_id,
            update=_toy_update(
                _cfg(), participant_id=participant_id, round_index=0, seed=seed
            ),
        )
    service.mark_dropout(participant_id="agent-c", reason="induced_dropout")

    assert service.close_round() is RoundState.CLOSED
    record = service.coordinator.ledger_records()[-1]
    assert record.participants == ("agent-a", "agent-b")
    assert service.report().round_index == 1
    assert service.report().dropout_policy.effective_quorum == 2
    trace = _events(tmp_path / "coordinator_trace.jsonl")
    assert {event["event"] for event in trace} >= {
        "service.started",
        "participant.joined",
        "round.assigned",
        "update.accepted",
        "participant.dropped",
        "round.closed",
    }


def test_service_consumes_shared_dataset_probe_registry(tmp_path: Path) -> None:
    manifest = _manifest()
    registry = phase3_registry_from_consortium_manifest(
        manifest, min_participant_count=3
    )

    service = _service_with_registry(tmp_path, registry)

    assert service.registry == registry
    assert service.report().participants[0].participant_id == "agent-a"


def test_late_join_is_rejected_after_assignment(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.join(participant_id="agent-a", endpoint="in-process://agent-a")
    service.assign_round(participant_id="agent-a")

    with pytest.raises(RoundError):
        service.join(participant_id="agent-b", endpoint="in-process://agent-b")

    trace = _events(tmp_path / "coordinator_trace.jsonl")
    assert trace[-1]["event"] == "participant.rejected"
    assert trace[-1]["reason"] == "late_join"


def test_duplicate_update_is_rejected(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _join_all(service)
    _assign_all(service)
    update = _toy_update(_cfg(), participant_id="agent-a", round_index=0, seed=10)
    service.submit_update(participant_id="agent-a", update=update)

    with pytest.raises(RoundError):
        service.submit_update(participant_id="agent-a", update=update)

    trace = _events(tmp_path / "coordinator_trace.jsonl")
    assert trace[-1]["event"] == "update.rejected"
    assert trace[-1]["reason"] == "duplicate_update"


def test_timeout_records_dropouts_and_retries_before_abort(tmp_path: Path) -> None:
    service = _service(tmp_path, retry_budget=1)
    _join_all(service)
    _assign_all(service)
    service.submit_update(
        participant_id="agent-a",
        update=_toy_update(_cfg(), participant_id="agent-a", round_index=0, seed=10),
    )

    assert service.close_round() is RoundState.ABORTED
    trace = _events(tmp_path / "coordinator_trace.jsonl")
    assert "participant.timeout" in {event["event"] for event in trace}
    assert trace[-1]["event"] == "round.retry"
    assert trace[-1]["payload"]["retry"] == 1

    assert service.close_round() is RoundState.ABORTED
    trace = _events(tmp_path / "coordinator_trace.jsonl")
    assert trace[-1]["event"] == "round.aborted"
    assert trace[-1]["reason"] == "below_quorum"


def test_operator_abort_flow_records_aborted_round(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.abort_round(reason="operator_abort")

    assert service.round_state() is RoundState.ABORTED
    trace = _events(tmp_path / "coordinator_trace.jsonl")
    assert trace[-1]["event"] == "round.aborted"
    assert trace[-1]["reason"] == "operator_abort"


def test_deterministic_round_close_is_order_independent(tmp_path: Path) -> None:
    def run_once(path: Path, order: tuple[str, str]) -> str:
        service = _service(path)
        _join_all(service)
        _assign_all(service)
        updates = {
            pid: _toy_update(_cfg(), participant_id=pid, round_index=0, seed=seed)
            for seed, pid in enumerate(("agent-a", "agent-b"), start=10)
        }
        for pid in order:
            service.submit_update(participant_id=pid, update=updates[pid])
        service.mark_dropout(participant_id="agent-c", reason="induced_dropout")
        assert service.close_round() is RoundState.CLOSED
        return service.coordinator.global_state_hash()

    first = run_once(tmp_path / "first", ("agent-a", "agent-b"))
    second = run_once(tmp_path / "second", ("agent-b", "agent-a"))

    assert first == second
