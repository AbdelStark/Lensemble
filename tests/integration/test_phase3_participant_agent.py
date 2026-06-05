"""Phase 3 participant-agent runtime preflight, round execution, and resume."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
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
from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
from lensemble.data import (
    Phase3DatasetProbeRegistry,
    phase3_registry_from_consortium_manifest,
)
from lensemble.data.episode import Window
from lensemble.data.probe import PublicProbe, probe_content_hash
from lensemble.errors import ConfigError
from lensemble.federation import (
    GlobalState,
    InProcessTransport,
    ParamRef,
    Participant,
    Phase3ParticipantAgent,
)
from lensemble.federation.agent import ParticipantFactory
from lensemble.federation.transport import weights_content_hash
from lensemble.model import build_encoder, build_predictor

_D = 8
_NUM_TOKENS = 4
_T, _C, _H, _W = 2, 3, 4, 4
_ACTION_DIM = 2
_WINDOW_STEPS = 2
_ROOT_BASE = bytes.fromhex("2a" * 32)


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


def _cfg(participant_id: str) -> LensembleConfig:
    base = LensembleConfig()
    fed = dataclasses.replace(
        base.federation,
        inner_horizon=2,
        participant_count=2,
        fault_tolerance_min_participants=2,
        secure_agg_threshold=2,
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
    data = dataclasses.replace(
        base.data,
        data_source=f"local://{participant_id}",
        format="hdf5",
        window_steps=_WINDOW_STEPS,
        residency_enforced=True,
    )
    return dataclasses.replace(
        base,
        model=_ModelConfig(),  # type: ignore[arg-type]
        federation=fed,
        objective=objective,
        privacy=privacy,
        data=data,
        run_mode="participant",
    )


def _action_spec() -> ActionSpec:
    return ActionSpec(
        embodiment_id="toy-agent",
        kind=ActionKind.CONTINUOUS,
        dim=_ACTION_DIM,
        low=(-1.0, -1.0),
        high=(1.0, 1.0),
        num_classes=None,
        units=("u", "u"),
        wmcp_version=WMCP_VERSION,
    )


def _action_contract() -> Phase3ActionContract:
    spec = _action_spec()
    return Phase3ActionContract(
        contract_id="toy-agent-action-v1",
        embodiment_id=spec.embodiment_id,
        kind="continuous",
        dim=spec.dim,
        low=spec.low,
        high=spec.high,
        num_classes=spec.num_classes,
        units=spec.units,
        wmcp_version=spec.wmcp_version,
    )


def _observation_contract() -> Phase3ObservationContract:
    return Phase3ObservationContract(
        contract_id="toy-agent-window-v1",
        shape=(_WINDOW_STEPS + 1, _T, _C, _H, _W),
        dtype="float32",
        frame_skip=1,
        wmcp_version=WMCP_VERSION,
    )


def _build_probe(seed: int = 0) -> PublicProbe:
    gen = torch.Generator().manual_seed(seed)
    points = torch.randn(_D, _T, _C, _H, _W, generator=gen)
    landmark_idx = torch.arange(_D)
    enc = build_encoder(_cfg("agent-a"))
    targets = enc(points[landmark_idx]).tokens.detach()
    return PublicProbe(
        points=points,
        landmark_idx=landmark_idx,
        landmark_targets=targets,
        content_hash=probe_content_hash(points, landmark_idx),
        probe_version=1,
    )


def _windows(seed: int) -> list[Window]:
    gen = torch.Generator().manual_seed(seed)
    return [
        Window(
            obs=torch.randn(_WINDOW_STEPS + 1, _T, _C, _H, _W, generator=gen),
            actions=torch.randn(_WINDOW_STEPS, _ACTION_DIM, generator=gen),
            num_steps=_WINDOW_STEPS,
            embodiment_id="toy-agent",
        ),
        Window(
            obs=torch.randn(_WINDOW_STEPS + 1, _T, _C, _H, _W, generator=gen),
            actions=torch.randn(_WINDOW_STEPS, _ACTION_DIM, generator=gen),
            num_steps=_WINDOW_STEPS,
            embodiment_id="toy-agent",
        ),
    ]


def _manifest(
    probe: PublicProbe, *, probe_hash: str | None = None
) -> Phase3ConsortiumManifest:
    action = _action_contract()
    observation = _observation_contract()
    public_probe = Phase3PublicProbe(
        probe_id="toy-agent-probe",
        version=probe.probe_version,
        content_hash=probe_hash or probe.content_hash.hex(),
    )
    participants = tuple(
        Phase3ParticipantDeclaration(
            participant_id=pid,
            role="trainer",
            contact=Phase3Contact(owner=pid, contact=f"{pid}@example.invalid"),
            action_contract=action,
            observation_contract=observation,
            accepted_probe_hash=public_probe.content_hash,
            accepted_probe_version=public_probe.version,
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
                window_steps=_WINDOW_STEPS,
                heldout_policy="last toy episode",
                license="test-only",
                raw_data_crosses_boundary=False,
            ),
        )
        for idx, pid in enumerate(("agent-a", "agent-b"), start=1)
    )
    return Phase3ConsortiumManifest(
        consortium_id="phase3-agent-test",
        run_id="phase3-agent-smoke",
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
        public_probe=public_probe,
        runtime=Phase3RuntimePolicy(
            transport="in_process",
            secure_aggregation_backend="simulated",
            secure_aggregation_required=False,
            dp_required=True,
            min_trainers=2,
            dropout_retry_budget=0,
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
        claim_boundary="test-only non-cryptographic participant-agent smoke",
    )


class _AgentParticipant(Participant):
    def __init__(
        self,
        config: LensembleConfig,
        *,
        participant_id: str,
        transport: InProcessTransport,
        probe: PublicProbe,
        windows: list[Window],
        dataset_root: bytes,
    ) -> None:
        super().__init__(config, participant_id=participant_id, transport=transport)
        self._probe = probe
        self._windows = windows
        self._root = dataset_root

    def _pinned_probe(self) -> PublicProbe:
        return self._probe

    def _local_windows(self) -> list[Window]:
        return self._windows

    def _local_windows_for_horizon(self, horizon: int) -> list[Window]:
        return self._windows[: max(1, horizon)]

    def _dataset_root(self) -> bytes:
        return self._root

    def _action_spec(self) -> ActionSpec:
        return _action_spec()


def _factory(
    *,
    probe: PublicProbe,
    windows_by_participant: dict[str, list[Window]],
) -> ParticipantFactory:
    def build(
        config: LensembleConfig,
        participant_id: str,
        transport: InProcessTransport,
    ) -> Participant:
        root = bytearray(_ROOT_BASE)
        root[-1] = len(participant_id)
        return _AgentParticipant(
            config,
            participant_id=participant_id,
            transport=transport,
            probe=probe,
            windows=windows_by_participant[participant_id],
            dataset_root=bytes(root),
        )

    return build  # type: ignore[return-value]


def _global_weights(
    cfg: LensembleConfig,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    torch.manual_seed(0)
    encoder = build_encoder(cfg)
    predictor = build_predictor(cfg)
    return dict(encoder.state_dict()), dict(predictor.state_dict())


def _seed_transport(
    cfg: LensembleConfig, probe: PublicProbe
) -> tuple[InProcessTransport, GlobalState]:
    theta, phi = _global_weights(cfg)
    gs = GlobalState(
        theta_ref=ParamRef(
            content_hash=weights_content_hash(theta), locator="mem://theta"
        ),
        phi_ref=ParamRef(content_hash=weights_content_hash(phi), locator="mem://phi"),
        round_index=1,
        sketch_seed=7,
        probe_hash=probe.content_hash,
        wmcp_version=WMCP_VERSION,
    )
    transport = InProcessTransport()
    transport.commit(gs, theta_weights=theta, phi_weights=phi)
    return transport, gs


def _agent(
    *,
    participant_id: str,
    transport: InProcessTransport,
    manifest: Phase3ConsortiumManifest,
    probe: PublicProbe,
    state_dir: Path,
    registry: Phase3DatasetProbeRegistry | None = None,
) -> Phase3ParticipantAgent:
    return Phase3ParticipantAgent(
        _cfg(participant_id),
        manifest=manifest,
        participant_id=participant_id,
        transport=transport,
        state_dir=state_dir,
        coordinator_endpoint="in-process://coordinator",
        registry=registry,
        participant_factory=_factory(
            probe=probe,
            windows_by_participant={
                "agent-a": _windows(11),
                "agent-b": _windows(12),
            },
        ),
    )


def test_preflight_refuses_probe_mismatch_before_registering(tmp_path: Path) -> None:
    probe = _build_probe()
    manifest = _manifest(probe, probe_hash="3" * 64)
    transport, _ = _seed_transport(_cfg("agent-a"), probe)
    agent = _agent(
        participant_id="agent-a",
        transport=transport,
        manifest=manifest,
        probe=probe,
        state_dir=tmp_path,
    )

    with pytest.raises(ConfigError) as exc:
        agent.preflight()

    assert exc.value.code == "config_invalid"
    assert getattr(transport, "_registered") == {}


def test_preflight_consumes_shared_dataset_probe_registry(tmp_path: Path) -> None:
    probe = _build_probe()
    manifest = _manifest(probe)
    registry = phase3_registry_from_consortium_manifest(
        manifest,
        min_participant_count=2,
        window_counts={"agent-a": 2, "agent-b": 2},
        episode_counts={"agent-a": 1, "agent-b": 1},
    )
    transport, _ = _seed_transport(_cfg("agent-a"), probe)
    agent = _agent(
        participant_id="agent-a",
        transport=transport,
        manifest=manifest,
        probe=probe,
        state_dir=tmp_path,
        registry=registry,
    )

    preflight = agent.preflight()

    assert "dataset_probe_registry" in preflight.checks


def test_two_participant_agents_complete_round_and_emit_safe_state(
    tmp_path: Path,
) -> None:
    probe = _build_probe()
    manifest = _manifest(probe)
    transport, _ = _seed_transport(_cfg("agent-a"), probe)

    result_a = _agent(
        participant_id="agent-a",
        transport=transport,
        manifest=manifest,
        probe=probe,
        state_dir=tmp_path,
    ).run_assigned_round()
    result_b = _agent(
        participant_id="agent-b",
        transport=transport,
        manifest=manifest,
        probe=probe,
        state_dir=tmp_path,
    ).run_assigned_round()

    updates = transport.collect_updates(1)
    assert set(updates) == {"agent-a", "agent-b"}
    assert result_a.preflight.local_window_count == 2
    assert result_b.state.submitted is True
    assert result_a.state.update_sha256 != result_b.state.update_sha256

    raw_state = json.loads(Path(result_a.state_path).read_text(encoding="utf-8"))
    forbidden = {
        "obs",
        "actions",
        "latents",
        "embeddings",
        "action_head",
        "private_action_head",
    }
    assert forbidden.isdisjoint(raw_state)
    with safe_open(result_a.delta_path, framework="pt") as f:  # type: ignore[no-untyped-call]
        assert list(f.keys()) == ["delta"]
    round_dir = Path(result_a.state_path).parent
    assert (round_dir / "lensemble.log.jsonl").exists()
    assert (round_dir / "metrics.jsonl").exists()


def test_resume_replays_same_committed_update_hash(tmp_path: Path) -> None:
    probe = _build_probe()
    manifest = _manifest(probe)
    transport, _ = _seed_transport(_cfg("agent-a"), probe)
    agent = _agent(
        participant_id="agent-a",
        transport=transport,
        manifest=manifest,
        probe=probe,
        state_dir=tmp_path,
    )
    first = agent.run_assigned_round()

    resumed = agent.run_assigned_round(resume=True)

    assert resumed.resumed is True
    assert resumed.state.update_sha256 == first.state.update_sha256
    assert resumed.state.update_delta_sha256 == first.state.update_delta_sha256
    update = transport.collect_updates(1)["agent-a"]
    assert update.delta.numel() == first.state.delta_numel
