"""Phase 3 reproducible consortium-run orchestration.

The long-run smoke here is deliberately small enough for CI, but it exercises
the operational Phase 3 coordinator-service loop over four simulated trust
domains for multiple closed rounds and emits a residency-safe run report.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
from pydantic import BaseModel, ConfigDict, Field

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
from lensemble.config.manifest import config_hash
from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
from lensemble.data.episode import Window
from lensemble.data.phase3 import (
    Phase3DatasetProbeRegistry,
    phase3_registry_from_consortium_manifest,
    validate_phase3_registry_against_manifest,
)
from lensemble.data.probe import PublicProbe, probe_content_hash, save_probe
from lensemble.eval.jepa_metrics import (
    evaluate_jepa_windows,
    frame_drift_deg_from_updates,
    load_round_models,
)
from lensemble.federation.agent import ParticipantFactory, Phase3ParticipantAgent
from lensemble.federation.participant import Participant
from lensemble.federation.round import RoundState
from lensemble.federation.service import Phase3CoordinatorService
from lensemble.federation.transport import InProcessTransport, Transport
from lensemble.model import build_action_head, build_encoder

PHASE3_LONG_RUN_REPORT_SCHEMA_VERSION = 1

_D = 8
_NUM_TOKENS = 4
_T, _C, _H, _W = 2, 3, 4, 4
_ACTION_DIM = 2
_WINDOW_STEPS = 2
_PARTICIPANTS = ("phase3-so100-a", "phase3-so100-b", "phase3-so100-c", "phase3-so100-d")
_GENERATED_AT = datetime(2026, 6, 5, 0, 0, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class _SmokeModelConfig:
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


class Phase3ArtifactTargets(BaseModel):
    """Declared artifact publication targets for a Phase 3 run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_repo: str = Field(min_length=1)
    dataset_repo: str = Field(min_length=1)
    reports_prefix: str = Field(min_length=1)
    publication_mode: Literal["local_smoke", "hf_jobs_release"]


class Phase3RunShape(BaseModel):
    """Pinned run shape declared before launching the Phase 3 orchestration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    participant_count: int = Field(ge=1)
    rounds: int = Field(ge=1)
    inner_horizon: int = Field(ge=1)
    model_latent_dim: int = Field(ge=1)
    model_num_tokens: int = Field(ge=1)
    root_seed: int
    dp_enabled: bool
    dp_clip_norm: float = Field(gt=0.0)
    dp_noise_multiplier: float = Field(ge=0.0)
    dp_epsilon: float = Field(gt=0.0)
    dp_delta: float = Field(gt=0.0, lt=1.0)
    dp_accountant: str = Field(min_length=1)
    secure_aggregation_backend: str = Field(min_length=1)
    secure_aggregation_threshold: int = Field(ge=1)
    eval_budget: str = Field(min_length=1)
    artifact_targets: Phase3ArtifactTargets


class Phase3RoundRunSummary(BaseModel):
    """Residency-safe summary for one closed Phase 3 round.

    The four learning metrics are optional: the local CI smoke leaves them unset, while a real HF Jobs
    consortium run (``run_phase3_consortium`` with ``compute_metrics=True``) fills them in from the
    committed global checkpoint and the public probe (RFC-0005 §4, RFC-0002). They carry no raw data.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    round_index: int = Field(ge=0)
    state: Literal["closed"]
    contributing_count: int = Field(ge=1)
    global_model_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    aggregation_backend_status: str = Field(min_length=1)
    dp_epsilon_spent: float | None = Field(default=None, ge=0.0)
    val_pred: float | None = Field(default=None)
    val_sigreg: float | None = Field(default=None)
    effective_rank: float | None = Field(default=None, ge=0.0)
    frame_drift_deg: float | None = Field(default=None, ge=0.0)


class Phase3ParticipantRunSummary(BaseModel):
    """Residency-safe lifecycle counts for one simulated trust domain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    participant_id: str = Field(min_length=1)
    joined: bool
    assigned_rounds: int = Field(ge=0)
    submitted_rounds: int = Field(ge=0)
    dropped_rounds: int = Field(ge=0)


class Phase3LongRunReport(BaseModel):
    """Machine-readable Phase 3 long-run orchestration report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = PHASE3_LONG_RUN_REPORT_SCHEMA_VERSION
    consortium_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    generated_at: datetime
    run_shape: Phase3RunShape
    dry_run_checks: tuple[str, ...] = Field(min_length=1)
    closed_rounds: int = Field(ge=0)
    target_rounds: int = Field(ge=1)
    completed_target: bool
    final_global_model_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_manifest_path: str = Field(min_length=1)
    trace_path: str = Field(min_length=1)
    ledger_path: str = Field(min_length=1)
    checkpoint_dir: str = Field(min_length=1)
    rounds: tuple[Phase3RoundRunSummary, ...]
    participants: tuple[Phase3ParticipantRunSummary, ...]
    blockers: tuple[str, ...] = ()
    claim_boundary: str = Field(min_length=1)


def phase3_long_run_smoke_config(
    *, rounds: int = 10, probe_path: Path | None = None
) -> LensembleConfig:
    """Return the deterministic tiny config used by the Phase 3 long-run smoke."""

    base = LensembleConfig()
    federation = dataclasses.replace(
        base.federation,
        participant_count=len(_PARTICIPANTS),
        inner_horizon=_WINDOW_STEPS,
        num_rounds=rounds,
        fault_tolerance_min_participants=3,
        secure_agg_threshold=3,
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
        clip_norm=0.5,
        noise_multiplier=1.0,
        epsilon=8.0,
        delta=1e-5,
        accountant="rdp",
    )
    data = dataclasses.replace(
        base.data,
        probe_path=str(probe_path) if probe_path is not None else base.data.probe_path,
        window_steps=_WINDOW_STEPS,
        residency_enforced=True,
    )
    return dataclasses.replace(
        base,
        model=_SmokeModelConfig(),  # type: ignore[arg-type]
        federation=federation,
        objective=objective,
        privacy=privacy,
        data=data,
        run_mode="coordinator",
    )


def _participant_config(
    cfg: LensembleConfig, *, participant_id: str
) -> LensembleConfig:
    data = dataclasses.replace(
        cfg.data,
        data_source=f"local://phase3-smoke/{participant_id}",
        format="hdf5",
        window_steps=_WINDOW_STEPS,
        residency_enforced=True,
    )
    return dataclasses.replace(cfg, data=data, run_mode="participant")


def phase3_long_run_smoke_manifest(
    cfg: LensembleConfig,
    *,
    public_probe_hash: str | None = None,
) -> Phase3ConsortiumManifest:
    """Return a four-participant manifest matching the long-run smoke config."""

    action = Phase3ActionContract(
        contract_id="phase3-smoke-action-v1",
        embodiment_id="phase3-smoke-arm",
        kind="continuous",
        dim=_ACTION_DIM,
        low=(-1.0, -1.0),
        high=(1.0, 1.0),
        num_classes=None,
        units=("u", "u"),
        wmcp_version=cfg.model.wmcp_version,
    )
    observation = Phase3ObservationContract(
        contract_id="phase3-smoke-window-v1",
        shape=(_WINDOW_STEPS + 1, _T, _C, _H, _W),
        dtype="float32",
        frame_skip=1,
        wmcp_version=cfg.model.wmcp_version,
    )
    probe = Phase3PublicProbe(
        probe_id="phase3-long-run-smoke-probe",
        version=1,
        content_hash=public_probe_hash or "b" * 64,
    )
    participants = tuple(
        Phase3ParticipantDeclaration(
            participant_id=participant_id,
            role="trainer",
            contact=Phase3Contact(
                owner=f"Phase 3 smoke trust domain {idx}",
                contact=f"phase3-smoke-{idx}@example.invalid",
            ),
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
                data_ref=f"local://phase3-smoke/{participant_id}",
                format="hdf5",
                smoke_report_uri=f"artifact://phase3-smoke/{participant_id}/dataset_smoke.json",
                smoke_report_sha256=f"{idx:064x}",
                window_steps=_WINDOW_STEPS,
                heldout_policy="last synthetic local episode held out",
                license="test-only",
                raw_data_crosses_boundary=False,
            ),
        )
        for idx, participant_id in enumerate(_PARTICIPANTS, start=1)
    )
    return Phase3ConsortiumManifest(
        consortium_id="lensemble-phase3-long-run-smoke",
        run_id="phase3-long-run-smoke-v1",
        coordinator_id="phase3-long-run-coordinator",
        created_at=_GENERATED_AT,
        model=Phase3ModelAgreement(
            model_family="LeWorldModel-claim-mode-tiny-smoke",
            wmcp_version=cfg.model.wmcp_version,
            latent_dim=int(cfg.model.latent_dim),
            num_tokens=int(cfg.model.num_tokens),
            objective_target_stop_gradient=False,
            lambda_anc=0.0,
            base_checkpoint_ref=None,
            config_hash=config_hash(asdict(cfg)),
        ),
        public_probe=probe,
        runtime=Phase3RuntimePolicy(
            transport="in_process",
            secure_aggregation_backend="simulated",
            secure_aggregation_required=True,
            dp_required=True,
            min_trainers=3,
            dropout_retry_budget=0,
        ),
        dp_policy=Phase3DPPolicy(
            enabled=True,
            clip_norm=float(cfg.privacy.clip_norm),
            noise_multiplier=float(cfg.privacy.noise_multiplier),
            epsilon=float(cfg.privacy.epsilon),
            delta=float(cfg.privacy.delta),
            accountant=cfg.privacy.accountant,
        ),
        accepted_action_contracts=(action,),
        accepted_observation_contracts=(observation,),
        participants=participants,
        claim_boundary=(
            "Tiny in-process Phase 3 orchestration smoke; demonstrates reproducible "
            "coordinator/participant lifecycle, not paper-scale robotics performance."
        ),
    )


def _action_spec() -> ActionSpec:
    return ActionSpec(
        embodiment_id="phase3-smoke-arm",
        kind=ActionKind.CONTINUOUS,
        dim=_ACTION_DIM,
        low=(-1.0, -1.0),
        high=(1.0, 1.0),
        num_classes=None,
        units=("u", "u"),
        wmcp_version=WMCP_VERSION,
    )


def _build_public_probe(cfg: LensembleConfig) -> PublicProbe:
    gen = torch.Generator().manual_seed(703)
    points = torch.randn(_D, _T, _C, _H, _W, generator=gen)
    landmark_idx = torch.arange(_D)
    encoder = build_encoder(cfg)
    targets = encoder(points[landmark_idx]).tokens.detach()
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
            embodiment_id="phase3-smoke-arm",
        ),
        Window(
            obs=torch.randn(_WINDOW_STEPS + 1, _T, _C, _H, _W, generator=gen),
            actions=torch.randn(_WINDOW_STEPS, _ACTION_DIM, generator=gen),
            num_steps=_WINDOW_STEPS,
            embodiment_id="phase3-smoke-arm",
        ),
    ]


class _SmokeAgentParticipant(Participant):
    def __init__(
        self,
        config: LensembleConfig,
        *,
        participant_id: str,
        transport: Transport,
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


def _agent_factory(
    *,
    probe: PublicProbe,
    windows_by_participant: dict[str, list[Window]],
) -> ParticipantFactory:
    def build(
        config: LensembleConfig,
        participant_id: str,
        transport: Transport,
    ) -> Participant:
        return _SmokeAgentParticipant(
            config,
            participant_id=participant_id,
            transport=transport,
            probe=probe,
            windows=windows_by_participant[participant_id],
            dataset_root=_root(participant_id),
        )

    return build  # type: ignore[return-value]


def _build_agents(
    cfg: LensembleConfig,
    *,
    manifest: Phase3ConsortiumManifest,
    registry: Phase3DatasetProbeRegistry,
    transport: InProcessTransport,
    probe: PublicProbe,
    run_dir: Path,
) -> dict[str, Phase3ParticipantAgent]:
    windows_by_participant = {
        participant_id: _windows(2000 + idx)
        for idx, participant_id in enumerate(_PARTICIPANTS)
    }
    factory = _agent_factory(
        probe=probe,
        windows_by_participant=windows_by_participant,
    )
    return {
        participant_id: Phase3ParticipantAgent(
            _participant_config(cfg, participant_id=participant_id),
            manifest=manifest,
            participant_id=participant_id,
            transport=transport,
            state_dir=run_dir / "participant-agents",
            coordinator_endpoint="in-process://phase3-long-run-coordinator",
            registry=registry,
            participant_factory=factory,
            emit_observability=True,
        )
        for participant_id in _PARTICIPANTS
    }


def phase3_long_run_smoke_inputs(
    *, run_dir: Path, rounds: int = 10
) -> Phase3ConsortiumInputs:
    """Build the deterministic tiny ``Phase3ConsortiumInputs`` used by the CI smoke + metrics test."""

    run_dir = Path(run_dir)
    probe = _build_public_probe(phase3_long_run_smoke_config(rounds=rounds))
    probe_path = run_dir / "phase3_public_probe.safetensors"
    cfg = phase3_long_run_smoke_config(rounds=rounds, probe_path=probe_path)
    manifest = phase3_long_run_smoke_manifest(
        cfg, public_probe_hash=probe.content_hash.hex()
    )
    registry = phase3_registry_from_consortium_manifest(manifest)
    windows_by_participant = {
        participant_id: _windows(2000 + idx)
        for idx, participant_id in enumerate(_PARTICIPANTS)
    }
    factory = _agent_factory(probe=probe, windows_by_participant=windows_by_participant)
    return Phase3ConsortiumInputs(
        cfg=cfg,
        participant_configs={
            participant_id: _participant_config(cfg, participant_id=participant_id)
            for participant_id in _PARTICIPANTS
        },
        manifest=manifest,
        registry=registry,
        probe=probe,
        probe_path=probe_path,
        participant_ids=_PARTICIPANTS,
        eval_windows=tuple(_windows(9_999)),
        eval_action_spec=_action_spec(),
        participant_factory=factory,
    )


def run_phase3_long_run_smoke(
    *,
    run_dir: Path,
    rounds: int = 10,
    generated_at: datetime = _GENERATED_AT,
    compute_metrics: bool = False,
) -> Phase3LongRunReport:
    """Run a deterministic four-participant, multi-round Phase 3 smoke.

    With ``compute_metrics=True`` the report's round rows carry real (tiny-model) per-round
    ``val_pred``/``val_sigreg``/``effective_rank``/``frame_drift_deg`` measured off the committed global
    checkpoints — the same path the HF Jobs consortium launcher exercises at non-toy scale.
    """

    inputs = phase3_long_run_smoke_inputs(run_dir=run_dir, rounds=rounds)
    return run_phase3_consortium(
        inputs,
        run_dir=run_dir,
        rounds=rounds,
        generated_at=generated_at,
        metric_windows=2,
        compute_metrics=compute_metrics,
        claim_boundary=(
            "Local deterministic Phase 3 long-run smoke: proves orchestration, "
            "artifact/report generation, secure-sum reporting, and DP accounting "
            "for a tiny model. It is not a published HF Jobs robotics result."
        ),
    )


@dataclass(frozen=True)
class _RoundMetrics:
    """Per-round learning metrics, all optional so an unmeasured round records absence, not zero."""

    val_pred: float | None = None
    val_sigreg: float | None = None
    effective_rank: float | None = None
    frame_drift_deg: float | None = None


_NO_METRICS = _RoundMetrics()


def _round_learning_metrics(
    cfg: LensembleConfig,
    *,
    run_dir: Path,
    round_index: int,
    probe: PublicProbe,
    transport: InProcessTransport,
    eval_windows: Sequence[Window],
    eval_action_spec: ActionSpec,
    metric_windows: int,
) -> _RoundMetrics:
    """Evaluate the just-committed global model: val_pred/val_sigreg/effective_rank + frame_drift_deg."""

    artifacts_dir = run_dir / "coordinator-artifacts"
    final_ckpt = artifacts_dir / f"round-{round_index + 1:05d}"
    prior_ckpt = artifacts_dir / f"round-{round_index:05d}"
    if not (final_ckpt / "weights.safetensors").exists():
        return _NO_METRICS
    encoder, predictor = load_round_models(cfg, final_ckpt)
    action_head = build_action_head(cfg, eval_action_spec).eval()
    window_metrics = evaluate_jepa_windows(
        cfg,
        encoder=encoder,
        predictor=predictor,
        action_head=action_head,
        windows=eval_windows,
        max_windows=metric_windows,
    )
    frame_drift_deg = frame_drift_deg_from_updates(
        cfg,
        prior_checkpoint_dir=prior_ckpt,
        final_encoder=encoder,
        probe=probe,
        updates=transport.collect_updates(round_index),
        prior_round=round_index,
    )
    if window_metrics is None:
        return _RoundMetrics(frame_drift_deg=frame_drift_deg)
    return _RoundMetrics(
        val_pred=window_metrics.val_pred,
        val_sigreg=window_metrics.val_sigreg,
        effective_rank=window_metrics.effective_rank,
        frame_drift_deg=frame_drift_deg,
    )


def _run_consortium_rounds(
    cfg: LensembleConfig,
    *,
    service: Phase3CoordinatorService,
    transport: InProcessTransport,
    agents: dict[str, Phase3ParticipantAgent],
    participant_ids: tuple[str, ...],
    rounds: int,
    run_dir: Path,
    probe: PublicProbe,
    eval_windows: Sequence[Window],
    eval_action_spec: ActionSpec,
    compute_metrics: bool,
    metric_windows: int,
) -> tuple[list[Phase3RoundRunSummary], list[str], dict[str, int], dict[str, int]]:
    """Drive ``rounds`` closed federated rounds, optionally measuring per-round learning metrics."""

    round_summaries: list[Phase3RoundRunSummary] = []
    blockers: list[str] = []
    submitted_counts = {participant_id: 0 for participant_id in participant_ids}
    assigned_counts = {participant_id: 0 for participant_id in participant_ids}
    for round_index in range(rounds):
        for participant_id in participant_ids:
            service.assign_round(participant_id=participant_id)
            assigned_counts[participant_id] += 1
        for participant_id in participant_ids:
            agents[participant_id].run_assigned_round()
            update = transport.collect_updates(round_index)[participant_id]
            service.submit_update(participant_id=participant_id, update=update)
            submitted_counts[participant_id] += 1
        close_state = service.close_round()
        if close_state is not RoundState.CLOSED:
            blockers.append(
                f"round {round_index} did not close: coordinator returned {close_state.value}"
            )
            break
        privacy_report = service.aggregation_privacy_report()
        record = service.coordinator.ledger_records()[-1]
        metrics = (
            _round_learning_metrics(
                cfg,
                run_dir=run_dir,
                round_index=round_index,
                probe=probe,
                transport=transport,
                eval_windows=eval_windows,
                eval_action_spec=eval_action_spec,
                metric_windows=metric_windows,
            )
            if compute_metrics
            else _NO_METRICS
        )
        # Stream the per-round gauge/learning metrics to stdout so a live GPU run is observable (and an
        # early divergence killable) via `hf jobs logs -f` — the report file is only written at the end.
        print(
            f"[round {round_index}] eff_rank={metrics.effective_rank} "
            f"val_pred={metrics.val_pred} val_sigreg={metrics.val_sigreg} "
            f"frame_drift_deg={metrics.frame_drift_deg} "
            f"hash={record.global_model_hash[:12]}",
            flush=True,
        )
        round_summaries.append(
            Phase3RoundRunSummary(
                round_index=round_index,
                state="closed",
                contributing_count=len(record.participants),
                global_model_hash=record.global_model_hash,
                aggregation_backend_status=(
                    privacy_report.secure_aggregation.backend_status
                    if privacy_report is not None
                    else "not_reported"
                ),
                dp_epsilon_spent=(
                    privacy_report.dp_accounting.epsilon_spent
                    if privacy_report is not None
                    else None
                ),
                val_pred=metrics.val_pred,
                val_sigreg=metrics.val_sigreg,
                effective_rank=metrics.effective_rank,
                frame_drift_deg=metrics.frame_drift_deg,
            )
        )
    return round_summaries, blockers, assigned_counts, submitted_counts


def _assemble_long_run_report(
    cfg: LensembleConfig,
    *,
    manifest: Phase3ConsortiumManifest,
    service: Phase3CoordinatorService,
    run_dir: Path,
    generated_at: datetime,
    target_rounds: int,
    round_summaries: list[Phase3RoundRunSummary],
    blockers: list[str],
    assigned_counts: dict[str, int],
    submitted_counts: dict[str, int],
    participant_ids: tuple[str, ...],
    run_manifest_path: Path,
    claim_boundary: str,
    artifact_targets: Phase3ArtifactTargets | None = None,
    eval_budget: str | None = None,
) -> Phase3LongRunReport:
    return Phase3LongRunReport(
        consortium_id=manifest.consortium_id,
        run_id=manifest.run_id,
        generated_at=generated_at,
        run_shape=_run_shape(
            cfg, artifact_targets=artifact_targets, eval_budget=eval_budget
        ),
        dry_run_checks=(
            "manifest_validated",
            "dataset_registry_validated_against_manifest",
            f"public_probe_hash_pinned:{manifest.public_probe.content_hash}",
            "participant_mounts_declared_no_raw_boundary",
            "participant_agents_preflighted",
            "participant_agents_released_updates",
            "secure_aggregation_threshold_validated",
            "dp_policy_validated",
            "artifact_publication_targets_declared",
            "report_publication_path_writable",
        ),
        closed_rounds=len(round_summaries),
        target_rounds=target_rounds,
        completed_target=len(round_summaries) >= target_rounds,
        final_global_model_hash=service.coordinator.global_state_hash(),
        config_hash=config_hash(asdict(cfg)),
        run_manifest_path=str(run_manifest_path),
        trace_path=str(service.trace_path),
        ledger_path=str(run_dir / "coordinator-artifacts" / "ledger.jsonl"),
        checkpoint_dir=str(run_dir / "coordinator-artifacts"),
        rounds=tuple(round_summaries),
        participants=tuple(
            Phase3ParticipantRunSummary(
                participant_id=participant_id,
                joined=True,
                assigned_rounds=assigned_counts[participant_id],
                submitted_rounds=submitted_counts[participant_id],
                dropped_rounds=0,
            )
            for participant_id in participant_ids
        ),
        blockers=tuple(blockers),
        claim_boundary=claim_boundary,
    )


@dataclass(frozen=True)
class Phase3ConsortiumInputs:
    """Everything ``run_phase3_consortium`` needs to drive a real (non-smoke) Phase 3 run.

    The HF Jobs entry point builds these from mounted participant-local data refs: a coordinator config,
    a per-participant config (each pinned to its own private data ref), the agreed manifest + registry,
    the pinned public probe, and a held-out eval split (disjoint from every participant silo) used only
    for the residency-safe per-round learning metrics. ``participant_factory`` is left ``None`` for real
    runs so each agent's default ``Participant`` streams windows from its own ``data_source``; tests pass
    a factory that injects in-memory windows for determinism.
    """

    cfg: LensembleConfig
    participant_configs: dict[str, LensembleConfig]
    manifest: Phase3ConsortiumManifest
    registry: Phase3DatasetProbeRegistry
    probe: PublicProbe
    probe_path: Path
    participant_ids: tuple[str, ...]
    eval_windows: tuple[Window, ...]
    eval_action_spec: ActionSpec
    participant_factory: ParticipantFactory | None = None


def run_phase3_consortium(
    inputs: Phase3ConsortiumInputs,
    *,
    run_dir: Path,
    rounds: int,
    generated_at: datetime,
    metric_windows: int = 8,
    compute_metrics: bool = True,
    artifact_targets: Phase3ArtifactTargets | None = None,
    eval_budget: str | None = None,
    enable_backstop: bool = False,
    claim_boundary: str,
) -> Phase3LongRunReport:
    """Drive the full Phase 3 consortium runtime and emit a long-run report with per-round metrics.

    This is the library entry point the HF Jobs launcher calls: it runs the networked
    coordinator-service + sovereign participant-agent runtime (not the claim-MVP path) over the supplied
    participant configs, and — when ``compute_metrics`` is set — measures ``val_pred``/``val_sigreg``/
    ``effective_rank`` on the held-out eval split plus per-round ``frame_drift_deg`` from released
    pseudo-gradients. No raw participant trajectory ever leaves a participant boundary.

    ``enable_backstop`` (#262) turns the LIVE Layer-3 Procrustes backstop ON in the coordinator: each
    over-threshold participant's encoder terminal frame + predictor I/O are aligned to the shared round-0
    reference before the outer step. Default OFF (the measured pass-through); the real anchored-federation
    run sets it ON.
    """

    run_dir = Path(run_dir)
    _prepare_run_dir(run_dir)
    save_probe(inputs.probe, inputs.probe_path)
    validate_phase3_registry_against_manifest(inputs.registry, inputs.manifest)
    transport = InProcessTransport()
    service = Phase3CoordinatorService(
        inputs.cfg,
        manifest=inputs.manifest,
        registry=inputs.registry,
        transport=transport,
        artifacts_dir=run_dir / "coordinator-artifacts",
        trace_path=run_dir / "phase3_coordinator_trace.jsonl",
        enable_backstop=enable_backstop,
    )
    coordinator_endpoint = f"in-process://{inputs.manifest.coordinator_id}"
    agents = {
        participant_id: Phase3ParticipantAgent(
            inputs.participant_configs[participant_id],
            manifest=inputs.manifest,
            participant_id=participant_id,
            transport=transport,
            state_dir=run_dir / "participant-agents",
            coordinator_endpoint=coordinator_endpoint,
            registry=inputs.registry,
            participant_factory=inputs.participant_factory,
            emit_observability=True,
        )
        for participant_id in inputs.participant_ids
    }
    for participant_id in inputs.participant_ids:
        service.join(
            participant_id=participant_id,
            endpoint=f"in-process://{participant_id}",
        )
    for agent in agents.values():
        agent.preflight()

    round_summaries, blockers, assigned_counts, submitted_counts = (
        _run_consortium_rounds(
            inputs.cfg,
            service=service,
            transport=transport,
            agents=agents,
            participant_ids=inputs.participant_ids,
            rounds=rounds,
            run_dir=run_dir,
            probe=inputs.probe,
            eval_windows=inputs.eval_windows,
            eval_action_spec=inputs.eval_action_spec,
            compute_metrics=compute_metrics,
            metric_windows=metric_windows,
        )
    )
    run_manifest_path = _write_run_manifest(
        inputs.cfg,
        inputs.manifest,
        run_dir=run_dir,
        generated_at=generated_at,
        artifact_targets=artifact_targets,
        eval_budget=eval_budget,
    )
    return _assemble_long_run_report(
        inputs.cfg,
        manifest=inputs.manifest,
        service=service,
        run_dir=run_dir,
        generated_at=generated_at,
        target_rounds=rounds,
        round_summaries=round_summaries,
        blockers=blockers,
        assigned_counts=assigned_counts,
        submitted_counts=submitted_counts,
        participant_ids=inputs.participant_ids,
        run_manifest_path=run_manifest_path,
        claim_boundary=claim_boundary,
        artifact_targets=artifact_targets,
        eval_budget=eval_budget,
    )


def _prepare_run_dir(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("coordinator-artifacts", "participant-agents"):
        shutil.rmtree(run_dir / child, ignore_errors=True)
    for name in (
        "phase3_coordinator_trace.jsonl",
        "phase3_public_probe.safetensors",
        "phase3_run_manifest.json",
    ):
        path = run_dir / name
        if path.exists():
            path.unlink()


def write_phase3_long_run_report(report: Phase3LongRunReport, path: Path) -> Path:
    """Write a Phase 3 long-run report as canonical JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_phase3_long_run_report_json(report) + "\n", encoding="utf-8")
    return path


def parse_phase3_long_run_report(raw: dict[str, Any]) -> Phase3LongRunReport:
    """Parse raw long-run report JSON, gating future schema versions first."""

    version = raw.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version > PHASE3_LONG_RUN_REPORT_SCHEMA_VERSION
    ):
        from lensemble.errors import LensembleErrorCode, SchemaVersionMismatch

        raise SchemaVersionMismatch(
            f"Phase 3 long-run report schema_version {version!r} exceeds reader max "
            f"{PHASE3_LONG_RUN_REPORT_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation="read with a build supporting this Phase 3 long-run report schema",
        )
    return Phase3LongRunReport.model_validate(raw)


def load_phase3_long_run_report(path: Path) -> Phase3LongRunReport:
    """Load and validate a Phase 3 long-run report JSON file."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_phase3_long_run_report(raw)


def to_phase3_long_run_report_json(report: Phase3LongRunReport) -> str:
    """Canonical JSON for a Phase 3 long-run report."""

    return json.dumps(
        report.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


_SMOKE_EVAL_BUDGET = "deferred to #228; smoke reserves synthetic downstream eval budget"


def _default_artifact_targets() -> Phase3ArtifactTargets:
    return Phase3ArtifactTargets(
        model_repo="hf://models/abdelstark/lensemble-phase3-consortium-checkpoint",
        dataset_repo="hf://datasets/abdelstark/lensemble-phase3-consortium-data",
        reports_prefix="reports/phase3/",
        publication_mode="local_smoke",
    )


def _run_shape(
    cfg: LensembleConfig,
    *,
    artifact_targets: Phase3ArtifactTargets | None = None,
    eval_budget: str | None = None,
) -> Phase3RunShape:
    return Phase3RunShape(
        participant_count=int(cfg.federation.participant_count),
        rounds=int(cfg.federation.num_rounds),
        inner_horizon=int(cfg.federation.inner_horizon),
        model_latent_dim=int(cfg.model.latent_dim),
        model_num_tokens=int(cfg.model.num_tokens),
        root_seed=int(cfg.determinism.root_seed),
        dp_enabled=bool(cfg.privacy.enabled),
        dp_clip_norm=float(cfg.privacy.clip_norm),
        dp_noise_multiplier=float(cfg.privacy.noise_multiplier),
        dp_epsilon=float(cfg.privacy.epsilon),
        dp_delta=float(cfg.privacy.delta),
        dp_accountant=cfg.privacy.accountant,
        secure_aggregation_backend=cfg.federation.aggregation_backend,
        secure_aggregation_threshold=int(cfg.federation.secure_agg_threshold),
        eval_budget=eval_budget if eval_budget is not None else _SMOKE_EVAL_BUDGET,
        artifact_targets=artifact_targets
        if artifact_targets is not None
        else _default_artifact_targets(),
    )


def _write_run_manifest(
    cfg: LensembleConfig,
    manifest: Phase3ConsortiumManifest,
    *,
    run_dir: Path,
    generated_at: datetime,
    artifact_targets: Phase3ArtifactTargets | None = None,
    eval_budget: str | None = None,
) -> Path:
    payload = {
        "schema": "phase3-long-run-manifest/v1",
        "generated_at": generated_at.isoformat(),
        "config_hash": config_hash(asdict(cfg)),
        "consortium_id": manifest.consortium_id,
        "run_id": manifest.run_id,
        "run_shape": _run_shape(
            cfg, artifact_targets=artifact_targets, eval_budget=eval_budget
        ).model_dump(mode="json"),
    }
    path = run_dir / "phase3_run_manifest.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def _root(participant_id: str) -> bytes:
    value = sum(ord(char) for char in participant_id) % 256
    return bytes([value]) * 32


__all__ = [
    "PHASE3_LONG_RUN_REPORT_SCHEMA_VERSION",
    "Phase3ArtifactTargets",
    "Phase3ConsortiumInputs",
    "Phase3LongRunReport",
    "Phase3ParticipantRunSummary",
    "Phase3RoundRunSummary",
    "Phase3RunShape",
    "load_phase3_long_run_report",
    "parse_phase3_long_run_report",
    "phase3_long_run_smoke_config",
    "phase3_long_run_smoke_manifest",
    "run_phase3_consortium",
    "run_phase3_long_run_smoke",
    "to_phase3_long_run_report_json",
    "write_phase3_long_run_report",
]
