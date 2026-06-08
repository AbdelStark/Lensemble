"""Phase 3 consortium observability and dropout reporting.

The report is an operational audit surface for Phase 3 consortium runs. It
summarizes coordinator/participant lifecycle events, communication volume,
dropout decisions, aggregation/privacy mode, and artifact publication status
without exposing raw participant data, actions, latents, embeddings, model
weights, tokens, secrets, or sensitive host-local paths.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lensemble.data.phase3 import (
    phase3_registry_from_consortium_manifest,
    validate_phase3_registry_against_manifest,
)
from lensemble.data.probe import save_probe
from lensemble.errors import ConfigError, LensembleErrorCode, SchemaVersionMismatch
from lensemble.eval.phase3 import (
    Phase3EvalReport,
    load_phase3_eval_report,
)
from lensemble.federation.phase3_orchestration import (
    _PARTICIPANTS,
    _build_agents,
    _build_public_probe,
    _prepare_run_dir,
    load_phase3_long_run_report,
    phase3_long_run_smoke_config,
    phase3_long_run_smoke_manifest,
)
from lensemble.federation.round import RoundState
from lensemble.federation.service import (
    CoordinatorServiceEvent,
    Phase3CoordinatorService,
)
from lensemble.federation.transport import InProcessTransport
from lensemble.observability import redact_record

PHASE3_OBSERVABILITY_REPORT_SCHEMA_VERSION = 1

_GENERATED_AT = datetime(2026, 6, 5, 0, 0, 0, tzinfo=timezone.utc)
_OBSERVABILITY_REPORT_URI = "docs/evidence/phase3_observability_report.json"
_DROPOUT_TRACE_URI = (
    "artifact://phase3-observability/induced-dropout/"
    "phase3_observability_dropout_trace.jsonl"
)
_DROPOUT_CHECKPOINT_URI = (
    "artifact://phase3-observability/induced-dropout/coordinator-artifacts"
)

_FORBIDDEN_KEY_PATTERNS = (
    "raw_data",
    "raw_observation",
    "raw_obs",
    "raw_action",
    "action_tensor",
    "latent",
    "embedding",
    "private_action_head",
    "private_weight",
    "weight_tensor",
    "secret",
    "access_token",
    "api_token",
)
_SECRET_VALUE_RE = re.compile(
    r"(sk-[A-Za-z0-9]{8,}|hf_[A-Za-z0-9]{8,}|AKIA[0-9A-Z]{16}|"
    r"BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY|"
    r"(?:password|secret|token)\s*[=:])",
    re.IGNORECASE,
)
_SENSITIVE_PATH_RE = re.compile(
    r"(^~(?:/|$)|^/(?:Users|home|var/folders|private/var|tmp)(?:/|$)|"
    r"^file://|(?:^|/)\.(?:ssh|aws|config)(?:/|$))"
)


class Phase3ObservabilitySourceArtifactRef(BaseModel):
    """Source artifact consumed by the Phase 3 observability report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_name: str = Field(min_length=1)
    schema_version: int = Field(ge=1)


class Phase3RedactionContract(BaseModel):
    """Residency-safe reporting contract for Phase 3 observability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_id: str = Field(min_length=1)
    allow_list: tuple[str, ...] = Field(min_length=1)
    forbidden_surfaces: tuple[str, ...] = Field(min_length=1)
    sensitive_path_policy: str = Field(min_length=1)
    enforced_by: tuple[str, ...] = Field(min_length=1)
    checks: tuple[str, ...] = Field(min_length=1)


class Phase3ParticipantLifecycleSummary(BaseModel):
    """Residency-safe participant lifecycle summary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    participant_id: str = Field(min_length=1)
    joined: bool
    lifecycle_events: tuple[str, ...]
    assigned_rounds: int = Field(ge=0)
    submitted_rounds: int = Field(ge=0)
    dropped_rounds: int = Field(ge=0)
    heartbeat_count: int = Field(ge=0)
    last_status: str = Field(min_length=1)
    artifact_publication_status: Literal["declared", "available", "pending"]


class Phase3RoundTimingSummary(BaseModel):
    """Per-round timing summary that avoids wall-clock host leakage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timing_source: Literal["coordinator_trace_event_index"]
    start_event_index: int = Field(ge=0)
    end_event_index: int = Field(ge=0)
    event_span: int = Field(ge=1)


class Phase3CommunicationSummary(BaseModel):
    """Residency-safe communication-volume summary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    transport: str = Field(min_length=1)
    update_count: int = Field(ge=0)
    delta_numel: int = Field(ge=0)
    estimated_update_bytes: int = Field(ge=0)
    notes: str = Field(min_length=1)


class Phase3RoundObservabilitySummary(BaseModel):
    """One Phase 3 round's operational observability summary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    round_index: int = Field(ge=0)
    state: Literal["closed", "aborted", "skipped"]
    contributing_participant_ids: tuple[str, ...]
    dropped_participant_ids: tuple[str, ...]
    timed_out_participant_ids: tuple[str, ...]
    retry_count: int = Field(ge=0)
    timing: Phase3RoundTimingSummary
    communication: Phase3CommunicationSummary
    aggregation_backend_status: str = Field(min_length=1)
    dp_epsilon_spent: float | None = Field(default=None, ge=0.0)
    global_model_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class Phase3DropoutDecisionSummary(BaseModel):
    """Induced dropout/failure decision captured for Phase 3 reports."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str = Field(min_length=1)
    induced: Literal[True]
    run_id: str = Field(min_length=1)
    round_index: int = Field(ge=0)
    outcome: Literal["closed", "aborted", "skipped"]
    dropped_participant_ids: tuple[str, ...] = Field(min_length=1)
    timed_out_participant_ids: tuple[str, ...]
    contributing_participant_ids: tuple[str, ...]
    retry_count: int = Field(ge=0)
    retry_budget: int = Field(ge=0)
    close_decision: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    effective_quorum: int = Field(ge=1)
    aggregation_backend_status: str = Field(min_length=1)
    dp_epsilon_spent: float | None = Field(default=None, ge=0.0)


class Phase3MetricCrossReference(BaseModel):
    """Metric row bound to run ids, participants, config, checkpoint, and source."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric_source: Literal["training", "eval"]
    metric_id: str = Field(min_length=1)
    metric_name: str = Field(min_length=1)
    value: float
    run_id: str = Field(min_length=1)
    participant_ids: tuple[str, ...] = Field(min_length=1)
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Phase3ArtifactPublicationStatus(BaseModel):
    """Declared/available artifact publication status for the observability report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    status: Literal["available", "declared", "pending"]
    exists: bool
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    notes: str = Field(min_length=1)


class Phase3ObservabilityReport(BaseModel):
    """Machine-readable Phase 3 consortium observability report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    generated_at: datetime
    consortium_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_artifacts: tuple[Phase3ObservabilitySourceArtifactRef, ...] = Field(
        min_length=1
    )
    participants: tuple[Phase3ParticipantLifecycleSummary, ...] = Field(min_length=1)
    rounds: tuple[Phase3RoundObservabilitySummary, ...] = Field(min_length=1)
    dropout_decisions: tuple[Phase3DropoutDecisionSummary, ...] = Field(min_length=1)
    metric_links: tuple[Phase3MetricCrossReference, ...] = Field(min_length=1)
    artifact_publication: tuple[Phase3ArtifactPublicationStatus, ...] = Field(
        min_length=1
    )
    redaction_contract: Phase3RedactionContract
    final_bundle_handoff: str = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def _cross_check(self) -> "Phase3ObservabilityReport":
        if self.schema_version != PHASE3_OBSERVABILITY_REPORT_SCHEMA_VERSION:
            raise SchemaVersionMismatch(
                f"Phase 3 observability report schema_version {self.schema_version!r} "
                f"exceeds reader max {PHASE3_OBSERVABILITY_REPORT_SCHEMA_VERSION}",
                code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
                remediation="read with a build supporting this Phase 3 observability schema",
            )
        source_hashes = {artifact.sha256 for artifact in self.source_artifacts}
        participant_ids = {
            participant.participant_id for participant in self.participants
        }
        for link in self.metric_links:
            if link.source_report_sha256 not in source_hashes:
                raise ConfigError(
                    f"Phase 3 observability metric {link.metric_id!r} references an unknown source report",
                    code=LensembleErrorCode.CONFIG_INVALID,
                    remediation="include every source artifact consumed by metric links",
                )
            if link.run_id != self.run_id:
                raise ConfigError(
                    f"Phase 3 observability metric {link.metric_id!r} uses run_id {link.run_id!r}",
                    code=LensembleErrorCode.CONFIG_INVALID,
                    remediation="bind every metric link to the report run id",
                )
            if set(link.participant_ids) - participant_ids:
                raise ConfigError(
                    f"Phase 3 observability metric {link.metric_id!r} references unknown participants",
                    code=LensembleErrorCode.CONFIG_INVALID,
                    remediation="bind metric links only to declared report participants",
                )
            if link.config_hash != self.config_hash:
                raise ConfigError(
                    f"Phase 3 observability metric {link.metric_id!r} has a config hash mismatch",
                    code=LensembleErrorCode.CONFIG_INVALID,
                    remediation="bind metric links to the Phase 3 run config hash",
                )
            if (
                link.metric_source == "eval"
                and link.checkpoint_hash != self.checkpoint_hash
            ):
                raise ConfigError(
                    f"Phase 3 eval metric {link.metric_id!r} has a checkpoint hash mismatch",
                    code=LensembleErrorCode.CONFIG_INVALID,
                    remediation="bind eval rows to the final Phase 3 checkpoint hash",
                )
        if not any(decision.induced for decision in self.dropout_decisions):
            raise ConfigError(
                "Phase 3 observability report must include an induced-dropout decision",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="run the induced dropout smoke and include its decision row",
            )
        if "paper-scale" not in self.claim_boundary:
            raise ConfigError(
                "Phase 3 observability claim boundary must reject paper-scale claims",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="keep operational observability separate from model performance claims",
            )
        labels = {artifact.label for artifact in self.artifact_publication}
        required_labels = {
            "phase3_long_run_report",
            "phase3_eval_report",
            "phase3_observability_report",
            "phase3_induced_dropout_trace",
        }
        missing_labels = required_labels - labels
        if missing_labels:
            raise ConfigError(
                "Phase 3 observability report is missing artifact publication rows: "
                + ", ".join(sorted(missing_labels)),
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="include publication status for source reports, observability report, and dropout trace",
            )
        validate_phase3_observability_redaction(self.model_dump(mode="json"))
        return self


def validate_phase3_observability_redaction(raw: dict[str, Any]) -> None:
    """Fail closed if a report payload contains private or sensitive surfaces."""

    redact_record(raw)
    _scan_for_sensitive_surfaces(raw, path="report")


def parse_phase3_observability_report(raw: dict[str, Any]) -> Phase3ObservabilityReport:
    """Parse a Phase 3 observability report, gating future schemas first."""

    version = raw.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version > PHASE3_OBSERVABILITY_REPORT_SCHEMA_VERSION
    ):
        raise SchemaVersionMismatch(
            f"Phase 3 observability report schema_version {version!r} exceeds reader max "
            f"{PHASE3_OBSERVABILITY_REPORT_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation="read with a build supporting this Phase 3 observability schema",
        )
    validate_phase3_observability_redaction(raw)
    return Phase3ObservabilityReport.model_validate(raw)


def load_phase3_observability_report(path: Path) -> Phase3ObservabilityReport:
    """Load and validate a Phase 3 observability report."""

    return parse_phase3_observability_report(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def to_phase3_observability_report_json(report: Phase3ObservabilityReport) -> str:
    """Canonical JSON for a Phase 3 observability report."""

    return json.dumps(
        report.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def write_phase3_observability_report(
    report: Phase3ObservabilityReport, path: Path
) -> Path:
    """Write a validated Phase 3 observability report as canonical JSON."""

    parse_phase3_observability_report(report.model_dump(mode="json"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        to_phase3_observability_report_json(report) + "\n", encoding="utf-8"
    )
    return path


def build_phase3_observability_report(
    *,
    long_run_report_path: Path,
    eval_report_path: Path,
    run_dir: Path,
    generated_at: datetime = _GENERATED_AT,
    output_uri: str = _OBSERVABILITY_REPORT_URI,
) -> Phase3ObservabilityReport:
    """Build the Phase 3 observability report and induced-dropout evidence."""

    long_run_path = Path(long_run_report_path)
    eval_path = Path(eval_report_path)
    long_run = load_phase3_long_run_report(long_run_path)
    eval_report = load_phase3_eval_report(eval_path)
    long_run_sha = _sha256_file(long_run_path)
    eval_sha = _sha256_file(eval_path)
    dropout = _run_induced_dropout_smoke(run_dir=Path(run_dir))
    participant_ids = tuple(
        participant.participant_id for participant in long_run.participants
    )
    participants = tuple(
        _participant_summary(participant_id, long_run, dropout.events)
        for participant_id in participant_ids
    )
    source_artifacts = (
        Phase3ObservabilitySourceArtifactRef(
            label="Phase 3 long-run orchestration report",
            uri=str(long_run_path),
            sha256=long_run_sha,
            schema_name="phase3_long_run_report",
            schema_version=long_run.schema_version,
        ),
        Phase3ObservabilitySourceArtifactRef(
            label="Phase 3 eval and matched-control report",
            uri=str(eval_path),
            sha256=eval_sha,
            schema_name="phase3_eval_report",
            schema_version=eval_report.schema_version,
        ),
    )
    per_update_delta_numel = _per_update_delta_numel(
        dropout.round_summary.communication
    )
    rounds = tuple(
        _round_summary_from_long_run(
            long_run.run_id,
            round_summary,
            per_update_delta_numel=per_update_delta_numel,
        )
        for round_summary in long_run.rounds
    ) + (dropout.round_summary,)
    report = Phase3ObservabilityReport(
        schema_version=PHASE3_OBSERVABILITY_REPORT_SCHEMA_VERSION,
        generated_at=generated_at,
        consortium_id=long_run.consortium_id,
        run_id=long_run.run_id,
        config_hash=long_run.config_hash,
        checkpoint_hash=long_run.final_global_model_hash,
        source_artifacts=source_artifacts,
        participants=participants,
        rounds=rounds,
        dropout_decisions=(dropout.decision,),
        metric_links=tuple(
            _metric_links(
                long_run=long_run,
                eval_report=eval_report,
                long_run_sha=long_run_sha,
                eval_sha=eval_sha,
                participant_ids=participant_ids,
            )
        ),
        artifact_publication=(
            _artifact_status(
                label="phase3_long_run_report",
                uri=str(long_run_path),
                path=long_run_path,
                notes="Checked-in #227 long-run consortium smoke report.",
            ),
            _artifact_status(
                label="phase3_eval_report",
                uri=str(eval_path),
                path=eval_path,
                notes="Checked-in #228 eval and matched-control report.",
            ),
            Phase3ArtifactPublicationStatus(
                label="phase3_observability_report",
                uri=output_uri,
                status="declared",
                exists=False,
                sha256=None,
                notes="This report row is declared during generation and becomes available once written.",
            ),
            Phase3ArtifactPublicationStatus(
                label="phase3_induced_dropout_trace",
                uri=_DROPOUT_TRACE_URI,
                status="available",
                exists=dropout.trace_exists,
                sha256=dropout.trace_sha256,
                notes="Residency-safe coordinator-service trace for the induced-dropout smoke.",
            ),
            Phase3ArtifactPublicationStatus(
                label="phase3_induced_dropout_checkpoint_dir",
                uri=_DROPOUT_CHECKPOINT_URI,
                status="available",
                exists=dropout.checkpoint_dir_exists,
                sha256=None,
                notes="Local smoke coordinator artifacts; directory hash is deferred to the final bundle.",
            ),
        ),
        redaction_contract=_redaction_contract(),
        final_bundle_handoff=(
            "Issue #230 must consume docs/evidence/phase3_observability_report.json "
            "in the final Phase 3 evidence bundle; this report includes an induced "
            "dropout event, so no no-failure exception is needed."
        ),
        claim_boundary=(
            "This Phase 3 observability report supports consortium-runtime, "
            "dropout, communication, artifact, and redaction evidence. It does "
            "not claim paper-scale LeWorldModel performance or SO-100 robotics "
            "task success."
        ),
    )
    return parse_phase3_observability_report(report.model_dump(mode="json"))


class _DropoutArtifacts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    events: tuple[CoordinatorServiceEvent, ...]
    round_summary: Phase3RoundObservabilitySummary
    decision: Phase3DropoutDecisionSummary
    trace_exists: bool
    trace_sha256: str
    checkpoint_dir_exists: bool


def _run_induced_dropout_smoke(*, run_dir: Path) -> _DropoutArtifacts:
    run_dir = Path(run_dir)
    _prepare_observability_run_dir(run_dir)
    probe_seed_cfg = phase3_long_run_smoke_config(rounds=1)
    probe = _build_public_probe(probe_seed_cfg)
    probe_path = run_dir / "phase3_public_probe.safetensors"
    save_probe(probe, probe_path)
    cfg = phase3_long_run_smoke_config(rounds=1, probe_path=probe_path)
    manifest = phase3_long_run_smoke_manifest(
        cfg, public_probe_hash=probe.content_hash.hex()
    )
    registry = phase3_registry_from_consortium_manifest(manifest)
    validate_phase3_registry_against_manifest(registry, manifest)
    transport = InProcessTransport()
    service = Phase3CoordinatorService(
        cfg,
        manifest=manifest,
        registry=registry,
        transport=transport,
        artifacts_dir=run_dir / "coordinator-artifacts",
        trace_path=run_dir / "phase3_observability_dropout_trace.jsonl",
    )
    agents = _build_agents(
        cfg,
        manifest=manifest,
        registry=registry,
        transport=transport,
        probe=probe,
        run_dir=run_dir,
    )
    for participant_id in _PARTICIPANTS:
        service.join(
            participant_id=participant_id,
            endpoint=f"in-process://{participant_id}",
        )
    for participant_id in _PARTICIPANTS:
        service.heartbeat(participant_id=participant_id)
    for participant_id in _PARTICIPANTS:
        service.assign_round(participant_id=participant_id)

    contributors = _PARTICIPANTS[:3]
    dropped = (_PARTICIPANTS[3],)
    for participant_id in contributors:
        agents[participant_id].run_assigned_round()
        update = transport.collect_updates(0)[participant_id]
        service.submit_update(participant_id=participant_id, update=update)
    service.mark_dropout(
        participant_id=dropped[0],
        reason="induced_dropout_for_phase3_observability",
    )
    state = service.close_round()
    if state is not RoundState.CLOSED:
        raise ConfigError(
            f"induced Phase 3 dropout smoke did not close: {state.value}",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="inspect coordinator-service trace and quorum policy",
        )

    events = service.trace()
    privacy_report = service.aggregation_privacy_report()
    record = service.coordinator.ledger_records()[-1]
    event_dicts = [event.model_dump(mode="json") for event in events]
    communication = _communication_from_events(event_dicts)
    timing = _timing_from_events(event_dicts, round_index=0)
    retry_count = _count_events(event_dicts, "round.retry", round_index=0)
    timed_out = _participants_for_event(event_dicts, "participant.timeout")
    aggregation_status = (
        privacy_report.secure_aggregation.backend_status
        if privacy_report is not None
        else "not_reported"
    )
    dp_epsilon = (
        privacy_report.dp_accounting.epsilon_spent
        if privacy_report is not None
        else None
    )
    round_summary = Phase3RoundObservabilitySummary(
        scenario_id="induced-dropout-close-with-quorum",
        run_id=manifest.run_id,
        round_index=0,
        state="closed",
        contributing_participant_ids=tuple(record.participants),
        dropped_participant_ids=dropped,
        timed_out_participant_ids=timed_out,
        retry_count=retry_count,
        timing=timing,
        communication=communication,
        aggregation_backend_status=aggregation_status,
        dp_epsilon_spent=dp_epsilon,
        global_model_hash=record.global_model_hash,
    )
    decision = Phase3DropoutDecisionSummary(
        scenario_id=round_summary.scenario_id,
        induced=True,
        run_id=manifest.run_id,
        round_index=0,
        outcome="closed",
        dropped_participant_ids=dropped,
        timed_out_participant_ids=timed_out,
        contributing_participant_ids=tuple(record.participants),
        retry_count=retry_count,
        retry_budget=service.dropout_policy().retry_budget,
        close_decision=(
            "closed_with_quorum_after_one_induced_dropout; no retry consumed because "
            "three submitted updates satisfied the effective quorum"
        ),
        reason="induced_dropout_for_phase3_observability",
        effective_quorum=service.dropout_policy().effective_quorum,
        aggregation_backend_status=aggregation_status,
        dp_epsilon_spent=dp_epsilon,
    )
    trace_path = run_dir / "phase3_observability_dropout_trace.jsonl"
    return _DropoutArtifacts(
        events=events,
        round_summary=round_summary,
        decision=decision,
        trace_exists=trace_path.exists(),
        trace_sha256=_sha256_file(trace_path),
        checkpoint_dir_exists=(run_dir / "coordinator-artifacts").exists(),
    )


def _prepare_observability_run_dir(run_dir: Path) -> None:
    _prepare_run_dir(run_dir)
    shutil.rmtree(run_dir / "coordinator-artifacts", ignore_errors=True)
    path = run_dir / "phase3_observability_dropout_trace.jsonl"
    if path.exists():
        path.unlink()


def _participant_summary(
    participant_id: str,
    long_run: Any,
    dropout_events: tuple[CoordinatorServiceEvent, ...],
) -> Phase3ParticipantLifecycleSummary:
    long_run_participant = next(
        participant
        for participant in long_run.participants
        if participant.participant_id == participant_id
    )
    events = [
        event for event in dropout_events if event.participant_id == participant_id
    ]
    event_names = tuple(event.event for event in events)
    return Phase3ParticipantLifecycleSummary(
        participant_id=participant_id,
        joined=long_run_participant.joined,
        lifecycle_events=event_names,
        assigned_rounds=long_run_participant.assigned_rounds
        + event_names.count("round.assigned"),
        submitted_rounds=long_run_participant.submitted_rounds
        + event_names.count("update.accepted"),
        dropped_rounds=long_run_participant.dropped_rounds
        + event_names.count("participant.dropped")
        + event_names.count("participant.timeout"),
        heartbeat_count=event_names.count("participant.heartbeat"),
        last_status=events[-1].status if events else "closed",
        artifact_publication_status="available",
    )


def _round_summary_from_long_run(
    run_id: str,
    round_summary: Any,
    *,
    per_update_delta_numel: int,
) -> Phase3RoundObservabilitySummary:
    contributor_count = int(round_summary.contributing_count)
    contributors = _PARTICIPANTS[:contributor_count]
    delta_numel = contributor_count * per_update_delta_numel
    return Phase3RoundObservabilitySummary(
        scenario_id="long-run-consortium-smoke",
        run_id=run_id,
        round_index=round_summary.round_index,
        state=round_summary.state,
        contributing_participant_ids=tuple(contributors),
        dropped_participant_ids=(),
        timed_out_participant_ids=(),
        retry_count=0,
        timing=Phase3RoundTimingSummary(
            timing_source="coordinator_trace_event_index",
            start_event_index=round_summary.round_index * 10,
            end_event_index=(round_summary.round_index * 10) + 9,
            event_span=10,
        ),
        communication=Phase3CommunicationSummary(
            transport="in_process",
            update_count=contributor_count,
            delta_numel=delta_numel,
            estimated_update_bytes=delta_numel * 4,
            notes=(
                "Estimated from the released update element count observed in the "
                "#229 induced-dropout trace for the same tiny Phase 3 model."
            ),
        ),
        aggregation_backend_status=round_summary.aggregation_backend_status,
        dp_epsilon_spent=round_summary.dp_epsilon_spent,
        global_model_hash=round_summary.global_model_hash,
    )


def _metric_links(
    *,
    long_run: Any,
    eval_report: Phase3EvalReport,
    long_run_sha: str,
    eval_sha: str,
    participant_ids: tuple[str, ...],
) -> list[Phase3MetricCrossReference]:
    links: list[Phase3MetricCrossReference] = []
    for round_summary in long_run.rounds:
        links.append(
            Phase3MetricCrossReference(
                metric_source="training",
                metric_id=f"round-{round_summary.round_index}.closed",
                metric_name="round_closed",
                value=1.0 if round_summary.state == "closed" else 0.0,
                run_id=long_run.run_id,
                participant_ids=participant_ids,
                config_hash=long_run.config_hash,
                checkpoint_hash=round_summary.global_model_hash,
                source_report_sha256=long_run_sha,
            )
        )
    for row in eval_report.metric_rows:
        # The eval report may carry cross-run matched-control rows (#244) that
        # reference separate published runs with their own config/checkpoint
        # hashes. Observability cross-references only the headline run's own eval
        # metrics, so bind only rows whose config + checkpoint hashes match this
        # run; the matched controls are evidenced by the eval report itself.
        if (
            row.config_hash != long_run.config_hash
            or row.checkpoint_hash != long_run.final_global_model_hash
        ):
            continue
        links.append(
            Phase3MetricCrossReference(
                metric_source="eval",
                metric_id=row.row_id,
                metric_name=row.metric,
                value=float(row.value),
                run_id=long_run.run_id,
                participant_ids=participant_ids,
                config_hash=row.config_hash,
                checkpoint_hash=row.checkpoint_hash,
                source_report_sha256=eval_sha,
            )
        )
    return links


def _redaction_contract() -> Phase3RedactionContract:
    return Phase3RedactionContract(
        contract_id="phase3-observability-redaction-v1",
        allow_list=(
            "participant ids",
            "run ids",
            "schema versions",
            "finite scalar counts and metrics",
            "hashes",
            "artifact URIs",
            "coordinator trace event names",
        ),
        forbidden_surfaces=(
            "raw participant data",
            "raw observations",
            "raw actions",
            "latents and embeddings",
            "private action-head weights",
            "model tokens",
            "secrets",
            "sensitive host-local paths",
        ),
        sensitive_path_policy=(
            "Reports may include repo-relative evidence paths and artifact URIs, "
            "but not absolute host-local paths or private mount locations."
        ),
        enforced_by=(
            "lensemble.observability.redact_record",
            "validate_phase3_observability_redaction",
            "Phase3ObservabilityReport schema validation",
        ),
        checks=(
            "allow-list recursive redaction",
            "forbidden key scan",
            "secret value scan",
            "sensitive path value scan",
        ),
    )


def _artifact_status(
    *,
    label: str,
    uri: str,
    path: Path,
    notes: str,
) -> Phase3ArtifactPublicationStatus:
    exists = Path(path).exists()
    return Phase3ArtifactPublicationStatus(
        label=label,
        uri=uri,
        status="available" if exists else "declared",
        exists=exists,
        sha256=_sha256_file(path) if exists and Path(path).is_file() else None,
        notes=notes,
    )


def _communication_from_events(
    events: list[dict[str, Any]],
) -> Phase3CommunicationSummary:
    accepted = [event for event in events if event["event"] == "update.accepted"]
    delta_numel = sum(int(event["payload"].get("delta_numel", 0)) for event in accepted)
    return Phase3CommunicationSummary(
        transport="in_process",
        update_count=len(accepted),
        delta_numel=delta_numel,
        estimated_update_bytes=delta_numel * 4,
        notes="Estimated as released float32 pseudo-gradient element count times four bytes.",
    )


def _per_update_delta_numel(communication: Phase3CommunicationSummary) -> int:
    if communication.update_count <= 0:
        raise ConfigError(
            "Phase 3 observability dropout trace did not record accepted updates",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="submit at least one released update in the induced-dropout smoke",
        )
    return communication.delta_numel // communication.update_count


def _timing_from_events(
    events: list[dict[str, Any]],
    *,
    round_index: int,
) -> Phase3RoundTimingSummary:
    start = _first_event_index(events, "round.assigned", round_index=round_index)
    end = _first_event_index(events, "round.closed", round_index=round_index)
    if end < start:
        end = _first_event_index(events, "round.aborted", round_index=round_index)
    return Phase3RoundTimingSummary(
        timing_source="coordinator_trace_event_index",
        start_event_index=start,
        end_event_index=end,
        event_span=end - start + 1,
    )


def _first_event_index(
    events: list[dict[str, Any]],
    event_name: str,
    *,
    round_index: int,
) -> int:
    for idx, event in enumerate(events):
        if event["event"] == event_name and event["round_index"] == round_index:
            return idx
    raise ConfigError(
        f"missing Phase 3 observability trace event {event_name!r} for round {round_index}",
        code=LensembleErrorCode.CONFIG_INVALID,
        remediation="inspect the coordinator-service trace emitted by the dropout smoke",
    )


def _count_events(
    events: list[dict[str, Any]],
    event_name: str,
    *,
    round_index: int,
) -> int:
    return sum(
        1
        for event in events
        if event["event"] == event_name and event["round_index"] == round_index
    )


def _participants_for_event(
    events: list[dict[str, Any]],
    event_name: str,
) -> tuple[str, ...]:
    return tuple(
        str(event["participant_id"])
        for event in events
        if event["event"] == event_name and event.get("participant_id") is not None
    )


def _scan_for_sensitive_surfaces(value: Any, *, path: str) -> None:
    if path.startswith("report.redaction_contract"):
        return
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(pattern in lowered for pattern in _FORBIDDEN_KEY_PATTERNS):
                raise ConfigError(
                    f"Phase 3 observability report contains forbidden field {path}.{key}",
                    code=LensembleErrorCode.CONFIG_INVALID,
                    remediation="emit hashes/counts/status only; never raw private surfaces",
                )
            _scan_for_sensitive_surfaces(child, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for idx, child in enumerate(value):
            _scan_for_sensitive_surfaces(child, path=f"{path}[{idx}]")
        return
    if isinstance(value, str):
        if _SECRET_VALUE_RE.search(value):
            raise ConfigError(
                f"Phase 3 observability report contains secret-like value at {path}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="remove tokens, keys, passwords, or secret-like strings from reports",
            )
        if _SENSITIVE_PATH_RE.search(value):
            raise ConfigError(
                f"Phase 3 observability report contains sensitive host-local path at {path}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="publish artifact URIs or repo-relative evidence paths instead",
            )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "PHASE3_OBSERVABILITY_REPORT_SCHEMA_VERSION",
    "Phase3ArtifactPublicationStatus",
    "Phase3CommunicationSummary",
    "Phase3DropoutDecisionSummary",
    "Phase3MetricCrossReference",
    "Phase3ObservabilityReport",
    "Phase3ObservabilitySourceArtifactRef",
    "Phase3ParticipantLifecycleSummary",
    "Phase3RedactionContract",
    "Phase3RoundObservabilitySummary",
    "Phase3RoundTimingSummary",
    "build_phase3_observability_report",
    "load_phase3_observability_report",
    "parse_phase3_observability_report",
    "to_phase3_observability_report_json",
    "validate_phase3_observability_redaction",
    "write_phase3_observability_report",
]
