"""Phase 3 networked coordinator service control plane.

The service wraps the existing deterministic :class:`Coordinator` round engine
with the operational Phase 3 control plane: governed admission from the
consortium manifest, participant heartbeat and round assignment, update
submission, dropout/timeout policy, explicit abort/close flows, and a
residency-safe state trace. It does not inspect raw participant data; it only
handles membership metadata, hashes, and released pseudo-gradients.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from lensemble.config.consortium import (
    Phase3ConsortiumManifest,
    validate_coordinator_run_agreement,
)
from lensemble.data.phase3 import (
    Phase3DatasetProbeRegistry,
    validate_coordinator_registry_preflight,
)
from lensemble.errors import ConfigError, LensembleErrorCode, RoundError
from lensemble.federation.coordinator import Coordinator
from lensemble.federation.phase3_privacy import (
    Phase3AggregationPrivacyReport,
    build_phase3_aggregation_privacy_report,
)
from lensemble.federation.round import RoundState
from lensemble.federation.transport import InProcessTransport

if TYPE_CHECKING:
    from lensemble.config.schema import LensembleConfig
    from lensemble.federation.pseudogradient import PseudoGradient
    from lensemble.federation.state import GlobalState
    from lensemble.federation.transport import Transport

COORDINATOR_SERVICE_TRACE_SCHEMA_VERSION = 1
COORDINATOR_SERVICE_REPORT_SCHEMA_VERSION = 1

ParticipantServiceStatus = Literal[
    "declared",
    "joined",
    "assigned",
    "updated",
    "dropped",
    "rejected",
]
CoordinatorServiceEventName = Literal[
    "service.started",
    "admission.closed",
    "participant.joined",
    "participant.rejected",
    "participant.heartbeat",
    "round.assigned",
    "update.accepted",
    "update.rejected",
    "participant.timeout",
    "participant.dropped",
    "round.closed",
    "round.retry",
    "round.aborted",
]


def _fail(key: str, value: object, expected: str, remediation: str) -> ConfigError:
    err = ConfigError(
        f"invalid coordinator-service agreement: {key}={value!r} ({expected})",
        code=LensembleErrorCode.CONFIG_INVALID,
        remediation=remediation,
    )
    err.key = key  # type: ignore[attr-defined]
    err.value = value  # type: ignore[attr-defined]
    err.expected = expected  # type: ignore[attr-defined]
    return err


def _round_error(message: str, remediation: str) -> RoundError:
    return RoundError(
        message,
        code=LensembleErrorCode.ROUND_FAILED,
        remediation=remediation,
    )


def _float_close(lhs: float, rhs: float) -> bool:
    return abs(lhs - rhs) <= 1e-9 + 1e-9 * max(abs(lhs), abs(rhs), 1.0)


class Phase3DropoutPolicy(BaseModel):
    """Explicit Phase 3 dropout/timeout policy derived from config + manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_participants: int = Field(ge=1)
    secure_agg_threshold: int = Field(ge=1)
    effective_quorum: int = Field(ge=1)
    collect_timeout_s: float = Field(gt=0.0)
    retry_budget: int = Field(ge=0)
    timeout_action: Literal["drop_absent_then_close_if_quorum_else_abort"] = (
        "drop_absent_then_close_if_quorum_else_abort"
    )
    duplicate_update_policy: Literal["reject"] = "reject"
    late_join_policy: Literal["reject_after_admission_closed"] = (
        "reject_after_admission_closed"
    )


class CoordinatorServiceEvent(BaseModel):
    """One residency-safe coordinator-service trace event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = COORDINATOR_SERVICE_TRACE_SCHEMA_VERSION
    event: CoordinatorServiceEventName
    consortium_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    round_index: int = Field(ge=0)
    participant_id: str | None = None
    status: str = Field(min_length=1)
    reason: str | None = None
    payload: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class CoordinatorParticipantReport(BaseModel):
    """Residency-safe participant lifecycle state for reports."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    participant_id: str = Field(min_length=1)
    status: ParticipantServiceStatus
    endpoint: str | None = None
    joined: bool
    assigned_round: int | None = None
    updated_round: int | None = None
    dropped_round: int | None = None
    last_heartbeat_round: int | None = None


class CoordinatorServiceReport(BaseModel):
    """Machine-readable startup/current-state report for the coordinator service."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = COORDINATOR_SERVICE_REPORT_SCHEMA_VERSION
    consortium_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    coordinator_id: str = Field(min_length=1)
    round_index: int = Field(ge=0)
    round_state: str = Field(min_length=1)
    global_model_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dropout_policy: Phase3DropoutPolicy
    participants: tuple[CoordinatorParticipantReport, ...]
    trace_path: str = Field(min_length=1)
    aggregation_privacy_report: Phase3AggregationPrivacyReport | None = None


@dataclass
class _ParticipantRuntimeState:
    participant_id: str
    status: ParticipantServiceStatus = "declared"
    endpoint: str | None = None
    joined: bool = False
    assigned_round: int | None = None
    updated_round: int | None = None
    dropped_round: int | None = None
    last_heartbeat_round: int | None = None
    submitted_rounds: set[int] = field(default_factory=set)

    def report(self) -> CoordinatorParticipantReport:
        return CoordinatorParticipantReport(
            participant_id=self.participant_id,
            status=self.status,
            endpoint=self.endpoint,
            joined=self.joined,
            assigned_round=self.assigned_round,
            updated_round=self.updated_round,
            dropped_round=self.dropped_round,
            last_heartbeat_round=self.last_heartbeat_round,
        )


class Phase3CoordinatorService:
    """Coordinator service control plane for Phase 3 consortium training."""

    def __init__(
        self,
        config: "LensembleConfig",
        *,
        manifest: Phase3ConsortiumManifest,
        registry: Phase3DatasetProbeRegistry | None = None,
        transport: "Transport | None" = None,
        artifacts_dir: Path | None = None,
        trace_path: Path | None = None,
        enable_backstop: bool = False,
    ) -> None:
        self.config = config
        self.manifest = validate_coordinator_run_agreement(manifest)
        self.registry = (
            validate_coordinator_registry_preflight(registry, self.manifest)
            if registry is not None
            else None
        )
        self._validate_config_agreement()
        self.transport: Transport = transport or InProcessTransport()
        # #262: pass the live Layer-3 Procrustes backstop flag through to the round engine. Default OFF so
        # every existing service test stays the measured pass-through; the consortium launcher turns it ON
        # for the real anchored-federation run.
        self.coordinator = Coordinator(
            config,
            transport=self.transport,
            artifacts_dir=artifacts_dir,
            enable_backstop=enable_backstop,
        )
        self.trace_path = Path(trace_path or Path("coordinator_trace.jsonl"))
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[CoordinatorServiceEvent] = []
        self._admission_open = True
        self._aborted_rounds: set[int] = set()
        self._retry_counts: dict[int, int] = {}
        self._last_aggregation_privacy_report: Phase3AggregationPrivacyReport | None = (
            None
        )
        trainer_ids = [
            p.participant_id for p in self.manifest.participants if p.role == "trainer"
        ]
        self._participants = {
            participant_id: _ParticipantRuntimeState(participant_id)
            for participant_id in trainer_ids
        }
        self._record(
            "service.started",
            status="open",
            payload={
                "declared_trainers": len(self._participants),
                "effective_quorum": self.dropout_policy().effective_quorum,
            },
        )

    def dropout_policy(self) -> Phase3DropoutPolicy:
        fed = self.config.federation
        effective = max(
            int(self.manifest.runtime.min_trainers),
            int(fed.fault_tolerance_min_participants),
            int(fed.secure_agg_threshold),
        )
        return Phase3DropoutPolicy(
            min_participants=int(self.manifest.runtime.min_trainers),
            secure_agg_threshold=int(fed.secure_agg_threshold),
            effective_quorum=effective,
            collect_timeout_s=float(fed.collect_timeout_s),
            retry_budget=int(self.manifest.runtime.dropout_retry_budget),
        )

    def report(self) -> CoordinatorServiceReport:
        return CoordinatorServiceReport(
            consortium_id=self.manifest.consortium_id,
            run_id=self.manifest.run_id,
            coordinator_id=self.manifest.coordinator_id,
            round_index=self.coordinator.global_state().round_index,
            round_state=self.round_state().value,
            global_model_hash=self.coordinator.global_state_hash(),
            dropout_policy=self.dropout_policy(),
            participants=tuple(
                self._participants[pid].report() for pid in sorted(self._participants)
            ),
            trace_path=str(self.trace_path),
            aggregation_privacy_report=self._last_aggregation_privacy_report,
        )

    def trace(self) -> tuple[CoordinatorServiceEvent, ...]:
        return tuple(self._events)

    def aggregation_privacy_report(self) -> Phase3AggregationPrivacyReport | None:
        """The latest successful round's Phase 3 aggregation/privacy report."""

        return self._last_aggregation_privacy_report

    def round_state(self) -> RoundState:
        current = self.coordinator.global_state().round_index
        if current in self._aborted_rounds:
            return RoundState.ABORTED
        return self.coordinator.round_state()

    def join(self, *, participant_id: str, endpoint: str) -> CoordinatorServiceEvent:
        state = self._participant_or_reject(participant_id)
        current = self.coordinator.global_state().round_index
        if not self._admission_open:
            state.status = "rejected"
            self._record(
                "participant.rejected",
                participant_id=participant_id,
                status="rejected",
                reason="late_join",
            )
            raise _round_error(
                f"participant {participant_id!r} attempted to join after admission closed for round {current}",
                "join before the coordinator assigns the first round, or wait for a new run",
            )
        if state.joined:
            self._record(
                "participant.rejected",
                participant_id=participant_id,
                status="rejected",
                reason="duplicate_join",
            )
            raise _round_error(
                f"participant {participant_id!r} already joined",
                "reuse the existing participant session instead of joining twice",
            )
        state.joined = True
        state.endpoint = endpoint
        state.status = "joined"
        register = getattr(self.transport, "register", None)
        if register is not None:
            register(participant_id, endpoint)
        return self._record(
            "participant.joined",
            participant_id=participant_id,
            status="joined",
            payload={"endpoint": endpoint},
        )

    def close_admission(self) -> CoordinatorServiceEvent:
        self._admission_open = False
        return self._record("admission.closed", status="closed")

    def heartbeat(self, *, participant_id: str) -> CoordinatorServiceEvent:
        state = self._require_joined(participant_id)
        round_index = self.coordinator.global_state().round_index
        state.last_heartbeat_round = round_index
        return self._record(
            "participant.heartbeat",
            participant_id=participant_id,
            status=state.status,
        )

    def assign_round(self, *, participant_id: str) -> "GlobalState":
        state = self._require_joined(participant_id)
        self._admission_open = False
        round_index = self.coordinator.global_state().round_index
        if round_index in self._aborted_rounds:
            raise _round_error(
                f"round {round_index} is aborted and cannot be assigned",
                "start a new coordinator run or inspect the abort trace",
            )
        state.assigned_round = round_index
        state.status = "assigned"
        gs = self.coordinator.global_state()
        self._record(
            "round.assigned",
            participant_id=participant_id,
            status="assigned",
            payload={"sketch_seed": gs.sketch_seed},
        )
        return gs

    def submit_update(
        self, *, participant_id: str, update: "PseudoGradient"
    ) -> CoordinatorServiceEvent:
        state = self._require_joined(participant_id)
        round_index = self.coordinator.global_state().round_index
        if state.assigned_round != round_index:
            self._record(
                "update.rejected",
                participant_id=participant_id,
                status=state.status,
                reason="unassigned_round",
            )
            raise _round_error(
                f"participant {participant_id!r} has not been assigned round {round_index}",
                "request a round assignment before submitting an update",
            )
        if update.round_index != round_index:
            self._record(
                "update.rejected",
                participant_id=participant_id,
                status=state.status,
                reason="wrong_round",
                payload={"update_round": update.round_index},
            )
            raise _round_error(
                f"participant {participant_id!r} submitted round {update.round_index}, expected {round_index}",
                "submit only the update for the currently assigned round",
            )
        if round_index in state.submitted_rounds:
            self._record(
                "update.rejected",
                participant_id=participant_id,
                status=state.status,
                reason="duplicate_update",
            )
            raise _round_error(
                f"duplicate update from participant {participant_id!r} for round {round_index}",
                "a participant may submit at most one released update per round",
            )
        commit_root = getattr(self.transport, "commit_root", None)
        if commit_root is not None:
            commit_root(
                participant_id=participant_id,
                round_index=round_index,
                root=update.dataset_root,
            )
        self.transport.submit_update(
            participant_id=participant_id, round_index=round_index, update=update
        )
        state.submitted_rounds.add(round_index)
        state.updated_round = round_index
        state.status = "updated"
        return self._record(
            "update.accepted",
            participant_id=participant_id,
            status="updated",
            payload={
                "delta_numel": int(update.delta.numel()),
                "l2_norm": float(update.l2_norm),
                "dataset_root": update.dataset_root.hex(),
            },
        )

    def mark_dropout(
        self, *, participant_id: str, reason: str = "dropout"
    ) -> CoordinatorServiceEvent:
        state = self._require_joined(participant_id)
        round_index = self.coordinator.global_state().round_index
        state.status = "dropped"
        state.dropped_round = round_index
        return self._record(
            "participant.dropped",
            participant_id=participant_id,
            status="dropped",
            reason=reason,
        )

    def abort_round(self, *, reason: str) -> CoordinatorServiceEvent:
        round_index = self.coordinator.global_state().round_index
        self._aborted_rounds.add(round_index)
        return self._record("round.aborted", status="aborted", reason=reason)

    def close_round(self) -> RoundState:
        round_index = self.coordinator.global_state().round_index
        if round_index in self._aborted_rounds:
            return RoundState.ABORTED
        self._record_missing_assigned_as_timeouts(round_index)
        updates = dict(self.transport.collect_updates(round_index))
        state = self.coordinator.try_round()
        if state is RoundState.CLOSED:
            self._last_aggregation_privacy_report = (
                build_phase3_aggregation_privacy_report(
                    self.config,
                    self.manifest,
                    updates,
                    round_index=round_index,
                )
            )
            record = self.coordinator.ledger_records()[-1]
            self._record(
                "round.closed",
                status="closed",
                round_index=record.round_index,
                payload={
                    "participants": ",".join(record.participants),
                    "global_model_hash": record.global_model_hash,
                    "contributing": len(record.participants),
                },
            )
            return state

        retries = self._retry_counts.get(round_index, 0)
        policy = self.dropout_policy()
        if retries < policy.retry_budget:
            self._retry_counts[round_index] = retries + 1
            self._record(
                "round.retry",
                status="retry",
                round_index=round_index,
                reason="below_quorum",
                payload={"retry": retries + 1, "retry_budget": policy.retry_budget},
            )
            return state
        self._aborted_rounds.add(round_index)
        self._record(
            "round.aborted",
            status="aborted",
            round_index=round_index,
            reason="below_quorum",
            payload={"effective_quorum": policy.effective_quorum},
        )
        return RoundState.ABORTED

    def _validate_config_agreement(self) -> None:
        if self.config.run_mode != "coordinator":
            raise _fail(
                "run_mode",
                self.config.run_mode,
                "coordinator",
                "coordinator-service runs require a coordinator-mode config",
            )
        model = self.manifest.model
        runtime = self.manifest.runtime
        dp = self.manifest.dp_policy
        if self.config.model.wmcp_version != model.wmcp_version:
            raise _fail(
                "model.wmcp_version",
                self.config.model.wmcp_version,
                f"== manifest model.wmcp_version ({model.wmcp_version})",
                "start the coordinator with the consortium WMCP version",
            )
        if int(self.config.model.latent_dim) != model.latent_dim:
            raise _fail(
                "model.latent_dim",
                self.config.model.latent_dim,
                f"== manifest model.latent_dim ({model.latent_dim})",
                "start the coordinator with the model shape accepted by the manifest",
            )
        if int(self.config.model.num_tokens) != model.num_tokens:
            raise _fail(
                "model.num_tokens",
                self.config.model.num_tokens,
                f"== manifest model.num_tokens ({model.num_tokens})",
                "start the coordinator with the token shape accepted by the manifest",
            )
        if bool(self.config.objective.target_stop_gradient) != (
            model.objective_target_stop_gradient
        ):
            raise _fail(
                "objective.target_stop_gradient",
                self.config.objective.target_stop_gradient,
                f"== manifest value ({model.objective_target_stop_gradient})",
                "start with the accepted LeWorldModel objective branch",
            )
        if not _float_close(float(self.config.objective.lambda_anc), model.lambda_anc):
            raise _fail(
                "objective.lambda_anc",
                self.config.objective.lambda_anc,
                f"== manifest value ({model.lambda_anc})",
                "start with the accepted public-probe anchor strength",
            )
        if self.config.federation.transport != runtime.transport:
            raise _fail(
                "federation.transport",
                self.config.federation.transport,
                f"== manifest runtime.transport ({runtime.transport})",
                "select the transport mode accepted by the consortium manifest",
            )
        if (
            self.config.federation.aggregation_backend
            != runtime.secure_aggregation_backend
        ):
            raise _fail(
                "federation.aggregation_backend",
                self.config.federation.aggregation_backend,
                f"== manifest secure_aggregation_backend ({runtime.secure_aggregation_backend})",
                "select the secure-aggregation backend accepted by the manifest",
            )
        if self.config.privacy.enabled != dp.enabled:
            raise _fail(
                "privacy.enabled",
                self.config.privacy.enabled,
                f"== manifest dp_policy.enabled ({dp.enabled})",
                "match the consortium DP policy",
            )
        if not _float_close(float(self.config.privacy.clip_norm), dp.clip_norm):
            raise _fail(
                "privacy.clip_norm",
                self.config.privacy.clip_norm,
                f"== manifest clip_norm ({dp.clip_norm})",
                "match the consortium DP clipping bound",
            )
        if not _float_close(
            float(self.config.privacy.noise_multiplier), dp.noise_multiplier
        ):
            raise _fail(
                "privacy.noise_multiplier",
                self.config.privacy.noise_multiplier,
                f"== manifest noise_multiplier ({dp.noise_multiplier})",
                "match the consortium DP noise multiplier",
            )
        if not _float_close(float(self.config.privacy.epsilon), dp.epsilon):
            raise _fail(
                "privacy.epsilon",
                self.config.privacy.epsilon,
                f"== manifest epsilon ({dp.epsilon})",
                "match the consortium DP budget",
            )
        if not _float_close(float(self.config.privacy.delta), dp.delta):
            raise _fail(
                "privacy.delta",
                self.config.privacy.delta,
                f"== manifest delta ({dp.delta})",
                "match the consortium DP budget",
            )
        if self.config.privacy.accountant != dp.accountant:
            raise _fail(
                "privacy.accountant",
                self.config.privacy.accountant,
                f"== manifest accountant ({dp.accountant})",
                "match the consortium DP accountant",
            )

    def _participant_or_reject(self, participant_id: str) -> _ParticipantRuntimeState:
        state = self._participants.get(participant_id)
        if state is None:
            self._record(
                "participant.rejected",
                participant_id=participant_id,
                status="rejected",
                reason="not_in_manifest",
            )
            raise _round_error(
                f"participant {participant_id!r} is not a trainer in the consortium manifest",
                "join with a trainer participant id declared in the manifest",
            )
        return state

    def _require_joined(self, participant_id: str) -> _ParticipantRuntimeState:
        state = self._participant_or_reject(participant_id)
        if not state.joined:
            raise _round_error(
                f"participant {participant_id!r} has not joined",
                "call join before heartbeat, assignment, dropout, or update submission",
            )
        return state

    def _record_missing_assigned_as_timeouts(self, round_index: int) -> None:
        for participant_id, state in sorted(self._participants.items()):
            if state.assigned_round != round_index:
                continue
            if (
                round_index in state.submitted_rounds
                or state.dropped_round == round_index
            ):
                continue
            state.status = "dropped"
            state.dropped_round = round_index
            self._record(
                "participant.timeout",
                participant_id=participant_id,
                status="dropped",
                reason="collect_timeout",
                round_index=round_index,
                payload={"collect_timeout_s": self.dropout_policy().collect_timeout_s},
            )

    def _record(
        self,
        event: CoordinatorServiceEventName,
        *,
        status: str,
        participant_id: str | None = None,
        reason: str | None = None,
        round_index: int | None = None,
        payload: dict[str, str | int | float | bool | None] | None = None,
    ) -> CoordinatorServiceEvent:
        record = CoordinatorServiceEvent(
            event=event,
            consortium_id=self.manifest.consortium_id,
            run_id=self.manifest.run_id,
            round_index=(
                self.coordinator.global_state().round_index
                if round_index is None
                else round_index
            ),
            participant_id=participant_id,
            status=status,
            reason=reason,
            payload=dict(payload or {}),
        )
        self._events.append(record)
        with self.trace_path.open("a", encoding="utf-8") as sink:
            sink.write(
                json.dumps(
                    record.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                + "\n"
            )
        return record


__all__ = [
    "COORDINATOR_SERVICE_REPORT_SCHEMA_VERSION",
    "COORDINATOR_SERVICE_TRACE_SCHEMA_VERSION",
    "CoordinatorParticipantReport",
    "CoordinatorServiceEvent",
    "CoordinatorServiceReport",
    "Phase3CoordinatorService",
    "Phase3DropoutPolicy",
]
