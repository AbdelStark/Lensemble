"""Browser federated-demo orchestration service.

This module intentionally implements a narrow educational demo backend, not a
production coordinator. It owns browser run allocation, participant admission,
residency-safe lifecycle events, bounded browser update artifacts, a tiny
coordinator-style aggregation path, inference artifact publication, and evidence
export for issues #294/#296-#301 plus the hackathon readiness epic #303.

The hard boundary is that browser participants submit only versioned derived
updates: a tiny bounded vector, shape, hash, norm, sample count, and runtime
labels. Raw observations, actions, labels, latents, tensors, participant tokens,
and model weights are rejected at the API boundary.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

RUN_STATES = (
    "created",
    "joining",
    "ready",
    "running_round",
    "aggregating",
    "checkpoint_ready",
    "inference_ready",
    "completed",
    "aborted",
    "failed",
)
TERMINAL_RUN_STATES = frozenset({"completed", "aborted", "failed"})
PARTICIPANT_STATES = (
    "joined",
    "ready",
    "assigned",
    "training",
    "submitted",
    "completed",
    "dropped",
    "error",
)
CONNECTION_STATES = (
    "connected",
    "reconnecting",
    "stale",
    "dropped",
    "completed",
)

RUN_SCHEMA = "demo-run/1"
EVENT_SCHEMA = "demo-event/1"
UPDATE_SCHEMA = "browser-update/1"
CHECKPOINT_SCHEMA = "demo-checkpoint/1"
MODEL_REVISION_SCHEMA = "demo-model-revision/1"
INFERENCE_SCHEMA = "demo-inference-artifact/1"
EVIDENCE_SCHEMA = "demo-evidence/1"
DEMO_PRESETS = frozenset({"swipe-dot-tiny"})
ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"
HACKATHON_MODEL_RUNTIME = "tiny-js-vector-v1"
INITIAL_REVISION_ID = "initial"
DEFAULT_CLIP_NORM = 1.0

CLAIM_BOUNDARY = (
    "Educational browser demo of federated JEPA world-model orchestration. "
    "It demonstrates run orchestration, residency-safe bounded browser update "
    "artifacts, aggregation plumbing, tiny model-revision artifacts, and "
    "browser inference/env-sim. It is not a benchmark win over local-only, not "
    "a production browser-training claim, not a cryptographic proof of honest "
    "computation, and not a physical SO-100 success claim."
)

_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "action",
        "actions",
        "data",
        "dataset",
        "example",
        "examples",
        "image",
        "images",
        "label",
        "labels",
        "latent",
        "latents",
        "observation",
        "observations",
        "raw",
        "sample",
        "samples",
        "tensor",
        "tensors",
        "token",
        "tokens",
        "weight",
        "weights",
    }
)

_REDACTED_UPDATE_FIELDS = frozenset(
    {
        "schema",
        "source",
        "runtime",
        "runId",
        "participantId",
        "round",
        "roundId",
        "modelRevisionId",
        "shape",
        "parameterCount",
        "sampleCount",
        "localSteps",
        "hash",
        "l2Norm",
        "clipNorm",
        "loss",
        "probe",
        "runtimeMs",
        "seed",
        "simulated",
    }
)


class FederatedDemoError(ValueError):
    """Structured local-demo failure."""

    def __init__(self, code: str, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status = status

    def as_payload(self) -> dict[str, str]:
        return {"ok": "false", "code": self.code, "message": str(self)}


def _token(prefix: str, length: int) -> str:
    return f"{prefix}{''.join(secrets.choice(ALPHABET) for _ in range(length))}"


def _hash_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _contains_forbidden_key(value: object) -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            if lowered in _FORBIDDEN_PAYLOAD_KEYS or lowered.startswith("raw_"):
                return str(key)
            found = _contains_forbidden_key(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _contains_forbidden_key(nested)
            if found is not None:
                return found
    return None


def _validate_hash(value: object, *, field_name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise FederatedDemoError(
            "invalid_artifact",
            f"{field_name} must be a 64-character lowercase hex hash",
        )
    return text


@dataclass(slots=True)
class DemoConfig:
    max_participants: int
    quorum: int
    rounds: int
    preset: str = "swipe-dot-tiny"

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DemoConfig":
        try:
            max_participants = int(
                payload.get("maxParticipants", payload.get("max_participants", 4))
            )
            quorum = int(payload.get("quorum", payload.get("minTrainers", 2)))
            rounds = int(payload.get("rounds", 2))
        except (TypeError, ValueError) as exc:
            raise FederatedDemoError(
                "invalid_config", "run shape fields must be integers"
            ) from exc
        preset = str(payload.get("preset", "swipe-dot-tiny"))
        if max_participants < 1 or max_participants > 64:
            raise FederatedDemoError(
                "invalid_config", "maxParticipants must be in [1, 64]"
            )
        if quorum < 1 or quorum > max_participants:
            raise FederatedDemoError(
                "invalid_config", "quorum must be in [1, maxParticipants]"
            )
        if rounds < 1 or rounds > 50:
            raise FederatedDemoError("invalid_config", "rounds must be in [1, 50]")
        if preset not in DEMO_PRESETS:
            raise FederatedDemoError("invalid_config", f"unknown demo preset: {preset}")
        return cls(
            max_participants=max_participants,
            quorum=quorum,
            rounds=rounds,
            preset=preset,
        )

    def as_payload(self) -> dict[str, int | str]:
        return {
            "maxParticipants": self.max_participants,
            "quorum": self.quorum,
            "rounds": self.rounds,
            "preset": self.preset,
        }


@dataclass(slots=True)
class DemoSafetyConfig:
    max_public_participants: int = 8
    max_public_rounds: int = 3
    max_artifact_bytes: int = 8192
    max_message_bytes: int = 16384
    max_update_vector_length: int = 32
    max_events: int = 1000
    token_ttl_ms: int = 4 * 60 * 60 * 1000
    heartbeat_stale_ms: int = 15_000
    participant_timeout_ms: int = 45_000
    rate_limit_per_minute: int = 120
    clip_norm: float = DEFAULT_CLIP_NORM

    def as_payload(self) -> dict[str, int | float]:
        return {
            "maxPublicParticipants": self.max_public_participants,
            "maxPublicRounds": self.max_public_rounds,
            "maxArtifactBytes": self.max_artifact_bytes,
            "maxMessageBytes": self.max_message_bytes,
            "maxUpdateVectorLength": self.max_update_vector_length,
            "maxEvents": self.max_events,
            "tokenTtlMs": self.token_ttl_ms,
            "heartbeatStaleMs": self.heartbeat_stale_ms,
            "participantTimeoutMs": self.participant_timeout_ms,
            "rateLimitPerMinute": self.rate_limit_per_minute,
            "clipNorm": self.clip_norm,
        }


@dataclass(slots=True)
class DemoEvent:
    seq: int
    at: int
    kind: str
    message: str
    run_state: str
    round: int
    severity: Literal["info", "warn", "error"] = "info"
    actor: str = "system"
    participant_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema": EVENT_SCHEMA,
            "seq": self.seq,
            "at": self.at,
            "kind": self.kind,
            "severity": self.severity,
            "actor": self.actor,
            "participantId": self.participant_id,
            "message": self.message,
            "runState": self.run_state,
            "round": self.round,
            "payload": deepcopy(self.payload),
            "residencySafe": True,
        }


@dataclass(slots=True)
class ParticipantState:
    id: str
    token: str
    session_id: str | None = None
    display_name: str | None = None
    state: str = "joined"
    connection_state: str = "connected"
    joined_at: int = 0
    last_heartbeat_at: int | None = None
    last_connection_at: int | None = None
    last_disconnect_at: int | None = None
    reconnect_count: int = 0
    last_seen_seq: int = -1
    round: int | None = None
    progress: float = 0.0
    submitted_rounds: set[int] = field(default_factory=set)
    error: str | None = None
    update_metadata: dict[int, dict[str, Any]] = field(default_factory=dict)

    def redacted_update_metadata(self) -> dict[str, dict[str, Any]]:
        redacted: dict[str, dict[str, Any]] = {}
        for round_index, metadata in sorted(self.update_metadata.items()):
            public = {
                key: deepcopy(metadata[key])
                for key in _REDACTED_UPDATE_FIELDS
                if key in metadata
            }
            if "vector" in metadata:
                public["vectorLength"] = len(metadata["vector"])
            redacted[str(round_index)] = public
        return redacted

    def public_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "displayName": self.display_name,
            "state": self.state,
            "connectionState": self.connection_state,
            "joinedAt": self.joined_at,
            "lastHeartbeatAt": self.last_heartbeat_at,
            "lastConnectionAt": self.last_connection_at,
            "lastDisconnectAt": self.last_disconnect_at,
            "reconnectCount": self.reconnect_count,
            "lastSeenSeq": self.last_seen_seq,
            "round": self.round,
            "progress": self.progress,
            "submittedRounds": sorted(self.submitted_rounds),
            "error": self.error,
            "updateMetadata": self.redacted_update_metadata(),
        }


@dataclass(slots=True)
class DemoRun:
    id: str
    join_token: str
    config: DemoConfig
    created_at: int
    state: str = "created"
    round: int = 0
    participants: dict[str, ParticipantState] = field(default_factory=dict)
    events: list[DemoEvent] = field(default_factory=list)
    next_event_seq: int = 0
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    abort_reason: str | None = None
    failure_reason: str | None = None
    aggregation_mode: str = "tiny-vector-mean"
    learner_runtime: str = "js-worker-tiny-jepa-v1"
    current_model_revision_id: str = INITIAL_REVISION_ID
    model_revisions: list[dict[str, Any]] = field(default_factory=list)
    round_metrics: list[dict[str, Any]] = field(default_factory=list)
    round_started_at: int | None = None


class FederatedDemoService:
    """In-memory local browser-demo service."""

    def __init__(
        self,
        *,
        public_base_url: str = "/web/federated-demo/",
        public_demo: bool = False,
        deployment_target: str = "local",
        transport_mode: str = "http-polling",
        safety: DemoSafetyConfig | None = None,
        allowed_origins: tuple[str, ...] = ("same-origin", "local", "tunnel"),
    ) -> None:
        self.public_base_url = public_base_url
        self.public_demo = public_demo
        self.deployment_target = deployment_target
        self.transport_mode = transport_mode
        self.safety = safety or DemoSafetyConfig()
        self.allowed_origins = allowed_origins
        self._runs: dict[str, DemoRun] = {}

    def create_run(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        config = DemoConfig.from_payload(payload or {})
        self._validate_public_demo_config(config)
        run = DemoRun(
            id=_token("run-", 8),
            join_token=_token("tok-", 18),
            config=config,
            created_at=_now_ms(),
        )
        self._runs[run.id] = run
        self._emit(
            run,
            "run.created",
            f"run {run.id} created",
            payload={
                **config.as_payload(),
                "claimBoundary": CLAIM_BOUNDARY,
                "learnerRuntime": run.learner_runtime,
                "modelRevisionId": run.current_model_revision_id,
                "deployment": self.deployment_payload(),
            },
        )
        return self.snapshot(run.id)

    def snapshot(self, run_id: str) -> dict[str, Any]:
        run = self._get_run(run_id)
        self.refresh_liveness(run.id)
        return {
            "schema": RUN_SCHEMA,
            "mode": "backend-api",
            "id": run.id,
            "state": run.state,
            "round": run.round,
            "config": run.config.as_payload(),
            "joinToken": run.join_token,
            "joinUrl": self.join_url(run),
            "webSocketUrl": self.websocket_url(run),
            "claimBoundary": CLAIM_BOUNDARY,
            "aggregationMode": run.aggregation_mode,
            "learnerRuntime": run.learner_runtime,
            "modelRuntime": HACKATHON_MODEL_RUNTIME,
            "currentModelRevisionId": run.current_model_revision_id,
            "deployment": self.deployment_payload(),
            "safety": self.safety.as_payload(),
            "participants": [p.public_payload() for p in run.participants.values()],
            "events": [e.as_payload() for e in run.events[-250:]],
            "artifacts": deepcopy(run.artifacts),
            "modelRevisions": deepcopy(run.model_revisions),
            "roundMetrics": deepcopy(run.round_metrics),
            "abortReason": run.abort_reason,
            "failureReason": run.failure_reason,
            "controls": {
                "canStart": run.state == "ready",
                "canAbort": run.state not in TERMINAL_RUN_STATES,
                "pauseSupported": False,
                "canDropParticipant": run.state not in TERMINAL_RUN_STATES,
                "canExport": True,
            },
        }

    def join_url(self, run: DemoRun) -> str:
        base = self.public_base_url.rstrip("/")
        return f"{base}/#/join/{run.id}?t={run.join_token}"

    def websocket_url(self, run: DemoRun) -> str:
        base = self.public_base_url.rstrip("/")
        if base.startswith("https://"):
            ws_base = "wss://" + base.removeprefix("https://")
        elif base.startswith("http://"):
            ws_base = "ws://" + base.removeprefix("http://")
        else:
            ws_base = base
        path = ws_base.split("#", 1)[0]
        if path.endswith("/web/federated-demo"):
            path = path[: -len("/web/federated-demo")]
        return f"{path.rstrip('/')}/api/runs/{run.id}/ws"

    def deployment_payload(self) -> dict[str, Any]:
        return {
            "publicDemo": self.public_demo,
            "target": self.deployment_target,
            "transportMode": self.transport_mode,
            "publicBaseUrl": self.public_base_url,
            "allowedOrigins": list(self.allowed_origins),
            "fallbacks": ["http-polling", "cloudflare-tunnel", "lan-hotspot"],
        }

    def events(self, run_id: str, *, after: int = -1) -> list[dict[str, Any]]:
        run = self._get_run(run_id)
        self.refresh_liveness(run.id)
        return [event.as_payload() for event in run.events if event.seq > after]

    def model_revision(self, run_id: str, revision_id: str) -> dict[str, Any]:
        run = self._get_run(run_id)
        for revision in run.model_revisions:
            if revision.get("modelRevisionId") == revision_id:
                return deepcopy(revision)
        raise FederatedDemoError(
            "not_found", f"unknown model revision {revision_id}", status=404
        )

    def join_run(
        self,
        run_id: str,
        *,
        join_token: str,
        display_name: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        run = self._get_run(run_id)
        self._require_join_token(run, join_token)
        if run.state in TERMINAL_RUN_STATES:
            raise FederatedDemoError("run_closed", f"run is {run.state}", status=409)
        if run.state not in {"created", "joining", "ready"}:
            raise FederatedDemoError(
                "already_started", "run already started; joining is closed", status=409
            )
        for participant in run.participants.values():
            if session_id is not None and participant.session_id == session_id:
                participant.last_heartbeat_at = _now_ms()
                participant.last_connection_at = _now_ms()
                participant.last_seen_seq = run.next_event_seq - 1
                if participant.connection_state in {"reconnecting", "stale"}:
                    participant.reconnect_count += 1
                participant.connection_state = "connected"
                self._emit(
                    run,
                    "participant.resumed",
                    f"participant {participant.id} resumed existing browser slot",
                    participant_id=participant.id,
                    actor="participant",
                    payload={"state": participant.state},
                )
                return {
                    "participantId": participant.id,
                    "participantToken": participant.token,
                    "run": self.snapshot(run.id),
                }
        if len(run.participants) >= run.config.max_participants:
            self._emit(
                run,
                "participant.rejected",
                "join rejected: run is full",
                severity="warn",
                payload={"reason": "run_full"},
            )
            raise FederatedDemoError("run_full", "run is full", status=409)

        participant = ParticipantState(
            id=_token("browser-", 6),
            token=_token("ptok-", 18),
            session_id=session_id,
            display_name=display_name or None,
            joined_at=_now_ms(),
            last_heartbeat_at=_now_ms(),
            last_connection_at=_now_ms(),
            last_seen_seq=run.next_event_seq,
        )
        run.participants[participant.id] = participant
        self._emit(
            run,
            "participant.joined",
            f"participant {participant.id} joined",
            participant_id=participant.id,
            actor="participant",
            payload={"slot": len(run.participants), "of": run.config.max_participants},
        )
        self._transition_run(run, "joining", "first participant joined")
        self._set_participant_state(run, participant, "ready")
        self._emit(
            run,
            "participant.ready",
            f"participant {participant.id} is ready",
            participant_id=participant.id,
            actor="participant",
        )
        if (
            run.state == "joining"
            and len(self._active_participants(run)) >= run.config.quorum
        ):
            self._transition_run(run, "ready", f"quorum of {run.config.quorum} reached")
        return {
            "participantId": participant.id,
            "participantToken": participant.token,
            "run": self.snapshot(run.id),
        }

    def heartbeat(
        self, run_id: str, participant_id: str, *, participant_token: str
    ) -> dict[str, Any]:
        run, participant = self._participant(run_id, participant_id, participant_token)
        participant.last_heartbeat_at = _now_ms()
        participant.connection_state = "connected"
        participant.last_seen_seq = run.next_event_seq - 1
        self._emit(
            run,
            "participant.heartbeat",
            f"heartbeat from {participant.id}",
            participant_id=participant.id,
            actor="participant",
            payload={"state": participant.state},
        )
        return {"ok": True, "run": self.snapshot(run.id)}

    def connection_opened(
        self,
        run_id: str,
        *,
        role: str,
        participant_id: str | None = None,
        participant_token: str | None = None,
        after: int = -1,
        transport: str = "websocket",
    ) -> dict[str, Any]:
        run = self._get_run(run_id)
        if role not in {"host", "participant"}:
            raise FederatedDemoError(
                "invalid_role", f"unknown connection role {role!r}"
            )
        if role == "participant":
            if participant_id is None or participant_token is None:
                raise FederatedDemoError(
                    "invalid_participant_token",
                    "participant WebSocket requires participant id and token",
                    status=403,
                )
            _, participant = self._participant(
                run_id, participant_id, participant_token
            )
            if participant.connection_state in {"reconnecting", "stale"}:
                participant.reconnect_count += 1
            participant.connection_state = "connected"
            participant.last_connection_at = _now_ms()
            participant.last_heartbeat_at = _now_ms()
            participant.last_seen_seq = after
        self._emit(
            run,
            "connection.opened",
            f"{role} connected over {transport}",
            actor=role,
            participant_id=participant_id,
            payload={"role": role, "transport": transport, "after": after},
        )
        return {
            "run": self.snapshot(run.id),
            "events": self.events(run.id, after=after),
        }

    def connection_closed(
        self,
        run_id: str,
        *,
        role: str,
        participant_id: str | None = None,
        transport: str = "websocket",
        reason: str = "socket closed",
    ) -> dict[str, Any]:
        run = self._get_run(run_id)
        if participant_id is not None:
            participant = run.participants.get(participant_id)
            if participant and participant.state not in {
                "completed",
                "dropped",
                "error",
            }:
                participant.connection_state = "reconnecting"
                participant.last_disconnect_at = _now_ms()
        self._emit(
            run,
            "connection.closed",
            f"{role} disconnected from {transport}",
            actor=role,
            participant_id=participant_id,
            severity="warn" if participant_id else "info",
            payload={"role": role, "transport": transport, "reason": reason},
        )
        return {"ok": True, "run": self.snapshot(run.id)}

    def start_run(self, run_id: str) -> dict[str, Any]:
        run = self._get_run(run_id)
        if run.state != "ready":
            raise FederatedDemoError(
                "not_ready", f"cannot start from state {run.state}", status=409
            )
        self._assign_round(run, 1)
        return self.snapshot(run.id)

    def abort_run(self, run_id: str, *, reason: str = "host abort") -> dict[str, Any]:
        run = self._get_run(run_id)
        if run.state in TERMINAL_RUN_STATES:
            raise FederatedDemoError(
                "run_closed", f"run is already {run.state}", status=409
            )
        run.abort_reason = reason
        run.state = "aborted"
        for participant in self._active_participants(run):
            participant.state = "dropped"
            participant.connection_state = "dropped"
            participant.error = reason
        self._emit(
            run, "run.aborted", f"run aborted: {reason}", severity="warn", actor="host"
        )
        return self.snapshot(run.id)

    def fail_run(self, run_id: str, *, reason: str = "demo failure") -> dict[str, Any]:
        run = self._get_run(run_id)
        if run.state in TERMINAL_RUN_STATES:
            raise FederatedDemoError(
                "run_closed", f"run is already {run.state}", status=409
            )
        run.failure_reason = reason
        run.state = "failed"
        for participant in self._active_participants(run):
            participant.state = "error"
            participant.connection_state = "dropped"
            participant.error = reason
        self._emit(
            run, "run.failed", f"run failed: {reason}", severity="error", actor="host"
        )
        return self.snapshot(run.id)

    def update_progress(
        self,
        run_id: str,
        participant_id: str,
        *,
        participant_token: str,
        progress: float,
    ) -> dict[str, Any]:
        run, participant = self._participant(run_id, participant_id, participant_token)
        if run.state != "running_round" or participant.round != run.round:
            raise FederatedDemoError(
                "wrong_round",
                "participant is not assigned to the active round",
                status=409,
            )
        if participant.state == "assigned":
            self._set_participant_state(run, participant, "training")
            self._emit(
                run,
                "participant.training",
                f"{participant.id} started browser-local tiny learner work",
                participant_id=participant.id,
                actor="participant",
                payload={
                    "runtime": run.learner_runtime,
                    "modelRevisionId": run.current_model_revision_id,
                },
            )
        if participant.state != "training":
            raise FederatedDemoError(
                "invalid_state",
                f"cannot report progress from {participant.state}",
                status=409,
            )
        participant.progress = min(1.0, max(0.0, float(progress)))
        participant.last_heartbeat_at = _now_ms()
        self._emit(
            run,
            "participant.heartbeat",
            f"{participant.id} progress {participant.progress:.2f}",
            participant_id=participant.id,
            actor="participant",
            payload={
                "progress": round(participant.progress, 4),
                "round": run.round,
                "connectionState": participant.connection_state,
            },
        )
        return {"ok": True, "run": self.snapshot(run.id)}

    def submit_update(
        self,
        run_id: str,
        participant_id: str,
        *,
        participant_token: str,
        artifact: dict[str, Any],
    ) -> dict[str, Any]:
        run, participant = self._participant(run_id, participant_id, participant_token)
        if run.state != "running_round":
            raise FederatedDemoError(
                "wrong_round", "run is not collecting browser updates", status=409
            )
        if participant.round != run.round:
            raise FederatedDemoError(
                "wrong_round",
                "participant is not assigned to the active round",
                status=409,
            )
        if run.round in participant.submitted_rounds:
            self._emit(
                run,
                "update.rejected",
                f"duplicate update from {participant.id}",
                participant_id=participant.id,
                actor="participant",
                severity="warn",
                payload={"reason": "duplicate_update", "round": run.round},
            )
            raise FederatedDemoError(
                "duplicate_update",
                "participant already submitted this round",
                status=409,
            )
        if participant.state not in {"assigned", "training"}:
            raise FederatedDemoError(
                "invalid_state", f"cannot submit from {participant.state}", status=409
            )
        metadata = self._validate_update_artifact(
            artifact, run=run, participant=participant
        )
        participant.submitted_rounds.add(run.round)
        participant.update_metadata[run.round] = metadata
        participant.state = "submitted"
        participant.progress = 1.0
        participant.last_heartbeat_at = _now_ms()
        self._emit(
            run,
            "update.submitted",
            f"{participant.id} submitted bounded browser update",
            participant_id=participant.id,
            actor="participant",
            payload={
                "schema": metadata["schema"],
                "round": metadata["round"],
                "roundId": metadata["roundId"],
                "modelRevisionId": metadata["modelRevisionId"],
                "shape": metadata["shape"],
                "parameterCount": metadata["parameterCount"],
                "hash": metadata["hash"],
                "l2Norm": metadata["l2Norm"],
                "clipNorm": metadata["clipNorm"],
                "source": metadata["source"],
                "simulated": metadata["simulated"],
            },
        )
        if self._submitted_count(run) >= run.config.quorum:
            self._close_round(run)
        return {"ok": True, "run": self.snapshot(run.id)}

    def expire_missing(
        self, run_id: str, *, reason: str = "participant timeout"
    ) -> dict[str, Any]:
        run = self._get_run(run_id)
        if run.state != "running_round":
            raise FederatedDemoError(
                "wrong_round", "no active round to expire", status=409
            )
        for participant in self._active_participants(run):
            if participant.round == run.round and participant.state in {
                "assigned",
                "training",
            }:
                self._drop_participant(run, participant, reason=reason)
        if self._submitted_count(run) >= run.config.quorum:
            self._close_round(run)
        else:
            self.fail_run(run.id, reason="quorum lost after participant timeout")
        return self.snapshot(run.id)

    def drop_participant(
        self, run_id: str, participant_id: str, *, reason: str = "host drop"
    ) -> dict[str, Any]:
        run = self._get_run(run_id)
        participant = run.participants.get(participant_id)
        if participant is None:
            raise FederatedDemoError(
                "unknown_participant",
                f"unknown participant {participant_id}",
                status=404,
            )
        self._drop_participant(run, participant, reason=reason)
        if (
            run.state == "running_round"
            and len(self._active_participants(run)) < run.config.quorum
        ):
            self.fail_run(run.id, reason="quorum lost after participant drop")
        elif (
            run.state == "running_round"
            and self._submitted_count(run) >= run.config.quorum
        ):
            self._close_round(run)
        elif (
            run.state in {"joining", "ready"}
            and len(self._active_participants(run)) < run.config.quorum
        ):
            self._transition_run(
                run, "joining", "quorum waiting after participant drop"
            )
        return self.snapshot(run.id)

    def export_evidence(self, run_id: str) -> dict[str, Any]:
        run = self._get_run(run_id)
        self.refresh_liveness(run.id)
        update_hashes = [
            metadata["hash"]
            for participant in run.participants.values()
            for metadata in participant.update_metadata.values()
        ]
        return {
            "schema": EVIDENCE_SCHEMA,
            "runId": run.id,
            "mode": "backend-api",
            "publicMode": "public" if self.public_demo else "local",
            "transportMode": self.transport_mode,
            "deploymentTarget": self.deployment_target,
            "deployment": self.deployment_payload(),
            "createdAt": run.created_at,
            "exportedAt": _now_ms(),
            "state": run.state,
            "round": run.round,
            "config": run.config.as_payload(),
            "claimBoundary": CLAIM_BOUNDARY,
            "nonClaimText": CLAIM_BOUNDARY,
            "aggregationMode": run.aggregation_mode,
            "learnerRuntime": run.learner_runtime,
            "modelRuntime": HACKATHON_MODEL_RUNTIME,
            "currentModelRevisionId": run.current_model_revision_id,
            "eventTrace": [e.as_payload() for e in run.events],
            "participants": [p.public_payload() for p in run.participants.values()],
            "artifacts": deepcopy(run.artifacts),
            "modelRevisions": deepcopy(run.model_revisions),
            "roundMetrics": deepcopy(run.round_metrics),
            "updateHashes": sorted(update_hashes),
            "modelRevisionRefs": [
                {
                    "modelRevisionId": revision["modelRevisionId"],
                    "sha256": revision["sha256"],
                    "round": revision["round"],
                    "sourceUpdateHashes": revision["sourceUpdateHashes"],
                }
                for revision in run.model_revisions
            ],
            "fallback": {
                "active": False,
                "mode": "tiny-js-vector",
                "reason": "hackathon tiny model runtime selected",
            },
            "liveness": {
                "participants": [
                    {
                        "id": participant.id,
                        "state": participant.state,
                        "connectionState": participant.connection_state,
                        "lastHeartbeatAt": participant.last_heartbeat_at,
                        "reconnectCount": participant.reconnect_count,
                    }
                    for participant in run.participants.values()
                ]
            },
            "redaction": {
                "residencySafe": True,
                "rawParticipantDataIncluded": False,
                "modelWeightsIncluded": False,
                "participantTokensIncluded": False,
                "updatePayload": "bounded derived tiny update vector plus shape/hash/norm/sample-count metadata",
            },
        }

    def _get_run(self, run_id: str) -> DemoRun:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise FederatedDemoError(
                "not_found", f"unknown run {run_id}", status=404
            ) from exc

    def _validate_public_demo_config(self, config: DemoConfig) -> None:
        if not self.public_demo:
            return
        if config.max_participants > self.safety.max_public_participants:
            raise FederatedDemoError(
                "invalid_config",
                f"public demo maxParticipants must be <= {self.safety.max_public_participants}",
            )
        if config.rounds > self.safety.max_public_rounds:
            raise FederatedDemoError(
                "invalid_config",
                f"public demo rounds must be <= {self.safety.max_public_rounds}",
            )

    def refresh_liveness(self, run_id: str, *, now_ms: int | None = None) -> None:
        run = self._get_run(run_id)
        if run.state in TERMINAL_RUN_STATES:
            return
        now = _now_ms() if now_ms is None else now_ms
        for participant in run.participants.values():
            if participant.state in {"completed", "dropped", "error"}:
                continue
            last = participant.last_heartbeat_at or participant.joined_at
            if now - last >= self.safety.participant_timeout_ms:
                if participant.connection_state != "dropped":
                    self._drop_participant(
                        run, participant, reason="participant heartbeat timeout"
                    )
            elif now - last >= self.safety.heartbeat_stale_ms:
                if participant.connection_state != "stale":
                    participant.connection_state = "stale"
                    self._emit(
                        run,
                        "participant.stale",
                        f"{participant.id} heartbeat is stale",
                        participant_id=participant.id,
                        severity="warn",
                        payload={"lastHeartbeatAt": last},
                    )
        if (
            run.state == "running_round"
            and len(self._active_participants(run)) < run.config.quorum
            and self._submitted_count(run) < run.config.quorum
        ):
            self.fail_run(run.id, reason="quorum lost after participant timeout")

    def _require_join_token(self, run: DemoRun, join_token: str) -> None:
        if _now_ms() - run.created_at > self.safety.token_ttl_ms:
            self._emit(
                run,
                "participant.rejected",
                "join rejected: token expired",
                severity="warn",
                payload={"reason": "token_expired"},
            )
            raise FederatedDemoError(
                "token_expired", "join token has expired", status=403
            )
        if not join_token or join_token != run.join_token:
            self._emit(
                run,
                "participant.rejected",
                "join rejected: invalid token",
                severity="warn",
                payload={"reason": "invalid_token"},
            )
            raise FederatedDemoError(
                "invalid_token", "join token is invalid", status=403
            )

    def _participant(
        self, run_id: str, participant_id: str, participant_token: str
    ) -> tuple[DemoRun, ParticipantState]:
        run = self._get_run(run_id)
        participant = run.participants.get(participant_id)
        if participant is None:
            raise FederatedDemoError(
                "unknown_participant",
                f"unknown participant {participant_id}",
                status=404,
            )
        if participant.token != participant_token:
            raise FederatedDemoError(
                "invalid_participant_token", "participant token is invalid", status=403
            )
        return run, participant

    def _emit(
        self,
        run: DemoRun,
        kind: str,
        message: str,
        *,
        severity: Literal["info", "warn", "error"] = "info",
        actor: str = "system",
        participant_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        safe_payload = deepcopy(payload or {})
        forbidden = _contains_forbidden_key(safe_payload)
        if forbidden is not None:
            raise FederatedDemoError(
                "unsafe_event_payload",
                f"event payload contains forbidden raw-data key {forbidden!r}",
            )
        run.events.append(
            DemoEvent(
                seq=run.next_event_seq,
                at=_now_ms(),
                kind=kind,
                severity=severity,
                actor=actor,
                participant_id=participant_id,
                message=message,
                run_state=run.state,
                round=run.round,
                payload=safe_payload,
            )
        )
        run.next_event_seq += 1
        if len(run.events) > self.safety.max_events:
            del run.events[: len(run.events) - self.safety.max_events]

    def _transition_run(self, run: DemoRun, state: str, message: str) -> None:
        if run.state == state:
            return
        if run.state in TERMINAL_RUN_STATES:
            raise FederatedDemoError(
                "run_closed", f"run is already {run.state}", status=409
            )
        run.state = state
        self._emit(run, "run.state", message)

    def _set_participant_state(
        self, run: DemoRun, participant: ParticipantState, state: str
    ) -> None:
        if participant.state in {"completed", "dropped", "error"}:
            return
        participant.state = state

    def _active_participants(self, run: DemoRun) -> list[ParticipantState]:
        return [
            p for p in run.participants.values() if p.state not in {"dropped", "error"}
        ]

    def _assigned_participants(self, run: DemoRun) -> list[ParticipantState]:
        return [p for p in self._active_participants(run) if p.round == run.round]

    def _submitted_count(self, run: DemoRun) -> int:
        return sum(
            1
            for p in self._assigned_participants(run)
            if run.round in p.submitted_rounds
        )

    def _assign_round(self, run: DemoRun, round_index: int) -> None:
        if len(self._active_participants(run)) < run.config.quorum:
            raise FederatedDemoError("not_ready", "quorum is not available", status=409)
        run.round = round_index
        run.state = "running_round"
        run.round_started_at = _now_ms()
        self._emit(
            run,
            "run.state",
            f"round {round_index} started",
            payload={"modelRevisionId": run.current_model_revision_id},
        )
        for participant in self._active_participants(run):
            if participant.state in {"ready", "submitted"}:
                participant.state = "assigned"
                participant.round = round_index
                participant.progress = 0.0
                self._emit(
                    run,
                    "round.assigned",
                    f"round {round_index} assigned to {participant.id}",
                    participant_id=participant.id,
                    payload={
                        "round": round_index,
                        "roundId": self._round_id(run),
                        "modelRevisionId": run.current_model_revision_id,
                    },
                )

    def _drop_participant(
        self, run: DemoRun, participant: ParticipantState, *, reason: str
    ) -> None:
        participant.state = "dropped"
        participant.connection_state = "dropped"
        participant.error = reason
        self._emit(
            run,
            "participant.dropped",
            f"{participant.id} dropped: {reason}",
            participant_id=participant.id,
            severity="warn",
            payload={"reason": reason, "round": run.round},
        )

    def _close_round(self, run: DemoRun) -> None:
        submitted = [
            p
            for p in self._assigned_participants(run)
            if run.round in p.submitted_rounds
        ]
        if len(submitted) < run.config.quorum:
            raise FederatedDemoError(
                "not_ready",
                "not enough submitted updates to close the round",
                status=409,
            )
        run.state = "aggregating"
        self._emit(
            run,
            "round.aggregating",
            f"aggregating round {run.round}",
            payload={
                "aggregationMode": run.aggregation_mode,
                "submitted": len(submitted),
                "quorum": run.config.quorum,
            },
        )
        update_metadata = [p.update_metadata[run.round] for p in submitted]
        update_hashes = [metadata["hash"] for metadata in update_metadata]
        aggregate_vector = self._mean_vector(
            [metadata["vector"] for metadata in update_metadata]
        )
        aggregate_norm = self._vector_norm(aggregate_vector)
        parent_revision = run.current_model_revision_id
        revision_hash = _hash_json(
            {
                "run": run.id,
                "round": run.round,
                "parentRevision": parent_revision,
                "aggregationMode": run.aggregation_mode,
                "vector": aggregate_vector,
                "updates": sorted(update_hashes),
            }
        )
        model_revision_id = f"rev-{revision_hash[:12]}"
        checkpoint_hash = _hash_json(
            {
                "run": run.id,
                "round": run.round,
                "modelRevisionId": model_revision_id,
                "aggregationMode": run.aggregation_mode,
                "updates": sorted(update_hashes),
            }
        )
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "kind": "checkpoint",
            "label": f"round-{run.round} tiny-vector checkpoint",
            "round": run.round,
            "sha256": checkpoint_hash,
            "source": "browser-submitted bounded update vectors",
            "aggregationMode": run.aggregation_mode,
            "contributingParticipants": [p.id for p in submitted],
            "sourceUpdateHashes": sorted(update_hashes),
            "modelRevisionId": model_revision_id,
            "simulated": False,
            "containsModelWeights": False,
            "containsTinyDerivedVector": False,
        }
        run.artifacts.append(checkpoint)
        model_revision = {
            "schema": MODEL_REVISION_SCHEMA,
            "kind": "model-revision",
            "label": f"round-{run.round} hackathon tiny model revision",
            "round": run.round,
            "roundId": self._round_id(run),
            "sha256": revision_hash,
            "modelRevisionId": model_revision_id,
            "parentModelRevisionId": parent_revision,
            "runtime": HACKATHON_MODEL_RUNTIME,
            "shape": [len(aggregate_vector)],
            "parameterCount": len(aggregate_vector),
            "vector": aggregate_vector,
            "aggregateNorm": round(aggregate_norm, 8),
            "sourceUpdateHashes": sorted(update_hashes),
            "contributingParticipants": [p.id for p in submitted],
            "aggregationMode": run.aggregation_mode,
            "containsModelWeights": False,
            "containsTinyDerivedVector": True,
            "fallback": False,
        }
        run.model_revisions.append(model_revision)
        run.current_model_revision_id = model_revision_id
        elapsed_ms = (
            max(0, _now_ms() - run.round_started_at)
            if run.round_started_at is not None
            else None
        )
        run.round_metrics.append(
            {
                "round": run.round,
                "roundId": self._round_id(run),
                "submitted": len(submitted),
                "quorum": run.config.quorum,
                "participantCount": len(self._assigned_participants(run)),
                "aggregateNorm": round(aggregate_norm, 8),
                "modelRevisionId": model_revision_id,
                "sourceUpdateHashes": sorted(update_hashes),
                "elapsedMs": elapsed_ms,
                "localLossMean": self._mean_optional(
                    [metadata.get("loss") for metadata in update_metadata]
                ),
                "probeMean": self._mean_optional(
                    [metadata.get("probe") for metadata in update_metadata]
                ),
            }
        )
        self._emit(
            run,
            "round.closed",
            f"round {run.round} closed",
            payload={
                "checkpoint": checkpoint_hash,
                "modelRevisionId": model_revision_id,
                "revisionHash": revision_hash,
                "contributing": len(submitted),
                "aggregateNorm": round(aggregate_norm, 8),
                "simulated": False,
            },
        )
        if run.round < run.config.rounds:
            self._assign_round(run, run.round + 1)
            return
        for participant in self._active_participants(run):
            if run.round in participant.submitted_rounds:
                participant.state = "completed"
                participant.connection_state = "completed"
            elif participant.state in {"assigned", "training"}:
                self._drop_participant(
                    run, participant, reason="round closed before update"
                )
        run.state = "checkpoint_ready"
        self._emit(run, "checkpoint.ready", "final checkpoint-like artifact ready")
        inference_hash = _hash_json(
            {
                "modelRevision": revision_hash,
                "schema": INFERENCE_SCHEMA,
                "preset": run.config.preset,
            }
        )
        run.artifacts.append(
            {
                "schema": INFERENCE_SCHEMA,
                "kind": "inference-model",
                "label": "browser inference attachment from hackathon tiny model",
                "round": run.round,
                "sha256": inference_hash,
                "sourceCheckpoint": checkpoint_hash,
                "sourceModelRevision": revision_hash,
                "modelRevisionId": model_revision_id,
                "modelId": f"lensemble-demo/{run.id}",
                "revision": inference_hash[:12],
                "runtime": HACKATHON_MODEL_RUNTIME,
                "shape": [len(aggregate_vector)],
                "parameterCount": len(aggregate_vector),
                "vector": aggregate_vector,
                "aggregateNorm": round(aggregate_norm, 8),
                "preset": run.config.preset,
                "simulated": False,
                "containsModelWeights": False,
                "containsTinyDerivedVector": True,
                "fallback": False,
            }
        )
        run.state = "inference_ready"
        self._emit(run, "inference.ready", "browser inference artifact metadata ready")
        run.state = "completed"
        self._emit(run, "run.completed", "run completed")

    def _validate_update_artifact(
        self, artifact: dict[str, Any], *, run: DemoRun, participant: ParticipantState
    ) -> dict[str, Any]:
        if not isinstance(artifact, dict):
            raise FederatedDemoError(
                "invalid_artifact", "update artifact must be a JSON object"
            )
        encoded_size = len(json.dumps(artifact, sort_keys=True).encode("utf-8"))
        if encoded_size > self.safety.max_artifact_bytes:
            raise FederatedDemoError(
                "artifact_too_large",
                f"update artifact exceeds {self.safety.max_artifact_bytes} bytes",
                status=413,
            )
        forbidden = _contains_forbidden_key(artifact)
        if forbidden is not None:
            raise FederatedDemoError(
                "raw_data_forbidden",
                f"browser update artifact contains forbidden raw-data key {forbidden!r}",
            )
        if artifact.get("schema") != UPDATE_SCHEMA:
            raise FederatedDemoError(
                "invalid_artifact", f"artifact schema must be {UPDATE_SCHEMA}"
            )
        if artifact.get("runId") != run.id:
            raise FederatedDemoError(
                "invalid_artifact", "artifact runId does not match run"
            )
        if artifact.get("participantId") != participant.id:
            raise FederatedDemoError(
                "invalid_artifact", "artifact participantId does not match participant"
            )
        if int(artifact.get("round", -1)) != run.round:
            raise FederatedDemoError(
                "wrong_round", "artifact round does not match active round", status=409
            )
        expected_round_id = self._round_id(run)
        if artifact.get("roundId", expected_round_id) != expected_round_id:
            raise FederatedDemoError(
                "wrong_round",
                "artifact roundId does not match active round",
                status=409,
            )
        if artifact.get("modelRevisionId") != run.current_model_revision_id:
            raise FederatedDemoError(
                "stale_model_revision",
                "artifact modelRevisionId does not match the active model revision",
                status=409,
            )
        shape = artifact.get("shape")
        if (
            not isinstance(shape, list)
            or not shape
            or any(not isinstance(v, int) or v <= 0 for v in shape)
        ):
            raise FederatedDemoError(
                "invalid_artifact",
                "artifact shape must be a non-empty positive integer list",
            )
        vector = artifact.get("vector")
        if not isinstance(vector, list):
            raise FederatedDemoError(
                "invalid_artifact",
                "artifact vector must be a list matching shape",
            )
        if len(vector) <= 0 or len(vector) > self.safety.max_update_vector_length:
            raise FederatedDemoError(
                "invalid_artifact",
                f"artifact vector length must be in [1, {self.safety.max_update_vector_length}]",
            )
        if len(shape) != 1 or shape[0] != len(vector):
            raise FederatedDemoError(
                "invalid_artifact",
                "only one-dimensional tiny update vectors are supported",
            )
        update_vector: list[float] = []
        for value in vector:
            if not isinstance(value, int | float):
                raise FederatedDemoError(
                    "invalid_artifact", "artifact vector values must be numeric"
                )
            update_vector.append(round(float(value), 8))
        parameter_count = int(artifact.get("parameterCount", len(update_vector)))
        if parameter_count != len(update_vector):
            raise FederatedDemoError(
                "invalid_artifact", "parameterCount must match vector length"
            )
        sample_count = int(artifact.get("sampleCount", 0))
        if sample_count <= 0:
            raise FederatedDemoError("invalid_artifact", "sampleCount must be positive")
        local_steps = int(artifact.get("localSteps", 0))
        if local_steps <= 0:
            raise FederatedDemoError("invalid_artifact", "localSteps must be positive")
        update_hash = _validate_hash(artifact.get("hash"), field_name="hash")
        l2_norm = float(artifact.get("l2Norm", 0.0))
        if l2_norm < 0:
            raise FederatedDemoError("invalid_artifact", "l2Norm must be non-negative")
        computed_norm = self._vector_norm(update_vector)
        if abs(l2_norm - computed_norm) > 1e-4:
            raise FederatedDemoError(
                "invalid_artifact", "l2Norm does not match the submitted vector"
            )
        clip_norm = float(artifact.get("clipNorm", self.safety.clip_norm))
        if clip_norm <= 0 or clip_norm > self.safety.clip_norm:
            raise FederatedDemoError(
                "invalid_artifact",
                f"clipNorm must be in (0, {self.safety.clip_norm}]",
            )
        if computed_norm > clip_norm + 1e-6:
            raise FederatedDemoError(
                "norm_bound_exceeded",
                "artifact vector exceeds the configured norm bound",
            )
        source = str(artifact.get("source", ""))
        if source not in {"browser-local-surrogate", "simulator"}:
            raise FederatedDemoError(
                "invalid_artifact",
                "artifact source must identify browser-local-surrogate or simulator",
            )
        if source == "simulator":
            run.aggregation_mode = "backend-simulator"
        else:
            run.aggregation_mode = "tiny-vector-mean"
        return {
            "schema": UPDATE_SCHEMA,
            "source": source,
            "runtime": str(artifact.get("runtime", run.learner_runtime)),
            "runId": run.id,
            "participantId": participant.id,
            "round": run.round,
            "roundId": expected_round_id,
            "modelRevisionId": run.current_model_revision_id,
            "shape": shape,
            "parameterCount": parameter_count,
            "vector": update_vector,
            "sampleCount": sample_count,
            "localSteps": local_steps,
            "hash": update_hash,
            "l2Norm": round(computed_norm, 8),
            "clipNorm": clip_norm,
            "loss": self._optional_float(artifact.get("loss")),
            "probe": self._optional_float(artifact.get("probe")),
            "runtimeMs": self._optional_float(artifact.get("runtimeMs")),
            "seed": int(artifact.get("seed", 0)),
            "simulated": bool(artifact.get("simulated", source == "simulator")),
        }

    def _round_id(self, run: DemoRun) -> str:
        return f"{run.id}:round-{run.round}"

    @staticmethod
    def _vector_norm(vector: list[float]) -> float:
        return sum(value * value for value in vector) ** 0.5

    @staticmethod
    def _mean_vector(vectors: list[list[float]]) -> list[float]:
        if not vectors:
            raise FederatedDemoError("not_ready", "no vectors to aggregate", status=409)
        width = len(vectors[0])
        if any(len(vector) != width for vector in vectors):
            raise FederatedDemoError(
                "invalid_artifact", "update vector shapes do not match"
            )
        return [
            round(sum(vector[index] for vector in vectors) / len(vectors), 8)
            for index in range(width)
        ]

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _mean_optional(values: list[object]) -> float | None:
        numeric = [float(value) for value in values if value is not None]
        if not numeric:
            return None
        return round(sum(numeric) / len(numeric), 8)
