"""Local browser federated-demo orchestration service.

This module intentionally implements a narrow educational demo backend, not a
production coordinator. It owns browser run allocation, participant admission,
residency-safe lifecycle events, browser-surrogate update artifacts, a tiny
coordinator-style aggregation path, inference artifact publication, and evidence
export for issues #294/#296-#301.

The hard boundary is that browser participants submit only versioned update
metadata: shape, hash, norm, sample count, and runtime labels. Raw observations,
actions, labels, latents, tensors, and model weights are rejected at the API
boundary.
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

RUN_SCHEMA = "demo-run/1"
EVENT_SCHEMA = "demo-event/1"
UPDATE_SCHEMA = "browser-update/1"
CHECKPOINT_SCHEMA = "demo-checkpoint/1"
INFERENCE_SCHEMA = "demo-inference-artifact/1"
EVIDENCE_SCHEMA = "demo-evidence/1"
DEMO_PRESETS = frozenset({"swipe-dot-tiny"})
ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"

CLAIM_BOUNDARY = (
    "Educational browser demo of federated JEPA world-model orchestration. "
    "It demonstrates local run orchestration, residency-safe browser update "
    "metadata, aggregation plumbing, checkpoint-like artifacts, and browser "
    "inference/env-sim. It is not a benchmark win over local-only, not a "
    "production browser-training claim, not a cryptographic proof of honest "
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
    display_name: str | None = None
    state: str = "joined"
    joined_at: int = 0
    last_heartbeat_at: int | None = None
    round: int | None = None
    progress: float = 0.0
    submitted_rounds: set[int] = field(default_factory=set)
    error: str | None = None
    update_metadata: dict[int, dict[str, Any]] = field(default_factory=dict)

    def public_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "displayName": self.display_name,
            "state": self.state,
            "joinedAt": self.joined_at,
            "lastHeartbeatAt": self.last_heartbeat_at,
            "round": self.round,
            "progress": self.progress,
            "submittedRounds": sorted(self.submitted_rounds),
            "error": self.error,
            "updateMetadata": {
                str(k): deepcopy(v) for k, v in sorted(self.update_metadata.items())
            },
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
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    abort_reason: str | None = None
    failure_reason: str | None = None
    aggregation_mode: str = "browser-surrogate-coordinator"
    learner_runtime: str = "js-worker-surrogate-v1"


class FederatedDemoService:
    """In-memory local browser-demo service."""

    def __init__(self, *, public_base_url: str = "/web/federated-demo/") -> None:
        self.public_base_url = public_base_url
        self._runs: dict[str, DemoRun] = {}

    def create_run(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        config = DemoConfig.from_payload(payload or {})
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
            },
        )
        return self.snapshot(run.id)

    def snapshot(self, run_id: str) -> dict[str, Any]:
        run = self._get_run(run_id)
        return {
            "schema": RUN_SCHEMA,
            "mode": "backend-api",
            "id": run.id,
            "state": run.state,
            "round": run.round,
            "config": run.config.as_payload(),
            "joinToken": run.join_token,
            "joinUrl": self.join_url(run),
            "claimBoundary": CLAIM_BOUNDARY,
            "aggregationMode": run.aggregation_mode,
            "learnerRuntime": run.learner_runtime,
            "participants": [p.public_payload() for p in run.participants.values()],
            "events": [e.as_payload() for e in run.events[-250:]],
            "artifacts": deepcopy(run.artifacts),
            "abortReason": run.abort_reason,
            "failureReason": run.failure_reason,
            "controls": {
                "canStart": run.state == "ready",
                "canAbort": run.state not in TERMINAL_RUN_STATES,
                "pauseSupported": False,
                "canExport": True,
            },
        }

    def join_url(self, run: DemoRun) -> str:
        base = self.public_base_url.rstrip("/")
        return f"{base}/#/join/{run.id}?t={run.join_token}"

    def events(self, run_id: str, *, after: int = -1) -> list[dict[str, Any]]:
        run = self._get_run(run_id)
        return [event.as_payload() for event in run.events if event.seq > after]

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
            if session_id is not None and participant.token == session_id:
                raise FederatedDemoError(
                    "duplicate_join", "this browser session already joined", status=409
                )
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
            token=session_id or _token("ptok-", 18),
            display_name=display_name or None,
            joined_at=_now_ms(),
            last_heartbeat_at=_now_ms(),
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
        self._emit(
            run,
            "participant.heartbeat",
            f"heartbeat from {participant.id}",
            participant_id=participant.id,
            actor="participant",
            payload={"state": participant.state},
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
                f"{participant.id} started browser-local surrogate work",
                participant_id=participant.id,
                actor="participant",
                payload={"runtime": run.learner_runtime},
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
            payload={"progress": round(participant.progress, 4), "round": run.round},
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
            f"{participant.id} submitted browser update metadata",
            participant_id=participant.id,
            actor="participant",
            payload={
                "schema": metadata["schema"],
                "round": metadata["round"],
                "shape": metadata["shape"],
                "hash": metadata["hash"],
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

    def export_evidence(self, run_id: str) -> dict[str, Any]:
        run = self._get_run(run_id)
        return {
            "schema": EVIDENCE_SCHEMA,
            "runId": run.id,
            "mode": "backend-api",
            "createdAt": run.created_at,
            "exportedAt": _now_ms(),
            "state": run.state,
            "round": run.round,
            "config": run.config.as_payload(),
            "claimBoundary": CLAIM_BOUNDARY,
            "aggregationMode": run.aggregation_mode,
            "learnerRuntime": run.learner_runtime,
            "eventTrace": [e.as_payload() for e in run.events],
            "participants": [p.public_payload() for p in run.participants.values()],
            "artifacts": deepcopy(run.artifacts),
            "redaction": {
                "residencySafe": True,
                "rawParticipantDataIncluded": False,
                "modelWeightsIncluded": False,
                "updatePayload": "shape/hash/norm/sample-count metadata only",
            },
        }

    def _get_run(self, run_id: str) -> DemoRun:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise FederatedDemoError(
                "not_found", f"unknown run {run_id}", status=404
            ) from exc

    def _require_join_token(self, run: DemoRun, join_token: str) -> None:
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
                seq=len(run.events),
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
        self._emit(run, "run.state", f"round {round_index} started")
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
                    payload={"round": round_index},
                )

    def _drop_participant(
        self, run: DemoRun, participant: ParticipantState, *, reason: str
    ) -> None:
        participant.state = "dropped"
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
        update_hashes = [p.update_metadata[run.round]["hash"] for p in submitted]
        checkpoint_hash = _hash_json(
            {
                "run": run.id,
                "round": run.round,
                "aggregationMode": run.aggregation_mode,
                "updates": sorted(update_hashes),
            }
        )
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "kind": "checkpoint",
            "label": f"round-{run.round} browser-surrogate checkpoint",
            "round": run.round,
            "sha256": checkpoint_hash,
            "source": "browser-submitted update metadata",
            "aggregationMode": run.aggregation_mode,
            "contributingParticipants": [p.id for p in submitted],
            "simulated": False,
            "containsModelWeights": False,
        }
        run.artifacts.append(checkpoint)
        self._emit(
            run,
            "round.closed",
            f"round {run.round} closed",
            payload={
                "checkpoint": checkpoint_hash,
                "contributing": len(submitted),
                "simulated": False,
            },
        )
        if run.round < run.config.rounds:
            self._assign_round(run, run.round + 1)
            return
        for participant in self._active_participants(run):
            if run.round in participant.submitted_rounds:
                participant.state = "completed"
            elif participant.state in {"assigned", "training"}:
                self._drop_participant(
                    run, participant, reason="round closed before update"
                )
        run.state = "checkpoint_ready"
        self._emit(run, "checkpoint.ready", "final checkpoint-like artifact ready")
        inference_hash = _hash_json(
            {
                "checkpoint": checkpoint_hash,
                "schema": INFERENCE_SCHEMA,
                "preset": run.config.preset,
            }
        )
        run.artifacts.append(
            {
                "schema": INFERENCE_SCHEMA,
                "kind": "inference-model",
                "label": "browser inference attachment from completed demo run",
                "round": run.round,
                "sha256": inference_hash,
                "sourceCheckpoint": checkpoint_hash,
                "modelId": f"lensemble-demo/{run.id}",
                "revision": inference_hash[:12],
                "preset": run.config.preset,
                "simulated": False,
                "containsModelWeights": False,
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
        sample_count = int(artifact.get("sampleCount", 0))
        if sample_count <= 0:
            raise FederatedDemoError("invalid_artifact", "sampleCount must be positive")
        update_hash = _validate_hash(artifact.get("hash"), field_name="hash")
        l2_norm = float(artifact.get("l2Norm", 0.0))
        if l2_norm < 0:
            raise FederatedDemoError("invalid_artifact", "l2Norm must be non-negative")
        source = str(artifact.get("source", ""))
        if source not in {"browser-local-surrogate", "simulator"}:
            raise FederatedDemoError(
                "invalid_artifact",
                "artifact source must identify browser-local-surrogate or simulator",
            )
        if source == "simulator":
            run.aggregation_mode = "backend-simulator"
        else:
            run.aggregation_mode = "browser-surrogate-coordinator"
        return {
            "schema": UPDATE_SCHEMA,
            "source": source,
            "runtime": str(artifact.get("runtime", run.learner_runtime)),
            "runId": run.id,
            "participantId": participant.id,
            "round": run.round,
            "shape": shape,
            "sampleCount": sample_count,
            "hash": update_hash,
            "l2Norm": l2_norm,
            "simulated": bool(artifact.get("simulated", source == "simulator")),
        }
