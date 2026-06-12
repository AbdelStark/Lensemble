"""End-to-end rehearsal gate for the Tapestry-like real-LeWM federated demo (#324, gate G7).

Exercises the real-mode service contract without browsers: a two-participant smoke run (auto +
manual mix), a four-participant run with a mid-round dropout and a reconnect, configurable round
counts for long-run checks, stale-revision rejection, adapter-state retrieval after aggregation,
evidence export, and the fail-closed claim audit (lensemble.demo.evidence_audit).

The adapter deltas here are deterministic rehearsal fixtures (marked ``simulated: true``) — they
validate orchestration, validation, aggregation, and evidence plumbing. The REAL-math gates are
separate and stay binding:

- one-browser training reality: ``scripts/lewm_adapter_overfit_check.py``
- federated before/after on real latents: ``scripts/lewm_probe_check.py``
- browser-side JS contracts: ``web/federated-demo/*_selftest.mjs`` (pytest harnesses)

Run: ``uv run python scripts/lewm_demo_rehearsal.py [--rounds 2] [--long-rounds 12]``
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

from lensemble.demo import FederatedDemoService, audit_real_lewm_evidence
from lensemble.demo.federated import (
    LEWM_UPDATE_SCHEMA,
    REAL_LEWM_MODE,
    lewm_adapter_parameter_count,
    lewm_adapter_spec,
)

CHECKPOINT_REVISION = "77adaae0bc31deab21c93740d1f8bb947cd0bdec"
PARAMS = lewm_adapter_parameter_count()


def _rehearsal_manifest() -> dict[str, Any]:
    """A rehearsal export binding; the real server loads the actual export manifest."""
    return {
        "schema": "lewm-browser-export/1",
        "graphVersion": 1,
        "checkpoint": {
            "repoId": "quentinll/lewm-tworooms",
            "revision": CHECKPOINT_REVISION,
            "weightsSha256": hashlib.sha256(b"rehearsal-weights").hexdigest(),
        },
        "files": {
            name: {"sha256": hashlib.sha256(name.encode()).hexdigest()}
            for name in (
                "lewm_tworooms_encoder.onnx",
                "lewm_tworooms_action.onnx",
                "lewm_tworooms_predictor.onnx",
            )
        },
    }


def _load_real_manifest() -> dict[str, Any]:
    from lensemble.demo.server import load_lewm_manifest

    manifest = load_lewm_manifest()
    return manifest if manifest else _rehearsal_manifest()


def _delta_artifact(
    service: FederatedDemoService,
    run_id: str,
    participant: dict[str, Any],
    *,
    seed: int,
    participant_mode: str = "auto",
) -> dict[str, Any]:
    snapshot = service.snapshot(run_id)
    binding = snapshot["lewmBinding"]
    fill = round(1e-4 * (1 + seed % 5), 8)
    delta = [fill] * PARAMS
    l2 = (fill * fill * PARAMS) ** 0.5
    payload = {
        "runId": run_id,
        "participantId": participant["participantId"],
        "round": snapshot["round"],
        "modelRevisionId": snapshot["currentModelRevisionId"],
        "seed": seed,
    }
    return {
        "schema": LEWM_UPDATE_SCHEMA,
        "runId": run_id,
        "participantId": participant["participantId"],
        "round": snapshot["round"],
        "roundId": f"{run_id}:round-{snapshot['round']}",
        "modelRevisionId": snapshot["currentModelRevisionId"],
        "baseCheckpoint": dict(binding["checkpoint"]),
        "exportGraphHashes": dict(binding["exportGraphHashes"]),
        "adapterSpec": lewm_adapter_spec(),
        "dtype": "float32",
        "parameterCount": PARAMS,
        "delta": delta,
        "l2Norm": round(l2, 8),
        "clipNorm": 3.0,
        "hash": hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest(),
        "metrics": {
            "pairCount": 24,
            "optimizerSteps": 20,
            "predLossFirst": round(0.06 - seed * 0.001, 8),
            "predLossLast": round(0.02 - seed * 0.0005, 8),
            "sigregStatistic": 0.02,
            "effectiveRank": 11.0,
            "latentStdMean": 0.85,
            "lossDecreased": True,
            "trainMs": 900.0 + seed,
        },
        "participantMode": participant_mode,
        "seed": seed,
        "simulated": True,  # rehearsal fixture, honestly labeled
    }


def _progress_and_submit(
    service: FederatedDemoService,
    run_id: str,
    participant: dict[str, Any],
    *,
    seed: int,
    participant_mode: str = "auto",
) -> dict[str, Any]:
    service.update_progress(
        run_id,
        participant["participantId"],
        participant_token=participant["participantToken"],
        progress=0.7,
    )
    return service.submit_update(
        run_id,
        participant["participantId"],
        participant_token=participant["participantToken"],
        artifact=_delta_artifact(
            service, run_id, participant, seed=seed, participant_mode=participant_mode
        ),
    )


def _smoke_two_participants(service: FederatedDemoService, rounds: int) -> dict[str, Any]:
    run = service.create_run(
        {"maxParticipants": 2, "quorum": 2, "rounds": rounds, "mode": REAL_LEWM_MODE}
    )
    auto = service.join_run(run["id"], join_token=run["joinToken"], display_name="auto-phone")
    manual = service.join_run(
        run["id"],
        join_token=run["joinToken"],
        display_name="manual-laptop",
        automation_mode="manual",
    )
    service.start_run(run["id"])
    result: dict[str, Any] = {}
    for round_index in range(1, rounds + 1):
        result = _progress_and_submit(service, run["id"], auto, seed=round_index)
        result = _progress_and_submit(
            service, run["id"], manual, seed=round_index + 50, participant_mode="manual"
        )
    snapshot = result["run"]
    assert snapshot["state"] == "completed", snapshot["state"]
    assert len(snapshot["modelRevisions"]) == rounds
    # inference after aggregation: the final global adapter state is retrievable and bound
    final = snapshot["modelRevisions"][-1]
    served = service.model_revision(run["id"], final["modelRevisionId"])
    assert len(served["adapterState"]) == PARAMS
    assert served["baseCheckpoint"]["revision"] == CHECKPOINT_REVISION
    evidence = service.export_evidence(run["id"])
    violations = audit_real_lewm_evidence(evidence)
    assert violations == [], violations
    modes = {p["automationMode"] for p in evidence["participants"]}
    assert modes == {"auto", "manual"}
    return {
        "runId": run["id"],
        "rounds": rounds,
        "state": evidence["state"],
        "finalRevision": final["modelRevisionId"],
        "adapterStateNorm": final["adapterStateNorm"],
        "claimAuditViolations": 0,
        "events": len(evidence["eventTrace"]),
    }


def _four_with_dropout_and_reconnect(service: FederatedDemoService) -> dict[str, Any]:
    run = service.create_run(
        {"maxParticipants": 4, "quorum": 3, "rounds": 2, "mode": REAL_LEWM_MODE}
    )
    participants = [
        service.join_run(
            run["id"],
            join_token=run["joinToken"],
            display_name=f"phone-{index + 1}",
            session_id=f"rehearsal-session-{index + 1}",
        )
        for index in range(4)
    ]
    service.start_run(run["id"])
    # round 1: one participant drops mid-round; quorum (3) is preserved
    service.drop_participant(
        run["id"], participants[3]["participantId"], reason="rehearsal dropout"
    )
    for index, participant in enumerate(participants[:3]):
        _progress_and_submit(service, run["id"], participant, seed=index + 1)
    # a stale-round replay is rejected after the round closes
    stale = _delta_artifact(service, run["id"], participants[0], seed=99)
    stale["round"] = 1
    stale["roundId"] = f"{run['id']}:round-1"
    rejected = False
    try:
        service.submit_update(
            run["id"],
            participants[0]["participantId"],
            participant_token=participants[0]["participantToken"],
            artifact=stale,
        )
    except Exception:
        rejected = True
    assert rejected, "stale-round artifact must be rejected"
    # reconnect: participant 1's socket drops and reattaches mid-run (the browser reconnect path)
    service.connection_closed(
        run["id"],
        role="participant",
        participant_id=participants[0]["participantId"],
        reason="rehearsal socket drop",
    )
    reopened = service.connection_opened(
        run["id"],
        role="participant",
        participant_id=participants[0]["participantId"],
        participant_token=participants[0]["participantToken"],
        after=-1,
    )
    me = next(
        p
        for p in reopened["run"]["participants"]
        if p["id"] == participants[0]["participantId"]
    )
    assert me["connectionState"] == "connected"
    assert me["reconnectCount"] >= 1
    result: dict[str, Any] = {}
    for index, participant in enumerate(participants[:3]):
        result = _progress_and_submit(service, run["id"], participant, seed=index + 20)
    snapshot = result["run"]
    assert snapshot["state"] == "completed"
    evidence = service.export_evidence(run["id"])
    violations = audit_real_lewm_evidence(evidence)
    assert violations == [], violations
    assert any(e["kind"] == "participant.dropped" for e in evidence["eventTrace"])
    kinds = {e["kind"] for e in evidence["eventTrace"]}
    assert "connection.closed" in kinds and "connection.opened" in kinds
    return {
        "runId": run["id"],
        "state": evidence["state"],
        "dropped": [p["id"] for p in evidence["participants"] if p["state"] == "dropped"],
        "reconnected": participants[0]["participantId"],
        "staleRoundRejected": rejected,
        "claimAuditViolations": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=2, help="smoke-gate rounds")
    parser.add_argument(
        "--long-rounds",
        type=int,
        default=0,
        help="optional longer configurable-round gate (0 skips it)",
    )
    args = parser.parse_args()
    manifest = _load_real_manifest()
    service = FederatedDemoService(
        public_base_url="https://rehearsal.example/web/federated-demo",
        deployment_target="rehearsal",
        transport_mode="websocket-primary",
        lewm_export_manifest=manifest,
    )
    report: dict[str, Any] = {
        "ok": True,
        "schema": "lewm-demo-rehearsal/1",
        "manifestSource": "real-export"
        if manifest.get("checkpoint", {}).get("weightsSha256")
        != _rehearsal_manifest()["checkpoint"]["weightsSha256"]
        else "rehearsal-fixture",
        "smokeTwoParticipants": _smoke_two_participants(service, args.rounds),
        "fourWithDropoutAndReconnect": _four_with_dropout_and_reconnect(service),
    }
    if args.long_rounds > 0:
        report["longRun"] = _smoke_two_participants(service, args.long_rounds)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
