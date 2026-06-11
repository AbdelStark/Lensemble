"""Deterministic rehearsal gate for the #303 hackathon browser demo epic.

The script exercises the same service contract as the browser/HTTP path without
requiring four physical phones in CI: host creates a run, four participants join,
the run starts at quorum, participants submit bounded tiny update artifacts, the
coordinator publishes a model revision, inference metadata becomes available,
and evidence export records the non-claim boundary. A second run drops one
participant while preserving quorum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

from lensemble.demo import FederatedDemoService


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact(
    *,
    run_id: str,
    participant_id: str,
    round_index: int,
    model_revision_id: str,
    seed: int,
) -> dict[str, Any]:
    vector = [
        round(0.04 + seed * 0.003, 8),
        round(0.02 + seed * 0.002, 8),
        round(0.01 * ((seed % 3) - 1), 8),
        round(-0.015 + seed * 0.001, 8),
    ]
    l2_norm = sum(value * value for value in vector) ** 0.5
    hash_payload = {
        "runId": run_id,
        "participantId": participant_id,
        "round": round_index,
        "modelRevisionId": model_revision_id,
        "vector": vector,
        "seed": seed,
    }
    return {
        "schema": "browser-update/1",
        "source": "browser-local-surrogate",
        "runtime": "js-worker-tiny-jepa-v1",
        "runId": run_id,
        "participantId": participant_id,
        "round": round_index,
        "roundId": f"{run_id}:round-{round_index}",
        "modelRevisionId": model_revision_id,
        "shape": [len(vector)],
        "parameterCount": len(vector),
        "vector": vector,
        "sampleCount": 24,
        "localSteps": 8,
        "hash": _hash_payload(hash_payload),
        "l2Norm": round(l2_norm, 8),
        "clipNorm": 1.0,
        "loss": round(0.2 - seed * 0.005, 8),
        "probe": round(0.8 + seed * 0.005, 8),
        "runtimeMs": 10 + seed,
        "seed": seed,
        "simulated": False,
    }


def _join_four(
    service: FederatedDemoService, run_id: str, join_token: str
) -> list[dict[str, Any]]:
    return [
        service.join_run(
            run_id,
            join_token=join_token,
            display_name=f"phone-{index + 1}",
            session_id=f"rehearsal-session-{index + 1}",
        )
        for index in range(4)
    ]


def _submit(
    service: FederatedDemoService,
    run_id: str,
    participant: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    snapshot = service.snapshot(run_id)
    service.update_progress(
        run_id,
        participant["participantId"],
        participant_token=participant["participantToken"],
        progress=1.0,
    )
    return service.submit_update(
        run_id,
        participant["participantId"],
        participant_token=participant["participantToken"],
        artifact=_artifact(
            run_id=run_id,
            participant_id=participant["participantId"],
            round_index=snapshot["round"],
            model_revision_id=snapshot["currentModelRevisionId"],
            seed=seed,
        ),
    )


def _happy_path(service: FederatedDemoService) -> dict[str, Any]:
    run = service.create_run({"maxParticipants": 4, "quorum": 4, "rounds": 1})
    participants = _join_four(service, run["id"], run["joinToken"])
    service.start_run(run["id"])
    result: dict[str, Any] = {}
    for index, participant in enumerate(participants):
        result = _submit(service, run["id"], participant, seed=index + 1)
    evidence = service.export_evidence(run["id"])
    assert result["run"]["state"] == "completed"
    assert evidence["modelRevisionRefs"]
    assert any(
        artifact["kind"] == "inference-model" for artifact in evidence["artifacts"]
    )
    return {
        "runId": run["id"],
        "state": evidence["state"],
        "revision": evidence["modelRevisionRefs"][-1]["modelRevisionId"],
        "participants": len(evidence["participants"]),
        "events": len(evidence["eventTrace"]),
    }


def _dropout_path(service: FederatedDemoService) -> dict[str, Any]:
    run = service.create_run({"maxParticipants": 4, "quorum": 3, "rounds": 1})
    participants = _join_four(service, run["id"], run["joinToken"])
    service.start_run(run["id"])
    service.drop_participant(
        run["id"], participants[-1]["participantId"], reason="rehearsal dropout"
    )
    result: dict[str, Any] = {}
    for index, participant in enumerate(participants[:3]):
        result = _submit(service, run["id"], participant, seed=index + 10)
    evidence = service.export_evidence(run["id"])
    assert result["run"]["state"] == "completed"
    assert any(
        event["kind"] == "participant.dropped" for event in evidence["eventTrace"]
    )
    return {
        "runId": run["id"],
        "state": evidence["state"],
        "revision": evidence["modelRevisionRefs"][-1]["modelRevisionId"],
        "participants": len(evidence["participants"]),
        "dropped": [
            participant["id"]
            for participant in evidence["participants"]
            if participant["state"] == "dropped"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the #303 demo rehearsal gate")
    parser.add_argument(
        "--public-base-url",
        default="https://demo.example/web/federated-demo",
        help="base URL recorded in generated join/WSS URLs",
    )
    args = parser.parse_args()
    service = FederatedDemoService(
        public_base_url=args.public_base_url,
        public_demo=True,
        deployment_target="rehearsal",
        transport_mode="websocket-primary",
    )
    report = {
        "ok": True,
        "schema": "hackathon-demo-rehearsal/1",
        "happyPath": _happy_path(service),
        "dropoutPath": _dropout_path(service),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
