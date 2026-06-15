"""System-composed LeWM federation probe driver (#327, epic #332).

The importable core behind ``scripts/lewm_system_probe.py``. It composes the *shipped* path in one
artifact: real ONNX-trained adapter deltas (``web/federated-demo/lewm_system_round.mjs``) flow
through ``FederatedDemoService.submit_update`` (the real fail-closed validation) and
``_close_round_lewm`` (the real deterministic-mean aggregation + hash-chained revisions); the final
before/after probe scores the SERVER-PRODUCED final ``modelRevisionId`` on held-out pairs.

Factored out of the CLI so the dataset-free unit suite can drive the identical composition on
synthetic 192-dim pairs (``tests/ml/test_lewm_system_probe.py``) and the multi-seed sweep
(``scripts/lewm_probe_seedsweep.py``) can reuse it per seed.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from lensemble.demo import FederatedDemoService, audit_real_lewm_evidence
from lensemble.demo.federated import (
    INITIAL_REVISION_ID,
    LEWM_UPDATE_SCHEMA,
    REAL_LEWM_MODE,
    lewm_adapter_parameter_count,
)

ROUND_TRAINER = "web/federated-demo/lewm_system_round.mjs"
PARAMS = lewm_adapter_parameter_count()
SYSTEM_PROBE_SCHEMA = "lewm-federated-probe/1"


def _node(request: dict[str, Any]) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(request, tmp)
        tmp_path = tmp.name
    result = subprocess.run(
        ["node", ROUND_TRAINER, tmp_path], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"node round trainer failed: {result.stdout}{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def _server_offset(
    service: FederatedDemoService, run_id: str, revision_id: str
) -> list[float] | None:
    """The current GLOBAL offset, read back from the server. None at the initial revision."""
    if revision_id == INITIAL_REVISION_ID:
        return None  # identity start: the server aggregates from all-zeros
    revision = service.model_revision(run_id, revision_id)
    state = revision.get("adapterState")
    return list(state) if state is not None else None


def _delta_artifact(
    *,
    run_id: str,
    participant: dict[str, Any],
    snapshot: dict[str, Any],
    trained: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    binding = snapshot["lewmBinding"]
    delta = [round(float(v), 8) for v in trained["delta"]]
    l2 = round(_vector_norm(delta), 8)
    payload = {
        "runId": run_id,
        "participantId": participant["participantId"],
        "round": snapshot["round"],
        "modelRevisionId": snapshot["currentModelRevisionId"],
        "seed": seed,
        "l2Norm": l2,
    }
    metrics = trained["metrics"]
    return {
        "schema": LEWM_UPDATE_SCHEMA,
        "runId": run_id,
        "participantId": participant["participantId"],
        "round": snapshot["round"],
        "roundId": f"{run_id}:round-{snapshot['round']}",
        "modelRevisionId": snapshot["currentModelRevisionId"],
        "baseCheckpoint": dict(binding["checkpoint"]),
        "exportGraphHashes": dict(binding["exportGraphHashes"]),
        "adapterSpec": binding["adapterSpec"],
        "dtype": "float32",
        "parameterCount": PARAMS,
        "delta": delta,
        "l2Norm": l2,
        "clipNorm": float(trained["clipNorm"]),
        "unclippedNorm": round(float(trained["unclippedNorm"]), 8),
        "metrics": {
            "pairCount": int(metrics["pairCount"]),
            "optimizerSteps": int(metrics["optimizerSteps"]),
            "predLossFirst": round(float(metrics["predLossFirst"]), 8),
            "predLossLast": round(float(metrics["predLossLast"]), 8),
            "sigregStatistic": round(float(metrics["sigregStatistic"]), 8),
            "effectiveRank": round(float(metrics["effectiveRank"]), 8),
            "latentStdMean": round(float(metrics["latentStdMean"]), 8),
            "lossDecreased": bool(metrics["lossDecreased"]),
            "trainMs": round(float(metrics["trainMs"]), 4),
        },
        "hash": hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest(),
        "seed": seed,
        "simulated": False,  # REAL node-trained delta through the real server path
    }


def run_system_composed_probe(
    *,
    participants: list[dict[str, Any]],
    validation: dict[str, Any],
    checkpoint: dict[str, Any],
    manifest: dict[str, Any],
    rounds: int = 3,
    steps_per_round: int = 20,
    batch_size: int = 32,
    seed: int,
    dim: int = 192,
    deployment_target: str = "system-probe",
) -> dict[str, Any]:
    """Compose real deltas through the real server path; return the evidence dict.

    ``participants``/``validation`` are ``{count, x, target}`` pair sets (flat float lists). The
    final probe scores the SERVER-PRODUCED final offset on ``validation``. Raises on claim-audit
    violations or a non-completed run.
    """
    service = FederatedDemoService(
        public_base_url=f"https://{deployment_target}.example/web/federated-demo",
        deployment_target=deployment_target,
        transport_mode="websocket-primary",
        lewm_export_manifest=manifest,
    )
    n_participants = len(participants)
    run = service.create_run(
        {
            "maxParticipants": n_participants,
            "quorum": n_participants,
            "rounds": rounds,
            "mode": REAL_LEWM_MODE,
        }
    )
    run_id = run["id"]
    joined = [
        service.join_run(
            run_id, join_token=run["joinToken"], display_name=f"system-node-{i + 1}"
        )
        for i in range(n_participants)
    ]
    service.start_run(run_id)

    round_telemetry: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        snapshot = service.snapshot(run_id)
        offset = _server_offset(service, run_id, snapshot["currentModelRevisionId"])
        trained = _node(
            {
                "op": "train-round",
                "dim": dim,
                "hiddenDim": 32,
                "initSeed": 42,
                "clipNorm": 3.0,
                "stepsPerRound": steps_per_round,
                "batchSize": batch_size,
                "round": round_index,
                "offset": offset,
                "participants": participants,
            }
        )
        losses = []
        for index, participant in enumerate(joined):
            service.update_progress(
                run_id,
                participant["participantId"],
                participant_token=participant["participantToken"],
                progress=0.7,
            )
            artifact = _delta_artifact(
                run_id=run_id,
                participant=participant,
                snapshot=snapshot,
                trained=trained["deltas"][index],
                seed=round_index * 100 + index,
            )
            service.submit_update(
                run_id,
                participant["participantId"],
                participant_token=participant["participantToken"],
                artifact=artifact,
            )
            losses.append(
                {
                    "first": artifact["metrics"]["predLossFirst"],
                    "last": artifact["metrics"]["predLossLast"],
                }
            )
        round_telemetry.append({"round": round_index, "losses": losses})

    final = service.snapshot(run_id)
    if final["state"] != "completed":
        raise RuntimeError(f"composed run did not complete: {final['state']}")
    final_revision = final["modelRevisions"][-1]
    final_id = final_revision["modelRevisionId"]
    server_offset = service.model_revision(run_id, final_id)["adapterState"]
    if len(server_offset) != PARAMS:
        raise RuntimeError("server offset has the wrong parameter count")

    probed = _node(
        {
            "op": "probe",
            "dim": dim,
            "hiddenDim": 32,
            "initSeed": 42,
            "offset": server_offset,
            "validation": validation,
            "seed": seed,
        }
    )
    report = probed["report"]
    report["rounds"] = rounds
    report["participants"] = n_participants
    report["modelRevisionId"] = final_id
    report["roundTelemetry"] = round_telemetry

    evidence_bundle = service.export_evidence(run_id)
    violations = audit_real_lewm_evidence(evidence_bundle)
    if violations:
        raise RuntimeError(f"claim-audit violations in the composed run: {violations}")

    passes = report["verdict"] == "improved" and not report.get("collapseRisk", False)
    return {
        "schema": SYSTEM_PROBE_SCHEMA,
        "role": "system-composed-headline",
        "seed": seed,
        "protocol": "system-composed: server aggregation path. REAL node-trained adapter deltas "
        "-> FederatedDemoService.submit_update (real fail-closed validation) -> _close_round_lewm "
        "(real deterministic-mean aggregation + hash-chained revisions) -> held-out before/after "
        "probe on the SERVER-PRODUCED final modelRevisionId. No reimplemented coordinator mean.",
        "checkpoint": checkpoint,
        "runId": run_id,
        "modelRevisionId": final_id,
        "modelRevisionChain": [
            rev["modelRevisionId"] for rev in final["modelRevisions"]
        ],
        "aggregateDeltaNorm": final_revision["aggregateDeltaNorm"],
        "adapterStateNorm": final_revision["adapterStateNorm"],
        "serverOffsetParameterCount": len(server_offset),
        "trainPairsPerParticipant": [p["count"] for p in participants],
        "claimAuditViolations": len(violations),
        "result": report,
        "passes": passes,
        "crossCheck": "docs/evidence/lewm_tworooms_probe_check.json "
        "(offline math cross-check; same protocol, reimplemented coordinator mean)",
        "nonClaims": [
            "System-composed before/after validation probe for the Tapestry-like demo's federated "
            "adapter path through the real server aggregation/validation code. Single local "
            "coordinator, mean-of-clipped-deltas (no robust aggregation / DP in this path); not "
            "paper-scale TwoRooms benchmark parity and not production browser training.",
        ],
    }


def write_evidence(out: Path, evidence: dict[str, Any]) -> None:
    out.write_text(json.dumps(evidence, indent=2) + "\n")
