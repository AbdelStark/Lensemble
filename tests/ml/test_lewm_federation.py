"""Tapestry-like adapter-delta federation contract (#320, epic #314) — gate G6.

Exercises the ``lewm-adapter-delta/1`` schema end-to-end against the demo coordinator: real-mode
run creation requires the export binding; participants submit bounded clipped adapter deltas;
invalid, stale, oversized, raw-data-like, shape-mismatched, checkpoint-mismatched, non-finite,
fabricated-metric, and replayed updates are rejected; aggregation produces deterministic global
adapter revisions bound to the parent checkpoint/export hashes; and the evidence export records
the privacy status honestly.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from lensemble.demo.federated import (
    LEWM_UPDATE_SCHEMA,
    REAL_LEWM_MODE,
    FederatedDemoError,
    FederatedDemoService,
    lewm_adapter_parameter_count,
    lewm_adapter_spec,
)

REVISION = "77adaae0bc31deab21c93740d1f8bb947cd0bdec"
WEIGHTS_SHA = "ab" * 32
GRAPHS = {
    "lewm_tworooms_encoder.onnx": {"sha256": "11" * 32},
    "lewm_tworooms_action.onnx": {"sha256": "22" * 32},
    "lewm_tworooms_predictor.onnx": {"sha256": "33" * 32},
}
PARAMS = lewm_adapter_parameter_count()


def _manifest() -> dict[str, Any]:
    return {
        "schema": "lewm-browser-export/1",
        "graphVersion": 1,
        "checkpoint": {
            "repoId": "quentinll/lewm-tworooms",
            "revision": REVISION,
            "weightsSha256": WEIGHTS_SHA,
        },
        "files": GRAPHS,
    }


def _service() -> FederatedDemoService:
    return FederatedDemoService(lewm_export_manifest=_manifest())


def _real_run(service: FederatedDemoService, participants: int = 2) -> dict[str, Any]:
    run = service.create_run(
        {"maxParticipants": participants, "quorum": participants, "rounds": 2, "mode": REAL_LEWM_MODE}
    )
    joins = []
    for index in range(participants):
        joins.append(
            service.join_run(
                run["id"], join_token=run["joinToken"], display_name=f"p{index}"
            )
        )
    service.start_run(run["id"])
    return {"run": service.snapshot(run["id"]), "joins": joins}


def _delta_artifact(
    run: dict[str, Any],
    join: dict[str, Any],
    *,
    fill: float = 1e-4,
    hash_suffix: str = "00",
    **overrides: Any,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "schema": LEWM_UPDATE_SCHEMA,
        "runId": run["id"],
        "participantId": join["participantId"],
        "round": run["round"],
        "roundId": f"{run['id']}:round-{run['round']}",
        "modelRevisionId": run["currentModelRevisionId"],
        "baseCheckpoint": {"repoId": "quentinll/lewm-tworooms", "revision": REVISION, "weightsSha256": WEIGHTS_SHA},
        "exportGraphHashes": {name: entry["sha256"] for name, entry in GRAPHS.items()},
        "adapterSpec": lewm_adapter_spec(),
        "dtype": "float32",
        "parameterCount": PARAMS,
        "delta": [fill] * PARAMS,
        "l2Norm": (fill * fill * PARAMS) ** 0.5,
        "clipNorm": 3.0,
        "hash": (hash_suffix * 32)[:64],
        "metrics": {
            "pairCount": 24,
            "optimizerSteps": 40,
            "predLossFirst": 0.06,
            "predLossLast": 0.01,
            "sigregStatistic": 0.02,
            "effectiveRank": 11.5,
            "latentStdMean": 0.9,
            "lossDecreased": True,
            "trainMs": 1200.0,
        },
        "participantMode": "auto",
        "seed": 7,
        "simulated": False,
    }
    artifact.update(overrides)
    return artifact


def _submit(service: FederatedDemoService, run: dict[str, Any], join: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    return service.submit_update(
        run["id"],
        join["participantId"],
        participant_token=join["participantToken"],
        artifact=artifact,
    )


# ---------------------------------------------------------------------------
# mode gating
# ---------------------------------------------------------------------------


def test_real_mode_requires_export_manifest() -> None:
    bare = FederatedDemoService()
    with pytest.raises(FederatedDemoError, match="real_mode_unavailable|export"):
        bare.create_run({"maxParticipants": 2, "quorum": 1, "rounds": 1, "mode": REAL_LEWM_MODE})
    # surrogate mode stays available without the manifest
    run = bare.create_run({"maxParticipants": 2, "quorum": 1, "rounds": 1})
    assert run["runMode"] == "surrogate-swipe-dot"


def test_real_run_snapshot_carries_binding_and_claim_boundary() -> None:
    service = _service()
    state = _real_run(service)
    run = state["run"]
    assert run["runMode"] == REAL_LEWM_MODE
    assert run["lewmBinding"]["checkpoint"]["revision"] == REVISION
    assert run["lewmBinding"]["adapterParameterCount"] == PARAMS
    assert run["aggregationMode"] == "lewm-adapter-mean-v1"
    assert "Tapestry-like" in run["claimBoundary"]
    assert "not" in run["claimBoundary"] and "from-scratch" in run["claimBoundary"]


def test_unknown_mode_rejected() -> None:
    with pytest.raises(FederatedDemoError, match="unknown run mode"):
        _service().create_run({"maxParticipants": 2, "quorum": 1, "rounds": 1, "mode": "surrogate-vector-2"})


def test_real_run_rejects_surrogate_schema() -> None:
    service = _service()
    state = _real_run(service)
    run, join = state["run"], state["joins"][0]
    artifact = _delta_artifact(run, join)
    artifact["schema"] = "browser-update/1"
    with pytest.raises(FederatedDemoError, match="accept only"):
        _submit(service, run, join, artifact)


def test_surrogate_run_rejects_adapter_schema() -> None:
    service = _service()
    run = service.create_run({"maxParticipants": 1, "quorum": 1, "rounds": 1})
    join = service.join_run(run["id"], join_token=run["joinToken"])
    service.start_run(run["id"])
    snapshot = service.snapshot(run["id"])
    artifact = _delta_artifact(snapshot, join)
    # the surrogate path rejects it (here at the tiny byte budget, before the schema check)
    with pytest.raises(FederatedDemoError, match="schema|bytes"):
        _submit(service, snapshot, join, artifact)


# ---------------------------------------------------------------------------
# validation fails closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda a: a.__setitem__("delta", a["delta"][:-1]), "exactly"),
        (lambda a: a.__setitem__("parameterCount", PARAMS - 1), "parameterCount"),
        (lambda a: a.__setitem__("dtype", "float64"), "float32"),
        (lambda a: a.__setitem__("adapterSpec", [{"name": "w1", "shape": [1, 1]}]), "adapterSpec"),
        (lambda a: a.__setitem__("modelRevisionId", "lewmrev-deadbeef"), "stale_model_revision|active model revision"),
        (lambda a: a.__setitem__("round", 99), "round"),
        (lambda a: a["baseCheckpoint"].__setitem__("revision", "f" * 40), "pinned checkpoint"),
        (lambda a: a["exportGraphHashes"].__setitem__("lewm_tworooms_encoder.onnx", "ee" * 32), "exportGraphHashes"),
        (lambda a: a.__setitem__("hash", "zz"), "hex"),
        (lambda a: a.__setitem__("rawFrames", [1, 2, 3]), "raw"),
        (lambda a: a.__setitem__("delta", [float("nan")] * PARAMS), "finite|numeric"),
        (lambda a: a.__setitem__("l2Norm", 99.0), "l2Norm"),
        (lambda a: a["metrics"].pop("sigregStatistic"), "fabricated|finite"),
        (lambda a: a["metrics"].__setitem__("predLossLast", "fine"), "finite"),
        (lambda a: a["metrics"].__setitem__("pairCount", 0), "real local work"),
        (lambda a: a.__setitem__("participantMode", "robot"), "participantMode"),
    ],
)
def test_invalid_adapter_deltas_rejected(mutate, match) -> None:
    service = _service()
    state = _real_run(service)
    run, join = state["run"], state["joins"][0]
    artifact = _delta_artifact(run, join)
    mutate(artifact)
    with pytest.raises(FederatedDemoError, match=match):
        _submit(service, run, join, artifact)


def test_norm_bound_and_byte_budget_rejected() -> None:
    service = _service()
    state = _real_run(service)
    run, join = state["run"], state["joins"][0]
    big = _delta_artifact(run, join, fill=0.05)  # norm ≈ 5.6 > 3.0
    big["l2Norm"] = (0.05 * 0.05 * PARAMS) ** 0.5
    with pytest.raises(FederatedDemoError, match="norm bound"):
        _submit(service, run, join, big)

    service.safety.max_lewm_artifact_bytes = 1024
    small = _delta_artifact(run, join)
    with pytest.raises(FederatedDemoError, match="bytes"):
        _submit(service, run, join, small)


def test_replayed_delta_hash_rejected() -> None:
    service = _service()
    state = _real_run(service)
    run = state["run"]
    first, second = state["joins"]
    _submit(service, run, first, _delta_artifact(run, first, hash_suffix="aa"))
    replay = _delta_artifact(run, second, hash_suffix="aa")
    with pytest.raises(FederatedDemoError, match="already submitted"):
        _submit(service, run, second, replay)


# ---------------------------------------------------------------------------
# deterministic aggregation + revision binding
# ---------------------------------------------------------------------------


def test_round_aggregates_deltas_into_bound_adapter_revision() -> None:
    service = _service()
    state = _real_run(service)
    run = state["run"]
    p0, p1 = state["joins"]
    _submit(service, run, p0, _delta_artifact(run, p0, fill=2e-4, hash_suffix="aa"))
    result = _submit(service, run, p1, _delta_artifact(run, p1, fill=4e-4, hash_suffix="bb"))

    snapshot = result["run"]
    assert snapshot["round"] == 2  # round 1 closed, round 2 assigned
    revisions = snapshot["modelRevisions"]
    assert len(revisions) == 1
    revision = revisions[0]
    assert revision["modelRevisionId"].startswith("lewmrev-")
    assert revision["runtime"] == "lewm-adapter-mean-v1"
    assert revision["baseCheckpoint"]["revision"] == REVISION
    assert revision["exportGraphHashes"] == {n: e["sha256"] for n, e in GRAPHS.items()}
    assert revision["adapterSpec"] == lewm_adapter_spec()
    assert len(revision["sourceUpdateHashes"]) == 2
    assert "adapterState" not in revision  # snapshots stay light
    assert revision["privacy"]["secureAggregation"].startswith("absent-in-demo-path")

    # the full state is served by the model-revision endpoint and equals the exact mean
    served = service.model_revision(snapshot["id"], revision["modelRevisionId"])
    assert len(served["adapterState"]) == PARAMS
    assert served["adapterState"][0] == pytest.approx(3e-4, abs=1e-9)
    assert all(v == served["adapterState"][0] for v in served["adapterState"])


def test_adapter_state_accumulates_across_rounds() -> None:
    service = _service()
    state = _real_run(service)
    run = state["run"]
    p0, p1 = state["joins"]
    _submit(service, run, p0, _delta_artifact(run, p0, fill=2e-4, hash_suffix="aa"))
    result = _submit(service, run, p1, _delta_artifact(run, p1, fill=4e-4, hash_suffix="bb"))
    round2 = result["run"]
    _submit(service, round2, p0, _delta_artifact(round2, p0, fill=1e-4, hash_suffix="cc"))
    result = _submit(service, round2, p1, _delta_artifact(round2, p1, fill=1e-4, hash_suffix="dd"))
    final = result["run"]
    assert final["state"] == "completed"
    last = final["modelRevisions"][-1]
    assert last["parentModelRevisionId"] == final["modelRevisions"][0]["modelRevisionId"]
    served = service.model_revision(final["id"], last["modelRevisionId"])
    assert served["adapterState"][0] == pytest.approx(3e-4 + 1e-4, abs=1e-9)
    metrics = final["roundMetrics"]
    assert len(metrics) == 2
    assert metrics[0]["predLossLastMean"] == pytest.approx(0.01)
    assert metrics[0]["sigregStatisticMean"] == pytest.approx(0.02)
    assert metrics[0]["lossDecreasedCount"] == 2


def test_evidence_export_is_honest_and_residency_safe() -> None:
    service = _service()
    state = _real_run(service)
    run = state["run"]
    p0, p1 = state["joins"]
    _submit(service, run, p0, _delta_artifact(run, p0, hash_suffix="aa"))
    _submit(service, run, p1, _delta_artifact(run, p1, hash_suffix="bb"))
    evidence = service.export_evidence(run["id"])
    assert evidence["runMode"] == REAL_LEWM_MODE
    assert evidence["lewmBinding"]["checkpoint"]["revision"] == REVISION
    assert evidence["privacy"]["secureAggregation"].startswith("absent-in-demo-path")
    assert evidence["privacy"]["differentialPrivacy"].startswith("absent-in-demo-path")
    assert "Tapestry-like" in evidence["nonClaimText"]
    assert len(evidence["updateHashes"]) == 2
    # residency: no raw adapter deltas, frames, or tokens anywhere in the export
    encoded = json.dumps(evidence)
    assert '"delta"' not in encoded
    assert "ptok-" not in encoded
    assert '"adapterState"' not in encoded


def test_progress_and_manual_mode_flow_unchanged_in_real_mode() -> None:
    service = _service()
    run = service.create_run(
        {"maxParticipants": 1, "quorum": 1, "rounds": 1, "mode": REAL_LEWM_MODE}
    )
    join = service.join_run(
        run["id"], join_token=run["joinToken"], automation_mode="manual"
    )
    service.start_run(run["id"])
    snapshot = service.snapshot(run["id"])
    service.update_progress(
        run["id"],
        join["participantId"],
        participant_token=join["participantToken"],
        progress=0.4,
    )
    artifact = _delta_artifact(snapshot, join, hash_suffix="ee", participantMode="manual")
    result = _submit(service, snapshot, join, artifact)
    assert result["run"]["state"] == "completed"
    participant = result["run"]["participants"][0]
    assert participant["automationMode"] == "manual"
    meta = participant["updateMetadata"]["1"]
    assert meta["participantMode"] == "manual"
    assert meta["metrics"]["predLossLast"] == pytest.approx(0.01)
    assert "delta" not in meta  # redacted view carries summaries, never the tensor
