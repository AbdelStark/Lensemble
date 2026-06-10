"""Browser federated demo app contracts (#294/#295).

The frontend-only run simulator must keep its simulated-only claim copy and
its JS lifecycle/selftest contract green. The JS-side assertions live in
web/federated-demo/selftest.mjs; this harness runs them under node (skipped
when node is unavailable) so they gate CI alongside the Python suite.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from lensemble.demo import FederatedDemoError, FederatedDemoService
from lensemble.demo.server import make_handler

WEB_DIR = Path("web/federated-demo")


def test_federated_demo_assets_document_simulated_scope() -> None:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    sim = (WEB_DIR / "sim_engine.mjs").read_text(encoding="utf-8")

    assert "Simulated demo" in html
    assert "frontend-only run simulator" in html
    assert "No backend, no real browser training" in html
    assert "does not materially beat local-only" in html
    assert 'SIMULATOR_MODE = "frontend-simulator"' in sim
    assert "simulated: true" in sim


def test_federated_demo_vendored_qr_keeps_license_header() -> None:
    vendor = (WEB_DIR / "vendor" / "qrcode.mjs").read_text(encoding="utf-8")
    assert "qrcode-generator" in vendor
    assert "MIT License" in vendor
    assert "Kazuhiko Arase" in vendor


def test_federated_demo_selftest_passes_under_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")

    result = subprocess.run(
        [node, str(WEB_DIR / "selftest.mjs")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["failed"] == 0
    assert report["passed"] == report["total"]
    assert report["total"] >= 25


def _update_artifact(
    run_id: str, participant_id: str, round_index: int
) -> dict[str, object]:
    return {
        "schema": "browser-update/1",
        "source": "browser-local-surrogate",
        "runtime": "js-worker-surrogate-v1",
        "runId": run_id,
        "participantId": participant_id,
        "round": round_index,
        "shape": [4],
        "sampleCount": 16,
        "hash": "a" * 64,
        "l2Norm": 0.25,
        "simulated": False,
    }


def test_backend_demo_service_closes_browser_submitted_round_and_exports_evidence() -> (
    None
):
    service = FederatedDemoService(
        public_base_url="http://127.0.0.1:8765/web/federated-demo"
    )
    run = service.create_run({"maxParticipants": 2, "quorum": 2, "rounds": 1})
    assert run["state"] == "created"
    assert run["joinUrl"].endswith(f"#/join/{run['id']}?t={run['joinToken']}")

    joined = [
        service.join_run(run["id"], join_token=run["joinToken"], display_name=f"p{i}")
        for i in range(2)
    ]
    snapshot = joined[-1]["run"]
    assert snapshot["state"] == "ready"
    assert len(snapshot["participants"]) == 2

    started = service.start_run(run["id"])
    assert started["state"] == "running_round"
    assert all(p["state"] == "assigned" for p in started["participants"])

    for reply in joined:
        participant_id = reply["participantId"]
        participant_token = reply["participantToken"]
        service.update_progress(
            run["id"],
            participant_id,
            participant_token=participant_token,
            progress=1.0,
        )
        result = service.submit_update(
            run["id"],
            participant_id,
            participant_token=participant_token,
            artifact=_update_artifact(run["id"], participant_id, 1),
        )

    assert result["run"]["state"] == "completed"
    assert result["run"]["aggregationMode"] == "browser-surrogate-coordinator"
    assert any(
        a["kind"] == "checkpoint" and not a["containsModelWeights"]
        for a in result["run"]["artifacts"]
    )
    assert any(a["kind"] == "inference-model" for a in result["run"]["artifacts"])

    evidence = service.export_evidence(run["id"])
    encoded = json.dumps(evidence, sort_keys=True)
    assert evidence["schema"] == "demo-evidence/1"
    assert evidence["redaction"]["rawParticipantDataIncluded"] is False
    assert evidence["redaction"]["modelWeightsIncluded"] is False
    assert "not a benchmark win over local-only" in evidence["claimBoundary"]
    for forbidden in ["observations", "actions", "latents", "weights"]:
        assert forbidden not in encoded


def test_backend_demo_rejects_limits_duplicates_wrong_round_and_raw_payloads() -> None:
    service = FederatedDemoService()
    run = service.create_run({"maxParticipants": 1, "quorum": 1, "rounds": 1})
    first = service.join_run(
        run["id"], join_token=run["joinToken"], session_id="browser-session"
    )

    with pytest.raises(FederatedDemoError, match="already joined"):
        service.join_run(
            run["id"], join_token=run["joinToken"], session_id="browser-session"
        )
    with pytest.raises(FederatedDemoError, match="run is full"):
        service.join_run(
            run["id"], join_token=run["joinToken"], session_id="other-session"
        )
    with pytest.raises(FederatedDemoError, match="join token is invalid"):
        service.join_run(run["id"], join_token="bad-token")

    service.start_run(run["id"])
    service.update_progress(
        run["id"],
        first["participantId"],
        participant_token=first["participantToken"],
        progress=0.5,
    )
    bad = _update_artifact(run["id"], first["participantId"], 2)
    with pytest.raises(FederatedDemoError, match="active round"):
        service.submit_update(
            run["id"],
            first["participantId"],
            participant_token=first["participantToken"],
            artifact=bad,
        )
    raw = _update_artifact(run["id"], first["participantId"], 1)
    raw["observations"] = [[0.1, 0.2]]
    with pytest.raises(FederatedDemoError, match="forbidden raw-data"):
        service.submit_update(
            run["id"],
            first["participantId"],
            participant_token=first["participantToken"],
            artifact=raw,
        )


def test_backend_demo_rejects_duplicate_update_before_round_closes() -> None:
    service = FederatedDemoService()
    run = service.create_run({"maxParticipants": 2, "quorum": 2, "rounds": 1})
    p1 = service.join_run(run["id"], join_token=run["joinToken"])
    service.join_run(run["id"], join_token=run["joinToken"])
    service.start_run(run["id"])
    service.update_progress(
        run["id"],
        p1["participantId"],
        participant_token=p1["participantToken"],
        progress=1.0,
    )
    service.submit_update(
        run["id"],
        p1["participantId"],
        participant_token=p1["participantToken"],
        artifact=_update_artifact(run["id"], p1["participantId"], 1),
    )

    with pytest.raises(FederatedDemoError, match="already submitted"):
        service.submit_update(
            run["id"],
            p1["participantId"],
            participant_token=p1["participantToken"],
            artifact=_update_artifact(run["id"], p1["participantId"], 1),
        )


def test_backend_demo_timeout_abort_and_ndjson_event_stream() -> None:
    service = FederatedDemoService()
    run = service.create_run({"maxParticipants": 2, "quorum": 1, "rounds": 1})
    p1 = service.join_run(run["id"], join_token=run["joinToken"])
    service.join_run(run["id"], join_token=run["joinToken"])
    service.start_run(run["id"])
    service.submit_update(
        run["id"],
        p1["participantId"],
        participant_token=p1["participantToken"],
        artifact=_update_artifact(run["id"], p1["participantId"], 1),
    )
    assert service.snapshot(run["id"])["state"] == "completed"
    assert any(e["kind"] == "participant.dropped" for e in service.events(run["id"]))

    aborted_run = service.create_run({"maxParticipants": 1, "quorum": 1, "rounds": 1})
    service.join_run(aborted_run["id"], join_token=aborted_run["joinToken"])
    aborted = service.abort_run(aborted_run["id"], reason="test abort")
    assert aborted["state"] == "aborted"
    exported = service.export_evidence(aborted_run["id"])
    assert exported["state"] == "aborted"

    failed_run = service.create_run({"maxParticipants": 1, "quorum": 1, "rounds": 1})
    service.join_run(failed_run["id"], join_token=failed_run["joinToken"])
    failed = service.fail_run(failed_run["id"], reason="test failure")
    assert failed["state"] == "failed"
    assert failed["participants"][0]["state"] == "error"


def test_demo_http_api_create_join_events_and_export() -> None:
    service = FederatedDemoService(
        public_base_url="http://127.0.0.1:0/web/federated-demo"
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        created = _post_json(
            f"{base}/api/runs",
            {"maxParticipants": 1, "quorum": 1, "rounds": 1},
        )
        joined = _post_json(
            f"{base}/api/runs/{created['id']}/join",
            {"joinToken": created["joinToken"], "displayName": "browser"},
        )
        _post_json(f"{base}/api/runs/{created['id']}/control", {"action": "start"})
        _post_json(
            f"{base}/api/runs/{created['id']}/participants/{joined['participantId']}/progress",
            {"participantToken": joined["participantToken"], "progress": 1.0},
        )
        completed = _post_json(
            f"{base}/api/runs/{created['id']}/participants/{joined['participantId']}/updates",
            {
                "participantToken": joined["participantToken"],
                "artifact": _update_artifact(created["id"], joined["participantId"], 1),
            },
        )
        assert completed["run"]["state"] == "completed"
        stream = urllib.request.urlopen(
            f"{base}/api/runs/{created['id']}/events?after=0"
        )
        lines = [
            json.loads(line) for line in stream.read().decode("utf-8").splitlines()
        ]
        assert any(line["kind"] == "round.closed" for line in lines)
        evidence = json.loads(
            urllib.request.urlopen(f"{base}/api/runs/{created['id']}/export")
            .read()
            .decode("utf-8")
        )
        assert evidence["schema"] == "demo-evidence/1"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_demo_cli_exposes_one_command_server_help() -> None:
    command = shutil.which("lensemble") or "lensemble"
    result = subprocess.run(
        [command, "demo", "federated", "--help"],
        capture_output=True,
        env={**os.environ, "COLUMNS": "120"},
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--port" in result.stdout
    assert "Serve the browser federated demo app" in result.stdout


def _post_json(url: str, payload: dict[str, object]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:  # pragma: no cover - assertion helper
        raise AssertionError(error.read().decode("utf-8")) from error
