"""Browser federated demo app contracts (#294/#295).

The frontend-only run simulator must keep its simulated-only claim copy and
its JS lifecycle/selftest contract green. The JS-side assertions live in
web/federated-demo/selftest.mjs; this harness runs them under node (skipped
when node is unavailable) so they gate CI alongside the Python suite.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import shutil
import socket
import struct
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from typer.models import OptionInfo

from lensemble.cli import demo_federated
from lensemble.demo import FederatedDemoError, FederatedDemoService
from lensemble.demo.federated import DemoSafetyConfig
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
    run_id: str,
    participant_id: str,
    round_index: int,
    *,
    model_revision_id: str = "initial",
    vector: list[float] | None = None,
) -> dict[str, object]:
    update_vector = vector or [0.1, 0.1, 0.0, 0.0]
    l2_norm = sum(value * value for value in update_vector) ** 0.5
    energy = [value * value for value in update_vector]
    effective_dim = (
        (sum(energy) * sum(energy)) / sum(value * value for value in energy)
        if sum(energy)
        else 0.0
    )
    effective_dim_ratio = effective_dim / len(update_vector)
    return {
        "schema": "browser-update/1",
        "source": "browser-local-surrogate",
        "runtime": "js-worker-tiny-jepa-v1",
        "runId": run_id,
        "participantId": participant_id,
        "round": round_index,
        "roundId": f"{run_id}:round-{round_index}",
        "modelRevisionId": model_revision_id,
        "shape": [len(update_vector)],
        "parameterCount": len(update_vector),
        "vector": update_vector,
        "sampleCount": 16,
        "localSteps": 8,
        "hash": "a" * 64,
        "l2Norm": round(l2_norm, 8),
        "clipNorm": 1.0,
        "unclippedNorm": round(l2_norm, 8),
        "clipSaturation": 0.0,
        "loss": 0.2,
        "probe": 0.8,
        "effectiveDim": round(effective_dim, 8),
        "effectiveDimRatio": round(effective_dim_ratio, 8),
        "collapseRisk": "watch" if effective_dim_ratio < 0.6 else "low",
        "runtimeMs": 12.5,
        "seed": 7,
        "simulated": False,
    }


def test_backend_demo_service_closes_browser_submitted_round_and_exports_evidence() -> (
    None
):
    service = FederatedDemoService(
        public_base_url="http://127.0.0.1:8765/web/federated-demo"
    )
    default_run = service.create_run({"maxParticipants": 1, "quorum": 1})
    assert default_run["config"]["rounds"] == 1000
    with pytest.raises(FederatedDemoError, match=r"rounds must be in \[1, 1000\]"):
        service.create_run({"maxParticipants": 1, "quorum": 1, "rounds": 1001})

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
    assert result["run"]["aggregationMode"] == "tiny-vector-mean"
    assert any(
        a["kind"] == "checkpoint" and not a["containsModelWeights"]
        for a in result["run"]["artifacts"]
    )
    assert any(a["kind"] == "inference-model" for a in result["run"]["artifacts"])
    assert result["run"]["modelRevisions"][0]["kind"] == "model-revision"
    assert result["run"]["modelRevisions"][0]["vector"] == [0.1, 0.1, 0.0, 0.0]
    assert result["run"]["roundMetrics"][0]["aggregateNorm"] == pytest.approx(
        0.14142136
    )
    assert result["run"]["roundMetrics"][0]["localLossMean"] == pytest.approx(0.2)
    assert result["run"]["roundMetrics"][0]["probeMean"] == pytest.approx(0.8)
    assert result["run"]["roundMetrics"][0]["aggregateEffectiveDim"] == pytest.approx(
        2.0
    )
    assert result["run"]["roundMetrics"][0]["collapseRisk"] == "watch"
    before_terminal_heartbeat = len(service.events(run["id"]))
    heartbeat = service.heartbeat(
        run["id"],
        joined[0]["participantId"],
        participant_token=joined[0]["participantToken"],
    )
    assert heartbeat["run"]["participants"][0]["connectionState"] == "completed"
    assert len(service.events(run["id"])) == before_terminal_heartbeat

    evidence = service.export_evidence(run["id"])
    encoded = json.dumps(evidence, sort_keys=True)
    assert evidence["schema"] == "demo-evidence/1"
    assert evidence["redaction"]["rawParticipantDataIncluded"] is False
    assert evidence["redaction"]["modelWeightsIncluded"] is False
    assert evidence["redaction"]["participantTokensIncluded"] is False
    assert evidence["modelRevisionRefs"][0]["modelRevisionId"].startswith("rev-")
    assert "not a benchmark win over local-only" in evidence["claimBoundary"]
    for forbidden in ["observations", "actions", "latents", "weights"]:
        assert forbidden not in encoded


def test_backend_demo_rejects_limits_duplicates_wrong_round_and_raw_payloads() -> None:
    service = FederatedDemoService()
    run = service.create_run({"maxParticipants": 1, "quorum": 1, "rounds": 1})
    first = service.join_run(
        run["id"],
        join_token=run["joinToken"],
        session_id="browser-session",
        automation_mode="manual",
    )
    assert first["run"]["participants"][0]["automationMode"] == "manual"

    resumed = service.join_run(
        run["id"], join_token=run["joinToken"], session_id="browser-session"
    )
    assert resumed["participantId"] == first["participantId"]
    assert resumed["run"]["participants"][0]["automationMode"] == "manual"
    assert len(resumed["run"]["participants"]) == 1
    switched = service.join_run(
        run["id"],
        join_token=run["joinToken"],
        session_id="browser-session",
        automation_mode="auto",
    )
    assert switched["run"]["participants"][0]["automationMode"] == "auto"
    with pytest.raises(FederatedDemoError, match="automationMode"):
        service.join_run(
            run["id"],
            join_token=run["joinToken"],
            session_id="browser-session",
            automation_mode="invalid",
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
    pending = service.submit_update(
        run["id"],
        p1["participantId"],
        participant_token=p1["participantToken"],
        artifact=_update_artifact(run["id"], p1["participantId"], 1),
    )
    assert pending["run"]["state"] == "running_round"
    service.expire_missing(run["id"], reason="test timeout")
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


def test_backend_demo_public_urls_safety_and_liveness() -> None:
    service = FederatedDemoService(
        public_base_url="https://demo.example/web/federated-demo",
        public_demo=True,
        deployment_target="public",
        transport_mode="websocket-primary",
        safety=DemoSafetyConfig(
            max_public_participants=4,
            max_public_rounds=2,
            heartbeat_stale_ms=10,
            participant_timeout_ms=30,
        ),
    )
    run = service.create_run({"maxParticipants": 4, "quorum": 2, "rounds": 1})
    assert run["joinUrl"].startswith("https://demo.example/web/federated-demo/")
    assert run["webSocketUrl"].startswith("wss://demo.example/api/runs/")
    assert run["deployment"]["publicDemo"] is True

    with pytest.raises(FederatedDemoError, match="maxParticipants"):
        service.create_run({"maxParticipants": 5, "quorum": 2, "rounds": 1})

    joined = service.join_run(run["id"], join_token=run["joinToken"])
    participant = service._runs[run["id"]].participants[joined["participantId"]]
    assert participant.last_heartbeat_at is not None
    service.refresh_liveness(run["id"], now_ms=participant.last_heartbeat_at + 11)
    stale = service.snapshot(run["id"])["participants"][0]
    assert stale["connectionState"] == "stale"
    service.refresh_liveness(run["id"], now_ms=participant.last_heartbeat_at + 31)
    dropped = service.snapshot(run["id"])["participants"][0]
    assert dropped["connectionState"] == "dropped"


def test_backend_demo_rejects_bad_norm_stale_revision_and_fetches_model_revision() -> (
    None
):
    service = FederatedDemoService()
    run = service.create_run({"maxParticipants": 1, "quorum": 1, "rounds": 1})
    joined = service.join_run(run["id"], join_token=run["joinToken"])
    service.start_run(run["id"])

    stale = _update_artifact(
        run["id"],
        joined["participantId"],
        1,
        model_revision_id="rev-stale",
    )
    with pytest.raises(FederatedDemoError, match="modelRevisionId"):
        service.submit_update(
            run["id"],
            joined["participantId"],
            participant_token=joined["participantToken"],
            artifact=stale,
        )

    bad_norm = _update_artifact(run["id"], joined["participantId"], 1)
    bad_norm["l2Norm"] = 0.99
    with pytest.raises(FederatedDemoError, match="l2Norm"):
        service.submit_update(
            run["id"],
            joined["participantId"],
            participant_token=joined["participantToken"],
            artifact=bad_norm,
        )

    completed = service.submit_update(
        run["id"],
        joined["participantId"],
        participant_token=joined["participantToken"],
        artifact=_update_artifact(run["id"], joined["participantId"], 1),
    )
    revision = completed["run"]["modelRevisions"][0]
    fetched = service.model_revision(run["id"], revision["modelRevisionId"])
    assert fetched["sha256"] == revision["sha256"]
    assert fetched["vector"] == [0.1, 0.1, 0.0, 0.0]


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


def test_demo_http_api_participant_protocol_uses_participant_rate_bucket() -> None:
    service = FederatedDemoService(
        public_base_url="http://127.0.0.1:0/web/federated-demo",
        safety=DemoSafetyConfig(
            rate_limit_per_minute=3,
            participant_rate_limit_per_minute=20,
        ),
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

        with pytest.raises(urllib.error.HTTPError) as host_limit:
            urllib.request.urlopen(f"{base}/api/runs/{created['id']}")
        assert host_limit.value.code == 429

        for _ in range(5):
            progress = _post_json(
                f"{base}/api/runs/{created['id']}/participants/{joined['participantId']}/progress",
                {"participantToken": joined["participantToken"], "progress": 0.5},
            )
            assert progress["run"]["state"] == "running_round"

        completed = _post_json(
            f"{base}/api/runs/{created['id']}/participants/{joined['participantId']}/updates",
            {
                "participantToken": joined["participantToken"],
                "artifact": _update_artifact(created["id"], joined["participantId"], 1),
            },
        )
        assert completed["run"]["state"] == "completed"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_demo_http_api_websocket_replay_and_commands() -> None:
    service = FederatedDemoService(
        public_base_url="http://127.0.0.1:0/web/federated-demo"
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    host_socket: socket.socket | None = None
    participant_socket: socket.socket | None = None
    try:
        created = _post_json(
            f"{base}/api/runs",
            {"maxParticipants": 1, "quorum": 1, "rounds": 1},
        )
        host_socket = _ws_connect(
            "127.0.0.1",
            server.server_port,
            f"/api/runs/{created['id']}/ws?role=host&after=-1",
        )
        snapshot = _ws_read_json(host_socket)
        assert snapshot["type"] == "snapshot"
        assert snapshot["run"]["id"] == created["id"]
        assert any(event["kind"] == "connection.opened" for event in snapshot["events"])

        joined = _post_json(
            f"{base}/api/runs/{created['id']}/join",
            {"joinToken": created["joinToken"], "displayName": "phone"},
        )
        joined_events = _ws_read_until_kind(host_socket, "participant.joined")
        assert joined_events["type"] == "events"
        assert joined_events["run"]["participants"][0]["displayName"] == "phone"

        participant_socket = _ws_connect(
            "127.0.0.1",
            server.server_port,
            f"/api/runs/{created['id']}/ws?role=participant&participantId={joined['participantId']}&after=-1",
            protocols=[f"ptok.{joined['participantToken']}"],
        )
        participant_snapshot = _ws_read_json(participant_socket)
        assert participant_snapshot["type"] == "snapshot"
        assert (
            participant_snapshot["run"]["participants"][0]["connectionState"]
            == "connected"
        )

        _ws_send_json(host_socket, {"type": "start"})
        started = _ws_read_until_state(host_socket, "running_round")
        assert started["run"]["state"] == "running_round"
    finally:
        for ws in [host_socket, participant_socket]:
            if ws is not None:
                ws.close()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_hackathon_demo_rehearsal_script_passes() -> None:
    result = subprocess.run(
        ["uv", "run", "python", "scripts/hackathon_demo_rehearsal.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["happyPath"]["state"] == "completed"
    assert report["dropoutPath"]["state"] == "completed"
    assert report["dropoutPath"]["dropped"]


def test_demo_cli_exposes_one_command_server_help() -> None:
    command = shutil.which("lensemble") or "lensemble"
    result = subprocess.run(
        [command, "demo", "federated", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    signature = inspect.signature(demo_federated)
    port_option = signature.parameters["port"].default
    assert isinstance(port_option, OptionInfo)
    assert port_option.param_decls == ("--port",)
    assert port_option.default == 8765
    assert port_option.help == "local port for the demo HTTP server"
    public_base_option = signature.parameters["public_base_url"].default
    assert isinstance(public_base_option, OptionInfo)
    assert public_base_option.param_decls == ("--public-base-url",)
    public_demo_option = signature.parameters["public_demo"].default
    assert isinstance(public_demo_option, OptionInfo)
    assert public_demo_option.param_decls == ("--public-demo",)
    assert "Serve the browser federated demo app" in (
        inspect.getdoc(demo_federated) or ""
    )


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


def _ws_connect(
    host: str, port: int, path: str, *, protocols: list[str] | None = None
) -> socket.socket:
    sock = socket.create_connection((host, port), timeout=5)
    key = base64.b64encode(hashlib.sha1(path.encode("utf-8")).digest()[:16]).decode(
        "ascii"
    )
    lines = [
        f"GET {path} HTTP/1.1",
        f"Host: {host}:{port}",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Key: {key}",
        "Sec-WebSocket-Version: 13",
    ]
    if protocols:
        lines.append(f"Sec-WebSocket-Protocol: {', '.join(protocols)}")
    request = "\r\n".join(lines) + "\r\n\r\n"
    sock.sendall(request.encode("ascii"))
    response = b""
    while b"\r\n\r\n" not in response:
        response += sock.recv(1)
    assert b" 101 " in response, response.decode("utf-8", errors="replace")
    if protocols:
        expected = f"Sec-WebSocket-Protocol: {protocols[0]}".encode("ascii")
        assert expected.lower() in response.lower(), response.decode(
            "utf-8", errors="replace"
        )
    sock.settimeout(5)
    return sock


def _ws_read_json(sock: socket.socket) -> dict[str, Any]:
    header = _recv_exact(sock, 2)
    first, second = header[0], header[1]
    opcode = first & 0x0F
    assert opcode == 1, f"unexpected websocket opcode {opcode}"
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", _recv_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    payload = _recv_exact(sock, length)
    return json.loads(payload.decode("utf-8"))


def _ws_send_json(sock: socket.socket, payload: dict[str, object]) -> None:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    mask = b"\x11\x22\x33\x44"
    header = bytearray([0x81])
    if len(raw) < 126:
        header.append(0x80 | len(raw))
    elif len(raw) <= 0xFFFF:
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", len(raw)))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack("!Q", len(raw)))
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(raw))
    sock.sendall(bytes(header) + mask + masked)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise AssertionError("websocket closed before frame completed")
        chunks.extend(chunk)
    return bytes(chunks)


def _ws_read_until_kind(sock: socket.socket, kind: str) -> dict[str, Any]:
    for _ in range(8):
        message = _ws_read_json(sock)
        if any(event.get("kind") == kind for event in message.get("events", [])):
            return message
    raise AssertionError(f"WebSocket stream did not emit {kind}")


def _ws_read_until_state(sock: socket.socket, state: str) -> dict[str, Any]:
    for _ in range(8):
        message = _ws_read_json(sock)
        run = message.get("run")
        if run and run.get("state") == state:
            return message
        for event in message.get("events", []):
            if event.get("runState") == state:
                return {"run": {"state": state}, "events": message.get("events", [])}
    raise AssertionError(f"WebSocket stream did not reach {state}")
