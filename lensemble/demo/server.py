"""Stdlib HTTP server for the local browser federated-demo app."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import socket
import struct
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from lensemble.demo.federated import (
    DemoSafetyConfig,
    FederatedDemoError,
    FederatedDemoService,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = REPO_ROOT / "web"
FEDERATED_DEMO_ROOT = WEB_ROOT / "federated-demo"
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _websocket_accept(key: str) -> str:
    digest = hashlib.sha1((key + WS_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = conn.recv(size - len(chunks))
        if not chunk:
            raise EOFError("websocket closed")
        chunks.extend(chunk)
    return bytes(chunks)


def _read_ws_text(conn: socket.socket, *, max_bytes: int) -> str | None:
    header = _recv_exact(conn, 2)
    first, second = header[0], header[1]
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", _recv_exact(conn, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _recv_exact(conn, 8))[0]
    if length > max_bytes:
        raise FederatedDemoError(
            "message_too_large",
            f"WebSocket message exceeds {max_bytes} bytes",
            status=413,
        )
    mask = _recv_exact(conn, 4) if masked else b""
    payload = _recv_exact(conn, length) if length else b""
    if masked:
        payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    if opcode == 8:
        return None
    if opcode == 9:
        return json.dumps({"type": "ping"})
    if opcode != 1:
        raise FederatedDemoError(
            "unsupported_ws_frame", "only text frames are supported"
        )
    return payload.decode("utf-8")


def _send_ws_text(conn: socket.socket, text: str) -> None:
    payload = text.encode("utf-8")
    header = bytearray([0x81])
    if len(payload) < 126:
        header.append(len(payload))
    elif len(payload) <= 0xFFFF:
        header.append(126)
        header.extend(struct.pack("!H", len(payload)))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", len(payload)))
    conn.sendall(bytes(header) + payload)


def _send_ws_json(conn: socket.socket, payload: dict[str, Any]) -> None:
    _send_ws_text(conn, json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _send_ws_close(conn: socket.socket) -> None:
    try:
        conn.sendall(b"\x88\x00")
    except OSError:
        pass


def make_handler(service: FederatedDemoService) -> type[BaseHTTPRequestHandler]:
    rate_limits: dict[tuple[str, ...], int] = {}

    class DemoHandler(BaseHTTPRequestHandler):
        server_version = "LensembleFederatedDemo/1"

        def do_GET(self) -> None:  # noqa: N802
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/api/health":
                    self._json(
                        {
                            "ok": True,
                            "service": "lensemble-federated-demo",
                            "deployment": service.deployment_payload(),
                        }
                    )
                    return
                if (
                    parsed.path.startswith("/api/runs/")
                    and self.headers.get("Upgrade", "").lower() == "websocket"
                ):
                    self._handle_ws(parsed.path, parse_qs(parsed.query))
                    return
                if parsed.path.startswith("/api/runs/"):
                    self._check_rate_limit(parsed.path)
                    self._handle_api_get(parsed.path, parse_qs(parsed.query))
                    return
                self._static(parsed.path)
            except FederatedDemoError as err:
                self._error(err.status, err.code, str(err))
            except Exception as exc:  # pragma: no cover - defensive local server guard
                self._error(500, "internal_error", str(exc))

        def _body_budget(self, path: str) -> int:
            # update submissions in the real mode carry ~150 KB lewm-adapter-delta/1 payloads;
            # every other endpoint keeps the tiny message cap
            if path.rstrip("/").endswith("/updates"):
                return max(
                    service.safety.max_message_bytes,
                    service.safety.max_lewm_artifact_bytes + 8192,
                )
            return service.safety.max_message_bytes

        def do_POST(self) -> None:  # noqa: N802
            try:
                parsed = urlparse(self.path)
                payload = self._read_json(max_bytes=self._body_budget(parsed.path))
                self._check_rate_limit(parsed.path, payload)
                if parsed.path == "/api/runs":
                    self._json(service.create_run(payload))
                    return
                if parsed.path.startswith("/api/runs/"):
                    self._handle_api_post(parsed.path, payload)
                    return
                self._error(404, "not_found", "unknown API route")
            except FederatedDemoError as err:
                self._error(err.status, err.code, str(err))
            except json.JSONDecodeError:
                self._error(400, "invalid_json", "request body must be JSON")
            except Exception as exc:  # pragma: no cover - defensive local server guard
                self._error(500, "internal_error", str(exc))

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._send_common_headers("application/json; charset=utf-8", 0)
            self.end_headers()

        def log_message(self, fmt: str, *args: object) -> None:
            # Keep CLI output focused; the demo is interactive and chatty.
            return

        def _handle_api_get(self, path: str, query: dict[str, list[str]]) -> None:
            parts = [unquote(p) for p in path.split("/") if p]
            if len(parts) == 3:
                self._json(service.snapshot(parts[2]))
                return
            if len(parts) == 4 and parts[3] == "events":
                after = int(query.get("after", ["-1"])[0])
                events = service.events(parts[2], after=after)
                body = "".join(
                    json.dumps(event, sort_keys=True) + "\n" for event in events
                )
                self._bytes(body.encode("utf-8"), "application/x-ndjson; charset=utf-8")
                return
            if len(parts) == 4 and parts[3] == "export":
                self._json(service.export_evidence(parts[2]))
                return
            if len(parts) == 5 and parts[3] == "model-revisions":
                self._json(service.model_revision(parts[2], parts[4]))
                return
            self._error(404, "not_found", "unknown run API route")

        def _handle_api_post(self, path: str, payload: dict[str, Any]) -> None:
            parts = [unquote(p) for p in path.split("/") if p]
            if len(parts) == 4 and parts[3] == "join":
                self._json(
                    service.join_run(
                        parts[2],
                        join_token=str(
                            payload.get("joinToken") or payload.get("token") or ""
                        ),
                        display_name=payload.get("displayName"),
                        session_id=payload.get("sessionId"),
                        automation_mode=payload.get("automationMode"),
                    )
                )
                return
            if len(parts) == 4 and parts[3] == "control":
                action = str(payload.get("action") or "")
                if action == "start":
                    self._json(service.start_run(parts[2]))
                elif action == "abort":
                    self._json(
                        service.abort_run(
                            parts[2], reason=str(payload.get("reason") or "host abort")
                        )
                    )
                elif action == "fail":
                    self._json(
                        service.fail_run(
                            parts[2],
                            reason=str(payload.get("reason") or "demo failure"),
                        )
                    )
                elif action == "timeout-missing":
                    self._json(
                        service.expire_missing(
                            parts[2],
                            reason=str(payload.get("reason") or "participant timeout"),
                        )
                    )
                elif action == "drop":
                    self._json(
                        service.drop_participant(
                            parts[2],
                            str(payload.get("participantId") or ""),
                            reason=str(payload.get("reason") or "host drop"),
                        )
                    )
                else:
                    self._error(
                        400, "invalid_control", f"unknown control action {action!r}"
                    )
                return
            if (
                len(parts) == 6
                and parts[3] == "participants"
                and parts[5] == "heartbeat"
            ):
                self._json(
                    service.heartbeat(
                        parts[2],
                        parts[4],
                        participant_token=str(payload.get("participantToken") or ""),
                    )
                )
                return
            if (
                len(parts) == 6
                and parts[3] == "participants"
                and parts[5] == "progress"
            ):
                self._json(
                    service.update_progress(
                        parts[2],
                        parts[4],
                        participant_token=str(payload.get("participantToken") or ""),
                        progress=float(payload.get("progress", 0.0)),
                    )
                )
                return
            if len(parts) == 6 and parts[3] == "participants" and parts[5] == "updates":
                self._json(
                    service.submit_update(
                        parts[2],
                        parts[4],
                        participant_token=str(payload.get("participantToken") or ""),
                        artifact=dict(payload.get("artifact") or {}),
                    )
                )
                return
            self._error(404, "not_found", "unknown run API route")

        def _handle_ws(self, path: str, query: dict[str, list[str]]) -> None:
            parts = [unquote(p) for p in path.split("/") if p]
            if len(parts) != 4 or parts[3] != "ws":
                self._error(404, "not_found", "unknown WebSocket route")
                return
            run_id = parts[2]
            role = query.get("role", ["host"])[0]
            participant_id = query.get("participantId", [None])[0]
            participant_token = query.get("participantToken", [None])[0]
            selected_protocol = None
            if participant_token is None:
                selected_protocol, participant_token = (
                    self._participant_protocol_and_token()
                )
            after = int(query.get("after", ["-1"])[0])
            key = self.headers.get("Sec-WebSocket-Key")
            if not key:
                self._error(400, "invalid_websocket", "missing Sec-WebSocket-Key")
                return

            self.send_response(101, "Switching Protocols")
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", _websocket_accept(key))
            if selected_protocol is not None:
                self.send_header("Sec-WebSocket-Protocol", selected_protocol)
            self.end_headers()
            self.close_connection = True
            self.connection.settimeout(0.35)

            last_seq = after
            try:
                opened = service.connection_opened(
                    run_id,
                    role=role,
                    participant_id=participant_id,
                    participant_token=participant_token,
                    after=after,
                    transport="websocket",
                )
                events = opened["events"]
                if events:
                    last_seq = max(event["seq"] for event in events)
                _send_ws_json(
                    self.connection,
                    {"type": "snapshot", "run": opened["run"], "events": events},
                )
                # participant sockets may submitUpdate over WS; allow the adapter-delta budget
                ws_max_bytes = (
                    max(
                        service.safety.max_message_bytes,
                        service.safety.max_lewm_artifact_bytes + 8192,
                    )
                    if role == "participant"
                    else service.safety.max_message_bytes
                )
                while True:
                    text: str | None = ""
                    try:
                        text = _read_ws_text(
                            self.connection, max_bytes=ws_max_bytes
                        )
                    except socket.timeout:
                        text = ""
                    if text is None:
                        break
                    if text:
                        self._handle_ws_message(
                            run_id,
                            role,
                            participant_id,
                            participant_token,
                            text,
                        )
                    events = service.events(run_id, after=last_seq)
                    if events:
                        last_seq = max(event["seq"] for event in events)
                        run = service.snapshot(run_id)
                        _send_ws_json(
                            self.connection,
                            {"type": "events", "run": run, "events": events},
                        )
            except (EOFError, OSError):
                pass
            except FederatedDemoError as err:
                _send_ws_json(
                    self.connection,
                    {"type": "error", "code": err.code, "message": str(err)},
                )
            finally:
                try:
                    service.connection_closed(
                        run_id,
                        role=role,
                        participant_id=participant_id,
                        transport="websocket",
                    )
                except FederatedDemoError:
                    pass
                _send_ws_close(self.connection)

        def _handle_ws_message(
            self,
            run_id: str,
            role: str,
            participant_id: str | None,
            participant_token: str | None,
            text: str,
        ) -> None:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise FederatedDemoError(
                    "invalid_json", "WebSocket command must be JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise FederatedDemoError(
                    "invalid_json", "WebSocket command must be a JSON object"
                )
            command = str(payload.get("type") or "")
            if command == "ping":
                _send_ws_json(
                    self.connection, {"type": "pong", "at": int(time.time() * 1000)}
                )
                return
            if role == "host":
                self._handle_host_ws_command(run_id, command, payload)
                return
            if role == "participant":
                if participant_id is None or participant_token is None:
                    raise FederatedDemoError(
                        "invalid_participant_token",
                        "participant command requires participant id and token",
                        status=403,
                    )
                self._handle_participant_ws_command(
                    run_id, participant_id, participant_token, command, payload
                )
                return
            raise FederatedDemoError(
                "invalid_role", f"unknown connection role {role!r}"
            )

        def _handle_host_ws_command(
            self, run_id: str, command: str, payload: dict[str, Any]
        ) -> None:
            if command == "start":
                run = service.start_run(run_id)
            elif command == "abort":
                run = service.abort_run(
                    run_id, reason=str(payload.get("reason") or "host abort")
                )
            elif command == "fail":
                run = service.fail_run(
                    run_id, reason=str(payload.get("reason") or "demo failure")
                )
            elif command == "timeout-missing":
                run = service.expire_missing(
                    run_id, reason=str(payload.get("reason") or "participant timeout")
                )
            elif command == "drop":
                run = service.drop_participant(
                    run_id,
                    str(payload.get("participantId") or ""),
                    reason=str(payload.get("reason") or "host drop"),
                )
            else:
                raise FederatedDemoError(
                    "invalid_control", f"unknown host command {command!r}"
                )
            _send_ws_json(self.connection, {"type": "command.ok", "run": run})

        def _handle_participant_ws_command(
            self,
            run_id: str,
            participant_id: str,
            participant_token: str,
            command: str,
            payload: dict[str, Any],
        ) -> None:
            if command == "heartbeat":
                run = service.heartbeat(
                    run_id, participant_id, participant_token=participant_token
                )["run"]
            elif command == "progress":
                run = service.update_progress(
                    run_id,
                    participant_id,
                    participant_token=participant_token,
                    progress=float(payload.get("progress", 0.0)),
                )["run"]
            elif command == "submitUpdate":
                run = service.submit_update(
                    run_id,
                    participant_id,
                    participant_token=participant_token,
                    artifact=dict(payload.get("artifact") or {}),
                )["run"]
            else:
                raise FederatedDemoError(
                    "invalid_control", f"unknown participant command {command!r}"
                )
            _send_ws_json(self.connection, {"type": "command.ok", "run": run})

        def _check_rate_limit(
            self, path: str, payload: dict[str, Any] | None = None
        ) -> None:
            limit, key = self._rate_limit_bucket(path, payload or {})
            if limit <= 0:
                return
            rate_limits[key] = rate_limits.get(key, 0) + 1
            if rate_limits[key] > limit:
                raise FederatedDemoError(
                    "rate_limited",
                    "too many demo API requests from this client",
                    status=429,
                )

        def _rate_limit_bucket(
            self, path: str, payload: dict[str, Any]
        ) -> tuple[int, tuple[str, ...]]:
            minute = str(int(time.time() // 60))
            participant = self._participant_rate_limit_identity(path, payload)
            if participant is not None:
                run_id, participant_id, token_hash = participant
                return (
                    service.safety.participant_rate_limit_per_minute,
                    ("participant", run_id, participant_id, token_hash, minute),
                )
            return (
                service.safety.rate_limit_per_minute,
                ("client", self.client_address[0], minute),
            )

        @staticmethod
        def _participant_rate_limit_identity(
            path: str, payload: dict[str, Any]
        ) -> tuple[str, str, str] | None:
            parts = [unquote(p) for p in path.split("/") if p]
            if (
                len(parts) == 6
                and parts[0] == "api"
                and parts[1] == "runs"
                and parts[3] == "participants"
                and parts[5] in {"heartbeat", "progress", "updates"}
            ):
                participant_token = str(payload.get("participantToken") or "")
                if participant_token:
                    token_hash = hashlib.sha256(
                        participant_token.encode("utf-8")
                    ).hexdigest()[:16]
                    return parts[2], parts[4], token_hash
            return None

        def _read_json(self, max_bytes: int | None = None) -> dict[str, Any]:
            limit = max_bytes if max_bytes is not None else service.safety.max_message_bytes
            length = int(self.headers.get("Content-Length", "0"))
            if length > limit:
                raise FederatedDemoError(
                    "message_too_large",
                    f"request body exceeds {limit} bytes",
                    status=413,
                )
            if length == 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise FederatedDemoError(
                    "invalid_json", "request body must be a JSON object"
                )
            return value

        def _static(self, path: str) -> None:
            if path in {"", "/"}:
                path = "/web/federated-demo/index.html"
            candidate = (REPO_ROOT / unquote(path).lstrip("/")).resolve()
            if not str(candidate).startswith(str(WEB_ROOT.resolve())):
                self._error(403, "forbidden", "static path is outside web/")
                return
            if candidate.is_dir():
                candidate = candidate / "index.html"
            if not candidate.exists() or not candidate.is_file():
                self._error(404, "not_found", "static file not found")
                return
            content_type = (
                mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
            )
            self._bytes(candidate.read_bytes(), content_type)

        def _json(self, payload: dict[str, Any], status: int = 200) -> None:
            self._bytes(
                json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
                "application/json; charset=utf-8",
                status=status,
            )

        def _error(self, status: int, code: str, message: str) -> None:
            self._json({"ok": False, "code": code, "message": message}, status=status)

        def _bytes(self, body: bytes, content_type: str, *, status: int = 200) -> None:
            self.send_response(status)
            self._send_common_headers(content_type, len(body))
            self.end_headers()
            self.wfile.write(body)

        def _send_common_headers(self, content_type: str, content_length: int) -> None:
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(content_length))
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header(
                "Access-Control-Allow-Headers",
                "Content-Type, Authorization, X-Lensemble-Demo",
            )
            origin = self.headers.get("Origin")
            allowed = self._cors_origin(origin)
            if allowed is not None:
                self.send_header("Access-Control-Allow-Origin", allowed)
                self.send_header("Vary", "Origin")

        def _cors_origin(self, origin: str | None) -> str | None:
            if origin is None:
                return "*"
            if "*" in service.allowed_origins:
                return origin
            if origin.startswith(("http://127.0.0.1", "http://localhost")):
                return origin
            if origin.startswith(
                service.public_base_url.split("/web/federated-demo")[0]
            ):
                return origin
            if "tunnel" in service.allowed_origins and (
                ".trycloudflare.com" in origin or ".ngrok-free.app" in origin
            ):
                return origin
            return None

        def _participant_protocol_and_token(self) -> tuple[str | None, str | None]:
            header = self.headers.get("Sec-WebSocket-Protocol", "")
            for item in header.split(","):
                protocol = item.strip()
                if protocol.startswith("ptok."):
                    return protocol, protocol.removeprefix("ptok.")
            return None, None

    return DemoHandler


def load_lewm_manifest(path: str | None = None) -> dict | None:
    """Load the lewm-browser-export/1 manifest that unlocks the real-lewm-tworooms mode.

    Defaults to the export location written by scripts/lewm_tworooms_export.py. A missing or
    unreadable manifest returns None: the server starts with the real mode unavailable (creating
    a real-mode run then fails closed with `real_mode_unavailable`).
    """
    candidate = Path(path) if path else Path("web/federated-demo/model/lewm-tworooms/manifest.json")
    try:
        manifest = json.loads(candidate.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return manifest if isinstance(manifest, dict) else None


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
    public_base_url: str | None = None,
    public_demo: bool = False,
    deployment_target: str = "local",
    rate_limit_per_minute: int = 0,
    participant_rate_limit_per_minute: int = 0,
    lewm_manifest_path: str | None = None,
) -> ThreadingHTTPServer:
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    public_base = public_base_url or f"http://{display_host}:{port}/web/federated-demo"
    lewm_manifest = load_lewm_manifest(lewm_manifest_path)
    service = FederatedDemoService(
        public_base_url=public_base,
        public_demo=public_demo,
        deployment_target=deployment_target,
        transport_mode="websocket-primary",
        safety=DemoSafetyConfig(
            rate_limit_per_minute=rate_limit_per_minute,
            participant_rate_limit_per_minute=participant_rate_limit_per_minute,
        ),
        lewm_export_manifest=lewm_manifest,
    )
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    if public_base_url is None:
        service.public_base_url = (
            f"http://{display_host}:{httpd.server_port}/web/federated-demo"
        )
    url = f"http://{display_host}:{httpd.server_port}/web/federated-demo/"
    print(f"host_url={url}")
    print(f"public_base_url={service.public_base_url}")
    print(
        f"participant_join_root={service.public_base_url.rstrip('/')}/#/join/<run_id>"
    )
    print("transport_mode=websocket-primary fallback=http-polling")
    if lewm_manifest is not None:
        revision = str(lewm_manifest.get("checkpoint", {}).get("revision", "?"))[:12]
        print(f"real_lewm_mode=available checkpoint_revision={revision}")
    else:
        print(
            "real_lewm_mode=unavailable (run scripts/lewm_tworooms_export.py to generate "
            "web/federated-demo/model/lewm-tworooms/)"
        )
    print(
        f"deployment_target={deployment_target} public_demo={str(public_demo).lower()}"
    )
    print(
        "safety="
        + json.dumps(service.safety.as_payload(), sort_keys=True, separators=(",", ":"))
    )
    if open_browser:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return httpd


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Serve the Lensemble browser federated demo"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--public-base-url",
        default=None,
        help="external HTTPS origin/path used in QR joins and WSS URLs",
    )
    parser.add_argument(
        "--public-demo",
        action="store_true",
        help="enforce short-event public-demo safety limits",
    )
    parser.add_argument(
        "--deployment-target",
        default="local",
        choices=("local", "lan", "cloudflare-tunnel", "public"),
        help="label printed in startup output and evidence exports",
    )
    parser.add_argument(
        "--open", action="store_true", help="open the browser after starting"
    )
    parser.add_argument(
        "--rate-limit-per-minute",
        type=int,
        default=0,
        help="shared client API request limit per minute; 0 disables rate limiting",
    )
    parser.add_argument(
        "--participant-rate-limit-per-minute",
        type=int,
        default=0,
        help="participant heartbeat/progress/update limit per minute; 0 disables rate limiting",
    )
    parser.add_argument(
        "--lewm-manifest",
        default=None,
        help="path to the lewm-browser-export/1 manifest unlocking real-lewm-tworooms "
        "(default: web/federated-demo/model/lewm-tworooms/manifest.json when present)",
    )
    args = parser.parse_args(argv)
    if args.rate_limit_per_minute < 0 or args.participant_rate_limit_per_minute < 0:
        parser.error("rate limits must be >= 0")
    serve(
        host=args.host,
        port=args.port,
        open_browser=args.open,
        public_base_url=args.public_base_url,
        public_demo=args.public_demo,
        deployment_target=args.deployment_target,
        rate_limit_per_minute=args.rate_limit_per_minute,
        participant_rate_limit_per_minute=args.participant_rate_limit_per_minute,
        lewm_manifest_path=args.lewm_manifest,
    )


if __name__ == "__main__":
    main()
