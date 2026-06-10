"""Stdlib HTTP server for the local browser federated-demo app."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from lensemble.demo.federated import FederatedDemoError, FederatedDemoService

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = REPO_ROOT / "web"
FEDERATED_DEMO_ROOT = WEB_ROOT / "federated-demo"


def make_handler(service: FederatedDemoService) -> type[BaseHTTPRequestHandler]:
    class DemoHandler(BaseHTTPRequestHandler):
        server_version = "LensembleFederatedDemo/1"

        def do_GET(self) -> None:  # noqa: N802
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/api/health":
                    self._json({"ok": True, "service": "lensemble-federated-demo"})
                    return
                if parsed.path.startswith("/api/runs/"):
                    self._handle_api_get(parsed.path, parse_qs(parsed.query))
                    return
                self._static(parsed.path)
            except FederatedDemoError as err:
                self._error(err.status, err.code, str(err))
            except Exception as exc:  # pragma: no cover - defensive local server guard
                self._error(500, "internal_error", str(exc))

        def do_POST(self) -> None:  # noqa: N802
            try:
                parsed = urlparse(self.path)
                payload = self._read_json()
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

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
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
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return DemoHandler


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
) -> ThreadingHTTPServer:
    public_base = f"http://{host}:{port}/web/federated-demo"
    service = FederatedDemoService(public_base_url=public_base)
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    service.public_base_url = f"http://{host}:{httpd.server_port}/web/federated-demo"
    url = f"http://{host}:{httpd.server_port}/web/federated-demo/"
    print(url)
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
        "--open", action="store_true", help="open the browser after starting"
    )
    args = parser.parse_args(argv)
    serve(host=args.host, port=args.port, open_browser=args.open)


if __name__ == "__main__":
    main()
