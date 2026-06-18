#!/usr/bin/env python3
"""CI-safe rehearsal gate for the #338/#359 surprise-meter path.

The real clean round needs the large TwoRooms H5 and onnxruntime. This rehearsal
uses synthetic bias-correctable latent pairs, but still drives the production
composition path: node-trained adapter deltas through FederatedDemoService,
server aggregation, audit, final probe, and offset sidecar export. It then checks
the browser-side surprise self-test and the committed fallback assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from lensemble.demo.system_probe import PARAMS, run_system_composed_probe


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offset-out",
        type=Path,
        default=Path("runs/surprise/rehearsal_offset.json"),
    )
    parser.add_argument(
        "--skip-viewer-assets",
        action="store_true",
        help="only skip static asset checks when developing the gate itself",
    )
    return parser.parse_args()


def _synthetic_manifest() -> dict[str, Any]:
    revision = "a" * 40
    return {
        "schema": "lewm-browser-export/1",
        "graphVersion": 1,
        "checkpoint": {
            "repoId": "synthetic/lewm-tworooms",
            "revision": revision,
            "weightsSha256": hashlib.sha256(b"synthetic-weights").hexdigest(),
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


def _bias_correctable_pairs(*, n: int, dim: int, seed: int) -> dict[str, Any]:
    rng = __import__("random").Random(seed)
    bias = [0.12 * math.sin(k) for k in range(dim)]
    xs: list[float] = []
    targets: list[float] = []
    for _ in range(n):
        for k in range(dim):
            target = rng.uniform(-0.5, 0.5)
            targets.append(round(target, 6))
            xs.append(round(target + bias[k] + rng.uniform(-0.01, 0.01), 6))
    return {"count": n, "x": xs, "target": targets}


def _load_offset(path: Path) -> list[Any]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or len(values) != PARAMS:
        raise SystemExit(f"{path} must contain {PARAMS} offset values")
    if not any(float(value) != 0.0 for value in values):
        raise SystemExit(f"{path} is all-zero")
    return values


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"required surprise-meter asset is missing: {path}")


def _validate_trajectory(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "lewm-surprise-traj/1":
        raise SystemExit(f"{path} has unexpected schema {payload.get('schema')!r}")
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise SystemExit(f"{path} must contain non-empty steps")
    if len(steps) > 600:
        raise SystemExit(f"{path} exceeds the 600-step fallback cap")
    event_rows = [row for row in steps if isinstance(row, dict) and row.get("event")]
    if event_rows:
        raise SystemExit(
            f"{path} must not autoplay perturbation events; found {len(event_rows)} event rows"
        )
    return payload


def _run_node_selftest() -> dict[str, Any]:
    if shutil.which("node") is None:
        raise SystemExit("node is required for the surprise-meter rehearsal")
    result = subprocess.run(
        ["node", "web/surprise-meter/surprise_selftest.mjs"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(result.stdout + result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> None:
    args = _args()
    dim = 192
    manifest = _synthetic_manifest()
    evidence = run_system_composed_probe(
        participants=[
            _bias_correctable_pairs(n=24, dim=dim, seed=11),
            _bias_correctable_pairs(n=24, dim=dim, seed=22),
        ],
        validation=_bias_correctable_pairs(n=16, dim=dim, seed=99),
        checkpoint=manifest["checkpoint"],
        manifest=manifest,
        rounds=3,
        steps_per_round=20,
        batch_size=16,
        seed=7,
        dim=dim,
        deployment_target="surprise-rehearsal",
        offset_out=args.offset_out,
    )
    offset = _load_offset(args.offset_out)
    selftest = _run_node_selftest()

    assets: dict[str, Any] = {}
    if not args.skip_viewer_assets:
        fixture = Path("web/surprise-meter/fixtures/adapter_offset.json")
        trajectory = Path("web/surprise-meter/data/surprise_trajectory.json")
        result_card = Path("web/surprise-meter/data/result_card.json")
        for path in (fixture, trajectory, result_card):
            _require_file(path)
        assets = {
            "fallbackOffsetLength": len(_load_offset(fixture)),
            "trajectorySteps": len(_validate_trajectory(trajectory)["steps"]),
            "resultCardSchema": json.loads(result_card.read_text(encoding="utf-8")).get(
                "schema"
            ),
        }

    report = {
        "ok": bool(
            evidence["passes"]
            and evidence["claimAuditViolations"] == 0
            and len(offset) == PARAMS
            and selftest.get("ok") is True
        ),
        "schema": "surprise-clean-round-rehearsal/1",
        "passes": evidence["passes"],
        "claimAuditViolations": evidence["claimAuditViolations"],
        "serverOffsetParameterCount": evidence["serverOffsetParameterCount"],
        "offsetFile": str(args.offset_out),
        "offsetLength": len(offset),
        "selftest": selftest,
        "viewerAssets": assets,
    }
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit("surprise rehearsal failed")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
