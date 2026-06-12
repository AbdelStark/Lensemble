#!/usr/bin/env python3
"""Export the pinned TwoRooms LeWM checkpoint to browser ONNX graphs and emit the gate-G2 manifest.

Loads the pinned ``quentinll/lewm-tworooms`` checkpoint (#316), exports the encoder/action/predictor
inference graphs for ONNX Runtime Web, validates PyTorch-vs-onnxruntime parity, and writes:

- ``web/federated-demo/model/lewm-tworooms/*.onnx`` (gitignored binaries the browser loads)
- ``web/federated-demo/model/lewm-tworooms/manifest.json`` (the same manifest, fetched by the app)
- ``docs/evidence/lewm_tworooms_browser_export_manifest.json`` (committed evidence)

Reproduce: ``uv run --with onnx --with onnxscript --with onnxruntime python
scripts/lewm_tworooms_export.py`` (onnx deps are export-time only, never lensemble runtime deps).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lensemble.model.lewm_checkpoint import (
    TWOROOMS_PINNED_REVISION,
    load_tworooms_model,
    resolve_checkpoint,
)
from lensemble.model.lewm_export import (
    browser_export_manifest,
    export_browser_graphs,
    onnxruntime_parity,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, default=None)
    parser.add_argument("--revision", default=TWOROOMS_PINNED_REVISION)
    parser.add_argument("--no-claim-grade", action="store_true")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("web/federated-demo/model/lewm-tworooms")
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("docs/evidence/lewm_tworooms_browser_export_manifest.json"),
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help="Export without onnxruntime parity (the manifest records the skip honestly).",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    resolved = resolve_checkpoint(
        local_dir=args.local_dir,
        revision=args.revision,
        claim_grade=not args.no_claim_grade,
    )
    model, _ = load_tworooms_model(resolved)
    paths = export_browser_graphs(model, args.out_dir, opset=args.opset)
    parity = onnxruntime_parity(
        model, paths, atol=args.atol, require=not args.skip_parity
    )
    import hashlib

    weights_sha = hashlib.sha256(resolved.weights_path.read_bytes()).hexdigest()
    manifest = browser_export_manifest(
        resolved, model, paths, parity, opset=args.opset, weights_sha256=weights_sha
    )
    payload = json.dumps(manifest, indent=2) + "\n"
    (args.out_dir / "manifest.json").write_text(payload)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(payload)
    status = parity.get("status")
    print(
        f"exported {len(paths)} graphs ({manifest['totalBytes'] / 1e6:.1f} MB) from "
        f"{resolved.repo_id}@{resolved.revision[:12]}; parity: {status}\n"
        f"graphs -> {args.out_dir}\nmanifest -> {args.manifest_out}"
    )
    if status == "failed":
        raise SystemExit("export parity FAILED — do not ship these graphs")


if __name__ == "__main__":
    main()
