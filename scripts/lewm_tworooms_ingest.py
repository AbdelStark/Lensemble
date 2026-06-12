#!/usr/bin/env python3
"""Ingest the pinned TwoRooms LeWorldModel checkpoint and emit the gate-G1 evidence artifacts.

Resolves ``quentinll/lewm-tworooms`` at the pinned revision (docs/roadmap/TAPESTRY_LEWM.md),
reconstructs the LeWM module tree in-tree, strictly loads ``weights.pt``, runs deterministic CPU
reference forwards, and writes:

- ``docs/evidence/lewm_tworooms_checkpoint_manifest.json`` (``lewm-checkpoint-manifest/1``)
- ``docs/evidence/lewm_tworooms_reference_report.json`` (fixed-fixture forward summaries)

Reproduce: ``uv run python scripts/lewm_tworooms_ingest.py`` (downloads via huggingface_hub), or
``--local-dir <snapshot>`` against a pre-downloaded snapshot for air-gapped runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lensemble.model.lewm_checkpoint import (
    TWOROOMS_PINNED_REVISION,
    checkpoint_manifest,
    load_tworooms_model,
    reference_forward_report,
    resolve_checkpoint,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Pre-downloaded checkpoint snapshot dir (skips the HF Hub download).",
    )
    parser.add_argument(
        "--revision",
        default=TWOROOMS_PINNED_REVISION,
        help="HF revision; claim-grade requires the pinned 40-hex commit SHA.",
    )
    parser.add_argument(
        "--no-claim-grade",
        action="store_true",
        help="Allow unpinned revisions (dev only; the manifest records claimGrade=false).",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("docs/evidence/lewm_tworooms_checkpoint_manifest.json"),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("docs/evidence/lewm_tworooms_reference_report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    resolved = resolve_checkpoint(
        local_dir=args.local_dir,
        revision=args.revision,
        claim_grade=not args.no_claim_grade,
    )
    model, _config = load_tworooms_model(resolved)
    manifest = checkpoint_manifest(resolved, model)
    report = reference_forward_report(model)
    report["checkpoint"] = {
        "repoId": resolved.repo_id,
        "revision": resolved.revision,
        "weightsSha256": manifest["files"]["weights.pt"]["sha256"],
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"ingested {resolved.repo_id}@{resolved.revision[:12]}: "
        f"{manifest['tensorCount']} tensors, {manifest['parameterCount']} params\n"
        f"manifest -> {args.manifest_out}\nreference report -> {args.report_out}"
    )


if __name__ == "__main__":
    main()
