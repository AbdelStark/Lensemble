#!/usr/bin/env python3
"""Run the #349 clean LeWM adapter-continuation round and export its offset.

This is the one-command wrapper around the system-composed probe. It regenerates
the headline evidence, writes the server-produced final adapter offset to an
ephemeral sidecar, and optionally copies that sidecar into the served
surprise-meter fallback fixture.

Example:
  uv run --with onnxruntime --with hdf5plugin python scripts/surprise/run_clean_round.py \
    --h5 ~/.cache/lensemble-lewm/tworoom.h5
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from lensemble.demo.server import load_lewm_manifest
from lensemble.demo.system_probe import (
    PARAMS,
    run_system_composed_probe,
    write_evidence,
)
from lensemble.eval.lewm_tworooms_probe_pairs import (
    DEFAULT_MODEL_DIR,
    build_probe_split,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/evidence/lewm_tworooms_system_probe.json"),
    )
    parser.add_argument(
        "--offset-out",
        type=Path,
        default=Path("runs/surprise/adapter_offset.json"),
    )
    parser.add_argument(
        "--fallback-out",
        type=Path,
        default=Path("web/surprise-meter/fixtures/adapter_offset.json"),
        help="served fallback copy; pass an empty string to skip",
    )
    parser.add_argument("--participants", type=int, default=2)
    parser.add_argument("--episodes-per-participant", type=int, default=8)
    parser.add_argument("--validation-episodes", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps-per-round", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260612)
    return parser.parse_args()


def _load_offset(path: Path) -> list[Any]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or len(values) != PARAMS:
        raise SystemExit(f"offset sidecar must contain {PARAMS} floats: {path}")
    if not any(float(value) != 0.0 for value in values):
        raise SystemExit("offset sidecar is all-zero; refusing to bless a dead demo")
    return values


def main() -> None:
    args = _args()
    split = build_probe_split(
        h5_path=args.h5,
        model_dir=args.model_dir,
        seed=args.seed,
        participants=args.participants,
        episodes_per_participant=args.episodes_per_participant,
        validation_episodes=args.validation_episodes,
    )
    manifest = load_lewm_manifest(str(args.model_dir / "manifest.json"))
    if not manifest:
        raise SystemExit(f"real export manifest not found under {args.model_dir}")

    evidence = run_system_composed_probe(
        participants=split.participants,
        validation=split.validation,
        checkpoint=split.checkpoint,
        manifest=manifest,
        rounds=args.rounds,
        steps_per_round=args.steps_per_round,
        batch_size=args.batch_size,
        seed=args.seed,
        dim=split.dim,
        offset_out=args.offset_out,
    )
    write_evidence(args.out, evidence)
    _load_offset(args.offset_out)

    fallback_path = args.fallback_out if str(args.fallback_out) else None
    if fallback_path is not None:
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.offset_out, fallback_path)
        _load_offset(fallback_path)

    report = evidence["result"]
    print(
        json.dumps(
            {
                "schema": "surprise-clean-round/1",
                "passes": evidence["passes"],
                "verdict": report["verdict"],
                "displayVerdict": report.get("displayVerdict"),
                "baselineMse": report["baselineMse"],
                "adaptedMse": report["adaptedMse"],
                "relativeImprovement": report["relativeImprovement"],
                "modelRevisionId": evidence["modelRevisionId"],
                "claimAuditViolations": evidence["claimAuditViolations"],
                "offsetFile": str(args.offset_out),
                "fallbackOffsetFile": str(fallback_path) if fallback_path else None,
                "offsetParameterCount": PARAMS,
            },
            indent=2,
        )
    )
    if not evidence["passes"]:
        raise SystemExit(
            "clean surprise round did not produce a claim-safe improvement"
        )


if __name__ == "__main__":
    main()
