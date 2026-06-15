#!/usr/bin/env python3
"""Gate #327: the system-composed end-to-end LeWM federation artifact (epic #332).

The headline +relativeImprovement must be produced by the SYSTEM THE DEMO SHIPS, not by offline
math. This driver composes the full path in one artifact:

    real ONNX-trained adapter deltas (web/federated-demo/lewm_system_round.mjs)
      -> FederatedDemoService.submit_update  (the real fail-closed validation)
      -> _close_round_lewm                   (the real coordinator aggregation + hash-chained revs)
      -> held-out before/after probe on the SERVER-PRODUCED final modelRevisionId

No browsers required: a scripted local-loopback run drives the actual service code. The composition
core lives in ``lensemble.demo.system_probe`` (so the dataset-free unit suite drives the same
path on synthetic pairs); this CLI just builds the real ONNX pairs and writes the headline
evidence ``docs/evidence/lewm_tworooms_system_probe.json``.

  uv run --with onnxruntime --with hdf5plugin python scripts/lewm_system_probe.py \
    --h5 ~/.cache/lensemble-lewm/tworoom.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lensemble.demo.server import load_lewm_manifest
from lensemble.demo.system_probe import run_system_composed_probe, write_evidence
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
    parser.add_argument("--participants", type=int, default=2)
    parser.add_argument("--episodes-per-participant", type=int, default=8)
    parser.add_argument("--validation-episodes", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps-per-round", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260612)
    return parser.parse_args()


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
    )
    write_evidence(args.out, evidence)
    report = evidence["result"]
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "displayVerdict": report.get("displayVerdict"),
                "collapseRisk": report.get("collapseRisk"),
                "baselineMse": report["baselineMse"],
                "adaptedMse": report["adaptedMse"],
                "relativeImprovement": report["relativeImprovement"],
                "modelRevisionId": evidence["modelRevisionId"],
                "claimAuditViolations": evidence["claimAuditViolations"],
                "passes": evidence["passes"],
            },
            indent=2,
        )
    )
    if not evidence["passes"]:
        raise SystemExit(
            "system-composed federated probe is not a clean 'improved' — recorded in evidence "
            "and blocks public positive claims"
        )


if __name__ == "__main__":
    main()
