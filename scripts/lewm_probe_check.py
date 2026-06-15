#!/usr/bin/env python3
"""Math-sanity cross-check: federated before/after probe with the shipping JS federation math.

This is the OFFLINE cross-check. It builds disjoint per-participant training pairs and a held-out
validation set through the exported checkpoint graphs (``lensemble.eval.lewm_tworooms_probe_pairs``)
and drives the shipping JS federation shape under node (web/federated-demo/lewm_probe_check.mjs):
shared deterministic adapter init, per-round local training from (init + global offset), clipped
deltas, deterministic mean, offset accumulation, and a final identity-vs-adapted comparison on the
held-out pairs.

It reimplements the coordinator mean inline (in JS), so it proves the *math*. It does NOT exercise
the shipped Python server aggregation/validation path — that is the system-composed gate
``scripts/lewm_system_probe.py`` (#327), whose artifact is the headline. This one is kept as a
labelled math cross-check and writes ``docs/evidence/lewm_tworooms_probe_check.json``.

  uv run --with onnxruntime --with hdf5plugin python scripts/lewm_probe_check.py \
    --h5 ~/.cache/lensemble-lewm/tworoom.h5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from lensemble.eval.lewm_tworooms_probe_pairs import (
    DEFAULT_MODEL_DIR,
    build_probe_split,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--out", type=Path, default=Path("docs/evidence/lewm_tworooms_probe_check.json")
    )
    parser.add_argument("--participants", type=int, default=2)
    # defaults validated by the #322 sweep: enough resident pairs that the adapter learns the
    # systematic predictor bias (which generalizes) instead of memorizing episodes
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

    fixture = {
        "dim": split.dim,
        "adapterHidden": 32,
        "adapterInitSeed": 42,
        "rounds": args.rounds,
        "stepsPerRound": args.steps_per_round,
        "batchSize": args.batch_size,
        "clipNorm": 3.0,
        "participants": split.participants,
        "validation": split.validation,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(fixture, tmp)
        tmp_path = tmp.name
    result = subprocess.run(
        ["node", "web/federated-demo/lewm_probe_check.mjs", tmp_path],
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise SystemExit(f"probe driver failed: {result.stdout}{result.stderr}")
    report = json.loads(result.stdout.strip().splitlines()[-1])
    evidence = {
        "schema": "lewm-federated-probe/1",
        "role": "offline-math-cross-check",
        "seed": args.seed,
        "protocol": "offline math cross-check: disjoint per-participant expert episodes -> "
        "shipping JS federation reimplemented inline (shared init + offset, clipped deltas, "
        "deterministic mean) -> held-out validation pairs; before = identity adapter (frozen "
        "predictor), after = final global revision. The system-composed headline is "
        "docs/evidence/lewm_tworooms_system_probe.json (scripts/lewm_system_probe.py).",
        "checkpoint": split.checkpoint,
        "trainPairsPerParticipant": [p["count"] for p in split.participants],
        "result": report,
        "passes": report["verdict"] == "improved"
        and not report.get("collapseRisk", False),
        "nonClaims": [
            "Offline math cross-check for the Tapestry-like demo's federated adapter path; not "
            "the system-composed artifact, not paper-scale TwoRooms benchmark parity, and not "
            "evidence of production browser training.",
        ],
    }
    args.out.write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "displayVerdict": report.get("displayVerdict", report["verdict"]),
                "collapseRisk": report.get("collapseRisk", False),
                "baselineMse": report["baselineMse"],
                "adaptedMse": report["adaptedMse"],
                "relativeImprovement": report["relativeImprovement"],
                "passes": evidence["passes"],
            },
            indent=2,
        )
    )
    if not evidence["passes"]:
        raise SystemExit(
            "federated probe verdict is not a clean 'improved' (worse/flat or collapse-risk) — "
            "the negative result is recorded in evidence and blocks public positive claims"
        )


if __name__ == "__main__":
    main()
