#!/usr/bin/env python3
"""Gate #330: seed-robust system-composed probe distribution (epic #332).

Turns the headline from "a documented favorable draw" into "a reproducible, seed-robust result".
Runs the SYSTEM-COMPOSED probe (lensemble.eval.lewm_system_probe — real deltas through the real
server aggregation path) across several seeds / episode splits and reports the distribution
(mean +/- spread and the WORST case), so the public headline cites the worst draw, not the best.

Writes ``docs/evidence/lewm_tworooms_probe_seedsweep.json``. A worst-case non-improving or
collapse-risk draw fails the gate.

  uv run --with onnxruntime --with hdf5plugin python scripts/lewm_probe_seedsweep.py \
    --h5 ~/.cache/lensemble-lewm/tworoom.h5 --seeds 20260612 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from lensemble.demo.server import load_lewm_manifest
from lensemble.eval.lewm_system_probe import run_system_composed_probe
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
        default=Path("docs/evidence/lewm_tworooms_probe_seedsweep.json"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260612, 1, 2, 3, 4])
    parser.add_argument("--participants", type=int, default=2)
    parser.add_argument("--episodes-per-participant", type=int, default=8)
    parser.add_argument("--validation-episodes", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps-per-round", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = _args()
    manifest = load_lewm_manifest(str(args.model_dir / "manifest.json"))
    if not manifest:
        raise SystemExit(f"real export manifest not found under {args.model_dir}")

    draws = []
    checkpoint = None
    for seed in args.seeds:
        split = build_probe_split(
            h5_path=args.h5,
            model_dir=args.model_dir,
            seed=seed,
            participants=args.participants,
            episodes_per_participant=args.episodes_per_participant,
            validation_episodes=args.validation_episodes,
        )
        checkpoint = split.checkpoint
        evidence = run_system_composed_probe(
            participants=split.participants,
            validation=split.validation,
            checkpoint=split.checkpoint,
            manifest=manifest,
            rounds=args.rounds,
            steps_per_round=args.steps_per_round,
            batch_size=args.batch_size,
            seed=seed,
            dim=split.dim,
        )
        result = evidence["result"]
        diag = result["diagnostics"]
        draws.append(
            {
                "seed": seed,
                "modelRevisionId": evidence["modelRevisionId"],
                "baselineMse": result["baselineMse"],
                "adaptedMse": result["adaptedMse"],
                "relativeImprovement": result["relativeImprovement"],
                "verdict": result["verdict"],
                "displayVerdict": result.get("displayVerdict"),
                "collapseRisk": result.get("collapseRisk", False),
                "validationPairs": result["pairCount"],
                "adaptedLatentStdMean": diag["adapted"]["latentStdMean"],
                "baselineLatentStdMean": diag["baseline"]["latentStdMean"],
            }
        )
        print(
            f"seed {seed}: rel={result['relativeImprovement']:.4f} "
            f"verdict={result.get('displayVerdict')} collapse={result.get('collapseRisk')}"
        )

    rels = [d["relativeImprovement"] for d in draws]
    worst = min(draws, key=lambda d: d["relativeImprovement"])
    all_improved = all(d["verdict"] == "improved" for d in draws)
    any_collapse = any(d["collapseRisk"] for d in draws)
    distribution = {
        "count": len(draws),
        "relativeImprovementMean": statistics.fmean(rels),
        "relativeImprovementStdev": statistics.pstdev(rels) if len(rels) > 1 else 0.0,
        "relativeImprovementMin": min(rels),
        "relativeImprovementMax": max(rels),
        "worstCaseSeed": worst["seed"],
        "worstCaseRelativeImprovement": worst["relativeImprovement"],
        "worstCaseVerdict": worst["verdict"],
        "allSeedsImproved": all_improved,
        "anySeedCollapseRisk": any_collapse,
    }
    passes = all_improved and not any_collapse
    evidence = {
        "schema": "lewm-federated-probe-seedsweep/1",
        "role": "system-composed-seed-robustness",
        "protocol": "system-composed probe (real deltas through the real server aggregation path) "
        "repeated across independent seeds / episode splits; the headline cites the WORST-case "
        "draw, not the best. See docs/evidence/lewm_tworooms_system_probe.json for one full run.",
        "checkpoint": checkpoint,
        "seeds": list(args.seeds),
        "distribution": distribution,
        "draws": draws,
        "passes": passes,
        "headline": (
            f"system-composed held-out MSE improvement across {len(draws)} seeds: "
            f"mean {distribution['relativeImprovementMean'] * 100:.1f}%, "
            f"worst {distribution['worstCaseRelativeImprovement'] * 100:.1f}% "
            f"(seed {distribution['worstCaseSeed']}); "
            f"{'all seeds improved, no collapse' if passes else 'NEGATIVE/COLLAPSE DRAW PRESENT'}."
        ),
        "nonClaims": [
            "Seed-robustness of the system-composed adapter probe; not paper-scale TwoRooms "
            "benchmark parity and not production browser training.",
        ],
    }
    args.out.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"distribution": distribution, "passes": passes}, indent=2))
    if not passes:
        raise SystemExit(
            "seed sweep has a non-improving or collapse-risk draw — recorded in evidence and "
            "blocks public positive claims"
        )


if __name__ == "__main__":
    main()
