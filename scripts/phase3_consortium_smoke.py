#!/usr/bin/env python3
"""Run or validate the deterministic Phase 3 coordinator/participant-agent smoke."""

from __future__ import annotations

import argparse
from pathlib import Path

from lensemble.federation import (
    PHASE3_HISTORICAL_CLAIM_BOUNDARY,
    PHASE3_LONG_RUN_REPORT_SCHEMA_VERSION,
    load_phase3_long_run_report,
    run_phase3_long_run_smoke,
    write_phase3_long_run_report,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/phase3_long_run_smoke_report.json"),
        help="Where to write the generated long-run smoke report.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/phase3-long-run-smoke"),
        help="Directory for local smoke manifests, traces, and checkpoints.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Closed federated rounds to run; defaults to the Phase 3 minimum.",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        default=None,
        help="Validate an existing report instead of running the smoke.",
    )
    parser.add_argument(
        "--normalize-legacy",
        type=Path,
        default=None,
        help=(
            "Load a schema-v1 report through the conservative migration and write "
            "the explicit historical schema-v2 representation to --output."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.validate is not None and args.normalize_legacy is not None:
        raise SystemExit("--validate and --normalize-legacy are mutually exclusive")
    if args.validate is not None:
        report = load_phase3_long_run_report(args.validate)
        print(
            "validated "
            f"{args.validate}: {report.consortium_id}/{report.run_id} "
            f"with {report.closed_rounds}/{report.target_rounds} closed rounds"
        )
        return
    if args.normalize_legacy is not None:
        report = load_phase3_long_run_report(args.normalize_legacy)
        if (
            report.schema_version != 1
            and report.evidence_status != "historical_pre_correctness_fix"
        ):
            raise SystemExit(
                "--normalize-legacy requires schema-v1 or already-classified "
                "historical evidence"
            )
        normalized = report.model_copy(
            update={
                "schema_version": PHASE3_LONG_RUN_REPORT_SCHEMA_VERSION,
                "claim_boundary": PHASE3_HISTORICAL_CLAIM_BOUNDARY,
            }
        )
        path = write_phase3_long_run_report(normalized, args.output)
        load_phase3_long_run_report(path)
        print(
            f"normalized {args.normalize_legacy} to {path}: "
            f"evidence_status={normalized.evidence_status}"
        )
        return

    report = run_phase3_long_run_smoke(run_dir=args.run_dir, rounds=args.rounds)
    path = write_phase3_long_run_report(report, args.output)
    load_phase3_long_run_report(path)
    print(
        f"wrote {path}: {report.closed_rounds}/{report.target_rounds} closed rounds, "
        f"completed_target={report.completed_target}"
    )


if __name__ == "__main__":
    main()
