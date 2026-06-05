#!/usr/bin/env python3
"""Run or validate the deterministic Phase 3 coordinator/participant-agent smoke."""

from __future__ import annotations

import argparse
from pathlib import Path

from lensemble.federation import (
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
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.validate is not None:
        report = load_phase3_long_run_report(args.validate)
        print(
            "validated "
            f"{args.validate}: {report.consortium_id}/{report.run_id} "
            f"with {report.closed_rounds}/{report.target_rounds} closed rounds"
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
