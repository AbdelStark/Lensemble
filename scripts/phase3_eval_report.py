#!/usr/bin/env python3
"""Generate or validate the Phase 3 eval and matched-control report."""

from __future__ import annotations

import argparse
from pathlib import Path

from lensemble.eval import (
    build_phase3_eval_report,
    load_phase3_eval_report,
    write_phase3_eval_report,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long-run-report",
        type=Path,
        default=Path("docs/evidence/phase3_long_run_smoke_report.json"),
        help="Phase 3 long-run report produced by scripts/phase3_consortium_smoke.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/phase3_eval_report.json"),
        help="Where to write the generated Phase 3 eval report.",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        default=None,
        help="Validate an existing eval report instead of generating one.",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.validate is not None:
        report = load_phase3_eval_report(args.validate)
        print(
            "validated "
            f"{args.validate}: {len(report.metric_rows)} metric rows, "
            f"{len(report.blocked_controls)} blocked controls"
        )
        return

    report = build_phase3_eval_report(args.long_run_report)
    path = write_phase3_eval_report(report, args.output)
    load_phase3_eval_report(path)
    print(
        f"wrote {path}: {len(report.metric_rows)} metric rows, "
        f"{len(report.blocked_controls)} blocked controls"
    )


if __name__ == "__main__":
    main()
