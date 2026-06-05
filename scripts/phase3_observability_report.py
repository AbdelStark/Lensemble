#!/usr/bin/env python3
"""Generate or validate the Phase 3 observability/dropout report."""

from __future__ import annotations

import argparse
from pathlib import Path

from lensemble.federation import (
    build_phase3_observability_report,
    load_phase3_observability_report,
    write_phase3_observability_report,
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
        "--eval-report",
        type=Path,
        default=Path("docs/evidence/phase3_eval_report.json"),
        help="Phase 3 eval report produced by scripts/phase3_eval_report.py.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/phase3-observability-smoke"),
        help="Directory for the induced-dropout trace and local artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/phase3_observability_report.json"),
        help="Where to write the generated Phase 3 observability report.",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        default=None,
        help="Validate an existing observability report instead of generating one.",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.validate is not None:
        report = load_phase3_observability_report(args.validate)
        print(
            "validated "
            f"{args.validate}: {len(report.rounds)} round summaries, "
            f"{len(report.dropout_decisions)} dropout decisions"
        )
        return

    report = build_phase3_observability_report(
        long_run_report_path=args.long_run_report,
        eval_report_path=args.eval_report,
        run_dir=args.run_dir,
        output_uri=_report_uri(args.output),
    )
    path = write_phase3_observability_report(report, args.output)
    load_phase3_observability_report(path)
    print(
        f"wrote {path}: {len(report.rounds)} round summaries, "
        f"{len(report.dropout_decisions)} dropout decisions"
    )


def _report_uri(path: Path) -> str:
    if path.is_absolute():
        return f"artifact://phase3-observability/{path.name}"
    return str(path)


if __name__ == "__main__":
    main()
