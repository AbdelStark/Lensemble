#!/usr/bin/env python3
"""Generate or validate the RFC-0017 dynamic-env observability/privacy report."""

from __future__ import annotations

import argparse
from pathlib import Path

from lensemble.federation import (
    build_dynamic_env_observability_report,
    load_dynamic_env_observability_report,
    write_dynamic_env_observability_report,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long-run-report",
        type=Path,
        default=Path("docs/evidence/dynamic_env_long_run_report.json"),
        help="Dynamic-env long-run training report from the C11 launcher run.",
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=Path("runs/dynamic-env/phase3_run_manifest.json"),
        help="Run manifest emitted by the C11 launcher run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/dynamic_env_observability_report.json"),
        help="Where to write the dynamic-env observability/privacy report.",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        default=None,
        help="Validate an existing dynamic-env observability report instead of generating one.",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.validate is not None:
        report = load_dynamic_env_observability_report(args.validate)
        print(
            f"validated {args.validate}: {len(report.rounds)} rounds, "
            f"{report.dp_accounted_rounds} DP-accounted rounds"
        )
        return

    report = build_dynamic_env_observability_report(
        long_run_report_path=args.long_run_report,
        run_manifest_path=args.run_manifest,
    )
    path = write_dynamic_env_observability_report(report, args.output)
    load_dynamic_env_observability_report(path)
    print(
        f"wrote {path}: {len(report.rounds)} rounds, "
        f"{report.secure_sum_rounds} secure-sum rounds"
    )


if __name__ == "__main__":
    main()
