#!/usr/bin/env python3
"""Smoke-test Phase 2 participant-silo dataset refs.

The command loads each source through the public Lensemble data adapter,
computes the dataset Merkle commitment, counts fixed-horizon windows, and emits
a residency-safe JSON report suitable for tracker comments, HF Job dry-runs,
and evidence bundles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from lensemble.data import build_phase2_dataset_smoke_report
from lensemble.data.dataset import Format

_FORMATS = ("lance", "hdf5", "lerobot", "lerobot-h5")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-source",
        action="append",
        required=True,
        help="Participant-local data source path or URI. Repeat once per silo.",
    )
    parser.add_argument(
        "--participant-id",
        action="append",
        help="Participant id matching a --data-source. Omit to auto-name participants.",
    )
    parser.add_argument(
        "--data-format",
        choices=_FORMATS,
        help="Adapter fmt override. Omit to infer from URI scheme or path suffix.",
    )
    parser.add_argument(
        "--window-steps",
        type=int,
        default=4,
        help="Fixed transition horizon used by EpisodeDataset.windows().",
    )
    parser.add_argument(
        "--min-windows-per-silo",
        type=int,
        default=1,
        help="Fail unless each participant produces at least this many windows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON report. stdout is always written.",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    report = build_phase2_dataset_smoke_report(
        tuple(cast(list[str], args.data_source)),
        participant_ids=(
            tuple(cast(list[str], args.participant_id))
            if args.participant_id is not None
            else None
        ),
        data_format=cast(Format | None, args.data_format),
        window_steps=int(args.window_steps),
        min_windows_per_silo=int(args.min_windows_per_silo),
    )
    payload = json.dumps(report.model_dump(mode="json"), indent=2)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
