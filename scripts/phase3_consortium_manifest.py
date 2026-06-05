#!/usr/bin/env python3
"""Generate or validate the Phase 3 example consortium manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from lensemble.config import (
    default_phase3_consortium_manifest,
    load_consortium_manifest,
    write_consortium_manifest,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/phase3_consortium_manifest.example.json"),
        help="Where to write the generated example manifest.",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        default=None,
        help="Validate an existing manifest instead of generating the example.",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.validate is not None:
        manifest = load_consortium_manifest(args.validate)
        print(
            "validated "
            f"{args.validate}: {manifest.consortium_id}/{manifest.run_id} "
            f"with {len(manifest.participants)} participants"
        )
        return
    path = write_consortium_manifest(default_phase3_consortium_manifest(), args.output)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
