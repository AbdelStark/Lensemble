#!/usr/bin/env python3
"""Generate or validate the Phase 3 dataset/public-probe registry."""

from __future__ import annotations

import argparse
from pathlib import Path

from lensemble.config import (
    default_phase3_consortium_manifest,
    load_consortium_manifest,
)
from lensemble.data import (
    load_phase3_dataset_registry,
    phase3_registry_from_consortium_manifest,
    validate_phase3_registry_against_manifest,
    write_phase3_dataset_registry,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/phase3_dataset_registry.example.json"),
        help="Where to write the generated example registry.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest to generate from; omit for the built-in Phase 3 example.",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        default=None,
        help="Validate an existing registry instead of generating the example.",
    )
    parser.add_argument(
        "--against-manifest",
        type=Path,
        default=None,
        help="Also validate the registry against this consortium manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.validate is not None:
        registry = load_phase3_dataset_registry(args.validate)
        if args.against_manifest is not None:
            manifest = load_consortium_manifest(args.against_manifest)
            validate_phase3_registry_against_manifest(registry, manifest)
        print(
            "validated "
            f"{args.validate}: {registry.consortium_id}/{registry.run_id} "
            f"with {len(registry.participants)} participants"
        )
        return

    manifest = (
        load_consortium_manifest(args.manifest)
        if args.manifest is not None
        else default_phase3_consortium_manifest()
    )
    registry = phase3_registry_from_consortium_manifest(manifest)
    path = write_phase3_dataset_registry(registry, args.output)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
