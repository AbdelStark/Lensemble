#!/usr/bin/env python3
"""Render the Phase 2 evidence matrix.

The default Markdown output is intended for roadmap docs and issue comments.
JSON output is available for automation that wants a machine-readable tracker
contract.
"""

from __future__ import annotations

import argparse
import json

from lensemble.eval import default_phase2_matrix, render_phase2_matrix_markdown


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    rows = default_phase2_matrix()
    if args.format == "json":
        print(json.dumps([row.model_dump(mode="json") for row in rows], indent=2))
        return
    print(render_phase2_matrix_markdown(rows))


if __name__ == "__main__":
    main()
