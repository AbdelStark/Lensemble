#!/usr/bin/env python3
"""Compute the TwoRooms expert-dataset action z-score statistics used by the LeWM checkpoint.

The upstream training pipeline z-scores the raw 2D actions with per-dimension statistics fitted on
the full expert dataset (``stable_worldmodel.data.column_normalizer`` → ``ZScoreScaler``) before
chunking them into frameskip-5 blocks. Browser inference must apply the same statistics, so this
script computes them from the official dataset (``quentinll/lewm-tworooms`` dataset repo,
``tworoom.h5``) and writes ``docs/evidence/lewm_tworooms_action_stats.json``.

The dataset stores one NaN action row per episode (the terminal step); statistics are computed
with ``nanmean``/``nanstd`` and the NaN handling is recorded in the artifact. Requires ``h5py``
plus ``hdf5plugin`` (the pixels column uses a compression plugin; the action column itself does
not, but opening the file can still require the plugin registry).

Reproduce:
  uv run --with hdf5plugin python scripts/lewm_tworooms_action_stats.py \
    --h5 ~/.cache/lensemble-lewm/tworoom.h5
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import h5py
import numpy as np

DATASET_REPO = "quentinll/lewm-tworooms"
DATASET_REVISION = "6903a2de048b13819d812da0b4dd661290bc01e4"
STATS_SCHEMA = "lewm-action-stats/1"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True, help="Path to tworoom.h5.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/evidence/lewm_tworooms_action_stats.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    with h5py.File(args.h5, "r") as f:
        actions = np.asarray(cast(h5py.Dataset, f["action"])[:])
        rows = int(actions.shape[0])
        episodes = int(cast(h5py.Dataset, f["ep_len"]).shape[0])
    nan_rows = int(np.isnan(actions).any(axis=1).sum())
    mean = np.nanmean(actions, axis=0)
    std = np.nanstd(actions, axis=0)
    artifact = {
        "schema": STATS_SCHEMA,
        "dataset": {
            "repoId": DATASET_REPO,
            "revision": DATASET_REVISION,
            "file": "tworoom.h5 (from tworoom.tar.zst)",
            "rows": rows,
            "episodes": episodes,
            "nanRows": nan_rows,
            "nanHandling": "nanmean/nanstd (one NaN action per episode terminal step)",
        },
        "method": "per-dimension z-score: (a - mean) / std, fitted on the raw 2D actions before "
        "frameskip-5 block chunking (mirrors stable_worldmodel ZScoreScaler)",
        "actionDim": 2,
        "mean": [round(float(v), 8) for v in mean],
        "std": [round(float(v), 8) for v in std],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"action stats -> {args.out} (sha256 {digest[:16]}…)")
    print(json.dumps({"mean": artifact["mean"], "std": artifact["std"]}))


if __name__ == "__main__":
    main()
