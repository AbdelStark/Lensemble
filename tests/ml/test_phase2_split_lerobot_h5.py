"""Phase 2 LeRobot-H5 episode split utility."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np

from lensemble.data import build_phase2_dataset_smoke_report, load_episodes

_H = _W = 8
_ADIM = 3
_EP_LENS = (5, 4, 6)


def _write_lerobot_h5(path: Path) -> None:
    import h5py

    n = sum(_EP_LENS)
    ep_index = np.concatenate(
        [np.full(length, i, dtype=np.int32) for i, length in enumerate(_EP_LENS)]
    )
    rng = np.random.default_rng(11)
    pixels = rng.integers(0, 256, size=(n, _H, _W, 3), dtype=np.uint8)
    actions = rng.standard_normal((n, _ADIM)).astype(np.float32)
    timestamps = np.arange(n, dtype=np.int64)
    with h5py.File(path, "w") as f:
        f.attrs["source"] = "phase2-test"
        f.create_dataset("episode_index", data=ep_index)
        f.create_dataset("action", data=actions)
        f.create_dataset("timestamp", data=timestamps)
        obs = f.create_group("observation")
        obs.attrs["camera"] = "top"
        obs.create_dataset("pixels_top", data=pixels)
        f.create_dataset("static_meta", data=np.asarray([1, 2], dtype=np.int32))


def _read_episode_index(path: Path) -> list[int]:
    import h5py

    with h5py.File(path, "r") as f:
        dataset = cast(h5py.Dataset, f["episode_index"])
        return [int(x) for x in dataset[:]]


def test_phase2_split_lerobot_h5_cli_writes_loadable_silos(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    out_dir = tmp_path / "silos"
    _write_lerobot_h5(source)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase2_split_lerobot_h5.py",
            "--input",
            str(source),
            "--output-dir",
            str(out_dir),
            "--prefix",
            "so100-silo",
            "--num-silos",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads(result.stdout)
    assert manifest["schema_version"] == 1
    assert manifest["policy"] == "episode_modulo"
    assert manifest["total_episodes"] == 3
    assert manifest["total_frames"] == sum(_EP_LENS)
    assert [silo["source_episode_indices"] for silo in manifest["silos"]] == [
        [0, 2],
        [1],
    ]

    silo0 = out_dir / "so100-silo0.h5"
    silo1 = out_dir / "so100-silo1.h5"
    assert _read_episode_index(silo0) == [0] * _EP_LENS[0] + [1] * _EP_LENS[2]
    assert _read_episode_index(silo1) == [0] * _EP_LENS[1]

    ds0 = load_episodes(f"lerobot-h5://{silo0}")
    ds1 = load_episodes(f"lerobot-h5://{silo1}")
    assert len(ds0) == 2
    assert len(ds1) == 1
    assert sum(1 for _ in ds0.windows(2)) == 7
    assert sum(1 for _ in ds1.windows(2)) == 2

    report = build_phase2_dataset_smoke_report(
        (f"lerobot-h5://{silo0}", f"lerobot-h5://{silo1}"),
        participant_ids=("phase2-a", "phase2-b"),
        window_steps=2,
    )
    assert report.participant_count == 2
    assert [silo.window_count for silo in report.silos] == [7, 2]
    assert report.silos[0].dataset_root != report.silos[1].dataset_root
