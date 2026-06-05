"""Phase 2 participant-silo data smoke reports."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from pydantic import ValidationError

from lensemble.data import (
    PHASE2_DATASET_SMOKE_SCHEMA_VERSION,
    build_phase2_dataset_smoke_report,
)

_H = _W = 8
_ADIM = 3


def _load_hfjobs_launcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "train_federated_lewm_test", Path("deploy/hfjobs/train_federated_lewm.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_lerobot_h5(
    path: Path, *, ep_lens: tuple[int, ...] = (5, 4), seed: int = 0
) -> None:
    import h5py

    n = sum(ep_lens)
    ep_index = np.concatenate(
        [np.full(length, i, dtype=np.int32) for i, length in enumerate(ep_lens)]
    )
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(n, _H, _W, 3), dtype=np.uint8)
    actions = rng.standard_normal((n, _ADIM)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("episode_index", data=ep_index)
        f.create_dataset("observation/pixels_top", data=pixels)
        f.create_dataset("action", data=actions)


def test_build_phase2_dataset_smoke_report_records_roots_and_shapes(
    tmp_path: Path,
) -> None:
    silo_a = tmp_path / "phase2-a.h5"
    silo_b = tmp_path / "phase2-b.h5"
    _write_lerobot_h5(silo_a, seed=1)
    _write_lerobot_h5(silo_b, seed=2)

    report = build_phase2_dataset_smoke_report(
        (f"lerobot-h5://{silo_a}", f"lerobot-h5://{silo_b}"),
        participant_ids=("silo-a", "silo-b"),
        window_steps=2,
        min_windows_per_silo=1,
    )

    assert report.schema_version == PHASE2_DATASET_SMOKE_SCHEMA_VERSION
    assert report.participant_count == 2
    assert report.blocker is None
    assert [silo.participant_id for silo in report.silos] == ["silo-a", "silo-b"]
    assert {silo.data_format for silo in report.silos} == {"lerobot-h5"}
    assert all(silo.episode_count == 2 for silo in report.silos)
    assert all(silo.window_count == 5 for silo in report.silos)
    assert all(len(silo.dataset_root) == 64 for silo in report.silos)
    assert report.silos[0].dataset_root != report.silos[1].dataset_root
    assert report.silos[0].observation_shape == (3, 1, 3, _H, _W)
    assert report.silos[0].action_shape == (2, _ADIM)
    assert report.silos[0].action_spec.dim == _ADIM
    assert report.silos[0].action_spec.embodiment_id == "lerobot-3dof"


def test_build_phase2_dataset_smoke_report_fails_on_zero_window_silo(
    tmp_path: Path,
) -> None:
    short_silo = tmp_path / "short.h5"
    _write_lerobot_h5(short_silo, ep_lens=(2,))

    with pytest.raises(ValidationError, match="below min_windows_per_silo"):
        build_phase2_dataset_smoke_report(
            (f"lerobot-h5://{short_silo}",),
            participant_ids=("short",),
            window_steps=2,
            min_windows_per_silo=1,
        )


def test_build_phase2_dataset_smoke_report_fails_on_duplicate_participants(
    tmp_path: Path,
) -> None:
    silo_a = tmp_path / "dup-a.h5"
    silo_b = tmp_path / "dup-b.h5"
    _write_lerobot_h5(silo_a, seed=3)
    _write_lerobot_h5(silo_b, seed=4)

    with pytest.raises(ValidationError, match="duplicate participant ids"):
        build_phase2_dataset_smoke_report(
            (f"lerobot-h5://{silo_a}", f"lerobot-h5://{silo_b}"),
            participant_ids=("dup", "dup"),
            window_steps=2,
        )


def test_phase2_dataset_smoke_script_outputs_json_and_file(tmp_path: Path) -> None:
    silo = tmp_path / "cli.h5"
    output = tmp_path / "report.json"
    _write_lerobot_h5(silo)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase2_dataset_smoke.py",
            "--data-source",
            str(silo),
            "--participant-id",
            "cli-silo",
            "--data-format",
            "lerobot-h5",
            "--window-steps",
            "2",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout_report = json.loads(result.stdout)
    file_report = json.loads(output.read_text(encoding="utf-8"))
    assert stdout_report == file_report
    assert stdout_report["participant_count"] == 1
    assert stdout_report["silos"][0]["participant_id"] == "cli-silo"
    assert stdout_report["silos"][0]["window_count"] == 5
    assert stdout_report["silos"][0]["data_format"] == "lerobot-h5"


def test_hfjobs_validate_sources_counts_windows_from_adapter(tmp_path: Path) -> None:
    silo = tmp_path / "launcher.h5"
    _write_lerobot_h5(silo)
    source = f"lerobot-h5://{silo}"
    launcher = _load_hfjobs_launcher()

    counts = getattr(launcher, "_validate_sources")(
        argparse.Namespace(
            data_source=[source],
            data_format="lerobot-h5",
            window_steps=2,
        )
    )

    assert counts == {source: 5}
