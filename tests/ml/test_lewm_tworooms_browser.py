"""TwoRooms browser environment + real-LeWM runtime wiring (#318, epic #314).

The deterministic JS tests live in ``web/federated-demo/tworooms_selftest.mjs`` (environment
geometry/dynamics/expert policy validated against the released dataset frames, plus rollout
windowing and CEM planning over injected fake sessions). This harness runs them under node so
they gate CI alongside the Python suite (skipped when node is unavailable), and pins the
claim-language and evidence contracts the panel relies on.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

WEB_DIR = Path("web/federated-demo")


def test_tworooms_selftest_passes_under_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    result = subprocess.run(
        [node, str(WEB_DIR / "tworooms_selftest.mjs")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["failed"] == 0
    assert report["passed"] == report["total"]
    assert report["total"] >= 18


def test_tworooms_env_documents_deviations() -> None:
    env = (WEB_DIR / "tworooms_env.mjs").read_text(encoding="utf-8")
    assert "TWOROOMS_DEVIATIONS" in env
    assert "Not the" in env and "upstream torch environment" in env
    # the geometry constants the dataset frames pinned
    for needle in ("WALL_CENTER = 112", "DOOR_CENTER_Y = 49", "ACTION_BLOCK = 5"):
        assert needle in env, needle


def test_lewm_runtime_refuses_silent_fallback() -> None:
    runtime = (WEB_DIR / "lewm_runtime.mjs").read_text(encoding="utf-8")
    assert "LewmUnsupportedError" in runtime
    assert "no silent fallback" in runtime
    assert "hash-mismatch" in runtime  # graph integrity is checked, not assumed
    panel = (WEB_DIR / "tworooms_panel.mjs").read_text(encoding="utf-8")
    assert "TWOROOMS_DEVIATIONS" in panel  # the deviation text is shown in the UI


def test_realdata_check_evidence_passes() -> None:
    committed = Path("docs/evidence/lewm_tworooms_realdata_check.json")
    if not committed.is_file():
        pytest.skip("real-data check not generated yet (scripts/lewm_tworooms_realdata_check.py)")
    report = json.loads(committed.read_text())
    assert report["schema"] == "lewm-realdata-check/1"
    assert report["passes"] is True
    # the exported pipeline must beat copy-last decisively, or the demo claim is hollow
    assert report["modelOverBaselineRatio"] < 0.5
    assert len(report["checkpoint"]["revision"]) == 40
    assert set(report["graphHashes"]) == {
        "lewm_tworooms_encoder.onnx",
        "lewm_tworooms_action.onnx",
        "lewm_tworooms_predictor.onnx",
    }


def test_action_stats_evidence_contract() -> None:
    committed = Path("docs/evidence/lewm_tworooms_action_stats.json")
    assert committed.is_file(), "action stats are required to regenerate the export"
    stats = json.loads(committed.read_text())
    assert stats["schema"] == "lewm-action-stats/1"
    assert len(stats["mean"]) == 2 and len(stats["std"]) == 2
    assert all(s > 0 for s in stats["std"])
    assert stats["dataset"]["repoId"] == "quentinll/lewm-tworooms"


def test_export_manifest_records_baked_normalizations() -> None:
    committed = Path("docs/evidence/lewm_tworooms_browser_export_manifest.json")
    if not committed.is_file():
        pytest.skip("export manifest not generated yet")
    manifest = json.loads(committed.read_text())
    norm = manifest["normalization"]
    assert norm["pixels"]["kind"] == "imagenet"
    assert norm["pixels"]["mean"] == [0.485, 0.456, 0.406]
    assert "z-score" in norm["actions"]["kind"]
    assert norm["actions"]["stats"]["schema"] == "lewm-action-stats/1"
