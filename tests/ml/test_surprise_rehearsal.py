from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

node = shutil.which("node")
needs_node = pytest.mark.skipif(node is None, reason="node is not installed")


@needs_node
def test_surprise_meter_node_selftest() -> None:
    result = subprocess.run(
        ["node", "web/surprise-meter/surprise_selftest.mjs"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert payload["schema"] == "lewm-surprise-selftest/1"
    assert payload["warmupSteps"] == 2


@needs_node
def test_surprise_rehearsal_script_runs_as_command(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/surprise/rehearsal.py",
            "--offset-out",
            str(tmp_path / "rehearsal_offset.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["schema"] == "surprise-clean-round-rehearsal/1"
    assert report["passes"] is True
    assert report["claimAuditViolations"] == 0
    assert report["serverOffsetParameterCount"] == 12512
    assert report["offsetLength"] == 12512
    assert report["viewerAssets"]["fallbackOffsetLength"] == 12512
    assert report["viewerAssets"]["trajectorySteps"] > 0
