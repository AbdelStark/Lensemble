from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

EVIDENCE = Path("docs/evidence")


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "lewm_surprise_check", Path("scripts/lewm_surprise_check.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_surprise_evidence_contract_matches_source_numbers() -> None:
    committed = EVIDENCE / "lewm_tworooms_surprise.json"
    if not committed.is_file():
        pytest.skip("surprise evidence not generated yet")
    payload = json.loads(committed.read_text(encoding="utf-8"))
    system = json.loads(
        (EVIDENCE / "lewm_tworooms_system_probe.json").read_text(encoding="utf-8")
    )
    seedsweep = json.loads(
        (EVIDENCE / "lewm_tworooms_probe_seedsweep.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["schema"] == "lewm-surprise/1"
    assert payload["passes"] is True
    assert payload["meanSurprisePost"] < payload["meanSurprisePre"]
    assert (
        abs(
            payload["federatedRelativeImprovement"]
            - system["result"]["relativeImprovement"]
        )
        <= 1e-6
    )
    assert (
        abs(
            payload["federatedSeedMean"]
            - seedsweep["distribution"]["relativeImprovementMean"]
        )
        <= 1e-6
    )
    assert (
        abs(
            payload["federatedSeedWorst"]
            - seedsweep["distribution"]["relativeImprovementMin"]
        )
        <= 1e-6
    )
    non_claims = " ".join(payload["nonClaims"])
    for needle in (
        "adapter-continuation-not-training",
        "surprise-is-scalar-CLS",
        "no-secure-agg/DP",
        "perturbation-illustrative",
    ):
        assert needle in non_claims


def test_surprise_producer_writes_docs_and_served_assets(tmp_path: Path) -> None:
    out = tmp_path / "lewm_tworooms_surprise.json"
    traj = tmp_path / "surprise_trajectory.json"
    card = tmp_path / "result_card.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/lewm_surprise_check.py",
            "--out",
            str(out),
            "--trajectory-out",
            str(traj),
            "--result-card-out",
            str(card),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    trajectory = json.loads(traj.read_text(encoding="utf-8"))
    result_card = json.loads(card.read_text(encoding="utf-8"))
    assert payload["schema"] == "lewm-surprise/1"
    assert payload["passes"] is True
    assert trajectory["schema"] == "lewm-surprise-traj/1"
    assert 0 < len(trajectory["steps"]) <= 600
    assert result_card["schema"] == "lewm-surprise-result-card/1"
    assert result_card["display"]["thisRun"] == "+12.3%"
    assert result_card["display"]["seedMean"] == "+16.8%"
    assert result_card["display"]["seedWorst"] == "+5.4%"


def test_surprise_pass_predicate_requires_mandatory_non_claims() -> None:
    module = _load_module()
    system = json.loads(
        (EVIDENCE / "lewm_tworooms_system_probe.json").read_text(encoding="utf-8")
    )
    seedsweep = json.loads(
        (EVIDENCE / "lewm_tworooms_probe_seedsweep.json").read_text(
            encoding="utf-8"
        )
    )
    trajectory = module.build_fallback_trajectory(
        mean_pre=system["result"]["baselineMse"],
        mean_post=system["result"]["adaptedMse"],
    )
    payload = module.build_surprise_evidence(
        system=system, seedsweep=seedsweep, trajectory=trajectory
    )
    assert module.surprise_passes(payload) is True
    payload["nonClaims"] = payload["nonClaims"][:-1]
    assert module.surprise_passes(payload) is False
