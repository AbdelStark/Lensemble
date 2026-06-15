"""Real-mode eval diagnostics: probe, before/after comparison, health flags (#322, epic #314).

The probe semantics are node-tested (``web/federated-demo/lewm_probe_selftest.mjs``: improved /
worse / flat verdicts over a known systematic bias, identity-baseline equivalence, deterministic
fixtures, health flags). This harness runs them, checks the server-side health flags mirror the
JS ones, and pins the committed gate-G5 federated probe evidence — including that a non-improving
verdict blocks the gate instead of being softened.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from lensemble.demo.federated import FederatedDemoService

WEB_DIR = Path("web/federated-demo")

node = shutil.which("node")
needs_node = pytest.mark.skipif(node is None, reason="node is not installed")


@needs_node
def test_probe_selftest_passes_under_node() -> None:
    assert node is not None
    result = subprocess.run(
        [node, str(WEB_DIR / "lewm_probe_selftest.mjs")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["failed"] == 0
    assert report["total"] >= 6


def test_server_health_flags_mirror_probe_semantics() -> None:
    flags = FederatedDemoService._lewm_health_flags(
        [
            {
                "effectiveRank": 2.0,
                "latentStdMean": 0.01,
                "predLossFirst": 0.06,
                "predLossLast": 0.06,
                "sigregStatistic": 2.5,
            }
        ]
    )
    text = " ".join(flags)
    assert "effective rank" in text
    assert "magnitude collapse" in text
    assert "flat-loss" in text
    assert "sigreg" in text

    healthy = FederatedDemoService._lewm_health_flags(
        [
            {
                "effectiveRank": 11.0,
                "latentStdMean": 0.8,
                "predLossFirst": 0.06,
                "predLossLast": 0.02,
                "sigregStatistic": 0.05,
            }
        ]
    )
    assert healthy == []

    worsened = FederatedDemoService._lewm_health_flags(
        [{"predLossFirst": 0.02, "predLossLast": 0.05}]
    )
    assert any("loss-worsened" in flag for flag in worsened)


def test_federated_probe_evidence_improved() -> None:
    committed = Path("docs/evidence/lewm_tworooms_probe_check.json")
    if not committed.is_file():
        pytest.skip(
            "federated probe evidence not generated yet (scripts/lewm_probe_check.py)"
        )
    evidence = json.loads(committed.read_text())
    assert evidence["schema"] == "lewm-federated-probe/1"
    result = evidence["result"]
    # the binding before/after metric: held-out validation MSE must actually improve, and the
    # pass flag must agree with the verdict (no softening)
    assert result["verdict"] == "improved"
    assert result["adaptedMse"] < result["baselineMse"]
    assert result["relativeImprovement"] > 0.02
    assert evidence["passes"] is (result["verdict"] == "improved")
    assert len(evidence["checkpoint"]["revision"]) == 40
    # disjoint train/validation protocol is recorded
    assert "held-out validation" in evidence["protocol"]
    assert result["participants"] >= 2


def test_federated_probe_evidence_carries_heldout_collapse_diagnostics() -> None:
    """#328: the probe evidence must carry held-out collapse diagnostics (the #259 blind spot)."""
    committed = Path("docs/evidence/lewm_tworooms_probe_check.json")
    if not committed.is_file():
        pytest.skip("federated probe evidence not generated yet")
    evidence = json.loads(committed.read_text())
    result = evidence["result"]
    assert result["collapseRisk"] is False
    diag = result["diagnostics"]
    for side in ("baseline", "adapted"):
        assert diag[side]["latentStdMean"] > 0
        assert diag[side]["effectiveRank"] > 0
        assert "sigregStatistic" in diag[side]
    # the gain is bias-correction, not magnitude/rank collapse on held-out
    assert diag["adapted"]["latentStdMean"] >= 0.7 * diag["baseline"]["latentStdMean"]
    assert diag["adapted"]["effectiveRank"] >= 0.7 * diag["baseline"]["effectiveRank"]


def test_dashboard_probe_reports_negative_results() -> None:
    app = (WEB_DIR / "app.mjs").read_text(encoding="utf-8")
    assert "compareRevisions" in app
    assert "did not beat the parent checkpoint" in app
    assert "reported, not hidden" in app
    assert "healthFlags" in app


def test_dashboard_renders_scoped_claim_boundary() -> None:
    """#329: the strict claim boundary is rendered in-UI and the H1 / probe badge don't oversell."""
    app = (WEB_DIR / "app.mjs").read_text(encoding="utf-8")
    # H1 names adaptation of a frozen checkpoint, not "federate a world model"
    assert "Federate adapter updates on a real world-model checkpoint" in app
    assert "Federate a world model across browsers" not in app
    # the scoped chips + single-coordinator note render near the probe, sourced from the snapshot
    assert "renderClaimBoundary" in app
    assert "bounded adapter on a frozen base" in app
    assert "fixed held-out probe, not a benchmark" in app
    assert "single local coordinator" in app.lower()
    assert "mean-of-clipped-deltas (no robust aggregation / DP in this path)" in app
    assert "run.claimBoundary" in app  # full boundary text comes from the run snapshot
    # the probe badge carries the "held-out probe, not a benchmark" caveat and a collapse override
    assert "Held-out probe, not a benchmark" in app
    assert "Collapse risk overrides the MSE verdict" in app
