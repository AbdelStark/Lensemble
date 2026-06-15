"""Browser-local LeWM adapter continuation (#319, epic #314) — gate G3.

The adapter math, training loop, SIGReg port, and resident-collection wiring are tested
deterministically in JS (``web/federated-demo/lewm_adapter_selftest.mjs``: identity-at-init,
analytic-vs-numerical gradients, loss decrease, variance-floor activation, collapse detection,
delta clipping). This harness runs them under node, replays the JS SIGReg fixture through the
canonical torch implementation (``lensemble.model.sigreg``) to prove the port matches, and pins
the committed real-latent overfit evidence.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from lensemble.model.sigreg import sigreg_statistic

WEB_DIR = Path("web/federated-demo")

node = shutil.which("node")
needs_node = pytest.mark.skipif(node is None, reason="node is not installed")


@needs_node
def test_adapter_selftest_passes_under_node() -> None:
    assert node is not None
    result = subprocess.run(
        [node, str(WEB_DIR / "lewm_adapter_selftest.mjs")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["failed"] == 0
    assert report["total"] >= 10


@needs_node
def test_js_sigreg_port_matches_torch_reference() -> None:
    """The JS Epps–Pulley statistic must agree with lensemble.model.sigreg on identical inputs."""
    assert node is not None
    result = subprocess.run(
        [node, str(WEB_DIR / "lewm_adapter_selftest.mjs"), "--dump-sigreg-fixture"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    fixture = json.loads(result.stdout.strip().splitlines()[-1])
    d, n, s = fixture["d"], fixture["n"], fixture["sketchDim"]
    sketch = torch.tensor(fixture["sketch"], dtype=torch.float32).reshape(d, s)
    for key, js_value in (
        ("normal", fixture["normalStatistic"]),
        ("collapsed", fixture["collapsedStatistic"]),
    ):
        emb = torch.tensor(fixture[key], dtype=torch.float32).reshape(n, d)
        torch_value = float(sigreg_statistic(emb, sketch))
        assert js_value == pytest.approx(torch_value, rel=1e-3, abs=1e-6), key
    # behavioral sanity carried by the same fixture
    assert fixture["collapsedStatistic"] > 50 * fixture["normalStatistic"]


def test_adapter_overfit_evidence_passes() -> None:
    committed = Path("docs/evidence/lewm_tworooms_adapter_overfit.json")
    if not committed.is_file():
        pytest.skip(
            "overfit evidence not generated yet (scripts/lewm_adapter_overfit_check.py)"
        )
    evidence = json.loads(committed.read_text())
    assert evidence["schema"] == "lewm-adapter-overfit/1"
    assert evidence["passes"] is True
    trainer = evidence["trainer"]
    assert trainer["lossDecreased"] is True
    assert trainer["relativeImprovement"] > 0.2
    assert trainer["lastPredLoss"] < trainer["firstPredLoss"]
    diag = trainer["finalDiagnostics"]
    # diagnostics must be real numbers from real tensors, not placeholders
    for key in ("sigregStatistic", "effectiveRank", "latentStdMean"):
        assert isinstance(diag[key], (int, float)) and diag[key] > 0, key
    assert trainer["delta"]["parameterCount"] > 0
    assert trainer["delta"]["l2Norm"] <= 3.0 + 1e-6
    assert len(evidence["checkpoint"]["revision"]) == 40
    assert any("not from-scratch" in t for t in evidence["nonClaims"])


def test_trainer_has_no_artificial_sleeps() -> None:
    for name in ("lewm_adapter.mjs", "lewm_local_trainer.mjs"):
        source = (WEB_DIR / name).read_text(encoding="utf-8")
        assert "setTimeout" not in source, f"{name} must not fake progress with sleeps"
        assert "setInterval" not in source, name


def test_trainer_documents_claim_boundary() -> None:
    source = (WEB_DIR / "lewm_adapter.mjs").read_text(encoding="utf-8")
    assert "not from-scratch LeWM browser pretraining" in source
    trainer = (WEB_DIR / "lewm_local_trainer.mjs").read_text(encoding="utf-8")
    assert "never leave the browser" in trainer
