"""System-composed federation gate (#327) + seed-robustness (#330), epic #332.

Two layers:
- a dataset-free unit run that drives the REAL composition (lensemble.eval.lewm_system_probe:
  real node-trained deltas -> FederatedDemoService.submit_update -> _close_round_lewm -> probe on
  the server-produced revision) over synthetic 192-dim bias-correctable pairs; and
- contract checks pinning the committed headline / cross-check / seed-sweep evidence so the shipped
  artifact cannot silently drift from "system-composed, collapse-checked, seed-robust".
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import pytest

WEB_DIR = Path("web/federated-demo")
EVIDENCE = Path("docs/evidence")

node = shutil.which("node")
needs_node = pytest.mark.skipif(node is None, reason="node is not installed")


def _synthetic_manifest() -> dict:
    revision = (
        "a" * 40
    )  # 40-hex like a real git sha; binding only checks length/presence
    return {
        "schema": "lewm-browser-export/1",
        "graphVersion": 1,
        "checkpoint": {
            "repoId": "synthetic/lewm-tworooms",
            "revision": revision,
            "weightsSha256": hashlib.sha256(b"synthetic-weights").hexdigest(),
        },
        "files": {
            name: {"sha256": hashlib.sha256(name.encode()).hexdigest()}
            for name in (
                "lewm_tworooms_encoder.onnx",
                "lewm_tworooms_action.onnx",
                "lewm_tworooms_predictor.onnx",
            )
        },
    }


def _bias_correctable_pairs(*, n: int, dim: int, seed: int) -> dict:
    """Frozen predictor x = target + systematic per-dim bias; the adapter can learn to subtract it.

    Pure-Python (no numpy dependency in the test path); deterministic for a fixed seed.
    """
    rng = __import__("random").Random(seed)
    bias = [
        0.12 * math.sin(k) for k in range(dim)
    ]  # the systematic, generalizing residual
    xs: list[float] = []
    targets: list[float] = []
    for _ in range(n):
        for k in range(dim):
            t = rng.uniform(-0.5, 0.5)
            targets.append(round(t, 6))
            xs.append(round(t + bias[k] + rng.uniform(-0.01, 0.01), 6))
    return {"count": n, "x": xs, "target": targets}


@needs_node
def test_system_composed_probe_drives_the_real_server_path() -> None:
    """The composition wiring works end-to-end on synthetic pairs, no dataset required."""
    from lensemble.eval.lewm_system_probe import run_system_composed_probe

    dim = 192
    participants = [_bias_correctable_pairs(n=24, dim=dim, seed=s) for s in (11, 22)]
    validation = _bias_correctable_pairs(n=16, dim=dim, seed=99)
    evidence = run_system_composed_probe(
        participants=participants,
        validation=validation,
        checkpoint=_synthetic_manifest()["checkpoint"],
        manifest=_synthetic_manifest(),
        rounds=3,
        steps_per_round=20,
        batch_size=16,
        seed=7,
        dim=dim,
    )
    # the artifact came from the real aggregation path (hash-chained revisions, clean claim audit)
    assert evidence["role"] == "system-composed-headline"
    assert "system-composed: server aggregation path" in evidence["protocol"]
    assert evidence["claimAuditViolations"] == 0
    assert evidence["serverOffsetParameterCount"] == 12512
    assert len(evidence["modelRevisionChain"]) == 3
    assert all(rev.startswith("lewmrev-") for rev in evidence["modelRevisionChain"])
    result = evidence["result"]
    # a learnable systematic bias must read as a clean, non-collapsed improvement
    assert result["verdict"] == "improved", result
    assert result["collapseRisk"] is False
    assert result["adaptedMse"] < result["baselineMse"]
    # held-out collapse diagnostics are present for baseline AND adapted (#328)
    diag = result["diagnostics"]
    assert diag["baseline"]["latentStdMean"] > 0
    assert diag["adapted"]["effectiveRank"] > 0
    assert evidence["passes"] is True


def test_system_composed_headline_evidence_is_pinned() -> None:
    committed = EVIDENCE / "lewm_tworooms_system_probe.json"
    if not committed.is_file():
        pytest.skip(
            "system-composed evidence not generated yet (scripts/lewm_system_probe.py)"
        )
    evidence = json.loads(committed.read_text())
    assert evidence["schema"] == "lewm-federated-probe/1"
    assert evidence["role"] == "system-composed-headline"
    assert "system-composed: server aggregation path" in evidence["protocol"]
    assert evidence["claimAuditViolations"] == 0
    assert len(evidence["checkpoint"]["revision"]) == 40
    result = evidence["result"]
    assert result["verdict"] == "improved"
    assert result["collapseRisk"] is False
    assert result["relativeImprovement"] > 0.02
    assert result["adaptedMse"] < result["baselineMse"]
    assert evidence["passes"] is True
    # #328: the headline carries held-out collapse diagnostics proving bias-correction, not collapse
    diag = result["diagnostics"]
    base_std = diag["baseline"]["latentStdMean"]
    adapted_std = diag["adapted"]["latentStdMean"]
    assert adapted_std >= 0.7 * base_std, (
        "adapted held-out latent std must not be collapsed"
    )
    assert diag["adapted"]["effectiveRank"] >= 0.7 * diag["baseline"]["effectiveRank"]


def test_offline_cross_check_is_labelled_and_agrees_with_headline() -> None:
    offline_path = EVIDENCE / "lewm_tworooms_probe_check.json"
    system_path = EVIDENCE / "lewm_tworooms_system_probe.json"
    if not (offline_path.is_file() and system_path.is_file()):
        pytest.skip("probe evidence not generated yet")
    offline = json.loads(offline_path.read_text())
    system = json.loads(system_path.read_text())
    # the offline harness is now explicitly the math cross-check, not the headline
    assert offline["role"] == "offline-math-cross-check"
    assert "system-composed" in offline["protocol"]
    # same protocol, same seed -> the system path must reproduce the offline math (server rounds
    # the adapter state to 8 decimals, so allow a tiny tolerance)
    if offline["seed"] == system["seed"]:
        assert (
            abs(
                offline["result"]["relativeImprovement"]
                - system["result"]["relativeImprovement"]
            )
            < 1e-4
        )


def test_seed_sweep_evidence_is_seed_robust() -> None:
    committed = EVIDENCE / "lewm_tworooms_probe_seedsweep.json"
    if not committed.is_file():
        pytest.skip(
            "seed-sweep evidence not generated yet (scripts/lewm_probe_seedsweep.py)"
        )
    evidence = json.loads(committed.read_text())
    assert evidence["schema"] == "lewm-federated-probe-seedsweep/1"
    dist = evidence["distribution"]
    assert dist["count"] >= 3
    assert dist["allSeedsImproved"] is True
    assert dist["anySeedCollapseRisk"] is False
    assert dist["worstCaseRelativeImprovement"] > 0
    assert len(evidence["draws"]) == dist["count"]
    for draw in evidence["draws"]:
        assert draw["collapseRisk"] is False
        assert "relativeImprovement" in draw
    # the headline cites the WORST case, not the best
    assert str(dist["worstCaseSeed"]) in evidence["headline"]
    assert evidence["passes"] is True


@pytest.mark.skipif(
    not os.environ.get("LENSEMBLE_LEWM_H5"),
    reason="dataset-gated: set LENSEMBLE_LEWM_H5 to the tworoom.h5 path to run the regeneration check",
)
@needs_node
def test_system_probe_regenerates_within_tolerance() -> None:
    """Nightly/dataset-gated: regenerate the headline and confirm it has not silently drifted."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("h5py")
    from lensemble.demo.server import load_lewm_manifest
    from lensemble.eval.lewm_system_probe import run_system_composed_probe
    from lensemble.eval.lewm_tworooms_probe_pairs import (
        DEFAULT_MODEL_DIR,
        build_probe_split,
    )

    committed = json.loads((EVIDENCE / "lewm_tworooms_system_probe.json").read_text())
    seed = committed["seed"]
    split = build_probe_split(
        h5_path=Path(os.environ["LENSEMBLE_LEWM_H5"]),
        model_dir=DEFAULT_MODEL_DIR,
        seed=seed,
    )
    manifest = load_lewm_manifest(str(DEFAULT_MODEL_DIR / "manifest.json"))
    assert manifest is not None
    fresh = run_system_composed_probe(
        participants=split.participants,
        validation=split.validation,
        checkpoint=split.checkpoint,
        manifest=manifest,
        seed=seed,
        dim=split.dim,
    )
    assert fresh["result"]["verdict"] == "improved"
    assert fresh["result"]["collapseRisk"] is False
    # the committed headline must match a fresh regeneration within tolerance (no silent drift)
    assert (
        abs(
            fresh["result"]["relativeImprovement"]
            - committed["result"]["relativeImprovement"]
        )
        < 1e-3
    )
