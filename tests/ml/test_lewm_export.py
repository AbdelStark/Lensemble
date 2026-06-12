"""Gate G2 — browser export of the TwoRooms LeWM inference graphs (#317).

Export tests need ``onnx``/``onnxscript``/``onnxruntime`` (export-time-only deps, not lensemble
runtime deps) and skip when absent — the blocking CPU gates stay download-free and torch-only. The
committed-evidence checks below run everywhere and pin the manifest contract that the browser app
and the evidence bundle rely on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from lensemble.model.lewm_checkpoint import ResolvedCheckpoint
from lensemble.model.lewm_export import (
    EXPORT_SCHEMA,
    browser_export_manifest,
    export_browser_graphs,
    onnxruntime_parity,
)
from lensemble.model.lewm_tworooms import build_lewm_tworooms

from .test_lewm_tworooms import _tiny_cfg


def _fake_resolved(tmp_path: Path) -> ResolvedCheckpoint:
    config = tmp_path / "config.json"
    weights = tmp_path / "weights.pt"
    config.write_text("{}")
    weights.write_bytes(b"\x00" * 16)
    return ResolvedCheckpoint(
        repo_id="quentinll/lewm-tworooms",
        revision="77adaae0bc31deab21c93740d1f8bb947cd0bdec",
        config_path=config,
        weights_path=weights,
        claim_grade=True,
    )


# ---------------------------------------------------------------------------
# manifest contract (no onnx required)
# ---------------------------------------------------------------------------


def test_manifest_binds_graphs_to_checkpoint(tmp_path: Path) -> None:
    model = build_lewm_tworooms(_tiny_cfg())
    paths = {}
    for name in (
        "lewm_tworooms_encoder.onnx",
        "lewm_tworooms_action.onnx",
        "lewm_tworooms_predictor.onnx",
    ):
        p = tmp_path / name
        p.write_bytes(name.encode() * 10)
        paths[name] = p
    parity = {"onnxruntimeAvailable": False, "status": "skipped", "reason": "test"}
    manifest = browser_export_manifest(
        _fake_resolved(tmp_path),
        model,
        paths,
        parity,
        opset=18,
        weights_sha256="ab" * 32,
    )
    assert manifest["schema"] == EXPORT_SCHEMA
    assert manifest["checkpoint"]["revision"] == "77adaae0bc31deab21c93740d1f8bb947cd0bdec"
    assert manifest["checkpoint"]["weightsSha256"] == "ab" * 32
    for name, path in paths.items():
        entry = manifest["files"][name]
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry["bytes"] == path.stat().st_size
        assert entry["inputs"] and entry["outputs"]
    assert manifest["parity"]["status"] == "skipped"
    assert any("not browser training" in t for t in manifest["nonClaims"])


def test_committed_export_manifest_is_claim_grade() -> None:
    committed = Path("docs/evidence/lewm_tworooms_browser_export_manifest.json")
    if not committed.is_file():
        pytest.skip("export manifest not generated yet (scripts/lewm_tworooms_export.py)")
    manifest = json.loads(committed.read_text())
    assert manifest["schema"] == EXPORT_SCHEMA
    assert manifest["checkpoint"]["repoId"] == "quentinll/lewm-tworooms"
    assert len(manifest["checkpoint"]["revision"]) == 40
    assert len(manifest["checkpoint"]["weightsSha256"]) == 64
    assert set(manifest["files"]) == {
        "lewm_tworooms_encoder.onnx",
        "lewm_tworooms_action.onnx",
        "lewm_tworooms_predictor.onnx",
    }
    for entry in manifest["files"].values():
        assert len(entry["sha256"]) == 64
        assert entry["bytes"] > 0
    # claim-grade evidence requires a real, passing parity record — not a skip
    assert manifest["parity"]["status"] == "passed"
    assert manifest["parity"]["maxAbsDiff"] <= manifest["parity"]["atol"]
    # parity must include the short-history predictor calls the browser rollout makes
    predictor_times = {
        shapes[0][1]
        for check in manifest["parity"]["checks"]
        if check["graph"] == "lewm_tworooms_predictor.onnx"
        for shapes in [check["inputShapes"]]
    }
    assert {1, 2, 3} <= predictor_times


# ---------------------------------------------------------------------------
# export + runtime parity (skip when onnx tooling is absent)
# ---------------------------------------------------------------------------


def _onnx_available() -> bool:
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
        import onnxscript  # noqa: F401
    except Exception:
        return False
    return True


needs_onnx = pytest.mark.skipif(
    not _onnx_available(), reason="onnx/onnxscript/onnxruntime not installed"
)


@needs_onnx
def test_tiny_export_parity(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = build_lewm_tworooms(_tiny_cfg())
    paths = export_browser_graphs(model, tmp_path)
    parity = onnxruntime_parity(model, paths, require=True)
    assert parity["status"] == "passed"
    assert parity["maxAbsDiff"] <= parity["atol"]
    # dynamic time axis: short-history predictor calls and longer action sequences both checked
    graphs = {c["graph"] for c in parity["checks"]}
    assert graphs == set(paths)


@needs_onnx
def test_export_is_deterministic(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = build_lewm_tworooms(_tiny_cfg())
    first = export_browser_graphs(model, tmp_path / "a")
    second = export_browser_graphs(model, tmp_path / "b")
    for name in first:
        h1 = hashlib.sha256(first[name].read_bytes()).hexdigest()
        h2 = hashlib.sha256(second[name].read_bytes()).hexdigest()
        assert h1 == h2, f"non-deterministic export for {name}"


@needs_onnx
def test_parity_fails_on_corrupted_graph(tmp_path: Path) -> None:
    """A graph whose weights were tampered with must not pass parity."""
    torch.manual_seed(3)
    model = build_lewm_tworooms(_tiny_cfg())
    paths = export_browser_graphs(model, tmp_path)
    with torch.no_grad():
        # random (non-constant) tamper — the projector input is LayerNormed (zero-mean), so a
        # constant weight shift would cancel exactly
        weight = model.projector.net[0].weight
        weight.add_(torch.randn_like(weight) * 0.05)
    parity = onnxruntime_parity(model, paths, require=True)
    assert parity["status"] == "failed"
