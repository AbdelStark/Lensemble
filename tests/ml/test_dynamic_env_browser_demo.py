"""RFC-0017 browser north-star demo contracts (#289)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_dynamic_env_web_assets_document_inference_only_scope() -> None:
    html = Path("web/dynamic-env-demo/index.html").read_text(encoding="utf-8")
    app = Path("web/dynamic-env-demo/app.mjs").read_text(encoding="utf-8")
    export_script = Path("scripts/dynamic_env_onnx_export.py").read_text(
        encoding="utf-8"
    )

    assert "onnxruntime-web" in html
    assert "Scope: browser inference and JS/Canvas env-sim only" in html
    assert "In-browser training is not claimed" in html
    assert "predicted_tokens" in app
    assert "Browser scope is ONNX inference plus JS/Canvas env-sim only" in (
        export_script
    )


def test_swipe_dot_js_core_matches_kinematic_contract() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")

    snippet = """
import { renderSwipeDotRGBA, stepSwipeDot } from './web/dynamic-env-demo/swipe_dot_core.mjs';
const next = stepSwipeDot({ x: 0.95, y: 0.05 }, [1, -1], 0.18);
const rgba = renderSwipeDotRGBA(next, 48);
console.log(JSON.stringify({
  x: next.x,
  y: next.y,
  len: rgba.length,
  alpha0: rgba[3],
  centerBright: rgba.some((value, idx) => idx % 4 === 0 && value > 200),
}));
"""
    result = subprocess.run(
        [node, "--input-type=module", "-e", snippet],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"x":1' in result.stdout
    assert '"y":0' in result.stdout
    assert '"len":9216' in result.stdout
    assert '"alpha0":255' in result.stdout
    assert '"centerBright":true' in result.stdout
