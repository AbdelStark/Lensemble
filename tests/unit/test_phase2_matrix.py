"""Phase 2 evidence matrix contract."""

from __future__ import annotations

import json
import subprocess
import sys

from lensemble.eval import default_phase2_matrix, render_phase2_matrix_markdown


def test_default_phase2_matrix_covers_tracker_children() -> None:
    rows = default_phase2_matrix()
    assert {row.issue for row in rows} == {201, 202, 203, 204, 205, 206}
    assert {row.slug for row in rows} == {
        "phase2-data-contract",
        "phase2-gpu-federated-run",
        "phase2-downstream-eval",
        "phase2-baselines-curves",
        "phase2-evidence-bundle",
        "phase2-roadmap-docs",
    }


def test_phase2_rows_have_falsifiers_and_artifact_gates() -> None:
    for row in default_phase2_matrix():
        assert row.falsifying_result
        assert row.artifact_gate
        assert row.expected_result != row.falsifying_result


def test_render_phase2_matrix_markdown_uses_reviewer_columns() -> None:
    markdown = render_phase2_matrix_markdown()
    assert markdown.startswith("| Issue | Claim | Metric | Dataset/env |")
    for issue in (201, 202, 203, 204, 205, 206):
        assert f"#{issue}" in markdown
    assert "Falsifying result" in markdown
    assert "Artifact gate" in markdown


def test_phase2_matrix_script_outputs_json() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/phase2_matrix.py", "--format", "json"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = json.loads(result.stdout)
    assert isinstance(rows, list)
    assert rows[0]["issue"] == 201
    assert rows[-1]["issue"] == 203
