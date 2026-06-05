"""Phase 3 eval report and matched-control contract (#228)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lensemble.errors import ConfigError, SchemaVersionMismatch
from lensemble.eval import (
    PHASE3_EVAL_REPORT_SCHEMA_VERSION,
    build_phase3_eval_report,
    load_phase3_eval_report,
    load_phase3_long_run_evidence,
    parse_phase3_eval_report,
    phase3_run_manifest_hash_from_report,
)

_LONG_RUN_REPORT = Path("docs/evidence/phase3_long_run_smoke_report.json")


def test_phase3_eval_report_links_metrics_to_hashes_and_blockers() -> None:
    report = build_phase3_eval_report(_LONG_RUN_REPORT)

    assert report.schema_version == PHASE3_EVAL_REPORT_SCHEMA_VERSION
    assert parse_phase3_eval_report(report.model_dump(mode="json")) == report
    assert {row.control_role for row in report.metric_rows} == {"anchored-federation"}
    assert {
        "closed_round_completion_rate",
        "participant_submission_rate",
        "secure_sum_round_rate",
        "dp_accounted_round_rate",
    } == {row.metric for row in report.metric_rows}
    assert all(row.value == 1.0 for row in report.metric_rows)
    assert all(row.task_env_id != "synthetic://toy" for row in report.metric_rows)

    blocked = {row.control_role for row in report.blocked_controls}
    assert {"local-only", "naive-fedavg", "fork-a-frozen-encoder"} == blocked
    for row in report.metric_rows:
        assert len(row.checkpoint_hash) == 64
        assert len(row.config_hash) == 64
        assert len(row.run_manifest_hash) == 64
        assert row.planner_budget.planner == "not_applicable"


def test_phase3_eval_report_rejects_missing_required_control() -> None:
    report = build_phase3_eval_report(_LONG_RUN_REPORT)
    raw = report.model_dump(mode="json")
    raw["blocked_controls"] = [
        row for row in raw["blocked_controls"] if row["control_role"] != "naive-fedavg"
    ]

    with pytest.raises(ConfigError):
        parse_phase3_eval_report(raw)


def test_phase3_eval_report_model_card_text_preserves_claim_boundary() -> None:
    report = build_phase3_eval_report(_LONG_RUN_REPORT)

    assert "Public task-scale SO-100 downstream evaluation remains blocked" in (
        report.model_card_eval_text
    )
    assert "must not be described as completed robotics performance" in (
        report.model_card_eval_text
    )
    assert "paper-scale LeWorldModel performance" in report.claim_boundary


def test_phase3_eval_report_reconstructs_run_manifest_hash() -> None:
    long_run = load_phase3_long_run_evidence(_LONG_RUN_REPORT)
    report = build_phase3_eval_report(_LONG_RUN_REPORT)
    expected = phase3_run_manifest_hash_from_report(long_run)

    assert {row.run_manifest_hash for row in report.metric_rows} == {expected}


def test_parse_phase3_eval_report_gates_future_schema_first() -> None:
    with pytest.raises(SchemaVersionMismatch):
        parse_phase3_eval_report(
            {"schema_version": PHASE3_EVAL_REPORT_SCHEMA_VERSION + 1}
        )


def test_phase3_eval_report_script_generates_and_validates(tmp_path: Path) -> None:
    output = tmp_path / "phase3_eval_report.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase3_eval_report.py",
            "--long-run-report",
            str(_LONG_RUN_REPORT),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote" in result.stdout
    report = load_phase3_eval_report(output)
    assert len(report.metric_rows) == 4

    validate = subprocess.run(
        [
            sys.executable,
            "scripts/phase3_eval_report.py",
            "--validate",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "validated" in validate.stdout


def test_checked_in_phase3_eval_report_is_schema_valid() -> None:
    path = Path("docs/evidence/phase3_eval_report.json")
    report = parse_phase3_eval_report(json.loads(path.read_text()))

    assert len(report.metric_rows) == 4
    assert {row.control_role for row in report.blocked_controls} == {
        "local-only",
        "naive-fedavg",
        "fork-a-frozen-encoder",
    }
