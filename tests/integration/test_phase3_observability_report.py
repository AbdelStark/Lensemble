"""Phase 3 consortium observability/dropout report (#229)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lensemble.errors import ConfigError, SchemaVersionMismatch
from lensemble.federation import (
    PHASE3_OBSERVABILITY_REPORT_SCHEMA_VERSION,
    build_phase3_observability_report,
    load_phase3_observability_report,
    parse_phase3_observability_report,
    to_phase3_observability_report_json,
    validate_phase3_observability_redaction,
    write_phase3_observability_report,
)

_LONG_RUN_REPORT = Path("docs/evidence/phase3_long_run_smoke_report.json")
_EVAL_REPORT = Path("docs/evidence/phase3_eval_report.json")


def test_phase3_observability_report_links_metrics_and_dropout(
    tmp_path: Path,
) -> None:
    report = build_phase3_observability_report(
        long_run_report_path=_LONG_RUN_REPORT,
        eval_report_path=_EVAL_REPORT,
        run_dir=tmp_path / "run",
    )

    assert report.schema_version == PHASE3_OBSERVABILITY_REPORT_SCHEMA_VERSION
    assert parse_phase3_observability_report(report.model_dump(mode="json")) == report
    assert {participant.participant_id for participant in report.participants} == {
        "phase3-so100-a",
        "phase3-so100-b",
        "phase3-so100-c",
        "phase3-so100-d",
    }
    decision = report.dropout_decisions[0]
    assert decision.induced is True
    assert decision.outcome == "closed"
    assert decision.dropped_participant_ids == ("phase3-so100-d",)
    assert "closed_with_quorum" in decision.close_decision

    dropout_round = next(
        row for row in report.rounds if row.scenario_id == decision.scenario_id
    )
    assert dropout_round.communication.update_count == 3
    assert dropout_round.communication.estimated_update_bytes > 0
    assert dropout_round.retry_count == 0
    assert dropout_round.aggregation_backend_status == "secure_sum"
    assert dropout_round.dp_epsilon_spent is not None

    eval_links = [row for row in report.metric_links if row.metric_source == "eval"]
    training_links = [
        row for row in report.metric_links if row.metric_source == "training"
    ]
    assert len(eval_links) == 4
    assert len(training_links) == 10
    assert all(row.run_id == report.run_id for row in report.metric_links)
    assert all(row.config_hash == report.config_hash for row in report.metric_links)
    assert all(
        set(row.participant_ids)
        == {participant.participant_id for participant in report.participants}
        for row in report.metric_links
    )

    artifact_labels = {artifact.label for artifact in report.artifact_publication}
    assert {
        "phase3_long_run_report",
        "phase3_eval_report",
        "phase3_observability_report",
        "phase3_induced_dropout_trace",
    }.issubset(artifact_labels)
    assert "Issue #230 must consume" in report.final_bundle_handoff


def test_phase3_observability_report_round_trips_canonical_json(
    tmp_path: Path,
) -> None:
    report = build_phase3_observability_report(
        long_run_report_path=_LONG_RUN_REPORT,
        eval_report_path=_EVAL_REPORT,
        run_dir=tmp_path / "run",
    )
    path = write_phase3_observability_report(report, tmp_path / "report.json")

    assert load_phase3_observability_report(path) == report
    assert json.loads(to_phase3_observability_report_json(report)) == json.loads(
        path.read_text()
    )


def test_phase3_observability_redaction_rejects_private_surfaces(
    tmp_path: Path,
) -> None:
    report = build_phase3_observability_report(
        long_run_report_path=_LONG_RUN_REPORT,
        eval_report_path=_EVAL_REPORT,
        run_dir=tmp_path / "run",
    )
    raw = report.model_dump(mode="json")
    raw["artifact_publication"][0]["uri"] = "/Users/alice/private/data.h5"
    with pytest.raises(ConfigError):
        parse_phase3_observability_report(raw)

    with pytest.raises(ConfigError):
        validate_phase3_observability_redaction({"raw_actions": [0.1, 0.2]})
    with pytest.raises(ConfigError):
        validate_phase3_observability_redaction({"safe": "hf_secretToken1234"})


def test_phase3_observability_report_validates_artifact_cross_refs(
    tmp_path: Path,
) -> None:
    report = build_phase3_observability_report(
        long_run_report_path=_LONG_RUN_REPORT,
        eval_report_path=_EVAL_REPORT,
        run_dir=tmp_path / "run",
    )
    raw = report.model_dump(mode="json")
    raw["metric_links"][0]["source_report_sha256"] = "0" * 64

    with pytest.raises(ConfigError):
        parse_phase3_observability_report(raw)


def test_parse_phase3_observability_report_gates_future_schema_first() -> None:
    with pytest.raises(SchemaVersionMismatch):
        parse_phase3_observability_report(
            {"schema_version": PHASE3_OBSERVABILITY_REPORT_SCHEMA_VERSION + 1}
        )


def test_phase3_observability_report_script_generates_and_validates(
    tmp_path: Path,
) -> None:
    output = tmp_path / "phase3_observability_report.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase3_observability_report.py",
            "--long-run-report",
            str(_LONG_RUN_REPORT),
            "--eval-report",
            str(_EVAL_REPORT),
            "--run-dir",
            str(tmp_path / "run"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote" in result.stdout
    report = load_phase3_observability_report(output)
    assert len(report.dropout_decisions) == 1

    validate = subprocess.run(
        [
            sys.executable,
            "scripts/phase3_observability_report.py",
            "--validate",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "validated" in validate.stdout


def test_checked_in_phase3_observability_report_is_schema_valid() -> None:
    path = Path("docs/evidence/phase3_observability_report.json")
    report = parse_phase3_observability_report(json.loads(path.read_text()))

    assert len(report.dropout_decisions) == 1
    assert report.dropout_decisions[0].outcome == "closed"
