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
    Phase3CompletedControlInput,
    Phase3ControlGaugeValue,
    build_phase3_eval_report,
    load_phase3_eval_report,
    load_phase3_long_run_evidence,
    parse_phase3_eval_report,
    phase3_run_manifest_hash_from_report,
)

_LONG_RUN_REPORT = Path("docs/evidence/phase3_long_run_smoke_report.json")


def _control_input(
    control_role: str,
    *,
    frame_drift_deg: float,
    effective_rank: float,
) -> Phase3CompletedControlInput:
    return Phase3CompletedControlInput(
        control_role=control_role,  # type: ignore[arg-type]
        task_env_id=f"phase3://consortium-control-{control_role}",
        repo=f"abdelstark/lensemble-phase3-consortium-{control_role}",
        revision="0" * 40,
        checkpoint_hash="a" * 64,
        config_hash="b" * 64,
        run_manifest_hash="c" * 64,
        seed=0,
        source_label=f"Phase 3 {control_role} control run report",
        source_uri=f"abdelstark/lensemble-phase3-consortium-{control_role}",
        source_report_sha256="d" * 64,
        source_schema_name="phase3_long_run_report",
        source_schema_version=1,
        gauges=(
            Phase3ControlGaugeValue(
                metric="latent_frame_drift_deg",
                value=frame_drift_deg,
                notes="round-0 inter-participant latent frame-drift",
            ),
            Phase3ControlGaugeValue(
                metric="effective_rank",
                value=effective_rank,
                notes="round-0 global-representation effective rank",
            ),
        ),
        note=f"matched {control_role} control bound to published run hashes",
    )


def _completed_controls() -> tuple[Phase3CompletedControlInput, ...]:
    return (
        _control_input("naive-fedavg", frame_drift_deg=180.0, effective_rank=1.08),
        _control_input(
            "fork-a-frozen-encoder", frame_drift_deg=0.0, effective_rank=2.39
        ),
        _control_input("local-only", frame_drift_deg=180.0, effective_rank=120.32),
    )


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


def test_phase3_eval_report_flips_controls_to_completed() -> None:
    report = build_phase3_eval_report(
        _LONG_RUN_REPORT, completed_controls=_completed_controls()
    )

    assert parse_phase3_eval_report(report.model_dump(mode="json")) == report
    # All three previously-blocked controls are now completed; none remain blocked.
    assert report.blocked_controls == ()
    completed_roles = {row.control_role for row in report.metric_rows}
    assert completed_roles == {
        "anchored-federation",
        "naive-fedavg",
        "fork-a-frozen-encoder",
        "local-only",
    }
    # 4 lifecycle rows for anchored + 2 gauge rows for each of 3 controls.
    assert len(report.metric_rows) == 10

    by_role: dict[str, dict[str, float]] = {}
    for row in report.metric_rows:
        if row.control_role == "anchored-federation":
            continue
        by_role.setdefault(row.control_role, {})[row.metric] = row.value
        assert len(row.checkpoint_hash) == 64
        assert len(row.config_hash) == 64
        assert len(row.run_manifest_hash) == 64
        assert row.source_report_sha256 in {
            artifact.sha256 for artifact in report.source_artifacts
        }
    assert by_role["naive-fedavg"]["latent_frame_drift_deg"] == 180.0
    assert by_role["fork-a-frozen-encoder"]["latent_frame_drift_deg"] == 0.0
    assert by_role["local-only"]["effective_rank"] == 120.32
    # Each control contributes a source-artifact reference (1 smoke + 3 controls).
    assert len(report.source_artifacts) == 4


def test_phase3_eval_report_rejects_anchored_as_completed_control() -> None:
    with pytest.raises(ConfigError):
        build_phase3_eval_report(
            _LONG_RUN_REPORT,
            completed_controls=(
                _control_input(
                    "anchored-federation",
                    frame_drift_deg=48.97,
                    effective_rank=2.23,
                ),
            ),
        )


def test_phase3_eval_report_rejects_duplicate_completed_control() -> None:
    dup = _control_input("naive-fedavg", frame_drift_deg=180.0, effective_rank=1.08)
    with pytest.raises(ConfigError):
        build_phase3_eval_report(_LONG_RUN_REPORT, completed_controls=(dup, dup))


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

    # 4 anchored lifecycle rows + 2 gauge rows for each of the 3 flipped controls.
    assert len(report.metric_rows) == 10
    # All three previously-blocked controls are now completed; nothing remains blocked.
    assert report.blocked_controls == ()
    completed_roles = {row.control_role for row in report.metric_rows}
    assert {"local-only", "naive-fedavg", "fork-a-frozen-encoder"} <= completed_roles

    # Each completed control gauge row is bound to a published source artifact.
    source_hashes = {artifact.sha256 for artifact in report.source_artifacts}
    for row in report.metric_rows:
        if row.control_role == "anchored-federation":
            continue
        assert row.metric in {"latent_frame_drift_deg", "effective_rank"}
        assert len(row.checkpoint_hash) == 64
        assert len(row.config_hash) == 64
        assert len(row.run_manifest_hash) == 64
        assert row.source_report_sha256 in source_hashes

    # The gauge contrast is stated honestly as the round-0 measurement.
    assert "round-0 48.97 deg vs naive-FedAvg 180 deg" in report.model_card_eval_text
    assert "collapses over rounds" in report.model_card_eval_text
    assert "paper-scale LeWorldModel performance" in report.claim_boundary
