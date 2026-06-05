"""Phase 3 long-run consortium orchestration smoke (#227)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lensemble.errors import SchemaVersionMismatch
from lensemble.federation import (
    PHASE3_LONG_RUN_REPORT_SCHEMA_VERSION,
    Phase3LongRunReport,
    load_phase3_long_run_report,
    parse_phase3_long_run_report,
    run_phase3_long_run_smoke,
    to_phase3_long_run_report_json,
    write_phase3_long_run_report,
)


def test_phase3_long_run_smoke_closes_small_ci_run(tmp_path: Path) -> None:
    report = run_phase3_long_run_smoke(run_dir=tmp_path / "run", rounds=2)

    assert report.schema_version == PHASE3_LONG_RUN_REPORT_SCHEMA_VERSION
    assert report.run_shape.participant_count == 4
    assert report.run_shape.rounds == 2
    assert report.run_shape.inner_horizon == 2
    assert report.run_shape.dp_enabled is True
    assert report.run_shape.secure_aggregation_backend == "simulated"
    assert report.run_shape.secure_aggregation_threshold == 3
    assert report.closed_rounds == 2
    assert report.completed_target is True
    assert report.blockers == ()
    assert len(report.rounds) == 2
    assert len(report.participants) == 4
    assert all(participant.submitted_rounds == 2 for participant in report.participants)
    assert {
        "manifest_validated",
        "dataset_registry_validated_against_manifest",
        "participant_mounts_declared_no_raw_boundary",
        "participant_agents_preflighted",
        "participant_agents_released_updates",
        "artifact_publication_targets_declared",
    }.issubset(set(report.dry_run_checks))
    assert any(
        check.startswith("public_probe_hash_pinned:") for check in report.dry_run_checks
    )
    assert all(
        round_summary.aggregation_backend_status == "secure_sum"
        for round_summary in report.rounds
    )
    assert all(
        round_summary.dp_epsilon_spent is not None for round_summary in report.rounds
    )
    assert Path(report.run_manifest_path).exists()
    assert Path(report.trace_path).exists()
    assert Path(report.ledger_path).exists()
    assert Path(report.checkpoint_dir).exists()


def test_phase3_long_run_report_round_trips_canonical_json(tmp_path: Path) -> None:
    report = run_phase3_long_run_smoke(run_dir=tmp_path / "run", rounds=1)
    path = write_phase3_long_run_report(report, tmp_path / "report.json")

    assert load_phase3_long_run_report(path) == report
    assert Phase3LongRunReport.model_validate_json(path.read_text()) == report
    assert json.loads(to_phase3_long_run_report_json(report)) == json.loads(
        path.read_text()
    )


def test_parse_phase3_long_run_report_gates_future_schema_first(
    tmp_path: Path,
) -> None:
    raw = run_phase3_long_run_smoke(
        run_dir=tmp_path / "schema-run", rounds=1
    ).model_dump(mode="json")
    raw["schema_version"] = PHASE3_LONG_RUN_REPORT_SCHEMA_VERSION + 1
    raw["rounds"] = "not-a-round-list"

    with pytest.raises(SchemaVersionMismatch):
        parse_phase3_long_run_report(raw)
