"""Phase 2 baselines/curves report contract."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lensemble.errors import ConfigError, SchemaVersionMismatch
from lensemble.eval import (
    EvalReport,
    Phase2CheckpointRef,
    Phase2ClaimCurveInput,
    Phase2DownstreamEvalReport,
    Phase2EvalTask,
    Phase2PlannerBudget,
    Phase2SourceReportRef,
    build_phase2_baselines_curves_report,
    parse_claim_mvp_report,
    parse_phase2_baselines_curves_report,
)


def _claim_report(*, lambda_anc: float, config_hash: str, final_hash: str):
    roots = {
        "phase2-so100-a": "a" * 64,
        "phase2-so100-b": "b" * 64,
    }
    return parse_claim_mvp_report(
        {
            "schema_version": 2,
            "claim": "federated-leworldmodel-claim-mvp",
            "config_hash": config_hash,
            "wmcp_version": "wmcp-test",
            "round_state": "closed",
            "committed_rounds": 2,
            "final_global_hash": final_hash,
            "objective_target_stop_gradient": False,
            "lambda_sig": 0.1,
            "lambda_anc": lambda_anc,
            "participant_count_configured": 2,
            "participants": [
                {
                    "participant_id": participant_id,
                    "data_source": f"lerobot-h5:///data/{participant_id}.h5",
                    "data_format": "lerobot-h5",
                    "dataset_root": root,
                    "update_l2_norm": 0.5,
                    "clipped": True,
                    "quantized": False,
                }
                for participant_id, root in roots.items()
            ],
            "ledger_records": [],
            "metrics": {
                "val_pred": 1.25,
                "val_sigreg": 0.2,
                "effective_rank": 3.0,
                "frame_drift_deg": 12.0,
                "run_manifest_hash": "c" * 64,
            },
            "round_metrics": [
                {
                    "round_index": 0,
                    "round_state": "closed",
                    "global_model_hash": "d" * 64,
                    "participant_ids": ["phase2-so100-a", "phase2-so100-b"],
                    "dataset_roots": roots,
                    "update_l2_norms": {
                        "phase2-so100-a": 0.4,
                        "phase2-so100-b": 0.45,
                    },
                },
                {
                    "round_index": 1,
                    "round_state": "closed",
                    "global_model_hash": final_hash,
                    "participant_ids": ["phase2-so100-a", "phase2-so100-b"],
                    "dataset_roots": roots,
                    "update_l2_norms": {
                        "phase2-so100-a": 0.5,
                        "phase2-so100-b": 0.55,
                    },
                },
            ],
            "publication": {
                "dataset_repos": [],
                "checkpoint_repo": "test/checkpoint",
                "checkpoint_path": "artifacts",
                "pushed": True,
                "dry_run": False,
                "blocker": None,
            },
            "created_at": "2026-06-05T00:00:00Z",
        }
    )


def _downstream_report(checkpoint_hash: str) -> Phase2DownstreamEvalReport:
    return Phase2DownstreamEvalReport(
        schema_version=1,
        generated_at=datetime(2026, 6, 5, tzinfo=timezone.utc),
        checkpoint=Phase2CheckpointRef(
            repo_id="test/checkpoint",
            revision="rev",
            artifact_path="artifacts/round-00002",
            checkpoint_hash=checkpoint_hash,
            training_job_id="train-job",
            training_job_url="https://example.test/jobs/train-job",
            code_sha="1" * 40,
            train_config_hash="1" * 64,
        ),
        eval_report=EvalReport(
            schema_version=1,
            checkpoint_hash=checkpoint_hash,
            env_id="synthetic://toy",
            planner="icem",
            success_rate=0.5,
            planning_samples=2,
            time_per_action_ms=7.0,
            effective_dim=1.1,
            probe_accuracy=None,
            run_manifest_hash="2" * 64,
        ),
        task=Phase2EvalTask(
            env_id="synthetic://toy",
            task_scale="tiny test",
            held_out_policy="synthetic held-out seeds",
            goal_policy="fixed synthetic goal",
            n_episodes=2,
            action_kind="continuous",
            action_units=("u", "u"),
        ),
        planner_budget=Phase2PlannerBudget(
            planner="icem",
            horizon=1,
            planning_samples=2,
            planner_iterations=1,
            action_dim=2,
            action_low=(-1.0, -1.0),
            action_high=(1.0, 1.0),
            action_clipping="none",
        ),
        eval_config_hash="3" * 64,
        eval_command="cmd",
        claim_boundary="test boundary",
    )


def _source_ref(label: str, schema_name: str, sha: str) -> Phase2SourceReportRef:
    return Phase2SourceReportRef(
        label=label,
        schema_name=schema_name,  # type: ignore[arg-type]
        schema_version=2 if schema_name == "claim_mvp_report" else 1,
        uri=f"hf://models/test/{label}.json",
        sha256=sha,
        repo_id="test/repo",
        repo_type="model",
        revision="rev",
        path_in_repo=f"{label}.json",
        job_id=f"{label}-job",
        job_url=f"https://example.test/jobs/{label}",
    )


def _build_report(with_naive: bool = True):
    final_hash = "f" * 64
    anchored = Phase2ClaimCurveInput(
        run_role="anchored-federation",
        run_label="anchored federation",
        report=_claim_report(
            lambda_anc=0.01, config_hash="1" * 64, final_hash=final_hash
        ),
        source_ref=_source_ref("anchored", "claim_mvp_report", "4" * 64),
    )
    controls: list[Phase2ClaimCurveInput] = []
    if with_naive:
        controls.append(
            Phase2ClaimCurveInput(
                run_role="naive-fedavg",
                run_label="naive FedAvg",
                report=_claim_report(
                    lambda_anc=0.0, config_hash="5" * 64, final_hash="6" * 64
                ),
                source_ref=_source_ref("naive", "claim_mvp_report", "7" * 64),
                ablation_axis="lambda_anc",
            )
        )
    return build_phase2_baselines_curves_report(
        anchored=anchored,
        downstream_report=_downstream_report(final_hash),
        downstream_source_ref=_source_ref(
            "downstream", "phase2_downstream_eval_report", "8" * 64
        ),
        control_reports=tuple(controls),
    )


def test_build_phase2_curves_report_links_points_to_hashes_and_blockers() -> None:
    report = _build_report(with_naive=True)

    assert (
        parse_phase2_baselines_curves_report(report.model_dump(mode="json")) == report
    )
    assert any(point.run_role == "naive-fedavg" for point in report.curve_points)
    assert any(point.ablation_axis == "lambda_anc" for point in report.curve_points)
    assert "naive-fedavg" not in {
        item.comparison for item in report.blocked_comparisons
    }
    assert {"local-only", "centralized-pooled", "fork-a"}.issubset(
        {item.comparison for item in report.blocked_comparisons}
    )
    source_hashes = {source.sha256 for source in report.source_reports}
    for point in report.curve_points:
        assert len(point.config_hash) == 64
        assert len(point.checkpoint_hash) == 64
        assert point.source_report_sha256 in source_hashes


def test_build_phase2_curves_report_marks_missing_control_and_ablation_blocked() -> (
    None
):
    report = _build_report(with_naive=False)
    blockers = {item.comparison for item in report.blocked_comparisons}
    assert "naive-fedavg" in blockers
    assert "lambda-anc-ablation" in blockers


def test_phase2_curves_rejects_unmatched_control_dataset_roots() -> None:
    final_hash = "f" * 64
    anchored_report = _claim_report(
        lambda_anc=0.01, config_hash="1" * 64, final_hash=final_hash
    )
    control_raw = _claim_report(
        lambda_anc=0.0, config_hash="5" * 64, final_hash="6" * 64
    ).model_dump(mode="json")
    control_raw["participants"][0]["dataset_root"] = "9" * 64
    control = parse_claim_mvp_report(control_raw)

    with pytest.raises(ConfigError):
        build_phase2_baselines_curves_report(
            anchored=Phase2ClaimCurveInput(
                run_role="anchored-federation",
                run_label="anchored federation",
                report=anchored_report,
                source_ref=_source_ref("anchored", "claim_mvp_report", "4" * 64),
            ),
            downstream_report=_downstream_report(final_hash),
            downstream_source_ref=_source_ref(
                "downstream", "phase2_downstream_eval_report", "8" * 64
            ),
            control_reports=(
                Phase2ClaimCurveInput(
                    run_role="naive-fedavg",
                    run_label="naive FedAvg",
                    report=control,
                    source_ref=_source_ref("naive", "claim_mvp_report", "7" * 64),
                    ablation_axis="lambda_anc",
                ),
            ),
        )


def test_parse_phase2_curves_report_rejects_future_schema() -> None:
    with pytest.raises(SchemaVersionMismatch):
        parse_phase2_baselines_curves_report({"schema_version": 2})


def test_phase2_curves_report_script_outputs_report_json(tmp_path: Path) -> None:
    final_hash = "f" * 64
    anchored_path = tmp_path / "anchored.json"
    naive_path = tmp_path / "naive.json"
    downstream_path = tmp_path / "downstream.json"
    output = tmp_path / "curves.json"
    anchored_path.write_text(
        _claim_report(
            lambda_anc=0.01, config_hash="1" * 64, final_hash=final_hash
        ).model_dump_json(),
        encoding="utf-8",
    )
    naive_path.write_text(
        _claim_report(
            lambda_anc=0.0, config_hash="5" * 64, final_hash="6" * 64
        ).model_dump_json(),
        encoding="utf-8",
    )
    downstream_path.write_text(
        _downstream_report(final_hash).model_dump_json(),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase2_curves_report.py",
            "--anchored-claim-report",
            str(anchored_path),
            "--downstream-report",
            str(downstream_path),
            "--naive-fedavg-claim-report",
            str(naive_path),
            "--naive-fedavg-job-id",
            "naive-job",
            "--naive-fedavg-revision",
            "naive-revision",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout_report = parse_phase2_baselines_curves_report(json.loads(result.stdout))
    file_report = parse_phase2_baselines_curves_report(json.loads(output.read_text()))
    assert stdout_report == file_report
    assert any(point.run_role == "naive-fedavg" for point in file_report.curve_points)


def test_checked_in_phase2_curves_report_is_schema_valid() -> None:
    path = Path("docs/evidence/phase2_baselines_curves_report.json")
    report = parse_phase2_baselines_curves_report(json.loads(path.read_text()))
    assert report.raw_data_in_report is False
    assert any(
        point.metric == "downstream_success_rate" for point in report.curve_points
    )
    assert "centralized-pooled" in {
        item.comparison for item in report.blocked_comparisons
    }
