"""RFC-0017 dynamic-env downstream report contract."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lensemble.errors import ConfigError, EvaluationError, SchemaVersionMismatch
from lensemble.eval import (
    DYNAMIC_ENV_DOWNSTREAM_REPORT_SCHEMA_VERSION,
    DynamicEnvCheckpointRef,
    DynamicEnvControlReport,
    DynamicEnvDownstreamEvalReport,
    parse_dynamic_env_downstream_eval_report,
)


def _control() -> DynamicEnvControlReport:
    return DynamicEnvControlReport(
        label="federated",
        checkpoint=DynamicEnvCheckpointRef(
            repo_id="abdelstark/lensemble-dynamic-checkpoint",
            revision="abc123",
            checkpoint_hash="a" * 64,
        ),
        state_probe_r2=0.72,
        success_rate=0.25,
        skill_vs_identity=1.8,
        latent_goal_success_rate=0.4,
        effective_rank=12.0,
        metric_boundary="state_probe_r2 is binding; latent metrics are supporting and gameable",
    )


def _report() -> DynamicEnvDownstreamEvalReport:
    return DynamicEnvDownstreamEvalReport(
        schema_version=DYNAMIC_ENV_DOWNSTREAM_REPORT_SCHEMA_VERSION,
        generated_at=datetime(2026, 6, 9, tzinfo=timezone.utc),
        task_env_id="kinematic://swipe-dot",
        held_out_data_ref="synthetic-dynamic://swipe-dot?seed=1&n_episodes=8&steps=64&image_size=48",
        controls=(_control(),),
        claim_boundary=(
            "synthetic control env; state_probe_r2 is the binding ground-truth metric; "
            "latent-MPC and skill metrics are gameable supporting signals; no paper-scale robotics claim"
        ),
        source_report_uri="local://dynamic-env-report",
    )


def test_dynamic_env_report_round_trips_without_old_phase3_blocker_lock() -> None:
    report = _report()
    parsed = parse_dynamic_env_downstream_eval_report(report.model_dump(mode="json"))
    assert parsed == report
    assert "#96" not in report.claim_boundary
    assert "#244" not in report.claim_boundary
    assert report.controls[0].state_probe_r2 == pytest.approx(0.72)
    assert report.controls[0].success_rate_role == "reported_non_binding"


def test_dynamic_env_report_rejects_future_schema_first() -> None:
    with pytest.raises(SchemaVersionMismatch):
        parse_dynamic_env_downstream_eval_report(
            {"schema_version": DYNAMIC_ENV_DOWNSTREAM_REPORT_SCHEMA_VERSION + 1}
        )


def test_dynamic_env_control_rejects_bad_metric_ranges() -> None:
    with pytest.raises(EvaluationError):
        DynamicEnvControlReport(
            label="bad",
            checkpoint=DynamicEnvCheckpointRef(
                repo_id="repo", revision="rev", checkpoint_hash="a" * 64
            ),
            state_probe_r2=1.01,
            success_rate=0.5,
            metric_boundary="supporting gameable",
        )
    with pytest.raises(EvaluationError):
        DynamicEnvControlReport(
            label="bad",
            checkpoint=DynamicEnvCheckpointRef(
                repo_id="repo", revision="rev", checkpoint_hash="a" * 64
            ),
            state_probe_r2=0.1,
            success_rate=1.5,
            metric_boundary="supporting gameable",
        )


def test_dynamic_env_report_requires_claim_boundary_phrases() -> None:
    raw = _report().model_dump(mode="json")
    raw["claim_boundary"] = "too vague"
    with pytest.raises(ConfigError):
        parse_dynamic_env_downstream_eval_report(raw)


def test_dynamic_env_control_requires_gameable_supporting_boundary() -> None:
    with pytest.raises(ConfigError):
        DynamicEnvControlReport(
            label="bad",
            checkpoint=DynamicEnvCheckpointRef(
                repo_id="repo", revision="rev", checkpoint_hash="a" * 64
            ),
            state_probe_r2=0.1,
            success_rate=0.5,
            metric_boundary="state_probe_r2 only",
        )
