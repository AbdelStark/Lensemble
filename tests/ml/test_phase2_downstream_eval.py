"""Phase 2 downstream eval report contract."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lensemble.artifacts import model_arch_from_config, save_checkpoint
from lensemble.config import LensembleConfig
from lensemble.contracts import WMCP_VERSION
from lensemble.errors import ConfigError, EvaluationError, SchemaVersionMismatch
from lensemble.eval import (
    Phase2CheckpointRef,
    build_phase2_downstream_eval_report,
    evaluate,
    parse_phase2_downstream_eval_report,
    phase2_eval_config_from_checkpoint,
)
from lensemble.model import build_encoder, build_predictor


def _tiny_cfg() -> LensembleConfig:
    base = LensembleConfig()
    model = dataclasses.replace(
        base.model,
        latent_dim=8,
        num_tokens=4,
        predictor_depth=1,
        predictor_width=8,
        num_frames=2,
        tubelet=2,
        image_size=4,
        patch_size=2,
        depth=1,
        num_heads=2,
        in_channels=3,
        wmcp_version=WMCP_VERSION,
    )
    eval_cfg = dataclasses.replace(
        base.eval,
        env_id="synthetic://toy",
        planner="icem",
        planning_samples=4,
        horizon=1,
    )
    return dataclasses.replace(base, model=model, eval=eval_cfg, run_mode="eval")


def _save_checkpoint(cfg: LensembleConfig, path: Path) -> str:
    encoder = build_encoder(cfg)
    predictor = build_predictor(cfg)
    weights = {
        **{f"encoder.{name}": tensor for name, tensor in encoder.state_dict().items()},
        **{
            f"predictor.{name}": tensor
            for name, tensor in predictor.state_dict().items()
        },
    }
    return save_checkpoint(
        path,
        weights,
        wmcp_version=WMCP_VERSION,
        round_index=3,
        config_hash="a" * 64,
        parent_hash="b" * 64,
        model_arch=model_arch_from_config(cfg),
    )


def _checkpoint_ref(checkpoint_hash: str) -> Phase2CheckpointRef:
    return Phase2CheckpointRef(
        repo_id="test/phase2-checkpoint",
        revision="test-revision",
        artifact_path="artifacts/round-00003",
        checkpoint_hash=checkpoint_hash,
        training_job_id="job-1",
        training_job_url="https://example.test/jobs/job-1",
        code_sha="c" * 40,
        train_config_hash="d" * 64,
    )


def test_phase2_eval_config_matches_self_describing_checkpoint(tmp_path: Path) -> None:
    source_cfg = _tiny_cfg()
    checkpoint_hash = _save_checkpoint(source_cfg, tmp_path / "ckpt")

    cfg = phase2_eval_config_from_checkpoint(
        tmp_path / "ckpt",
        expected_checkpoint_hash=checkpoint_hash,
        env_id="synthetic://toy",
        planning_samples=4,
        horizon=1,
    )

    assert cfg.model.latent_dim == 8
    assert cfg.model.num_tokens == 4
    assert cfg.model.predictor_depth == 1
    assert cfg.model.predictor_width == 8
    assert cfg.model.image_size == 4
    assert cfg.eval.env_id == "synthetic://toy"


def test_build_phase2_downstream_eval_report_is_schema_valid(tmp_path: Path) -> None:
    source_cfg = _tiny_cfg()
    checkpoint_hash = _save_checkpoint(source_cfg, tmp_path / "ckpt")
    cfg = phase2_eval_config_from_checkpoint(
        tmp_path / "ckpt",
        expected_checkpoint_hash=checkpoint_hash,
        env_id="synthetic://toy",
        planning_samples=4,
        horizon=1,
    )

    report = build_phase2_downstream_eval_report(
        tmp_path / "ckpt",
        checkpoint_ref=_checkpoint_ref(checkpoint_hash),
        cfg=cfg,
        eval_command="uv run --extra dev python scripts/phase2_eval_checkpoint.py ...",
        task_scale="tiny synthetic test",
        held_out_policy="synthetic seeds are not training data",
        goal_policy="fixed synthetic goal seed 7919",
        action_clipping="none; bounds recorded",
        source_report_uri="hf://models/test/report.json",
        claim_boundary="test claim boundary",
        generated_at=datetime(2026, 6, 5, tzinfo=timezone.utc),
    )

    assert report.eval_report.success_rate == 0.5
    assert report.eval_report.checkpoint_hash == checkpoint_hash
    assert report.task.env_id == "synthetic://toy"
    assert report.task.n_episodes == 4
    assert report.task.raw_data_in_report is False
    assert report.planner_budget.action_dim == 2
    assert report.planner_budget.action_low == (-1.0, -1.0)
    assert report.planner_budget.action_high == (1.0, 1.0)
    assert "none" in report.planner_budget.action_clipping
    assert len(report.eval_config_hash) == 64
    assert parse_phase2_downstream_eval_report(report.model_dump(mode="json")) == report


def test_build_phase2_downstream_eval_report_records_explicit_episode_budget(
    tmp_path: Path,
) -> None:
    source_cfg = _tiny_cfg()
    checkpoint_hash = _save_checkpoint(source_cfg, tmp_path / "ckpt")
    cfg = phase2_eval_config_from_checkpoint(
        tmp_path / "ckpt",
        expected_checkpoint_hash=checkpoint_hash,
        env_id="synthetic://toy",
        planning_samples=4,
        horizon=1,
    )

    report = build_phase2_downstream_eval_report(
        tmp_path / "ckpt",
        checkpoint_ref=_checkpoint_ref(checkpoint_hash),
        cfg=cfg,
        eval_command="cmd",
        task_scale="tiny synthetic test",
        held_out_policy="synthetic seeds are not training data",
        goal_policy="fixed synthetic goal seed 7919",
        action_clipping="none",
        num_episodes=2,
        planner_iterations=1,
        claim_boundary="test claim boundary",
    )

    assert report.task.n_episodes == 2
    assert report.eval_report.success_rate == 0.5
    assert report.planner_budget.planner_iterations == 1


def test_evaluate_rejects_non_positive_episode_count(tmp_path: Path) -> None:
    cfg = _tiny_cfg()
    _save_checkpoint(cfg, tmp_path / "ckpt")

    with pytest.raises(EvaluationError):
        evaluate(tmp_path / "ckpt", env_id="synthetic://toy", cfg=cfg, num_episodes=1)
    with pytest.raises(EvaluationError):
        evaluate(tmp_path / "ckpt", env_id="synthetic://toy", cfg=cfg, planner_iters=0)


def test_phase2_downstream_report_rejects_mismatched_checkpoint_hash(
    tmp_path: Path,
) -> None:
    source_cfg = _tiny_cfg()
    checkpoint_hash = _save_checkpoint(source_cfg, tmp_path / "ckpt")
    cfg = phase2_eval_config_from_checkpoint(
        tmp_path / "ckpt",
        expected_checkpoint_hash=checkpoint_hash,
        env_id="synthetic://toy",
        planning_samples=4,
        horizon=1,
    )
    bad_ref = _checkpoint_ref("e" * 64)

    with pytest.raises(ConfigError):
        build_phase2_downstream_eval_report(
            tmp_path / "ckpt",
            checkpoint_ref=bad_ref,
            cfg=cfg,
            eval_command="cmd",
            task_scale="tiny synthetic test",
            held_out_policy="synthetic seeds are not training data",
            goal_policy="fixed synthetic goal seed 7919",
            action_clipping="none",
            claim_boundary="test claim boundary",
        )


def test_parse_phase2_downstream_report_rejects_future_schema() -> None:
    with pytest.raises(SchemaVersionMismatch):
        parse_phase2_downstream_eval_report({"schema_version": 2})


def test_phase2_eval_checkpoint_script_outputs_report_json(tmp_path: Path) -> None:
    source_cfg = _tiny_cfg()
    checkpoint_hash = _save_checkpoint(source_cfg, tmp_path / "ckpt")
    output = tmp_path / "phase2_downstream_eval_report.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase2_eval_checkpoint.py",
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
            "--output",
            str(output),
            "--expected-checkpoint-hash",
            checkpoint_hash,
            "--checkpoint-repo",
            "test/phase2-checkpoint",
            "--checkpoint-revision",
            "test-revision",
            "--checkpoint-artifact-path",
            "artifacts/round-00003",
            "--env-id",
            "synthetic://toy",
            "--planning-samples",
            "4",
            "--horizon",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout_report = parse_phase2_downstream_eval_report(json.loads(result.stdout))
    file_report = parse_phase2_downstream_eval_report(json.loads(output.read_text()))
    assert stdout_report == file_report
    assert file_report.eval_report.checkpoint_hash == checkpoint_hash


def test_checked_in_phase2_downstream_report_is_schema_valid() -> None:
    path = Path("docs/evidence/phase2_downstream_eval_report.json")
    report = parse_phase2_downstream_eval_report(json.loads(path.read_text()))
    assert report.eval_report.env_id == "synthetic://toy"
    assert report.eval_report.success_rate == 0.5
    assert report.task.raw_data_in_report is False
