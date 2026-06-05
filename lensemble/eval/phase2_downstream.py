"""Phase 2 downstream-eval evidence reports.

The stable :class:`lensemble.eval.report.EvalReport` is intentionally narrow.
This module wraps it with Phase 2 provenance: the public checkpoint ref, the
held-out task policy, planner budget, action clipping rule, and claim boundary.
The wrapper is still residency-safe: it carries only scalars, hashes, ids, and
configuration counts.
"""

from __future__ import annotations

import dataclasses
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lensemble.artifacts import CheckpointHeader, verify
from lensemble.config import (
    EvalConfig,
    LensembleConfig,
    config_hash,
)
from lensemble.contracts import ActionSpec
from lensemble.errors import ConfigError, LensembleErrorCode, SchemaVersionMismatch
from lensemble.eval.harness import evaluate
from lensemble.eval.report import EvalReport
from lensemble.eval.world import resolve_env

PHASE2_DOWNSTREAM_REPORT_SCHEMA_VERSION = 1
PHASE2_SYNTHETIC_TOY_EPISODES = 4

_LAYER_RE = re.compile(r"^predictor\.blocks\.layers\.(\d+)\.")


class Phase2CheckpointRef(BaseModel):
    """Public checkpoint identity evaluated by a Phase 2 downstream report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo_id: str
    repo_type: Literal["model"] = "model"
    revision: str
    artifact_path: str
    checkpoint_hash: str
    training_job_id: str | None = None
    training_job_url: str | None = None
    code_sha: str | None = None
    train_config_hash: str | None = None


class Phase2PlannerBudget(BaseModel):
    """The explicit planner budget and action clipping rule for the eval run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    planner: Literal["cem", "icem", "mppi"]
    horizon: int = Field(gt=0)
    planning_samples: int = Field(gt=0)
    planner_iterations: int = Field(gt=0, default=4)
    action_dim: int = Field(gt=0)
    action_low: tuple[float, ...] | None = None
    action_high: tuple[float, ...] | None = None
    action_clipping: str


class Phase2EvalTask(BaseModel):
    """Residency-safe task metadata for a Phase 2 downstream eval report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    env_id: str
    task_scale: str
    held_out_policy: str
    goal_policy: str
    n_episodes: int = Field(gt=0)
    action_kind: Literal["continuous", "discrete"]
    action_units: tuple[str, ...]
    raw_data_in_report: Literal[False] = False


class Phase2DownstreamEvalReport(BaseModel):
    """A schema-validated Phase 2 wrapper around :class:`EvalReport`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    generated_at: datetime
    checkpoint: Phase2CheckpointRef
    eval_report: EvalReport
    task: Phase2EvalTask
    planner_budget: Phase2PlannerBudget
    eval_config_hash: str
    eval_command: str
    published_report_uri: str | None = None
    source_report_uri: str | None = None
    claim_boundary: str

    @model_validator(mode="after")
    def _cross_check(self) -> "Phase2DownstreamEvalReport":
        if self.schema_version != PHASE2_DOWNSTREAM_REPORT_SCHEMA_VERSION:
            raise SchemaVersionMismatch(
                f"phase2 downstream report schema_version {self.schema_version!r} "
                f"exceeds reader max {PHASE2_DOWNSTREAM_REPORT_SCHEMA_VERSION}",
                code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
                remediation="read with a build supporting this phase2 downstream report schema",
            )
        if self.checkpoint.checkpoint_hash != self.eval_report.checkpoint_hash:
            raise ConfigError(
                "checkpoint ref hash does not match embedded EvalReport checkpoint_hash",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="bind the report to the exact checkpoint directory evaluated",
            )
        if self.task.env_id != self.eval_report.env_id:
            raise ConfigError(
                "task env_id does not match embedded EvalReport env_id",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="record the resolved env id used by evaluate",
            )
        if self.planner_budget.planner != self.eval_report.planner:
            raise ConfigError(
                "planner budget does not match embedded EvalReport planner",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="record the planner family used by evaluate",
            )
        if self.planner_budget.planning_samples != self.eval_report.planning_samples:
            raise ConfigError(
                "planner budget sample count does not match embedded EvalReport",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="record the configured planning sample count used by evaluate",
            )
        return self


def parse_phase2_downstream_eval_report(
    raw: dict[str, object],
) -> Phase2DownstreamEvalReport:
    """Parse a Phase 2 downstream report, gating schema version first."""

    version = raw.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version > PHASE2_DOWNSTREAM_REPORT_SCHEMA_VERSION
    ):
        raise SchemaVersionMismatch(
            f"phase2 downstream report schema_version {version!r} exceeds reader max "
            f"{PHASE2_DOWNSTREAM_REPORT_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation="read with a build supporting this phase2 downstream report schema",
        )
    return Phase2DownstreamEvalReport.model_validate(raw)


def phase2_eval_config_from_checkpoint(
    checkpoint_dir: Path,
    *,
    env_id: str = "synthetic://toy",
    planner: Literal["cem", "icem", "mppi"] = "icem",
    planning_samples: int = 8,
    horizon: int = 2,
    root_seed: int = 0,
    expected_checkpoint_hash: str | None = None,
) -> LensembleConfig:
    """Build an eval config matching a self-describing Phase 2 checkpoint."""

    header = verify(Path(checkpoint_dir), expected_hash=expected_checkpoint_hash)
    if header.model_arch is None:
        raise ConfigError(
            "Phase 2 downstream eval requires a self-describing checkpoint header",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="evaluate a checkpoint written with header.model_arch",
        )
    predictor_depth, predictor_width, cond_dim = _predictor_dims(header)
    if cond_dim != header.model_arch.d:
        raise ConfigError(
            f"Phase 2 eval config only supports cond_dim == latent_dim, got {cond_dim} "
            f"and {header.model_arch.d}",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="extend ModelConfig with cond_dim before evaluating this checkpoint",
        )

    base = LensembleConfig()
    model = dataclasses.replace(
        base.model,
        latent_dim=header.model_arch.d,
        num_tokens=header.model_arch.num_tokens,
        predictor_depth=predictor_depth,
        predictor_width=predictor_width,
        num_frames=header.model_arch.num_frames,
        tubelet=header.model_arch.tubelet,
        image_size=header.model_arch.image_size,
        patch_size=header.model_arch.patch_size,
        depth=header.model_arch.depth,
        num_heads=header.model_arch.num_heads,
        in_channels=header.model_arch.in_channels,
        mlp_ratio=header.model_arch.mlp_ratio,
        wmcp_version=header.model_arch.wmcp_version,
    )
    eval_cfg = EvalConfig(
        env_id=env_id,
        planner=planner,
        planning_samples=int(planning_samples),
        horizon=int(horizon),
    )
    determinism = dataclasses.replace(base.determinism, root_seed=int(root_seed))
    return dataclasses.replace(
        base,
        model=model,
        eval=eval_cfg,
        determinism=determinism,
        run_mode="eval",
    )


def build_phase2_downstream_eval_report(
    checkpoint_dir: Path,
    *,
    checkpoint_ref: Phase2CheckpointRef,
    cfg: LensembleConfig,
    eval_command: str,
    task_scale: str,
    held_out_policy: str,
    goal_policy: str,
    action_clipping: str,
    num_episodes: int = PHASE2_SYNTHETIC_TOY_EPISODES,
    planner_iterations: int = 4,
    published_report_uri: str | None = None,
    source_report_uri: str | None = None,
    claim_boundary: str,
    generated_at: datetime | None = None,
) -> Phase2DownstreamEvalReport:
    """Run ``evaluate`` and wrap its ``EvalReport`` with Phase 2 metadata."""

    checkpoint_dir = Path(checkpoint_dir)
    report = evaluate(
        checkpoint_dir,
        cfg.eval.env_id,
        cfg=cfg,
        num_episodes=num_episodes,
        planner_iters=planner_iterations,
    )
    world = resolve_env(cfg.eval.env_id, cfg=cfg)
    action_spec = world.action_spec
    resolved = dataclasses.asdict(cfg)
    return Phase2DownstreamEvalReport(
        schema_version=PHASE2_DOWNSTREAM_REPORT_SCHEMA_VERSION,
        generated_at=generated_at or datetime.now(timezone.utc),
        checkpoint=checkpoint_ref,
        eval_report=report,
        task=Phase2EvalTask(
            env_id=cfg.eval.env_id,
            task_scale=task_scale,
            held_out_policy=held_out_policy,
            goal_policy=goal_policy,
            n_episodes=num_episodes,
            action_kind=action_spec.kind.value,
            action_units=tuple(action_spec.units),
        ),
        planner_budget=_planner_budget(
            cfg, action_spec, action_clipping, planner_iterations
        ),
        eval_config_hash=config_hash(resolved),
        eval_command=eval_command,
        published_report_uri=published_report_uri,
        source_report_uri=source_report_uri,
        claim_boundary=claim_boundary,
    )


def _predictor_dims(header: CheckpointHeader) -> tuple[int, int, int]:
    entries = {entry.name: entry for entry in header.tensor_manifest}
    try:
        in_proj = entries["predictor.in_proj.weight"].shape
        cond_proj = entries["predictor.cond_proj.weight"].shape
    except KeyError as exc:
        raise ConfigError(
            f"checkpoint tensor manifest is missing predictor tensor {exc.args[0]!r}",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="evaluate a checkpoint containing shared encoder and predictor weights",
        ) from exc
    if len(in_proj) != 2 or len(cond_proj) != 2:
        raise ConfigError(
            "predictor tensor manifest has malformed projection shapes",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="evaluate a valid Lensemble shared checkpoint",
        )
    layers = {
        int(match.group(1))
        for entry in header.tensor_manifest
        if (match := _LAYER_RE.match(entry.name)) is not None
    }
    if not layers:
        raise ConfigError(
            "checkpoint tensor manifest has no predictor transformer layers",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="evaluate a checkpoint written by build_predictor",
        )
    return max(layers) + 1, int(in_proj[0]), int(cond_proj[1])


def _planner_budget(
    cfg: LensembleConfig,
    action_spec: ActionSpec,
    action_clipping: str,
    planner_iterations: int,
) -> Phase2PlannerBudget:
    return Phase2PlannerBudget(
        planner=cfg.eval.planner,
        horizon=int(cfg.eval.horizon),
        planning_samples=int(cfg.eval.planning_samples),
        planner_iterations=planner_iterations,
        action_dim=int(action_spec.dim),
        action_low=tuple(action_spec.low) if action_spec.low is not None else None,
        action_high=tuple(action_spec.high) if action_spec.high is not None else None,
        action_clipping=action_clipping,
    )
