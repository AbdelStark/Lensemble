"""Phase 3 evaluation plan, matched controls, and claim-boundary reports.

The Phase 3 eval package is intentionally conservative. It can bind completed
local consortium-orchestration evidence to reproducible metric rows, and it
represents task-scale downstream eval or matched controls as explicit blocked
rows when the required public checkpoint/data artifacts do not exist yet.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lensemble.errors import ConfigError, LensembleErrorCode, SchemaVersionMismatch

PHASE3_EVAL_REPORT_SCHEMA_VERSION = 1

Phase3ControlRole = Literal[
    "anchored-federation",
    "local-only",
    "naive-fedavg",
    "fork-a-frozen-encoder",
]
Phase3MetricName = Literal[
    "closed_round_completion_rate",
    "participant_submission_rate",
    "secure_sum_round_rate",
    "dp_accounted_round_rate",
    "latent_frame_drift_deg",
    "effective_rank",
]
Phase3ControlGaugeMetric = Literal[
    "latent_frame_drift_deg",
    "effective_rank",
]
Phase3EvalStatus = Literal["completed", "blocked"]
Phase3PlannerName = Literal["icem", "cem", "mppi", "not_applicable"]

_REQUIRED_CONTROLS: tuple[Phase3ControlRole, ...] = (
    "anchored-federation",
    "local-only",
    "naive-fedavg",
    "fork-a-frozen-encoder",
)
_TASK_SCALE_BLOCKER_ENV_ID = "so100-heldout://phase3-public-task-scale"
_LOCAL_SMOKE_ENV_ID = "phase3://consortium-long-run-smoke"


class Phase3SourceArtifactRef(BaseModel):
    """Public or local source artifact consumed by the Phase 3 eval report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_name: str = Field(min_length=1)
    schema_version: int = Field(ge=1)


class Phase3LongRunRoundEvidence(BaseModel):
    """Minimal #227 round evidence consumed by the eval layer."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    aggregation_backend_status: str = Field(min_length=1)
    dp_epsilon_spent: float | None = Field(default=None, ge=0.0)


class Phase3LongRunParticipantEvidence(BaseModel):
    """Minimal #227 participant evidence consumed by the eval layer."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    submitted_rounds: int = Field(ge=0)


class Phase3LongRunEvidence(BaseModel):
    """Minimal long-run report schema read by Phase 3 eval without layering back-edges."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    schema_version: int = Field(ge=1)
    generated_at: datetime
    consortium_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    final_global_model_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    closed_rounds: int = Field(ge=0)
    target_rounds: int = Field(ge=1)
    run_shape: dict[str, Any]
    rounds: tuple[Phase3LongRunRoundEvidence, ...]
    participants: tuple[Phase3LongRunParticipantEvidence, ...]

    @property
    def root_seed(self) -> int:
        """Root seed from the long-run shape, defaulting to zero for old fixtures."""

        return int(self.run_shape.get("root_seed", 0))

    @property
    def participant_count(self) -> int:
        """Declared participant count, falling back to parsed participant rows."""

        return int(self.run_shape.get("participant_count", len(self.participants)))


class Phase3PlannerBudget(BaseModel):
    """Planner/eval budget declared before Phase 3 evaluation runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    planner: Phase3PlannerName
    horizon: int = Field(ge=0)
    planning_samples: int = Field(ge=0)
    planner_iterations: int = Field(ge=0)
    eval_episodes: int = Field(ge=0)
    action_dim: int = Field(ge=0)
    notes: str = Field(min_length=1)


class Phase3ControlGaugeValue(BaseModel):
    """One bound gauge metric (frame drift / effective rank) for a completed control."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: Phase3ControlGaugeMetric
    value: float
    notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def _finite_value(self) -> "Phase3ControlGaugeValue":
        if not math.isfinite(self.value):
            raise ConfigError(
                "Phase 3 control gauge values must be finite",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="remove NaN/Inf gauge values before publishing the eval report",
            )
        return self


class Phase3CompletedControlInput(BaseModel):
    """A real, published matched control bound to immutable run evidence.

    This carries everything needed to flip a previously-blocked matched control
    into completed metric rows: the immutable checkpoint revision, the run's
    final global checkpoint hash, config hash, run-manifest hash (or report
    sha256 for the no-aggregation local-only control), the residency-safe gauge
    metrics, and the source-report binding.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    control_role: Phase3ControlRole
    task_env_id: str = Field(min_length=1)
    repo: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    source_label: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    source_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_schema_name: str = Field(min_length=1)
    source_schema_version: int = Field(ge=1)
    gauges: tuple[Phase3ControlGaugeValue, ...] = Field(min_length=1)
    note: str = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_gauges(self) -> "Phase3CompletedControlInput":
        names = [gauge.metric for gauge in self.gauges]
        if len(names) != len(set(names)):
            raise ConfigError(
                f"Phase 3 completed control {self.control_role!r} has duplicate gauge metrics",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="bind each gauge metric at most once per completed control",
            )
        if self.task_env_id == "synthetic://toy":
            raise ConfigError(
                "Phase 3 completed controls must use a non-synthetic://toy env id",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="bind a real consortium control env id, not synthetic://toy",
            )
        return self


class Phase3EvalTaskPlan(BaseModel):
    """One declared Phase 3 eval target and its pre-run expectation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_env_id: str = Field(min_length=1)
    status: Phase3EvalStatus
    task_scale: str = Field(min_length=1)
    held_out_policy: str = Field(min_length=1)
    goal_policy: str = Field(min_length=1)
    seeds: tuple[int, ...]
    metrics: tuple[Phase3MetricName | str, ...] = Field(min_length=1)
    planner_budget: Phase3PlannerBudget
    expected_outcomes: tuple[str, ...] = Field(min_length=1)
    falsifying_outcomes: tuple[str, ...] = Field(min_length=1)
    blocker: str | None = Field(default=None, min_length=1)


class Phase3EvalPlan(BaseModel):
    """Reviewer-facing Phase 3 eval plan declared before runs launch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str = Field(min_length=1)
    generated_at: datetime
    tasks: tuple[Phase3EvalTaskPlan, ...] = Field(min_length=1)
    matched_control_policy: str = Field(min_length=1)
    raw_data_in_report: Literal[False] = False


class Phase3EvalMetricRow(BaseModel):
    """One completed Phase 3 eval metric bound to hashes and run shape."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    row_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    control_role: Phase3ControlRole
    task_env_id: str = Field(min_length=1)
    metric: Phase3MetricName
    value: float
    checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    planner_budget: Phase3PlannerBudget
    source_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def _finite_value(self) -> "Phase3EvalMetricRow":
        if not math.isfinite(self.value):
            raise ConfigError(
                "Phase 3 eval metric rows must use finite numeric values",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="remove NaN/Inf metrics before publishing the eval report",
            )
        return self


class Phase3BlockedControlRow(BaseModel):
    """A required matched control that lacks completed public evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    control_role: Phase3ControlRole
    status: Literal["blocked"] = "blocked"
    task_env_id: str = Field(min_length=1)
    checkpoint_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    config_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    run_manifest_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    seed: int | None = None
    planner_budget: Phase3PlannerBudget
    blocker_source: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    required_match: tuple[str, ...] = Field(min_length=1)


class Phase3EvalReport(BaseModel):
    """Machine-readable Phase 3 downstream eval and matched-control report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    generated_at: datetime
    source_artifacts: tuple[Phase3SourceArtifactRef, ...] = Field(min_length=1)
    eval_plan: Phase3EvalPlan
    metric_rows: tuple[Phase3EvalMetricRow, ...]
    blocked_controls: tuple[Phase3BlockedControlRow, ...]
    model_card_eval_text: str = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def _cross_check(self) -> "Phase3EvalReport":
        if self.schema_version != PHASE3_EVAL_REPORT_SCHEMA_VERSION:
            raise SchemaVersionMismatch(
                f"phase3 eval report schema_version {self.schema_version!r} exceeds "
                f"reader max {PHASE3_EVAL_REPORT_SCHEMA_VERSION}",
                code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
                remediation="read with a build supporting this phase3 eval report schema",
            )
        source_hashes = {artifact.sha256 for artifact in self.source_artifacts}
        for row in self.metric_rows:
            if row.source_report_sha256 not in source_hashes:
                raise ConfigError(
                    f"Phase 3 eval row {row.row_id!r} references an unknown source artifact",
                    code=LensembleErrorCode.CONFIG_INVALID,
                    remediation="include every source artifact consumed by metric rows",
                )
        completed_controls = {row.control_role for row in self.metric_rows}
        blocked_controls = {row.control_role for row in self.blocked_controls}
        missing = set(_REQUIRED_CONTROLS) - completed_controls - blocked_controls
        if missing:
            raise ConfigError(
                "Phase 3 eval report is missing required matched controls: "
                + ", ".join(sorted(missing)),
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="add completed metric rows or explicit blocked rows for every matched control",
            )
        has_beyond_toy = any(
            row.task_env_id != "synthetic://toy" for row in self.metric_rows
        )
        has_blocker_beyond_toy = any(
            task.task_env_id != "synthetic://toy" and task.status == "blocked"
            for task in self.eval_plan.tasks
        )
        if not has_beyond_toy and not has_blocker_beyond_toy:
            raise ConfigError(
                "Phase 3 eval must include a non-synthetic://toy target or an explicit blocker",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="declare a task-scale eval target and either run it or block it explicitly",
            )
        if "paper-scale" not in self.claim_boundary:
            raise ConfigError(
                "Phase 3 eval claim boundary must reject paper-scale performance claims",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="keep model-card text conservative until task-scale evidence exists",
            )
        return self


def parse_phase3_eval_report(raw: dict[str, Any]) -> Phase3EvalReport:
    """Parse a Phase 3 eval report, gating future schemas first."""

    version = raw.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version > PHASE3_EVAL_REPORT_SCHEMA_VERSION
    ):
        raise SchemaVersionMismatch(
            f"phase3 eval report schema_version {version!r} exceeds reader max "
            f"{PHASE3_EVAL_REPORT_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation="read with a build supporting this phase3 eval report schema",
        )
    return Phase3EvalReport.model_validate(raw)


def load_phase3_eval_report(path: Path) -> Phase3EvalReport:
    """Load and validate a Phase 3 eval report JSON file."""

    return parse_phase3_eval_report(json.loads(Path(path).read_text(encoding="utf-8")))


def load_phase3_long_run_evidence(path: Path) -> Phase3LongRunEvidence:
    """Load the minimal #227 long-run evidence needed by Phase 3 eval."""

    return Phase3LongRunEvidence.model_validate(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def to_phase3_eval_report_json(report: Phase3EvalReport) -> str:
    """Canonical JSON for a Phase 3 eval report."""

    return json.dumps(
        report.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def write_phase3_eval_report(report: Phase3EvalReport, path: Path) -> Path:
    """Write a validated Phase 3 eval report as canonical JSON."""

    parse_phase3_eval_report(report.model_dump(mode="json"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_phase3_eval_report_json(report) + "\n", encoding="utf-8")
    return path


def build_phase3_eval_report(
    long_run_report_path: Path,
    *,
    generated_at: datetime | None = None,
    completed_controls: Sequence[Phase3CompletedControlInput] | None = None,
) -> Phase3EvalReport:
    """Build the Phase 3 eval report from the #227 long-run evidence.

    When ``completed_controls`` is provided, the named matched controls are
    flipped from blocked rows to completed metric rows bound to the published
    run hashes and gauge metrics, and a source-artifact reference is added for
    each control report consumed.
    """

    completed = tuple(completed_controls or ())
    _reject_duplicate_completed_controls(completed)
    report_path = Path(long_run_report_path)
    long_run = load_phase3_long_run_evidence(report_path)
    generated = generated_at or long_run.generated_at
    source_sha = _sha256_file(report_path)
    run_manifest_hash = phase3_run_manifest_hash_from_report(long_run)
    planner_budget = Phase3PlannerBudget(
        planner="not_applicable",
        horizon=0,
        planning_samples=0,
        planner_iterations=0,
        eval_episodes=0,
        action_dim=0,
        notes=(
            "No latent-MPC planner is executed for the local consortium lifecycle "
            "smoke; task-scale planner eval is blocked separately."
        ),
    )
    metric_rows = _metric_rows(
        long_run,
        source_sha=source_sha,
        run_manifest_hash=run_manifest_hash,
        planner_budget=planner_budget,
    )
    control_metric_rows = _completed_control_metric_rows(
        completed, planner_budget=planner_budget
    )
    metric_rows = [*metric_rows, *control_metric_rows]
    completed_roles = {control.control_role for control in completed}
    blocked_controls = [
        row
        for row in _blocked_controls(planner_budget=planner_budget)
        if row.control_role not in completed_roles
    ]
    model_card_text = _model_card_eval_text(
        blocked_controls, completed_controls=completed
    )
    control_artifacts = tuple(
        Phase3SourceArtifactRef(
            label=control.source_label,
            uri=control.source_uri,
            sha256=control.source_report_sha256,
            schema_name=control.source_schema_name,
            schema_version=control.source_schema_version,
        )
        for control in completed
    )
    return Phase3EvalReport(
        schema_version=PHASE3_EVAL_REPORT_SCHEMA_VERSION,
        generated_at=generated,
        source_artifacts=(
            Phase3SourceArtifactRef(
                label="Phase 3 long-run orchestration report",
                uri=str(report_path),
                sha256=source_sha,
                schema_name="phase3_long_run_report",
                schema_version=long_run.schema_version,
            ),
            *control_artifacts,
        ),
        eval_plan=Phase3EvalPlan(
            plan_id="phase3-consortium-eval-plan-v1",
            generated_at=generated,
            tasks=(
                Phase3EvalTaskPlan(
                    task_env_id=_LOCAL_SMOKE_ENV_ID,
                    status="completed",
                    task_scale="local deterministic tiny-model consortium lifecycle eval",
                    held_out_policy=(
                        "synthetic participant-local windows are generated inside each "
                        "Phase3ParticipantAgent and never serialized in this report"
                    ),
                    goal_policy=(
                        "not applicable; lifecycle metrics evaluate round closure, "
                        "participant release, secure aggregation, and DP accounting"
                    ),
                    seeds=(long_run.root_seed,),
                    metrics=tuple(row.metric for row in metric_rows),
                    planner_budget=planner_budget,
                    expected_outcomes=(
                        "all declared participant agents submit every assigned round",
                        "every closed round consumes secure-sum reporting and DP accounting",
                    ),
                    falsifying_outcomes=(
                        "any missing participant update without an explicit dropout row",
                        "any closed round without secure aggregation or DP accounting status",
                    ),
                ),
                Phase3EvalTaskPlan(
                    task_env_id=_TASK_SCALE_BLOCKER_ENV_ID,
                    status="blocked",
                    task_scale="public SO-100 held-out task-scale downstream eval",
                    held_out_policy=(
                        "final local participant episodes or an equivalent public held-out "
                        "SO-100 split must be published before this eval can run"
                    ),
                    goal_policy=(
                        "task goals and planner budget must be fixed before launch and "
                        "cited in the model card"
                    ),
                    seeds=(0,),
                    metrics=(
                        "task_success_rate",
                        "time_per_action_ms",
                        "effective_dim",
                    ),
                    planner_budget=Phase3PlannerBudget(
                        planner="icem",
                        horizon=2,
                        planning_samples=8,
                        planner_iterations=4,
                        eval_episodes=4,
                        action_dim=2,
                        notes=(
                            "Reserved task-scale budget matching the compact Phase 2 "
                            "downstream eval until public Phase 3 checkpoints/data exist."
                        ),
                    ),
                    expected_outcomes=(
                        "published checkpoint loads and executes latent-MPC on held-out task episodes",
                    ),
                    falsifying_outcomes=(
                        "checkpoint cannot be loaded",
                        "metric rows cannot be bound to checkpoint/config/run-manifest hashes",
                    ),
                    blocker=(
                        "No public Phase 3 task-scale checkpoint and held-out SO-100 eval "
                        "dataset are published yet; #230 owns final artifact publication."
                    ),
                ),
            ),
            matched_control_policy=(
                "A matched Phase 3 control must use the same participant ids, dataset/probe "
                "registry, model size, seeds, DP policy, secure-aggregation mode, eval "
                "budget, and report bindings as the anchored federation run."
            ),
        ),
        metric_rows=tuple(metric_rows),
        blocked_controls=tuple(blocked_controls),
        model_card_eval_text=model_card_text,
        claim_boundary=(
            "This Phase 3 eval report supports consortium-runtime engineering evidence "
            "from a local tiny-model run and explicit blockers for public task-scale "
            "controls. It does not claim paper-scale LeWorldModel performance or SO-100 "
            "robotics task success."
        ),
    )


def phase3_run_manifest_hash_from_report(report: Phase3LongRunEvidence) -> str:
    """Reconstruct and hash the deterministic #227 run-manifest payload."""

    payload = {
        "schema": "phase3-long-run-manifest/v1",
        "generated_at": report.generated_at.isoformat(),
        "config_hash": report.config_hash,
        "consortium_id": report.consortium_id,
        "run_id": report.run_id,
        "run_shape": report.run_shape,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256((raw + "\n").encode("utf-8")).hexdigest()


def _metric_rows(
    report: Phase3LongRunEvidence,
    *,
    source_sha: str,
    run_manifest_hash: str,
    planner_budget: Phase3PlannerBudget,
) -> list[Phase3EvalMetricRow]:
    closed = max(int(report.closed_rounds), 1)
    participant_count = max(report.participant_count, 1)
    target_updates = max(int(report.target_rounds) * participant_count, 1)
    total_submitted = sum(p.submitted_rounds for p in report.participants)
    secure_sum_rounds = sum(
        1 for row in report.rounds if row.aggregation_backend_status == "secure_sum"
    )
    dp_accounted_rounds = sum(
        1 for row in report.rounds if row.dp_epsilon_spent is not None
    )
    values: tuple[tuple[Phase3MetricName, float, str], ...] = (
        (
            "closed_round_completion_rate",
            float(report.closed_rounds) / float(report.target_rounds),
            "closed rounds divided by declared target rounds",
        ),
        (
            "participant_submission_rate",
            float(total_submitted) / float(target_updates),
            "participant submitted rounds divided by assigned participant-rounds",
        ),
        (
            "secure_sum_round_rate",
            float(secure_sum_rounds) / float(closed),
            "closed rounds whose aggregation report used secure_sum status",
        ),
        (
            "dp_accounted_round_rate",
            float(dp_accounted_rounds) / float(closed),
            "closed rounds with non-null DP epsilon accounting",
        ),
    )
    return [
        Phase3EvalMetricRow(
            row_id=f"anchored-federation.{metric.replace('_', '-')}",
            control_role="anchored-federation",
            task_env_id=_LOCAL_SMOKE_ENV_ID,
            metric=metric,
            value=value,
            checkpoint_hash=report.final_global_model_hash,
            config_hash=report.config_hash,
            run_manifest_hash=run_manifest_hash,
            seed=report.root_seed,
            planner_budget=planner_budget,
            source_report_sha256=source_sha,
            notes=notes,
        )
        for metric, value, notes in values
    ]


def _reject_duplicate_completed_controls(
    completed: Sequence[Phase3CompletedControlInput],
) -> None:
    roles = [control.control_role for control in completed]
    if len(roles) != len(set(roles)):
        raise ConfigError(
            "Phase 3 completed controls must declare each control role at most once",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="pass one Phase3CompletedControlInput per control role",
        )
    if "anchored-federation" in roles:
        raise ConfigError(
            "anchored-federation is already a completed metric row from the long-run "
            "report and must not be passed as a completed control input",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="only pass the previously-blocked controls as completed inputs",
        )


def _completed_control_metric_rows(
    completed: Sequence[Phase3CompletedControlInput],
    *,
    planner_budget: Phase3PlannerBudget,
) -> list[Phase3EvalMetricRow]:
    rows: list[Phase3EvalMetricRow] = []
    for control in completed:
        for gauge in control.gauges:
            rows.append(
                Phase3EvalMetricRow(
                    row_id=f"{control.control_role}.{gauge.metric.replace('_', '-')}",
                    control_role=control.control_role,
                    task_env_id=control.task_env_id,
                    metric=gauge.metric,
                    value=gauge.value,
                    checkpoint_hash=control.checkpoint_hash,
                    config_hash=control.config_hash,
                    run_manifest_hash=control.run_manifest_hash,
                    seed=control.seed,
                    planner_budget=planner_budget,
                    source_report_sha256=control.source_report_sha256,
                    notes=f"{control.note} ({gauge.notes}); checkpoint revision "
                    f"{control.revision} of {control.repo}",
                )
            )
    return rows


def _blocked_controls(
    *,
    planner_budget: Phase3PlannerBudget,
) -> list[Phase3BlockedControlRow]:
    required_match = (
        "same four participant ids and dataset/probe registry",
        "same tiny Phase 3 model shape and root seed",
        "same DP policy and secure-aggregation mode",
        "same eval task/env id and planner budget",
        "metric rows bound to checkpoint, config, and run-manifest hashes",
    )
    return [
        Phase3BlockedControlRow(
            control_role="local-only",
            task_env_id=_TASK_SCALE_BLOCKER_ENV_ID,
            planner_budget=planner_budget,
            blocker_source="GitHub issue #228 Phase 3 eval acceptance",
            reason=(
                "No matched local-only Phase 3 run is published for the same "
                "participant data refs, seed, model size, and eval budget."
            ),
            required_match=required_match,
        ),
        Phase3BlockedControlRow(
            control_role="naive-fedavg",
            task_env_id=_TASK_SCALE_BLOCKER_ENV_ID,
            planner_budget=planner_budget,
            blocker_source="GitHub issue #228 Phase 3 eval acceptance",
            reason=(
                "No matched lambda_anc=0 / unanchored Phase 3 consortium control "
                "is published for the same run shape and eval budget."
            ),
            required_match=required_match,
        ),
        Phase3BlockedControlRow(
            control_role="fork-a-frozen-encoder",
            task_env_id=_TASK_SCALE_BLOCKER_ENV_ID,
            planner_budget=planner_budget,
            blocker_source="GitHub issue #228 Phase 3 eval acceptance",
            reason=(
                "The RFC-0002 Fork A frozen-encoder safe-degrade baseline has not "
                "been run for the Phase 3 consortium manifest."
            ),
            required_match=required_match,
        ),
    ]


def _model_card_eval_text(
    blocked_controls: list[Phase3BlockedControlRow],
    *,
    completed_controls: Sequence[Phase3CompletedControlInput] = (),
) -> str:
    intro = (
        "Phase 3 evaluation evidence covers the local deterministic "
        "consortium-runtime smoke (participant-agent updates, ten closed rounds, "
        "secure-sum reporting, and DP accounting) plus four real matched control "
        "runs published on HF Jobs (DP-off, latent_dim=256, 6 rounds, "
        "window_steps=4, simulated secure-agg, four participants phase3-so100-a..d, "
        "held-out silo4)."
    )
    gauge_finding = (
        "Gauge finding: the frame anchor reduces inter-participant latent "
        "frame-drift at aggregation (anchored round-0 48.97 deg vs naive-FedAvg "
        "180 deg); Fork-A's frozen encoder is the 0 deg safe-degrade baseline; and "
        "local-only silos train healthily (effective_rank ~120) but diverge "
        "maximally (180 deg) - the divergence federation is designed to close."
    )
    limitation = (
        "Honest limitation: at the default outer-step (outer_lr=0.7) with a "
        "random-init warm-start (real V-JEPA weights remain unvendored, #96), the "
        "federated global representation collapses over rounds (effective_rank -> "
        "1), so the clean anchored-vs-naive contrast is the round-0 measurement; "
        "sustained non-collapsing federated training is a documented follow-up. "
        "This report is consortium-engineering and training evidence, NOT a "
        "cryptographic proof of honest participant computation."
    )
    task_scale = (
        "Public task-scale SO-100 downstream evaluation remains blocked until the "
        "Phase 3 checkpoint and held-out eval data are published."
    )
    if blocked_controls:
        blocked = ", ".join(row.control_role for row in blocked_controls)
        controls_line = (
            f"Blocked controls: {blocked}. These rows must not be described as "
            "completed robotics performance comparisons."
        )
    else:
        completed = ", ".join(control.control_role for control in completed_controls)
        controls_line = (
            f"Completed matched controls bound to published run hashes: {completed}. "
            "These are representation-gauge controls and must not be described as "
            "completed robotics performance comparisons."
        )
    return " ".join((intro, gauge_finding, limitation, task_scale, controls_line))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "PHASE3_EVAL_REPORT_SCHEMA_VERSION",
    "Phase3BlockedControlRow",
    "Phase3CompletedControlInput",
    "Phase3ControlGaugeMetric",
    "Phase3ControlGaugeValue",
    "Phase3ControlRole",
    "Phase3EvalMetricRow",
    "Phase3EvalPlan",
    "Phase3EvalReport",
    "Phase3EvalTaskPlan",
    "Phase3LongRunEvidence",
    "Phase3LongRunParticipantEvidence",
    "Phase3LongRunRoundEvidence",
    "Phase3PlannerBudget",
    "Phase3SourceArtifactRef",
    "build_phase3_eval_report",
    "load_phase3_eval_report",
    "load_phase3_long_run_evidence",
    "parse_phase3_eval_report",
    "phase3_run_manifest_hash_from_report",
    "to_phase3_eval_report_json",
    "write_phase3_eval_report",
]
