"""Phase 3 downstream latent-MPC eval report (GitHub issue #245).

The only prior downstream number was a ``synthetic://toy``, single-sample,
``success_rate=0.5`` placeholder. This module produces an honest, bounded Phase
3 downstream eval report that:

1. Goes beyond ``synthetic://toy`` by citing the REAL held-out SO-100 latent
   metrics (final-round ``effective_rank`` / ``val_pred``) computed by the
   headline consortium run on the disjoint held-out split
   ``phase3-so100-silo4.h5`` (#242).
2. Records a NON-TOY latent-MPC planner budget — the budget a closed-loop run
   WOULD use — without executing a planner.
3. Honestly documents the two specific blockers that make a real closed-loop
   task-success number infeasible right now: the unvendored
   ``stable-worldmodel`` suite (#96) and the collapsing federated checkpoints
   (#244). It does NOT fabricate a task-success pass.

The report mirrors :class:`lensemble.eval.phase2_downstream.Phase2DownstreamEvalReport`
(checkpoint ref + planner budget + claim boundary) but is residency-safe: it
carries only scalars, hashes, ids, and configuration counts — no raw
observation or action arrays.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lensemble.errors import ConfigError, LensembleErrorCode, SchemaVersionMismatch

PHASE3_DOWNSTREAM_REPORT_SCHEMA_VERSION = 1

# The synthetic placeholder values the real held-out latent metrics must NOT be.
_SYNTHETIC_TOY_ENV_ID = "synthetic://toy"
_SYNTHETIC_TOY_SUCCESS_RATE = 0.5
_SYNTHETIC_TOY_VAL_PRED = 1.0

Phase3DownstreamPlannerName = Literal["icem", "cem", "mppi"]
Phase3TaskSuccessStatus = Literal["blocked"]


class Phase3DownstreamCheckpointRef(BaseModel):
    """Immutable public checkpoint identity evaluated by the downstream report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo_id: str = Field(min_length=1)
    repo_type: Literal["model"] = "model"
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class Phase3HeldOutLatentMetrics(BaseModel):
    """Real held-out SO-100 latent metrics measured on the disjoint eval split.

    These are NOT synthetic://toy placeholders: ``effective_rank`` and
    ``val_pred`` are the FINAL-round values computed by the headline consortium
    run on the held-out SO-100 split ``phase3-so100-silo4.h5`` (#242), the split
    disjoint from every participant's training silo.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    effective_rank: float = Field(gt=0.0)
    val_pred: float = Field(ge=0.0)
    latent_dim: int = Field(gt=0)
    round_index: int = Field(ge=0)
    held_out_windows: int = Field(gt=0)
    window_steps: int = Field(gt=0)
    measured_on: str = Field(min_length=1)
    note: str = Field(min_length=1)

    @model_validator(mode="after")
    def _real_not_synthetic(self) -> "Phase3HeldOutLatentMetrics":
        if not math.isfinite(self.effective_rank) or not math.isfinite(self.val_pred):
            raise ConfigError(
                "Phase 3 held-out latent metrics must be finite",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="bind the real per-round effective_rank/val_pred from the run report",
            )
        # The real held-out latent signal must be beyond the synthetic toy
        # placeholders. effective_rank must exceed the toy success_rate=0.5 and
        # val_pred must be the real (large) prediction error, not the toy 1.0.
        if self.effective_rank <= _SYNTHETIC_TOY_SUCCESS_RATE:
            raise ConfigError(
                "Phase 3 held-out effective_rank looks like a synthetic://toy placeholder",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="bind the real held-out effective_rank (~36-47 of 256), not the toy 0.5",
            )
        if self.val_pred == _SYNTHETIC_TOY_VAL_PRED:
            raise ConfigError(
                "Phase 3 held-out val_pred looks like a synthetic://toy placeholder",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="bind the real held-out val_pred from the consortium run report",
            )
        if self.effective_rank > self.latent_dim:
            raise ConfigError(
                "Phase 3 held-out effective_rank cannot exceed the latent dimension",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="bind effective_rank measured in the model latent space",
            )
        return self


class Phase3DownstreamPlannerBudget(BaseModel):
    """The NON-TOY latent-MPC planner budget a closed-loop run WOULD use.

    No planner is executed: this records the budget so reviewers know the eval
    is a real task-scale latent-MPC plan, not a 1-sample toy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    planner: Phase3DownstreamPlannerName
    horizon: int = Field(gt=0)
    planning_samples: int = Field(gt=0)
    planner_iterations: int = Field(gt=0)
    eval_episodes: int = Field(gt=0)
    action_dim: int = Field(gt=0)
    executed: Literal[False] = False
    note: str = Field(min_length=1)

    @model_validator(mode="after")
    def _non_toy(self) -> "Phase3DownstreamPlannerBudget":
        # A toy budget is the compact Phase 2 synthetic one (horizon<=2,
        # planning_samples<=8, eval_episodes<=4). Reject it here so the recorded
        # budget is demonstrably non-toy.
        if (
            self.horizon <= 2
            or self.planning_samples <= 8
            or self.eval_episodes <= 4
            or self.planner_iterations <= 4
        ):
            raise ConfigError(
                "Phase 3 downstream planner budget must be non-toy "
                "(horizon>2, planning_samples>8, planner_iterations>4, eval_episodes>4)",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="record a real task-scale latent-MPC budget, not the Phase 2 toy budget",
            )
        return self


class Phase3TaskSuccessBlocker(BaseModel):
    """One specific, structured blocker on closed-loop task-success."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    blocker_ref: str = Field(pattern=r"^#\d+$")
    reason: str = Field(min_length=1)


class Phase3TaskSuccess(BaseModel):
    """An explicitly blocked task-success outcome — never a synthetic pass."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Phase3TaskSuccessStatus = "blocked"
    # Typed as float | None so a fabricated pass is rejected by the structured
    # validator below (raising ConfigError), not only by a pydantic type error.
    success_rate: float | None = None
    blockers: tuple[Phase3TaskSuccessBlocker, ...]

    @model_validator(mode="after")
    def _structured_blockers(self) -> "Phase3TaskSuccess":
        if self.success_rate is not None:
            raise ConfigError(
                "blocked Phase 3 task-success must not record a success_rate "
                "(do not fabricate a task-success pass)",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="leave success_rate null until a closed-loop run is feasible",
            )
        if len(self.blockers) < 2:
            raise ConfigError(
                "blocked Phase 3 task-success must document at least two blockers",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="document the #96 and #244 blockers explicitly",
            )
        refs = {blocker.blocker_ref for blocker in self.blockers}
        if not {"#96", "#244"}.issubset(refs):
            raise ConfigError(
                "Phase 3 task-success must cite the #96 and #244 blockers explicitly",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="document the unvendored stable-worldmodel (#96) and collapse (#244) blockers",
            )
        return self


class Phase3DownstreamEvalReport(BaseModel):
    """Machine-readable, residency-safe Phase 3 downstream latent-MPC eval report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    generated_at: datetime
    checkpoint: Phase3DownstreamCheckpointRef
    task_env_id: str = Field(min_length=1)
    held_out_data_ref: str = Field(min_length=1)
    held_out_latent_metrics: Phase3HeldOutLatentMetrics
    planner_budget: Phase3DownstreamPlannerBudget
    task_success: Phase3TaskSuccess
    claim_boundary: str = Field(min_length=1)
    raw_data_in_report: Literal[False] = False
    source_report_uri: str = Field(min_length=1)

    @model_validator(mode="after")
    def _cross_check(self) -> "Phase3DownstreamEvalReport":
        if self.schema_version != PHASE3_DOWNSTREAM_REPORT_SCHEMA_VERSION:
            raise SchemaVersionMismatch(
                f"phase3 downstream report schema_version {self.schema_version!r} "
                f"exceeds reader max {PHASE3_DOWNSTREAM_REPORT_SCHEMA_VERSION}",
                code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
                remediation="read with a build supporting this phase3 downstream report schema",
            )
        if self.task_env_id == _SYNTHETIC_TOY_ENV_ID:
            raise ConfigError(
                "Phase 3 downstream eval must use a real held-out task env id, "
                "not synthetic://toy",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="bind the so100-heldout:// task env id, not the synthetic toy placeholder",
            )
        if "paper-scale" not in self.claim_boundary:
            raise ConfigError(
                "Phase 3 downstream claim boundary must reject paper-scale performance claims",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="keep the claim boundary conservative: latent evidence only, task-success deferred",
            )
        if "#96" not in self.claim_boundary or "#244" not in self.claim_boundary:
            raise ConfigError(
                "Phase 3 downstream claim boundary must cite the #96 and #244 deferral blockers",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="state that closed-loop task-success is deferred pending #96 and #244",
            )
        required_correction_phrases = (
            "magnitude collapse",
            "central ceiling",
            "skill_vs_identity is gameable",
            "effective_rank is scale-invariant",
        )
        missing = [
            phrase
            for phrase in required_correction_phrases
            if phrase not in self.claim_boundary
        ]
        if missing:
            raise ConfigError(
                "Phase 3 downstream claim boundary is missing SO-100 correction phrases: "
                + ", ".join(missing),
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="disclose the held-out collapse, central ceiling, and gameable/scale-invariant metrics",
            )
        return self


def parse_phase3_downstream_eval_report(
    raw: dict[str, object],
) -> Phase3DownstreamEvalReport:
    """Parse a Phase 3 downstream report, gating future schemas first."""

    version = raw.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version > PHASE3_DOWNSTREAM_REPORT_SCHEMA_VERSION
    ):
        raise SchemaVersionMismatch(
            f"phase3 downstream report schema_version {version!r} exceeds reader max "
            f"{PHASE3_DOWNSTREAM_REPORT_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation="read with a build supporting this phase3 downstream report schema",
        )
    return Phase3DownstreamEvalReport.model_validate(raw)


def load_phase3_downstream_eval_report(path: Path) -> Phase3DownstreamEvalReport:
    """Load and validate a Phase 3 downstream eval report JSON file."""

    return parse_phase3_downstream_eval_report(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def to_phase3_downstream_eval_report_json(report: Phase3DownstreamEvalReport) -> str:
    """Canonical JSON for a Phase 3 downstream eval report."""

    return json.dumps(
        report.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def write_phase3_downstream_eval_report(
    report: Phase3DownstreamEvalReport, path: Path
) -> Path:
    """Write a validated Phase 3 downstream eval report as canonical JSON."""

    parse_phase3_downstream_eval_report(report.model_dump(mode="json"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        to_phase3_downstream_eval_report_json(report) + "\n", encoding="utf-8"
    )
    return path


def _require_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"consortium run report field {field!r} is not numeric",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="read a well-formed phase3 consortium run report",
        )
    return float(value)


def _round_index(row: object) -> int:
    if isinstance(row, dict):
        value = row.get("round_index")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return -1


def _final_round(run_report: dict[str, object]) -> dict[str, object]:
    rounds = run_report.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ConfigError(
            "consortium run report has no rounds to read held-out latent metrics from",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="point at the headline phase3 consortium run report with closed rounds",
        )
    final = max(rounds, key=_round_index)
    if not isinstance(final, dict):
        raise ConfigError(
            "consortium run report final round is malformed",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="read a well-formed phase3 consortium run report",
        )
    return final


def build_phase3_downstream_eval_report(
    consortium_run_report_path: Path,
    *,
    checkpoint: Phase3DownstreamCheckpointRef,
    task_env_id: str,
    held_out_data_ref: str,
    planner_budget: Phase3DownstreamPlannerBudget,
    blockers: tuple[Phase3TaskSuccessBlocker, ...],
    claim_boundary: str,
    source_report_uri: str,
    held_out_windows: int,
    window_steps: int,
    generated_at: datetime | None = None,
) -> Phase3DownstreamEvalReport:
    """Build the Phase 3 downstream eval report from the headline run report.

    The held-out latent metrics are read from the FINAL closed round of the
    headline consortium run report — the real held-out SO-100 latent signal,
    not a synthetic placeholder.
    """

    run_report = json.loads(
        Path(consortium_run_report_path).read_text(encoding="utf-8")
    )
    if not isinstance(run_report, dict):
        raise ConfigError(
            "consortium run report must be a JSON object",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="point at a phase3 consortium run report JSON file",
        )
    final = _final_round(run_report)
    run_shape = run_report.get("run_shape")
    latent_dim = 256
    if isinstance(run_shape, dict):
        shape_latent = run_shape.get("model_latent_dim")
        if isinstance(shape_latent, int) and not isinstance(shape_latent, bool):
            latent_dim = shape_latent
    metrics = Phase3HeldOutLatentMetrics(
        effective_rank=_require_number(final.get("effective_rank"), "effective_rank"),
        val_pred=_require_number(final.get("val_pred"), "val_pred"),
        latent_dim=latent_dim,
        round_index=int(_require_number(final.get("round_index"), "round_index")),
        held_out_windows=held_out_windows,
        window_steps=window_steps,
        measured_on=held_out_data_ref,
        note=(
            "Corrected SO-100 latent metrics: final-round effective_rank and "
            "val_pred were computed on the disjoint held-out split "
            "phase3-so100-silo4.h5 (#242), but effective_rank is "
            "scale-invariant and blind to held-out magnitude collapse "
            "(~7.5e-6 latent variance; thoughts/collapse_fix_probe.py). "
            "The central ceiling probe (thoughts/central_ceiling_probe.py) "
            "keeps this from being a downstream usefulness claim."
        ),
    )
    return Phase3DownstreamEvalReport(
        schema_version=PHASE3_DOWNSTREAM_REPORT_SCHEMA_VERSION,
        generated_at=generated_at or datetime.now(timezone.utc),
        checkpoint=checkpoint,
        task_env_id=task_env_id,
        held_out_data_ref=held_out_data_ref,
        held_out_latent_metrics=metrics,
        planner_budget=planner_budget,
        task_success=Phase3TaskSuccess(blockers=blockers),
        claim_boundary=claim_boundary,
        source_report_uri=source_report_uri,
    )
