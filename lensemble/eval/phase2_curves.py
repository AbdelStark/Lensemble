"""Phase 2 baselines, ablations, and curve evidence reports.

The report aggregates existing residency-safe artifacts: the Phase 2
``ClaimMVPReport`` training report, the downstream-eval wrapper, and optional
matched control/ablation claim reports. It does not run training. Its job is to
turn completed run artifacts into a generated table where every point carries
the run, config, checkpoint, and report hashes needed for review.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lensemble.errors import ConfigError, LensembleErrorCode, SchemaVersionMismatch
from lensemble.eval.claim_mvp import ClaimMVPReport
from lensemble.eval.phase2_downstream import Phase2DownstreamEvalReport

PHASE2_CURVES_REPORT_SCHEMA_VERSION = 1

Phase2RunRole = Literal[
    "anchored-federation",
    "naive-fedavg",
    "local-only",
    "centralized-pooled",
    "fork-a",
    "lambda-sig-ablation",
    "lambda-anc-ablation",
    "participant-count-ablation",
    "inner-horizon-ablation",
    "model-scale-ablation",
]
Phase2Comparison = Literal[
    "naive-fedavg",
    "local-only",
    "centralized-pooled",
    "fork-a",
    "lambda-sig-ablation",
    "lambda-anc-ablation",
    "participant-count-ablation",
    "inner-horizon-ablation",
    "model-scale-ablation",
]
Phase2AblationAxis = Literal[
    "lambda_sig",
    "lambda_anc",
    "participant_count",
    "inner_horizon",
    "model_scale",
]
Phase2CurveFamily = Literal[
    "training-final-scalars",
    "training-round-update-norms",
    "downstream-eval",
]
Phase2CurveMetric = Literal[
    "val_pred",
    "val_sigreg",
    "effective_rank",
    "frame_drift_deg",
    "update_l2_norm",
    "downstream_success_rate",
    "downstream_time_per_action_ms",
    "downstream_effective_dim",
]


class Phase2SourceReportRef(BaseModel):
    """Public identity of a source report consumed by the curves artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(min_length=1)
    schema_name: Literal["claim_mvp_report", "phase2_downstream_eval_report"]
    schema_version: int = Field(ge=1)
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repo_id: str | None = None
    repo_type: Literal["model", "dataset"] | None = None
    revision: str | None = None
    path_in_repo: str | None = None
    job_id: str | None = None
    job_url: str | None = None


class Phase2MatchedPolicy(BaseModel):
    """How the generated report interprets matched controls."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    data: str = Field(min_length=1)
    seeds: str = Field(min_length=1)
    model_size: str = Field(min_length=1)
    eval_budget: str = Field(min_length=1)


class Phase2CurvePoint(BaseModel):
    """One generated curve/table row bound to run and artifact hashes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    row_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    curve: Phase2CurveFamily
    metric: Phase2CurveMetric
    run_role: Phase2RunRole
    run_label: str = Field(min_length=1)
    ablation_axis: Phase2AblationAxis | None = None
    x_axis: Literal["round", "eval"]
    x_value: float
    value: float
    participant_id: str | None = None
    job_id: str | None = None
    job_url: str | None = None
    source_report_uri: str = Field(min_length=1)
    source_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    global_model_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    run_manifest_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    eval_config_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    match_notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _finite_numbers(self) -> "Phase2CurvePoint":
        if not math.isfinite(self.x_value) or not math.isfinite(self.value):
            raise ConfigError(
                "curve points must use finite numeric coordinates and values",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="remove NaN/Inf metrics before emitting Phase 2 curves",
            )
        return self


class Phase2BlockedComparison(BaseModel):
    """A required comparison that has not produced matched public evidence yet."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    comparison: Phase2Comparison
    status: Literal["blocked"] = "blocked"
    blocker_source: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    required_match: tuple[str, ...]
    issue_url: str | None = None


class Phase2BaselinesCurvesReport(BaseModel):
    """Schema-validated Phase 2 baseline/curve evidence artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    generated_at: datetime
    source_reports: tuple[Phase2SourceReportRef, ...]
    matched_policy: Phase2MatchedPolicy
    curve_points: tuple[Phase2CurvePoint, ...]
    blocked_comparisons: tuple[Phase2BlockedComparison, ...]
    raw_data_in_report: Literal[False] = False
    model_card_baseline_text: str = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def _cross_check(self) -> "Phase2BaselinesCurvesReport":
        if self.schema_version != PHASE2_CURVES_REPORT_SCHEMA_VERSION:
            raise SchemaVersionMismatch(
                f"phase2 curves report schema_version {self.schema_version!r} "
                f"exceeds reader max {PHASE2_CURVES_REPORT_SCHEMA_VERSION}",
                code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
                remediation="read with a build supporting this phase2 curves schema",
            )
        if not self.curve_points:
            raise ConfigError(
                "phase2 curves report needs at least one generated curve point",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="supply at least the anchored training report",
            )

        source_hashes = {source.sha256 for source in self.source_reports}
        for point in self.curve_points:
            if point.source_report_sha256 not in source_hashes:
                raise ConfigError(
                    f"curve point {point.row_id!r} references an unknown source report hash",
                    code=LensembleErrorCode.CONFIG_INVALID,
                    remediation="include every source report in source_reports",
                )

        metrics = {point.metric for point in self.curve_points}
        required_metrics = {
            "val_pred",
            "val_sigreg",
            "effective_rank",
            "frame_drift_deg",
            "downstream_success_rate",
            "downstream_time_per_action_ms",
        }
        missing_metrics = required_metrics - metrics
        if missing_metrics:
            raise ConfigError(
                "phase2 curves report is missing required metric families: "
                + ", ".join(sorted(missing_metrics)),
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="include anchored training scalars and the downstream eval report",
            )

        covered_roles = {point.run_role for point in self.curve_points}
        blocked = {item.comparison for item in self.blocked_comparisons}
        for comparison in ("local-only", "centralized-pooled", "naive-fedavg"):
            if comparison not in covered_roles and comparison not in blocked:
                raise ConfigError(
                    f"missing baseline {comparison!r} is neither covered nor blocked",
                    code=LensembleErrorCode.CONFIG_INVALID,
                    remediation="add a matched control report or an explicit blocker",
                )
        if not any(point.ablation_axis is not None for point in self.curve_points):
            ablation_blockers = {
                item.comparison
                for item in self.blocked_comparisons
                if item.comparison.endswith("-ablation")
            }
            if not ablation_blockers:
                raise ConfigError(
                    "phase2 curves report needs an ablation point or an explicit ablation blocker",
                    code=LensembleErrorCode.CONFIG_INVALID,
                    remediation="run a matched lambda/model/participant ablation or mark it blocked",
                )
        return self


@dataclass(frozen=True)
class Phase2ClaimCurveInput:
    """One claim report to fold into the Phase 2 curves table."""

    run_role: Phase2RunRole
    run_label: str
    report: ClaimMVPReport
    source_ref: Phase2SourceReportRef
    ablation_axis: Phase2AblationAxis | None = None


def phase2_source_report_ref_from_path(
    path: Path,
    *,
    label: str,
    schema_name: Literal["claim_mvp_report", "phase2_downstream_eval_report"],
    schema_version: int,
    uri: str,
    repo_id: str | None = None,
    repo_type: Literal["model", "dataset"] | None = None,
    revision: str | None = None,
    path_in_repo: str | None = None,
    job_id: str | None = None,
    job_url: str | None = None,
) -> Phase2SourceReportRef:
    """Build a source-report ref by hashing the exact local copy consumed."""

    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return Phase2SourceReportRef(
        label=label,
        schema_name=schema_name,
        schema_version=schema_version,
        uri=uri,
        sha256=digest,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        path_in_repo=path_in_repo,
        job_id=job_id,
        job_url=job_url,
    )


def parse_phase2_baselines_curves_report(
    raw: dict[str, Any],
) -> Phase2BaselinesCurvesReport:
    """Parse a Phase 2 curves report, gating schema version first."""

    version = raw.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version > PHASE2_CURVES_REPORT_SCHEMA_VERSION
    ):
        raise SchemaVersionMismatch(
            f"phase2 curves report schema_version {version!r} exceeds reader max "
            f"{PHASE2_CURVES_REPORT_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation="read with a build supporting this phase2 curves schema",
        )
    return Phase2BaselinesCurvesReport.model_validate(raw)


def build_phase2_baselines_curves_report(
    *,
    anchored: Phase2ClaimCurveInput,
    downstream_report: Phase2DownstreamEvalReport,
    downstream_source_ref: Phase2SourceReportRef,
    control_reports: Sequence[Phase2ClaimCurveInput] = (),
    generated_at: datetime | None = None,
    blocked_comparisons: Sequence[Phase2BlockedComparison] = (),
) -> Phase2BaselinesCurvesReport:
    """Generate the Phase 2 baseline/curve report from completed artifacts."""

    if anchored.run_role != "anchored-federation":
        raise ConfigError(
            "the primary Phase 2 curves input must be the anchored-federation run",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="pass the main Phase 2 claim report as anchored",
        )
    source_reports = [anchored.source_ref, downstream_source_ref]
    curve_points = _claim_curve_points(anchored)
    for control in control_reports:
        _validate_control_matches_anchor(anchored.report, control.report, control)
        source_reports.append(control.source_ref)
        curve_points.extend(_claim_curve_points(control))

    if (
        downstream_report.checkpoint.checkpoint_hash
        != anchored.report.final_global_hash
    ):
        raise ConfigError(
            "downstream eval checkpoint hash does not match anchored final global hash",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="evaluate the exact anchored checkpoint used for the curve report",
        )
    curve_points.extend(
        _downstream_curve_points(
            downstream_report,
            downstream_source_ref,
            train_config_hash=anchored.report.config_hash,
        )
    )

    blockers = list(blocked_comparisons)
    blockers.extend(_default_blockers(curve_points))
    blocked_names = ", ".join(item.comparison for item in blockers) or "none"
    return Phase2BaselinesCurvesReport(
        schema_version=PHASE2_CURVES_REPORT_SCHEMA_VERSION,
        generated_at=generated_at or datetime.now(timezone.utc),
        source_reports=tuple(_dedupe_source_reports(source_reports)),
        matched_policy=Phase2MatchedPolicy(
            data=(
                "A control is treated as matched only when participant ids and "
                "dataset roots equal the anchored Phase 2 SO-100 run."
            ),
            seeds=(
                "HF launcher defaults are used unless the source job command "
                "overrides them; each row records the resolved config hash."
            ),
            model_size=(
                "Matched controls use the compact Phase 2 launcher shape from "
                "the source job command; each row is bound to a checkpoint hash."
            ),
            eval_budget=(
                "Training scalars use the completed job's metric window budget; "
                "downstream rows record the explicit planner/eval config hash."
            ),
        ),
        curve_points=tuple(curve_points),
        blocked_comparisons=tuple(blockers),
        model_card_baseline_text=(
            "Phase 2 baseline coverage is partial: the generated table includes "
            "only completed, hash-bound public runs. Blocked comparisons: "
            f"{blocked_names}. Blocked rows must not be described as completed "
            "comparisons."
        ),
        claim_boundary=(
            "This report supports an engineering-scale Phase 2 evidence claim: "
            "published SO-100 federated training artifacts, a downstream "
            "synthetic planning eval, and any completed matched controls. It "
            "does not claim paper-scale LeWorldModel performance or exhaustive "
            "baseline coverage."
        ),
    )


def _claim_curve_points(input_: Phase2ClaimCurveInput) -> list[Phase2CurvePoint]:
    report = input_.report
    ref = input_.source_ref
    match_notes = _claim_match_notes(report)
    points: list[Phase2CurvePoint] = []
    for round_report in report.round_metrics:
        for participant_id, value in sorted(round_report.update_l2_norms.items()):
            points.append(
                Phase2CurvePoint(
                    row_id=_row_id(
                        input_.run_role,
                        "round",
                        round_report.round_index,
                        participant_id,
                        "update-l2-norm",
                    ),
                    curve="training-round-update-norms",
                    metric="update_l2_norm",
                    run_role=input_.run_role,
                    run_label=input_.run_label,
                    ablation_axis=input_.ablation_axis,
                    x_axis="round",
                    x_value=float(round_report.round_index),
                    value=float(value),
                    participant_id=participant_id,
                    job_id=ref.job_id,
                    job_url=ref.job_url,
                    source_report_uri=ref.uri,
                    source_report_sha256=ref.sha256,
                    config_hash=report.config_hash,
                    checkpoint_hash=round_report.global_model_hash,
                    global_model_hash=round_report.global_model_hash,
                    run_manifest_hash=report.metrics.run_manifest_hash,
                    match_notes=match_notes,
                )
            )

    x_value = float(
        max(
            (round_report.round_index for round_report in report.round_metrics),
            default=0,
        )
    )
    scalar_metrics: tuple[tuple[Phase2CurveMetric, float | None], ...] = (
        ("val_pred", report.metrics.val_pred),
        ("val_sigreg", report.metrics.val_sigreg),
        ("effective_rank", report.metrics.effective_rank),
        ("frame_drift_deg", report.metrics.frame_drift_deg),
    )
    for metric, value in scalar_metrics:
        if value is None:
            continue
        points.append(
            Phase2CurvePoint(
                row_id=_row_id(input_.run_role, "final", metric),
                curve="training-final-scalars",
                metric=metric,
                run_role=input_.run_role,
                run_label=input_.run_label,
                ablation_axis=input_.ablation_axis,
                x_axis="round",
                x_value=x_value,
                value=float(value),
                job_id=ref.job_id,
                job_url=ref.job_url,
                source_report_uri=ref.uri,
                source_report_sha256=ref.sha256,
                config_hash=report.config_hash,
                checkpoint_hash=report.final_global_hash,
                global_model_hash=report.final_global_hash,
                run_manifest_hash=report.metrics.run_manifest_hash,
                match_notes=match_notes,
            )
        )
    return points


def _downstream_curve_points(
    report: Phase2DownstreamEvalReport,
    ref: Phase2SourceReportRef,
    *,
    train_config_hash: str,
) -> list[Phase2CurvePoint]:
    metrics: tuple[tuple[Phase2CurveMetric, float], ...] = (
        ("downstream_success_rate", report.eval_report.success_rate),
        ("downstream_time_per_action_ms", report.eval_report.time_per_action_ms),
        ("downstream_effective_dim", report.eval_report.effective_dim),
    )
    points: list[Phase2CurvePoint] = []
    for metric, value in metrics:
        points.append(
            Phase2CurvePoint(
                row_id=_row_id("anchored-federation", "downstream", metric),
                curve="downstream-eval",
                metric=metric,
                run_role="anchored-federation",
                run_label="anchored federation downstream eval",
                x_axis="eval",
                x_value=0.0,
                value=float(value),
                job_id=ref.job_id or report.checkpoint.training_job_id,
                job_url=ref.job_url or report.checkpoint.training_job_url,
                source_report_uri=ref.uri,
                source_report_sha256=ref.sha256,
                config_hash=train_config_hash,
                checkpoint_hash=report.checkpoint.checkpoint_hash,
                global_model_hash=report.checkpoint.checkpoint_hash,
                run_manifest_hash=report.eval_report.run_manifest_hash,
                eval_config_hash=report.eval_config_hash,
                match_notes=(
                    f"env={report.eval_report.env_id}",
                    f"planner={report.eval_report.planner}",
                    f"planning_samples={report.eval_report.planning_samples}",
                    f"n_episodes={report.task.n_episodes}",
                ),
            )
        )
    return points


def _validate_control_matches_anchor(
    anchored: ClaimMVPReport,
    control: ClaimMVPReport,
    input_: Phase2ClaimCurveInput,
) -> None:
    if _participant_roots(control) != _participant_roots(anchored):
        raise ConfigError(
            f"control {input_.run_role!r} does not match anchored participant dataset roots",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="run controls on the same Phase 2 silos before comparing curves",
        )
    if control.participant_count_configured != anchored.participant_count_configured:
        raise ConfigError(
            f"control {input_.run_role!r} has a different participant count",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="run controls with the same participant count or mark them blocked",
        )
    if input_.run_role == "naive-fedavg":
        if control.lambda_anc != 0.0:
            raise ConfigError(
                "naive-fedavg control must set lambda_anc=0.0",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="pass the lambda_anc=0 negative-control report",
            )
        if control.lambda_sig != anchored.lambda_sig:
            raise ConfigError(
                "naive-fedavg control must keep lambda_sig matched to anchored run",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="only vary the anchor when using naive-fedavg as lambda_anc ablation",
            )


def _default_blockers(
    points: Sequence[Phase2CurvePoint],
) -> tuple[Phase2BlockedComparison, ...]:
    covered_roles = {point.run_role for point in points}
    has_ablation = any(point.ablation_axis is not None for point in points)
    blockers: list[Phase2BlockedComparison] = []
    for comparison, reason in (
        (
            "local-only",
            "No matched local-only Phase 2 SO-100 run is published for the same silos, seed, model size, and eval budget.",
        ),
        (
            "centralized-pooled",
            "No centralized/pooled run is published; pooling these participant silos was not executed for this artifact.",
        ),
        (
            "naive-fedavg",
            "No matched lambda_anc=0 negative-control run is published for the same SO-100 silos, seed, model size, and eval budget.",
        ),
        (
            "fork-a",
            "The RFC-0005 Fork-A safe-degrade baseline has not been run on the Phase 2 SO-100 silos.",
        ),
    ):
        if comparison not in covered_roles:
            blockers.append(
                _blocked(
                    comparison=comparison,  # type: ignore[arg-type]
                    reason=reason,
                )
            )
    if not has_ablation:
        blockers.append(
            _blocked(
                comparison="lambda-anc-ablation",
                reason=(
                    "No matched lambda_anc, lambda_sig, participant-count, "
                    "inner-horizon, or model-scale ablation report is available."
                ),
            )
        )
    return tuple(blockers)


def _blocked(*, comparison: Phase2Comparison, reason: str) -> Phase2BlockedComparison:
    return Phase2BlockedComparison(
        comparison=comparison,
        blocker_source="GitHub issue #205 Phase 2 acceptance audit",
        reason=reason,
        required_match=(
            "same Phase 2 SO-100 participant dataset roots",
            "same launcher seed/defaults unless explicitly varied by the ablation",
            "same compact model size and round budget where possible",
            "same downstream planner/eval budget for eval comparisons",
        ),
        issue_url="https://github.com/AbdelStark/Lensemble/issues/205",
    )


def _dedupe_source_reports(
    refs: Sequence[Phase2SourceReportRef],
) -> list[Phase2SourceReportRef]:
    seen: set[str] = set()
    deduped: list[Phase2SourceReportRef] = []
    for ref in refs:
        if ref.sha256 in seen:
            continue
        seen.add(ref.sha256)
        deduped.append(ref)
    return deduped


def _participant_roots(report: ClaimMVPReport) -> dict[str, str]:
    return {
        participant.participant_id: participant.dataset_root
        for participant in report.participants
    }


def _claim_match_notes(report: ClaimMVPReport) -> tuple[str, ...]:
    participants = ",".join(sorted(_participant_roots(report)))
    return (
        f"participants={participants}",
        f"committed_rounds={report.committed_rounds}",
        f"lambda_sig={report.lambda_sig}",
        f"lambda_anc={report.lambda_anc}",
    )


def _row_id(*parts: object) -> str:
    return "-".join(_slug(str(part)) for part in parts if str(part))


def _slug(text: str) -> str:
    lowered = text.lower().replace("_", "-")
    return "".join(ch if ch.isalnum() or ch in ".-" else "-" for ch in lowered)
