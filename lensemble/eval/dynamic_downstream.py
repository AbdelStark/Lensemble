"""Dynamic-env downstream report for RFC-0017.

This report is intentionally separate from ``phase3_downstream.py``. The old Phase 3 SO-100 container is
locked to the #96/#244 closed-loop deferral; the dynamic env runs an in-process control world and reports
the binding ground-truth ``state_probe_r2`` plus a populated, non-binding true-state success rate.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lensemble.errors import (
    ConfigError,
    EvaluationError,
    LensembleErrorCode,
    SchemaVersionMismatch,
)

DYNAMIC_ENV_DOWNSTREAM_REPORT_SCHEMA_VERSION = 1


class DynamicEnvCheckpointRef(BaseModel):
    """Checkpoint identity for one dynamic-env control row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class DynamicEnvControlReport(BaseModel):
    """One dynamic-env control row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(min_length=1)
    checkpoint: DynamicEnvCheckpointRef
    state_probe_r2: float
    binding_metric: Literal["state_probe_r2"] = "state_probe_r2"
    success_rate: float
    success_rate_role: Literal["reported_non_binding"] = "reported_non_binding"
    skill_vs_identity: float | None = None
    latent_goal_success_rate: float | None = None
    effective_rank: float | None = None
    metric_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def _ranges_and_boundary(self) -> "DynamicEnvControlReport":
        for field in ("state_probe_r2", "success_rate"):
            value = getattr(self, field)
            if not math.isfinite(value):
                raise EvaluationError(
                    f"dynamic-env {field} must be finite, got {value!r}",
                    code=LensembleErrorCode.EVALUATION_FAILED,
                    remediation="write only finite scalar metric values",
                )
        if self.state_probe_r2 > 1.0:
            raise EvaluationError(
                f"state_probe_r2 must be <= 1, got {self.state_probe_r2}",
                code=LensembleErrorCode.EVALUATION_FAILED,
                remediation="R2 is upper-bounded by 1; inspect the probe computation",
            )
        if not 0.0 <= self.success_rate <= 1.0:
            raise EvaluationError(
                f"success_rate must be in [0, 1], got {self.success_rate}",
                code=LensembleErrorCode.EVALUATION_FAILED,
                remediation="success_rate is a fraction of closed-loop episodes",
            )
        if (
            "gameable" not in self.metric_boundary
            or "supporting" not in self.metric_boundary
        ):
            raise ConfigError(
                "dynamic-env control metric_boundary must label latent metrics as supporting/gameable",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="state that latent-MPC/skill metrics are supporting and gameable",
            )
        return self


class DynamicEnvDownstreamEvalReport(BaseModel):
    """Schema-versioned dynamic-env downstream report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    generated_at: datetime
    task_env_id: Literal["kinematic://swipe-dot"]
    held_out_data_ref: str = Field(min_length=1)
    controls: tuple[DynamicEnvControlReport, ...] = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)
    raw_data_in_report: Literal[False] = False
    source_report_uri: str = Field(min_length=1)

    @model_validator(mode="after")
    def _cross_check(self) -> "DynamicEnvDownstreamEvalReport":
        if self.schema_version != DYNAMIC_ENV_DOWNSTREAM_REPORT_SCHEMA_VERSION:
            raise SchemaVersionMismatch(
                f"dynamic-env downstream report schema_version {self.schema_version!r} exceeds reader max "
                f"{DYNAMIC_ENV_DOWNSTREAM_REPORT_SCHEMA_VERSION}",
                code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
                remediation="read with a build supporting this dynamic-env downstream report schema",
            )
        required = (
            "synthetic control env",
            "state_probe_r2",
            "binding",
            "gameable",
            "paper-scale",
        )
        missing = [phrase for phrase in required if phrase not in self.claim_boundary]
        if missing:
            raise ConfigError(
                f"dynamic-env claim_boundary missing required phrases: {missing}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="carry RFC-0017's synthetic-env, binding-R2, gameable-metric, and no-paper-scale boundary",
            )
        return self


def parse_dynamic_env_downstream_eval_report(
    raw: dict[str, object],
) -> DynamicEnvDownstreamEvalReport:
    """Parse a dynamic-env report, gating future schemas first."""

    version = raw.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version > DYNAMIC_ENV_DOWNSTREAM_REPORT_SCHEMA_VERSION
    ):
        raise SchemaVersionMismatch(
            f"dynamic-env downstream report schema_version {version!r} exceeds reader max "
            f"{DYNAMIC_ENV_DOWNSTREAM_REPORT_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation="read with a build supporting this dynamic-env downstream report schema",
        )
    return DynamicEnvDownstreamEvalReport.model_validate(raw)


def load_dynamic_env_downstream_eval_report(
    path: Path,
) -> DynamicEnvDownstreamEvalReport:
    """Load and validate a dynamic-env downstream report JSON file."""

    return parse_dynamic_env_downstream_eval_report(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def to_dynamic_env_downstream_eval_report_json(
    report: DynamicEnvDownstreamEvalReport,
) -> str:
    """Canonical JSON for a dynamic-env downstream report."""

    return json.dumps(
        report.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def write_dynamic_env_downstream_eval_report(
    report: DynamicEnvDownstreamEvalReport, path: Path
) -> Path:
    """Write a validated dynamic-env downstream report as canonical JSON."""

    parse_dynamic_env_downstream_eval_report(report.model_dump(mode="json"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        to_dynamic_env_downstream_eval_report_json(report) + "\n", encoding="utf-8"
    )
    return path
