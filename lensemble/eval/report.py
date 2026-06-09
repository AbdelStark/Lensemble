"""lensemble.eval.report — the ``EvalReport`` on-disk reporting type (docs/spec/03-data-model §13.1).

The latent-MPC evaluation output: a frozen, ``extra="forbid"`` pydantic v2 document carrying ONLY scalar
metrics, hashes, and counts — never a raw observation / action / latent tensor (``INV-RESIDENCY``). The
field set is normative in [03-data-model §13.1] and the metric definitions in
[RFC-0005 §3-4](docs/rfcs/RFC-0005-evaluation.md); this module is the stable contract the harness (#52)
assembles and writes.

Validation (03 §13.1). The ``@model_validator`` enforces the documented ranges — ``0 <= success_rate <=
1``, ``effective_dim > 0``, ``probe_accuracy in [0, 1]`` when present — and raises
:class:`~lensemble.errors.EvaluationError` (code ``EVALUATION_FAILED``) on a violation, never a silent
coercion. (pydantic v2 propagates a non-``ValueError`` raised inside a validator, so the typed
``EvaluationError`` surfaces directly to the caller.) :func:`parse_eval_report` checks ``schema_version``
*first* and raises :class:`~lensemble.errors.SchemaVersionMismatch` on a too-new document — fail-closed,
never a best-effort parse — mirroring ``parse_frame_drift_record``.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from lensemble.errors import EvaluationError, LensembleErrorCode, SchemaVersionMismatch

# Current on-disk EvalReport schema version (03 §15; conventions 10).
EVAL_REPORT_SCHEMA_VERSION = 1


class EvalReport(BaseModel):
    """The latent-MPC evaluation output (03 §13.1; RFC-0005 §3-4). Frozen; unknown fields rejected.

    Carries the headline downstream metric (``success_rate`` + planning cost), the collapse guard
    (``effective_dim``), the supporting probe accuracy, and the hashes that bind the report to its
    checkpoint and ``RunManifest``. ``INV-RESIDENCY``: every field is a scalar / hash / count — no field
    is a tensor, so a raw observation/action/latent can never structurally reach this report sink.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: (
        int  # on-disk schema version (03 §15); validated first by parse_eval_report
    )
    checkpoint_hash: (
        str  # the ModelArtifact content_hash evaluated (INV-CHECKPOINT-HASH)
    )
    env_id: str  # the resolved eval environment id
    planner: Literal["cem", "icem", "mppi"]  # the planner family used
    success_rate: float  # held-out MPC success fraction, in [0, 1] (RFC-0005 §3)
    planning_samples: int  # planner samples drawn per action
    time_per_action_ms: float  # mean planning wall-cost per action, milliseconds
    effective_dim: (
        float  # embedding-covariance effective dimension (collapse guard); > 0
    )
    probe_accuracy: (
        float | None
    )  # supporting linear-probe accuracy in [0, 1], or None if unwired
    state_probe_r2: float | None = (
        None  # binding RFC-0017 ground-truth state probe R2, or None if unavailable
    )
    run_manifest_hash: str  # binds the report to its RunManifest

    @model_validator(mode="after")
    def _enforce_ranges(self) -> "EvalReport":
        """Enforce the 03 §13.1 ranges; an out-of-range value is an ``EvaluationError``, never a clamp."""
        if not 0.0 <= self.success_rate <= 1.0:
            raise EvaluationError(
                f"success_rate must be in [0, 1], got {self.success_rate}",
                code=LensembleErrorCode.EVALUATION_FAILED,
                remediation="success_rate is a fraction of held-out episodes; fix the metric computation",
            )
        if self.effective_dim <= 0.0:
            raise EvaluationError(
                f"effective_dim must be > 0, got {self.effective_dim}",
                code=LensembleErrorCode.EVALUATION_FAILED,
                remediation="effective_dim is a participation ratio in [1, d]; a non-positive value is a bug",
            )
        if self.probe_accuracy is not None and not 0.0 <= self.probe_accuracy <= 1.0:
            raise EvaluationError(
                f"probe_accuracy must be in [0, 1] when present, got {self.probe_accuracy}",
                code=LensembleErrorCode.EVALUATION_FAILED,
                remediation="probe_accuracy is a held-out accuracy fraction; fix the probe metric",
            )
        if self.state_probe_r2 is not None and (
            not math.isfinite(self.state_probe_r2) or self.state_probe_r2 > 1.0
        ):
            raise EvaluationError(
                f"state_probe_r2 must be finite and <= 1 when present, got {self.state_probe_r2}",
                code=LensembleErrorCode.EVALUATION_FAILED,
                remediation="state_probe_r2 is the held-out ground-truth state R2; fix the probe computation",
            )
        return self


def parse_eval_report(raw: dict[str, Any]) -> EvalReport:
    """Parse a raw dict back to an :class:`EvalReport`; raise on a too-new ``schema_version`` (03 §13.1).

    Checks ``schema_version`` FIRST (mirroring ``parse_frame_drift_record``): a non-integer or a version
    exceeding ``EVAL_REPORT_SCHEMA_VERSION`` raises :class:`~lensemble.errors.SchemaVersionMismatch` —
    fail-closed, the body is never best-effort parsed. Otherwise ``model_validate`` runs (and its
    ``@model_validator`` enforces the field ranges).
    """
    version = raw.get("schema_version")
    if not isinstance(version, int) or version > EVAL_REPORT_SCHEMA_VERSION:
        raise SchemaVersionMismatch(
            f"eval-report schema_version {version!r} exceeds reader max {EVAL_REPORT_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation=f"read with a build supporting schema_version <= {version!r}",
        )
    return EvalReport.model_validate(raw)
