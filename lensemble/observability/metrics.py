"""lensemble.observability.metrics — the metric taxonomy emission (RFC-0015 2 / 05-observability 2).

A canonical, closed metric vocabulary with fixed units, emitted as a ``metrics.jsonl`` stream through the
redaction guard. Units are part of the contract (``gauge/drift_angle_deg`` is degrees, never radians). All
metric values are derived statistics — never raw tensors — and every sample passes the guard fail-closed.

A ``NaN``/``Inf`` value is rejected with :class:`~lensemble.errors.EvaluationError` (a metric that is not a
finite scalar is a bug, surfaced loudly, never silently logged). Under strict mode an unknown metric name
raises :class:`~lensemble.errors.ConfigError`; strict mode is off pre-1.0 and becomes the default at 1.0.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from lensemble.errors import (
    ConfigError,
    EvaluationError,
    LensembleErrorCode,
    SchemaVersionMismatch,
)
from lensemble.observability.redaction import redact

METRIC_SCHEMA_VERSION = 1

# The canonical taxonomy (RFC-0015 2 / 05 2): metric name -> its fixed unit. Closed and additive; a
# rename is a schema_version bump. Units are part of the contract.
_TAXONOMY: dict[str, str] = {
    "loss/pred": "loss",
    "loss/sigreg": "loss",
    "loss/anchor": "loss",
    "grad_norm": "l2",
    "gauge/drift_angle_deg": "deg",
    "gauge/procrustes_residual": "frobenius",
    "gauge/effective_dim": "count",
    "fed/round_seconds": "s",
    "fed/participants": "count",
    "fed/comm_bytes": "bytes",
    "fed/quant_ratio": "ratio",
    "dp/epsilon_cumulative": "epsilon",
    "dp/clip_fraction": "fraction",
    "eval/success_rate": "fraction",
    "eval/planning_samples": "count",
    "eval/time_per_action_ms": "ms",
}


def canonical_unit(name: str) -> str | None:
    """The canonical unit for a taxonomy metric name, or ``None`` if the name is not in the taxonomy."""
    return _TAXONOMY.get(name)


class MetricSample(BaseModel):
    """One metric sample (RFC-0015 2). Frozen; unknown fields rejected; ``value`` is a finite float."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=METRIC_SCHEMA_VERSION, ge=1)
    name: str
    value: float
    unit: str
    round: int | None = None
    step: int | None = None
    participant_id: str | None = None
    correlation_id: str
    timestamp: datetime


def to_json(sample: MetricSample) -> str:
    """Serialize a sample to one canonical JSON line (sorted keys, compact, ASCII)."""
    return json.dumps(
        sample.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def parse_metric_sample(line: str) -> MetricSample:
    """Parse one JSONL line back to a :class:`MetricSample`; raise on a too-new ``schema_version``."""
    raw = json.loads(line)
    version = raw.get("schema_version")
    if not isinstance(version, int) or version > METRIC_SCHEMA_VERSION:
        raise SchemaVersionMismatch(
            f"metric schema_version {version!r} exceeds reader max {METRIC_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation=f"read with a build supporting schema_version <= {version!r}",
        )
    return MetricSample.model_validate(raw)


def emit_metric(
    name: str,
    value: float,
    *,
    run_dir: Path,
    correlation_id: str,
    unit: str | None = None,
    round: int | None = None,
    step: int | None = None,
    participant_id: str | None = None,
    strict: bool = False,
) -> MetricSample:
    """Append one metric sample to ``<run_dir>/metrics.jsonl`` (RFC-0015 2).

    Precondition: ``value`` is a finite scalar; a ``NaN``/``Inf`` or non-scalar raises
    :class:`~lensemble.errors.EvaluationError`. For a taxonomy name the unit is the canonical one (a
    mismatched ``unit`` raises :class:`~lensemble.errors.ConfigError`); under ``strict`` an unknown name
    also raises ``ConfigError``. The value is routed through the redaction guard before the write.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(
            f"metric {name!r} value must be a finite scalar, got {type(value).__name__}",
            code=LensembleErrorCode.EVALUATION_FAILED,
            remediation="emit a derived scalar statistic, never a raw tensor or non-number",
        )
    fvalue = float(value)
    if not math.isfinite(fvalue):
        raise EvaluationError(
            f"metric {name!r} value is non-finite ({fvalue})",
            code=LensembleErrorCode.EVALUATION_FAILED,
            remediation="a NaN/Inf metric indicates a diverged computation; investigate, do not log it",
        )

    canonical = canonical_unit(name)
    if canonical is not None:
        if unit is not None and unit != canonical:
            raise ConfigError(
                f"metric {name!r} unit must be {canonical!r} (the contract), got {unit!r}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation=f"emit {name!r} in {canonical!r}; units are part of the taxonomy contract",
            )
        unit = canonical
    elif strict:
        raise ConfigError(
            f"unknown metric name {name!r} under strict mode",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="use a name from the RFC-0015 2 taxonomy, or disable strict mode pre-1.0",
        )
    elif unit is None:
        unit = "unknown"

    redact(fvalue, field=name)  # every sample passes the guard (a finite scalar passes)
    sample = MetricSample(
        name=name,
        value=fvalue,
        unit=unit,
        round=round,
        step=step,
        participant_id=participant_id,
        correlation_id=correlation_id,
        timestamp=datetime.now(timezone.utc),
    )
    path = Path(run_dir) / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as sink:
        sink.write(to_json(sample) + "\n")
    return sample
