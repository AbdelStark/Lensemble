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
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from lensemble.errors import (
    CheckpointIntegrityError,
    ConfigError,
    EvaluationError,
    LensembleErrorCode,
    ProbeError,
    SchemaVersionMismatch,
)
from lensemble.observability.redaction import redact

if TYPE_CHECKING:  # annotation-only; L1 observability never runtime-imports L4 gauge
    from collections.abc import Mapping

    from lensemble.gauge.drift import FrameDriftReport

METRIC_SCHEMA_VERSION = 1
FRAME_DRIFT_RECORD_SCHEMA_VERSION = 1

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


# --- the frame-drift diagnostic emission contract (RFC-0015 3 / 05 3; #60) -------------------------
#
# The headline artifact: per round, per participant pair, the recovered inter-frame rotation angle and
# Procrustes residual on the pinned probe, plus each participant's drift against the global model. The
# record is a pure function of the committed checkpoint hashes it names and the pinned probe identified by
# `probe_hash` (riding INV-AGG-DETERMINISM / INV-PROBE-PIN / INV-CHECKPOINT-HASH), so two honest emissions
# are byte-identical and an independent recomputation reproduces it. The frame-drift *algorithm*
# (`lensemble.gauge.frame_drift`) is consumed, not reimplemented here.


class PairAngle(BaseModel):
    """One unordered participant pair's recovered inter-frame rotation angle on the probe (RFC-0015 3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    participant_a: str  # the lexicographically smaller id (canonical c < c')
    participant_b: str
    angle_deg: float


class PairResidual(BaseModel):
    """One unordered participant pair's optimal-Procrustes residual on the probe (RFC-0015 3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    participant_a: str
    participant_b: str
    residual: float


class FrameDriftRecord(BaseModel):
    """A per-round frame-drift diagnostic record (RFC-0015 3 / 05 3). Frozen; unknown fields rejected.

    On-disk JSONL entry tagged ``record_kind == "frame_drift"``, carrying the pinned ``probe_hash`` and
    the committed global / per-participant checkpoint hashes it was computed from, plus the canonical
    (``c < c'``, deduplicated) pairwise angles and residuals and each participant's drift against the
    global model. Floats serialize via ``repr`` (the JSON shortest round-trippable decimal), so two
    emissions of identical fp64 values are byte-identical across platforms.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=FRAME_DRIFT_RECORD_SCHEMA_VERSION, ge=1)
    record_kind: Literal["frame_drift"] = "frame_drift"
    round_index: int
    probe_hash: str  # the pinned probe (INV-PROBE-PIN)
    global_checkpoint_hash: (
        str  # committed (theta_{t+1}, phi_{t+1}) content hash (INV-CHECKPOINT-HASH)
    )
    participant_checkpoint_hash: dict[
        str, str
    ]  # participant id -> committed checkpoint hash
    pairwise_angle_deg: tuple[PairAngle, ...]  # one entry per unordered pair, c < c'
    pairwise_residual: tuple[PairResidual, ...]
    drift_from_global_deg: dict[
        str, float
    ]  # participant id -> angle vs the global model
    pair_sampling: str  # the pair-sampling policy ("all_pairs" | a sampling scheme), so the figure stays honest
    timestamp: datetime


def _frame_drift_to_json(record: FrameDriftRecord) -> str:
    """One canonical JSON line (sorted keys, compact, ASCII; floats via ``repr``) — byte-stable."""
    return json.dumps(
        record.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def parse_frame_drift_record(line: str) -> FrameDriftRecord:
    """Parse one JSONL line back to a :class:`FrameDriftRecord`; raise on a too-new ``schema_version``."""
    raw = json.loads(line)
    version = raw.get("schema_version")
    if not isinstance(version, int) or version > FRAME_DRIFT_RECORD_SCHEMA_VERSION:
        raise SchemaVersionMismatch(
            f"frame-drift schema_version {version!r} exceeds reader max {FRAME_DRIFT_RECORD_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation=f"read with a build supporting schema_version <= {version!r}",
        )
    return FrameDriftRecord.model_validate(raw)


def emit_diagnostic(
    report: "FrameDriftReport",
    *,
    run_dir: Path,
    global_checkpoint_hash: str,
    participant_checkpoint_hash: "Mapping[str, str]",
    pinned_probe_hash: str,
    pair_sampling: str = "all_pairs",
    timestamp: datetime | None = None,
) -> FrameDriftRecord:
    """Emit one :class:`FrameDriftRecord` per outer round to ``<run_dir>/metrics.jsonl`` (RFC-0015 3).

    ``report`` is the :class:`~lensemble.gauge.drift.FrameDriftReport` from ``lensemble.gauge.frame_drift``
    (consumed structurally; this module never imports gauge at runtime). Pin binding fails closed: a
    ``report.probe_hash`` that differs from ``pinned_probe_hash`` raises
    :class:`~lensemble.errors.ProbeError`, and a participant named in the report without a committed
    checkpoint hash raises :class:`~lensemble.errors.CheckpointIntegrityError` — the diagnostic is never
    silently partial. Pairs are emitted in the canonical ``c < c'`` order, deduplicated. The record is
    routed through the redaction guard before the write. Cadence (after ``agg.aligned``, before
    ``commit.checkpoint``) is the runtime's; this facade owns the record and its reproducibility.
    """
    if report.probe_hash != pinned_probe_hash:
        raise ProbeError(
            f"frame-drift probe_hash {report.probe_hash!r} != the pinned {pinned_probe_hash!r}; "
            "refusing to emit a diagnostic against an unpinned probe",
            code=LensembleErrorCode.PROBE_INVALID,
            remediation="recompute the diagnostic against the RoundOpen-pinned probe (INV-PROBE-PIN)",
        )

    participants = {p.participant_a for p in report.pairs} | {
        p.participant_b for p in report.pairs
    }
    participants |= set(report.drift_from_global)
    missing = sorted(participants - set(participant_checkpoint_hash))
    if missing:
        raise CheckpointIntegrityError(
            f"no committed checkpoint hash for participant(s) {missing}; the diagnostic would be partial",
            code=LensembleErrorCode.CHECKPOINT_INTEGRITY,
            remediation="supply the committed checkpoint hash for every participant in the report",
        )

    angles: dict[tuple[str, str], float] = {}
    residuals: dict[tuple[str, str], float] = {}
    for pair in report.pairs:
        key = tuple(sorted((pair.participant_a, pair.participant_b)))
        angles[key] = pair.rotation_angle_deg  # type: ignore[index]
        residuals[key] = pair.procrustes_residual  # type: ignore[index]

    record = FrameDriftRecord(
        round_index=report.round_index,
        probe_hash=report.probe_hash,
        global_checkpoint_hash=global_checkpoint_hash,
        participant_checkpoint_hash=dict(participant_checkpoint_hash),
        pairwise_angle_deg=tuple(
            PairAngle(participant_a=a, participant_b=b, angle_deg=angles[(a, b)])
            for (a, b) in sorted(angles)
        ),
        pairwise_residual=tuple(
            PairResidual(participant_a=a, participant_b=b, residual=residuals[(a, b)])
            for (a, b) in sorted(residuals)
        ),
        drift_from_global_deg=dict(report.drift_from_global),
        pair_sampling=pair_sampling,
        timestamp=timestamp if timestamp is not None else datetime.now(timezone.utc),
    )
    redact(
        record.model_dump(mode="json"), field="frame_drift"
    )  # fail closed on any tensor leaf
    path = Path(run_dir) / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as sink:
        sink.write(_frame_drift_to_json(record) + "\n")
    return record
