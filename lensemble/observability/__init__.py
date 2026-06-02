"""lensemble.observability — structured logging, metrics, redaction (docs/rfcs/RFC-0015).

The redaction guard (``INV-RESIDENCY``) is the single allow-list every log/metric/diagnostic record
passes before a sink write. Structured logging and the metric taxonomy land with #57 / #58.
"""

from __future__ import annotations

from lensemble.observability.logging import (
    LOG_SCHEMA_VERSION,
    JsonFormatter,
    LogLevel,
    LogRecord,
    emit_log,
    parse_log_record,
)
from lensemble.observability.metrics import (
    FRAME_DRIFT_RECORD_SCHEMA_VERSION,
    METRIC_SCHEMA_VERSION,
    FrameDriftRecord,
    MetricSample,
    PairAngle,
    PairResidual,
    canonical_unit,
    emit_diagnostic,
    emit_metric,
    parse_frame_drift_record,
    parse_metric_sample,
)
from lensemble.observability.redaction import redact, redact_record

__all__ = [
    "redact",
    "redact_record",
    "LogRecord",
    "LogLevel",
    "emit_log",
    "parse_log_record",
    "JsonFormatter",
    "LOG_SCHEMA_VERSION",
    "MetricSample",
    "emit_metric",
    "parse_metric_sample",
    "canonical_unit",
    "METRIC_SCHEMA_VERSION",
    "FrameDriftRecord",
    "PairAngle",
    "PairResidual",
    "emit_diagnostic",
    "parse_frame_drift_record",
    "FRAME_DRIFT_RECORD_SCHEMA_VERSION",
]
