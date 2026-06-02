"""lensemble.observability.logging — structured JSON logging (RFC-0015 1 / 05-observability 1).

A fixed-schema :class:`LogRecord` (pydantic v2), the :func:`emit_log` facade that routes every record's
``payload`` through the redaction guard before a single JSONL line is appended, and a stdlib
:class:`JsonFormatter`. ``event`` is a closed, additive, dotted vocabulary (``round.open``,
``agg.outer_step``, ...); a rename is a ``schema_version`` bump. An ``ERROR`` record always carries a
``code`` from ``LensembleErrorCode`` so the error catalog is auditable from the logs alone.

Residency (``INV-RESIDENCY``): ``payload`` carries only emittable scalars/hashes/counts; a raw
observation/action/embedding reaching it raises :class:`~lensemble.errors.ResidencyViolation` (delegated
to ``observability.redaction``) and no line is written (fail-closed).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lensemble.errors import LensembleErrorCode, SchemaVersionMismatch
from lensemble.observability.redaction import redact_record

LOG_SCHEMA_VERSION = 1


class LogLevel(str, Enum):
    """Normative log levels (RFC-0015 1): ``WARN`` = recovered/degraded; ``ERROR`` = round/run-fatal."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


class LogRecord(BaseModel):
    """One structured log line (RFC-0015 1 / 05 1.1). Frozen; unknown fields rejected."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=LOG_SCHEMA_VERSION, ge=1)
    timestamp: datetime  # RFC 3339 UTC, microsecond precision
    level: LogLevel
    event: str  # dotted vocabulary, e.g. round.open / agg.outer_step
    logger: str
    correlation_id: str
    round: int | None = None
    participant_id: str | None = None
    code: str | None = None  # a LensembleErrorCode value; REQUIRED on ERROR
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _error_carries_a_code(self) -> LogRecord:
        if self.level is LogLevel.ERROR and self.code is None:
            raise ValueError(
                "an ERROR log record must carry a LensembleErrorCode `code`"
            )
        return self


def to_json(record: LogRecord) -> str:
    """Serialize a record to one canonical JSON line (sorted keys, compact, ASCII)."""
    return json.dumps(
        record.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def parse_log_record(line: str) -> LogRecord:
    """Parse one JSONL line back to a :class:`LogRecord`.

    Raises :class:`~lensemble.errors.SchemaVersionMismatch` when ``schema_version`` exceeds this reader's
    max (never a best-effort parse).
    """
    raw = json.loads(line)
    version = raw.get("schema_version")
    if not isinstance(version, int) or version > LOG_SCHEMA_VERSION:
        raise SchemaVersionMismatch(
            f"log schema_version {version!r} exceeds reader max {LOG_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation=f"read with a build supporting schema_version <= {version!r}",
        )
    return LogRecord.model_validate(raw)


def emit_log(
    level: LogLevel | str,
    event: str,
    *,
    run_dir: Path,
    correlation_id: str,
    logger: str = "lensemble",
    round: int | None = None,
    participant_id: str | None = None,
    code: str | None = None,
    **payload: Any,
) -> LogRecord:
    """Append one redacted JSON log line to ``<run_dir>/lensemble.log.jsonl`` (RFC-0015 1).

    ``payload`` is passed through the redaction guard *before* any write; a non-emittable value raises
    :class:`~lensemble.errors.ResidencyViolation` and no line is written. An ``ERROR`` level without
    ``code`` is rejected at construction.
    """
    safe_payload = redact_record(
        payload
    )  # INV-RESIDENCY: fail-closed before opening the file
    record = LogRecord(
        timestamp=datetime.now(timezone.utc),
        level=LogLevel(level),
        event=event,
        logger=logger,
        correlation_id=correlation_id,
        round=round,
        participant_id=participant_id,
        code=code,
        payload=safe_payload,
    )
    path = Path(run_dir) / "lensemble.log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as sink:
        sink.write(to_json(record) + "\n")
    return record


class JsonFormatter(logging.Formatter):
    """A stdlib ``logging.Formatter`` that serializes a structured record to one JSON line.

    Attach a :class:`LogRecord` as ``record.lensemble_record`` and it is emitted verbatim; otherwise a
    minimal record is built from the stdlib level/message so existing stdlib loggers still produce JSONL.
    """

    def format(self, record: logging.LogRecord) -> str:
        structured = getattr(record, "lensemble_record", None)
        if isinstance(structured, LogRecord):
            return to_json(structured)
        level = record.levelname if record.levelname in LogLevel.__members__ else "INFO"
        minimal = LogRecord(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel[level],
            event=record.name,
            logger=record.name,
            correlation_id=getattr(record, "correlation_id", "-"),
        )
        return to_json(minimal)
