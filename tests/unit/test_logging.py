"""Structured JSON logging: schema, level semantics, redaction (RFC-0015 1 / 05 1). Issue #57.

LogRecord round-trips through canonical JSONL; a too-new schema_version raises SchemaVersionMismatch; an
ERROR record without a code is rejected; a non-emittable payload value raises ResidencyViolation and no
line is written. Placed in tests/unit (pure-Python; the §8 gate does not collect tests/observability).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from lensemble.errors import ResidencyViolation, SchemaVersionMismatch
from lensemble.observability import (
    LOG_SCHEMA_VERSION,
    JsonFormatter,
    LogLevel,
    LogRecord,
    emit_log,
    parse_log_record,
)

_TS = datetime(2026, 6, 2, 12, 0, 0, 123456, tzinfo=timezone.utc)


def _record(level: LogLevel = LogLevel.INFO, code: str | None = None) -> LogRecord:
    return LogRecord(
        timestamp=_TS,
        level=level,
        event="round.open",
        logger="lensemble.federation",
        correlation_id="run-7:round-3",
        round=3,
        code=code,
        payload={"participants": 4, "global_hash": "deadbeef"},
    )


def test_log_record_round_trips_through_canonical_jsonl() -> None:
    record = _record()
    assert parse_log_record(json.dumps(record.model_dump(mode="json"))) == record


def test_too_new_schema_version_raises(tmp_path: Path) -> None:
    raw = _record().model_dump(mode="json")
    raw["schema_version"] = LOG_SCHEMA_VERSION + 1
    with pytest.raises(SchemaVersionMismatch):
        parse_log_record(json.dumps(raw))


def test_error_record_requires_a_code() -> None:
    with pytest.raises(ValidationError):
        _record(
            level=LogLevel.ERROR, code=None
        )  # ERROR must carry a LensembleErrorCode
    # ...but an ERROR with a code is fine
    assert _record(level=LogLevel.ERROR, code="round_failed").code == "round_failed"


def test_emit_log_writes_one_jsonl_line(tmp_path: Path) -> None:
    record = emit_log(
        LogLevel.INFO,
        "agg.outer_step",
        run_dir=tmp_path,
        correlation_id="run-1:round-0",
        round=0,
        grad_norm=3.4,
        participants=4,
    )
    path = tmp_path / "lensemble.log.jsonl"
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert parse_log_record(lines[0]) == record
    # a second emit appends (one line per record)
    emit_log(
        LogLevel.WARN, "agg.drift", run_dir=tmp_path, correlation_id="run-1:round-0"
    )
    assert len(path.read_text().splitlines()) == 2


def test_non_emittable_payload_raises_and_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(ResidencyViolation):
        emit_log(
            LogLevel.INFO,
            "agg.outer_step",
            run_dir=tmp_path,
            correlation_id="run-1:round-0",
            raw_delta=torch.zeros(4),  # a raw tensor must never reach the sink
        )
    assert not (
        tmp_path / "lensemble.log.jsonl"
    ).exists()  # fail-closed, no partial write


def test_emit_log_accepts_str_level(tmp_path: Path) -> None:
    record = emit_log("DEBUG", "round.open", run_dir=tmp_path, correlation_id="c")
    assert record.level is LogLevel.DEBUG


def test_json_formatter_serializes_structured_and_minimal() -> None:
    import logging as _logging

    formatter = JsonFormatter()
    structured = _logging.makeLogRecord({"lensemble_record": _record()})
    assert json.loads(formatter.format(structured))["event"] == "round.open"
    minimal = _logging.makeLogRecord({"name": "lensemble.x", "levelname": "WARN"})
    decoded = json.loads(formatter.format(minimal))
    assert decoded["level"] == "WARN" and decoded["logger"] == "lensemble.x"
