"""Metric taxonomy emission: canonical names/units, finiteness, strict mode (RFC-0015 2). Issue #58.

Every taxonomy name resolves to its canonical unit; a NaN/Inf value raises EvaluationError; an unknown
name under strict mode raises ConfigError; MetricSample round-trips and a too-new schema_version raises.
Placed in tests/unit (the §8 gate does not collect tests/observability).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lensemble.errors import ConfigError, EvaluationError, SchemaVersionMismatch
from lensemble.observability import (
    METRIC_SCHEMA_VERSION,
    MetricSample,
    canonical_unit,
    emit_metric,
    parse_metric_sample,
)

_TAXONOMY_UNITS = {
    "loss/pred": "loss",
    "gauge/drift_angle_deg": "deg",
    "gauge/procrustes_residual": "frobenius",
    "gauge/effective_dim": "count",
    "fed/round_seconds": "s",
    "fed/comm_bytes": "bytes",
    "eval/success_rate": "fraction",
    "eval/time_per_action_ms": "ms",
}


def test_taxonomy_names_resolve_to_canonical_units() -> None:
    for name, unit in _TAXONOMY_UNITS.items():
        assert canonical_unit(name) == unit
    assert canonical_unit("not/a/metric") is None


def test_metric_sample_round_trips_and_rejects_future_schema() -> None:
    sample = MetricSample(
        name="gauge/drift_angle_deg",
        value=12.5,
        unit="deg",
        round=3,
        correlation_id="run-1:round-3",
        timestamp=datetime(2026, 6, 2, tzinfo=timezone.utc),
    )
    assert parse_metric_sample(json.dumps(sample.model_dump(mode="json"))) == sample
    raw = sample.model_dump(mode="json")
    raw["schema_version"] = METRIC_SCHEMA_VERSION + 1
    with pytest.raises(SchemaVersionMismatch):
        parse_metric_sample(json.dumps(raw))


def test_emit_metric_writes_line_with_canonical_unit(tmp_path: Path) -> None:
    sample = emit_metric(
        "gauge/drift_angle_deg", 12.5, run_dir=tmp_path, correlation_id="c", round=3
    )
    assert sample.unit == "deg"  # canonical unit applied even though none was passed
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 1 and parse_metric_sample(lines[0]) == sample


def test_unit_mismatch_for_known_name_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        emit_metric(
            "gauge/drift_angle_deg",
            1.0,
            run_dir=tmp_path,
            correlation_id="c",
            unit="rad",
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_value_raises_evaluation_error(bad: float, tmp_path: Path) -> None:
    with pytest.raises(EvaluationError):
        emit_metric("loss/pred", bad, run_dir=tmp_path, correlation_id="c")
    assert not (tmp_path / "metrics.jsonl").exists()  # nothing written


def test_non_scalar_value_raises(tmp_path: Path) -> None:
    import torch

    with pytest.raises(EvaluationError):
        emit_metric("loss/pred", torch.zeros(3), run_dir=tmp_path, correlation_id="c")  # type: ignore[arg-type]


def test_strict_mode_rejects_unknown_name(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        emit_metric(
            "custom/thing", 1.0, run_dir=tmp_path, correlation_id="c", strict=True
        )
    # non-strict: an unknown name is accepted (pre-1.0)
    sample = emit_metric(
        "custom/thing", 1.0, run_dir=tmp_path, correlation_id="c", unit="widget"
    )
    assert sample.unit == "widget"
