"""CheckpointHeader schema validation (RFC-0010 2). Issue #31."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from lensemble.artifacts import SCHEMA_VERSION, CheckpointHeader, TensorEntry

_H = "a" * 64


def _header(**over: object) -> CheckpointHeader:
    base: dict[str, object] = dict(
        schema_version=SCHEMA_VERSION,
        content_hash=_H,
        parent_hash=None,
        wmcp_version="wmcp-1.0.0",
        round_index=0,
        config_hash=_H,
        param_groups=("encoder", "predictor"),
        tensor_manifest=(
            TensorEntry(
                name="encoder.w", group="encoder", dtype="float32", shape=(2, 2)
            ),
        ),
        weight_files=("weights.safetensors",),
        created_at=datetime.now(timezone.utc),
    )
    base.update(over)
    return CheckpointHeader(**base)  # type: ignore[arg-type]


def test_header_json_roundtrip() -> None:
    h = _header()
    restored = CheckpointHeader.model_validate_json(h.model_dump_json())
    assert restored == h


def test_extra_field_rejected() -> None:
    h = _header()
    payload = h.model_dump()
    payload["surprise"] = 1
    with pytest.raises(ValidationError):
        CheckpointHeader.model_validate(payload)


def test_missing_field_rejected() -> None:
    h = _header()
    payload = h.model_dump()
    del payload["wmcp_version"]
    with pytest.raises(ValidationError):
        CheckpointHeader.model_validate(payload)


def test_bad_length_content_hash_rejected() -> None:
    with pytest.raises(ValidationError):
        _header(content_hash="deadbeef")  # not 64 hex chars


def test_bad_schema_version_rejected() -> None:
    with pytest.raises(ValidationError):
        _header(schema_version=0)
