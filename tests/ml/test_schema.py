"""CheckpointHeader schema validation (RFC-0010 2). Issue #31."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from lensemble.artifacts import (
    SCHEMA_VERSION,
    CheckpointHeader,
    TensorEntry,
    migrate_header,
)
from lensemble.errors import SchemaVersionMismatch

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


# --- forward-compatible migration chain (RFC-0010 §7 / 07 §2.10; #33) ---


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_schema_roundtrip_and_migration(offset: int) -> None:
    header = _header()
    raw = header.model_dump(mode="json")
    version = SCHEMA_VERSION + offset
    raw["schema_version"] = version

    if (
        offset == 0
    ):  # at the reader version: passes through and round-trips JSON without loss
        migrated = migrate_header(raw)
        assert migrated["schema_version"] == SCHEMA_VERSION
        assert CheckpointHeader.model_validate(migrated) == header
    else:  # current-1 (unknown / below floor) and current+1 (too-new) both fail closed
        with pytest.raises(SchemaVersionMismatch) as exc:
            migrate_header(raw)
        assert exc.value.file_schema_version == version  # type: ignore[attr-defined]
        assert exc.value.reader_max_version == SCHEMA_VERSION  # type: ignore[attr-defined]


def test_migration_chain_applies_in_order() -> None:
    # Exercise the dispatcher with a synthetic v1->v2->v3 chain (production is at v1 with no migrations).
    applied: list[int] = []

    def v1_to_v2(h: dict) -> dict:
        applied.append(1)
        return {**h, "config_hash": "b" * 64}  # a representative field transform

    def v2_to_v3(h: dict) -> dict:
        applied.append(2)
        return dict(h)

    raw = _header().model_dump(mode="json")  # schema_version == 1
    out = migrate_header(raw, target=3, migrations={1: v1_to_v2, 2: v2_to_v3})
    assert applied == [1, 2]  # ordered
    assert out["schema_version"] == 3
    assert out["config_hash"] == "b" * 64
    # the migrated header still validates as a CheckpointHeader (schema_version 3 >= 1)
    assert CheckpointHeader.model_validate(out).schema_version == 3


def test_missing_chain_link_fails_closed() -> None:
    raw = _header().model_dump(mode="json")  # v1
    with pytest.raises(SchemaVersionMismatch):
        migrate_header(raw, target=3, migrations={1: lambda h: dict(h)})  # no 2->3 step


def test_non_integer_version_is_a_version_problem() -> None:
    raw = _header().model_dump(mode="json")
    raw["schema_version"] = "1"  # a string, not an int
    with pytest.raises(SchemaVersionMismatch):
        migrate_header(raw)
