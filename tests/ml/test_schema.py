"""CheckpointHeader schema validation (RFC-0010 2). Issue #31."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from lensemble.artifacts import (
    SCHEMA_VERSION,
    CheckpointHeader,
    ModelArchDescriptor,
    TensorEntry,
    migrate_header,
)
from lensemble.errors import SchemaVersionMismatch

_H = "a" * 64


def _arch() -> ModelArchDescriptor:
    return ModelArchDescriptor(
        d=8,
        depth=1,
        num_heads=2,
        num_tokens=4,
        in_channels=3,
        num_frames=2,
        image_size=4,
        patch_size=2,
        tubelet=2,
        mlp_ratio=2.0,
        wmcp_version="wmcp-1.0.0",
    )


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
        model_arch=_arch(),  # schema v2: the self-describing architecture descriptor (#171)
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


def test_schema_at_reader_version_passes_through() -> None:
    # At the reader version (SCHEMA_VERSION == 2): passes through and round-trips JSON without loss.
    header = _header()
    raw = header.model_dump(mode="json")
    migrated = migrate_header(raw)
    assert migrated["schema_version"] == SCHEMA_VERSION
    assert CheckpointHeader.model_validate(migrated) == header


def test_schema_too_new_fails_closed() -> None:
    # current+1 (too-new) fails closed with the version metadata set.
    raw = _header().model_dump(mode="json")
    version = SCHEMA_VERSION + 1
    raw["schema_version"] = version
    with pytest.raises(SchemaVersionMismatch) as exc:
        migrate_header(raw)
    assert exc.value.file_schema_version == version  # type: ignore[attr-defined]
    assert exc.value.reader_max_version == SCHEMA_VERSION  # type: ignore[attr-defined]


def test_below_floor_version_fails_closed() -> None:
    # An unknown / below-floor version (0) is a version problem, fail closed.
    raw = _header().model_dump(mode="json")
    raw["schema_version"] = 0
    with pytest.raises(SchemaVersionMismatch) as exc:
        migrate_header(raw)
    assert exc.value.file_schema_version == 0  # type: ignore[attr-defined]


def test_v1_header_migrates_to_v2_with_model_arch_none() -> None:
    # A legacy v1 header (no model_arch) migrates up the chain: the no-op migrate_v1_to_v2 (#171) leaves
    # model_arch absent, so it validates as model_arch=None (a non-self-describing checkpoint).
    raw = _header().model_dump(mode="json")
    raw["schema_version"] = 1
    del raw["model_arch"]  # v1 headers carry no architecture descriptor
    migrated = migrate_header(raw)
    assert migrated["schema_version"] == SCHEMA_VERSION == 2
    header = CheckpointHeader.model_validate(migrated)
    assert header.model_arch is None


def test_migration_chain_applies_in_order() -> None:
    # Exercise the dispatcher with a synthetic v1->v2->v3 chain (production is at v1 with no migrations).
    applied: list[int] = []

    def v1_to_v2(h: dict) -> dict:
        applied.append(1)
        return {**h, "config_hash": "b" * 64}  # a representative field transform

    def v2_to_v3(h: dict) -> dict:
        applied.append(2)
        return dict(h)

    raw = _header().model_dump(mode="json")
    raw["schema_version"] = 1  # drive the synthetic v1->v2->v3 chain from the start
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
