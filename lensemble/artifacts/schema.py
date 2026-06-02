"""lensemble.artifacts.schema — the checkpoint header schema (docs/rfcs/RFC-0010 2).

``CheckpointHeader`` is the pydantic v2 sidecar (``header.json``) validated before any tensor is
materialized. ``schema_version`` is the first-validated field; ``extra="forbid"`` makes a malformed
header fail closed. The canonical data-model definition is 03-data-model 10.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator

# Current on-disk header schema version (conventions 10). The migration chain is #33.
SCHEMA_VERSION: int = 1

_HEX64 = set("0123456789abcdef")


def _is_hex64(value: str) -> bool:
    return len(value) == 64 and all(c in _HEX64 for c in value)


class TensorEntry(BaseModel):
    """One stored tensor's structural contract (name/group/dtype/shape), name-sorted in the header."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str  # fully-qualified parameter name
    group: str  # one of the header's param_groups
    dtype: str  # canonical dtype token, e.g. "float32", "bfloat16"
    shape: tuple[int, ...]  # tensor shape


class CheckpointHeader(BaseModel):
    """The JSON header of a model artifact (RFC-0010 2). Frozen; rejects extra/missing fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: (
        int  # on-disk schema version (conventions 10); validated before body
    )
    content_hash: str  # SHA-256 (lowercase hex, 64 chars) over canonical weight bytes
    parent_hash: str | None  # previous round's content_hash; None at round 0
    wmcp_version: str  # pinned latent-contract version (INV-WMCP)
    round_index: int  # the round t these params belong to (>= 0)
    config_hash: str  # the RunManifest config content hash that produced them
    param_groups: tuple[
        str, ...
    ]  # e.g. ("encoder", "predictor"); action heads NEVER included
    tensor_manifest: tuple[
        TensorEntry, ...
    ]  # per-tensor name/dtype/shape/group, name-sorted
    weight_files: tuple[str, ...]  # ("weights.safetensors",) or the ordered shard list
    created_at: datetime  # UTC, RFC 3339 (informational; never hashed)

    @field_validator("schema_version")
    @classmethod
    def _schema_version_ge_1(cls, v: int) -> int:
        if v < 1:
            raise ValueError("schema_version must be >= 1")
        return v

    @field_validator("round_index")
    @classmethod
    def _round_index_ge_0(cls, v: int) -> int:
        if v < 0:
            raise ValueError("round_index must be >= 0")
        return v

    @field_validator("content_hash")
    @classmethod
    def _content_hash_hex64(cls, v: str) -> str:
        if not _is_hex64(v):
            raise ValueError("content_hash must be 64 lowercase hex characters")
        return v

    @field_validator("parent_hash")
    @classmethod
    def _parent_hash_hex64(cls, v: str | None) -> str | None:
        if v is not None and not _is_hex64(v):
            raise ValueError("parent_hash must be None or 64 lowercase hex characters")
        return v
