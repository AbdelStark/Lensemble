"""lensemble.artifacts.schema — the checkpoint header schema (docs/rfcs/RFC-0010 2).

``CheckpointHeader`` is the pydantic v2 sidecar (``header.json``) validated before any tensor is
materialized. ``schema_version`` is the first-validated field; ``extra="forbid"`` makes a malformed
header fail closed. The canonical data-model definition is 03-data-model 10.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, field_validator

from lensemble.errors import LensembleErrorCode, SchemaVersionMismatch

if TYPE_CHECKING:
    from collections.abc import Callable

# Current on-disk header schema version (conventions 10).
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


# --- the ordered, forward-compatible migration chain (RFC-0010 §7 / 03 §15; #33) ---

# Ordered, append-only chain: `_HEADER_MIGRATIONS[N]` upgrades a schema-version-N header dict to N+1. It
# is empty at SCHEMA_VERSION == 1 (no prior versions). A reader is forward-compatible: it accepts any
# `schema_version <= SCHEMA_VERSION` by running the chain, and fails closed on an unknown/too-new version.
#
# To bump the schema (e.g. v1 -> v2), in one PR:
#   1. def migrate_v1_to_v2(header: dict) -> dict: ...    # transform fields; do NOT set schema_version
#   2. _HEADER_MIGRATIONS[1] = migrate_v1_to_v2
#   3. SCHEMA_VERSION = 2
#   4. a round-trip migration test (07 §2.10), and a Keep-a-Changelog `Changed` entry, e.g.:
#        ### Changed
#        - `artifacts`: `CheckpointHeader` schema_version 1 -> 2 (<field>); readers migrate v1 on load
#          (`INV-CHECKPOINT-HASH` unaffected). [area:artifacts]
_HEADER_MIGRATIONS: dict[int, "Callable[[dict[str, Any]], dict[str, Any]]"] = {}


def _schema_mismatch(file_version: object, reader_max: int) -> SchemaVersionMismatch:
    err = SchemaVersionMismatch(
        f"checkpoint header schema_version {file_version!r} is unreadable by this reader "
        f"(supported: 1..{reader_max})",
        code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
        remediation=f"upgrade lensemble to read this artifact, or re-export at schema_version <= {reader_max}",
    )
    err.file_schema_version = file_version  # type: ignore[attr-defined]
    err.reader_max_version = reader_max  # type: ignore[attr-defined]
    return err


def migrate_header(
    raw: dict[str, Any],
    *,
    target: int = SCHEMA_VERSION,
    migrations: dict[int, "Callable[[dict[str, Any]], dict[str, Any]]"] | None = None,
) -> dict[str, Any]:
    """Bring an older header dict up to ``target`` via the ordered migration chain (RFC-0010 §7).

    Fail-closed: a non-integer / unknown (``< 1``) / too-new (``> target``) version, or a missing chain
    link, raises :class:`~lensemble.errors.SchemaVersionMismatch` (``file_schema_version`` /
    ``reader_max_version`` set); the header body is never best-effort parsed. An at-target header passes
    through unchanged. Returns a header dict at ``target`` (each step bumps ``schema_version``).
    """
    chain = _HEADER_MIGRATIONS if migrations is None else migrations
    version = raw.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version < 1
        or version > target
    ):
        raise _schema_mismatch(version, target)
    out = dict(raw)
    current = version
    while current < target:
        migrate = chain.get(current)
        if migrate is None:
            raise _schema_mismatch(version, target)  # the chain is missing a step
        out = dict(migrate(out))
        current += 1
        out["schema_version"] = current
    return out
