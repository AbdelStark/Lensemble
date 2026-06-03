"""lensemble.artifacts — the schema-versioned, hash-committed checkpoint format (docs/rfcs/RFC-0010)."""

from __future__ import annotations

from lensemble.artifacts.checkpoint import (
    load_checkpoint,
    model_arch_from_config,
    save_checkpoint,
    verify,
)
from lensemble.artifacts.hashing import StructuralFields, content_hash, verify_hash
from lensemble.artifacts.schema import (
    SCHEMA_VERSION,
    CheckpointHeader,
    ModelArchDescriptor,
    TensorEntry,
    migrate_header,
)

__all__ = [
    "CheckpointHeader",
    "ModelArchDescriptor",
    "TensorEntry",
    "SCHEMA_VERSION",
    "migrate_header",
    "save_checkpoint",
    "load_checkpoint",
    "verify",
    "content_hash",
    "verify_hash",
    "StructuralFields",
    "model_arch_from_config",
]
