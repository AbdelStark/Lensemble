"""lensemble.artifacts.hashing — canonical-byte content hashing (docs/rfcs/RFC-0010 4).

``content_hash`` is a deterministic SHA-256 over the model weights plus the structural header fields,
independent of host OS, CPU endianness/architecture, torch build, and **shard count**. This is where
``INV-CHECKPOINT-HASH`` is enforced. Domain separation distinguishes an artifact hash from a Merkle-leaf
hash (RFC-0014) so the two surfaces never collide.

Cross-platform determinism contract: each tensor's element bytes are emitted **little-endian** in its
declared dtype, tensors are concatenated in ascending fully-qualified-name order, and each tensor is
framed ``len(name)||name||dtype_token||rank||shape_dims||element_bytes``; a trailing structural frame
binds ``schema_version``, ``wmcp_version``, ``round_index``, ``parent_hash`` (empty when ``None``), and
the sorted ``param_groups``. ``content_hash``/``config_hash``/``created_at`` are excluded.

Stored dtypes are restricted to ``{float32, bfloat16, float16}`` for v0.1.

RISK (RFC-0010 4): a torch/safetensors change to an exotic dtype's storage layout could make "raw element
bytes" ambiguous; mitigated by the dtype restriction and the cross-platform hash-stability release gate.

Note: ``weight_files`` is intentionally **excluded** from the hash so the hash is shard-independent
(RFC-0010 4 step 2; #32). RFC-0010 4 step 4 lists ``weight_files`` in the structural bind, which
contradicts shard-independence — the step-4 list should drop ``weight_files`` (flagged for an RFC fix).
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import torch

from lensemble.artifacts.schema import CheckpointHeader, migrate_header
from lensemble.errors import (
    ArtifactError,
    CheckpointIntegrityError,
    LensembleErrorCode,
)

if TYPE_CHECKING:
    from torch import Tensor

_DOMAIN = b"lensemble/artifact/v1"
_HEADER = "header.json"

# dtype -> (canonical token, little-endian numpy view dtype). bfloat16 has no numpy dtype, so its raw
# 2-byte bit pattern is taken via an int16 view.
_DTYPE_TOKEN = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}


@dataclass(frozen=True)
class StructuralFields:
    """The structural header fields bound into the content hash (RFC-0010 4 step 4)."""

    schema_version: int
    wmcp_version: str
    round_index: int
    parent_hash: str | None
    param_groups: tuple[str, ...]


def _le_element_bytes(t: "Tensor") -> bytes:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().astype("<i2", copy=False).tobytes()
    if t.dtype == torch.float16:
        return t.numpy().astype("<f2", copy=False).tobytes()
    if t.dtype == torch.float32:
        return t.numpy().astype("<f4", copy=False).tobytes()
    raise ArtifactError(
        f"unsupported stored dtype {t.dtype}; v0.1 allows {{float32, bfloat16, float16}}",
        code=LensembleErrorCode.ARTIFACT_INVALID,
        remediation="cast to float32/bfloat16/float16 before saving, or extend the canonical-dtype table",
    )


def content_hash(weights: Mapping[str, "Tensor"], fields: StructuralFields) -> str:
    """Canonical SHA-256 content hash (64 lowercase hex), shard-independent (RFC-0010 4)."""
    h = hashlib.sha256()
    h.update(_DOMAIN)
    h.update(b"\x00tensors\x00")
    for name in sorted(weights):  # ascending UTF-8 fully-qualified name
        t = weights[name]
        if t.dtype not in _DTYPE_TOKEN:
            _le_element_bytes(t)  # raises ArtifactError with the dtype message
        token = _DTYPE_TOKEN[t.dtype].encode("utf-8")
        name_b = name.encode("utf-8")
        h.update(struct.pack("<I", len(name_b)))
        h.update(name_b)
        h.update(struct.pack("<B", len(token)))
        h.update(token)
        shape = tuple(int(d) for d in t.shape)
        h.update(struct.pack("<I", len(shape)))
        for dim in shape:
            h.update(struct.pack("<Q", dim))
        h.update(_le_element_bytes(t))
    h.update(b"\x00struct\x00")
    frame = json.dumps(
        {
            "schema_version": fields.schema_version,
            "wmcp_version": fields.wmcp_version,
            "round_index": fields.round_index,
            "parent_hash": fields.parent_hash or "",
            "param_groups": sorted(fields.param_groups),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    h.update(struct.pack("<I", len(frame)))
    h.update(frame)
    return h.hexdigest()


def read_header(artifact_dir: Path) -> CheckpointHeader:
    """Read and validate ``header.json`` (``schema_version`` first); fail closed (RFC-0010 7)."""
    path = Path(artifact_dir) / _HEADER
    if not path.exists():
        raise ArtifactError(
            f"no {_HEADER} in {artifact_dir}",
            code=LensembleErrorCode.ARTIFACT_INVALID,
            remediation="point at a directory written by save_checkpoint",
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Gate the version first: a too-new / unknown version raises SchemaVersionMismatch (a version problem,
    # distinct from a bytes/field problem); an older known version is migrated up the chain (#33).
    raw = migrate_header(raw)
    try:
        return CheckpointHeader.model_validate(raw)
    except (
        Exception
    ) as exc:  # pydantic ValidationError -> typed ArtifactError (fail-closed)
        raise ArtifactError(
            f"malformed checkpoint header: {exc}",
            code=LensembleErrorCode.ARTIFACT_INVALID,
            remediation="the header is missing/extra a field or has a bad hash; the artifact is invalid",
        ) from exc


def load_weights_no_pickle(
    artifact_dir: Path, header: CheckpointHeader
) -> dict[str, "Tensor"]:
    """Load the safetensors weights, rejecting any non-safetensors (pickle/torch.save/npz) payload.

    The loader never falls back to ``torch.load``; a non-safetensors file raises
    :class:`~lensemble.errors.CheckpointIntegrityError` with a safetensors-only remediation.
    """
    from safetensors.torch import load_file

    artifact_dir = Path(artifact_dir)
    merged: dict[str, "Tensor"] = {}
    for fname in header.weight_files:
        try:
            merged.update(load_file(str(artifact_dir / fname)))
        except Exception as exc:
            raise CheckpointIntegrityError(
                f"{fname} is not a readable safetensors artifact: {exc}",
                code=LensembleErrorCode.CHECKPOINT_INTEGRITY,
                remediation="artifacts are safetensors only; pickle/torch.save/npz payloads are rejected",
            ) from exc
    return merged


def _integrity_error(
    artifact_dir: Path, expected: str, got: str
) -> CheckpointIntegrityError:
    err = CheckpointIntegrityError(
        f"content hash mismatch for {artifact_dir}: expected {expected}, got {got}",
        code=LensembleErrorCode.CHECKPOINT_INTEGRITY,
        remediation="the artifact is corrupt or tampered; do not load it (INV-CHECKPOINT-HASH)",
    )
    err.expected_hash = expected  # type: ignore[attr-defined]
    err.got_hash = got  # type: ignore[attr-defined]
    return err


def verify_hash(
    artifact_dir: Path, expected_hash: str | None = None
) -> CheckpointHeader:
    """Recompute the content hash over the stored weights and assert it equals the header hash.

    Raises :class:`~lensemble.errors.CheckpointIntegrityError` on a mismatch (and no tensors are returned
    for downstream use); if ``expected_hash`` is given it must also equal the header ``content_hash``.
    Enforces ``INV-CHECKPOINT-HASH``.
    """
    artifact_dir = Path(artifact_dir)
    header = read_header(artifact_dir)
    weights = load_weights_no_pickle(artifact_dir, header)
    fields = StructuralFields(
        schema_version=header.schema_version,
        wmcp_version=header.wmcp_version,
        round_index=header.round_index,
        parent_hash=header.parent_hash,
        param_groups=header.param_groups,
    )
    got = content_hash(weights, fields)
    if got != header.content_hash:
        raise _integrity_error(artifact_dir, header.content_hash, got)
    if expected_hash is not None and expected_hash != header.content_hash:
        raise _integrity_error(artifact_dir, expected_hash, header.content_hash)
    return header
