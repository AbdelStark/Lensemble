"""lensemble.artifacts.checkpoint — the save / load / verify lifecycle (docs/rfcs/RFC-0010 5).

An artifact directory holds a ``safetensors`` weight payload (encoder + predictor param groups, tensors
only, no pickle) and a ``header.json`` sidecar. ``save_checkpoint`` writes to a temporary directory and
atomically renames it into place, so a reader never sees a half-written artifact. ``load_checkpoint``
reads and validates the header (``schema_version`` first) before materializing any tensor.

The ``content_hash`` here is a deterministic, shard-independent provisional hash (SHA-256 over the
name-sorted tensor payload plus the structural header fields). The canonical cross-platform byte
procedure of RFC-0010 4, ``verify_hash``, the ``INV-ACTIONHEAD-LOCAL`` serialization check, and the
no-pickle/tamper rejection land in artifacts-hashing (#32) on this same path.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from safetensors import safe_open
from safetensors.torch import load_file, save, save_file

from lensemble.artifacts.schema import SCHEMA_VERSION, CheckpointHeader, TensorEntry
from lensemble.errors import ArtifactError, LensembleErrorCode, SchemaVersionMismatch

if TYPE_CHECKING:
    from torch import Tensor

_HEADER = "header.json"


def _dtype_token(t: "Tensor") -> str:
    return str(t.dtype).replace("torch.", "")


def _ordered(weights: Mapping[str, "Tensor"]) -> "OrderedDict[str, Tensor]":
    return OrderedDict((k, weights[k]) for k in sorted(weights))


def _tensor_manifest(weights: "OrderedDict[str, Tensor]") -> tuple[TensorEntry, ...]:
    return tuple(
        TensorEntry(
            name=name,
            group=name.split(".", 1)[0],
            dtype=_dtype_token(t),
            shape=tuple(t.shape),
        )
        for name, t in weights.items()
    )


def _structural_frame(
    *,
    schema_version: int,
    wmcp_version: str,
    round_index: int,
    parent_hash: str | None,
    param_groups: tuple[str, ...],
    weight_files: tuple[str, ...],
    manifest: tuple[TensorEntry, ...],
) -> bytes:
    payload = {
        "schema_version": schema_version,
        "wmcp_version": wmcp_version,
        "round_index": round_index,
        "parent_hash": parent_hash or "",
        "param_groups": sorted(param_groups),
        "weight_files": list(weight_files),
        "tensors": [[e.name, e.group, e.dtype, list(e.shape)] for e in manifest],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _provisional_content_hash(weights: "OrderedDict[str, Tensor]", frame: bytes) -> str:
    body = save({k: v.detach().cpu().contiguous() for k, v in weights.items()})
    return hashlib.sha256(body + b"\x00lensemble-artifact-v1\x00" + frame).hexdigest()


def _shard_plan(
    weights: "OrderedDict[str, Tensor]", shard_size_bytes: int | None
) -> list[list[str]]:
    names = list(weights)
    if shard_size_bytes is None:
        return [names]
    shards: list[list[str]] = []
    current: list[str] = []
    used = 0
    for name in names:
        nbytes = weights[name].element_size() * weights[name].nelement()
        if current and used + nbytes > shard_size_bytes:
            shards.append(current)
            current, used = [], 0
        current.append(name)
        used += nbytes
    if current:
        shards.append(current)
    return shards or [[]]


def save_checkpoint(
    artifact_dir: Path,
    weights: Mapping[str, "Tensor"],
    *,
    wmcp_version: str,
    round_index: int,
    config_hash: str,
    parent_hash: str | None,
    param_groups: tuple[str, ...] = ("encoder", "predictor"),
    shard_size_bytes: int | None = None,
) -> str:
    """Write a model artifact and return its ``content_hash`` (the value to commit, RFC-0010 5).

    Writes ``weights.safetensors`` (optionally sharded) and ``header.json`` into a temporary directory,
    then atomically renames it to ``artifact_dir`` so a reader never sees a partial write.
    """
    artifact_dir = Path(artifact_dir)
    ordered = _ordered(weights)
    plan = _shard_plan(ordered, shard_size_bytes)
    n = len(plan)
    weight_files = (
        ("weights.safetensors",)
        if n == 1
        else tuple(f"weights-{i:05d}-of-{n:05d}.safetensors" for i in range(n))
    )
    manifest = _tensor_manifest(ordered)
    frame = _structural_frame(
        schema_version=SCHEMA_VERSION,
        wmcp_version=wmcp_version,
        round_index=round_index,
        parent_hash=parent_hash,
        param_groups=param_groups,
        weight_files=weight_files,
        manifest=manifest,
    )
    content_hash = _provisional_content_hash(ordered, frame)
    header = CheckpointHeader(
        schema_version=SCHEMA_VERSION,
        content_hash=content_hash,
        parent_hash=parent_hash,
        wmcp_version=wmcp_version,
        round_index=round_index,
        config_hash=config_hash,
        param_groups=param_groups,
        tensor_manifest=manifest,
        weight_files=weight_files,
        created_at=datetime.now(timezone.utc),
    )

    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(
        tempfile.mkdtemp(prefix=f".{artifact_dir.name}.tmp-", dir=artifact_dir.parent)
    )
    try:
        for fname, names in zip(weight_files, plan, strict=True):
            shard = {name: ordered[name].detach().cpu().contiguous() for name in names}
            save_file(shard, str(tmp / fname))
        (tmp / _HEADER).write_text(header.model_dump_json(indent=2), encoding="utf-8")
        if artifact_dir.exists():
            raise ArtifactError(
                f"artifact_dir already exists: {artifact_dir}",
                code=LensembleErrorCode.ARTIFACT_INVALID,
                remediation="write to a fresh artifact directory; checkpoints are immutable once committed",
            )
        os.replace(tmp, artifact_dir)
    except BaseException:
        if tmp.exists():
            for child in tmp.iterdir():
                child.unlink()
            tmp.rmdir()
        raise
    return content_hash


def _read_header(artifact_dir: Path) -> CheckpointHeader:
    path = Path(artifact_dir) / _HEADER
    if not path.exists():
        raise ArtifactError(
            f"no {_HEADER} in {artifact_dir}",
            code=LensembleErrorCode.ARTIFACT_INVALID,
            remediation="point at a directory written by save_checkpoint",
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    version = raw.get("schema_version")
    if isinstance(version, int) and version > SCHEMA_VERSION:
        raise SchemaVersionMismatch(
            f"header schema_version {version} is newer than this reader ({SCHEMA_VERSION})",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation="upgrade lensemble to read this artifact, or re-export at the supported schema",
        )
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


def _recompute_hash(
    weights: "OrderedDict[str, Tensor]", header: CheckpointHeader
) -> str:
    frame = _structural_frame(
        schema_version=header.schema_version,
        wmcp_version=header.wmcp_version,
        round_index=header.round_index,
        parent_hash=header.parent_hash,
        param_groups=header.param_groups,
        weight_files=header.weight_files,
        manifest=header.tensor_manifest,
    )
    return _provisional_content_hash(weights, frame)


def load_checkpoint(artifact_dir: Path) -> tuple[dict[str, "Tensor"], CheckpointHeader]:
    """Validate the header (``schema_version`` first), verify the hash, then load tensors (RFC-0010 5)."""
    artifact_dir = Path(artifact_dir)
    header = _read_header(artifact_dir)
    merged: dict[str, "Tensor"] = {}
    for fname in header.weight_files:
        merged.update(load_file(str(artifact_dir / fname)))
    ordered = _ordered(merged)
    recomputed = _recompute_hash(ordered, header)
    if recomputed != header.content_hash:
        raise _integrity_error(artifact_dir, header.content_hash, recomputed)
    return dict(ordered), header


def verify(artifact_dir: Path, expected_hash: str | None = None) -> CheckpointHeader:
    """Header-and-hash-only integrity check (RFC-0010 5); used by public recomputation and ingress.

    Recomputes the content hash over the stored weights and asserts it equals the header
    ``content_hash`` (``INV-CHECKPOINT-HASH``); if ``expected_hash`` is given it must also match.
    """
    artifact_dir = Path(artifact_dir)
    header = _read_header(artifact_dir)
    merged: dict[str, "Tensor"] = {}
    for fname in header.weight_files:
        with safe_open(str(artifact_dir / fname), framework="pt") as f:  # type: ignore[no-untyped-call]
            for key in f.keys():
                merged[key] = f.get_tensor(key)
    recomputed = _recompute_hash(_ordered(merged), header)
    if recomputed != header.content_hash:
        raise _integrity_error(artifact_dir, header.content_hash, recomputed)
    if expected_hash is not None and expected_hash != header.content_hash:
        raise _integrity_error(artifact_dir, expected_hash, header.content_hash)
    return header


def _integrity_error(artifact_dir: Path, expected: str, got: str):
    from lensemble.errors import CheckpointIntegrityError

    err = CheckpointIntegrityError(
        f"content hash mismatch for {artifact_dir}: expected {expected}, got {got}",
        code=LensembleErrorCode.CHECKPOINT_INTEGRITY,
        remediation="the artifact is corrupt or tampered; do not load it (INV-CHECKPOINT-HASH)",
    )
    err.expected_hash = expected  # type: ignore[attr-defined]
    err.got_hash = got  # type: ignore[attr-defined]
    return err
