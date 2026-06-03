"""lensemble.artifacts.checkpoint — the save / load / verify lifecycle (docs/rfcs/RFC-0010 5).

An artifact directory holds a ``safetensors`` weight payload (encoder + predictor param groups, tensors
only, no pickle) and a ``header.json`` sidecar. ``save_checkpoint`` writes to a temporary directory and
atomically renames it into place, so a reader never sees a half-written artifact. ``load_checkpoint``
verifies the canonical content hash (``INV-CHECKPOINT-HASH``, ``hashing.content_hash``) before returning
tensors for downstream use.

``INV-ACTIONHEAD-LOCAL`` at the serialization boundary (RFC-0010 6): ``save_checkpoint`` rejects any
tensor whose param group is outside the allowed shared set ``{encoder, predictor}`` — emitting an action
head into a *shared* artifact is a residency breach (``ResidencyViolation``, fail-closed) — before any
bytes are written.
"""

from __future__ import annotations

import os
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from safetensors.torch import save_file

from lensemble.artifacts.hashing import (
    StructuralFields,
    content_hash,
    load_weights_no_pickle,
    verify_hash,
)
from lensemble.artifacts.schema import (
    SCHEMA_VERSION,
    CheckpointHeader,
    ModelArchDescriptor,
    TensorEntry,
)
from lensemble.contracts import WMCP_VERSION
from lensemble.errors import ArtifactError, LensembleErrorCode, ResidencyViolation

if TYPE_CHECKING:
    from typing import Any

    from torch import Tensor

_HEADER = "header.json"
# The only param groups a shared artifact may carry (RFC-0010 6; INV-ACTIONHEAD-LOCAL).
_ALLOWED_GROUPS = frozenset({"encoder", "predictor"})


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


def _reject_action_heads(ordered: "OrderedDict[str, Tensor]") -> None:
    """INV-ACTIONHEAD-LOCAL: refuse any tensor outside {encoder, predictor} before writing."""
    for name in ordered:
        group = name.split(".", 1)[0]
        if group not in _ALLOWED_GROUPS:
            err = ResidencyViolation(
                f"tensor {name!r} (group {group!r}) is not a shared param group; per-embodiment "
                "action heads must never enter a shared artifact",
                code=LensembleErrorCode.RESIDENCY_VIOLATION,
                remediation="store only encoder/predictor params; persist action heads to private "
                "local state (INV-ACTIONHEAD-LOCAL)",
            )
            err.tensor_role = "action_head"  # type: ignore[attr-defined]
            err.boundary = "shared-artifact"  # type: ignore[attr-defined]
            raise err


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
    model_arch: ModelArchDescriptor | None = None,
) -> str:
    """Write a model artifact and return its ``content_hash`` (the value to commit, RFC-0010 5).

    Rejects any action-head tensor (``INV-ACTIONHEAD-LOCAL``) before writing, then writes
    ``weights.safetensors`` (optionally sharded) + ``header.json`` into a temporary directory and
    atomically renames it to ``artifact_dir``.

    ``model_arch`` (#171) is the optional self-describing encoder architecture (schema v2). It is HEADER
    metadata only — it is NOT fed into :class:`StructuralFields` / ``content_hash`` (like ``created_at``
    and ``config_hash``), so the returned hash is byte-identical with or without it
    (``INV-CHECKPOINT-HASH`` stays metadata-independent). ``None`` writes a non-self-describing checkpoint.
    """
    artifact_dir = Path(artifact_dir)
    ordered = _ordered(weights)
    _reject_action_heads(ordered)
    plan = _shard_plan(ordered, shard_size_bytes)
    n = len(plan)
    weight_files = (
        ("weights.safetensors",)
        if n == 1
        else tuple(f"weights-{i:05d}-of-{n:05d}.safetensors" for i in range(n))
    )
    fields = StructuralFields(
        schema_version=SCHEMA_VERSION,
        wmcp_version=wmcp_version,
        round_index=round_index,
        parent_hash=parent_hash,
        param_groups=param_groups,
    )
    ch = content_hash(ordered, fields)
    header = CheckpointHeader(
        schema_version=SCHEMA_VERSION,
        content_hash=ch,
        parent_hash=parent_hash,
        wmcp_version=wmcp_version,
        round_index=round_index,
        config_hash=config_hash,
        param_groups=param_groups,
        tensor_manifest=_tensor_manifest(ordered),
        weight_files=weight_files,
        created_at=datetime.now(timezone.utc),
        model_arch=model_arch,  # header metadata only; NOT in `fields`/content_hash (#171)
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
    return ch


def load_checkpoint(artifact_dir: Path) -> tuple[dict[str, "Tensor"], CheckpointHeader]:
    """Validate the header, verify the content hash, then return the tensors (RFC-0010 5).

    Rejects non-safetensors payloads and a hash mismatch (``CheckpointIntegrityError``) before the
    tensors are used downstream (``INV-CHECKPOINT-HASH``).
    """
    artifact_dir = Path(artifact_dir)
    header = verify_hash(artifact_dir)
    weights = load_weights_no_pickle(artifact_dir, header)
    return dict(_ordered(weights)), header


def verify(artifact_dir: Path, expected_hash: str | None = None) -> CheckpointHeader:
    """Header-and-hash-only integrity check (RFC-0010 5); used by public recomputation and ingress."""
    return verify_hash(artifact_dir, expected_hash)


def model_arch_from_config(cfg: "Any") -> ModelArchDescriptor:
    """Build the self-describing :class:`ModelArchDescriptor` from a model config (#171).

    Reads ``cfg.model`` EXACTLY as :func:`~lensemble.model.encoder.build_encoder` does so the descriptor
    records the architecture the committed weights were actually built with: ``latent_dim`` for ``d`` (with
    the legacy ``d`` alias fallback some SimpleNamespace test/CLI configs use), the ViT-shape fields, and
    the ``in_channels``/``mlp_ratio``/``wmcp_version`` ``getattr`` defaults. ``num_tokens`` is the derived
    token count (``(num_frames//tubelet) * (image_size//patch_size)**2``), matching the encoder.

    The result is HEADER metadata only — it never enters ``content_hash`` (``INV-CHECKPOINT-HASH``).
    """
    m = getattr(cfg, "model", None)
    if m is None:
        raise ArtifactError(
            "config has no `model` sub-config; cannot build a ModelArchDescriptor",
            code=LensembleErrorCode.ARTIFACT_INVALID,
            remediation="provide cfg.model with latent_dim, num_frames, image_size, patch_size, "
            "tubelet, depth, num_heads (the build_encoder fields, #171)",
        )
    d = int(m.latent_dim if hasattr(m, "latent_dim") else m.d)
    num_frames = int(m.num_frames)
    tubelet = int(m.tubelet)
    image_size = int(m.image_size)
    patch_size = int(m.patch_size)
    num_tokens = (num_frames // tubelet) * (image_size // patch_size) ** 2
    return ModelArchDescriptor(
        d=d,
        depth=int(m.depth),
        num_heads=int(m.num_heads),
        num_tokens=num_tokens,
        in_channels=int(getattr(m, "in_channels", 3)),
        num_frames=num_frames,
        image_size=image_size,
        patch_size=patch_size,
        tubelet=tubelet,
        mlp_ratio=float(getattr(m, "mlp_ratio", 4.0)),
        wmcp_version=str(getattr(m, "wmcp_version", WMCP_VERSION)),
    )
