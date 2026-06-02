"""Checkpoint save/load/verify lifecycle (RFC-0010 5). Issue #31."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from lensemble.artifacts import load_checkpoint, save_checkpoint, verify
from lensemble.errors import (
    ArtifactError,
    CheckpointIntegrityError,
    SchemaVersionMismatch,
)


def _weights() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "encoder.patch_embed.weight": torch.randn(4, 3, dtype=torch.float32),
        "encoder.norm.bias": torch.randn(4, dtype=torch.float32),
        "predictor.proj.weight": torch.randn(4, 4, dtype=torch.float32),
    }


def _save(d: Path, w: dict[str, torch.Tensor], **kw: Any) -> str:
    return save_checkpoint(
        d,
        w,
        wmcp_version="wmcp-1.0.0",
        round_index=0,
        config_hash="b" * 64,
        parent_hash=None,
        **kw,
    )


def test_save_load_roundtrip_bitwise(tmp_path: Path) -> None:
    w = _weights()
    h = _save(tmp_path / "ckpt", w)
    loaded, header = load_checkpoint(tmp_path / "ckpt")
    assert header.content_hash == h
    assert set(loaded) == set(w)
    for k in w:
        assert torch.equal(loaded[k], w[k])  # fp32, exact, no atol/rtol
    assert verify(tmp_path / "ckpt").content_hash == h


def test_sharded_and_unsharded_both_roundtrip(tmp_path: Path) -> None:
    w = _weights()
    _save(tmp_path / "u", w)
    _save(tmp_path / "s", w, shard_size_bytes=16)  # forces multiple shards
    lu, hu = load_checkpoint(tmp_path / "u")
    ls, hs = load_checkpoint(tmp_path / "s")
    assert len(hu.weight_files) == 1 and len(hs.weight_files) > 1
    # RFC-0010 4 (#32): the canonical content hash is shard-independent.
    assert hu.content_hash == hs.content_hash
    for k in w:
        assert torch.equal(lu[k], w[k]) and torch.equal(ls[k], w[k])


def test_save_is_atomic(tmp_path: Path) -> None:
    _save(tmp_path / "ckpt", _weights())
    # no temporary directory residue; final directory is complete
    assert not list(tmp_path.glob(".ckpt.tmp-*"))
    files = {p.name for p in (tmp_path / "ckpt").iterdir()}
    assert "header.json" in files and "weights.safetensors" in files


def test_existing_dir_refused(tmp_path: Path) -> None:
    _save(tmp_path / "ckpt", _weights())
    with pytest.raises(ArtifactError):
        _save(tmp_path / "ckpt", _weights())  # checkpoints are immutable once committed


def test_tamper_detected(tmp_path: Path) -> None:
    _save(tmp_path / "ckpt", _weights())
    # overwrite a weight file with different (valid safetensors) bytes
    save_file(
        {"encoder.patch_embed.weight": torch.ones(4, 3)},
        str(tmp_path / "ckpt" / "weights.safetensors"),
    )
    with pytest.raises(CheckpointIntegrityError):
        load_checkpoint(tmp_path / "ckpt")


def test_schema_too_new_rejected(tmp_path: Path) -> None:
    _save(tmp_path / "ckpt", _weights())
    hp = tmp_path / "ckpt" / "header.json"
    raw = json.loads(hp.read_text())
    raw["schema_version"] = 999
    hp.write_text(json.dumps(raw))
    with pytest.raises(SchemaVersionMismatch):
        load_checkpoint(tmp_path / "ckpt")
