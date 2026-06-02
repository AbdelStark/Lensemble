"""Canonical content hashing, tamper detection, no-pickle, action-head exclusion (RFC-0010 4/6). #32."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lensemble.artifacts import (
    StructuralFields,
    content_hash,
    load_checkpoint,
    save_checkpoint,
    verify,
)
from lensemble.errors import CheckpointIntegrityError, ResidencyViolation


def _w() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "encoder.a": torch.randn(4, 3, dtype=torch.float32),
        "encoder.b": torch.randn(4, dtype=torch.float32),
        "predictor.c": torch.randn(2, 2, dtype=torch.float32),
    }


def _fields(round_index: int = 0) -> StructuralFields:
    return StructuralFields(
        schema_version=1,
        wmcp_version="wmcp-1.0.0",
        round_index=round_index,
        parent_hash=None,
        param_groups=("encoder", "predictor"),
    )


def _save(d: Path, w: dict[str, torch.Tensor], **kw: object) -> str:
    return save_checkpoint(
        d,
        w,
        wmcp_version="wmcp-1.0.0",
        round_index=0,
        config_hash="b" * 64,
        parent_hash=None,
        **kw,
    )


def test_content_hash_is_64_hex_and_deterministic() -> None:
    w, f = _w(), _fields()
    h1 = content_hash(w, f)
    h2 = content_hash(
        {k: w[k].clone() for k in w}, f
    )  # "second process": fresh tensors, same values
    assert h1 == h2
    assert len(h1) == 64 and all(c in "0123456789abcdef" for c in h1)


def test_hash_changes_on_value_rename_reshape_or_field() -> None:
    w, f = _w(), _fields()
    base = content_hash(w, f)
    flipped = {**w, "encoder.a": w["encoder.a"] + 1.0}
    assert content_hash(flipped, f) != base  # a changed weight
    renamed = {("encoder.z" if k == "encoder.a" else k): v for k, v in w.items()}
    assert content_hash(renamed, f) != base  # a renamed tensor
    reshaped = {**w, "encoder.a": w["encoder.a"].reshape(3, 4)}
    assert content_hash(reshaped, f) != base  # a reshaped tensor
    assert content_hash(w, _fields(round_index=1)) != base  # a changed structural field


def test_hash_excludes_config_and_created_at(tmp_path: Path) -> None:
    # two artifacts with the same weights/structural fields but different config_hash hash equal
    h1 = save_checkpoint(
        tmp_path / "a",
        _w(),
        wmcp_version="wmcp-1.0.0",
        round_index=0,
        config_hash="1" * 64,
        parent_hash=None,
    )
    h2 = save_checkpoint(
        tmp_path / "b",
        _w(),
        wmcp_version="wmcp-1.0.0",
        round_index=0,
        config_hash="2" * 64,
        parent_hash=None,
    )
    assert h1 == h2  # config_hash and created_at are excluded from the hash input


def test_checkpoint_hash_roundtrip_and_tamper(tmp_path: Path) -> None:
    committed = _save(tmp_path / "ckpt", _w())
    assert verify(tmp_path / "ckpt").content_hash == committed
    # mismatch vs the committed RoundClose hash
    with pytest.raises(CheckpointIntegrityError):
        verify(tmp_path / "ckpt", expected_hash="f" * 64)
    # flip a weight byte -> integrity failure, no tensors returned
    save_file(
        {
            "encoder.a": torch.ones(4, 3),
            "encoder.b": torch.zeros(4),
            "predictor.c": torch.zeros(2, 2),
        },
        str(tmp_path / "ckpt" / "weights.safetensors"),
    )
    with pytest.raises(CheckpointIntegrityError):
        load_checkpoint(tmp_path / "ckpt")
    # edit the header content_hash -> integrity failure
    hp = tmp_path / "ckpt" / "header.json"
    raw = json.loads(hp.read_text())
    raw["content_hash"] = "a" * 64
    hp.write_text(json.dumps(raw))
    with pytest.raises(CheckpointIntegrityError):
        load_checkpoint(tmp_path / "ckpt")


def test_no_pickle_payload_rejected(tmp_path: Path) -> None:
    d = tmp_path / "ckpt"
    _save(d, _w())
    # replace the weights file with a pickle payload; the loader must never fall back to torch.load
    (d / "weights.safetensors").write_bytes(pickle.dumps({"encoder.a": [1, 2, 3]}))
    with pytest.raises(CheckpointIntegrityError) as exc:
        load_checkpoint(d)
    assert "safetensors" in exc.value.remediation


def test_action_head_rejected_before_write(tmp_path: Path) -> None:
    bad = {**_w(), "action_head.so101.weight": torch.zeros(2, 2)}
    with pytest.raises(ResidencyViolation) as exc:
        _save(tmp_path / "ckpt", bad)
    assert exc.value.code.value == "residency_violation"
    assert not (
        tmp_path / "ckpt"
    ).exists()  # nothing written (fail-closed before any bytes)


def test_domain_separation_from_plain_sha256() -> None:
    # the artifact hash is domain-separated, so it never equals a plain SHA-256 of the same bytes
    # (guaranteeing it cannot collide with a RFC-0014 Merkle-leaf hash of the same content)
    w, f = {"encoder.a": torch.zeros(2, 2, dtype=torch.float32)}, _fields()
    raw = w["encoder.a"].numpy().astype("<f4").tobytes()
    assert content_hash(w, f) != hashlib.sha256(raw).hexdigest()
