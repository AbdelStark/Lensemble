"""Encoder warm-start, f_ref snapshot, and WMCP conformance (RFC-0008 2). Issue #10. CPU fixtures."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lensemble.contracts import LatentState, check_latent_state
from lensemble.errors import CheckpointIntegrityError, ContractViolation
from lensemble.model.encoder import (
    _canonical_bytes,
    build_encoder,
    encoder_content_hash,
    load_warmstart,
    snapshot_reference,
)


def _cfg() -> SimpleNamespace:
    # derived num_tokens = (num_frames//tubelet) * (image_size//patch_size)**2 = (4//2)*(8//4)**2 = 8
    return SimpleNamespace(
        model=SimpleNamespace(
            d=16,
            in_channels=3,
            num_frames=4,
            image_size=8,
            patch_size=4,
            tubelet=2,
            depth=2,
            num_heads=4,
        )
    )


def _clip(batch: int = 2) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, 4, 3, 8, 8)


def _write_checkpoint(encoder: object, path: Path) -> str:
    raw = _canonical_bytes(encoder.state_dict())  # type: ignore[attr-defined]
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_forward_emits_conformant_latentstate() -> None:
    enc = build_encoder(_cfg())
    assert enc.d == 16 and enc.num_tokens == 8
    out = enc(_clip())
    assert isinstance(out, LatentState)
    assert tuple(out.tokens.shape) == (2, 8, 16)
    check_latent_state(out)  # conformant at the contract boundary (INV-WMCP)


def test_load_warmstart_hash_mismatch_raises(tmp_path: Path) -> None:
    src = build_encoder(_cfg())
    ckpt = tmp_path / "warmstart.safetensors"
    _write_checkpoint(src, ckpt)
    dst = build_encoder(_cfg())
    with pytest.raises(CheckpointIntegrityError) as exc:
        load_warmstart(dst, ckpt, expected_hash="0" * 64)
    assert exc.value.code.value == "checkpoint_integrity"
    assert exc.value.remediation


def test_load_warmstart_success_is_byte_identical(tmp_path: Path) -> None:
    src = build_encoder(_cfg())
    ckpt = tmp_path / "warmstart.safetensors"
    h = _write_checkpoint(src, ckpt)
    dst = build_encoder(_cfg())
    assert encoder_content_hash(dst) != h  # different random init
    load_warmstart(dst, ckpt, expected_hash=h)
    # INV-WARMSTART-T0: weights byte-identical after loading the pinned hash
    assert encoder_content_hash(dst) == h
    for a, b in zip(src.state_dict().values(), dst.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_snapshot_reference_hash_equals_warmstart(tmp_path: Path) -> None:
    src = build_encoder(_cfg())
    ckpt = tmp_path / "warmstart.safetensors"
    h = _write_checkpoint(src, ckpt)
    enc = build_encoder(_cfg())
    load_warmstart(enc, ckpt, expected_hash=h)
    f_ref = snapshot_reference(enc)
    # f_ref content hash equals the pinned warm-start hash at round 0 (INV-WARMSTART-T0)
    assert f_ref.content_hash == h
    # frozen: no parameter requires grad; eval mode
    assert all(not p.requires_grad for p in f_ref.parameters())
    out = f_ref(_clip())
    check_latent_state(out)
    assert not out.tokens.requires_grad


def test_non_conformant_output_raises_contract_violation() -> None:
    enc = build_encoder(_cfg())
    out = enc(_clip())
    # tamper the declared metadata: the boundary check (INV-WMCP) rejects it
    bad = dataclasses.replace(out, num_tokens=999)
    with pytest.raises(ContractViolation):
        check_latent_state(bad)
