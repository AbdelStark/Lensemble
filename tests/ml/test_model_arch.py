"""ModelArchDescriptor + self-describing checkpoints (RFC-0010 §2; #171, unblocks #62).

Pins the three contracts the issue requires:
  - the descriptor round-trips through ``save_checkpoint`` / ``load_checkpoint`` (the header carries it);
  - the descriptor is HEADER metadata only — ``content_hash`` is byte-identical with or without it
    (``INV-CHECKPOINT-HASH`` stays metadata-independent);
  - ``Encoder.from_header`` reconstructs ``f_theta`` from the descriptor (and fails closed on a legacy,
    non-self-describing header so #62 can never silently mis-reconstruct ``num_heads``).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from lensemble.artifacts import (
    SCHEMA_VERSION,
    CheckpointHeader,
    ModelArchDescriptor,
    load_checkpoint,
    model_arch_from_config,
    save_checkpoint,
)
from lensemble.config import load_config
from lensemble.errors import ArtifactError
from lensemble.model.encoder import Encoder, build_encoder, build_encoder_from_arch

# A tiny, internally-consistent ViT shape: num_tokens = (num_frames//tubelet) * (image_size//patch_size)**2
# = (2//2) * (4//2)**2 = 1 * 4 = 4. d=8 with num_heads=2 (2 | 8).
_D, _DEPTH, _HEADS, _TUBELET, _FRAMES, _PATCH, _IMG = 8, 1, 2, 2, 2, 2, 4
_NUM_TOKENS = (_FRAMES // _TUBELET) * (_IMG // _PATCH) ** 2  # == 4


def _tiny_model_cfg() -> SimpleNamespace:
    """A minimal model config build_encoder/model_arch_from_config both consume."""
    return SimpleNamespace(
        latent_dim=_D,
        depth=_DEPTH,
        num_heads=_HEADS,
        num_tokens=_NUM_TOKENS,
        in_channels=3,
        num_frames=_FRAMES,
        image_size=_IMG,
        patch_size=_PATCH,
        tubelet=_TUBELET,
        mlp_ratio=2.0,
        wmcp_version="wmcp-1.0.0",
    )


def _tiny_cfg() -> SimpleNamespace:
    return SimpleNamespace(model=_tiny_model_cfg())


def _descriptor() -> ModelArchDescriptor:
    return model_arch_from_config(_tiny_cfg())


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


# --- the descriptor round-trips through the header ---


def test_descriptor_round_trips_through_header(tmp_path: Path) -> None:
    md = _descriptor()
    h = _save(tmp_path / "ckpt", _weights(), model_arch=md)
    _loaded, header = load_checkpoint(tmp_path / "ckpt")
    assert header.content_hash == h
    assert header.model_arch == md  # frozen pydantic equality, every field preserved
    assert header.schema_version == SCHEMA_VERSION == 2


def test_legacy_header_loads_with_model_arch_none(tmp_path: Path) -> None:
    # No model_arch passed: a non-self-describing checkpoint loads with model_arch=None (additive field).
    _save(tmp_path / "ckpt", _weights())  # model_arch defaults to None
    _loaded, header = load_checkpoint(tmp_path / "ckpt")
    assert header.model_arch is None


# --- the metadata-independence INVARIANT (RFC-0010 §4; INV-CHECKPOINT-HASH) ---


def test_content_hash_is_unchanged_by_the_descriptor(tmp_path: Path) -> None:
    # Same weights / StructuralFields, once WITHOUT and once WITH the descriptor -> identical content_hash.
    # The descriptor is header metadata (like created_at/config_hash), never in StructuralFields.
    h_none = _save(tmp_path / "none", _weights(), model_arch=None)
    h_arch = _save(tmp_path / "arch", _weights(), model_arch=_descriptor())
    assert h_none == h_arch  # byte-identical: the descriptor never enters content_hash

    # And the on-disk header content_hash matches too (verify_hash recomputes from weights+StructuralFields).
    _wn, hn = load_checkpoint(tmp_path / "none")
    _wa, ha = load_checkpoint(tmp_path / "arch")
    assert hn.content_hash == ha.content_hash == h_none
    assert hn.model_arch is None and ha.model_arch is not None


# --- Encoder.from_header reconstructs f_theta ---


def test_from_header_reconstructs_encoder_and_runs_forward(tmp_path: Path) -> None:
    md = _descriptor()
    _save(tmp_path / "ckpt", _weights(), model_arch=md)
    _loaded, header = load_checkpoint(tmp_path / "ckpt")

    enc = Encoder.from_header(header)
    assert isinstance(enc, Encoder)
    # Same dims as a fresh build_encoder over the minting cfg.
    reference = build_encoder(_tiny_cfg())
    assert (
        (enc.d, enc.num_tokens)
        == (reference.d, reference.num_tokens)
        == (md.d, md.num_tokens)
    )
    assert enc.wmcp_version == md.wmcp_version

    # Runs a forward on a tiny clip (B, T, C, Hpx, Wpx) -> a conformant LatentState (B, N, d).
    clip = torch.randn(1, _FRAMES, 3, _IMG, _IMG)
    out = enc(clip)
    assert out.tokens.shape == (1, _NUM_TOKENS, _D)
    assert out.num_tokens == _NUM_TOKENS and out.dim == _D


def test_build_encoder_from_arch_matches_build_encoder() -> None:
    md = _descriptor()
    enc_arch = build_encoder_from_arch(md)
    enc_cfg = build_encoder(_tiny_cfg())
    assert (enc_arch.d, enc_arch.num_tokens, enc_arch.wmcp_version) == (
        enc_cfg.d,
        enc_cfg.num_tokens,
        enc_cfg.wmcp_version,
    )
    # Same parameter set + shapes (same architecture), independent of init values.
    a = {k: tuple(v.shape) for k, v in enc_arch.state_dict().items()}
    b = {k: tuple(v.shape) for k, v in enc_cfg.state_dict().items()}
    assert a == b


def test_from_header_on_legacy_header_fails_closed(tmp_path: Path) -> None:
    _save(tmp_path / "ckpt", _weights())  # model_arch=None
    _loaded, header = load_checkpoint(tmp_path / "ckpt")
    with pytest.raises(ArtifactError) as exc:
        Encoder.from_header(header)
    # the remediation points at #171 / re-committing with a descriptor
    assert (
        "171" in exc.value.remediation or "ModelArchDescriptor" in exc.value.remediation
    )


# --- model_arch_from_config matches the default ModelConfig dims ---


def test_model_arch_from_default_config_matches_modelconfig() -> None:
    cfg = load_config()
    md = model_arch_from_config(cfg)
    m = cfg.model
    assert md.d == m.latent_dim
    assert md.depth == m.depth
    assert md.num_heads == m.num_heads
    assert md.num_tokens == m.num_tokens
    assert md.in_channels == m.in_channels
    assert md.num_frames == m.num_frames
    assert md.image_size == m.image_size
    assert md.patch_size == m.patch_size
    assert md.tubelet == m.tubelet
    assert md.mlp_ratio == m.mlp_ratio
    assert md.wmcp_version == m.wmcp_version


# --- descriptor validation (frozen, positive-int, extra=forbid) ---


def test_descriptor_rejects_non_positive_and_extra() -> None:
    from pydantic import ValidationError

    good = _descriptor().model_dump()
    with pytest.raises(ValidationError):
        ModelArchDescriptor(**{**good, "num_heads": 0})
    with pytest.raises(ValidationError):
        ModelArchDescriptor(**{**good, "mlp_ratio": 0.0})
    with pytest.raises(ValidationError):
        ModelArchDescriptor(**{**good, "surprise": 1})


def test_header_with_model_arch_json_roundtrips(tmp_path: Path) -> None:
    md = _descriptor()
    _save(tmp_path / "ckpt", _weights(), model_arch=md)
    header_path = tmp_path / "ckpt" / "header.json"
    restored = CheckpointHeader.model_validate_json(header_path.read_text())
    assert restored.model_arch == md


# --- defensive guards: inconsistent descriptor / no-model config (mirror build_encoder) ---


def test_model_arch_from_config_without_model_fails_closed() -> None:
    with pytest.raises(ArtifactError):
        model_arch_from_config(SimpleNamespace())  # no `model` sub-config


def _arch(**over: object) -> ModelArchDescriptor:
    base = _descriptor().model_dump()
    base.update(over)
    return ModelArchDescriptor(**base)  # type: ignore[arg-type]


def test_build_encoder_from_arch_rejects_indivisible_heads() -> None:
    from lensemble.errors import ConfigError

    # num_heads=3 does not divide d=8 (the descriptor validates positives, not cross-field divisibility).
    with pytest.raises(ConfigError):
        build_encoder_from_arch(_arch(num_heads=3))


def test_build_encoder_from_arch_rejects_inconsistent_patching() -> None:
    from lensemble.errors import ConfigError

    # tubelet=3 does not divide num_frames=2: inconsistent patching.
    with pytest.raises(ConfigError):
        build_encoder_from_arch(_arch(tubelet=3, num_frames=2))


def test_build_encoder_from_arch_rejects_wrong_num_tokens() -> None:
    from lensemble.errors import ConfigError

    # num_tokens=99 does not equal the derived token count for this shape (==4).
    with pytest.raises(ConfigError):
        build_encoder_from_arch(_arch(num_tokens=99))
