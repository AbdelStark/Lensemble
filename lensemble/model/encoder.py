"""lensemble.model.encoder — the video-ViT encoder f_theta (docs/rfcs/RFC-0008 2).

A video Vision Transformer warm-started from released V-JEPA 2 weights and co-trained (Fork B). Its
``forward`` emits a WMCP :class:`~lensemble.contracts.LatentState` of shape ``(B, N, d)``. Warm-start
loading is the only path that establishes ``INV-WARMSTART-T0``: every participant loads weights whose
content hash equals the same ``expected_hash``, so round-0 encoder weights are byte-identical across the
federation and the latent gauge is closed at ``t=0``. ``snapshot_reference`` freezes the round-0 encoder
as ``f_ref``, the source of the anchor targets (``INV-PROBE-PIN``); ``f_ref`` is never trained or
broadcast.

The architecture here is a compact, configurable video ViT (the V-JEPA 2 shape family) so the same
construction loads the released weights and runs on a tiny synthetic CPU config in tests — no large model
download is required by the unit suite. The forward returns the ``LatentState`` directly; conformance is
checked at the contract boundary by the caller (``check_latent_state``), not inside the forward.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save
from torch import Tensor, nn

from lensemble.contracts import WMCP_VERSION, LatentState
from lensemble.errors import (
    ArtifactError,
    CheckpointIntegrityError,
    ConfigError,
    LensembleErrorCode,
)
from lensemble.model.numerics import (
    apply_numerics,
    autocast_forward,
    module_input_tensor,
    resolve_device,
)

if (
    TYPE_CHECKING
):  # avoids importing artifacts at module load (keeps the dep direction inward)
    from lensemble.artifacts.schema import CheckpointHeader, ModelArchDescriptor


def _canonical_bytes(state_dict: dict[str, Tensor]) -> bytes:
    """Deterministic serialization of a weight ``state_dict`` (safetensors; sorted keys)."""
    contiguous = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    return st_save(contiguous)


def _module_content_hash(module: nn.Module) -> str:
    """SHA-256 over the canonical serialization of a module's weights (RFC-0010 ``INV-CHECKPOINT-HASH``)."""
    return hashlib.sha256(_canonical_bytes(module.state_dict())).hexdigest()


class Encoder(nn.Module):
    """Video ViT encoder f_theta. Warm-started from V-JEPA 2; co-trained (Fork B).

    ``forward(clip)`` maps ``clip`` of shape ``(B, T, C, Hpx, Wpx)`` to a ``LatentState`` carrying
    ``(B, N, d)`` tokens conforming to ``wmcp_version`` (``INV-WMCP``).
    """

    wmcp_version: str
    d: int
    num_tokens: int
    compute_dtype: torch.dtype = (
        torch.float32
    )  # bf16 on CUDA (set by build_encoder); fp32 default

    def __init__(
        self,
        *,
        d: int,
        num_tokens: int,
        in_channels: int,
        num_frames: int,
        image_size: int,
        patch_size: int,
        tubelet: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        wmcp_version: str = WMCP_VERSION,
    ) -> None:
        super().__init__()
        self.d = d
        self.num_tokens = num_tokens
        self.wmcp_version = wmcp_version
        self.patch_embed = nn.Conv3d(
            in_channels,
            d,
            kernel_size=(tubelet, patch_size, patch_size),
            stride=(tubelet, patch_size, patch_size),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=num_heads,
            dim_feedforward=int(d * mlp_ratio),
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, clip: Tensor) -> LatentState:
        if clip.ndim != 5:
            raise ValueError(
                f"clip must be (B, T, C, Hpx, Wpx) rank-5, got rank {clip.ndim} shape {tuple(clip.shape)}"
            )
        clip = module_input_tensor(self, clip)
        # bf16 forward on CUDA / fp32 on the CPU fallback (RFC-0008 7); a no-op for fp32 so the CPU path
        # is unchanged. Master weights stay fp32; loss/statistic accumulation downstream is fp32.
        with autocast_forward(
            clip.device, getattr(self, "compute_dtype", torch.float32)
        ):
            x = clip.movedim(2, 1)  # (B, T, C, H, W) -> (B, C, T, H, W) for Conv3d
            x = self.patch_embed(x)  # (B, d, T', H', W')
            b, d = x.shape[0], x.shape[1]
            x = x.reshape(b, d, -1).transpose(1, 2)  # (B, L, d)
            if x.shape[1] != self.num_tokens:
                raise ConfigError(
                    f"patching produced {x.shape[1]} tokens but num_tokens={self.num_tokens}; "
                    "reconcile num_frames/tubelet and image_size/patch_size with num_tokens",
                    code=LensembleErrorCode.CONFIG_INVALID,
                    remediation="set num_tokens = (num_frames//tubelet) * (image_size//patch_size)**2",
                )
            x = x + self.pos_embed
            x = self.norm(self.blocks(x))  # (B, N, d)
        return LatentState(
            tokens=x,
            num_tokens=self.num_tokens,
            dim=self.d,
            wmcp_version=self.wmcp_version,
        )

    @classmethod
    def from_header(cls, header: "CheckpointHeader") -> "Encoder":
        """Reconstruct ``f_theta`` from a self-describing checkpoint header (#171; unblocks #62).

        Reads ``header.model_arch`` (the :class:`~lensemble.artifacts.schema.ModelArchDescriptor` written
        by ``save_checkpoint``) and builds an :class:`Encoder` with the SAME dims ``build_encoder`` would
        produce for the config that minted it — so loading the committed weights into it reproduces a
        byte-identical forward (``recompute_alignment`` can compute ``f_theta(P)``, #62).

        Raises :class:`~lensemble.errors.ArtifactError` when ``header.model_arch is None`` (a legacy /
        non-self-describing checkpoint, schema v1): the architecture — ``num_heads`` in particular — is
        unrecoverable from weight shapes, so reconstruction fails closed with a clear remediation.
        """
        if header.model_arch is None:
            raise ArtifactError(
                "checkpoint is not self-describing: header.model_arch is None, so the encoder "
                "architecture (num_heads is unrecoverable from weight shapes) cannot be reconstructed",
                code=LensembleErrorCode.ARTIFACT_INVALID,
                remediation="re-commit the checkpoint with a ModelArchDescriptor (#171) — e.g. "
                "save_checkpoint(..., model_arch=model_arch_from_config(cfg))",
            )
        return build_encoder_from_arch(header.model_arch)


class ReferenceEncoder(nn.Module):
    """A frozen, eval-mode round-0 snapshot of the encoder: ``f_ref`` (``INV-PROBE-PIN``).

    Never trained, never broadcast. Used to produce the anchor targets ``t_i = f_ref(p_i)`` and
    ``E_ref = f_ref(P)``; the target computation itself lives in ``lensemble.gauge.anchor``.
    """

    def __init__(self, encoder: Encoder) -> None:
        super().__init__()
        self._encoder = copy.deepcopy(encoder)
        self._encoder.eval()
        for p in self._encoder.parameters():
            p.requires_grad_(False)
        self._content_hash = _module_content_hash(self._encoder)

    @property
    def content_hash(self) -> str:
        """SHA-256 of the frozen weights; equals the pinned warm-start hash at round 0."""
        return self._content_hash

    @torch.no_grad()
    def forward(self, clip: Tensor) -> LatentState:
        return self._encoder(clip)


def _model_cfg(cfg: Any) -> Any:
    model = getattr(cfg, "model", None)
    if model is None:
        raise ConfigError(
            "config has no `model` sub-config",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="provide cfg.model with d, num_frames, image_size, patch_size, tubelet, depth, num_heads",
        )
    return model


def build_encoder(cfg: Any) -> Encoder:
    """Construct an :class:`Encoder` per the model config.

    Pre: ``cfg.model.latent_dim > 0`` and the patching is consistent with ``cfg.model.num_tokens``.
    Post: an ``Encoder`` whose forward emits a conformant ``LatentState`` (``INV-WMCP``).
    Raises: :class:`~lensemble.errors.ConfigError` on inconsistent dims/patching.
    """
    m = _model_cfg(cfg)
    # The ViT hidden dim is ModelConfig.latent_dim (#166 bridge); `d` is the legacy alias some
    # SimpleNamespace test/CLI configs still use, so fall back to it when latent_dim is absent.
    d = int(m.latent_dim if hasattr(m, "latent_dim") else m.d)
    in_channels = int(getattr(m, "in_channels", 3))
    num_frames = int(m.num_frames)
    image_size = int(m.image_size)
    patch_size = int(m.patch_size)
    tubelet = int(m.tubelet)
    depth = int(m.depth)
    num_heads = int(m.num_heads)
    mlp_ratio = float(getattr(m, "mlp_ratio", 4.0))
    wmcp_version = str(getattr(m, "wmcp_version", WMCP_VERSION))

    def _bad(msg: str, remediation: str) -> ConfigError:
        return ConfigError(
            msg, code=LensembleErrorCode.CONFIG_INVALID, remediation=remediation
        )

    if d <= 0:
        raise _bad(
            f"model.latent_dim must be > 0, got {d}", "set a positive latent_dim"
        )
    if d % num_heads != 0:
        raise _bad(
            f"model.latent_dim ({d}) must be divisible by num_heads ({num_heads})",
            "choose num_heads dividing d",
        )
    if num_frames % tubelet != 0 or image_size % patch_size != 0:
        raise _bad(
            "num_frames must be divisible by tubelet and image_size by patch_size",
            "align num_frames/tubelet and image_size/patch_size",
        )
    derived_tokens = (num_frames // tubelet) * (image_size // patch_size) ** 2
    declared = getattr(m, "num_tokens", None)
    if declared is not None and int(declared) != derived_tokens:
        raise _bad(
            f"model.num_tokens ({declared}) inconsistent with patching (derived {derived_tokens})",
            f"set num_tokens = {derived_tokens} or adjust patch/tubelet/frame/image sizes",
        )
    encoder = Encoder(
        d=d,
        num_tokens=derived_tokens,
        in_channels=in_channels,
        num_frames=num_frames,
        image_size=image_size,
        patch_size=patch_size,
        tubelet=tubelet,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        wmcp_version=wmcp_version,
    )
    apply_numerics(
        encoder, resolve_device()
    )  # CUDA primary (bf16 forward) / CPU fallback (fp32)
    if bool(getattr(m, "encoder_frozen", False)):
        # Fork A (RFC-0002): freeze f_theta at warm-start and federate g_phi only. The inner loop
        # (`_inner_loop`) optimizes only `requires_grad` params, so a frozen encoder contributes a zero
        # encoder-delta while the predictor still trains. The weight values are untouched, so the
        # warm-start content hash (INV-WARMSTART-T0) is unchanged.
        for param in encoder.parameters():
            param.requires_grad_(False)
    return encoder


def build_encoder_from_arch(arch: "ModelArchDescriptor") -> Encoder:
    """Construct an :class:`Encoder` from a self-describing :class:`ModelArchDescriptor` (#171).

    The descriptor records the exact ViT shape ``build_encoder`` derived from the minting config (notably
    ``num_heads``, unrecoverable from weight shapes), and ``num_tokens`` is already the derived token count.
    The resulting encoder has the SAME dims as ``build_encoder(cfg)`` for that cfg, so loading the committed
    weights reproduces the saved forward (``INV-CHECKPOINT-HASH`` / #62). ``apply_numerics`` selects the
    same CUDA(bf16)/CPU(fp32) compute path.

    Raises :class:`~lensemble.errors.ConfigError` on an inconsistent descriptor (``num_heads`` not dividing
    ``d``, or patching that does not derive ``num_tokens``) — the same guards ``build_encoder`` enforces.
    """

    def _bad(msg: str, remediation: str) -> ConfigError:
        return ConfigError(
            msg, code=LensembleErrorCode.CONFIG_INVALID, remediation=remediation
        )

    if arch.d % arch.num_heads != 0:
        raise _bad(
            f"model_arch.d ({arch.d}) must be divisible by num_heads ({arch.num_heads})",
            "the descriptor is inconsistent; num_heads must divide d",
        )
    if arch.num_frames % arch.tubelet != 0 or arch.image_size % arch.patch_size != 0:
        raise _bad(
            "model_arch: num_frames must be divisible by tubelet and image_size by patch_size",
            "the descriptor's patching is inconsistent",
        )
    derived_tokens = (arch.num_frames // arch.tubelet) * (
        arch.image_size // arch.patch_size
    ) ** 2
    if arch.num_tokens != derived_tokens:
        raise _bad(
            f"model_arch.num_tokens ({arch.num_tokens}) inconsistent with patching (derived "
            f"{derived_tokens})",
            f"the descriptor is inconsistent; expected num_tokens = {derived_tokens}",
        )
    encoder = Encoder(
        d=arch.d,
        num_tokens=arch.num_tokens,
        in_channels=arch.in_channels,
        num_frames=arch.num_frames,
        image_size=arch.image_size,
        patch_size=arch.patch_size,
        tubelet=arch.tubelet,
        depth=arch.depth,
        num_heads=arch.num_heads,
        mlp_ratio=arch.mlp_ratio,
        wmcp_version=arch.wmcp_version,
    )
    apply_numerics(encoder, resolve_device())
    return encoder


def load_warmstart(encoder: Encoder, checkpoint: Path, *, expected_hash: str) -> None:
    """Load pinned V-JEPA 2 warm-start weights into ``encoder`` (establishes ``INV-WARMSTART-T0``).

    Pre: the checkpoint file content hash (SHA-256) equals ``expected_hash`` (the pinned warm-start
    hash, RFC-0010 ``INV-CHECKPOINT-HASH``). A mismatch raises
    :class:`~lensemble.errors.CheckpointIntegrityError` and the weights are not loaded.
    Post: ``encoder`` weights are byte-identical across all participants that loaded the same hash.
    """
    raw = Path(checkpoint).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_hash:
        raise CheckpointIntegrityError(
            f"warm-start checkpoint hash mismatch: expected {expected_hash}, got {actual}",
            code=LensembleErrorCode.CHECKPOINT_INTEGRITY,
            remediation="re-download the pinned V-JEPA 2 warm-start; do not train from an unverified checkpoint",
        )
    state = st_load(raw)
    encoder.load_state_dict(state, strict=True)


def snapshot_reference(encoder: Encoder) -> ReferenceEncoder:
    """Freeze the round-0 encoder as ``f_ref`` (``INV-PROBE-PIN`` / ``INV-WARMSTART-T0``).

    Post: a frozen, eval-mode encoder whose ``content_hash`` equals the pinned warm-start hash at round 0.
    ``f_ref`` is never trained and never broadcast; anchor targets derive only from it.
    """
    return ReferenceEncoder(encoder)


def encoder_content_hash(encoder: Encoder) -> str:
    """The SHA-256 content hash of an encoder's weights (canonical safetensors serialization)."""
    return _module_content_hash(encoder)
