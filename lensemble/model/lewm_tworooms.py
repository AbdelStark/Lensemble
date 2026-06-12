"""lensemble.model.lewm_tworooms — in-tree reconstruction of the LeWorldModel TwoRooms checkpoint.

The Tapestry-like LeWM pivot ([#316](https://github.com/AbdelStark/Lensemble/issues/314); contract in
docs/roadmap/TAPESTRY_LEWM.md) starts from the real ``quentinll/lewm-tworooms`` checkpoint, whose
``config.json`` instantiates ``stable_worldmodel.wm.lewm.LeWM`` with a HuggingFace ``ViTModel`` encoder
(``stable_pretraining.backbone.utils.vit_hf``). Those upstream packages are **not vendored**
(third_party manifests, RFC-0016 §2), so this module reconstructs the exact module tree in plain torch,
**state-dict-key compatible** with the released ``weights.pt``: the HF-ViT key schema for the encoder
(``embeddings.*``, ``encoder.layer.N.*``, ``layernorm.*``) and the upstream LeWM key schema for the
predictor/action-encoder/projection heads. ``load_lewm_state_dict`` is strict by construction: an
unknown or missing tensor name is a ``CheckpointIntegrityError``, never a silent partial load.

This is an inference/reference surface for checkpoint parity (gate G1) and browser export (gate G2) —
it is not the Lensemble training encoder (``lensemble.model.encoder``) and does not claim from-scratch
LeWM training. Forward semantics are kept numerically faithful to upstream (einops rearrangements are
replaced by reshape/permute; ``scaled_dot_product_attention(is_causal=True)`` is replaced by an
explicit additive causal mask, which is mathematically identical and ONNX-exportable).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lensemble.errors import (
    CheckpointIntegrityError,
    ConfigError,
    ContractViolation,
    LensembleErrorCode,
)

__all__ = [
    "LeWMTwoRoomsConfig",
    "LeWMTwoRooms",
    "HFViT",
    "LeWMPredictor",
    "ActionEmbedder",
    "ProjectionMLP",
    "build_lewm_tworooms",
    "load_lewm_state_dict",
]

# Upstream `_target_` strings pinned by the released config.json. Reconstruction refuses configs that
# name any other implementation: a different target means a different module tree, and a silent
# best-effort load would fake parity (gate G1 fails closed).
_EXPECTED_TARGETS = {
    "_target_": "stable_worldmodel.wm.lewm.LeWM",
    "encoder": "stable_pretraining.backbone.utils.vit_hf",
    "predictor": "stable_worldmodel.wm.lewm.module.Predictor",
    "action_encoder": "stable_worldmodel.wm.lewm.module.Embedder",
    "projector": "stable_worldmodel.wm.lewm.module.MLP",
    "pred_proj": "stable_worldmodel.wm.lewm.module.MLP",
}

# vit_hf size table (stable_pretraining.backbone.utils): hidden/layers/heads per named size.
_VIT_SIZES = {
    "tiny": (192, 12, 3),
    "small": (384, 12, 6),
    "base": (768, 12, 12),
}


@dataclass(frozen=True)
class LeWMTwoRoomsConfig:
    """Validated architecture fields extracted from the checkpoint ``config.json``."""

    vit_size: str = "tiny"
    patch_size: int = 14
    image_size: int = 224
    hidden_dim: int = 192
    vit_layers: int = 12
    vit_heads: int = 3
    pred_num_frames: int = 3
    pred_depth: int = 6
    pred_heads: int = 16
    pred_dim_head: int = 64
    pred_mlp_dim: int = 2048
    pred_dropout: float = 0.1
    pred_emb_dropout: float = 0.0
    action_input_dim: int = 10
    action_emb_dim: int = 192
    proj_hidden_dim: int = 2048
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def num_patches(self) -> int:
        side = self.image_size // self.patch_size
        return side * side

    @staticmethod
    def from_upstream(config: dict[str, Any]) -> "LeWMTwoRoomsConfig":
        """Validate and translate the upstream Hydra-style ``config.json`` dict.

        Fails closed (``ConfigError``) on any unexpected ``_target_``, BatchNorm substitution, masked
        encoder, or pretrained-encoder flag — every one of those would change the module tree or the
        numerics relative to the released TwoRooms artifact.
        """

        def _section(name: str) -> dict[str, Any]:
            sec = config.get(name)
            if not isinstance(sec, dict):
                raise ConfigError(
                    f"checkpoint config is missing section {name!r}",
                    code=LensembleErrorCode.CONFIG_INVALID,
                    remediation="point at an unmodified quentinll/lewm-tworooms config.json",
                )
            return sec

        for key, expected in _EXPECTED_TARGETS.items():
            actual = (
                config.get("_target_") if key == "_target_" else _section(key).get("_target_")
            )
            if actual != expected:
                raise ConfigError(
                    f"checkpoint config {key} _target_ is {actual!r}, expected {expected!r}",
                    code=LensembleErrorCode.CONFIG_INVALID,
                    remediation="this loader reconstructs the released LeWM TwoRooms module tree "
                    "only; use the pinned quentinll/lewm-tworooms config.json",
                )

        enc = _section("encoder")
        if enc.get("pretrained", False):
            raise ConfigError(
                "encoder.pretrained=true is not the released TwoRooms artifact",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="the released checkpoint trains the ViT from scratch; refuse "
                "substituted encoders",
            )
        if enc.get("use_mask_token", False):
            raise ConfigError(
                "encoder.use_mask_token=true does not match the released state dict",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="the released weights have no mask token tensor",
            )
        size = enc.get("size", "tiny")
        if size not in _VIT_SIZES:
            raise ConfigError(
                f"unsupported vit size {size!r}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation=f"supported sizes: {sorted(_VIT_SIZES)}",
            )
        hidden, layers, heads = _VIT_SIZES[size]

        pred = _section("predictor")
        act = _section("action_encoder")
        proj = _section("projector")
        pred_proj = _section("pred_proj")
        for head_name, head in (("projector", proj), ("pred_proj", pred_proj)):
            norm = head.get("norm_fn", {})
            if norm.get("_target_") != "torch.nn.BatchNorm1d":
                raise ConfigError(
                    f"{head_name}.norm_fn must be torch.nn.BatchNorm1d",
                    code=LensembleErrorCode.CONFIG_INVALID,
                    remediation="the released projection heads are BatchNorm MLPs; a LayerNorm "
                    "substitute changes both keys and numerics",
                )

        cfg = LeWMTwoRoomsConfig(
            vit_size=size,
            patch_size=int(enc.get("patch_size", 16)),
            image_size=int(enc.get("image_size", 224)),
            hidden_dim=hidden,
            vit_layers=layers,
            vit_heads=heads,
            pred_num_frames=int(pred["num_frames"]),
            pred_depth=int(pred["depth"]),
            pred_heads=int(pred["heads"]),
            pred_dim_head=int(pred.get("dim_head", 64)),
            pred_mlp_dim=int(pred["mlp_dim"]),
            pred_dropout=float(pred.get("dropout", 0.0)),
            pred_emb_dropout=float(pred.get("emb_dropout", 0.0)),
            action_input_dim=int(act["input_dim"]),
            action_emb_dim=int(act["emb_dim"]),
            proj_hidden_dim=int(proj["hidden_dim"]),
            raw=dict(config),
        )
        if cfg.hidden_dim != int(pred["input_dim"]):
            raise ConfigError(
                f"predictor input_dim {pred['input_dim']} != encoder hidden {cfg.hidden_dim}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="encoder latent width and predictor input width must agree",
            )
        if cfg.image_size % cfg.patch_size != 0:
            raise ConfigError(
                f"image_size {cfg.image_size} not divisible by patch_size {cfg.patch_size}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="the HF ViT patchifier requires an integer patch grid",
            )
        return cfg


# ---------------------------------------------------------------------------
# Encoder — HuggingFace ViTModel reconstruction (state-dict compatible)
# ---------------------------------------------------------------------------


class _Dense(nn.Module):
    """A single Linear registered as ``dense`` (the HF ViT nesting unit)."""

    def __init__(self, din: int, dout: int) -> None:
        super().__init__()
        self.dense = nn.Linear(din, dout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dense(x)


class _HFViTQKV(nn.Module):
    """HF ``ViTSelfAttention`` projections, registered as ``query``/``key``/``value``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)


class _HFViTSelfAttention(nn.Module):
    """``transformers.ViTSelfAttention`` + ``ViTSelfOutput`` under HF key names."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ConfigError(
                f"vit hidden {dim} not divisible by heads {heads}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="use a vit size from the released table",
            )
        self.num_heads = heads
        self.head_dim = dim // heads
        # HF nests these as attention.attention.{query,key,value} and attention.output.dense.
        self.attention = _HFViTQKV(dim)
        self.output = _Dense(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        h, hd = self.num_heads, self.head_dim

        def _split(t: Tensor) -> Tensor:
            return t.reshape(b, n, h, hd).permute(0, 2, 1, 3)

        q = _split(self.attention.query(x))
        k = _split(self.attention.key(x))
        v = _split(self.attention.value(x))
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(hd), dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(b, n, d)
        return self.output(out)


class _HFViTLayer(nn.Module):
    """``transformers.ViTLayer``: pre-norm attention and MLP with HF residual placement."""

    def __init__(self, dim: int, heads: int, eps: float = 1e-12) -> None:
        super().__init__()
        self.attention = _HFViTSelfAttention(dim, heads)
        self.intermediate = _Dense(dim, dim * 4)
        self.output = _Dense(dim * 4, dim)
        self.layernorm_before = nn.LayerNorm(dim, eps=eps)
        self.layernorm_after = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.layernorm_before(x))
        y = self.output(F.gelu(self.intermediate(self.layernorm_after(x))))
        return x + y


class HFViT(nn.Module):
    """``transformers.ViTModel`` (no pooler, no mask token) reconstructed in plain torch.

    Key schema, forward order, GELU variant, and LayerNorm epsilon (1e-12) follow HF exactly so the
    released ``encoder.*`` tensors load strictly and outputs match the upstream reference. With the
    released 224px/patch-14 geometry the position table matches the patch grid, so the HF
    ``interpolate_pos_encoding=True`` path that upstream enables is the identity.
    """

    class _PatchEmbeddings(nn.Module):
        def __init__(self, dim: int, patch_size: int) -> None:
            super().__init__()
            self.projection = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

    class _Embeddings(nn.Module):
        def __init__(self, dim: int, patch_size: int, num_patches: int) -> None:
            super().__init__()
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
            self.patch_embeddings = HFViT._PatchEmbeddings(dim, patch_size)

    class _LayerStack(nn.Module):
        # HF registers encoder.layer.N; this inner module named `encoder` reproduces the nesting.
        def __init__(self, dim: int, heads: int, layers: int) -> None:
            super().__init__()
            self.layer = nn.ModuleList(_HFViTLayer(dim, heads) for _ in range(layers))

    def __init__(self, image_size: int, patch_size: int, dim: int, layers: int, heads: int) -> None:
        super().__init__()
        side = image_size // patch_size
        num_patches = side * side
        self.patch_size = patch_size
        self.hidden_dim = dim
        self.embeddings = HFViT._Embeddings(dim, patch_size, num_patches)
        self.encoder = HFViT._LayerStack(dim, heads, layers)
        self.layernorm = nn.LayerNorm(dim, eps=1e-12)

    def forward(self, pixels: Tensor) -> Tensor:
        """``(B, 3, H, W)`` float pixels → all token states ``(B, 1 + num_patches, D)``."""
        b = pixels.shape[0]
        patches = self.embeddings.patch_embeddings.projection(pixels)
        tokens = patches.flatten(2).transpose(1, 2)  # (B, N, D)
        cls = self.embeddings.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, tokens], dim=1) + self.embeddings.position_embeddings
        for layer in self.encoder.layer:
            x = layer(x)
        return self.layernorm(x)


# ---------------------------------------------------------------------------
# Predictor / action encoder / projection heads — upstream LeWM reconstruction
# ---------------------------------------------------------------------------


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class _LeWMAttention(nn.Module):
    """Upstream ``module.Attention``: pre-LN, bias-free QKV, causal mask, projected output."""

    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float) -> None:
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.attn_dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        h, hd = self.heads, self.dim_head
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (part.reshape(b, t, h, hd).permute(0, 2, 1, 3) for part in qkv)
        scores = q @ k.transpose(-1, -2) / math.sqrt(hd)
        causal = torch.triu(
            torch.full((t, t), float("-inf"), dtype=scores.dtype, device=scores.device),
            diagonal=1,
        )
        attn = torch.softmax(scores + causal, dim=-1)
        if self.training and self.attn_dropout > 0:
            attn = F.dropout(attn, p=self.attn_dropout)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(b, t, h * hd)
        return self.to_out(out)


class _LeWMFeedForward(nn.Module):
    """Upstream ``module.FeedForward``: LN → Linear → GELU → Dropout → Linear → Dropout."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _LeWMConditionalBlock(nn.Module):
    """Upstream ``module.ConditionalBlock``: AdaLN-zero conditioning on the action embedding."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn = _LeWMAttention(dim, heads, dim_head, dropout)
        self.mlp = _LeWMFeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        modulation = nn.Linear(dim, 6 * dim, bias=True)
        nn.init.constant_(modulation.weight, 0)
        nn.init.constant_(modulation.bias, 0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), modulation)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class _LeWMTransformer(nn.Module):
    """Upstream ``module.Transformer`` with conditional blocks (identity projections at D=192)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        )
        self.layers = nn.ModuleList(
            _LeWMConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        x = self.input_proj(x)
        c = self.cond_proj(c)
        for block in self.layers:
            x = block(x, c)
        return self.output_proj(self.norm(x))


class LeWMPredictor(nn.Module):
    """Upstream ``module.Predictor``: learned positions + AdaLN-conditioned causal transformer."""

    def __init__(self, cfg: LeWMTwoRoomsConfig) -> None:
        super().__init__()
        self.num_frames = cfg.pred_num_frames
        self.pos_embedding = nn.Parameter(
            torch.randn(1, cfg.pred_num_frames, cfg.hidden_dim)
        )
        self.dropout = nn.Dropout(cfg.pred_emb_dropout)
        self.transformer = _LeWMTransformer(
            cfg.hidden_dim,
            cfg.hidden_dim,
            cfg.hidden_dim,
            cfg.pred_depth,
            cfg.pred_heads,
            cfg.pred_dim_head,
            cfg.pred_mlp_dim,
            cfg.pred_dropout,
        )

    def forward(self, emb: Tensor, act_emb: Tensor) -> Tensor:
        t = emb.shape[1]
        if t > self.num_frames:
            raise ContractViolation(
                f"predictor history {t} exceeds num_frames {self.num_frames}",
                code=LensembleErrorCode.WMCP_CONTRACT_VIOLATION,
                remediation="truncate the rollout history to the predictor window",
            )
        x = emb + self.pos_embedding[:, :t]
        x = self.dropout(x)
        return self.transformer(x, act_emb)


class ActionEmbedder(nn.Module):
    """Upstream ``module.Embedder``: 1x1 Conv1d smoothing + SiLU MLP over action blocks."""

    def __init__(self, input_dim: int, emb_dim: int, mlp_scale: int = 4) -> None:
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(input_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, actions: Tensor) -> Tensor:
        """``(B, T, A)`` action blocks → ``(B, T, emb_dim)``."""
        x = actions.float().permute(0, 2, 1)
        x = self.patch_embed(x).permute(0, 2, 1)
        return self.embed(x)


class ProjectionMLP(nn.Module):
    """Upstream ``module.MLP`` with BatchNorm1d: Linear → BN → GELU → Linear over ``(B*T, D)``."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# The assembled world model
# ---------------------------------------------------------------------------


class LeWMTwoRooms(nn.Module):
    """The released TwoRooms LeWM: encoder → projector latents; AdaLN predictor → pred_proj.

    Inference semantics mirror upstream ``LeWM`` (encode / predict / rollout / goal cost). BatchNorm
    projection heads make batch-size-dependent statistics in train mode, so every reference/parity/
    export path runs in ``eval()`` with the checkpoint's running statistics.
    """

    def __init__(self, cfg: LeWMTwoRoomsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = HFViT(
            cfg.image_size, cfg.patch_size, cfg.hidden_dim, cfg.vit_layers, cfg.vit_heads
        )
        self.predictor = LeWMPredictor(cfg)
        self.action_encoder = ActionEmbedder(cfg.action_input_dim, cfg.action_emb_dim)
        self.projector = ProjectionMLP(cfg.hidden_dim, cfg.proj_hidden_dim, cfg.hidden_dim)
        self.pred_proj = ProjectionMLP(cfg.hidden_dim, cfg.proj_hidden_dim, cfg.hidden_dim)

    def encode_frames(self, pixels: Tensor) -> Tensor:
        """``(B, T, 3, H, W)`` pixels → projected CLS latents ``(B, T, D)``."""
        b, t = pixels.shape[:2]
        flat = pixels.reshape(b * t, *pixels.shape[2:])
        cls = self.encoder(flat)[:, 0]
        emb = self.projector(cls)
        return emb.reshape(b, t, -1)

    def encode_actions(self, actions: Tensor) -> Tensor:
        """``(B, T, A)`` raw action blocks → ``(B, T, D)`` action embeddings."""
        return self.action_encoder(actions)

    def predict(self, emb: Tensor, act_emb: Tensor) -> Tensor:
        """Teacher-forced next-latent prediction: ``(B, T, D)`` × ``(B, T, D)`` → ``(B, T, D)``."""
        preds = self.predictor(emb, act_emb)
        b, t, d = preds.shape
        return self.pred_proj(preds.reshape(b * t, d)).reshape(b, t, d)

    @torch.no_grad()
    def rollout(self, emb_init: Tensor, act_emb: Tensor) -> Tensor:
        """Autoregressive latent rollout with the upstream truncated-history loop.

        ``emb_init``: ``(B, H, D)`` encoded history; ``act_emb``: ``(B, H + n_steps, D)`` embedded
        action blocks. Returns ``(B, H + n_steps + 1, D)``: history plus each predicted next latent
        (one step is predicted past the final action, matching upstream ``rollout``).
        """
        h = emb_init.shape[1]
        n_steps = act_emb.shape[1] - h
        window = self.predictor.num_frames
        frames = list(emb_init.unbind(dim=1))
        for step in range(n_steps + 1):
            lo = max(0, h + step - window)
            history = torch.stack(frames[lo:], dim=1)
            acts = act_emb[:, lo : h + step]
            frames.append(self.predict(history, acts)[:, -1])
        return torch.stack(frames, dim=1)

    @torch.no_grad()
    def goal_cost(self, predicted: Tensor, goal_emb: Tensor) -> Tensor:
        """Terminal latent goal distance per candidate: ``(S, T, D)`` × ``(D,)`` → ``(S,)``."""
        terminal = predicted[:, -1]
        return ((terminal - goal_emb.unsqueeze(0)) ** 2).sum(dim=-1)


def build_lewm_tworooms(config: dict[str, Any] | LeWMTwoRoomsConfig) -> LeWMTwoRooms:
    """Build the reconstruction from the upstream ``config.json`` dict (validated, fail-closed)."""
    cfg = (
        config
        if isinstance(config, LeWMTwoRoomsConfig)
        else LeWMTwoRoomsConfig.from_upstream(config)
    )
    model = LeWMTwoRooms(cfg)
    model.eval()
    return model


def load_lewm_state_dict(model: LeWMTwoRooms, state_dict: dict[str, Tensor]) -> LeWMTwoRooms:
    """Strictly load the released ``weights.pt`` state dict into the reconstruction.

    Unknown tensor names, missing tensor names, and shape mismatches each raise
    ``CheckpointIntegrityError`` with the offending names — a partial load would silently break
    parity, so there is no non-strict path.
    """
    expected = {name: tuple(t.shape) for name, t in model.state_dict().items()}
    provided = {name: tuple(t.shape) for name, t in state_dict.items()}
    unknown = sorted(set(provided) - set(expected))
    missing = sorted(set(expected) - set(provided))
    if unknown or missing:
        raise CheckpointIntegrityError(
            f"state dict mismatch: {len(unknown)} unknown, {len(missing)} missing "
            f"(unknown={unknown[:5]}, missing={missing[:5]})",
            code=LensembleErrorCode.CHECKPOINT_INTEGRITY,
            remediation="use the unmodified weights.pt from the pinned checkpoint revision",
        )
    mismatched = sorted(
        name for name in expected if expected[name] != provided[name]
    )
    if mismatched:
        raise CheckpointIntegrityError(
            f"tensor shape mismatch for {mismatched[:5]} (and {max(0, len(mismatched) - 5)} more)",
            code=LensembleErrorCode.CHECKPOINT_INTEGRITY,
            remediation="the config and weights come from different artifacts; re-download the "
            "pinned revision",
        )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
