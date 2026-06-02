"""lensemble.model.predictor — the action-conditioned latent predictor g_phi (RFC-0008 3).

``Predictor.forward(latent, action_embedding)`` maps a ``(B, N, d)`` latent and a ``(B, cond_dim)``
conditioning embedding to a predicted ``(B, N, d)`` latent (a WMCP :class:`~lensemble.contracts.LatentState`).
The conditioning embedding lives in the shared latent-conditioning space, so ``g_phi`` is one federated
model across embodiments; the per-embodiment head that produces it is local (``INV-ACTIONHEAD-LOCAL``,
RFC-0008 4).

Stop-gradient contract (RFC-0008 3). The prediction loss is
``E || g_phi(f_theta(x_t), a_t) - sg[f_theta(x_{t+1})] ||^2``. :meth:`Predictor.prediction_residual`
applies ``Tensor.detach()`` to the target ``f_theta(x_{t+1})`` *before* the residual, so gradients flow
into ``g_phi`` and into ``f_theta`` through the input branch ``f_theta(x_t)`` only. The detach is a
contract, not an optimization detail — a missing detach silently changes the objective. With SIGReg
preventing collapse, no EMA/teacher target is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from lensemble.contracts import WMCP_VERSION, LatentState
from lensemble.errors import ConfigError, LensembleErrorCode

if TYPE_CHECKING:
    from torch import Tensor


def _default_heads(width: int) -> int:
    """The largest of {8,4,2,1} that divides ``width`` — a sane head count when config omits one."""
    for h in (8, 4, 2):
        if width % h == 0:
            return h
    return 1


class Predictor(nn.Module):
    """Action-conditioned latent predictor ``g_phi`` (RFC-0008 3): a transformer over the ``N`` tokens.

    The conditioning embedding ``(B, cond_dim)`` is projected to the model width and added to every token
    (a global conditioning bias), then a pre-norm Transformer predicts the next-step latent. ``forward``
    returns a conformant ``LatentState`` of the same ``(B, N, d)`` shape; the prediction is direct (no
    EMA/teacher target).
    """

    wmcp_version: str
    d: int
    num_tokens: int
    cond_dim: int

    def __init__(
        self,
        *,
        d: int,
        num_tokens: int,
        cond_dim: int,
        depth: int,
        width: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        wmcp_version: str = WMCP_VERSION,
    ) -> None:
        super().__init__()
        self.d = d
        self.num_tokens = num_tokens
        self.cond_dim = cond_dim
        self.wmcp_version = wmcp_version
        self.in_proj = nn.Linear(d, width)
        self.cond_proj = nn.Linear(cond_dim, width)
        self.out_proj = nn.Linear(width, d)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=num_heads,
            dim_feedforward=int(width * mlp_ratio),
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(width)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, latent: LatentState, action_embedding: Tensor) -> LatentState:
        """Predict the next-step latent: ``(B, N, d)`` x ``(B, cond_dim)`` -> ``(B, N, d)`` LatentState."""
        x = latent.tokens
        if x.ndim != 3 or x.shape[1] != self.num_tokens or x.shape[2] != self.d:
            raise ConfigError(
                f"predictor expects a batched latent (B, {self.num_tokens}, {self.d}), "
                f"got shape {tuple(x.shape)}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="pass a batched LatentState whose num_tokens and dim match the predictor",
            )
        if action_embedding.ndim != 2 or action_embedding.shape[1] != self.cond_dim:
            raise ConfigError(
                f"action_embedding must be (B, {self.cond_dim}), got "
                f"{tuple(action_embedding.shape)}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="produce the conditioning embedding with last dim == predictor cond_dim",
            )
        cond = self.cond_proj(action_embedding.to(x.dtype))  # (B, width)
        h = (
            self.in_proj(x) + self.pos_embed + cond.unsqueeze(1)
        )  # broadcast cond over N
        h = self.norm(self.blocks(h))
        out = self.out_proj(h)  # (B, N, d)
        return LatentState(
            tokens=out,
            num_tokens=self.num_tokens,
            dim=self.d,
            wmcp_version=self.wmcp_version,
        )

    def rollout(
        self, latent: LatentState, action_embeddings: Tensor
    ) -> list[LatentState]:
        """Autoregressive multi-step rollout: feed each prediction back as the next input.

        ``action_embeddings`` is ``(B, H, cond_dim)``; returns ``H`` predicted ``LatentState``s, each of
        shape ``(B, N, d)`` (shape + WMCP conformance preserved over the horizon).
        """
        if action_embeddings.ndim != 3 or action_embeddings.shape[2] != self.cond_dim:
            raise ConfigError(
                f"rollout action_embeddings must be (B, H, {self.cond_dim}), got "
                f"{tuple(action_embeddings.shape)}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="stack per-step conditioning embeddings on a horizon axis",
            )
        states: list[LatentState] = []
        current = latent
        for step in range(action_embeddings.shape[1]):
            current = self.forward(current, action_embeddings[:, step, :])
            states.append(current)
        return states

    def prediction_residual(
        self,
        latent: LatentState,
        action_embedding: Tensor,
        next_latent: LatentState,
    ) -> Tensor:
        """``g_phi(f(x_t), a_t) - sg[f(x_{t+1})]`` — the residual the Objective squares (RFC-0008 3).

        The ``detach`` on the target ``next_latent`` is the stop-gradient contract: no gradient flows
        into ``f_theta`` through the target branch, only through the input branch and ``g_phi``.
        """
        predicted = self.forward(latent, action_embedding)
        return predicted.tokens - next_latent.tokens.detach()


def build_predictor(cfg: Any) -> Predictor:
    """Construct a :class:`Predictor` from the model config (RFC-0008 3).

    Reads ``cfg.model``: ``d`` (latent dim), ``num_tokens`` (``N``), ``cond_dim`` (default ``d``),
    ``predictor_depth``/``predictor_width`` (defaults ``depth``/``d``), ``num_heads`` (default a divisor
    of width). Raises :class:`~lensemble.errors.ConfigError` on inconsistent dims (non-positive, or a
    width not divisible by ``num_heads``).
    """
    model = getattr(cfg, "model", None)
    if model is None:
        raise ConfigError(
            "config has no `model` sub-config",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="provide cfg.model with d, num_tokens, predictor_depth, predictor_width",
        )
    d = int(model.d)
    num_tokens = int(model.num_tokens)
    cond_dim = int(getattr(model, "cond_dim", d))
    width = int(getattr(model, "predictor_width", d))
    depth = int(getattr(model, "predictor_depth", getattr(model, "depth", 12)))
    num_heads = int(getattr(model, "num_heads", _default_heads(width)))
    mlp_ratio = float(getattr(model, "mlp_ratio", 4.0))
    wmcp_version = str(getattr(model, "wmcp_version", WMCP_VERSION))

    if min(d, num_tokens, cond_dim, width, depth, num_heads) <= 0:
        raise ConfigError(
            f"predictor dims must be positive: d={d} num_tokens={num_tokens} "
            f"cond_dim={cond_dim} width={width} depth={depth} num_heads={num_heads}",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="set all predictor dimensions > 0",
        )
    if width % num_heads != 0:
        raise ConfigError(
            f"predictor width ({width}) must be divisible by num_heads ({num_heads})",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="choose a num_heads that divides predictor_width",
        )
    return Predictor(
        d=d,
        num_tokens=num_tokens,
        cond_dim=cond_dim,
        depth=depth,
        width=width,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        wmcp_version=wmcp_version,
    )
