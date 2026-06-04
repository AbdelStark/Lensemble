"""lensemble.model.sigreg — SIGReg anti-collapse regularizer (docs/rfcs/RFC-0008 6).

SIGReg pushes the embedding marginal toward ``N(0, I_d)`` by projecting onto ``sketch_dim`` shared random
directions and matching each projected 1-D marginal to a standard Gaussian via the Epps-Pulley
characteristic-function statistic (Cramer-Wold: a distribution is standard normal iff all of its 1-D
projections are). The sketch ``A`` is built deterministically from the broadcast round seed ``s_t``, so
every participant minimizes the identical regularizer (``INV-SKETCH-CONSISTENCY``). A shared sketch gives
objective consistency but does **not** close the gauge — the frame fix is the anchor term, not the sketch.

Reduce-within-trust-domain (``INV-RESIDENCY``): the per-sample projection statistics may be reduced
freely within a participant's inner-parallel group, but neither the raw projections ``Az``, the
embeddings, nor any private-batch statistic may be serialized across a participant boundary; only scalar
SIGReg loss values (aggregate statistics) may be logged. The egress guard (``lensemble.data.residency``)
rejects an attempt to emit a projection/embedding, fail-closed.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Epps-Pulley integration grid: `ep_knots` points over [-T_MAX, T_MAX] with a Gaussian weight. The
# standard-normal characteristic function exp(-t^2/2) is ~3e-6 at t=5, so the grid covers its support.
_T_MAX = 5.0


def build_sketch(seed: int, d: int, sketch_dim: int = 64) -> Tensor:
    """Deterministic SIGReg projection matrix ``A`` of shape ``(d, sketch_dim)`` (RFC-0008 6a).

    Pre: ``seed == GlobalState.sketch_seed`` for the current round (``INV-SKETCH-CONSISTENCY``).
    Post: bitwise-identical across participants for identical ``(seed, d, sketch_dim)``; columns are
    i.i.d. Gaussian directions L2-normalized to unit norm (each projection is a coordinate of a random
    rotation, matching the ``O(d)`` symmetry argument of RFC-0002 2).
    """
    gen = torch.Generator().manual_seed(int(seed) % (2**63))
    a = torch.randn(d, sketch_dim, generator=gen, dtype=torch.float32)
    return a / a.norm(dim=0, keepdim=True).clamp_min(1e-12)


def sigreg_statistic(
    embeddings: Tensor, sketch: Tensor, *, ep_knots: int = 17
) -> Tensor:
    """Mean Epps-Pulley characteristic-function distance to ``N(0,1)`` over projected directions (RFC-0008 6b).

    Input: ``embeddings`` of shape ``(M, d)`` (flattened over ``(B, N)``) and ``sketch`` ``A`` of shape
    ``(d, sketch_dim)``. Output: a 0-dim fp32 tensor — ~0 for a standard-normal sample, large for
    non-normal. Projection and standardization are fp32 with a fixed per-knot reduction order, so the
    statistic is reproducible given identical inputs (``conventions 9``).
    """
    emb = embeddings.reshape(-1, embeddings.shape[-1]).to(torch.float32)
    # Device-follow the embeddings: the sketch is built deterministically on CPU (build_sketch) and the
    # Epps-Pulley grid below would default to CPU, but `emb` lives on the encoder's device (CUDA in the
    # GPU inner loop). Move both onto `emb.device` so the matmul/CF reduction stay on one device. The
    # moves are value-preserving (CPU<->CUDA fp32 copy), so the statistic is unchanged (conventions 9).
    device = emb.device
    proj = emb @ sketch.to(device=device, dtype=torch.float32)  # (M, sketch_dim)
    mu = proj.mean(dim=0, keepdim=True)
    sd = proj.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    u = (proj - mu) / sd  # standardized per direction

    t = torch.linspace(
        -_T_MAX, _T_MAX, ep_knots, dtype=torch.float32, device=device
    )  # (K,)
    weight = torch.exp(-0.5 * t * t)  # Gaussian integration weight (K,)
    target_re = torch.exp(-0.5 * t * t)  # standard-normal CF (real; imag 0)

    tu = u.unsqueeze(-1) * t  # (M, S, K)
    re = torch.cos(tu).mean(dim=0)  # (S, K) empirical CF real part
    im = torch.sin(tu).mean(dim=0)  # (S, K) empirical CF imaginary part
    diff_sq = (re - target_re) ** 2 + im**2  # (S, K) |phi_emp - phi_target|^2
    per_direction = (diff_sq * weight).sum(
        dim=-1
    ) / weight.sum()  # (S,) weighted CF distance
    return per_direction.mean()
