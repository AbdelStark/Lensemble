"""Objective gradients: finite-difference check per term + the stop-gradient contract (RFC-0008). #13.

Verifies each term's analytic gradient against a central finite-difference estimate (fp32, loose by
construction — fp32 FD truncation+roundoff is ~1e-3), and that the prediction target f(x_{t+1}) is
stop-gradiented: the objective's encoder gradient equals the gradient of the explicitly-detached
residual and differs from the no-detach residual (07 §2.1).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from lensemble.contracts import WMCP_VERSION, LatentState
from lensemble.model.objective import Objective
from lensemble.model.predictor import Predictor, build_predictor

_D, _N, _COND, _FEAT, _STEPS = 8, 4, 8, 6, 3


class _TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(_FEAT, _N * _D)

    def forward(self, obs: Tensor) -> LatentState:
        tokens = self.lin(obs).reshape(obs.shape[0], _N, _D)
        return LatentState(
            tokens=tokens, num_tokens=_N, dim=_D, wmcp_version=WMCP_VERSION
        )


def _setup() -> tuple[_TinyEncoder, Predictor, SimpleNamespace, Tensor]:
    torch.manual_seed(0)
    encoder = _TinyEncoder().eval()  # eval: deterministic forward (no dropout) for FD
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            d=_D,
            num_tokens=_N,
            cond_dim=_COND,
            predictor_depth=2,
            predictor_width=16,
            num_heads=4,
        )
    )
    predictor = build_predictor(cfg).eval()
    window = SimpleNamespace(obs=torch.randn(_STEPS + 1, _FEAT))
    action_embedding = torch.randn(_STEPS, _COND)
    return encoder, predictor, window, action_embedding


def _fd_check_param(loss_fn, param: Tensor) -> None:
    """Central-difference the largest-magnitude gradient entry of ``param`` against the analytic value.

    ``param`` must NOT also feed the stop-gradiented target branch — otherwise FD (which moves the actual
    target value) would legitimately disagree with the detached analytic gradient. We therefore FD-check
    predictor params for the prediction term and encoder params for SIGReg (no target there).
    """
    param.grad = None
    loss_fn().backward()
    grad = param.grad
    assert grad is not None
    flat_idx = int(grad.abs().argmax())
    idx = divmod(flat_idx, grad.shape[1]) if grad.ndim == 2 else (flat_idx,)
    analytic = float(grad[idx])

    eps = 2e-3
    orig = float(param.data[idx])
    with torch.no_grad():
        param.data[idx] = orig + eps
        plus = float(loss_fn())
        param.data[idx] = orig - eps
        minus = float(loss_fn())
        param.data[idx] = orig
    fd = (plus - minus) / (2 * eps)
    assert math.isclose(analytic, fd, rel_tol=3e-2, abs_tol=1e-3), (analytic, fd)


def test_prediction_term_gradient_matches_finite_difference() -> None:
    # FD-check a PREDICTOR param: g_phi feeds only the input branch, never the detached target.
    encoder, predictor, window, action_embedding = _setup()
    objective = Objective(
        lambda_pred=1.0, lambda_sig=0.0, lambda_anc=0.0, sketch_seed=7
    )
    _fd_check_param(
        lambda: objective(encoder, predictor, window, action_embedding).total,
        predictor.out_proj.weight,
    )


def test_sigreg_term_gradient_matches_finite_difference() -> None:
    # FD-check an ENCODER param: SIGReg depends only on the embeddings, with no stop-gradient.
    encoder, predictor, window, action_embedding = _setup()
    objective = Objective(
        lambda_pred=0.0, lambda_sig=1.0, lambda_anc=0.0, sketch_seed=7, sketch_dim=16
    )
    _fd_check_param(
        lambda: objective(encoder, predictor, window, action_embedding).total,
        encoder.lin.weight,
    )


def test_target_branch_is_stop_gradiented() -> None:
    encoder, predictor, window, action_embedding = _setup()
    objective = Objective(
        lambda_pred=1.0, lambda_sig=0.0, lambda_anc=0.0, sketch_seed=7
    )

    def encode_split() -> tuple[LatentState, LatentState]:
        tokens = encoder(window.obs).tokens
        inp = LatentState(tokens[:-1], _N, _D, WMCP_VERSION)
        tgt = LatentState(tokens[1:], _N, _D, WMCP_VERSION)
        return inp, tgt

    def encoder_grad() -> Tensor:
        grad = encoder.lin.weight.grad
        assert grad is not None
        return grad.clone()

    # the objective's encoder gradient (target detached inside prediction_residual)
    encoder.zero_grad(set_to_none=True)
    objective(encoder, predictor, window, action_embedding).total.backward()
    g_objective = encoder_grad()

    # manual residual WITH detach -> must match the objective exactly
    encoder.zero_grad(set_to_none=True)
    inp, tgt = encode_split()
    res_detached = predictor.forward(inp, action_embedding).tokens - tgt.tokens.detach()
    res_detached.pow(2).mean().backward()
    g_detached = encoder_grad()

    # manual residual WITHOUT detach -> target branch contributes, gradient must differ
    encoder.zero_grad(set_to_none=True)
    inp, tgt = encode_split()
    res_nodetach = predictor.forward(inp, action_embedding).tokens - tgt.tokens
    res_nodetach.pow(2).mean().backward()
    g_nodetach = encoder_grad()

    assert torch.allclose(
        g_objective, g_detached, atol=1e-6
    )  # detach is the objective's behavior
    assert not torch.allclose(
        g_objective, g_nodetach, atol=1e-6
    )  # and it removes the target branch


def test_lewm_base_mode_keeps_target_branch_live() -> None:
    """Claim-mode LeWorldModel base recipe: no stop-gradient on f(x_{t+1}) (#191)."""
    encoder, predictor, window, action_embedding = _setup()
    objective = Objective(
        lambda_pred=1.0,
        lambda_sig=0.0,
        lambda_anc=0.0,
        sketch_seed=7,
        target_stop_gradient=False,
    )

    def encode_split() -> tuple[LatentState, LatentState]:
        tokens = encoder(window.obs).tokens
        inp = LatentState(tokens[:-1], _N, _D, WMCP_VERSION)
        tgt = LatentState(tokens[1:], _N, _D, WMCP_VERSION)
        return inp, tgt

    def encoder_grad() -> Tensor:
        grad = encoder.lin.weight.grad
        assert grad is not None
        return grad.clone()

    encoder.zero_grad(set_to_none=True)
    objective(encoder, predictor, window, action_embedding).total.backward()
    g_objective = encoder_grad()

    encoder.zero_grad(set_to_none=True)
    inp, tgt = encode_split()
    res_live = predictor.forward(inp, action_embedding).tokens - tgt.tokens
    res_live.pow(2).mean().backward()
    g_live = encoder_grad()

    encoder.zero_grad(set_to_none=True)
    inp, tgt = encode_split()
    res_detached = predictor.forward(inp, action_embedding).tokens - tgt.tokens.detach()
    res_detached.pow(2).mean().backward()
    g_detached = encoder_grad()

    assert torch.allclose(g_objective, g_live, atol=1e-6)
    assert not torch.allclose(g_objective, g_detached, atol=1e-6)
