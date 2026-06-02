"""Action-conditioned predictor g_phi: shape, AR rollout, stop-gradient contract (RFC-0008 3). #11.

CPU fp32, tiny dims. The stop-gradient (detach on the target f_theta(x_{t+1})) is a contract, not an
optimization detail — a missing detach silently changes the objective (07 §2.1); it is asserted here at
the residual level. The objective-level gradient finite-difference check is owned by #13.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lensemble.contracts import WMCP_VERSION, LatentState, check_latent_state
from lensemble.errors import ConfigError, LensembleErrorCode
from lensemble.model.predictor import Predictor, build_predictor

_D, _N, _COND = 8, 4, 8


def _tiny_cfg(**model_overrides: object) -> SimpleNamespace:
    model: dict[str, object] = {
        "d": _D,
        "num_tokens": _N,
        "cond_dim": _COND,
        "predictor_depth": 2,
        "predictor_width": 16,
        "num_heads": 4,
    }
    model.update(model_overrides)
    return SimpleNamespace(model=SimpleNamespace(**model))


def _latent(batch: int = 2, *, requires_grad: bool = False) -> LatentState:
    tokens = torch.randn(
        batch, _N, _D, dtype=torch.float32, requires_grad=requires_grad
    )
    return LatentState(tokens=tokens, num_tokens=_N, dim=_D, wmcp_version=WMCP_VERSION)


def test_ar_predictor_shape() -> None:
    predictor = build_predictor(_tiny_cfg())
    assert isinstance(predictor, Predictor)
    latent = _latent(batch=2)
    action_embedding = torch.zeros(2, _COND)
    out = predictor(latent, action_embedding)
    assert isinstance(out, LatentState)
    assert tuple(out.tokens.shape) == (2, _N, _D)
    assert check_latent_state(out) is None  # the prediction is WMCP-conformant

    # an autoregressive rollout preserves the (B, N, d) shape + conformance over the horizon
    horizon = 3
    rollout = predictor.rollout(latent, torch.zeros(2, horizon, _COND))
    assert len(rollout) == horizon
    for state in rollout:
        assert tuple(state.tokens.shape) == (2, _N, _D)
        assert check_latent_state(state) is None


def test_stop_gradient_severs_the_target_branch() -> None:
    # prediction_residual detaches f(x_{t+1}); gradient flows into g_phi and the input branch only.
    predictor = build_predictor(_tiny_cfg())
    latent_t = _latent(batch=2, requires_grad=True)
    target_tokens = torch.randn(2, _N, _D, requires_grad=True)
    next_latent = LatentState(
        tokens=target_tokens, num_tokens=_N, dim=_D, wmcp_version=WMCP_VERSION
    )
    residual = predictor.prediction_residual(
        latent_t, torch.zeros(2, _COND), next_latent
    )
    residual.pow(2).sum().backward()

    assert target_tokens.grad is None  # the detached target branch carries no gradient
    assert latent_t.tokens.grad is not None  # the input branch f(x_t) does
    assert any(p.grad is not None for p in predictor.parameters())  # and g_phi learns


def test_build_predictor_rejects_inconsistent_dims() -> None:
    with pytest.raises(ConfigError) as exc:
        build_predictor(_tiny_cfg(predictor_width=10, num_heads=4))  # 10 % 4 != 0
    assert exc.value.code == LensembleErrorCode.CONFIG_INVALID
    with pytest.raises(ConfigError):
        build_predictor(_tiny_cfg(cond_dim=0))  # non-positive dim


def test_forward_rejects_mismatched_inputs() -> None:
    predictor = build_predictor(_tiny_cfg())
    with pytest.raises(ConfigError):
        predictor(_latent(batch=2), torch.zeros(2, _COND + 1))  # wrong cond_dim
    bad_latent = LatentState(
        tokens=torch.zeros(2, _N + 1, _D),
        num_tokens=_N + 1,
        dim=_D,
        wmcp_version=WMCP_VERSION,
    )
    with pytest.raises(ConfigError):
        predictor(bad_latent, torch.zeros(2, _COND))  # wrong num_tokens


def test_conditioning_changes_the_prediction() -> None:
    # the action embedding actually conditions g_phi: different a -> different prediction.
    predictor = build_predictor(_tiny_cfg())
    predictor.eval()
    latent = _latent(batch=1)
    with torch.no_grad():
        a0 = predictor(latent, torch.zeros(1, _COND)).tokens
        a1 = predictor(latent, torch.ones(1, _COND)).tokens
    assert not torch.allclose(a0, a1)
