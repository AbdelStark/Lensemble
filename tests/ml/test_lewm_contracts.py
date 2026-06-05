"""LeWorldModel claim-contract checks: action alignment, no future leakage, and collapse guard."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from lensemble.contracts import WMCP_VERSION, LatentState
from lensemble.model import build_predictor, build_sketch, sigreg_statistic
from lensemble.model.objective import Objective


class _SequenceEncoder:
    """Encode a scalar sequence as one-token, one-dimensional latents."""

    def __call__(self, obs: object) -> LatentState:
        obs_tensor = torch.as_tensor(obs)
        tokens = obs_tensor.reshape(-1, 1, 1).to(torch.float32)
        return LatentState(
            tokens=tokens, num_tokens=1, dim=1, wmcp_version=WMCP_VERSION
        )


class _AddActionPredictor:
    """Known latent dynamics: z_hat[t+1] = z[t] + a[t]."""

    def __call__(
        self, latent: LatentState, action_embedding: torch.Tensor
    ) -> LatentState:
        out = latent.tokens + action_embedding.reshape(-1, 1, 1)
        return LatentState(
            tokens=out,
            num_tokens=latent.num_tokens,
            dim=latent.dim,
            wmcp_version=latent.wmcp_version,
        )

    def prediction_residual(
        self,
        latent: LatentState,
        action_embedding: torch.Tensor,
        next_latent: LatentState,
    ) -> torch.Tensor:
        return self(latent, action_embedding).tokens - next_latent.tokens.detach()


def test_objective_aligns_action_t_to_next_latent_target() -> None:
    """A deterministic system is zero-loss only when action a_t is aligned to z_{t+1}."""
    objective = Objective(
        lambda_pred=1.0,
        lambda_sig=0.0,
        lambda_anc=0.0,
        sketch_seed=0,
        target_stop_gradient=False,
    )
    encoder = _SequenceEncoder()
    predictor = _AddActionPredictor()
    window = SimpleNamespace(obs=torch.tensor([0.0, 1.0, 3.0]))

    aligned = torch.tensor([[1.0], [2.0]])
    misaligned = torch.tensor([[0.0], [0.0]])

    assert float(objective(encoder, predictor, window, aligned).pred) == 0.0
    assert float(objective(encoder, predictor, window, misaligned).pred) > 0.0


def test_predictor_batch_path_does_not_leak_future_transition_rows() -> None:
    """The current teacher-forced transition batch predicts each row independently."""
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            d=8,
            latent_dim=8,
            num_tokens=4,
            cond_dim=8,
            predictor_depth=1,
            predictor_width=8,
            num_heads=2,
        )
    )
    predictor = build_predictor(cfg).eval()
    latent = LatentState(
        tokens=torch.randn(3, 4, 8),
        num_tokens=4,
        dim=8,
        wmcp_version=WMCP_VERSION,
    )
    actions = torch.randn(3, 8)
    with torch.no_grad():
        baseline_first = predictor(latent, actions).tokens[0].clone()
        perturbed = LatentState(
            tokens=latent.tokens.clone(),
            num_tokens=4,
            dim=8,
            wmcp_version=WMCP_VERSION,
        )
        perturbed.tokens[2].add_(100.0)
        perturbed_actions = actions.clone()
        perturbed_actions[2].add_(100.0)
        first_after_future_perturb = predictor(perturbed, perturbed_actions).tokens[0]
    assert torch.allclose(baseline_first, first_after_future_perturb, atol=1e-6)


def test_sigreg_penalizes_zero_and_low_rank_latents_more_than_normal_latents() -> None:
    torch.manual_seed(0)
    sketch = build_sketch(seed=123, d=8, sketch_dim=64)
    normal = torch.randn(2048, 8)
    zero = torch.zeros(2048, 8)
    low_rank = torch.randn(2048, 1).expand(-1, 8)

    normal_stat = float(sigreg_statistic(normal, sketch))
    assert float(sigreg_statistic(zero, sketch)) > normal_stat
    assert float(sigreg_statistic(low_rank, sketch)) > normal_stat
