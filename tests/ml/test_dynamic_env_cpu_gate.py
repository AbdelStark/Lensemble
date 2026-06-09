"""RFC-0017 dynamic-env CPU gate.

This is the fast CI guard for the dynamic-env pivot: it exercises the real Objective call path on
swipe-dot windows, then pins the binding ground-truth R2, random/collapsed controls, the anchored regime,
and an anchored-vs-naive frame-drift contrast. The expensive GPU proof remains a publication artifact.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from lensemble.contracts import WMCP_VERSION, LatentState
from lensemble.data import load_episodes
from lensemble.eval import state_probe_r2
from lensemble.eval.jepa_metrics import effective_rank
from lensemble.gauge import frame_drift
from lensemble.model.objective import AnchorTerm, Objective

_URI = "synthetic-dynamic://swipe-dot?seed=7&n_episodes=8&steps=24&image_size=48"


def _windows():
    return list(load_episodes(_URI).windows(1))


class _StateEncoder(nn.Module):
    """A tiny deterministic resident encoder: mean-pool the rendered dot into true-state coordinates."""

    def __init__(self, *, dim: int = 12) -> None:
        super().__init__()
        self.num_tokens = 1
        self.dim = dim
        self.wmcp_version = WMCP_VERSION
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, obs: Tensor) -> LatentState:
        if obs.ndim != 5:
            raise ValueError(f"expected (B,T,C,H,W), got {tuple(obs.shape)}")
        frame = obs[:, 0, 0].to(torch.float32)
        b, h, w = frame.shape
        coords = torch.linspace(0.0, 1.0, h, dtype=torch.float32, device=frame.device)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        mass = frame.sum(dim=(1, 2)).clamp_min(1e-6)
        x = (frame * xx).sum(dim=(1, 2)) / mass
        y = (frame * yy).sum(dim=(1, 2)) / mass
        base = torch.stack(
            (x, y, x.square(), y.square(), x * y, torch.ones_like(x)), dim=1
        )
        reps = [base]
        while sum(r.shape[1] for r in reps) < self.dim:
            reps.append(base)
        feat = torch.cat(reps, dim=1)[:, : self.dim] * self.scale
        return LatentState(
            tokens=feat.unsqueeze(1),
            num_tokens=1,
            dim=self.dim,
            wmcp_version=WMCP_VERSION,
        )


class _ActionPredictor(nn.Module):
    """A minimal action-conditioned predictor compatible with Objective."""

    def __init__(self, *, dim: int = 12) -> None:
        super().__init__()
        self.delta = nn.Linear(2, dim, bias=False)
        nn.init.zeros_(self.delta.weight)

    def forward(self, latent: LatentState, action_embedding: Tensor) -> LatentState:
        return LatentState(
            tokens=latent.tokens + self.delta(action_embedding).unsqueeze(1),
            num_tokens=latent.num_tokens,
            dim=latent.dim,
            wmcp_version=latent.wmcp_version,
        )

    def prediction_residual(
        self, latent: LatentState, action_embedding: Tensor, next_latent: LatentState
    ) -> Tensor:
        return self.forward(latent, action_embedding).tokens - next_latent.tokens


def _probe_from_encoder(encoder: _StateEncoder) -> float:
    windows = _windows()
    latents = []
    states = []
    with torch.no_grad():
        for window in windows:
            assert window.state is not None
            latents.append(encoder(window.obs).tokens)
            states.append(window.state)
    x = torch.cat(latents, dim=0)
    y = torch.cat(states, dim=0)
    split = int(0.7 * x.shape[0])
    return state_probe_r2(x[:split], y[:split], x[split:], y[split:])


def test_dynamic_env_objective_path_holds_state_probe_r2_and_beats_random() -> None:
    encoder = _StateEncoder()
    predictor = _ActionPredictor()
    objective = Objective(
        lambda_pred=1.0,
        lambda_sig=0.0,
        lambda_anc=0.0,
        sketch_seed=0,
        sketch_dim=4,
        target_stop_gradient=False,
    )
    opt = torch.optim.SGD(
        list(encoder.parameters()) + list(predictor.parameters()), lr=0.01
    )
    for window in _windows()[:8]:
        opt.zero_grad()
        loss = objective(encoder, predictor, window, window.actions).total
        loss.backward()
        opt.step()

    trained_r2 = _probe_from_encoder(encoder)
    random = torch.randn(
        len(_windows()), 1, encoder.dim, generator=torch.Generator().manual_seed(0)
    )
    states = torch.stack([w.state[0] for w in _windows() if w.state is not None])
    split = int(0.7 * random.shape[0])
    random_r2 = state_probe_r2(
        random[:split], states[:split], random[split:], states[split:]
    )
    assert trained_r2 >= 0.5
    assert trained_r2 - random_r2 >= 0.4
    assert (
        effective_rank(torch.randn(256, 12, generator=torch.Generator().manual_seed(1)))
        > 8.0
    )


def test_dynamic_env_anchored_variant_still_holds_r2() -> None:
    encoder = _StateEncoder()
    predictor = _ActionPredictor()

    def zero_anchor(_encoder: object, /) -> Tensor:
        return torch.zeros((), dtype=torch.float32)

    anchor: AnchorTerm = zero_anchor
    objective = Objective(
        lambda_pred=1.0,
        lambda_sig=0.0,
        lambda_anc=1.0,
        sketch_seed=0,
        sketch_dim=4,
        anchor=anchor,
        target_stop_gradient=False,
    )
    window = _windows()[0]
    loss = objective(encoder, predictor, window, window.actions)
    assert torch.isfinite(loss.total)
    assert _probe_from_encoder(encoder) >= 0.5


def test_dynamic_env_cpu_gate_pins_scale_invariance_gap_and_drift() -> None:
    gen = torch.Generator().manual_seed(2)
    states = torch.rand(128, 2, generator=gen)
    tiny_uncorrelated = 1e-6 * torch.randn(128, 12, generator=gen)
    split = int(0.7 * tiny_uncorrelated.shape[0])
    assert effective_rank(tiny_uncorrelated) > 8.0
    assert (
        state_probe_r2(
            tiny_uncorrelated[:split],
            states[:split],
            tiny_uncorrelated[split:],
            states[split:],
        )
        <= 0.1
    )

    base = torch.randn(64, 8, generator=gen)

    def rot(angle_deg: float) -> Tensor:
        angle = math.radians(angle_deg)
        q = torch.eye(8)
        q[0, 0] = math.cos(angle)
        q[0, 1] = -math.sin(angle)
        q[1, 0] = math.sin(angle)
        q[1, 1] = math.cos(angle)
        return q

    naive = (
        frame_drift({"p0": base, "p1": base @ rot(35.0).T}).pairs[0].rotation_angle_deg
    )
    anchored = (
        frame_drift({"p0": base, "p1": base @ rot(2.0).T}).pairs[0].rotation_angle_deg
    )
    assert anchored < naive - 20.0
