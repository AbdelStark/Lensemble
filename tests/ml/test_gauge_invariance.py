"""Gauge invariance of the composite objective (07 §2.1 / RFC-0002 4). Issue #13. CPU fp32.

Under the gauge transform f -> Qf, g -> QgQ^T for Q in O(d), the prediction term is invariant
(||Qr|| = ||r||) and SIGReg is invariant when its sketch co-rotates (A -> QA): the sketch is a frame of
latent-space directions, and SIGReg_{QA}(Qf) = SIGReg_A(f) exactly (projections are unchanged). Only the
anchor term breaks the symmetry, by design. Mode (a): lambda_anc=0 must be invariant; mode (b): with the
anchor active the loss must NOT be invariant (a guard against an anchor that silently does nothing).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from lensemble.contracts import WMCP_VERSION, LatentState
from lensemble.model.objective import Objective
from lensemble.model.predictor import Predictor, build_predictor
from lensemble.model.sigreg import build_sketch

_D, _N, _COND, _FEAT, _STEPS = 8, 4, 8, 6, 3


class _TinyEncoder(nn.Module):
    """A linear stand-in f_theta: maps obs (B, feat) -> a LatentState (B, N, d)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(_FEAT, _N * _D)

    def forward(self, obs: Tensor) -> LatentState:
        tokens = self.lin(obs).reshape(obs.shape[0], _N, _D)
        return LatentState(
            tokens=tokens, num_tokens=_N, dim=_D, wmcp_version=WMCP_VERSION
        )


class _RotatedEncoder(nn.Module):
    """f -> Qf: rotate every d-vector of the encoder output by Q (tokens @ Q^T)."""

    def __init__(self, base: nn.Module, q: Tensor) -> None:
        super().__init__()
        self.base = base
        self.q = q

    def forward(self, obs: Tensor) -> LatentState:
        ls = self.base(obs)
        return LatentState(
            tokens=ls.tokens @ self.q.T,
            num_tokens=ls.num_tokens,
            dim=ls.dim,
            wmcp_version=ls.wmcp_version,
        )


class _ConjugatedPredictor(nn.Module):
    """g -> QgQ^T: rotate the latent into the base frame (Q^T), predict, rotate back (Q)."""

    def __init__(self, base: nn.Module, q: Tensor) -> None:
        super().__init__()
        self.base = base
        self.q = q

    def forward(self, latent: LatentState, action_embedding: Tensor) -> LatentState:
        z = LatentState(
            tokens=latent.tokens @ self.q,
            num_tokens=latent.num_tokens,
            dim=latent.dim,
            wmcp_version=latent.wmcp_version,
        )
        out = self.base(z, action_embedding)
        return LatentState(
            tokens=out.tokens @ self.q.T,
            num_tokens=out.num_tokens,
            dim=out.dim,
            wmcp_version=out.wmcp_version,
        )

    def prediction_residual(
        self, latent: LatentState, action_embedding: Tensor, next_latent: LatentState
    ) -> Tensor:
        return (
            self.forward(latent, action_embedding).tokens - next_latent.tokens.detach()
        )


def _random_orthogonal(d: int, gen: torch.Generator) -> Tensor:
    a = torch.randn(d, d, generator=gen)
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)  # Haar-distributed


def _window() -> SimpleNamespace:
    torch.manual_seed(1)
    return SimpleNamespace(obs=torch.randn(_STEPS + 1, _FEAT))


def _tiny_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            d=_D,
            num_tokens=_N,
            cond_dim=_COND,
            predictor_depth=2,
            predictor_width=16,
            num_heads=4,
        )
    )


def _build(
    gen: torch.Generator,
) -> tuple[_TinyEncoder, Predictor, SimpleNamespace, Tensor, Tensor]:
    torch.manual_seed(0)
    encoder = _TinyEncoder().eval()
    predictor = build_predictor(_tiny_cfg()).eval()
    window = _window()
    action_embedding = torch.randn(_STEPS, _COND, generator=gen)
    sketch = build_sketch(seed=123, d=_D, sketch_dim=32)
    return encoder, predictor, window, action_embedding, sketch


def test_objective_invariant_under_random_rotation(rng: torch.Generator) -> None:
    encoder, predictor, window, action_embedding, sketch = _build(rng)
    q = _random_orthogonal(_D, rng)

    base = Objective(
        lambda_pred=1.0, lambda_sig=0.5, lambda_anc=0.0, sketch_seed=123, sketch=sketch
    )
    rot = Objective(
        lambda_pred=1.0,
        lambda_sig=0.5,
        lambda_anc=0.0,
        sketch_seed=123,
        sketch=q @ sketch,
    )
    with torch.no_grad():
        base_terms = base(encoder, predictor, window, action_embedding)
        rot_terms = rot(
            _RotatedEncoder(encoder, q),
            _ConjugatedPredictor(predictor, q),
            window,
            action_embedding,
        )
    # prediction + SIGReg are invariant under the co-rotated gauge transform
    assert math.isclose(
        float(base_terms.total), float(rot_terms.total), rel_tol=1e-4, abs_tol=1e-6
    )
    assert math.isclose(
        float(base_terms.pred), float(rot_terms.pred), rel_tol=1e-4, abs_tol=1e-6
    )
    assert math.isclose(
        float(base_terms.sigreg), float(rot_terms.sigreg), rel_tol=1e-4, abs_tol=1e-6
    )


def test_anchor_breaks_the_symmetry(rng: torch.Generator) -> None:
    encoder, predictor, window, action_embedding, sketch = _build(rng)
    q = _random_orthogonal(_D, rng)
    fixed_target = torch.randn(_N, _D, generator=rng)
    probe = torch.ones(1, _FEAT)

    def anchor(enc: object) -> Tensor:
        emb = enc(probe).tokens.reshape(_N, _D)  # type: ignore[operator]
        return ((emb - fixed_target) ** 2).mean()

    base = Objective(
        lambda_pred=1.0,
        lambda_sig=0.5,
        lambda_anc=1.0,
        sketch_seed=123,
        sketch=sketch,
        anchor=anchor,
    )
    rot = Objective(
        lambda_pred=1.0,
        lambda_sig=0.5,
        lambda_anc=1.0,
        sketch_seed=123,
        sketch=q @ sketch,
        anchor=anchor,
    )
    with torch.no_grad():
        base_terms = base(encoder, predictor, window, action_embedding)
        rot_terms = rot(
            _RotatedEncoder(encoder, q),
            _ConjugatedPredictor(predictor, q),
            window,
            action_embedding,
        )
    # the anchor pins the frame: rotating the encoder must change the total (guards a no-op anchor)
    assert not math.isclose(
        float(base_terms.total), float(rot_terms.total), rel_tol=1e-3, abs_tol=1e-5
    )
    assert not math.isclose(
        float(base_terms.anchor), float(rot_terms.anchor), rel_tol=1e-3, abs_tol=1e-5
    )


def test_loss_terms_are_fp32_scalars_and_total_requires_grad(
    rng: torch.Generator,
) -> None:
    encoder, predictor, window, action_embedding, sketch = _build(rng)
    terms = Objective(
        lambda_pred=1.0, lambda_sig=0.5, lambda_anc=0.0, sketch_seed=123, sketch=sketch
    )(encoder, predictor, window, action_embedding)
    for field in (terms.pred, terms.sigreg, terms.anchor, terms.total):
        assert field.dtype == torch.float32
        assert field.ndim == 0
    assert terms.total.requires_grad  # the value .backward() is called on
