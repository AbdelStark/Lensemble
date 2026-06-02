"""Property variant of the gauge-invariance test (07 §2.1): many random Q via hypothesis. Issue #13.

Draws many Haar-distributed Q in O(d) and asserts the co-rotated objective (lambda_anc=0) is invariant,
catching non-generic rotations an example test would miss. The sketch co-rotates with the gauge frame
(A -> QA), under which prediction + SIGReg are invariant.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from hypothesis import given
from hypothesis import strategies as st
from torch import Tensor, nn

from lensemble.contracts import WMCP_VERSION, LatentState
from lensemble.model.objective import Objective
from lensemble.model.predictor import build_predictor
from lensemble.model.sigreg import build_sketch

_D, _N, _COND, _FEAT, _STEPS = 8, 4, 8, 6, 3


class _Enc(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(_FEAT, _N * _D)

    def forward(self, obs: Tensor) -> LatentState:
        return LatentState(
            tokens=self.lin(obs).reshape(obs.shape[0], _N, _D),
            num_tokens=_N,
            dim=_D,
            wmcp_version=WMCP_VERSION,
        )


class _Rot(nn.Module):
    def __init__(self, base: nn.Module, q: Tensor) -> None:
        super().__init__()
        self.base, self.q = base, q

    def forward(self, obs: Tensor) -> LatentState:
        ls = self.base(obs)
        return LatentState(ls.tokens @ self.q.T, ls.num_tokens, ls.dim, ls.wmcp_version)


class _Conj(nn.Module):
    def __init__(self, base: nn.Module, q: Tensor) -> None:
        super().__init__()
        self.base, self.q = base, q

    def forward(self, latent: LatentState, action_embedding: Tensor) -> LatentState:
        z = LatentState(
            latent.tokens @ self.q, latent.num_tokens, latent.dim, latent.wmcp_version
        )
        out = self.base(z, action_embedding)
        return LatentState(
            out.tokens @ self.q.T, out.num_tokens, out.dim, out.wmcp_version
        )

    def prediction_residual(
        self, latent: LatentState, action_embedding: Tensor, next_latent: LatentState
    ) -> Tensor:
        return (
            self.forward(latent, action_embedding).tokens - next_latent.tokens.detach()
        )


torch.manual_seed(0)
_ENC = _Enc().eval()
_PRED = build_predictor(
    SimpleNamespace(
        model=SimpleNamespace(
            d=_D,
            num_tokens=_N,
            cond_dim=_COND,
            predictor_depth=2,
            predictor_width=16,
            num_heads=4,
        )
    )
).eval()
_WINDOW = SimpleNamespace(obs=torch.randn(_STEPS + 1, _FEAT))
_ACTION = torch.randn(_STEPS, _COND)
_SKETCH = build_sketch(seed=123, d=_D, sketch_dim=32)


def _orthogonal(seed: int) -> Tensor:
    gen = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(_D, _D, generator=gen))
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


@given(seed=st.integers(min_value=1, max_value=10_000))
def test_objective_invariant_over_many_rotations(seed: int) -> None:
    q = _orthogonal(seed)
    base = Objective(
        lambda_pred=1.0, lambda_sig=0.5, lambda_anc=0.0, sketch_seed=123, sketch=_SKETCH
    )
    rot = Objective(
        lambda_pred=1.0,
        lambda_sig=0.5,
        lambda_anc=0.0,
        sketch_seed=123,
        sketch=q @ _SKETCH,
    )
    with torch.no_grad():
        base_total = float(base(_ENC, _PRED, _WINDOW, _ACTION).total)
        rot_total = float(rot(_Rot(_ENC, q), _Conj(_PRED, q), _WINDOW, _ACTION).total)
    assert abs(base_total - rot_total) <= 1e-4 * max(1.0, abs(base_total))
