"""Shared pytest harness: seeded fixtures, the hypothesis profile, and the named tolerance constants.

The test pyramid layout (07-testing-strategy 1) is ``tests/{unit,property,integration,ml,regression,
e2e}/``. Fixtures take a seeded generator and never touch a global RNG, so the property layer is
reproducible on hypothesis replay (07 7). Numerical tolerances are named constants set in one place and
cited by id — never inlined as magic numbers (07 6). All fixtures are tiny, CPU-only, and download
nothing (07 7).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import SimpleNamespace

import pytest
import torch
from hypothesis import HealthCheck, settings

# --- hypothesis profile: deadline off, derandomized so a failing case replays (07 7) ---
settings.register_profile(
    "lensemble",
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("lensemble")


# --- numerical tolerance policy (07 6): named, single-sourced, cited by id ---
@dataclass(frozen=True)
class Tolerances:
    RTOL_LOSS: float = 1e-4  # fp32 loss-term comparisons
    ATOL_LOSS: float = 1e-6
    SIGREG_NULL_TOL: float = 2e-4  # SIGReg statistic on a standard-normal sample (≈0)
    SIGREG_SIGNAL_FLOOR: float = (
        5e-4  # SIGReg statistic on a strongly non-normal sample
    )
    RTOL_PROC: float = 1e-4  # Procrustes residual / closed-form vs brute force
    ATOL_ORTHO: float = 1e-5  # orthogonality of Q (QᵀQ ≈ I)
    ANGLE_TOL_DEG: float = 1.0  # frame-drift angle agreement (degrees)
    RTOL_DP: float = 1e-1  # DP noised-statistic mean (loose; statistical)
    RTOL_DP_STD: float = 1e-1  # DP noised-statistic std
    RTOL_BF16: float = 1e-2  # bf16 forward vs fp32 reference
    ATOL_BF16: float = 1e-2
    RTOL_AGG: float = (
        0.0  # aggregation/outer-step path is bitwise-exact (INV-AGG-DETERMINISM)
    )


TOLERANCES = Tolerances()


@pytest.fixture
def tol() -> Tolerances:
    """The named numerical tolerances (07 6). Cite by id, e.g. ``tol.RTOL_LOSS``."""
    return TOLERANCES


# --- seeded RNG (never the global RNG) ---
@pytest.fixture
def rng() -> torch.Generator:
    """A fresh, seeded ``torch.Generator`` (seed 0). Fixtures and tests draw from it, not global RNG."""
    g = torch.Generator()
    g.manual_seed(0)
    return g


# --- tiny, CPU-only synthetic fixtures (07 7): no download, no module larger than 2 layers ---
class _TinyEncoder(torch.nn.Module):
    """A 2-layer linear stand-in for the encoder (``d=8``): maps ``(B, d) -> (B, d)`` embeddings."""

    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(d, d)
        self.fc2 = torch.nn.Linear(d, d)
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(x)))


def build_tiny_warmstart(seed: int = 0, d: int = 8) -> _TinyEncoder:
    """Deterministically-initialized tiny encoder; byte-identical for a fixed ``(seed, d)``."""
    gen = torch.Generator().manual_seed(seed)
    enc = _TinyEncoder(d)
    with torch.no_grad():
        for p in enc.parameters():
            p.copy_(torch.randn(p.shape, generator=gen))
    enc.eval()
    return enc


def build_synthetic_probe(
    seed: int = 0, d: int = 8, num_points: int = 16, k: int = 8
) -> SimpleNamespace:
    """Deterministic probe points with ``k >= d`` landmarks (07 7)."""
    assert k >= d, "synthetic probe must carry k >= d landmarks"
    gen = torch.Generator().manual_seed(seed)
    points = torch.randn(num_points, d, generator=gen)
    return SimpleNamespace(points=points, landmark_idx=torch.arange(k), d=d, k=k)


class ToyEnv:
    """An in-memory stable-worldmodel-style env with a closed-form linear transition (07 7).

    ``step(action)`` advances ``state -> decay * state + action``; the goal is the origin. Deterministic,
    CPU-only, no rollout randomness — a stand-in for latent-MPC eval tests.
    """

    def __init__(self, dim: int = 4, decay: float = 0.9) -> None:
        self.dim = dim
        self.decay = decay
        self.goal = torch.zeros(dim)
        self._state = torch.ones(dim)

    def reset(self) -> torch.Tensor:
        self._state = torch.ones(self.dim)
        return self._state.clone()

    def step(self, action: torch.Tensor) -> torch.Tensor:
        self._state = self.decay * self._state + action
        return self._state.clone()

    def goal_distance(self) -> float:
        return float((self._state - self.goal).norm())


@pytest.fixture
def tiny_warmstart() -> _TinyEncoder:
    return build_tiny_warmstart()


@pytest.fixture
def synthetic_probe() -> SimpleNamespace:
    return build_synthetic_probe()


@pytest.fixture
def toy_env() -> ToyEnv:
    return ToyEnv()


@pytest.fixture
def make_tiny_warmstart():
    """The ``build_tiny_warmstart(seed, d)`` builder (so tests can construct multiple instances)."""
    return build_tiny_warmstart


@pytest.fixture
def make_synthetic_probe():
    """The ``build_synthetic_probe(seed, d, num_points, k)`` builder."""
    return build_synthetic_probe


@pytest.fixture
def tolerance_field_names() -> set[str]:
    return {f.name for f in fields(Tolerances)}
