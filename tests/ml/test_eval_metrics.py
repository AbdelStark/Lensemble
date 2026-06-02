"""Evaluation metric bodies (RFC-0005 §3-4; #53).

Effective dimension (the collapse guard) matches the analytic participation ratio of the same covariance
within the named fp32 tolerance and reports ~1 on a rank-1 (collapsed) input; the byte accountant matches
08 §4 with and without int8; planning success/cost and the linear probe report in-range values; an
out-of-range metric raises EvaluationError. Placed in tests/ml (CI-gated).
"""

from __future__ import annotations

import pytest
import torch

from lensemble.errors import EvaluationError
from lensemble.eval import (
    comm_bytes,
    effective_dim,
    linear_probe_accuracy,
    planning_cost,
    quant_ratio,
    success_rate,
)
from lensemble.eval.mpc import PlanResult

# --- effective dimension (the collapse guard) ---


def test_effective_dim_matches_analytic_participation_ratio(tol) -> None:
    gen = torch.Generator().manual_seed(0)
    spectrum = torch.tensor([4.0, 2.0, 1.0, 0.25])  # known eigenvalue scale
    x = torch.randn(4000, 4, generator=gen) * spectrum.sqrt()
    # independent reference: (sum lambda)^2 / sum lambda^2 over the same sample covariance
    centered = x - x.mean(0, keepdim=True)
    cov = centered.T @ centered / (x.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    analytic = float((eig.sum() ** 2) / eig.square().sum())
    assert effective_dim(x) == pytest.approx(
        analytic, rel=tol.RTOL_EFFDIM, abs=tol.ATOL_EFFDIM
    )
    assert 1.0 <= effective_dim(x) <= 4.0  # in [1, d]


def test_effective_dim_detects_rank_one_collapse() -> None:
    gen = torch.Generator().manual_seed(1)
    direction = torch.randn(4, generator=gen)
    scales = torch.randn(500, 1, generator=gen)
    collapsed = scales * direction  # every row is a multiple of one direction -> rank 1
    assert effective_dim(collapsed) == pytest.approx(1.0, abs=1e-3)


def test_effective_dim_rejects_degenerate_inputs() -> None:
    with pytest.raises(EvaluationError):
        effective_dim(torch.randn(1, 4))  # < 2 vectors
    with pytest.raises(EvaluationError):
        effective_dim(torch.ones(8, 4))  # zero variance (all identical)


# --- the communication-byte accountant (08 §4) ---


def test_comm_bytes_matches_perf_budget() -> None:
    params = 1000
    assert comm_bytes(params) == 4 * params  # fp32 wire dtype
    assert comm_bytes(params, quantized=True) == 1 * params + 4  # int8 + 4-byte scale
    assert quant_ratio(params) == pytest.approx(4.0, rel=0.01)  # ~4x reduction


def test_comm_bytes_rejects_negative_params() -> None:
    with pytest.raises(EvaluationError):
        comm_bytes(-1)


# --- planning success and cost ---


def test_success_rate_is_a_fraction() -> None:
    assert success_rate([True, True, False, True]) == 0.75
    assert success_rate([False, False]) == 0.0
    with pytest.raises(EvaluationError):
        success_rate([])


def test_planning_cost_from_plan_results() -> None:
    results = [
        PlanResult(
            actions=torch.zeros(4, 2),
            cost=1.0,
            planner="icem",
            num_samples=64,
            num_iters=3,
            wall_time_s=0.002,
        )
        for _ in range(5)
    ]
    samples, ms = planning_cost(results)
    assert samples == 64 * 3  # samples per action = num_samples * num_iters
    assert ms == pytest.approx(2.0, rel=1e-6)  # 0.002 s -> 2.0 ms
    with pytest.raises(EvaluationError):
        planning_cost([])


# --- supporting: the linear probe ---


def test_linear_probe_accuracy_on_separable_data() -> None:
    gen = torch.Generator().manual_seed(2)
    # two linearly-separable clusters in 4-D
    a = torch.randn(50, 4, generator=gen) + torch.tensor([3.0, 0, 0, 0])
    b = torch.randn(50, 4, generator=gen) - torch.tensor([3.0, 0, 0, 0])
    x = torch.cat([a, b])
    y = torch.cat(
        [torch.zeros(50, dtype=torch.int64), torch.ones(50, dtype=torch.int64)]
    )
    acc = linear_probe_accuracy(x[::2], y[::2], x[1::2], y[1::2])
    assert 0.0 <= acc <= 1.0
    assert acc > 0.9  # separable -> a linear probe recovers it
    with pytest.raises(EvaluationError):
        linear_probe_accuracy(x, y[:10], x, y)  # length mismatch
