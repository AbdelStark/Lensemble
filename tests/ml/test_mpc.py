"""Latent MPC planner: families, reproducibility, goal-energy minimization (RFC-0005 3). Issue #51.

A deterministic stub latent dynamics (z' = decay*z + action) with the origin as goal: each planner
family constructs and plans, a fixed seed yields a reproducible plan, the planner drives the latent
goal-energy below the no-op baseline, and an invalid family / diverging rollout raises EvaluationError.

Placed in tests/ml (the issue named tests/eval, which the §8 CI gate does not collect).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from lensemble.errors import EvaluationError, LensembleErrorCode
from lensemble.eval import Planner

_D = 4
_Dynamics = Callable[[Tensor, Tensor], Tensor]


def _linear_dynamics(decay: float = 0.9) -> _Dynamics:
    def dynamics(latents: Tensor, actions: Tensor) -> Tensor:
        return decay * latents + actions  # action_dim == d; goal is the origin

    return dynamics


def _planner(family: str, **kw: object) -> Planner:
    params: dict[str, object] = {
        "family": family,
        "horizon": 4,
        "num_samples": 64,
        "action_dim": _D,
        "seed": 0,
        "num_iters": 3,
    }
    params.update(kw)
    return Planner(**params)  # type: ignore[arg-type]


def test_each_family_constructs_and_plans() -> None:
    z0 = torch.ones(_D)
    goal = torch.zeros(_D)
    for family in ("cem", "icem", "mppi"):
        result = _planner(family).plan(_linear_dynamics(), z0, goal)
        assert result.planner == family
        assert tuple(result.actions.shape) == (4, _D)
        assert result.cost >= 0.0 and torch.isfinite(torch.tensor(result.cost))
        assert result.num_samples == 64 and result.num_iters == 3


def test_plan_is_reproducible_under_fixed_seed() -> None:
    z0, goal = torch.ones(_D), torch.zeros(_D)
    a = _planner("icem", seed=7).plan(_linear_dynamics(), z0, goal)
    b = _planner("icem", seed=7).plan(_linear_dynamics(), z0, goal)
    assert torch.equal(a.actions, b.actions) and a.cost == b.cost


def test_planner_beats_the_no_op_baseline() -> None:
    z0, goal = torch.ones(_D), torch.zeros(_D)
    dynamics = _linear_dynamics()
    result = _planner("cem", num_samples=256, num_iters=5).plan(dynamics, z0, goal)

    # cost of doing nothing (zero actions): the latent decays toward, but never reaches, the goal
    zero_actions = torch.zeros(1, 4, _D)
    no_op_cost = float(
        _planner("cem")._rollout_costs(dynamics, zero_actions, z0, goal, _D)[0]
    )
    assert (
        result.cost < no_op_cost
    )  # the planner actually drives the latent to the goal


def test_invalid_family_raises_evaluation_error() -> None:
    with pytest.raises(EvaluationError) as exc:
        _planner("greedy")
    assert exc.value.code == LensembleErrorCode.EVALUATION_FAILED


def test_diverging_dynamics_raises() -> None:
    def nan_dynamics(latents: Tensor, actions: Tensor) -> Tensor:
        return latents * float("nan")

    with pytest.raises(EvaluationError):
        _planner("cem").plan(nan_dynamics, torch.ones(_D), torch.zeros(_D))


def test_from_config_constructs() -> None:
    from types import SimpleNamespace

    eval_cfg = SimpleNamespace(planner="mppi", horizon=3, planning_samples=32)
    planner = Planner.from_config(eval_cfg, action_dim=_D, seed=1)
    assert (
        planner.family == "mppi" and planner.horizon == 3 and planner.num_samples == 32
    )
