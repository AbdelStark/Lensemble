"""lensemble.eval.mpc — sampling-based latent model-predictive control (RFC-0005 3).

:class:`Planner` turns the trained world model into a controller: it searches action sequences that
minimize an **L1 goal-energy** in latent space — the accumulated ``sum_t ||z_t - z_goal||_1`` of the
predicted future latents (rolled out through the action-conditioned predictor ``g_phi``) against the
goal-image latent (both produced by the frozen encoder ``f_theta``). The planner family is a config
choice — ``cem`` / ``icem`` / ``mppi`` (``cfg.eval.planner``), ``icem`` the default.

The search is decoupled from the concrete model: :meth:`Planner.plan` takes a batched ``dynamics``
callable ``(latents, actions) -> next_latents`` so it is unit-testable with a stub and the eval harness
wires ``g_phi`` (+ the action head) into it. Determinism is best-effort and seed-pinned (conventions 9):
all sampling draws from one seeded generator, so a plan is reproducible on the same device class. A
diverging / non-finite rollout raises :class:`~lensemble.errors.EvaluationError`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from lensemble.errors import EvaluationError, LensembleErrorCode

_FAMILIES = ("cem", "icem", "mppi")
Dynamics = Callable[
    [Tensor, Tensor], Tensor
]  # (latents (N,d), actions (N,A)) -> next (N,d)


@dataclass(frozen=True)
class PlanResult:
    """A selected action sequence plus the planning-cost telemetry ``evaluate`` records (RFC-0005 3)."""

    actions: Tensor  # (horizon, action_dim) — the best action sequence found
    cost: float  # the L1 latent goal-energy of the selected plan
    planner: str  # the planner family used
    num_samples: int  # samples drawn per refinement iteration
    num_iters: int  # refinement iterations
    wall_time_s: float  # wall-clock planning time (telemetry; not part of the deterministic plan)


class Planner:
    """Latent MPC planner over a batched latent ``dynamics`` (CEM / iCEM / MPPI), RFC-0005 3."""

    def __init__(
        self,
        *,
        family: str,
        horizon: int,
        num_samples: int,
        action_dim: int,
        seed: int = 0,
        num_iters: int = 4,
        elite_frac: float = 0.1,
        init_std: float = 1.0,
        temperature: float = 1.0,
    ) -> None:
        if family not in _FAMILIES:
            raise EvaluationError(
                f"unknown planner family {family!r}; expected one of {_FAMILIES}",
                code=LensembleErrorCode.EVALUATION_FAILED,
                remediation=f"set cfg.eval.planner to one of {_FAMILIES}",
            )
        if min(horizon, num_samples, action_dim, num_iters) <= 0:
            raise EvaluationError(
                f"planner dims must be positive: horizon={horizon} num_samples={num_samples} "
                f"action_dim={action_dim} num_iters={num_iters}",
                code=LensembleErrorCode.EVALUATION_FAILED,
                remediation="set positive horizon / sample count / action_dim / iterations",
            )
        self.family = family
        self.horizon = horizon
        self.num_samples = num_samples
        self.action_dim = action_dim
        self.seed = seed
        self.num_iters = num_iters
        self.elite_frac = elite_frac
        self.init_std = init_std
        self.temperature = temperature

    @classmethod
    def from_config(
        cls, eval_cfg: object, *, action_dim: int, seed: int = 0
    ) -> Planner:
        """Construct from ``cfg.eval`` (``planner``, ``horizon``, ``planning_samples``)."""
        return cls(
            family=str(eval_cfg.planner),  # type: ignore[attr-defined]
            horizon=int(eval_cfg.horizon),  # type: ignore[attr-defined]
            num_samples=int(eval_cfg.planning_samples),  # type: ignore[attr-defined]
            action_dim=action_dim,
            seed=seed,
        )

    def plan(
        self, dynamics: Dynamics, initial_latent: Tensor, goal_latent: Tensor
    ) -> PlanResult:
        """Search for the action sequence minimizing the L1 latent goal-energy (RFC-0005 3).

        ``dynamics`` rolls a batch of latents one step under a batch of actions; ``initial_latent`` and
        ``goal_latent`` are ``(d,)`` latents from the frozen encoder. Returns the best sequence found and
        the planning-cost telemetry. Raises :class:`~lensemble.errors.EvaluationError` if the rollout
        diverges (a non-finite cost).
        """
        device = initial_latent.device
        gen = torch.Generator(device=device).manual_seed(self.seed)
        d = initial_latent.shape[-1]
        n, h, a = self.num_samples, self.horizon, self.action_dim
        elites_k = max(1, int(self.elite_frac * n))

        mean = torch.zeros(h, a, device=device)
        std = self.init_std * torch.ones(h, a, device=device)
        best_actions = mean.clone()
        best_cost = float("inf")
        prev_elites: Tensor | None = None

        start = time.perf_counter()
        for _ in range(self.num_iters):
            samples = mean + std * torch.randn(n, h, a, generator=gen, device=device)
            if self.family == "icem" and prev_elites is not None:
                # iCEM: carry the previous iteration's elites into the population (elite memory).
                keep = max(0, n - prev_elites.shape[0])
                samples = torch.cat([samples[:keep], prev_elites], dim=0)
            costs = self._rollout_costs(
                dynamics, samples, initial_latent, goal_latent, d
            )

            iter_best = int(torch.argmin(costs))
            if float(costs[iter_best]) < best_cost:
                best_cost = float(costs[iter_best])
                best_actions = samples[iter_best].clone()

            if self.family == "mppi":
                weights = torch.softmax(-costs / self.temperature, dim=0)
                mean = (weights.view(-1, 1, 1) * samples).sum(dim=0)
            else:  # cem / icem: refit to the elite set
                elite_idx = torch.topk(costs, elites_k, largest=False).indices
                elites = samples[elite_idx]
                mean = elites.mean(dim=0)
                std = elites.std(dim=0, unbiased=False).clamp_min(1e-6)
                prev_elites = elites

        if not torch.isfinite(torch.tensor(best_cost, device=device)):
            raise EvaluationError(
                "latent MPC rollout diverged (non-finite goal-energy)",
                code=LensembleErrorCode.EVALUATION_FAILED,
                remediation="check the dynamics model and the planner std/temperature",
            )
        return PlanResult(
            actions=best_actions,
            cost=best_cost,
            planner=self.family,
            num_samples=n,
            num_iters=self.num_iters,
            wall_time_s=time.perf_counter() - start,
        )

    def _rollout_costs(
        self,
        dynamics: Dynamics,
        samples: Tensor,
        initial_latent: Tensor,
        goal_latent: Tensor,
        d: int,
    ) -> Tensor:
        """Accumulated L1 latent goal-energy ``sum_t ||z_t - z_goal||_1`` for each sampled sequence."""
        n, h, _ = samples.shape
        latent = initial_latent.reshape(1, d).expand(n, d).contiguous()
        goal = goal_latent.reshape(1, d)
        cost = torch.zeros(n, device=samples.device)
        for t in range(h):
            latent = dynamics(latent, samples[:, t, :])
            cost = cost + (latent - goal).abs().sum(dim=-1)
        return cost
