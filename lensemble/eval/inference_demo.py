"""lensemble.eval.inference_demo — latent-space inference demonstration on held-out data (RFC-0005, #265).

Two falsifiable "the converged world model is *usable*" signals computed from a committed checkpoint plus
real held-out windows — with **no simulator** (closed-loop physical task-success stays gated on the
unvendored ``stable-worldmodel`` simulator, #96; this is latent-space goal-reaching / prediction only):

1. :func:`multistep_prediction_report` — open-loop multi-step latent prediction quality vs trivial
   baselines. Rolling the action-conditioned predictor ``g_phi`` forward ``h`` steps from the encoded start
   and comparing to the true future latents, against the *predict-current* (identity) and *predict-random*
   baselines. A model that learned dynamics beats predict-current (``skill_vs_identity < 1``); a collapsed
   model that emits a near-constant latent does not.

2. :func:`latent_mpc_goal_reaching` — the predictor *as a controller*. A CEM/iCEM planner
   (:class:`~lensemble.eval.mpc.Planner`) searches action sequences that reduce the L1 latent goal-energy
   against the true ``h``-step-ahead goal latent. Success = the plan reduces the goal-energy materially
   below the *zero-action* baseline. A collapsed model's predictor is action-insensitive, so the planner
   cannot beat doing nothing; a converged model's predictor responds to actions, so it can — the headline
   "it works where the collapsed control does not" contrast.

Both consume only a frozen ``encoder``/``predictor`` (+ a fresh local action head) and held-out
:class:`~lensemble.data.episode.Window` s; nothing is mutated and no raw tensor is returned (only scalars).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch

from lensemble.contracts import LatentState
from lensemble.eval.jepa_metrics import effective_rank
from lensemble.eval.metrics import effective_dim, success_rate
from lensemble.eval.mpc import Planner

if TYPE_CHECKING:
    from torch import Tensor

    from lensemble.data.episode import Window
    from lensemble.model.action_head import ActionHead
    from lensemble.model.encoder import Encoder
    from lensemble.model.predictor import Predictor


def _device_of(encoder: "Encoder") -> torch.device:
    return next(encoder.parameters()).device


def _latent_dynamics(
    encoder: "Encoder", predictor: "Predictor", action_head: "ActionHead"
) -> Any:
    """A batched latent dynamics ``(flat_latents (N,n*d), actions (N,A)) -> flat_next`` (mirrors the harness).

    Reshapes the flat latent to the ``(N, n_tokens, d)`` :class:`LatentState`, conditions the predictor on
    the encoded actions, and re-flattens — the exact dynamics the latent-MPC planner rolls out.
    """
    n_tokens, d = encoder.num_tokens, encoder.d

    def dynamics(latents: "Tensor", actions: "Tensor") -> "Tensor":
        tokens = latents.reshape(latents.shape[0], n_tokens, d)
        state = LatentState(
            tokens=tokens,
            num_tokens=n_tokens,
            dim=d,
            wmcp_version=encoder.wmcp_version,
        )
        cond = action_head.encode(actions)
        nxt = predictor.forward(state, cond)
        return nxt.tokens.reshape(latents.shape[0], n_tokens * d)

    return dynamics


def _window_to_device(window: "Window", device: torch.device) -> "Window":
    return replace(window, obs=window.obs.to(device), actions=window.actions.to(device))


def multistep_prediction_report(
    *,
    encoder: "Encoder",
    predictor: "Predictor",
    action_head: "ActionHead",
    windows: Sequence["Window"],
    horizon: int,
    max_windows: int = 64,
) -> dict[str, Any]:
    """Multi-step open-loop latent prediction quality vs predict-current / predict-random (RFC-0005, #265).

    For each held-out window (obs ``(h+1, …)``, actions ``(h, A)``) rolls ``g_phi`` forward ``h`` steps from
    the encoded start under the TRUE actions and accumulates the per-step latent MSE against the true future
    latents. Compares against the *predict-current* (the start latent held constant) and *predict-random*
    baselines. Returns the mean model / identity / random val_pred, ``skill_vs_identity`` (model/identity; <1
    means the model beat predict-current), and the held-out frame's ``effective_rank``.
    """
    device = _device_of(encoder)
    encoder.eval()
    predictor.eval()
    action_head.eval()
    used = 0
    sum_model = 0.0
    sum_identity = 0.0
    sum_random = 0.0
    frames: list[Tensor] = []
    gen = torch.Generator(device="cpu").manual_seed(2026)
    with torch.no_grad():
        for raw in windows:
            if used >= max_windows:
                break
            if int(raw.num_steps) < horizon or raw.obs.shape[0] <= horizon:
                continue
            window = _window_to_device(raw, device)
            true_tokens = encoder(window.obs[: horizon + 1]).tokens.to(
                torch.float32
            )  # (h+1, N, d)
            frames.append(true_tokens.reshape(-1, true_tokens.shape[-1]).cpu())
            z = true_tokens[0:1]  # (1, N, d)
            model_err = 0.0
            identity_err = 0.0
            random_err = 0.0
            for t in range(horizon):
                cond = action_head.encode(window.actions[t : t + 1])
                state = LatentState(
                    tokens=z,
                    num_tokens=encoder.num_tokens,
                    dim=encoder.d,
                    wmcp_version=encoder.wmcp_version,
                )
                z = predictor.forward(state, cond).tokens.to(torch.float32)
                target = true_tokens[t + 1 : t + 2]
                model_err += float((z - target).pow(2).mean())
                identity_err += float((true_tokens[0:1] - target).pow(2).mean())
                rand = torch.randn(target.shape, generator=gen, dtype=torch.float32).to(
                    device
                )
                random_err += float((rand - target).pow(2).mean())
            sum_model += model_err / horizon
            sum_identity += identity_err / horizon
            sum_random += random_err / horizon
            used += 1
    if used == 0:
        raise ValueError(
            "no held-out window had enough steps for the requested multi-step horizon"
        )
    val_pred_model = sum_model / used
    val_pred_identity = sum_identity / used
    val_pred_random = sum_random / used
    eff_rank = float(effective_rank(torch.cat(frames, dim=0))) if frames else 0.0
    return {
        "windows_used": used,
        "horizon": horizon,
        "val_pred_model": val_pred_model,
        "val_pred_identity": val_pred_identity,
        "val_pred_random": val_pred_random,
        # < 1.0 ⟺ the model's multi-step rollout beats holding the start latent constant (learned dynamics).
        "skill_vs_identity": (
            val_pred_model / val_pred_identity
            if val_pred_identity > 0
            else float("inf")
        ),
        "effective_rank": eff_rank,
    }


def latent_mpc_goal_reaching(
    *,
    encoder: "Encoder",
    predictor: "Predictor",
    action_head: "ActionHead",
    windows: Sequence["Window"],
    horizon: int,
    planner: str = "icem",
    planning_samples: int = 256,
    planner_iters: int = 4,
    max_episodes: int = 24,
    success_margin: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Latent-MPC goal-reaching: can the planner use ``g_phi`` to drive toward a held-out goal? (RFC-0005, #265).

    For each held-out episode, encodes the start ``z0`` and the true ``h``-step-ahead frame as the goal
    ``zg``, then runs the CEM/iCEM planner over the predictor dynamics. Success = the best plan's L1 latent
    goal-energy is at least ``success_margin`` below the *zero-action* rollout cost — i.e. selecting actions
    measurably beats doing nothing. A collapsed (action-insensitive) predictor cannot improve on the
    zero-action baseline, so its ``success_rate`` → 0; a converged predictor's responds, so it can.

    Returns ``success_rate``, the mean fractional cost reduction vs zero-action, the mean planning cost +
    wall-time, and the ``effective_dim`` of the start latents (the collapse guard).
    """
    device = _device_of(encoder)
    encoder.eval()
    predictor.eval()
    action_head.eval()
    dynamics = _latent_dynamics(encoder, predictor, action_head)
    n_tokens, d = encoder.num_tokens, encoder.d
    flat = n_tokens * d

    successes: list[bool] = []
    reductions: list[float] = []
    costs: list[float] = []
    wall: list[float] = []
    start_latents: list[Tensor] = []
    used = 0
    with torch.no_grad():
        for raw in windows:
            if used >= max_episodes:
                break
            if int(raw.num_steps) < horizon or raw.obs.shape[0] <= horizon:
                continue
            window = _window_to_device(raw, device)
            action_dim = int(window.actions.shape[-1])
            tokens = encoder(window.obs[: horizon + 1]).tokens.to(torch.float32)
            z0 = tokens[0:1].reshape(1, flat)
            zg = tokens[horizon : horizon + 1].reshape(1, flat)
            start_latents.append(z0.reshape(-1).cpu())

            # The zero-action baseline goal-energy (the predictor's free run toward the goal).
            zero_actions = torch.zeros(1, horizon, action_dim, device=device)
            latent = z0.clone()
            zero_cost = 0.0
            for t in range(horizon):
                latent = dynamics(latent, zero_actions[:, t, :])
                zero_cost += float((latent - zg).abs().sum())
            if zero_cost <= 0.0:
                used += 1
                successes.append(False)
                reductions.append(0.0)
                continue

            plan = Planner(
                family=planner,
                horizon=horizon,
                num_samples=planning_samples,
                action_dim=action_dim,
                seed=seed + used,
                num_iters=planner_iters,
            ).plan(dynamics, z0, zg)
            reduction = (zero_cost - plan.cost) / zero_cost
            successes.append(reduction >= success_margin)
            reductions.append(reduction)
            costs.append(plan.cost)
            wall.append(plan.wall_time_s)
            used += 1
    if used == 0:
        raise ValueError(
            "no held-out window had enough steps for the requested planning horizon"
        )
    eff_dim = (
        float(effective_dim(torch.stack(start_latents)))
        if len(start_latents) >= 2
        else 0.0
    )
    return {
        "episodes": used,
        "horizon": horizon,
        "planner": planner,
        "planning_samples": planning_samples,
        "success_rate": float(success_rate(successes)),
        "mean_cost_reduction_vs_zero_action": (
            sum(reductions) / len(reductions) if reductions else 0.0
        ),
        "mean_planning_cost": sum(costs) / len(costs) if costs else 0.0,
        "mean_planning_wall_s": sum(wall) / len(wall) if wall else 0.0,
        "effective_dim": eff_dim,
    }
