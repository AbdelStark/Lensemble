"""Latent-space inference demonstration on held-out windows (RFC-0005, #265).

Exercises both falsifiable "the world model is usable" signals on a tiny CPU model: the multi-step
open-loop prediction report (model vs predict-current vs predict-random) and the latent-MPC goal-reaching
loop (the planner reduces the goal-energy below the zero-action baseline). Mirrors the tiny V-JEPA shape
used across tests/ml.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
from lensemble.data.episode import Window
from lensemble.eval.inference_demo import (
    latent_mpc_goal_reaching,
    multistep_prediction_report,
)
from lensemble.model import build_action_head, build_encoder, build_predictor

_D = 8
_T, _C, _H, _W = 2, 3, 4, 4
_ACTION_DIM = 2
_STEPS = 4


@dataclass(frozen=True)
class _ModelConfig:
    encoder: str = "vjepa2-vit-l"
    warm_start_release: str = "vjepa2-2.0"
    latent_dim: int = _D
    num_tokens: int = 4  # (2//2)*(4//2)**2
    predictor_depth: int = 1
    predictor_width: int = _D
    wmcp_version: str = WMCP_VERSION
    encoder_frozen: bool = False
    d: int = _D
    in_channels: int = _C
    num_frames: int = _T
    image_size: int = _H
    patch_size: int = 2
    tubelet: int = 2
    depth: int = 1
    num_heads: int = 2
    cond_dim: int = _D


@dataclass(frozen=True)
class _Cfg:
    model: _ModelConfig = _ModelConfig()


def _spec() -> ActionSpec:
    return ActionSpec(
        embodiment_id="toy",
        kind=ActionKind.CONTINUOUS,
        dim=_ACTION_DIM,
        low=(-1.0, -1.0),
        high=(1.0, 1.0),
        num_classes=None,
        units=("u", "u"),
        wmcp_version=WMCP_VERSION,
    )


def _windows(n: int = 6, seed: int = 0) -> list[Window]:
    gen = torch.Generator().manual_seed(seed)
    return [
        Window(
            obs=torch.randn(_STEPS + 1, _T, _C, _H, _W, generator=gen),
            actions=torch.randn(_STEPS, _ACTION_DIM, generator=gen),
            num_steps=_STEPS,
            embodiment_id="toy",
        )
        for _ in range(n)
    ]


def _models() -> tuple[object, object, object]:
    cfg = _Cfg()
    torch.manual_seed(0)
    encoder = build_encoder(cfg).eval()
    predictor = build_predictor(cfg).eval()
    action_head = build_action_head(cfg, _spec()).eval()
    return encoder, predictor, action_head


def test_multistep_prediction_report_runs_and_reports_finite_metrics() -> None:
    encoder, predictor, action_head = _models()
    report = multistep_prediction_report(
        encoder=encoder,
        predictor=predictor,
        action_head=action_head,
        windows=_windows(),
        horizon=3,
        max_windows=4,
    )
    assert report["windows_used"] == 4
    assert report["horizon"] == 3
    for key in ("val_pred_model", "val_pred_identity", "val_pred_random"):
        assert torch.isfinite(torch.tensor(report[key]))
        assert report[key] >= 0.0
    assert report["effective_rank"] > 0.0
    assert torch.isfinite(torch.tensor(report["skill_vs_identity"]))


def test_latent_mpc_goal_reaching_runs_and_reports_in_range() -> None:
    encoder, predictor, action_head = _models()
    out = latent_mpc_goal_reaching(
        encoder=encoder,
        predictor=predictor,
        action_head=action_head,
        windows=_windows(),
        horizon=3,
        planning_samples=32,
        planner_iters=2,
        max_episodes=4,
    )
    assert out["episodes"] == 4
    assert 0.0 <= out["success_rate"] <= 1.0
    assert out["planner"] == "icem"
    assert torch.isfinite(torch.tensor(out["mean_cost_reduction_vs_zero_action"]))
    assert out["effective_dim"] > 0.0


def test_multistep_prediction_raises_when_no_window_long_enough() -> None:
    encoder, predictor, action_head = _models()
    import pytest

    with pytest.raises(ValueError, match="enough steps"):
        multistep_prediction_report(
            encoder=encoder,
            predictor=predictor,
            action_head=action_head,
            windows=_windows(),
            horizon=_STEPS + 5,  # longer than any window
        )
