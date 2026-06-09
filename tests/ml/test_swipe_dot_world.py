"""RFC-0017 swipe-dot eval world."""

from __future__ import annotations

import pytest
import torch

from lensemble.config import load_config
from lensemble.data.residency import guard_egress
from lensemble.errors import ResidencyViolation
from lensemble.eval import KINEMATIC_SWIPE_DOT_ENV_ID, resolve_env
from lensemble.eval.world import SwipeDotWorld


def _cfg():
    cfg = load_config(
        overrides=[
            "model.encoder=scratch",
            "model.latent_dim=128",
            "model.num_tokens=9",
            "model.num_frames=1",
            "model.tubelet=1",
            "model.image_size=48",
            "model.patch_size=16",
            "model.depth=4",
            "model.num_heads=4",
            "model.predictor_depth=1",
            "model.predictor_width=128",
            "eval.env_id=kinematic://swipe-dot",
            "eval.planning_samples=4",
            "eval.horizon=2",
        ]
    )
    return cfg


def _world() -> SwipeDotWorld:
    world = resolve_env(KINEMATIC_SWIPE_DOT_ENV_ID, cfg=_cfg())
    assert isinstance(world, SwipeDotWorld)
    return world


def test_swipe_dot_world_resolves_and_exposes_true_state() -> None:
    world = _world()
    clip = world.reset(0)
    state = world.state()
    assert tuple(clip.shape) == (1, 3, 48, 48)
    assert tuple(state.shape) == (2,)
    assert torch.all((state >= 0.0) & (state <= 1.0))
    assert world.succeeded() is False


def test_swipe_dot_world_step_is_action_sensitive_and_clamped() -> None:
    world = _world()
    world.reset(123)
    before = world.state()
    world.step(torch.tensor([1.0, -1.0]))
    after = world.state()
    assert torch.allclose(
        after, (before + 0.12 * torch.tensor([1.0, -1.0])).clamp(0.0, 1.0)
    )
    world.step(torch.tensor([100.0, 100.0]))
    assert torch.all(world.state() <= 1.0)


def test_swipe_dot_world_success_uses_true_position_not_seed_parity() -> None:
    world = _world()
    world.reset(1)
    goal = torch.tensor([0.82, 0.82])
    for _ in range(8):
        action = ((goal - world.state()) / 0.12).clamp(-1.0, 1.0)
        world.step(action)
    assert world.succeeded() is True


def test_swipe_dot_world_state_tensor_is_residency_bound() -> None:
    world = _world()
    world.reset(0)
    with pytest.raises(ResidencyViolation):
        guard_egress({"state": world.state()})
