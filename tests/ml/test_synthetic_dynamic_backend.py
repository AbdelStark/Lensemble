"""RFC-0017 synthetic-dynamic swipe-dot data source."""

from __future__ import annotations

import pytest
import torch

from lensemble.config import load_config
from lensemble.contracts import ActionKind
from lensemble.data import load_episodes
from lensemble.data.residency import guard_egress
from lensemble.errors import ResidencyViolation
from lensemble.model import build_encoder

_URI = "synthetic-dynamic://swipe-dot?seed=0&n_episodes=3&steps=6&image_size=48"


def test_synthetic_dynamic_scheme_resolves_read_only_dataset() -> None:
    ds = load_episodes(_URI)
    assert ds.fmt == "synthetic-dynamic"
    assert ds.path is None
    assert ds.exportable is False
    assert len(ds) == 3
    spec = ds.episodes[0].action_spec
    assert spec.embodiment_id == "swipe-dot-2dof"
    assert spec.kind == ActionKind.CONTINUOUS
    assert spec.dim == 2
    assert spec.low == (-1.0, -1.0)
    assert spec.high == (1.0, 1.0)


def test_synthetic_dynamic_is_byte_identical_for_same_seed() -> None:
    left = load_episodes(_URI)
    right = load_episodes(_URI)
    for ep_left, ep_right in zip(left.episodes, right.episodes, strict=True):
        for tr_left, tr_right in zip(
            ep_left.transitions, ep_right.transitions, strict=True
        ):
            assert torch.equal(tr_left.obs_t, tr_right.obs_t)
            assert torch.equal(tr_left.action_t, tr_right.action_t)
            assert torch.equal(tr_left.obs_tp1, tr_right.obs_tp1)
            assert tr_left.state_t is not None and tr_right.state_t is not None
            assert tr_left.state_tp1 is not None and tr_right.state_tp1 is not None
            assert torch.equal(tr_left.state_t, tr_right.state_t)
            assert torch.equal(tr_left.state_tp1, tr_right.state_tp1)


def test_synthetic_dynamic_does_not_touch_global_torch_rng() -> None:
    torch.manual_seed(123)
    before = torch.rand(4)
    load_episodes(_URI)
    after = torch.rand(4)

    torch.manual_seed(123)
    expected_before = torch.rand(4)
    expected_after = torch.rand(4)
    assert torch.equal(before, expected_before)
    assert torch.equal(after, expected_after)


def test_synthetic_dynamic_windows_include_true_state() -> None:
    ds = load_episodes(_URI)
    windows = list(ds.windows(num_steps=3))
    assert windows
    window = windows[0]
    assert tuple(window.obs.shape) == (4, 1, 3, 48, 48)
    assert tuple(window.actions.shape) == (3, 2)
    assert window.state is not None
    assert tuple(window.state.shape) == (4, 2)
    assert float(window.obs.min()) >= 0.0
    assert float(window.obs.max()) <= 1.0
    assert float(window.actions.min()) >= -1.0
    assert float(window.actions.max()) <= 1.0


def test_synthetic_dynamic_dataset_and_state_are_residency_bound() -> None:
    ds = load_episodes(_URI)
    with pytest.raises(ResidencyViolation):
        guard_egress(ds)
    state = next(ds.windows(num_steps=2)).state
    assert state is not None
    with pytest.raises(ResidencyViolation):
        guard_egress({"state": state})
    assert all(
        isinstance(value, str)
        for episode in ds.episodes
        for value in episode.collection_meta.values()
    )


def test_synthetic_dynamic_small_shape_builds_nine_token_encoder() -> None:
    cfg = load_config(
        overrides=[
            "model.encoder=scratch",
            "model.latent_dim=128",
            "model.num_tokens=9",
            "model.num_frames=1",
            "model.tubelet=1",
            "model.image_size=48",
            "model.patch_size=16",
            "model.depth=1",
            "model.num_heads=4",
            "model.predictor_depth=1",
            "model.predictor_width=128",
        ]
    )
    encoder = build_encoder(cfg)
    assert encoder.num_tokens == 9
    assert encoder(torch.zeros(1, 1, 3, 48, 48)).tokens.shape == (1, 9, 128)
