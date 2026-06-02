"""Episode data layer + windowed loader (RFC-0004 1 / 03-data-model 4-5). Issue #21."""

from __future__ import annotations

import torch

from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
from lensemble.data import Episode, EpisodeDataset, Transition, Window


def _spec(dim: int = 4) -> ActionSpec:
    return ActionSpec(
        embodiment_id="so101-arm-7dof",
        kind=ActionKind.CONTINUOUS,
        dim=dim,
        low=tuple(-1.0 for _ in range(dim)),
        high=tuple(1.0 for _ in range(dim)),
        num_classes=None,
        units=tuple("rad" for _ in range(dim)),
        wmcp_version=WMCP_VERSION,
    )


def _episode(
    n_transitions: int = 3, *, dim: int = 4, obs_shape: tuple[int, ...] = (3, 4, 4)
) -> Episode:
    spec = _spec(dim)
    transitions = [
        Transition(
            obs_t=torch.zeros(*obs_shape),
            action_t=torch.zeros(dim),
            obs_tp1=torch.zeros(*obs_shape),
        )
        for _ in range(n_transitions)
    ]
    return Episode(
        episode_id="ep-0",
        transitions=transitions,
        embodiment_id=spec.embodiment_id,
        modality="rgb-video",
        action_spec=spec,
        collection_meta={"site": "lab-a"},
    )


def test_windows_have_correct_shapes() -> None:
    ds = EpisodeDataset([_episode(n_transitions=3, dim=4)], exportable=False)
    windows = list(ds.windows(num_steps=2))
    # K - num_steps + 1 = 3 - 2 + 1 = 2 windows
    assert len(windows) == 2
    for w in windows:
        assert isinstance(w, Window)
        assert w.obs.shape[0] == w.num_steps + 1 == 3
        assert w.actions.shape[0] == w.num_steps == 2
        assert w.actions.shape[1] == 4
        assert w.embodiment_id == "so101-arm-7dof"


def test_episode_shorter_than_window_yields_nothing() -> None:
    ds = EpisodeDataset([_episode(n_transitions=1)])
    assert list(ds.windows(num_steps=2)) == []


def test_dataset_exposes_no_raw_serialization() -> None:
    ds = EpisodeDataset([_episode()], exportable=False)
    forbidden = ["export", "serialize", "to_bytes", "dump", "save", "write", "emit"]
    for name in forbidden:
        assert not hasattr(ds, name), (
            f"EpisodeDataset must not expose a raw-tensor egress method: {name}"
        )
    # residency flag is carried for the guard (#23) to read
    assert ds.exportable is False
