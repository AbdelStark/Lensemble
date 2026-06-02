"""Window shape validation (03-data-model 5). Issue #21."""

from __future__ import annotations

import pytest
import torch

from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
from lensemble.data import Episode, EpisodeDataset, Transition, Window
from lensemble.data.dataset import _check_window
from lensemble.errors import ContractViolation, LensembleErrorCode


def _win(*, obs_len: int, n_actions: int, action_dim: int, num_steps: int) -> Window:
    return Window(
        obs=torch.zeros(obs_len, 3, 4, 4),
        actions=torch.zeros(n_actions, action_dim),
        num_steps=num_steps,
        embodiment_id="emb",
    )


def test_valid_window_passes() -> None:
    _check_window(_win(obs_len=3, n_actions=2, action_dim=4, num_steps=2), action_dim=4)


def test_obs_length_mismatch_rejected() -> None:
    with pytest.raises(ContractViolation) as exc:
        _check_window(
            _win(obs_len=2, n_actions=2, action_dim=4, num_steps=2), action_dim=4
        )
    assert exc.value.code == LensembleErrorCode.WMCP_CONTRACT_VIOLATION


def test_actions_length_mismatch_rejected() -> None:
    with pytest.raises(ContractViolation):
        _check_window(
            _win(obs_len=3, n_actions=1, action_dim=4, num_steps=2), action_dim=4
        )


def test_action_dim_mismatch_rejected() -> None:
    with pytest.raises(ContractViolation) as exc:
        _check_window(
            _win(obs_len=3, n_actions=2, action_dim=3, num_steps=2), action_dim=4
        )
    assert "action_spec.dim" in exc.value.remediation or "4" in str(exc.value)


def test_loader_rejects_action_dim_mismatch() -> None:
    # episode whose transitions carry action_t of the wrong dimensionality vs its ActionSpec
    spec = ActionSpec(
        embodiment_id="emb",
        kind=ActionKind.CONTINUOUS,
        dim=4,
        low=(-1.0, -1.0, -1.0, -1.0),
        high=(1.0, 1.0, 1.0, 1.0),
        num_classes=None,
        units=("rad", "rad", "rad", "rad"),
        wmcp_version=WMCP_VERSION,
    )
    transitions = [
        Transition(
            obs_t=torch.zeros(3, 4, 4),
            action_t=torch.zeros(3),
            obs_tp1=torch.zeros(3, 4, 4),
        )
        for _ in range(3)
    ]
    ds = EpisodeDataset(
        [
            Episode(
                episode_id="ep",
                transitions=transitions,
                embodiment_id="emb",
                modality="rgb-video",
                action_spec=spec,
                collection_meta={},
            )
        ]
    )
    with pytest.raises(ContractViolation):
        list(ds.windows(num_steps=2))
