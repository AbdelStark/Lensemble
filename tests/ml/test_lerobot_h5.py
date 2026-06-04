"""LeRobot-layout single-file HDF5 data source (RFC-0004 §1; the #22 data-layer boundary).

Exercises ``lensemble.data.adapters.lerobot_h5_backend``: a flat ``episode_index`` + ``observation/
pixels_*`` + ``action`` HDF5 (the de-facto LeRobot / stable-worldmodel export) resolves to a read-only
``EpisodeDataset`` whose ``Window``s feed the official ``train_local`` / federated path. Placed in
tests/ml (the §8 CI gate scans tests/{unit,property,integration,ml,e2e,regression}) since the adapter
materializes raw model-bearing tensors, mirroring tests/ml/test_format_roundtrip.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from lensemble.data import load_episodes
from lensemble.errors import ContractViolation

_H = _W = 8
_ADIM = 3
# episode lengths (frames); each yields (len - 1) transitions
_EP_LENS = (5, 4)


def _write_lerobot_h5(
    path: Path, *, ep_lens=_EP_LENS, with_action: bool = True
) -> None:
    import h5py

    n = sum(ep_lens)
    ep_index = np.concatenate(
        [np.full(L, i, dtype=np.int32) for i, L in enumerate(ep_lens)]
    )
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(n, _H, _W, 3), dtype=np.uint8)
    actions = rng.standard_normal((n, _ADIM)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("episode_index", data=ep_index)
        f.create_dataset("observation/pixels_top", data=pixels)
        if with_action:
            f.create_dataset("action", data=actions)


def test_loads_episodes_and_windows(tmp_path: Path) -> None:
    h5 = tmp_path / "robot.h5"
    _write_lerobot_h5(h5)

    ds = load_episodes(f"lerobot-h5://{h5}")
    assert ds.fmt == "lerobot-h5"
    assert ds.exportable is False  # read-only view (INV-RESIDENCY)
    assert len(ds) == len(_EP_LENS)

    # a window over num_steps transitions: obs (num_steps+1, 1, 3, H, W), actions (num_steps, ADIM)
    num_steps = 2
    windows = list(ds.windows(num_steps))
    # per-episode contiguous windows: sum(max(0, (L-1) - num_steps + 1))
    expected = sum(max(0, (L - 1) - num_steps + 1) for L in _EP_LENS)
    assert len(windows) == expected
    w = windows[0]
    assert tuple(w.obs.shape) == (num_steps + 1, 1, 3, _H, _W)
    assert tuple(w.actions.shape) == (num_steps, _ADIM)
    assert w.obs.dtype == torch.float32
    assert float(w.obs.min()) >= 0.0 and float(w.obs.max()) <= 1.0  # decoded to [0,1]
    assert w.embodiment_id == f"lerobot-{_ADIM}dof"


def test_explicit_fmt_resolves_without_scheme(tmp_path: Path) -> None:
    h5 = tmp_path / "robot.h5"
    _write_lerobot_h5(h5)
    # a bare .h5 path would otherwise select the lensemble `hdf5` store; fmt= picks the LeRobot backend
    ds = load_episodes(h5, fmt="lerobot-h5")
    assert ds.fmt == "lerobot-h5" and len(ds) == len(_EP_LENS)


def test_missing_file_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ContractViolation):
        load_episodes(f"lerobot-h5://{tmp_path / 'nope.h5'}")


def test_missing_action_column_fails_closed(tmp_path: Path) -> None:
    h5 = tmp_path / "noaction.h5"
    _write_lerobot_h5(h5, with_action=False)
    with pytest.raises(ContractViolation):
        load_episodes(f"lerobot-h5://{h5}")


def test_action_spec_dim_inferred_and_conformant(tmp_path: Path) -> None:
    h5 = tmp_path / "robot.h5"
    _write_lerobot_h5(h5)
    ds = load_episodes(f"lerobot-h5://{h5}")
    spec = ds.episodes[0].action_spec
    assert spec.dim == _ADIM
    assert spec.embodiment_id == f"lerobot-{_ADIM}dof"
    # low/high bounds were inferred per-dim from the data (len == dim)
    assert spec.low is not None and len(spec.low) == _ADIM
    assert spec.high is not None and len(spec.high) == _ADIM
