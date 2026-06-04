"""lensemble.data.adapters.lerobot_h5_backend — read-only LeRobot-layout HDF5 source (RFC-0004 §1).

A single-file HDF5 in the de-facto **LeRobot episode layout** — a flat ``episode_index`` column, one or
more ``observation/pixels_*`` image stacks ``(N, H, W, 3)`` uint8, and an ``action`` column ``(N, dim)`` —
as produced by stable-worldmodel / LeRobot exports (e.g. ``abdelstark/so100-pickplace-lewm-ready``). This
is DISTINCT from the :mod:`hdf5_backend` store, which expects lensemble's own ``ep_len``/``ep_offset``
schema; this backend reads the LeRobot layout directly so a real robot dataset is a first-class
``cfg.data.data_source`` for the official ``train_local`` / federated path (no bespoke loader).

Resolution: ``lerobot-h5://<path>`` (or ``load_episodes(path, fmt="lerobot-h5")``). Read-only
(``exportable=False``; no saver registered). Every resolved episode is conformance-checked on load
(:func:`~lensemble.data.adapters.lerobot_adapter._validate_episode_conformance`). The materialized
observations/actions are RAW and local (``INV-RESIDENCY``): pixels are decoded to ``[0,1]`` float clips
``(1, 3, H, W)`` (single-frame ``rgb-video``) and actions kept as raw floats; the view never re-exports
them and registers no egress path.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import h5py
import numpy as np
import torch

from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
from lensemble.data.adapters.lerobot_adapter import _validate_episode_conformance
from lensemble.data.episode import Episode, Transition
from lensemble.errors import ContractViolation, LensembleErrorCode

if TYPE_CHECKING:
    from lensemble.data.dataset import EpisodeDataset

_SCHEME = "lerobot-h5://"
# Preferred camera stack; on absence the first `observation/pixels*` dataset is used.
_DEFAULT_PIXELS_KEY = "observation/pixels_top"


def _fail(message: str, remediation: str) -> ContractViolation:
    return ContractViolation(
        message,
        code=LensembleErrorCode.WMCP_CONTRACT_VIOLATION,
        remediation=remediation,
    )


def load_lerobot_h5(source: str | Path) -> "EpisodeDataset":
    """Resolve a LeRobot-layout HDF5 ``source`` to a read-only :class:`EpisodeDataset` (RFC-0004 §1).

    ``source`` is a path or a ``lerobot-h5://<path>`` URI. Raises :class:`~lensemble.errors.ContractViolation`
    when the file is missing or lacks the required ``episode_index`` / ``action`` / ``observation/pixels*``
    columns, and (via on-load conformance) when an episode's action space or modality is non-conformant.
    """
    text = str(source)
    if text.startswith(_SCHEME):
        text = text[len(_SCHEME) :]
    path = Path(text)
    if not path.exists():
        raise _fail(
            f"cannot resolve lerobot-h5 source {text!r}: file does not exist",
            "point cfg.data.data_source at an existing LeRobot-layout HDF5 file",
        )
    episodes = _read_episodes(path)
    for episode in episodes:
        _validate_episode_conformance(episode, episode.action_spec)

    from lensemble.data.dataset import EpisodeDataset

    return EpisodeDataset(episodes, path=path, fmt="lerobot-h5", exportable=False)


def _pixels_key(f: h5py.File) -> str:
    """The camera stack to read: the preferred top camera, else the first ``observation/pixels*`` dataset."""
    if _DEFAULT_PIXELS_KEY in f:
        return _DEFAULT_PIXELS_KEY
    obs = f.get("observation")
    if isinstance(obs, h5py.Group):
        for name in obs:
            if str(name).startswith("pixels"):
                return f"observation/{name}"
    raise _fail(
        f"no observation/pixels* image stack found in the HDF5 (top-level keys: {sorted(f.keys())})",
        "export the LeRobot record with an observation/pixels_<cam> uint8 (N,H,W,3) stack",
    )


def _read_episodes(path: Path) -> list[Episode]:
    """Materialize the LeRobot-layout HDF5 at ``path`` into RAW, conformance-ready ``Episode``s."""
    with h5py.File(str(path), "r") as f:
        for required in ("episode_index", "action"):
            if required not in f:
                raise _fail(
                    f"LeRobot-layout HDF5 missing required dataset {required!r} (have: {sorted(f.keys())})",
                    "a LeRobot-layout export needs flat episode_index + action columns",
                )
        pixels_key = _pixels_key(f)
        ep_index = np.asarray(cast(h5py.Dataset, f["episode_index"])[:])
        actions_np = np.asarray(cast(h5py.Dataset, f["action"])[:]).astype("float32")
        pixels_np = np.asarray(cast(h5py.Dataset, f[pixels_key])[:])  # (N,H,W,3) uint8

    if actions_np.ndim != 2:
        raise _fail(
            f"action column must be 2-D (N, dim); got shape {tuple(actions_np.shape)}",
            "store actions as a flat (N, action_dim) float column",
        )
    action_dim = int(actions_np.shape[-1])
    lo = tuple(float(x) for x in actions_np.min(axis=0))
    hi = tuple(float(x) for x in actions_np.max(axis=0))
    spec = ActionSpec(
        embodiment_id=f"lerobot-{action_dim}dof",
        kind=ActionKind.CONTINUOUS,
        dim=action_dim,
        low=lo,
        high=hi,
        num_classes=None,
        units=tuple(["unit"] * action_dim),
        wmcp_version=WMCP_VERSION,
    )

    # Pixels -> [0,1] float, channels-first, single-frame rgb-video clips (1, 3, H, W).
    frames = (
        torch.from_numpy(pixels_np).permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
    ).unsqueeze(1)  # (N, 1, 3, H, W)
    actions = torch.from_numpy(actions_np)  # (N, dim)

    bounds = np.flatnonzero(np.diff(ep_index)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(ep_index)]])

    episodes: list[Episode] = []
    for ei, (s, e) in enumerate(zip(starts.tolist(), ends.tolist())):
        if e - s < 2:  # need ≥ 2 frames to form ≥ 1 transition
            continue
        transitions = [
            Transition(obs_t=frames[i], action_t=actions[i], obs_tp1=frames[i + 1])
            for i in range(s, e - 1)
        ]
        episodes.append(
            Episode(
                episode_id=f"{path.stem}-ep{ei}",
                transitions=transitions,
                embodiment_id=spec.embodiment_id,
                modality="rgb-video",
                action_spec=spec,
                collection_meta={"source": "lerobot-h5", "pixels_key": pixels_key},
            )
        )
    if not episodes:
        raise _fail(
            f"no usable episodes (>= 2 frames) in {path}",
            "verify the episode_index column delimits multi-frame episodes",
        )
    return episodes
