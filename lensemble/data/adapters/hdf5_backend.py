"""lensemble.data.adapters.hdf5_backend — the portable single-file ``hdf5`` store (RFC-0004 §1).

``hdf5`` is the portable, archival format: one self-contained ``.h5`` file, easy to transfer between a
participant's own machines ([conventions §11](../../../docs/spec/conventions.md#11-external-dependencies),
pinned ``h5py >= 3.10``). The layout is one HDF5 *group* per episode, holding three stacked datasets —
``obs_t`` ``(N, *obs_shape)``, ``action_t`` ``(N, action_dim)``, ``obs_tp1`` ``(N, *obs_shape)`` — so a
transition ``i`` is row ``i`` of each. (We never assume ``obs_tp1[i] == obs_t[i+1]``: storing all three
stacks round-trips each :class:`~lensemble.data.episode.Transition` exactly, contiguous trajectory or
not.) Per-episode metadata — ``episode_id``, ``embodiment_id``, ``modality``, ``collection_meta``, and
every ``ActionSpec`` field — rides as group/dataset attrs. ``bfloat16`` (no native HDF5/numpy type) is
stored bit-cast to ``uint16`` with the true dtype label recorded in an attr, then restored on read.

Residency (``INV-RESIDENCY``): the ``.h5`` file is a LOCAL participant artifact written inside the trust
boundary. This backend has no egress / serialize-outbound path; a boundary-crossing payload is inspected
only by ``lensemble.data.residency.guard_egress``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import h5py
import numpy as np
import torch

from lensemble.data.adapters._serialize import (
    action_spec_from_meta,
    action_spec_to_meta,
    dtype_label,
)
from lensemble.data.episode import Episode, Transition

if TYPE_CHECKING:
    from lensemble.data.dataset import EpisodeDataset

_STR_TO_DTYPE: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def _stack_to_dataset(
    group: h5py.Group, name: str, tensors: list[torch.Tensor]
) -> None:
    """Write a length-N list of equal-shape tensors as one ``(N, *shape)`` dataset + a dtype-label attr."""
    label = dtype_label(tensors[0].dtype)
    cpu = [t.detach().cpu().contiguous() for t in tensors]
    if tensors[0].dtype is torch.bfloat16:
        arr = np.stack([t.view(torch.uint16).numpy() for t in cpu])
    else:
        arr = np.stack([t.numpy() for t in cpu])
    dset = group.create_dataset(name, data=arr)
    dset.attrs["dtype_label"] = label


def _dataset_to_tensors(group: h5py.Group, name: str) -> list[torch.Tensor]:
    """Read a ``(N, *shape)`` dataset back into N byte-identical tensors of the recorded dtype."""
    dset = cast(h5py.Dataset, group[name])
    label = cast(str, dset.attrs["dtype_label"])
    arr = np.asarray(dset)
    dtype = _STR_TO_DTYPE[label]
    out: list[torch.Tensor] = []
    for row in arr:
        if dtype is torch.bfloat16:
            out.append(
                torch.from_numpy(row.copy()).view(torch.uint16).view(torch.bfloat16)
            )
        else:
            out.append(torch.from_numpy(row.copy()))
    return out


def save_hdf5(dataset: "EpisodeDataset", path: Path) -> None:
    """Write every episode to one portable ``.h5`` file at ``path`` (one group per episode, RFC-0004 §1).

    The file is a LOCAL participant artifact (``INV-RESIDENCY``). Group keys are the enumeration index
    (so a non-filesystem-safe ``episode_id`` is never an HDF5 path); the true id rides as a group attr,
    and read order follows the index.
    """
    with h5py.File(str(path), "w") as f:
        for i, episode in enumerate(dataset.episodes):
            group = f.create_group(f"episode_{i:08d}")
            group.attrs["episode_id"] = episode.episode_id
            group.attrs["embodiment_id"] = episode.embodiment_id
            group.attrs["modality"] = episode.modality
            group.attrs["collection_meta"] = json.dumps(dict(episode.collection_meta))
            group.attrs["action_spec"] = json.dumps(
                action_spec_to_meta(episode.action_spec)
            )
            _stack_to_dataset(group, "obs_t", [t.obs_t for t in episode.transitions])
            _stack_to_dataset(
                group, "action_t", [t.action_t for t in episode.transitions]
            )
            _stack_to_dataset(
                group, "obs_tp1", [t.obs_tp1 for t in episode.transitions]
            )


def load_hdf5(source: "str | Path") -> "EpisodeDataset":
    """Read a ``.h5`` file back into an ``EpisodeDataset(..., fmt="hdf5")`` (RFC-0004 §1).

    Episodes are returned in group-index order; each transition is reconstructed byte-identically from
    the stacked datasets and the recorded dtype labels. The materialized episodes are RAW and local
    (``INV-RESIDENCY``).
    """
    from lensemble.data.dataset import EpisodeDataset

    episodes: list[Episode] = []
    with h5py.File(str(source), "r") as f:
        for key in sorted(f.keys()):
            group = cast(h5py.Group, f[key])
            spec = action_spec_from_meta(
                json.loads(cast(str, group.attrs["action_spec"]))
            )
            collection_meta = json.loads(cast(str, group.attrs["collection_meta"]))
            obs_t = _dataset_to_tensors(group, "obs_t")
            action_t = _dataset_to_tensors(group, "action_t")
            obs_tp1 = _dataset_to_tensors(group, "obs_tp1")
            transitions = [
                Transition(obs_t=o, action_t=a, obs_tp1=o1)
                for o, a, o1 in zip(obs_t, action_t, obs_tp1, strict=True)
            ]
            episodes.append(
                Episode(
                    episode_id=str(group.attrs["episode_id"]),
                    transitions=transitions,
                    embodiment_id=str(group.attrs["embodiment_id"]),
                    modality=str(group.attrs["modality"]),
                    action_spec=spec,
                    collection_meta=collection_meta,
                )
            )

    return EpisodeDataset(episodes, path=Path(source), fmt="hdf5")
