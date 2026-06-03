"""lensemble.data.adapters.lance_backend — the ``lance`` reference store (RFC-0004 §1).

``lance`` is the *default* (never canonical) on-disk format: append-friendly, columnar, with fast indexed
random reads of windows ([conventions §11](../../../docs/spec/conventions.md#11-external-dependencies),
pinned ``lance >= 0.10``). One row per :class:`~lensemble.data.episode.Transition`, in episode/transition
order, so a downstream window read is a contiguous indexed scan. Each transition's three tensors are
stored as raw little-endian bytes (a ``binary`` column) alongside a recorded dtype label + shape
(``_serialize``), so the read reshapes to byte-identical tensors (``torch.equal`` holds). Per-episode
metadata — ``episode_id``, ``embodiment_id``, ``modality``, ``collection_meta``, and every ``ActionSpec``
field — rides as string columns (denormalized per row; an episode is a contiguous run of one
``episode_id``).

Residency (``INV-RESIDENCY``): the ``.lance`` directory is a LOCAL participant artifact written inside
the trust boundary. This backend has no egress / serialize-outbound path; a boundary-crossing payload is
inspected only by ``lensemble.data.residency.guard_egress``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import lance
import pyarrow as pa

from lensemble.data.adapters._serialize import (
    action_spec_from_meta,
    action_spec_to_meta,
    tensor_from_bytes,
    tensor_to_bytes,
)
from lensemble.data.episode import Episode, Transition

if TYPE_CHECKING:
    from lensemble.data.dataset import EpisodeDataset

# The columnar schema: per-transition blobs + their dtype/shape, plus denormalized per-episode metadata.
_SCHEMA = pa.schema(
    [
        ("episode_id", pa.string()),
        ("transition_idx", pa.int64()),
        ("obs_t", pa.binary()),
        ("obs_t_dtype", pa.string()),
        ("obs_t_shape", pa.string()),  # JSON list, e.g. "[2, 2, 3, 3]"
        ("action_t", pa.binary()),
        ("action_t_dtype", pa.string()),
        ("action_t_shape", pa.string()),
        ("obs_tp1", pa.binary()),
        ("obs_tp1_dtype", pa.string()),
        ("obs_tp1_shape", pa.string()),
        ("embodiment_id", pa.string()),
        ("modality", pa.string()),
        ("collection_meta", pa.string()),  # JSON object
        ("action_spec", pa.string()),  # JSON object (the flat _serialize map)
    ]
)


def _episode_rows(episode: Episode) -> list[dict]:
    spec_json = json.dumps(action_spec_to_meta(episode.action_spec))
    meta_json = json.dumps(dict(episode.collection_meta))
    rows: list[dict] = []
    for idx, t in enumerate(episode.transitions):
        obs_t, obs_t_dt, obs_t_sh = tensor_to_bytes(t.obs_t)
        act, act_dt, act_sh = tensor_to_bytes(t.action_t)
        obs_tp1, obs_tp1_dt, obs_tp1_sh = tensor_to_bytes(t.obs_tp1)
        rows.append(
            {
                "episode_id": episode.episode_id,
                "transition_idx": idx,
                "obs_t": obs_t,
                "obs_t_dtype": obs_t_dt,
                "obs_t_shape": json.dumps(list(obs_t_sh)),
                "action_t": act,
                "action_t_dtype": act_dt,
                "action_t_shape": json.dumps(list(act_sh)),
                "obs_tp1": obs_tp1,
                "obs_tp1_dtype": obs_tp1_dt,
                "obs_tp1_shape": json.dumps(list(obs_tp1_sh)),
                "embodiment_id": episode.embodiment_id,
                "modality": episode.modality,
                "collection_meta": meta_json,
                "action_spec": spec_json,
            }
        )
    return rows


def save_lance(dataset: "EpisodeDataset", path: Path) -> None:
    """Write every episode's transitions to a ``.lance`` dataset at ``path`` (RFC-0004 §1).

    Rows are emitted in episode-then-transition order; a single ``write_dataset`` call lands them in one
    indexed fragment. The file is a LOCAL participant artifact (``INV-RESIDENCY``).
    """
    rows: list[dict] = []
    for episode in dataset.episodes:
        rows.extend(_episode_rows(episode))
    table = pa.Table.from_pylist(rows, schema=_SCHEMA)
    lance.write_dataset(table, str(path), mode="overwrite")


def load_lance(source: "str | Path") -> "EpisodeDataset":
    """Read a ``.lance`` dataset back into an ``EpisodeDataset(..., fmt="lance")`` (RFC-0004 §1).

    Reconstructs byte-identical tensors from the recorded dtype/shape and groups rows by ``episode_id``
    (preserving first-seen episode order and ``transition_idx`` order). The materialized episodes are
    RAW and local (``INV-RESIDENCY``).
    """
    from lensemble.data.dataset import EpisodeDataset

    ds = lance.dataset(str(source))
    table = ds.to_table()
    rows = table.to_pylist()
    rows.sort(key=lambda r: (r["episode_id"], r["transition_idx"]))

    grouped: dict[str, list[dict]] = {}
    order: list[str] = []
    for r in rows:
        eid = r["episode_id"]
        if eid not in grouped:
            grouped[eid] = []
            order.append(eid)
        grouped[eid].append(r)

    episodes: list[Episode] = []
    for eid in order:
        eps_rows = grouped[eid]
        head = eps_rows[0]
        spec = action_spec_from_meta(json.loads(head["action_spec"]))
        collection_meta = json.loads(head["collection_meta"])
        transitions = [
            Transition(
                obs_t=tensor_from_bytes(
                    r["obs_t"], r["obs_t_dtype"], tuple(json.loads(r["obs_t_shape"]))
                ),
                action_t=tensor_from_bytes(
                    r["action_t"],
                    r["action_t_dtype"],
                    tuple(json.loads(r["action_t_shape"])),
                ),
                obs_tp1=tensor_from_bytes(
                    r["obs_tp1"],
                    r["obs_tp1_dtype"],
                    tuple(json.loads(r["obs_tp1_shape"])),
                ),
            )
            for r in eps_rows
        ]
        episodes.append(
            Episode(
                episode_id=eid,
                transitions=transitions,
                embodiment_id=head["embodiment_id"],
                modality=head["modality"],
                action_spec=spec,
                collection_meta=collection_meta,
            )
        )

    return EpisodeDataset(episodes, path=Path(source), fmt="lance")
