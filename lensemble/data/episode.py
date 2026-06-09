"""lensemble.data.episode — the per-participant data-layer types (docs/rfcs/RFC-0004 1).

``Transition`` is the atomic learning tuple ``(o_t, a_t, o_{t+1})``; an ``Episode`` is an ordered
trajectory of transitions plus declared metadata; a ``Window`` is the fixed-length slice the loader
yields for next-embedding prediction. The canonical schemas are 03-data-model 4/5.

All observation/action tensors here are RAW, private, and **residency-bound** (``INV-RESIDENCY``): they
never cross a trust boundary and no embedding derived from them is serialized outbound. The egress guard
that enforces this lives in ``lensemble.data.residency`` (#23); these types only carry the data and the
``exportable`` flag the guard reads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only; the loader (dataset.py) imports torch at runtime
    from torch import Tensor

    from lensemble.contracts import ActionSpec


@dataclass(frozen=True)
class Transition:
    """A single ``(o_t, a_t, o_{t+1})`` step. Raw, private, residency-bound (``INV-RESIDENCY``).

    ``state_t`` / ``state_tp1`` are optional true environment state labels, used only by local
    ground-truth probes such as RFC-0017's swipe-dot ``(x,y)`` metric. They are resident data like
    observations and actions; only aggregate scalar metrics derived from them may cross a boundary.
    """

    obs_t: "Tensor"  # observation at t; modality-shaped (e.g. video clip C,T,H,W)
    action_t: "Tensor"  # action applied at t; shape (action_dim,) per the ActionSpec
    obs_tp1: "Tensor"  # observation at t+1; same modality-shape as obs_t
    state_t: "Tensor | None" = None  # optional true env state at t; resident label
    state_tp1: "Tensor | None" = None  # optional true env state at t+1; resident label


@dataclass(frozen=True)
class Episode:
    """An ordered trajectory of transitions plus declared metadata. Residency-bound (``INV-RESIDENCY``)."""

    episode_id: str  # participant-local stable id (a Merkle-leaf preimage component)
    transitions: "Sequence[Transition]"
    embodiment_id: str  # must match an ActionSpec in scope
    modality: str  # e.g. "rgb-video"
    action_spec: "ActionSpec"  # the embodiment action contract for this episode
    collection_meta: (
        "Mapping[str, str]"  # declared, non-private collection conditions (RFC-0004 7)
    )


@dataclass(frozen=True)
class Window:
    """A fixed-length slice the loader yields for next-latent prediction. Residency-bound (``INV-RESIDENCY``).

    ``obs`` is ``o_t … o_{t+num_steps}`` (length ``num_steps + 1``); ``actions`` is
    ``a_t … a_{t+num_steps-1}`` (length ``num_steps``).
    """

    obs: "Tensor"  # (num_steps + 1, *modality_shape)
    actions: "Tensor"  # (num_steps, action_dim)
    num_steps: int  # fixed horizon; equals config data.num_steps
    embodiment_id: str
    state: "Tensor | None" = (
        None  # optional (num_steps + 1, state_dim) resident true-state labels
    )
