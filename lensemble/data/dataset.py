"""lensemble.data.dataset — the participant-local ``EpisodeDataset`` windowed loader (RFC-0004 1).

Iterates fixed-length :class:`~lensemble.data.episode.Window` slices for next-embedding prediction. The
loader materializes RAW windows only and computes no embeddings; the local trainer that consumes them is
inside the trust boundary. ``EpisodeDataset`` is residency-bound (``INV-RESIDENCY``): it carries the
``exportable`` flag the egress guard reads and exposes no method that serializes raw tensors outbound.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal

import torch

from lensemble.data.episode import Episode, Transition, Window
from lensemble.errors import ContractViolation, LensembleErrorCode

Format = Literal["lance", "hdf5", "lerobot"]


def _build_window(
    transitions: Sequence[Transition], embodiment_id: str, num_steps: int
) -> Window:
    obs = torch.stack([t.obs_t for t in transitions] + [transitions[-1].obs_tp1])
    actions = torch.stack([t.action_t for t in transitions])
    return Window(
        obs=obs, actions=actions, num_steps=num_steps, embodiment_id=embodiment_id
    )


def _check_window(window: Window, *, action_dim: int) -> None:
    """Validate a window's shapes (03-data-model 5). Raises ``ContractViolation`` on mismatch."""

    def fail(msg: str, remediation: str) -> None:
        raise ContractViolation(
            msg,
            code=LensembleErrorCode.WMCP_CONTRACT_VIOLATION,
            remediation=remediation,
        )

    if window.obs.shape[0] != window.num_steps + 1:
        fail(
            f"obs length {window.obs.shape[0]} != num_steps + 1 ({window.num_steps + 1})",
            "a window over num_steps transitions has num_steps + 1 observations",
        )
    if window.actions.shape[0] != window.num_steps:
        fail(
            f"actions length {window.actions.shape[0]} != num_steps ({window.num_steps})",
            "supply exactly num_steps conditioning actions",
        )
    if window.actions.ndim < 2 or window.actions.shape[1] != action_dim:
        got = tuple(window.actions.shape)
        fail(
            f"actions trailing dim != action_spec.dim ({action_dim}); got shape {got}",
            "store actions of shape (num_steps, action_spec.dim) matching the episode's ActionSpec",
        )


class EpisodeDataset:
    """Participant-local store iterating :class:`Window` slices (RFC-0004 1).

    Residency-bound (``INV-RESIDENCY``): never exposes a method that serializes raw tensors across
    egress. Backend loading (``lance`` / ``hdf5`` / ``lerobot``) lands with the data-format adapters
    (#22); this loader holds episodes in memory and yields windows.
    """

    path: Path | None
    fmt: Format
    exportable: bool

    def __init__(
        self,
        episodes: Sequence[Episode],
        *,
        path: Path | None = None,
        fmt: Format = "lance",
        exportable: bool = False,
    ) -> None:
        self._episodes: tuple[Episode, ...] = tuple(episodes)
        self.path = path
        self.fmt = fmt
        self.exportable = exportable

    def __len__(self) -> int:
        return len(self._episodes)

    def windows(self, num_steps: int) -> Iterator[Window]:
        """Yield every contiguous ``num_steps``-transition window across the episodes.

        Each window is shape-validated against the episode's ``ActionSpec`` (``ContractViolation`` on a
        mismatch). Episodes shorter than ``num_steps`` transitions yield no windows.
        """
        if num_steps <= 0:
            raise ContractViolation(
                f"num_steps must be > 0, got {num_steps}",
                code=LensembleErrorCode.WMCP_CONTRACT_VIOLATION,
                remediation="set data.num_steps to a positive window horizon",
            )
        for episode in self._episodes:
            transitions = episode.transitions
            action_dim = episode.action_spec.dim
            for start in range(0, len(transitions) - num_steps + 1):
                window = _build_window(
                    transitions[start : start + num_steps],
                    episode.embodiment_id,
                    num_steps,
                )
                _check_window(window, action_dim=action_dim)
                yield window
