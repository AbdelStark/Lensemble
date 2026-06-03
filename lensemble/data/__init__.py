"""lensemble.data — the per-participant data layer (docs/rfcs/RFC-0004).

Episodes/transitions/windows are residency-bound (``INV-RESIDENCY``); only pseudo-gradients leave a
boundary. The on-disk backend adapters (``lance``/``hdf5``/``lerobot``) land here behind the
``EpisodeDataset.fmt`` selector and the ``register_adapter`` extension point (#22); the egress guard
lands with #23. Importing this package self-registers the three built-in adapters.
"""

from __future__ import annotations

from lensemble.data.adapters import load_episodes, register_adapter, save_episodes
from lensemble.data.dataset import EpisodeDataset
from lensemble.data.episode import Episode, Transition, Window
from lensemble.data.quality import DataQualityMetadata, validate_join_precondition
from lensemble.data.residency import EgressRole, guard_egress

__all__ = [
    "Transition",
    "Episode",
    "Window",
    "EpisodeDataset",
    "save_episodes",
    "load_episodes",
    "register_adapter",
    "guard_egress",
    "EgressRole",
    "DataQualityMetadata",
    "validate_join_precondition",
]
