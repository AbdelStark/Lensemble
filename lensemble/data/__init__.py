"""lensemble.data — the per-participant data layer (docs/rfcs/RFC-0004).

Episodes/transitions/windows are residency-bound (``INV-RESIDENCY``); only pseudo-gradients leave a
boundary. Backend adapters (``lance``/``hdf5``/``lerobot``) and the egress guard land with #22 and #23.
"""

from __future__ import annotations

from lensemble.data.dataset import EpisodeDataset
from lensemble.data.episode import Episode, Transition, Window
from lensemble.data.quality import DataQualityMetadata, validate_join_precondition
from lensemble.data.residency import EgressRole, guard_egress

__all__ = [
    "Transition",
    "Episode",
    "Window",
    "EpisodeDataset",
    "guard_egress",
    "EgressRole",
    "DataQualityMetadata",
    "validate_join_precondition",
]
