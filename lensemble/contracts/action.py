"""lensemble.contracts.action — the per-embodiment ``ActionSpec`` descriptor (docs/rfcs/RFC-0007 3).

``ActionSpec`` describes one embodiment's action space. It is the input to action-head construction
and the declared metadata carried in a ``DatasetCommitment``. It is frozen and hashable so its content
hash is a stable, join-declared identifier (recorded in the ``RunManifest``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionKind(str, Enum):
    """Whether an embodiment's action space is continuous or discrete."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Per-embodiment action-space descriptor (conventions 8). Local; declared at join.

    ``num_classes`` is the per-dimension category count tuple for a discrete space (``len == dim``,
    each ``>= 2``) and ``None`` for a continuous one. (RFC-0007 3 annotates it ``int | None`` but its
    own validation rule requires per-dim counts of length ``dim``; the per-dim tuple is the coherent
    type and is implemented here.)
    """

    embodiment_id: str  # stable id, e.g. "so101-arm-7dof"
    kind: ActionKind  # continuous | discrete
    dim: int  # action dimensionality (>0)
    low: tuple[float, ...] | None  # per-dim lower bounds; len==dim if continuous
    high: tuple[float, ...] | None  # per-dim upper bounds; len==dim if continuous
    num_classes: (
        tuple[int, ...] | None
    )  # per-dim category counts if discrete; else None
    units: tuple[str, ...]  # per-dim unit label, len==dim (e.g. "rad", "m/s")
    wmcp_version: str  # MUST equal WMCP_VERSION (INV-WMCP)
