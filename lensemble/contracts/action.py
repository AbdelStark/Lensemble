"""lensemble.contracts.action — the per-embodiment ``ActionSpec`` descriptor (docs/rfcs/RFC-0007 3).

``ActionSpec`` describes one embodiment's action space. It is the input to action-head construction
and the declared metadata carried in a ``DatasetCommitment``. It is frozen and hashable so its content
hash is a stable, join-declared identifier (recorded in the ``RunManifest``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
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


def union_action_specs(specs: Sequence[ActionSpec]) -> ActionSpec:
    """The single consortium-agreed ``ActionSpec`` covering a set of per-silo specs.

    Sovereign silos partition the same embodiment's episodes, so every silo agrees on
    ``embodiment_id``/``kind``/``dim``/``num_classes``/``units``/``wmcp_version`` but reports its own
    *observed* continuous ``low``/``high`` (per-file action min/max). The consortium action contract a
    coordinator and every participant must agree on (`INV-WMCP`, the manifest's accepted action contract)
    therefore uses the element-wise union — ``min`` of the lows and ``max`` of the highs — so every silo's
    local actions fall inside the agreed bounds. Discrete specs (no continuous bounds) must be identical.

    Raises :class:`ValueError` on an empty sequence or any disagreement on the non-bound fields.
    """

    if not specs:
        raise ValueError("union_action_specs needs at least one ActionSpec")
    head = specs[0]
    for other in specs[1:]:
        for field in (
            "embodiment_id",
            "kind",
            "dim",
            "num_classes",
            "units",
            "wmcp_version",
        ):
            if getattr(other, field) != getattr(head, field):
                raise ValueError(
                    f"action specs disagree on {field!r}: {getattr(head, field)!r} vs "
                    f"{getattr(other, field)!r}; only continuous low/high may differ across silos"
                )
    if head.low is None or head.high is None:
        # Discrete (or unbounded) spaces carry no continuous bounds to union; they must match exactly.
        for other in specs[1:]:
            if other.low != head.low or other.high != head.high:
                raise ValueError(
                    "action specs without continuous bounds must be identical across silos"
                )
        return head
    lows = [spec.low for spec in specs]
    highs = [spec.high for spec in specs]
    if any(low is None or high is None for low, high in zip(lows, highs)):
        raise ValueError("mixed bounded/unbounded action specs cannot be unioned")
    union_low = tuple(min(low[i] for low in lows) for i in range(head.dim))  # type: ignore[index]
    union_high = tuple(max(high[i] for high in highs) for i in range(head.dim))  # type: ignore[index]
    return replace(head, low=union_low, high=union_high)
