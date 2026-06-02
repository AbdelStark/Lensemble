"""lensemble.federation — DiLoCo outer loop, round state machine, roles (docs/rfcs/RFC-0013)."""

from __future__ import annotations

from .coordinator import Coordinator
from .participant import Participant, train_local
from .pseudogradient import PseudoGradient, build_pseudogradient
from .round import RoundState

__all__ = [
    "Coordinator",
    "Participant",
    "RoundState",
    "train_local",
    "PseudoGradient",
    "build_pseudogradient",
]
