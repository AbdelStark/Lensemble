"""lensemble.federation.coordinator — see docs/rfcs/RFC-0013. Stub scaffolded by #2."""
from __future__ import annotations
from typing import Any


class Coordinator:
    """Orchestrates the DiLoCo outer loop and the round state machine (RFC-0013). Implemented by #42."""

    def __init__(self, cfg: Any, participants: Any | None = None) -> None:
        raise NotImplementedError("lensemble.federation.Coordinator is implemented by #42")

    def run(self, num_rounds: int) -> None:
        raise NotImplementedError("Coordinator.run is implemented by #42")
