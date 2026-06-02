"""lensemble.federation.participant — see docs/rfcs/RFC-0013 / RFC-0003. Stub scaffolded by #2."""
from __future__ import annotations
from typing import Any


class RunResult:
    """Result of a local training run: checkpoint, manifest, final metrics (02-public-api 1.2). #43."""


class Participant:
    """Holds sovereign data; runs H inner steps; emits a `PseudoGradient` (RFC-0013). Implemented by #43."""

    def __init__(self, cfg: Any, dataset: Any | None = None) -> None:
        raise NotImplementedError("lensemble.federation.Participant is implemented by #43")

    def local_round(self, global_state: Any, round_seed: int) -> Any:
        raise NotImplementedError("Participant.local_round is implemented by #43")


def train_local(config: Any) -> "RunResult":
    """Single-site / participant-local training (the Stage-A inner loop, 02-public-api 1.2). #43."""
    raise NotImplementedError("lensemble.train_local is implemented by #43 (participant local round)")
