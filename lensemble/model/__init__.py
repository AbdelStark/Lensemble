"""lensemble.model — encoder, predictor, action heads, objective (docs/rfcs/RFC-0008)."""

from __future__ import annotations

from .action_head import build_action_head
from .encoder import (
    Encoder,
    build_encoder,
    build_encoder_from_arch,
    load_warmstart,
    snapshot_reference,
)
from .objective import AnchorTerm, LossTerms, Objective
from .predictor import Predictor, build_predictor
from .sigreg import build_sketch, sigreg_statistic

__all__ = [
    "build_encoder",
    "build_encoder_from_arch",
    "Encoder",
    "load_warmstart",
    "snapshot_reference",
    "build_predictor",
    "Predictor",
    "build_action_head",
    "Objective",
    "LossTerms",
    "AnchorTerm",
    "build_sketch",
    "sigreg_statistic",
]
