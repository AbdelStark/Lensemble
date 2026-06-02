"""lensemble.gauge — frame anchoring, Procrustes alignment, drift (docs/rfcs/RFC-0002)."""

from __future__ import annotations

from .anchor import FrameAnchor
from .drift import FrameDriftReport, PairDrift, frame_drift
from .procrustes import procrustes_align

__all__ = [
    "frame_drift",
    "FrameDriftReport",
    "PairDrift",
    "procrustes_align",
    "FrameAnchor",
]
