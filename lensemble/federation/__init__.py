"""lensemble.federation — DiLoCo outer loop, round state machine, roles (docs/rfcs/RFC-0013)."""

from __future__ import annotations

from .coordinator import Coordinator
from .outer import OuterOptimizer, assert_bitwise_reproducible
from .participant import Participant, train_local
from .pseudogradient import PseudoGradient, build_pseudogradient
from .quant import (
    dequantize_int8,
    int8_roundtrip_l2_bound,
    quantize_int8,
    wire_roundtrip,
)
from .round import RoundDriver, RoundPhase, RoundState

__all__ = [
    "Coordinator",
    "Participant",
    "RoundState",
    "RoundPhase",
    "RoundDriver",
    "train_local",
    "PseudoGradient",
    "build_pseudogradient",
    "OuterOptimizer",
    "assert_bitwise_reproducible",
    "quantize_int8",
    "dequantize_int8",
    "int8_roundtrip_l2_bound",
    "wire_roundtrip",
]
