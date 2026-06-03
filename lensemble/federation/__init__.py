"""lensemble.federation — DiLoCo outer loop, round state machine, roles (docs/rfcs/RFC-0013)."""

from __future__ import annotations

from .coordinator import Coordinator
from .messages import (
    CONTROL_MESSAGE_SCHEMA_VERSION,
    Commitment,
    ControlMessage,
    RoundClose,
    RoundOpen,
    Update,
    from_pseudogradient,
    parse_control_message,
    to_delta_tensor,
)
from .network import LoopbackChannel, MessageChannel, NetworkedTransport
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
from .state import GlobalState, ParamRef
from .transport import InProcessTransport, Transport

__all__ = [
    "Coordinator",
    "Participant",
    "RoundState",
    "RoundPhase",
    "RoundDriver",
    "GlobalState",
    "ParamRef",
    "Transport",
    "InProcessTransport",
    "MessageChannel",
    "LoopbackChannel",
    "NetworkedTransport",
    "ControlMessage",
    "RoundOpen",
    "Commitment",
    "Update",
    "RoundClose",
    "CONTROL_MESSAGE_SCHEMA_VERSION",
    "parse_control_message",
    "from_pseudogradient",
    "to_delta_tensor",
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
