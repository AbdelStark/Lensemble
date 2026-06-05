"""lensemble.federation — DiLoCo outer loop, round state machine, roles (docs/rfcs/RFC-0013)."""

from __future__ import annotations

from .ablation import run_ablation_ladder
from .agent import (
    PARTICIPANT_AGENT_PREFLIGHT_SCHEMA_VERSION,
    PARTICIPANT_AGENT_ROUND_STATE_SCHEMA_VERSION,
    ParticipantAgentPreflight,
    ParticipantAgentRoundResult,
    ParticipantAgentRoundState,
    Phase3ParticipantAgent,
)
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
from .phase3_privacy import (
    PHASE3_AGGREGATION_PRIVACY_REPORT_SCHEMA_VERSION,
    Phase3AggregationPrivacyReport,
    Phase3DPAccountingReport,
    Phase3SecureAggregationReport,
    build_phase3_aggregation_privacy_report,
)
from .pseudogradient import PseudoGradient, build_pseudogradient
from .quant import (
    dequantize_int8,
    int8_roundtrip_l2_bound,
    quantize_int8,
    wire_roundtrip,
)
from .round import RoundDriver, RoundPhase, RoundState
from .service import (
    COORDINATOR_SERVICE_REPORT_SCHEMA_VERSION,
    COORDINATOR_SERVICE_TRACE_SCHEMA_VERSION,
    CoordinatorParticipantReport,
    CoordinatorServiceEvent,
    CoordinatorServiceReport,
    Phase3CoordinatorService,
    Phase3DropoutPolicy,
)
from .simulation import (
    RoundMetrics,
    SiloData,
    SimulationResult,
    run_federated_simulation,
)
from .state import GlobalState, ParamRef
from .sweeps import (
    non_iid_severity_sweep,
    participant_horizon_sweep,
    scale_sweep,
)
from .transport import InProcessTransport, Transport

__all__ = [
    "Coordinator",
    "Participant",
    "Phase3ParticipantAgent",
    "ParticipantAgentPreflight",
    "ParticipantAgentRoundState",
    "ParticipantAgentRoundResult",
    "PARTICIPANT_AGENT_PREFLIGHT_SCHEMA_VERSION",
    "PARTICIPANT_AGENT_ROUND_STATE_SCHEMA_VERSION",
    "RoundState",
    "RoundPhase",
    "RoundDriver",
    "Phase3CoordinatorService",
    "Phase3DropoutPolicy",
    "CoordinatorServiceEvent",
    "CoordinatorServiceReport",
    "CoordinatorParticipantReport",
    "COORDINATOR_SERVICE_TRACE_SCHEMA_VERSION",
    "COORDINATOR_SERVICE_REPORT_SCHEMA_VERSION",
    "PHASE3_AGGREGATION_PRIVACY_REPORT_SCHEMA_VERSION",
    "Phase3AggregationPrivacyReport",
    "Phase3SecureAggregationReport",
    "Phase3DPAccountingReport",
    "build_phase3_aggregation_privacy_report",
    "SiloData",
    "RoundMetrics",
    "SimulationResult",
    "run_federated_simulation",
    "run_ablation_ladder",
    "non_iid_severity_sweep",
    "participant_horizon_sweep",
    "scale_sweep",
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
