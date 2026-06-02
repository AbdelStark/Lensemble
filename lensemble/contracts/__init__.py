"""lensemble.contracts — the WMCP latent/action contract (docs/rfcs/RFC-0007).

The lowest typed layer that makes heterogeneous-embodiment federation well-posed: it pins the shape,
dtype, and frame semantics every encoder emits and every predictor consumes (``INV-WMCP``).
"""

from __future__ import annotations

from lensemble.contracts.action import ActionKind, ActionSpec
from lensemble.contracts.action_head import ActionHead
from lensemble.contracts.conformance import check_latent_state, validate_action_spec
from lensemble.contracts.latent import WMCP_VERSION, LatentState

__all__ = [
    "LatentState",
    "WMCP_VERSION",
    "check_latent_state",
    "ActionSpec",
    "ActionKind",
    "ActionHead",
    "validate_action_spec",
]
