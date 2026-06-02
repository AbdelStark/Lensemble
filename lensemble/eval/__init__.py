"""lensemble.eval — latent MPC planner, eval harness, metrics (docs/rfcs/RFC-0005)."""

from __future__ import annotations

from .harness import evaluate
from .metrics import (
    comm_bytes,
    effective_dim,
    linear_probe_accuracy,
    planning_cost,
    quant_ratio,
    success_rate,
)
from .mpc import Planner

__all__ = [
    "evaluate",
    "Planner",
    "success_rate",
    "planning_cost",
    "effective_dim",
    "linear_probe_accuracy",
    "comm_bytes",
    "quant_ratio",
]
