"""lensemble.eval — latent MPC planner, eval harness, metrics (docs/rfcs/RFC-0005)."""

from __future__ import annotations

from .ablation import (
    LADDER_RUNGS,
    RungReport,
    RungSpec,
    lambda_anc_sweep,
)
from .baselines import BASELINES, gap_recovery_fraction, load_baseline
from .claim_mvp import (
    CLAIM_MVP_REPORT_SCHEMA_VERSION,
    ClaimMetricEvidence,
    ClaimMVPReport,
    ClaimParticipantEvidence,
    ClaimPublicationEvidence,
    build_claim_mvp_report,
    parse_claim_mvp_report,
)
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
from .report import EVAL_REPORT_SCHEMA_VERSION, EvalReport, parse_eval_report
from .sweeps import (
    SiloPartition,
    partition_synthetic_noniid,
    sample_drift_pairs,
)
from .world import EvalWorld, register_env, resolve_env

__all__ = [
    "evaluate",
    "Planner",
    "EvalReport",
    "EVAL_REPORT_SCHEMA_VERSION",
    "parse_eval_report",
    "ClaimMVPReport",
    "ClaimMetricEvidence",
    "ClaimParticipantEvidence",
    "ClaimPublicationEvidence",
    "CLAIM_MVP_REPORT_SCHEMA_VERSION",
    "build_claim_mvp_report",
    "parse_claim_mvp_report",
    "EvalWorld",
    "register_env",
    "resolve_env",
    "success_rate",
    "planning_cost",
    "effective_dim",
    "linear_probe_accuracy",
    "comm_bytes",
    "quant_ratio",
    "BASELINES",
    "load_baseline",
    "gap_recovery_fraction",
    "LADDER_RUNGS",
    "RungReport",
    "RungSpec",
    "lambda_anc_sweep",
    "SiloPartition",
    "partition_synthetic_noniid",
    "sample_drift_pairs",
]
