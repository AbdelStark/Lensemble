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
    ClaimRoundMetricEvidence,
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
from .phase2 import (
    Phase2MatrixRow,
    default_phase2_matrix,
    render_phase2_matrix_markdown,
)
from .phase2_curves import (
    PHASE2_CURVES_REPORT_SCHEMA_VERSION,
    Phase2BaselinesCurvesReport,
    Phase2BlockedComparison,
    Phase2ClaimCurveInput,
    Phase2CurvePoint,
    Phase2SourceReportRef,
    build_phase2_baselines_curves_report,
    parse_phase2_baselines_curves_report,
    phase2_source_report_ref_from_path,
)
from .phase2_downstream import (
    PHASE2_DOWNSTREAM_REPORT_SCHEMA_VERSION,
    Phase2CheckpointRef,
    Phase2DownstreamEvalReport,
    Phase2EvalTask,
    Phase2PlannerBudget,
    build_phase2_downstream_eval_report,
    parse_phase2_downstream_eval_report,
    phase2_eval_config_from_checkpoint,
)
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
    "Phase2MatrixRow",
    "default_phase2_matrix",
    "render_phase2_matrix_markdown",
    "Phase2CheckpointRef",
    "Phase2DownstreamEvalReport",
    "Phase2EvalTask",
    "Phase2PlannerBudget",
    "PHASE2_DOWNSTREAM_REPORT_SCHEMA_VERSION",
    "build_phase2_downstream_eval_report",
    "parse_phase2_downstream_eval_report",
    "phase2_eval_config_from_checkpoint",
    "Phase2BaselinesCurvesReport",
    "Phase2BlockedComparison",
    "Phase2ClaimCurveInput",
    "Phase2CurvePoint",
    "Phase2SourceReportRef",
    "PHASE2_CURVES_REPORT_SCHEMA_VERSION",
    "build_phase2_baselines_curves_report",
    "parse_phase2_baselines_curves_report",
    "phase2_source_report_ref_from_path",
    "EvalReport",
    "EVAL_REPORT_SCHEMA_VERSION",
    "parse_eval_report",
    "ClaimMVPReport",
    "ClaimMetricEvidence",
    "ClaimParticipantEvidence",
    "ClaimPublicationEvidence",
    "ClaimRoundMetricEvidence",
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
