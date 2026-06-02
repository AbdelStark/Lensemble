"""lensemble.errors — the canonical error taxonomy.

Implements the exception hierarchy of ``docs/spec/04-error-model.md`` (2) and the
``LensembleErrorCode`` enum (4); summary in ``docs/spec/conventions.md`` (6). Owned by ``core``
(``docs/rfcs/RFC-0001-architecture.md`` 2).

``LensembleError`` is the single base; every error carries a ``.code`` and a ``.remediation``.
The category parents (``GaugeError``, ``AggregationError``, ``ProvenanceError``, ``ArtifactError``,
``RoundError``) let a caller catch a whole family or a specific leaf. ``ResidencyViolation``,
``CommitmentMismatch``, and ``NonDeterministicAggregation`` are security-critical and must never be
caught-and-ignored (04-error-model 1, principle 3).
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "LensembleErrorCode",
    "LensembleError",
    "ConfigError",
    "ContractViolation",
    "ResidencyViolation",
    "GaugeError",
    "FrameDriftExceeded",
    "DegenerateProcrustes",
    "AggregationError",
    "SecureAggregationError",
    "NonDeterministicAggregation",
    "PrivacyBudgetExceeded",
    "ProvenanceError",
    "CommitmentMismatch",
    "MerkleVerificationError",
    "ArtifactError",
    "SchemaVersionMismatch",
    "CheckpointIntegrityError",
    "RoundError",
    "FaultToleranceExceeded",
    "ProbeError",
    "EvaluationError",
]


class LensembleErrorCode(str, Enum):
    """Stable, string-valued error codes (04-error-model 4).

    Codes are append-only across pre-1.0 minors and frozen at 1.0; a removed code is reserved,
    never reused. Emitted into structured logs and the CLI exit-code mapping.
    """

    # core / config
    INTERNAL = "internal"
    CONFIG_INVALID = "config_invalid"
    # contracts (WMCP)
    WMCP_CONTRACT_VIOLATION = "wmcp_contract_violation"
    # residency (security-critical)
    RESIDENCY_VIOLATION = "residency_violation"
    # gauge
    GAUGE_FAILED = "gauge_failed"
    FRAME_DRIFT_EXCEEDED = "frame_drift_exceeded"
    PROCRUSTES_DEGENERATE = "procrustes_degenerate"
    # aggregation
    AGGREGATION_FAILED = "aggregation_failed"
    SECURE_AGG_FAILED = "secure_agg_failed"
    AGG_NONDETERMINISTIC = "agg_nondeterministic"  # security-critical
    # privacy
    DP_BUDGET_EXCEEDED = "dp_budget_exceeded"
    # provenance
    PROVENANCE_FAILED = "provenance_failed"
    COMMITMENT_MISMATCH = "commitment_mismatch"  # security-critical
    MERKLE_VERIFY_FAILED = "merkle_verify_failed"
    # artifacts
    SCHEMA_VERSION_MISMATCH = "schema_version_mismatch"
    CHECKPOINT_INTEGRITY = "checkpoint_integrity"
    # round lifecycle
    ROUND_FAILED = "round_failed"
    FAULT_TOLERANCE_EXCEEDED = "fault_tolerance_exceeded"
    # probe
    PROBE_INVALID = "probe_invalid"
    # evaluation
    EVALUATION_FAILED = "evaluation_failed"


class LensembleError(Exception):
    """Base for all Lensemble errors. Always carries a code and a remediation."""

    code: LensembleErrorCode
    remediation: str

    def __init__(self, message: str, *, code: LensembleErrorCode, remediation: str) -> None:
        super().__init__(message)
        self.code = code
        self.remediation = remediation


class ConfigError(LensembleError):
    """Invalid or inconsistent configuration (raised at config load, never lazily mid-run)."""


class ContractViolation(LensembleError):
    """WMCP nonconformance: a latent shape/dtype/semantics or ``ActionSpec`` mismatch."""


class ResidencyViolation(LensembleError):
    """A raw observation/action/private embedding would cross a trust boundary.

    Security-critical; fail-closed and never caught-and-ignored (``INV-RESIDENCY``).
    """


class GaugeError(LensembleError):
    """Category parent for latent-frame / alignment failures."""


class FrameDriftExceeded(GaugeError):
    """Inter-participant latent frame drift exceeded the configured threshold."""


class DegenerateProcrustes(GaugeError):
    """The SVD is ill-conditioned (near-degenerate singular values) in Procrustes alignment."""


class AggregationError(LensembleError):
    """Category parent for outer-step / secure-sum failures."""


class SecureAggregationError(AggregationError):
    """The masked-sum protocol failed (for example, the live set dropped below the threshold)."""


class NonDeterministicAggregation(AggregationError):
    """The aggregation path was not bitwise-reproducible.

    Security-critical (proof-readiness); abort and recompute, never swallow (``INV-AGG-DETERMINISM``).
    """


class PrivacyBudgetExceeded(LensembleError):
    """The planned ``(eps, delta)`` differential-privacy budget over the rounds is spent."""


class ProvenanceError(LensembleError):
    """Category parent for dataset-commitment / Merkle failures."""


class CommitmentMismatch(ProvenanceError):
    """A pseudo-gradient is bound to the wrong dataset root, or to none.

    Security-critical; reject the update, never swallow (``INV-COMMIT-BINDING``).
    """


class MerkleVerificationError(ProvenanceError):
    """A Merkle root or inclusion proof did not verify."""


class ArtifactError(LensembleError):
    """Category parent for checkpoint / artifact failures."""


class SchemaVersionMismatch(ArtifactError):
    """An on-disk ``schema_version`` is unknown or too new for this reader."""


class CheckpointIntegrityError(ArtifactError):
    """A checkpoint content hash did not match (tamper or corruption)."""


class RoundError(LensembleError):
    """Category parent for round-lifecycle failures."""


class FaultToleranceExceeded(RoundError):
    """Too few participants remain to complete a valid round."""


class ProbeError(LensembleError):
    """The public probe hash did not match the pinned hash, or the probe under-covers the frame."""


class EvaluationError(LensembleError):
    """A latent-MPC rollout, metric, or evaluation-harness failure (or an unregistered env)."""
