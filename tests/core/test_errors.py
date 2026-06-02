"""The error taxonomy matches docs/spec/04-error-model.md (2 hierarchy, 4 codes). Issue #3."""

from __future__ import annotations

import pytest

from lensemble import errors as E
from lensemble.errors import LensembleError, LensembleErrorCode

# Expected enum members -> string values (04-error-model 4), exactly.
EXPECTED_CODES = {
    "INTERNAL": "internal",
    "CONFIG_INVALID": "config_invalid",
    "WMCP_CONTRACT_VIOLATION": "wmcp_contract_violation",
    "RESIDENCY_VIOLATION": "residency_violation",
    "GAUGE_FAILED": "gauge_failed",
    "FRAME_DRIFT_EXCEEDED": "frame_drift_exceeded",
    "PROCRUSTES_DEGENERATE": "procrustes_degenerate",
    "AGGREGATION_FAILED": "aggregation_failed",
    "SECURE_AGG_FAILED": "secure_agg_failed",
    "AGG_NONDETERMINISTIC": "agg_nondeterministic",
    "DP_BUDGET_EXCEEDED": "dp_budget_exceeded",
    "PROVENANCE_FAILED": "provenance_failed",
    "COMMITMENT_MISMATCH": "commitment_mismatch",
    "MERKLE_VERIFY_FAILED": "merkle_verify_failed",
    "SCHEMA_VERSION_MISMATCH": "schema_version_mismatch",
    "CHECKPOINT_INTEGRITY": "checkpoint_integrity",
    "ROUND_FAILED": "round_failed",
    "FAULT_TOLERANCE_EXCEEDED": "fault_tolerance_exceeded",
    "PROBE_INVALID": "probe_invalid",
    "EVALUATION_FAILED": "evaluation_failed",
}

# leaf -> category parent (04-error-model 2)
CATEGORY = {
    "FrameDriftExceeded": "GaugeError",
    "DegenerateProcrustes": "GaugeError",
    "SecureAggregationError": "AggregationError",
    "NonDeterministicAggregation": "AggregationError",
    "CommitmentMismatch": "ProvenanceError",
    "MerkleVerificationError": "ProvenanceError",
    "SchemaVersionMismatch": "ArtifactError",
    "CheckpointIntegrityError": "ArtifactError",
    "FaultToleranceExceeded": "RoundError",
}

DIRECT = [
    "ConfigError",
    "ContractViolation",
    "ResidencyViolation",
    "GaugeError",
    "AggregationError",
    "PrivacyBudgetExceeded",
    "ProvenanceError",
    "ArtifactError",
    "RoundError",
    "ProbeError",
    "EvaluationError",
]


def test_code_enum_is_str_subclass() -> None:
    assert issubclass(LensembleErrorCode, str)


def test_code_member_set_matches_spec_exactly() -> None:
    actual = {m.name: m.value for m in LensembleErrorCode}
    assert actual == EXPECTED_CODES


def test_all_errors_subclass_base() -> None:
    for name in [*DIRECT, *CATEGORY]:
        cls = getattr(E, name)
        assert issubclass(cls, LensembleError), name


def test_leaves_subclass_category_parent() -> None:
    for leaf, parent in CATEGORY.items():
        assert issubclass(getattr(E, leaf), getattr(E, parent)), f"{leaf} not under {parent}"


def test_construction_requires_code_and_remediation() -> None:
    err = LensembleError("boom", code=LensembleErrorCode.INTERNAL, remediation="do X")
    assert err.code is LensembleErrorCode.INTERNAL
    assert err.remediation == "do X"
    with pytest.raises(TypeError):
        LensembleError("boom", remediation="x")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        LensembleError("boom", code=LensembleErrorCode.INTERNAL)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        E.FrameDriftExceeded("boom")  # type: ignore[call-arg]


def test_subclass_carries_code_and_remediation() -> None:
    err = E.NonDeterministicAggregation(
        "outer step not reproducible",
        code=LensembleErrorCode.AGG_NONDETERMINISTIC,
        remediation="recompute with the canonical reduction order",
    )
    assert isinstance(err, E.AggregationError)
    assert isinstance(err, LensembleError)
    assert err.code == "agg_nondeterministic"  # str-valued enum compares to its value


def test_all_exports_present() -> None:
    for name in E.__all__:
        assert hasattr(E, name), name
