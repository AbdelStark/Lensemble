"""Outer-round state machine: transitions + the ABORTED short-circuit (RFC-0013 2/7). Issue #41.

Drive a toy round OPEN -> ... -> CLOSED asserting each legal transition and its precondition; the
ABORTED path on each failure-mode error leaves the canonical global hash unchanged (no partial commit).
Pure-Python state machine -> tests/unit.
"""

from __future__ import annotations

import pytest

from lensemble.errors import (
    CommitmentMismatch,
    FaultToleranceExceeded,
    LensembleError,
    LensembleErrorCode,
    NonDeterministicAggregation,
    PrivacyBudgetExceeded,
    RoundError,
    SecureAggregationError,
)
from lensemble.federation import RoundDriver, RoundPhase, RoundState

_H0 = "a" * 64  # the prior canonical global hash
_H1 = "b" * 64  # the committed next hash


def test_enum_values_and_phase_alias() -> None:
    assert [s.value for s in RoundState] == [
        "open",
        "collecting",
        "aggregating",
        "aligning",
        "committing",
        "closed",
        "aborted",
    ]
    assert RoundPhase is RoundState  # 03 §8 RoundPhase == the public RoundState enum


def test_happy_path_open_to_closed_advances_hash() -> None:
    driver = RoundDriver(global_hash=_H0, round_index=3)
    for target in (
        RoundState.COLLECTING,
        RoundState.AGGREGATING,
        RoundState.ALIGNING,
        RoundState.COMMITTING,
    ):
        assert driver.to(target) is target
    assert driver.global_hash == _H0  # unchanged until commit
    assert driver.commit(_H1) is RoundState.CLOSED
    assert driver.global_hash == _H1  # commit is the only thing that advances the hash

    assert driver.open_next() is RoundState.OPEN
    assert driver.round_index == 4


def test_illegal_transition_raises_round_error() -> None:
    driver = RoundDriver(global_hash=_H0)
    with pytest.raises(RoundError) as exc:
        driver.to(RoundState.AGGREGATING)  # skips COLLECTING
    assert exc.value.code == LensembleErrorCode.ROUND_FAILED
    assert driver.state is RoundState.OPEN  # no state change on an illegal transition


def _abort_error(name: str) -> LensembleError:
    table: dict[str, LensembleError] = {
        "quorum": FaultToleranceExceeded(
            "quorum below K",
            code=LensembleErrorCode.FAULT_TOLERANCE_EXCEEDED,
            remediation="wait for K participants",
        ),
        "dropout": SecureAggregationError(
            "live set below the reveal threshold",
            code=LensembleErrorCode.SECURE_AGG_FAILED,
            remediation="recover dropped masks or abort",
        ),
        "determinism": NonDeterministicAggregation(
            "aggregation not bitwise-reproducible",
            code=LensembleErrorCode.AGG_NONDETERMINISTIC,
            remediation="fix the reduction order",
        ),
        "budget": PrivacyBudgetExceeded(
            "DP budget spent",
            code=LensembleErrorCode.DP_BUDGET_EXCEEDED,
            remediation="stop training; the (eps, delta) budget is exhausted",
        ),
        "binding": CommitmentMismatch(
            "delta not bound to a valid R_c",
            code=LensembleErrorCode.COMMITMENT_MISMATCH,
            remediation="reject the unbound delta",
        ),
    }
    return table[name]


@pytest.mark.parametrize(
    "reach,error_name",
    [
        ([], "quorum"),  # OPEN: quorum failure
        ([RoundState.COLLECTING, RoundState.AGGREGATING], "dropout"),
        ([RoundState.COLLECTING, RoundState.AGGREGATING], "determinism"),
        ([RoundState.COLLECTING], "budget"),
        ([RoundState.COLLECTING], "binding"),
    ],
)
def test_abort_paths_leave_global_hash_unchanged(
    reach: list[RoundState], error_name: str
) -> None:
    driver = RoundDriver(global_hash=_H0)
    for target in reach:
        driver.to(target)
    error = _abort_error(error_name)
    with pytest.raises(type(error)):
        driver.abort(error)
    assert driver.state is RoundState.ABORTED
    assert driver.global_hash == _H0  # ABORTED never advances the canonical hash


def test_cannot_abort_a_terminal_round() -> None:
    driver = RoundDriver(global_hash=_H0)
    for target in (
        RoundState.COLLECTING,
        RoundState.AGGREGATING,
        RoundState.ALIGNING,
        RoundState.COMMITTING,
    ):
        driver.to(target)
    driver.commit(_H1)  # CLOSED
    with pytest.raises(RoundError):
        driver.abort(_abort_error("quorum"))


def test_commit_requires_committing_state() -> None:
    driver = RoundDriver(global_hash=_H0)
    with pytest.raises(RoundError):
        driver.commit(_H1)  # still OPEN, not COMMITTING
    assert driver.global_hash == _H0  # no partial commit
