"""lensemble.federation.round — the outer-round state machine (RFC-0013 2 / RFC-0003 2).

One DiLoCo outer round advances ``OPEN -> COLLECTING -> AGGREGATING -> ALIGNING -> COMMITTING -> CLOSED``
with an ``any -> ABORTED`` short-circuit. :class:`RoundDriver` enforces the legal-transition precondition
and the atomicity at ``COMMITTING``: either the outer step applies and the global hash advances
(``-> CLOSED``), or the round aborts and the canonical global hash is left unchanged (no partial commit).

Transition triggers (RFC-0013 2; the orchestration that supplies them is the Coordinator, #42):

- ``OPEN -> COLLECTING``: a quorum of ``K`` participants joined and ``Commitment``-bound.
- ``COLLECTING -> AGGREGATING``: all ``Update``s received, or a collect timeout with quorum held.
- ``AGGREGATING -> ALIGNING``: the revealed secure sum reproduces (``INV-AGG-DETERMINISM``).
- ``ALIGNING -> COMMITTING``: frame drift measured, the Procrustes backstop applied if drift > tau.
- ``COMMITTING -> CLOSED``: outer step applied, the global hash committed, the ledger appended.
- ``CLOSED -> OPEN``: the next round opens.

Failure dispatch is the caller's (RFC-0013 7): a quorum below ``K`` -> ``FaultToleranceExceeded``; below
the secure-agg threshold -> ``SecureAggregationError``; a determinism mismatch ->
``NonDeterministicAggregation``; an exhausted budget -> ``PrivacyBudgetExceeded``; an unbound delta ->
``CommitmentMismatch``; a tampered checkpoint -> ``CheckpointIntegrityError``; a malformed message ->
``RoundError``. :meth:`RoundDriver.abort` routes any of these to ``ABORTED`` and re-raises, leaving the
global hash unchanged. These errors are security-critical and never swallowed.
"""

from __future__ import annotations

from enum import Enum

from lensemble.errors import LensembleError, LensembleErrorCode, RoundError

# RFC-0013 §1 imports ``GlobalState`` / ``PseudoGradient`` from ``lensemble.federation.round`` (the
# runtime-class module). Re-export them here so that import resolves; the canonical definitions live in
# ``state.py`` (03 §7) and ``pseudogradient.py`` (03 §6).
from lensemble.federation.pseudogradient import (  # noqa: E402,F401  (re-export for RFC-0013 §1)
    PseudoGradient,
)
from lensemble.federation.state import GlobalState, ParamRef  # noqa: E402,F401


class RoundState(str, Enum):
    """The outer-round lifecycle state (RFC-0013 2). String values agree with 03 8 ``RoundPhase``."""

    OPEN = "open"
    COLLECTING = "collecting"
    AGGREGATING = "aggregating"
    ALIGNING = "aligning"
    COMMITTING = "committing"
    CLOSED = "closed"
    ABORTED = "aborted"


# 03 8 names the lifecycle enum ``RoundPhase``; it is the same value set as the public ``RoundState``.
RoundPhase = RoundState

# The legal forward transitions; ``any -> ABORTED`` is handled by :meth:`RoundDriver.abort`.
_ALLOWED: dict[RoundState, frozenset[RoundState]] = {
    RoundState.OPEN: frozenset({RoundState.COLLECTING}),
    RoundState.COLLECTING: frozenset({RoundState.AGGREGATING}),
    RoundState.AGGREGATING: frozenset({RoundState.ALIGNING}),
    RoundState.ALIGNING: frozenset({RoundState.COMMITTING}),
    RoundState.COMMITTING: frozenset({RoundState.CLOSED}),
    RoundState.CLOSED: frozenset({RoundState.OPEN}),  # the next round opens
    RoundState.ABORTED: frozenset(),  # terminal
}
_TERMINAL = (RoundState.CLOSED, RoundState.ABORTED)


class RoundDriver:
    """Drives one outer round through :class:`RoundState`, enforcing legal transitions and commit atomicity.

    Holds the canonical ``global_hash``; it advances only on a successful :meth:`commit` and is left
    unchanged on :meth:`abort` (no partial commit).
    """

    def __init__(self, *, global_hash: str, round_index: int = 0) -> None:
        self.round_index = round_index
        self.state = RoundState.OPEN
        self._global_hash = global_hash

    @property
    def global_hash(self) -> str:
        return self._global_hash

    def to(self, target: RoundState) -> RoundState:
        """Advance to ``target`` if it is a legal successor; else raise ``RoundError`` (precondition)."""
        if target not in _ALLOWED[self.state]:
            raise RoundError(
                f"illegal round transition {self.state.value!r} -> {target.value!r}",
                code=LensembleErrorCode.ROUND_FAILED,
                remediation="advance OPEN->COLLECTING->AGGREGATING->ALIGNING->COMMITTING->CLOSED, or abort",
            )
        self.state = target
        return self.state

    def commit(self, new_global_hash: str) -> RoundState:
        """Atomically apply the outer step: ``COMMITTING -> CLOSED`` and advance the global hash.

        If the round is not in ``COMMITTING`` the transition raises and the hash is unchanged.
        """
        self.to(RoundState.CLOSED)
        self._global_hash = new_global_hash
        return self.state

    def abort(self, error: LensembleError) -> None:
        """Route any non-terminal round to ``ABORTED`` and re-raise ``error``; the global hash is unchanged.

        Aborting an already-terminal (``CLOSED``/``ABORTED``) round is itself a ``RoundError``.
        """
        if self.state in _TERMINAL:
            raise RoundError(
                f"cannot abort a {self.state.value!r} round",
                code=LensembleErrorCode.ROUND_FAILED,
                remediation="abort only an in-flight round (OPEN..COMMITTING)",
            )
        self.state = RoundState.ABORTED
        raise error

    def open_next(self) -> RoundState:
        """Open the next round: ``CLOSED -> OPEN`` with an incremented round index."""
        self.to(RoundState.OPEN)
        self.round_index += 1
        return self.state
