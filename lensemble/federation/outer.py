"""lensemble.federation.outer — the DiLoCo outer step, the proof-ready aggregation path (RFC-0003 7).

``OuterOptimizer.step`` folds the averaged pseudo-gradient into the canonical global model with Nesterov
momentum: ``(theta, phi)_{t+1} = (theta, phi)_t - eta_out * Nesterov((1/C) * sum_c Delta_c)`` (round step
7). It is the one path required to be **bitwise-reproducible** (``INV-AGG-DETERMINISM``): the deltas are
summed in a fixed, participant-id-sorted order in fp32 (or fp64), with no atomics and no nondeterministic
reductions, so the step can be publicly recomputed (RFC-0006 3).

A per-step determinism self-check recomputes the averaged sum under the same fixed order and compares
content hashes; a mismatch raises :class:`~lensemble.errors.NonDeterministicAggregation` (carrying
``expected_hash``/``got_hash``) and the step does NOT commit, so the round can recompute. This error is
security-critical and never swallowed. Nesterov is stable under a varying participant count ``C``, so a
step proceeds with whatever participants are present.

Sign note: this follows the issue formula ``- eta * Nesterov(avg Delta)`` literally; with the
``Delta = local - global`` convention of the PseudoGradient (#38) the runtime wiring (#41) settles the
descent direction. The determinism guarantee — the point of this module — is independent of sign.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lensemble.errors import LensembleErrorCode, NonDeterministicAggregation

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lensemble.federation.pseudogradient import PseudoGradient


def _content_hash(tensor: Tensor) -> str:
    """SHA-256 over the canonical little-endian bytes of a tensor (platform-stable)."""
    array = tensor.detach().cpu().contiguous().numpy()
    little_endian = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return hashlib.sha256(little_endian.tobytes()).hexdigest()


def assert_bitwise_reproducible(first: Tensor, second: Tensor) -> None:
    """Raise :class:`NonDeterministicAggregation` unless two computations are byte-identical.

    The aggregation path must be bitwise-reproducible (``INV-AGG-DETERMINISM``); a mismatch is
    security-critical and aborts the outer step (never swallowed).
    """
    expected, got = _content_hash(first), _content_hash(second)
    if expected != got or not torch.equal(first, second):
        err = NonDeterministicAggregation(
            "outer-step aggregation was not bitwise-reproducible; aborting (INV-AGG-DETERMINISM)",
            code=LensembleErrorCode.AGG_NONDETERMINISTIC,
            remediation="ensure a fixed fp32/fp64 reduction order, no atomics, no nondeterministic kernels",
        )
        err.expected_hash = expected  # type: ignore[attr-defined]
        err.got_hash = got  # type: ignore[attr-defined]
        raise err


class OuterOptimizer:
    """Nesterov-momentum DiLoCo outer optimizer over a set of ``PseudoGradient`` deltas (RFC-0003 7).

    Stateful across rounds (it carries the Nesterov velocity). Two instances with the same configuration,
    the same prior global params, and the same deltas produce byte-identical results.
    """

    def __init__(
        self,
        *,
        lr: float,
        momentum: float = 0.9,
        nesterov: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.nesterov = nesterov
        self.dtype = dtype
        self._velocity: Tensor | None = None

    def average_deltas(self, deltas: Mapping[str, PseudoGradient]) -> Tensor:
        """``(1/C) * sum_c Delta_c`` summed in a fixed participant-id-sorted order (deterministic)."""
        if not deltas:
            raise NonDeterministicAggregation(
                "no pseudo-gradients to aggregate; the outer step has nothing to fold",
                code=LensembleErrorCode.AGG_NONDETERMINISTIC,
                remediation="aggregate over at least one participant delta",
            )
        ordered = [deltas[pid].delta.to(self.dtype) for pid in sorted(deltas)]
        accumulator = torch.zeros_like(ordered[0])
        for delta in ordered:  # fixed reduction order — fp32/fp64, no atomics
            accumulator = accumulator + delta
        return accumulator / len(ordered)

    def step(
        self, global_params: Tensor, deltas: Mapping[str, PseudoGradient]
    ) -> Tensor:
        """One outer step: average the deltas, apply Nesterov momentum, return the new global params.

        Bitwise-reproducible: the averaged sum is recomputed under the same fixed order and compared
        (``assert_bitwise_reproducible``); a mismatch raises ``NonDeterministicAggregation`` and the step
        does not commit.
        """
        averaged = self.average_deltas(deltas)
        assert_bitwise_reproducible(averaged, self.average_deltas(deltas))  # self-check

        velocity = (
            torch.zeros_like(averaged) if self._velocity is None else self._velocity
        )
        velocity = self.momentum * velocity + averaged
        update = averaged + self.momentum * velocity if self.nesterov else velocity
        self._velocity = velocity
        return global_params.to(self.dtype) - self.lr * update
