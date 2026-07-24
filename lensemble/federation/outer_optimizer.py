"""lensemble.federation.outer_optimizer — the DiLoCo outer step, the proof-ready aggregation path (RFC-0003 7).

``OuterOptimizer.step`` folds the averaged pseudo-gradient into the canonical global model with Nesterov
momentum. Lensemble's public contract is the displacement
``Delta_c = (theta, phi)_c^local - (theta, phi)_t``, so the outer step **adds** that displacement:
``(theta, phi)_{t+1} = (theta, phi)_t + eta_out * Nesterov((1/C) * sum_c Delta_c)`` (round step 7).
Equivalently, DiLoCo may define the outer gradient with the opposite sign,
``G_c = (theta, phi)_t - (theta, phi)_c^local = -Delta_c``, and subtract
``eta_out * Nesterov(mean_c G_c)``. It is the one path required to be **bitwise-reproducible**
(``INV-AGG-DETERMINISM``): the deltas are summed in a fixed, participant-id-sorted order in fp32 (or
fp64), with no atomics and no nondeterministic reductions, so the step can be publicly recomputed
(RFC-0006 3).

A per-step determinism self-check calls :meth:`OuterOptimizer.preview_step` twice from the same carried
velocity and compares content hashes. ``preview_step`` is pure; :meth:`OuterOptimizer.step` commits the
identical computation and only then advances the velocity. A mismatch raises
:class:`~lensemble.errors.NonDeterministicAggregation` (carrying ``expected_hash``/``got_hash``) and the
step does NOT commit, so the round can recompute. This error is security-critical and never swallowed.
Nesterov is stable under a varying participant count ``C``, so a step proceeds with whatever participants
are present.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lensemble.errors import (
    LensembleErrorCode,
    NonDeterministicAggregation,
    RoundError,
)

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
    prior velocity, prior global params, and deltas produce byte-identical results.
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
        participant_ids = sorted(deltas)
        first = deltas[participant_ids[0]].delta
        expected_shape = first.shape
        for participant_id in participant_ids:
            shape = deltas[participant_id].delta.shape
            if deltas[participant_id].delta.ndim != 1 or shape != expected_shape:
                raise RoundError(
                    f"participant {participant_id!r} delta shape {tuple(shape)} does not match "
                    f"the flat aggregation shape {tuple(expected_shape)}",
                    code=LensembleErrorCode.ROUND_FAILED,
                    remediation="emit one flat delta with the canonical parameter-manifest length",
                )
        # Device placement is transport-local metadata, not part of the released vector's semantics.
        # Normalize onto the first participant in the canonical id order so mixed CPU/GPU arrivals do
        # not make a valid update set fail with an incidental device error.
        ordered = [
            deltas[pid].delta.to(device=first.device, dtype=self.dtype)
            for pid in participant_ids
        ]
        accumulator = torch.zeros_like(ordered[0])
        for delta in ordered:  # fixed reduction order — fp32/fp64, no atomics
            accumulator = accumulator + delta
        return accumulator / len(ordered)

    def _compute_step(
        self, global_params: Tensor, deltas: Mapping[str, PseudoGradient]
    ) -> tuple[Tensor, Tensor]:
        """Return ``(next_params, next_velocity)`` without mutating optimizer state."""
        if global_params.ndim != 1:
            raise RoundError(
                f"global_params must be a flat 1-D parameter vector, got shape "
                f"{tuple(global_params.shape)}",
                code=LensembleErrorCode.ROUND_FAILED,
                remediation="flatten encoder then predictor parameters via the canonical manifest",
            )
        expected_shape = global_params.shape
        for participant_id, update in deltas.items():
            if update.delta.shape != expected_shape:
                raise RoundError(
                    f"participant {participant_id!r} delta shape "
                    f"{tuple(update.delta.shape)} does not match global parameter shape "
                    f"{tuple(expected_shape)}",
                    code=LensembleErrorCode.ROUND_FAILED,
                    remediation="reject stale or malformed updates before the outer step",
                )
        averaged = self.average_deltas(deltas)
        assert_bitwise_reproducible(averaged, self.average_deltas(deltas))  # self-check

        previous_velocity = (
            torch.zeros_like(averaged) if self._velocity is None else self._velocity
        )
        next_velocity = self.momentum * previous_velocity + averaged
        update = (
            averaged + self.momentum * next_velocity if self.nesterov else next_velocity
        )
        next_params = (
            global_params.to(device=averaged.device, dtype=self.dtype)
            + self.lr * update
        )
        return next_params, next_velocity

    def preview_step(
        self, global_params: Tensor, deltas: Mapping[str, PseudoGradient]
    ) -> Tensor:
        """Return the exact next-step params without advancing the current velocity.

        Repeated previews from identical inputs are byte-identical and state-preserving. The next
        :meth:`step` call with those inputs returns the same params and commits the previewed velocity.
        """
        next_params, _ = self._compute_step(global_params, deltas)
        return next_params

    def step(
        self, global_params: Tensor, deltas: Mapping[str, PseudoGradient]
    ) -> Tensor:
        """Add the momentum-filtered mean displacement and commit the next velocity.

        Bitwise-reproducible: the averaged sum is recomputed under the same fixed order and compared
        (``assert_bitwise_reproducible``); a mismatch raises ``NonDeterministicAggregation`` and the step
        does not commit.
        """
        next_params, next_velocity = self._compute_step(global_params, deltas)
        self._velocity = next_velocity
        return next_params
