"""lensemble.aggregation.secure_agg — in-process simulated secure aggregation (RFC-0011 1/6, Stage B).

A single-process secure-sum harness: ``C`` simulated participants submit field-encoded updates and the
aggregator reconstructs the deterministic plaintext ``sum_c Delta_c`` with no network and no cryptographic
masking. Its job is to validate the integer fixed-point encode/decode, the no-wrap modulus sizing, and the
determinism self-check before the pairwise-mask backend lands; the ``FieldParams``/``MaskedUpdate`` contract
is stable from v0.2 (Stage C swaps the transport, not the contract).

Security posture: :meth:`SimulatedSecureAggregator.aggregate` returns ONLY the fp32 sum and never
materializes, stores, or returns an individual ``Delta_c`` (``INV-RESIDENCY``). The integer field makes
addition associative, so the revealed sum is order-independent and bitwise-reproducible
(``INV-AGG-DETERMINISM``, RFC-0011 3); a determinism self-check re-derives the sum in the fixed coordinate
order and raises :class:`~lensemble.errors.NonDeterministicAggregation` on any mismatch (never swallowed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lensemble.errors import (
    LensembleErrorCode,
    NonDeterministicAggregation,
    SecureAggregationError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class FieldParams:
    """Modular integer field the masks (would) cancel over (RFC-0011 1/3).

    ``modulus`` is sized so ``C * max|encoded delta| < modulus/2`` (no wrap on the sum); ``scale`` is the
    fixed-point scale (``encoded = round(value * scale)``); ``dim`` is the flat ``(theta, phi)`` length.
    """

    modulus: int
    scale: float
    dim: int


@dataclass(frozen=True)
class MaskedUpdate:
    """Participant ``c``'s encoded pseudo-gradient (RFC-0011 1). Carries no recoverable ``Delta_c`` alone.

    In this simulated harness ``masked`` is ``encode(Delta_c)`` with no mask (the sum is revealed directly);
    the field is the same one the pairwise-mask backend will cancel over.
    """

    participant_id: str
    round_index: int
    masked: Tensor  # int64, shape (dim,), values in [0, modulus)
    dataset_root: bytes


def encode_delta(
    delta: Tensor,
    field: FieldParams,
    *,
    participant_id: str,
    round_index: int,
    dataset_root: bytes,
) -> MaskedUpdate:
    """Fixed-point encode a pseudo-gradient into the integer field, lifted to ``[0, modulus)``."""
    quantized = torch.round(delta.to(torch.float32) * field.scale).to(torch.int64)
    masked = quantized.remainder(
        field.modulus
    )  # lift signed -> [0, modulus) (no division by zero)
    return MaskedUpdate(
        participant_id=participant_id,
        round_index=round_index,
        masked=masked,
        dataset_root=dataset_root,
    )


def _lift_to_signed(field_elements: Tensor, modulus: int) -> Tensor:
    """Recentre field elements ``[0, modulus)`` to signed integers ``(-modulus/2, modulus/2]``."""
    half = modulus // 2
    return torch.where(field_elements > half, field_elements - modulus, field_elements)


def assert_no_wrap(num_participants: int, clip_norm: float, field: FieldParams) -> None:
    """Reject a modulus too small for ``C`` participants at ``C_clip`` (the field sum could wrap)."""
    max_encoded = round(clip_norm * field.scale)
    if num_participants * max_encoded >= field.modulus // 2:
        raise SecureAggregationError(
            f"field too small: {num_participants} * {max_encoded} >= modulus/2 "
            f"({field.modulus // 2}); the encoded sum could wrap",
            code=LensembleErrorCode.SECURE_AGG_FAILED,
            remediation="increase modulus or decrease scale so C * round(C_clip*scale) < modulus/2",
        )


def assert_field_sum_reproducible(first: Tensor, second: Tensor) -> None:
    """Raise :class:`NonDeterministicAggregation` unless two field-sum derivations are byte-identical."""
    if not torch.equal(first, second):
        raise NonDeterministicAggregation(
            "secure-aggregation field sum was not reproducible; aborting (INV-AGG-DETERMINISM)",
            code=LensembleErrorCode.AGG_NONDETERMINISTIC,
            remediation="reduce in a fixed coordinate order over the integer field; no float reductions",
        )


def _sum_mod(elements: Sequence[Tensor], modulus: int) -> Tensor:
    """Modular sum of integer field vectors in the given fixed order (associative, order-independent)."""
    accumulator = torch.zeros_like(elements[0])
    for element in elements:
        accumulator = (accumulator + element).remainder(modulus)
    return accumulator


class SimulatedSecureAggregator:
    """In-process secure-sum aggregator (RFC-0011 6, v0.2). Reveals the sum directly (no masks)."""

    def aggregate(
        self,
        updates: Mapping[str, MaskedUpdate],
        *,
        field: FieldParams,
        round_index: int,
        threshold: int,
        recovery: object | None = None,
    ) -> Tensor:
        """Reconstruct the fp32 plaintext ``sum_c Delta_c`` over the surviving set (RFC-0011 1).

        Below ``threshold`` survivors raises :class:`SecureAggregationError` and returns no partial sum.
        ``recovery`` (dropout secret-sharing) is unused in the simulated harness — it has no masks.
        Never materializes or returns an individual ``Delta_c`` (``INV-RESIDENCY``).
        """
        if len(updates) < threshold:
            raise SecureAggregationError(
                f"only {len(updates)} survivors < threshold {threshold}; refusing a partial sum",
                code=LensembleErrorCode.SECURE_AGG_FAILED,
                remediation="wait for the secure-aggregation threshold of survivors, or abort the round",
            )
        ordered = [
            updates[pid].masked for pid in sorted(updates)
        ]  # fixed participant/coordinate order
        total = _sum_mod(ordered, field.modulus)
        assert_field_sum_reproducible(
            total, _sum_mod(ordered, field.modulus)
        )  # determinism self-check
        signed = _lift_to_signed(total, field.modulus)
        return signed.to(torch.float32) / field.scale
