"""Simulated secure aggregation: exact reveal + order-independence (RFC-0011 1/6, 07 §2.5). Issue #46.

C participants field-encode their deltas; the aggregator reveals sum_c Delta_c to within the fixed-point
scale, bit-identically regardless of submission order (INV-AGG-DETERMINISM), and refuses below threshold.
"""

from __future__ import annotations

import pytest
import torch

from lensemble.aggregation import (
    FieldParams,
    SimulatedSecureAggregator,
    assert_no_wrap,
    encode_delta,
)
from lensemble.errors import SecureAggregationError

_DIM = 8
_FIELD = FieldParams(modulus=2**32, scale=2.0**16, dim=_DIM)


def _deltas(c: int = 4) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {f"p{i}": torch.randn(_DIM, generator=g) for i in range(c)}


def _updates(deltas: dict[str, torch.Tensor]) -> dict:
    return {
        pid: encode_delta(
            d, _FIELD, participant_id=pid, round_index=0, dataset_root=b"\x00" * 32
        )
        for pid, d in deltas.items()
    }


def test_revealed_sum_equals_plaintext_within_scale() -> None:
    deltas = _deltas(4)
    plaintext = torch.stack(list(deltas.values())).sum(dim=0)
    revealed = SimulatedSecureAggregator().aggregate(
        _updates(deltas), field=_FIELD, round_index=0, threshold=3
    )
    # exact to the fixed-point scale (each round() costs <= 0.5/scale per participant)
    assert torch.allclose(revealed, plaintext, atol=len(deltas) / _FIELD.scale)
    assert revealed.dtype == torch.float32


def test_reveal_is_order_independent_and_bit_identical() -> None:
    deltas = _deltas(5)
    updates = _updates(deltas)
    agg = SimulatedSecureAggregator()
    baseline = agg.aggregate(updates, field=_FIELD, round_index=0, threshold=3)
    for permutation in ([*reversed(updates)], sorted(updates), list(updates)):
        shuffled = {pid: updates[pid] for pid in permutation}
        assert torch.equal(
            agg.aggregate(shuffled, field=_FIELD, round_index=0, threshold=3), baseline
        )  # INV-AGG-DETERMINISM


def test_recovers_signed_sum_both_directions() -> None:
    # a sum with both positive and negative coordinates exercises the lift-to-signed decode
    deltas = {"a": torch.full((_DIM,), 0.5), "b": torch.full((_DIM,), -0.9)}
    revealed = SimulatedSecureAggregator().aggregate(
        _updates(deltas), field=_FIELD, round_index=0, threshold=2
    )
    assert torch.allclose(revealed, torch.full((_DIM,), -0.4), atol=2 / _FIELD.scale)


def test_below_threshold_refuses_partial_sum() -> None:
    updates = _updates(_deltas(2))
    with pytest.raises(SecureAggregationError):
        SimulatedSecureAggregator().aggregate(
            updates, field=_FIELD, round_index=0, threshold=3
        )


def test_no_wrap_assertion() -> None:
    assert_no_wrap(
        num_participants=4, clip_norm=1.5, field=_FIELD
    )  # comfortably sized -> no raise
    tiny = FieldParams(modulus=2**8, scale=2.0**16, dim=_DIM)
    with pytest.raises(SecureAggregationError):
        assert_no_wrap(
            num_participants=4, clip_norm=1.5, field=tiny
        )  # modulus too small -> wrap risk
