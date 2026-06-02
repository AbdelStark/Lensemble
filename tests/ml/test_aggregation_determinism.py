"""Outer-step aggregation determinism — INV-AGG-DETERMINISM (RFC-0003 7 / 07 §2.5). Issue #39.

The Nesterov outer step is a bitwise-reproducible function of (deltas, prior global params): two
in-process runs and a fresh subprocess agree byte-for-byte; the fixed participant-id-sorted reduction is
input-order-independent; a non-bitwise recomputation raises NonDeterministicAggregation; Nesterov is
stable across a varying participant count C.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from lensemble.errors import LensembleErrorCode, NonDeterministicAggregation
from lensemble.federation import (
    OuterOptimizer,
    assert_bitwise_reproducible,
    build_pseudogradient,
)
from lensemble.federation.outer import _content_hash

# A delta-build reproducible in this process and in a subprocess (seeded, fixed shapes/order).
_BUILD = """
import torch, hashlib
from lensemble.federation import OuterOptimizer, build_pseudogradient

def deltas(c):
    g = torch.Generator().manual_seed(0)
    return {
        f"p{i}": build_pseudogradient(
            {"encoder.w": torch.randn(6, generator=g), "predictor.w": torch.randn(4, generator=g)},
            dataset_root=bytes([i]) * 32, round_index=0,
        )
        for i in range(c)
    }

res = OuterOptimizer(lr=0.7, momentum=0.9).step(torch.zeros(10), deltas(3))
arr = res.detach().cpu().numpy()
print(hashlib.sha256(arr.astype(arr.dtype.newbyteorder("<")).tobytes()).hexdigest())
"""


def _deltas(c: int = 3) -> dict:
    g = torch.Generator().manual_seed(0)
    return {
        f"p{i}": build_pseudogradient(
            {
                "encoder.w": torch.randn(6, generator=g),
                "predictor.w": torch.randn(4, generator=g),
            },
            dataset_root=bytes([i]) * 32,
            round_index=0,
        )
        for i in range(c)
    }


def test_outer_step_is_bitwise_reproducible() -> None:
    g0 = torch.zeros(10)
    first = OuterOptimizer(lr=0.7, momentum=0.9).step(g0, _deltas())
    second = OuterOptimizer(lr=0.7, momentum=0.9).step(g0, _deltas())
    assert torch.equal(first, second)
    assert _content_hash(first) == _content_hash(second)

    # ...and a fresh subprocess reproduces the same content hash (no process-dependent state)
    out = subprocess.run(
        [sys.executable, "-c", _BUILD], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == _content_hash(first)


def test_fixed_reduction_order_is_input_order_independent() -> None:
    g0 = torch.zeros(10)
    deltas = _deltas()
    sorted_result = OuterOptimizer(lr=0.7, momentum=0.9).step(g0, deltas)
    shuffled = {
        k: deltas[k] for k in reversed(list(deltas))
    }  # different insertion order
    shuffled_result = OuterOptimizer(lr=0.7, momentum=0.9).step(g0, shuffled)
    assert torch.equal(
        sorted_result, shuffled_result
    )  # participant-id-sorted reduction


def test_nondeterministic_reduction_raises() -> None:
    a = torch.zeros(4)
    b = torch.zeros(4)
    b[0] = 1e-7  # a recomputation that does not reproduce byte-for-byte
    with pytest.raises(NonDeterministicAggregation) as exc:
        assert_bitwise_reproducible(a, b)
    assert exc.value.code == LensembleErrorCode.AGG_NONDETERMINISTIC
    assert exc.value.expected_hash != exc.value.got_hash  # type: ignore[attr-defined]


def test_nesterov_stable_across_varying_participant_count() -> None:
    g0 = torch.zeros(10)
    for c in (2, 3, 4, 8):
        result = OuterOptimizer(lr=0.7, momentum=0.9).step(g0, _deltas(c))
        assert result.shape == g0.shape
        assert bool(
            torch.isfinite(result).all()
        )  # an outer step proceeds with whatever C is present


def test_empty_deltas_rejected() -> None:
    with pytest.raises(NonDeterministicAggregation):
        OuterOptimizer(lr=0.7).step(torch.zeros(10), {})


def test_secure_agg_nondeterministic_field_sum_raises() -> None:
    # the secure-aggregation self-check aborts (never silently averages) on a non-reproducing field sum
    from lensemble.aggregation import assert_field_sum_reproducible

    a = torch.zeros(4, dtype=torch.int64)
    b = a.clone()
    b[0] = 1
    with pytest.raises(NonDeterministicAggregation) as exc:
        assert_field_sum_reproducible(a, b)
    assert exc.value.code == LensembleErrorCode.AGG_NONDETERMINISTIC
