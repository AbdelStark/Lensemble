"""Observability redaction guard (RFC-0015 5 / 07 2.7). Issue #59. Security-critical (INV-RESIDENCY)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from lensemble.errors import LensembleErrorCode, ResidencyViolation
from lensemble.observability import redact, redact_record


def _emit(record: dict, sink: list) -> None:
    """Stub sink-write: redact all-or-nothing, then append. A reject leaves the sink unchanged."""
    sink.append(redact_record(record))


def test_emittable_scalars_pass() -> None:
    assert redact(True, field="flag") is True
    assert redact(7, field="count") == 7
    assert redact(0.5, field="loss") == 0.5
    assert redact("bfloat16", field="dtype") == "bfloat16"
    assert redact((4, 8), field="shape") == (4, 8)  # shape tuple
    assert redact(b"a" * 64, field="hash") == b"a" * 64  # hex digest bytes


def test_tensor_and_ndarray_rejected() -> None:
    for bad in (torch.zeros(4, 8), np.zeros((4, 8), dtype=np.float32)):
        with pytest.raises(ResidencyViolation) as exc:
            redact(bad, field="embedding")
        assert exc.value.code == LensembleErrorCode.RESIDENCY_VIOLATION
        assert exc.value.remediation


def test_non_finite_and_non_hex_bytes_rejected() -> None:
    with pytest.raises(ResidencyViolation):
        redact(float("nan"), field="loss")
    with pytest.raises(ResidencyViolation):
        redact(float("inf"), field="loss")
    with pytest.raises(ResidencyViolation):
        redact(b"\x00\x01\x02\x03", field="blob")  # raw bytes, not a hex digest


def test_record_fail_closed_no_partial_write() -> None:
    sink: list = []
    with pytest.raises(ResidencyViolation):
        _emit({"loss/pred": 0.1, "embedding": torch.randn(4, 8)}, sink)
    assert sink == []  # nothing written — fail-closed, no partial record
    _emit({"loss/pred": 0.1, "gauge/effective_dim": 7, "shape": (4, 8)}, sink)
    assert sink == [{"loss/pred": 0.1, "gauge/effective_dim": 7, "shape": (4, 8)}]


def test_residency_violation_not_swallowed() -> None:
    # the guard re-raises; a caller using it cannot silently drop the violation
    raised = False
    try:
        redact_record({"x": torch.ones(2)})
    except ResidencyViolation:
        raised = True
    assert raised


_scalars = st.one_of(
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=8),
)
_safe = st.recursive(
    _scalars,
    lambda c: st.one_of(
        st.lists(c, max_size=4),
        st.dictionaries(st.text(min_size=1, max_size=5), c, max_size=4),
    ),
    max_leaves=10,
)


@given(payload=_safe)
def test_property_all_emittable_passes(payload: object) -> None:
    redact(payload, field="root")  # no tensors anywhere => emittable


@given(prefix=st.lists(st.text(min_size=1, max_size=4), max_size=3))
def test_property_tensor_leaf_rejects_whole_record(prefix: list[str]) -> None:
    payload: object = torch.zeros(2, 2)
    for key in prefix:
        payload = {key: payload}
    with pytest.raises(ResidencyViolation):
        redact(payload, field="root")
