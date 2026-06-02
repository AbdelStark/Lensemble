"""Property test: the residency guard refuses any embedded tensor (07-testing-strategy 2.7). Issue #23."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from lensemble.data.residency import guard_egress
from lensemble.errors import ResidencyViolation

# "Safe" payloads: scalars / hashes / shapes nested in dicts and lists — no tensors.
_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=12),
    st.binary(max_size=8),
)
_safe = st.recursive(
    _scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(min_size=1, max_size=6), children, max_size=4),
    ),
    max_leaves=12,
)


@given(payload=_safe)
def test_scalar_payloads_pass(payload: object) -> None:
    # No tensors anywhere => permitted to cross.
    assert guard_egress(payload) is None


@given(
    prefix=st.lists(st.text(min_size=1, max_size=4), min_size=0, max_size=3),
)
def test_tensor_at_any_depth_is_rejected(prefix: list[str]) -> None:
    # Bury a raw tensor under `prefix` nested dicts; the guard must still refuse it (fail-closed).
    payload: object = torch.zeros(3)
    for key in prefix:
        payload = {key: payload}
    with pytest.raises(ResidencyViolation):
        guard_egress(payload)
