"""lensemble.aggregation — secure-sum aggregation backends (docs/rfcs/RFC-0011).

The aggregator returns only the plaintext ``sum_c Delta_c`` and never an individual ``Delta_c``
(``INV-RESIDENCY``); the integer field makes the revealed sum order-independent (``INV-AGG-DETERMINISM``).
"""

from __future__ import annotations

from lensemble.aggregation.secure_agg import (
    FieldParams,
    MaskedUpdate,
    SimulatedSecureAggregator,
    assert_field_sum_reproducible,
    assert_no_wrap,
    encode_delta,
)

__all__ = [
    "FieldParams",
    "MaskedUpdate",
    "SimulatedSecureAggregator",
    "encode_delta",
    "assert_no_wrap",
    "assert_field_sum_reproducible",
]
