"""Pairwise-mask secure aggregation (RFC-0011 §2/§4, Backend A; #47).

Mask-cancellation correctness (exact integer recovery), order-independence (INV-AGG-DETERMINISM), dropout
robustness via Shamir recovery, the below-threshold fail-closed posture (no partial sum), the
double-masking rule (exactly one seed kind per participant), and the Shamir round-trip. The aggregator
returns only the sum — never an individual delta (the no-leak property is in tests/property/).
"""

from __future__ import annotations

import pytest
import torch

from lensemble.aggregation import (
    DropoutRecovery,
    PairwiseMaskAggregator,
    build_masked_update,
    dh_keypair,
    shamir_reconstruct,
    shamir_split,
)
from lensemble.aggregation.secure_agg import FieldParams
from lensemble.errors import LensembleErrorCode, SecureAggregationError

_DIM = 6
_FIELD = FieldParams(modulus=2**31 - 1, scale=1e6, dim=_DIM)
_ROOT = b"\x00" * 32


def _setup(pids, *, seed=0):
    """Build each participant's MaskedUpdate + the round keys/secrets for a C-participant round."""
    g = torch.Generator().manual_seed(seed)
    deltas = {p: torch.randn(_DIM, generator=g) * 0.5 for p in pids}
    keys = {p: dh_keypair(1000 + i) for i, p in enumerate(pids)}
    self_secrets = {p: 5000 + i for i, p in enumerate(pids)}
    pubs = {p: keys[p][1] for p in pids}
    updates = {
        p: build_masked_update(
            deltas[p],
            participant_id=p,
            round_index=0,
            field=_FIELD,
            secret_key=keys[p][0],
            self_mask_secret=self_secrets[p],
            peer_public_keys={q: pubs[q] for q in pids if q != p},
            dataset_root=_ROOT,
        )
        for p in pids
    }
    return deltas, keys, self_secrets, pubs, updates


def _decoded_sum(deltas, over) -> torch.Tensor:
    total = torch.zeros(_DIM)
    for p in over:
        total = total + torch.round(deltas[p].to(torch.float32) * _FIELD.scale)
    return total / _FIELD.scale


# --- mask cancellation (all present) ---


def test_mask_cancellation_recovers_plaintext_sum() -> None:
    pids = ["p0", "p1", "p2", "p3"]
    deltas, keys, selfs, pubs, updates = _setup(pids)
    t = 3
    recovery = DropoutRecovery(
        threshold=t,
        self_mask_shares={
            p: shamir_split(selfs[p], num_shares=len(pids), threshold=t) for p in pids
        },
        pairwise_shares={},
    )
    got = PairwiseMaskAggregator(pubs).aggregate(
        updates, field=_FIELD, round_index=0, threshold=t, recovery=recovery
    )
    assert torch.equal(got, _decoded_sum(deltas, pids))  # exact integer recovery


# --- order-independence (INV-AGG-DETERMINISM) ---


def test_reconstruction_is_order_independent() -> None:
    pids = ["p0", "p1", "p2", "p3"]
    deltas, keys, selfs, pubs, updates = _setup(pids)
    t = 3
    recovery = DropoutRecovery(
        threshold=t,
        self_mask_shares={
            p: shamir_split(selfs[p], num_shares=len(pids), threshold=t) for p in pids
        },
        pairwise_shares={},
    )
    agg = PairwiseMaskAggregator(pubs)
    base = agg.aggregate(
        updates, field=_FIELD, round_index=0, threshold=t, recovery=recovery
    )
    shuffled = {p: updates[p] for p in reversed(pids)}  # different insertion order
    assert torch.equal(
        base,
        agg.aggregate(
            shuffled, field=_FIELD, round_index=0, threshold=t, recovery=recovery
        ),
    )


# --- dropout robustness ---


def test_dropout_recovery_recovers_survivor_sum() -> None:
    pids = ["p0", "p1", "p2", "p3"]
    deltas, keys, selfs, pubs, updates = _setup(pids)
    t = 3
    dropped = "p1"
    present = {p: updates[p] for p in pids if p != dropped}
    recovery = DropoutRecovery(
        threshold=t,
        self_mask_shares={
            p: shamir_split(selfs[p], num_shares=len(pids), threshold=t)
            for p in present
        },
        pairwise_shares={
            dropped: shamir_split(keys[dropped][0], num_shares=len(pids), threshold=t)
        },
    )
    got = PairwiseMaskAggregator(pubs).aggregate(
        present, field=_FIELD, round_index=0, threshold=t, recovery=recovery
    )
    assert torch.equal(got, _decoded_sum(deltas, present))


def test_double_masking_rule_one_seed_kind_per_participant() -> None:
    # a survivor appears only in self_mask_shares; a dropout only in pairwise_shares — never both.
    pids = ["p0", "p1", "p2", "p3"]
    _, keys, selfs, _, updates = _setup(pids)
    t = 3
    dropped = {"p1"}
    survivors = [p for p in pids if p not in dropped]
    recovery = DropoutRecovery(
        threshold=t,
        self_mask_shares={
            p: shamir_split(selfs[p], num_shares=len(pids), threshold=t)
            for p in survivors
        },
        pairwise_shares={
            d: shamir_split(keys[d][0], num_shares=len(pids), threshold=t)
            for d in dropped
        },
    )
    assert set(recovery.self_mask_shares) & set(recovery.pairwise_shares) == set()
    assert set(recovery.self_mask_shares) == set(survivors)
    assert set(recovery.pairwise_shares) == dropped


# --- below-threshold: fail closed, no partial sum ---


def test_below_threshold_raises_and_returns_no_sum() -> None:
    pids = ["p0", "p1", "p2", "p3"]
    _, keys, selfs, pubs, updates = _setup(pids)
    t = 3
    present = {p: updates[p] for p in ("p0", "p2")}  # only 2 survivors < t_agg = 3
    recovery = DropoutRecovery(
        threshold=t,
        self_mask_shares={
            p: shamir_split(selfs[p], num_shares=len(pids), threshold=t)
            for p in present
        },
        pairwise_shares={},
    )
    with pytest.raises(SecureAggregationError) as exc:
        PairwiseMaskAggregator(pubs).aggregate(
            present, field=_FIELD, round_index=7, threshold=t, recovery=recovery
        )
    assert exc.value.code == LensembleErrorCode.SECURE_AGG_FAILED
    assert exc.value.present == 2  # type: ignore[attr-defined]
    assert exc.value.threshold == 3  # type: ignore[attr-defined]
    assert exc.value.round == 7  # type: ignore[attr-defined]
    assert exc.value.cause == "below_threshold"  # type: ignore[attr-defined]


# --- Shamir round-trip ---


def test_shamir_split_reconstructs_with_threshold_shares() -> None:
    secret = 1234567890123
    shares = shamir_split(secret, num_shares=5, threshold=3)
    assert shamir_reconstruct(shares[:3]) == secret  # any 3 of 5
    assert shamir_reconstruct(shares[1:4]) == secret  # a different 3
    assert shamir_reconstruct(shares) == secret  # all 5


def test_shamir_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        shamir_split(7, num_shares=3, threshold=5)  # threshold > num_shares
    shares = shamir_split(7, num_shares=3, threshold=2)
    with pytest.raises(ValueError):
        shamir_reconstruct([shares[0], shares[0]])  # duplicate x-coordinate
