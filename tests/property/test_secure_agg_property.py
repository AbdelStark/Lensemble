"""No-individual-leak property for pairwise-mask secure aggregation (RFC-0011 §7; #47 / 07 §2.7).

For random delta sets and participant counts: the full surviving set reconstructs the plaintext sum
exactly, but no single masked update reveals its delta (the mask hides it), and an incomplete recovery
(one survivor's self-mask share missing) does NOT reveal the sum — so no combination short of the full
surviving set, with the full recovery, isolates an individual ``encode(Δ_c)`` (``INV-RESIDENCY``).
"""

from __future__ import annotations

import torch
from hypothesis import given
from hypothesis import strategies as st

from lensemble.aggregation import (
    DropoutRecovery,
    PairwiseMaskAggregator,
    build_masked_update,
    dh_keypair,
    shamir_split,
)
from lensemble.aggregation.secure_agg import FieldParams, _lift_to_signed

_DIM = 5
_FIELD = FieldParams(modulus=2**31 - 1, scale=1e5, dim=_DIM)
_ROOT = b"\x00" * 32


def _plaintext_sum(deltas, pids, scale) -> torch.Tensor:
    total = torch.zeros(_DIM)
    for p in pids:
        total = total + torch.round(deltas[p].float() * scale)
    return total / scale


def _round(num_participants: int, seed: int):
    pids = [f"p{i}" for i in range(num_participants)]
    g = torch.Generator().manual_seed(seed)
    deltas = {p: torch.randn(_DIM, generator=g) * 0.3 for p in pids}
    keys = {p: dh_keypair(1 + i + seed) for i, p in enumerate(pids)}
    selfs = {p: 10_000 + i + seed for i, p in enumerate(pids)}
    pubs = {p: keys[p][1] for p in pids}
    updates = {
        p: build_masked_update(
            deltas[p],
            participant_id=p,
            round_index=0,
            field=_FIELD,
            secret_key=keys[p][0],
            self_mask_secret=selfs[p],
            peer_public_keys={q: pubs[q] for q in pids if q != p},
            dataset_root=_ROOT,
        )
        for p in pids
    }
    return pids, deltas, selfs, pubs, updates


@given(
    num_participants=st.integers(min_value=2, max_value=6), seed=st.integers(0, 10_000)
)
def test_full_set_recovers_sum_but_no_single_update_leaks(
    num_participants, seed
) -> None:
    pids, deltas, selfs, pubs, updates = _round(num_participants, seed)
    t = num_participants  # full set
    recovery = DropoutRecovery(
        threshold=t,
        self_mask_shares={
            p: shamir_split(selfs[p], num_shares=num_participants, threshold=t)
            for p in pids
        },
        pairwise_shares={},
    )
    got = PairwiseMaskAggregator(pubs).aggregate(
        updates, field=_FIELD, round_index=0, threshold=t, recovery=recovery
    )
    expected = _plaintext_sum(deltas, pids, _FIELD.scale)
    assert torch.equal(
        got, expected
    )  # the full set reconstructs the plaintext sum exactly

    # ...but no single masked update, decoded naively (no mask removal), reveals its delta.
    for p in pids:
        naive = (
            _lift_to_signed(updates[p].masked, _FIELD.modulus).float() / _FIELD.scale
        )
        assert not torch.allclose(naive, deltas[p], atol=1e-2)  # the mask hides Δ_c


@given(seed=st.integers(0, 10_000))
def test_incomplete_recovery_does_not_reveal_the_sum(seed) -> None:
    # dropping one survivor's self-mask share from the recovery leaves its b_c in the result -> garbage,
    # so the plaintext sum is not revealed without the full surviving set's recovery data.
    pids, deltas, selfs, pubs, updates = _round(4, seed)
    t = 4
    full = {p: shamir_split(selfs[p], num_shares=4, threshold=t) for p in pids}
    plaintext = _plaintext_sum(deltas, pids, _FIELD.scale)
    agg = PairwiseMaskAggregator(pubs)
    # the full recovery reveals the sum
    assert torch.equal(
        agg.aggregate(
            updates,
            field=_FIELD,
            round_index=0,
            threshold=t,
            recovery=DropoutRecovery(
                threshold=t, self_mask_shares=full, pairwise_shares={}
            ),
        ),
        plaintext,
    )
    # a recovery missing one participant's self-mask seed (replaced by a wrong seed) does NOT
    missing = dict(full)
    missing["p0"] = shamir_split(
        selfs["p0"] + 1, num_shares=4, threshold=t
    )  # wrong seed
    wrong = agg.aggregate(
        updates,
        field=_FIELD,
        round_index=0,
        threshold=t,
        recovery=DropoutRecovery(
            threshold=t, self_mask_shares=missing, pairwise_shares={}
        ),
    )
    assert not torch.allclose(wrong, plaintext, atol=1e-2)
