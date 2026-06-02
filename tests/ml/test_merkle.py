"""Dataset Merkle tree, root R_c, and inclusion proofs (RFC-0014 §3, §5; #28).

Exact-byte tests, no numerical tolerance. Roots are checked against roots hand-computed inline from the
``_h`` primitive (sizes 1-5 exercise odd-node promotion); inclusion proofs round-trip for every leaf
including one whose path crosses a promoted odd node; the ``False`` (no match) vs raise (malformed)
contract is pinned. Provenance tests live under ``tests/ml/`` so the CI gate (07 §8.4) runs them.
"""

from __future__ import annotations

import hashlib
import itertools

import pytest

from lensemble.errors import MerkleVerificationError, ProvenanceError
from lensemble.provenance.merkle import (
    DIGEST_SIZE,
    CommitmentScheme,
    HashDomain,
    MerkleProof,
    _h,
    merkle_root,
    prove_inclusion,
    verify_inclusion,
)

_SCHEME = CommitmentScheme()


def _leaf(i: int) -> bytes:
    """A deterministic, distinct 32-byte LEAF-domain stand-in digest."""
    return hashlib.sha256(f"leaf-{i}".encode()).digest()


def _node(left: bytes, right: bytes) -> bytes:
    return _h(HashDomain.NODE, left + right)


def _root(node: bytes) -> bytes:
    return _h(HashDomain.ROOT, node)


# --- root correctness against inline hand-computed roots (sizes 1-5, then 8) ---


def test_root_size_1() -> None:
    a = _leaf(0)
    assert merkle_root([a]) == _root(a)


def test_root_size_2() -> None:
    a, b = sorted([_leaf(0), _leaf(1)])
    assert merkle_root([_leaf(0), _leaf(1)]) == _root(_node(a, b))


def test_root_size_3_promotes_odd_leaf() -> None:
    a, b, c = sorted(_leaf(i) for i in range(3))
    # level0 [a,b,c]: pair (a,b)->n01, promote c. level1 [n01,c]: pair -> n. root = ROOT(n).
    n01 = _node(a, b)
    assert merkle_root([_leaf(i) for i in range(3)]) == _root(_node(n01, c))


def test_root_size_4() -> None:
    a, b, c, d = sorted(_leaf(i) for i in range(4))
    top = _node(_node(a, b), _node(c, d))
    assert merkle_root([_leaf(i) for i in range(4)]) == _root(top)


def test_root_size_5_double_promotion() -> None:
    a, b, c, d, e = sorted(_leaf(i) for i in range(5))
    # level0: (a,b)->p0, (c,d)->p1, promote e. level1 [p0,p1,e]: (p0,p1)->q0, promote e.
    # level2 [q0,e]: -> r. root = ROOT(r).
    p0, p1 = _node(a, b), _node(c, d)
    q0 = _node(p0, p1)
    assert merkle_root([_leaf(i) for i in range(5)]) == _root(_node(q0, e))


def test_root_size_8() -> None:
    leaves = sorted(_leaf(i) for i in range(8))
    lvl1 = [_node(leaves[i], leaves[i + 1]) for i in range(0, 8, 2)]
    lvl2 = [_node(lvl1[0], lvl1[1]), _node(lvl1[2], lvl1[3])]
    top = _node(lvl2[0], lvl2[1])
    assert merkle_root([_leaf(i) for i in range(8)]) == _root(top)
    assert len(merkle_root([_leaf(i) for i in range(8)])) == DIGEST_SIZE


# --- empty / malformed input ---


def test_empty_dataset_raises_provenance_error() -> None:
    with pytest.raises(ProvenanceError):
        merkle_root([])


def test_non_digest_sized_leaf_raises() -> None:
    with pytest.raises(ProvenanceError):
        merkle_root([_leaf(0), b"too-short"])


# --- leaf-sorting property: enumeration order does not change R_c ---


def test_root_is_order_independent() -> None:
    leaves = [_leaf(i) for i in range(4)]
    base = merkle_root(leaves)
    for perm in itertools.permutations(leaves):
        assert merkle_root(list(perm)) == base


def test_duplicate_leaves_change_the_root() -> None:
    # A set with a duplicate episode is a different multiset, so a different root.
    one = merkle_root([_leaf(0), _leaf(1)])
    dup = merkle_root([_leaf(0), _leaf(0), _leaf(1)])
    assert one != dup


# --- inclusion proofs: round-trip for every leaf, every size ---


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8])
def test_inclusion_roundtrip_every_leaf(n: int) -> None:
    leaves = [_leaf(i) for i in range(n)]
    root = merkle_root(leaves)
    for idx in range(n):
        proof = prove_inclusion(leaves, idx)
        assert proof.leaf == leaves[idx]
        assert verify_inclusion(proof, root, _SCHEME) is True


def test_inclusion_proof_crosses_promoted_odd_node() -> None:
    # In a 3-leaf tree the lexicographically-largest leaf is promoted at level 0 (no sibling there),
    # so its proof carries exactly one sibling — the level-1 internal node.
    leaves = [_leaf(i) for i in range(3)]
    root = merkle_root(leaves)
    largest = max(range(3), key=lambda i: leaves[i])
    proof = prove_inclusion(leaves, largest)
    assert len(proof.siblings) == 1
    assert verify_inclusion(proof, root, _SCHEME) is True


def test_prove_inclusion_disambiguates_duplicate_leaves() -> None:
    leaves = [_leaf(0), _leaf(0), _leaf(1)]
    root = merkle_root(leaves)
    for idx in range(3):
        assert verify_inclusion(prove_inclusion(leaves, idx), root, _SCHEME) is True


def test_prove_inclusion_rejects_out_of_range_index() -> None:
    with pytest.raises(ProvenanceError):
        prove_inclusion([_leaf(0), _leaf(1)], 5)


# --- well-formed but non-matching proof returns False (no raise) ---


def test_wrong_root_returns_false() -> None:
    leaves = [_leaf(i) for i in range(4)]
    proof = prove_inclusion(leaves, 0)
    assert verify_inclusion(proof, _leaf(99), _SCHEME) is False


def test_flipped_path_bit_returns_false() -> None:
    leaves = [_leaf(i) for i in range(4)]
    root = merkle_root(leaves)
    proof = prove_inclusion(leaves, 0)
    flipped = MerkleProof(
        leaf=proof.leaf,
        siblings=proof.siblings,
        path_bits=tuple(not b for b in proof.path_bits),
    )
    assert verify_inclusion(flipped, root, _SCHEME) is False


def test_corrupted_sibling_returns_false() -> None:
    leaves = [_leaf(i) for i in range(4)]
    root = merkle_root(leaves)
    proof = prove_inclusion(leaves, 0)
    corrupted = MerkleProof(
        leaf=proof.leaf,
        siblings=(_leaf(123),) + proof.siblings[1:],
        path_bits=proof.path_bits,
    )
    assert verify_inclusion(corrupted, root, _SCHEME) is False


# --- structurally malformed proof / scheme raises MerkleVerificationError ---


def test_mismatched_sibling_bit_lengths_raise() -> None:
    bad = MerkleProof(leaf=_leaf(0), siblings=(_leaf(1),), path_bits=())
    with pytest.raises(MerkleVerificationError):
        verify_inclusion(bad, _leaf(0), _SCHEME)


def test_non_digest_sized_proof_element_raises() -> None:
    bad = MerkleProof(leaf=b"short", siblings=(), path_bits=())
    with pytest.raises(MerkleVerificationError):
        verify_inclusion(bad, _leaf(0), _SCHEME)


def test_unsupported_scheme_raises() -> None:
    leaves = [_leaf(i) for i in range(2)]
    root = merkle_root(leaves)
    proof = prove_inclusion(leaves, 0)
    with pytest.raises(MerkleVerificationError):
        verify_inclusion(proof, root, CommitmentScheme(hash_name="poseidon2"))
