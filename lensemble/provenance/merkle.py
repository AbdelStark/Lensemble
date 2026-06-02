"""lensemble.provenance.merkle — domain-separated episode hashing (docs/rfcs/RFC-0014 §1-2).

This module is the leaf-construction layer every dataset commitment is built on. It hashes an
``Episode`` to a SHA-256 leaf via a canonical, **backend-independent** byte serialization: the same
logical episode commits to the same leaf whether it was stored as ``lance``, ``hdf5``, or read through
the ``lerobot://`` adapter (RFC-0004 §2). The hash is therefore taken over the normalized *logical*
form — fixed dtype, fixed little-endian byte order, fixed time ordering — not over raw on-disk bytes.

Domain separation (``HashDomain``) prepends a one-byte tag to every preimage so an episode digest can
never be reinterpreted as a leaf/internal-node/root digest (RFC-0014 §1); the tree builder
(``merkle_root``, ``prove_inclusion``, ``verify_inclusion``) consumes the ``LEAF``/``NODE``/``ROOT``
domains. Odd levels **promote** the trailing node unchanged rather than duplicating it — duplication is
the CVE-2012-2459 Merkle-malleability source, where distinct leaf multisets hash to the same root. The
hash is selected by a versioned :class:`CommitmentScheme` rather than hard-coded at each call site, so
the Stage-D migration to a STARK-friendly hash is a version bump, not a rewrite (RFC-0014 §1).

RISK (RFC-0014 §2): cross-platform stability rests on a fixed dtype/endianness and a fixed frame
ordering. An episode captured at ``bf16`` and an ``fp32`` re-export of "the same" trajectory hash
differently — which is correct: a re-quantized copy is a different episode, and the stored dtype is part
of the canonical bytes. The canonical form is pinned behind ``CommitmentScheme.scheme_version`` and only
changes with a version bump. This byte ordering is shared surface with the checkpoint-hash
canonicalization Open Question (RFC-0010 §Open Questions) and the artifact hash in
``lensemble.artifacts.hashing``.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from lensemble.errors import (
    LensembleErrorCode,
    MerkleVerificationError,
    ProvenanceError,
)

if TYPE_CHECKING:  # annotation-only; torch is a runtime import in the byte encoder
    from torch import Tensor

    from lensemble.contracts import ActionSpec
    from lensemble.data.episode import Episode

# --- the Phase-1 commitment hash, pinned and versioned (RFC-0014 §1) ---
COMMITMENT_HASH = "sha256"  # Phase-1 canonical; "poseidon2" reserved for Stage D
DIGEST_SIZE = 32  # bytes, SHA-256

# A self-describing tag inside the EPISODE-domain preimage. Bumping the canonical serialization (frame
# framing, metadata key set, tensor encoding) requires bumping both this tag and
# CommitmentScheme.scheme_version.
_EPISODE_CANON_TAG = b"lensemble/provenance/episode/v1\x00"


class HashDomain(IntEnum):
    """One-byte domain-separation tag prepended to every preimage (RFC-0014 §1)."""

    EPISODE = 0x00  # canonical episode bytes -> leaf preimage
    LEAF = 0x01  # leaf node in the Merkle tree
    NODE = 0x02  # internal node (concatenation of two child digests)
    ROOT = 0x03  # the committed dataset root R_c


def _h(domain: HashDomain, payload: bytes) -> bytes:
    """Domain-separated SHA-256: ``H(domain_tag || payload)`` -> 32 bytes (RFC-0014 §1)."""
    return hashlib.sha256(bytes([domain]) + payload).digest()


@dataclass(frozen=True)
class CommitmentScheme:
    """The pinned hash + tree parameters a ``DatasetCommitment`` was built under (RFC-0014 §1).

    Recorded on every commitment so a verifier reconstructs the exact scheme. ``scheme_version`` is
    bumped on any change to the hash, the tree construction, or the canonical episode serialization.
    """

    hash_name: str = (
        COMMITMENT_HASH  # "sha256" (Phase 1); "poseidon2" reserved (Stage D)
    )
    digest_size: int = DIGEST_SIZE  # bytes
    scheme_version: int = 1  # bumped on any tree/hash/serialization change


def _frame(payload: bytes) -> bytes:
    """Length-prefix a byte field so concatenation is unambiguous (8-byte little-endian length)."""
    return struct.pack("<Q", len(payload)) + payload


def _le_raw_bytes(t: "Tensor") -> bytes:
    """Raw little-endian element bytes of a tensor at its stored dtype, host-independent.

    ``bfloat16`` has no numpy dtype, so its 2-byte bit pattern is taken via an ``int16`` view (matching
    ``lensemble.artifacts.hashing``). Every other dtype is materialized through numpy with an explicit
    little-endian byte order, so the bytes are identical on a big-endian or little-endian host. The tensor
    is detached, moved to CPU, and made C-contiguous first.
    """
    import torch

    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().astype("<i2", copy=False).tobytes()
    arr = t.numpy()
    return arr.astype(arr.dtype.newbyteorder("<"), copy=False).tobytes()


def _encode_tensor(t: "Tensor") -> bytes:
    """Encode one tensor as ``(dtype string, shape tuple, C-contiguous little-endian raw bytes)``.

    The dtype string (e.g. ``"torch.uint8"``, ``"torch.float32"``) and the shape are framed before the
    raw bytes so a re-quantized or reshaped tensor serializes differently. No pickle, no platform-native
    layout.
    """
    dtype_token = str(t.dtype).encode("utf-8")
    shape = tuple(int(d) for d in t.shape)
    out = bytearray()
    out += _frame(dtype_token)
    out += struct.pack("<I", len(shape))
    for dim in shape:
        out += struct.pack("<q", dim)
    out += _frame(_le_raw_bytes(t))
    return bytes(out)


def _action_spec_digest(spec: "ActionSpec") -> str:
    """A stable content digest (hex SHA-256) of an ``ActionSpec`` over its declared fields.

    Python's builtin ``hash`` of a frozen dataclass is salted per process (string hashing), so it is not
    cross-process stable; this canonical-JSON digest is. Tuples serialize as JSON arrays; the enum
    serializes as its string value.
    """
    payload = json.dumps(
        {
            "embodiment_id": spec.embodiment_id,
            "kind": spec.kind.value,
            "dim": spec.dim,
            "low": list(spec.low) if spec.low is not None else None,
            "high": list(spec.high) if spec.high is not None else None,
            "num_classes": list(spec.num_classes)
            if spec.num_classes is not None
            else None,
            "units": list(spec.units),
            "wmcp_version": spec.wmcp_version,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_episode_bytes(episode: "Episode") -> bytes:
    """Canonical, backend-independent byte serialization of one episode (RFC-0014 §2).

    Ordering and encoding are fixed so the bytes are identical cross-platform and across the ``lance`` /
    ``hdf5`` / ``lerobot`` backends:

    - a self-describing version tag (``_EPISODE_CANON_TAG``) opens the stream;
    - declared scalar metadata as canonical JSON (sorted keys, no whitespace, UTF-8), restricted to the
      data-quality fields ``modality``, ``embodiment_id``, the ``ActionSpec`` digest, and the episode
      length (number of transitions) — **never** a raw observation beyond the hashed frames
      (``INV-RESIDENCY``);
    - the transition frames in time order ``t = 0..T-1``; each transition contributes ``obs_t``,
      ``action_t``, ``obs_tp1``, each tensor encoded at its stored dtype as ``(dtype string, shape,
      C-contiguous little-endian raw bytes)``.

    The serialization contains no pickle opcodes: only the version tag, ``struct``-framed integers,
    canonical JSON for scalars, and raw tensor element bytes. An empty transition list is permitted here
    and handled by the tree builder (#28); this function never raises.
    """
    transitions = list(episode.transitions)
    meta = json.dumps(
        {
            "action_spec_digest": _action_spec_digest(episode.action_spec),
            "embodiment_id": episode.embodiment_id,
            "episode_length": len(transitions),
            "modality": episode.modality,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    out = bytearray()
    out += _EPISODE_CANON_TAG
    out += _frame(meta)
    out += struct.pack("<Q", len(transitions))
    for tr in transitions:
        out += _encode_tensor(tr.obs_t)
        out += _encode_tensor(tr.action_t)
        out += _encode_tensor(tr.obs_tp1)
    return bytes(out)


def episode_leaf_hash(episode: "Episode") -> bytes:
    """SHA-256 over the ``EPISODE``-domain canonical bytes -> a 32-byte leaf preimage source.

    This is the per-episode digest the tree builder re-hashes under the ``LEAF`` domain. Returns
    exactly ``DIGEST_SIZE`` (32) bytes (RFC-0014 §2).
    """
    return _h(HashDomain.EPISODE, canonical_episode_bytes(episode))


# --- the dataset Merkle tree and inclusion proofs (RFC-0014 §3, §5) ---------------------------------


def _require_digests(leaf_digests: list[bytes]) -> None:
    """Reject an empty set or any non-``DIGEST_SIZE`` leaf (the ``merkle_root`` precondition)."""
    if not leaf_digests:
        raise ProvenanceError(
            "cannot commit a dataset with zero episodes",
            code=LensembleErrorCode.PROVENANCE_FAILED,
            remediation="commit at least one episode; an empty dataset has no Merkle root",
        )
    for d in leaf_digests:
        if not isinstance(d, (bytes, bytearray)) or len(d) != DIGEST_SIZE:
            raise ProvenanceError(
                f"leaf digest must be {DIGEST_SIZE} bytes, got {type(d).__name__} of "
                f"len {len(d) if isinstance(d, (bytes, bytearray)) else 'n/a'}",
                code=LensembleErrorCode.PROVENANCE_FAILED,
                remediation="pass LEAF-domain digests (_h(LEAF, episode_leaf_hash(ep))), each 32 bytes",
            )


def _levels(sorted_leaves: list[bytes]) -> list[list[bytes]]:
    """Build every tree level from the sorted leaves up to the single top node.

    ``level[0]`` is the sorted leaves; each subsequent level combines adjacent pairs as
    ``_h(NODE, left || right)`` left-to-right and **promotes** (does not duplicate) a trailing odd node.
    The top level is ``[top_node]``; ``merkle_root`` applies the ``ROOT`` domain to it.
    """
    levels = [sorted_leaves]
    level = sorted_leaves
    while len(level) > 1:
        nxt: list[bytes] = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(_h(HashDomain.NODE, level[i] + level[i + 1]))
        if len(level) % 2 == 1:
            nxt.append(level[-1])  # promote the odd node unchanged (no duplication)
        levels.append(nxt)
        level = nxt
    return levels


def merkle_root(leaf_digests: list[bytes]) -> bytes:
    """Deterministic Merkle root ``R_c`` over ``LEAF``-domain digests (RFC-0014 §3).

    The leaves are sorted lexicographically before the tree is built, so ``R_c`` is a commitment to the
    episode *set* independent of dataset enumeration order — two honest builds over the same episodes
    yield the identical root, and duplicate episodes intentionally produce duplicate leaves. Adjacent
    pairs combine as ``_h(NODE, left || right)``; a trailing odd node is promoted unchanged (not
    duplicated — CVE-2012-2459); the single top node is wrapped as ``R_c = _h(ROOT, top)``.

    Callers pass the leaf construction ``_h(LEAF, episode_leaf_hash(ep))`` per episode (done by
    ``commit_dataset``). Empty input or a non-32-byte leaf raises
    :class:`~lensemble.errors.ProvenanceError` (``PROVENANCE_FAILED``).
    """
    _require_digests(leaf_digests)
    levels = _levels(sorted(leaf_digests))
    return _h(HashDomain.ROOT, levels[-1][0])


@dataclass(frozen=True)
class MerkleProof:
    """Membership witness for one leaf under a committed root (RFC-0014 §5).

    ``siblings`` is the sibling digest at each combined level from the leaf upward; ``path_bits`` is the
    side bit per level (``False`` = sibling on the right, ``True`` = sibling on the left). A level where
    the leaf's node was promoted (a trailing odd node) contributes no entry — the proof skips it,
    matching ``merkle_root``'s promotion rule — so ``len(siblings) == len(path_bits)`` need not equal the
    tree height.
    """

    leaf: bytes  # the LEAF-domain digest being proven
    siblings: tuple[bytes, ...]  # sibling digest at each combined level, leaf -> top
    path_bits: tuple[bool, ...]  # False = sibling on right, True = sibling on left


def prove_inclusion(leaf_digests: list[bytes], target_index: int) -> MerkleProof:
    """Produce the inclusion proof for ``leaf_digests[target_index]`` under ``merkle_root(leaf_digests)``.

    ``target_index`` indexes the *original* (unsorted) list; the proof is built at the leaf's position in
    the sorted tree, so duplicate leaves are disambiguated by original position. Raises
    :class:`~lensemble.errors.ProvenanceError` on an empty set, a malformed leaf, or an out-of-range
    index.
    """
    _require_digests(leaf_digests)
    if not 0 <= target_index < len(leaf_digests):
        raise ProvenanceError(
            f"target_index {target_index} out of range for {len(leaf_digests)} leaves",
            code=LensembleErrorCode.PROVENANCE_FAILED,
            remediation="pass an index in [0, len(leaf_digests))",
        )
    target = leaf_digests[target_index]
    # Sort (digest, original_index) so a duplicate digest maps to its specific sorted slot.
    order = sorted(range(len(leaf_digests)), key=lambda i: (leaf_digests[i], i))
    idx = order.index(target_index)
    levels = _levels([leaf_digests[i] for i in order])

    siblings: list[bytes] = []
    path_bits: list[bool] = []
    for level in levels[:-1]:
        is_promoted = idx == len(level) - 1 and len(level) % 2 == 1
        if not is_promoted:
            if idx % 2 == 0:
                siblings.append(level[idx + 1])
                path_bits.append(False)  # sibling on the right
            else:
                siblings.append(level[idx - 1])
                path_bits.append(True)  # sibling on the left
        idx //= 2
    return MerkleProof(
        leaf=target, siblings=tuple(siblings), path_bits=tuple(path_bits)
    )


def verify_inclusion(proof: MerkleProof, root: bytes, scheme: CommitmentScheme) -> bool:
    """Recompute the root from the proof and return ``True`` iff it equals ``root`` (RFC-0014 §5).

    A pure function of public inputs (proof, published root, scheme) — "free", checkable with no prover.
    A *structurally malformed* proof (mismatched sibling/bit counts, a non-32-byte digest, an unsupported
    scheme) raises :class:`~lensemble.errors.MerkleVerificationError`; a well-formed proof that simply
    does not match the root returns ``False`` (the caller decides whether a ``False`` is an error).
    """
    if scheme.hash_name != COMMITMENT_HASH or scheme.digest_size != DIGEST_SIZE:
        raise MerkleVerificationError(
            f"unsupported commitment scheme {scheme.hash_name}/{scheme.digest_size}",
            code=LensembleErrorCode.MERKLE_VERIFY_FAILED,
            remediation=f"this build verifies {COMMITMENT_HASH}/{DIGEST_SIZE}-byte commitments only",
        )
    if len(proof.siblings) != len(proof.path_bits):
        raise MerkleVerificationError(
            f"malformed proof: {len(proof.siblings)} siblings vs {len(proof.path_bits)} path bits",
            code=LensembleErrorCode.MERKLE_VERIFY_FAILED,
            remediation="siblings and path_bits must have equal length (one entry per combined level)",
        )
    for d in (proof.leaf, root, *proof.siblings):
        if not isinstance(d, (bytes, bytearray)) or len(d) != DIGEST_SIZE:
            raise MerkleVerificationError(
                f"malformed proof: a digest is not {DIGEST_SIZE} bytes",
                code=LensembleErrorCode.MERKLE_VERIFY_FAILED,
                remediation="leaf, root, and every sibling must be 32-byte digests",
            )
    cur = bytes(proof.leaf)
    for sibling, on_left in zip(proof.siblings, proof.path_bits):
        cur = (
            _h(HashDomain.NODE, bytes(sibling) + cur)
            if on_left
            else _h(HashDomain.NODE, cur + bytes(sibling))
        )
    return _h(HashDomain.ROOT, cur) == bytes(root)
