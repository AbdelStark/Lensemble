"""lensemble.provenance.merkle — domain-separated episode hashing (docs/rfcs/RFC-0014 §1-2).

This module is the leaf-construction layer every dataset commitment is built on. It hashes an
``Episode`` to a SHA-256 leaf via a canonical, **backend-independent** byte serialization: the same
logical episode commits to the same leaf whether it was stored as ``lance``, ``hdf5``, or read through
the ``lerobot://`` adapter (RFC-0004 §2). The hash is therefore taken over the normalized *logical*
form — fixed dtype, fixed little-endian byte order, fixed time ordering — not over raw on-disk bytes.

Domain separation (``HashDomain``) prepends a one-byte tag to every preimage so an episode digest can
never be reinterpreted as a leaf/internal-node/root digest (RFC-0014 §1); the three tree domains
(``LEAF``/``NODE``/``ROOT``) are consumed by the tree builder (#28), not here. The hash is selected by a
versioned :class:`CommitmentScheme` rather than hard-coded at each call site, so the Stage-D migration to
a STARK-friendly hash is a version bump, not a rewrite (RFC-0014 §1).

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

    This is the per-episode digest the tree builder (#28) re-hashes under the ``LEAF`` domain. Returns
    exactly ``DIGEST_SIZE`` (32) bytes (RFC-0014 §2).
    """
    return _h(HashDomain.EPISODE, canonical_episode_bytes(episode))
