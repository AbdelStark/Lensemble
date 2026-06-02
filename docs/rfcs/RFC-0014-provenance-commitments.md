# RFC-0014 — Provenance Commitments & Merkle Scheme

| | |
|---|---|
| **RFC** | 0014 |
| **Title** | Provenance Commitments & Merkle Scheme |
| **Slug** | provenance-commitments |
| **Status** | Accepted |
| **Track** | Standards |
| **Authors** | @AbdelStark |
| **Created** | 2026-06-02 |
| **Target milestone** | v0.1 (episode hashing, Merkle commitment, binding, ledger scaffold); residency over a real boundary and the networked ledger in v0.3; the STARK-friendly-hash migration in Stage D (post-v1.0) |
| **Area** | provenance |
| **Requires** | [RFC-0003](RFC-0003-federated-protocol.md) (the `Commitment` message and binding), [RFC-0004](RFC-0004-data-provenance.md) (the data layer and accounting requirements) |
| **Defers to** | [RFC-0006](RFC-0006-verifiable-contribution.md) (the Phase-2 proofs the commitments enable), [RFC-0010](RFC-0010-artifact-checkpoint-format.md) (the global-model content hash recorded in the ledger) |

## Summary

This RFC specifies the cryptographic commitment scheme that binds each participant's contribution to
the data it was computed from. It is the Phase-1 bridge to the Phase-2 verifiable layer
([RFC-0006](RFC-0006-verifiable-contribution.md)): the commitments are cheap engineering disciplines we
honor from day one ([RFC-0006 §3](RFC-0006-verifiable-contribution.md)) so that adding proofs later
needs no rework.

The scheme has four parts. (1) **Episode hashing**: a canonical, domain-separated byte serialization of
an episode hashes to a SHA-256 leaf. (2) **A dataset Merkle tree**: a binary tree over the sorted leaf
hashes with domain-separated node hashing, whose root is the dataset commitment `R_c`
([RFC-0004 §4](RFC-0004-data-provenance.md)). (3) **Binding**: every released pseudo-gradient `Δ_c`
carries exactly one `R_c` (`INV-COMMIT-BINDING`); a delta not bound to a valid root is rejected at the
coordinator with `CommitmentMismatch`. (4) **A contribution ledger**: an append-only log of
`(round, contributing participants, their roots, the resulting global-model hash)` — the audit substrate
([RFC-0004 §5](RFC-0004-data-provenance.md)) the Phase-2 layer formalizes.

The public surface this RFC stabilizes — `commit_dataset`, `DatasetCommitment`, `ContributionLedger`
([02 §Provenance](../spec/02-public-api.md)) — is consumed by the federation protocol
([RFC-0003 §3, §8](RFC-0003-federated-protocol.md)), the data layer
([RFC-0004 §4–5](RFC-0004-data-provenance.md)), the data model ([03 §DatasetCommitment](../spec/03-data-model.md)),
and the security model ([06 §Provenance](../spec/06-security.md)). The boundary it states plainly: this
proves data **origin/provenance**, not data quality and not honest computation.

## Motivation

In a data-sovereign federation, raw trajectories never cross a boundary (`INV-RESIDENCY`,
[RFC-0004 §2](RFC-0004-data-provenance.md)); only privatized, aggregated model deltas do
([RFC-0003 §2](RFC-0003-federated-protocol.md)). That property is also a verifiability problem: a
coordinator that never sees the data cannot, by inspection of a delta, say *which* data produced it. For
contribution accounting (who improved the model, on what data), for licensing claims, and for the
Phase-2 statement "this update was computed from data under committed root `R_c`"
([RFC-0006 §3](RFC-0006-verifiable-contribution.md)), the federation needs a succinct, tamper-evident
commitment to each participant's dataset that the participant produces locally and publishes once.

A Merkle tree is the natural construction: the root `R_c` is a single 32-byte value that commits to the
entire dataset, while inclusion proofs let a Phase-2 prover argue that specific episodes are members of
the committed set without revealing the rest of the dataset. The commitment must be deterministic and
cross-platform stable so two honest implementations compute the identical `R_c` from the identical
episodes — the same discipline as the checkpoint content hash
([RFC-0010 §Hashing](RFC-0010-artifact-checkpoint-format.md), `INV-CHECKPOINT-HASH`).

These are inexpensive disciplines in Phase 1 (hashing the data once per epoch) and are the entire
prerequisite for the Phase-2 provenance binding ([RFC-0006 §3](RFC-0006-verifiable-contribution.md),
roadmap step 2b). Honoring them now is strictly cheaper than retrofitting them after the science ships.

## Goals

- Define episode hashing: a canonical, domain-separated byte serialization of an `Episode`
  ([03 §Episode](../spec/03-data-model.md)) to a SHA-256 leaf hash, stable across platforms and
  serialization backends (`lance`/`hdf5`/`lerobot`, [RFC-0004 §2](RFC-0004-data-provenance.md)).
- Define the dataset Merkle tree: leaf ordering, domain-separated leaf vs internal-node hashing,
  odd-node handling, and the root `R_c` carried by a `DatasetCommitment`.
- Define inclusion proofs (the membership witness Phase 2 needs) and their verification.
- Define the binding (`INV-COMMIT-BINDING`): each released `Δ_c` is associated with exactly one `R_c`;
  the `Commitment` message ([RFC-0003 §8](RFC-0003-federated-protocol.md)); the ingress check and the
  errors on failure (`CommitmentMismatch`, `MerkleVerificationError`).
- Define the `ContributionLedger`: the append-only record schema and its append-only invariant.
- Pin the Phase-1 commitment hash (SHA-256) and state the STARK-friendly-hash migration as a versioned,
  deferred choice shared with [RFC-0006](RFC-0006-verifiable-contribution.md).
- State the provenance boundary plainly: origin, not quality, not honesty.

## Non-Goals

- The Phase-2 proofs themselves (aggregation STARK, provenance-binding proof, TEE attestation). Owned by
  [RFC-0006](RFC-0006-verifiable-contribution.md); this RFC ships only the commitments those proofs bind
  to.
- The data layer (formats, loaders, residency flag) and the public-probe governance. Owned by
  [RFC-0004](RFC-0004-data-provenance.md); this RFC consumes the `Episode` schema and the data-quality
  metadata it declares.
- Data-quality gating or scoring. Provenance proves origin; quality enforcement beyond declared metadata
  is out of scope for v0.1 ([RFC-0004 §6](RFC-0004-data-provenance.md)) and is an Open Question.
- The pseudo-gradient construction and the round lifecycle. Owned by
  [RFC-0003](RFC-0003-federated-protocol.md); this RFC owns only the `Commitment` payload and its
  binding semantics.
- Incentive/payment mechanisms, slashing, on-chain settlement. Out of scope
  ([RFC-0006 §Non-Goals](RFC-0006-verifiable-contribution.md)); the ledger is an audit substrate, not
  an economic layer.
- The checkpoint/artifact format and its content hash. Owned by
  [RFC-0010](RFC-0010-artifact-checkpoint-format.md); the ledger records that hash, it does not define it.

## Proposed Design

The provenance subsystem lives in `lensemble.provenance` ([01 §Module Map](../spec/01-architecture.md)):
`merkle.py` (tree construction, inclusion proofs), `commit.py` (`commit_dataset`, `DatasetCommitment`),
`ledger.py` (`ContributionLedger`, `ContributionRecord`). It depends only on `core`/`errors`,
`config`, and the `Episode` type from `data`; nothing in `provenance` imports `model`, `gauge`, or
`federation` ([01 §Dependency Layering](../spec/01-architecture.md)).

### 1. Domain-separated hashing primitive

All hashing is SHA-256 (Phase-1 canonical commitment hash,
[09 §Dependencies](../spec/09-release-and-versioning.md)) with a
single-byte domain-separation tag prepended to the preimage. Domain separation prevents a leaf hash from
ever colliding with or being reinterpreted as an internal-node hash (the classic second-preimage attack
on naive Merkle trees), and isolates the episode/leaf/node/root domains.

```python
import hashlib
from enum import IntEnum

class HashDomain(IntEnum):
    """One-byte domain-separation tag prepended to every preimage."""
    EPISODE = 0x00   # canonical episode bytes -> leaf preimage
    LEAF    = 0x01   # leaf node in the Merkle tree
    NODE    = 0x02   # internal node (concatenation of two child digests)
    ROOT    = 0x03   # the committed dataset root R_c

COMMITMENT_HASH = "sha256"   # Phase-1 canonical; versioned (see CommitmentScheme below)
DIGEST_SIZE = 32             # bytes, SHA-256

def _h(domain: HashDomain, payload: bytes) -> bytes:
    """Domain-separated SHA-256: H(domain_tag || payload) -> 32 bytes."""
    return hashlib.sha256(bytes([domain]) + payload).digest()
```

The hash is selected by a `CommitmentScheme` record (below), not hard-coded into call sites, so the
Stage-D migration to a STARK-friendly hash is a versioned change, not a rewrite.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class CommitmentScheme:
    """The pinned hash + tree parameters a DatasetCommitment was built under.
    Recorded on every commitment so a verifier reconstructs the exact scheme."""
    hash_name: str = COMMITMENT_HASH   # "sha256" (Phase 1); "poseidon2" reserved (Stage D)
    digest_size: int = DIGEST_SIZE     # bytes
    scheme_version: int = 1            # bumped on any tree/hash/serialization change
```

### 2. Episode hashing

An `Episode` ([03 §Episode](../spec/03-data-model.md)) is a sequence of `Transition` tuples
`(o_t, a_t, o_{t+1})` ([03 §Transition](../spec/03-data-model.md)) plus declared metadata
([RFC-0004 §6](RFC-0004-data-provenance.md)). Its leaf hash MUST be independent of the on-disk
serialization backend (`lance` default, `hdf5` portable, `lerobot://` adapter,
[RFC-0004 §2](RFC-0004-data-provenance.md)) so the same logical episode commits to the same leaf
regardless of how it is stored. The canonical episode serialization therefore hashes a normalized
*logical* form, not raw file bytes:

```python
from typing import Any, Mapping

def canonical_episode_bytes(episode: "Episode") -> bytes:
    """Canonical, backend-independent byte serialization of one episode.

    Ordering and encoding are fixed so the bytes are identical cross-platform:
      - frames in time order t = 0..T-1;
      - each tensor encoded as (shape tuple, dtype string, C-contiguous little-endian
        raw bytes) — fixed dtype/endianness, never the platform-native layout;
      - scalar metadata serialized as canonical JSON (sorted keys, no whitespace,
        UTF-8), restricted to declared data-quality fields (modality, embodiment_id,
        ActionSpec digest, episode length) — NEVER raw observations beyond the frames
        already covered.
    """
    ...

def episode_leaf_hash(episode: "Episode") -> bytes:
    """SHA-256 over the EPISODE-domain canonical bytes -> 32-byte leaf preimage source."""
    return _h(HashDomain.EPISODE, canonical_episode_bytes(episode))
```

RISK: the cross-platform stability of `canonical_episode_bytes` rests on a fixed dtype/endianness and a
fixed frame ordering. Float tensors stored at different precisions (a `bf16` capture vs an `fp32`
re-export of the same episode) hash differently. Resolution plan: episodes are hashed at their stored
dtype and the stored dtype is part of the canonical bytes (so a re-quantized copy is, correctly, a
different episode); the conformance test (Testing Strategy) asserts determinism across two processes and
two backends for a fixed-dtype fixture. The exact canonical byte ordering is shared with the checkpoint
canonicalization Open Question ([RFC-0010 §Open Questions](RFC-0010-artifact-checkpoint-format.md)).

### 3. The dataset Merkle tree and `R_c`

Given the multiset of episode leaf preimages, the tree is built deterministically:

1. **Leaf construction.** For each episode, compute `leaf_i = _h(LEAF, episode_leaf_hash(episode_i))`.
   The `LEAF` domain re-hash binds the leaf to its position in the tree-domain and separates it from any
   raw `EPISODE` digest.
2. **Leaf ordering (`INV` of the commitment).** Sort the leaf digests lexicographically (byte order).
   Sorting makes `R_c` a commitment to the *set* of episodes independent of dataset enumeration order, so
   two honest builds over the same episodes yield the identical root. (Duplicate episodes produce
   duplicate leaves; this is intentional — a dataset with two identical episodes is a different set than
   one with a single copy.)
3. **Internal nodes.** Combine adjacent pairs: `parent = _h(NODE, left || right)`, left-to-right. An odd
   node at any level is promoted unchanged to the next level (it is *not* duplicated — duplication is the
   well-known source of the CVE-2012-2459 Merkle malleability bug; promotion avoids it).
4. **Root.** When one node remains, the committed root is `R_c = _h(ROOT, top_node)`. The `ROOT` domain
   tag ensures the published root is never confusable with an internal node digest.

```python
def merkle_root(leaf_digests: list[bytes]) -> bytes:
    """Deterministic Merkle root over LEAF-domain digests.

    Preconditions: every element is DIGEST_SIZE bytes; len >= 1.
    Empty input raises ProvenanceError (a dataset with zero episodes cannot commit).
    """
    if not leaf_digests:
        raise ProvenanceError(code=LensembleErrorCode.EMPTY_DATASET, remediation=...)
    level = sorted(leaf_digests)                 # lexicographic, deterministic
    while len(level) > 1:
        nxt: list[bytes] = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(_h(HashDomain.NODE, level[i] + level[i + 1]))
        if len(level) % 2 == 1:
            nxt.append(level[-1])                # promote odd node (no duplication)
        level = nxt
    return _h(HashDomain.ROOT, level[0])
```

ASCII view of a four-leaf tree (the prose above says the same): leaves `L0..L3` are sorted, paired into
internal nodes `N01 = H(N, L0||L1)` and `N23 = H(N, L2||L3)`, combined into `T = H(N, N01||N23)`, and the
published root is `R_c = H(ROOT, T)`.

```text
                 R_c = H(ROOT, T)
                       T = H(N, N01 || N23)
                ┌──────────┴──────────┐
        N01 = H(N, L0||L1)     N23 = H(N, L2||L3)
          ┌─────┴─────┐          ┌─────┴─────┐
         L0          L1         L2          L3        (L* sorted lexicographically)
```

### 4. `DatasetCommitment` and `commit_dataset`

`commit_dataset` ([02 §Provenance](../spec/02-public-api.md)) hashes every episode, builds the tree, and
returns a `DatasetCommitment` carrying the root `R_c`, the episode count, the WMCP metadata, and the
scheme it was built under. The commitment is the on-disk, schema-versioned
([09 §Schema Versioning](../spec/09-release-and-versioning.md)) provenance record validated by a pydantic v2
model ([03 §Serialization](../spec/03-data-model.md)).

```python
from pydantic import BaseModel

class DatasetCommitment(BaseModel):
    """Tamper-evident commitment to a participant's local dataset (03 §Data Model).
    On-disk metadata: pydantic v2 JSON with an explicit integer schema_version."""
    schema_version: int            # on-disk schema (09 §Schema Versioning); SchemaVersionMismatch on unknown/too-new
    root: bytes                    # R_c, 32 bytes (DIGEST_SIZE)
    episode_count: int             # number of leaves; >= 1
    wmcp_version: str              # the latent-contract version the episodes conform to (RFC-0007)
    embodiment_id: str             # the embodiment whose ActionSpec the actions satisfy (RFC-0007)
    scheme: CommitmentScheme       # hash + tree params used (versioned for the Stage-D migration)
    created_at: str                # RFC 3339 UTC

def commit_dataset(dataset: "Dataset") -> DatasetCommitment:
    """Build the Merkle commitment for a local dataset (02 §Provenance).

    Precondition: dataset is non-empty and WMCP-conformant (RFC-0007); every episode's
      ActionSpec matches the declared embodiment_id (else ContractViolation, raised by data).
    Postcondition: commit_dataset(D).root == commit_dataset(D).root for any two builds over
      the same logical episodes on any platform (deterministic; cf. INV-CHECKPOINT-HASH style).
    Raises: ProvenanceError (empty dataset); the per-episode ContractViolation is raised upstream
      by the data layer during ingest (RFC-0004), not here.
    Determinism: pure function of the episode set + the CommitmentScheme. No RNG, no I/O ordering
      dependence (leaves are sorted).
    """
    ...
```

The participant publishes `R_c` once per epoch via the `Commitment` message
([RFC-0003 §8](RFC-0003-federated-protocol.md)) before contributing; the full episodes never leave the
boundary (`INV-RESIDENCY`, [RFC-0004 §2](RFC-0004-data-provenance.md)) — only the 32-byte root and the
declared metadata.

### 5. Inclusion proofs (the Phase-2 membership witness)

An inclusion proof witnesses that a specific episode is a member of the committed set without revealing
the rest of the dataset — the membership argument the Phase-2 provenance binding
([RFC-0006 §3](RFC-0006-verifiable-contribution.md), roadmap 2b) is built on. It is the ordered list of
sibling digests on the path from the leaf to the root, with a side bit per level.

```python
@dataclass(frozen=True)
class MerkleProof:
    """Membership witness for one leaf under a committed root."""
    leaf: bytes                       # the LEAF-domain digest being proven
    siblings: tuple[bytes, ...]       # sibling digest at each level, leaf -> top
    path_bits: tuple[bool, ...]       # False=sibling on right, True=sibling on left, per level

def prove_inclusion(leaf_digests: list[bytes], target_index: int) -> MerkleProof: ...

def verify_inclusion(proof: MerkleProof, root: bytes, scheme: CommitmentScheme) -> bool:
    """Recompute the root from the leaf + siblings using the scheme's domains.
    Returns True iff the recomputed root equals `root`. Promoted odd nodes contribute
    no sibling at that level (the proof skips them), matching merkle_root's promotion rule.
    A structurally malformed proof raises MerkleVerificationError; a well-formed proof that
    simply does not match returns False (callers decide whether a False is an error)."""
    ...
```

`verify_inclusion` is a pure function of public inputs (the proof, the published root, the scheme); like
the frame-alignment recomputation ([RFC-0006 §4](RFC-0006-verifiable-contribution.md),
[RFC-0002 §5](RFC-0002-gauge-and-aggregation.md)) it is "free" — anyone can check it with no prover.

### 6. Binding `Δ_c` to `R_c` (`INV-COMMIT-BINDING`)

> **INV-COMMIT-BINDING** ([03 §Invariants](../spec/03-data-model.md)): every released `Δ_c` is bound to exactly
> one dataset Merkle root `R_c`. **Enforced** at coordinator ingress in `lensemble.federation` against
> the `Commitment` message, and carried structurally by `PseudoGradient.dataset_root`
> ([RFC-0003 §3](RFC-0003-federated-protocol.md)).

The binding has two halves. (a) **At construction**, the participant sets
`PseudoGradient.dataset_root = R_c` for the commitment under which the `H` inner steps were run
([RFC-0003 §3](RFC-0003-federated-protocol.md)); the field is part of the frozen `PseudoGradient`
dataclass, so a delta cannot be released without a root. (b) **At ingress**, the coordinator validates
that the `dataset_root` on the incoming `Update` matches the `R_c` in the participant's `Commitment`
message for that round and that the root is a syntactically valid 32-byte digest under the round's
`CommitmentScheme`. The check is a binding check, not a membership recomputation (the coordinator has no
episodes); membership is the Phase-2 inclusion-proof job.

```python
def verify_binding(committed_root: bytes, declared_root: bytes, scheme: CommitmentScheme) -> None:
    """Raise unless the released delta's root matches the participant's committed root.
    Pure check: no dataset access. CommitmentMismatch on inequality (never swallowed,
    04 §Error Model); MerkleVerificationError on a malformed/short root."""
    if len(declared_root) != scheme.digest_size:
        raise MerkleVerificationError(code=LensembleErrorCode.MERKLE_VERIFICATION, remediation=...)
    if declared_root != committed_root:
        raise CommitmentMismatch(code=LensembleErrorCode.COMMITMENT_MISMATCH, remediation=...)
```

`CommitmentMismatch` is security-critical and is **never** caught-and-ignored
([04 §Error-Handling Rules](../spec/04-error-model.md)); the
mismatched update is rejected ([RFC-0003 §7, §9](RFC-0003-federated-protocol.md)) and does not enter the
sum, so a participant cannot launder an unattributed delta into the global model.

### 7. The `ContributionLedger`

The ledger is the append-only audit substrate ([RFC-0004 §5](RFC-0004-data-provenance.md)): one record
per round, recording the contributing participants, the roots they were bound to, and the global-model
content hash that round produced ([RFC-0010](RFC-0010-artifact-checkpoint-format.md),
`INV-CHECKPOINT-HASH`). It is the data substrate the Phase-2 layer formalizes
([RFC-0006 §5](RFC-0006-verifiable-contribution.md)).

```python
class ContributionRecord(BaseModel):
    """One round's contribution provenance (03 §Data Model). On-disk pydantic v2, schema_versioned."""
    schema_version: int
    round_index: int                       # the outer round t
    participants: tuple[str, ...]           # contributing participant ids (sorted, deterministic)
    dataset_roots: tuple[bytes, ...]        # R_c per participant, index-aligned with `participants`
    global_model_hash: bytes               # content hash of (θ_{t+1}, φ_{t+1}) (RFC-0010)
    prev_record_hash: bytes | None         # hash of the prior record (the ledger chain); None at t=0
    created_at: str                        # RFC 3339 UTC

class ContributionLedger:
    """Append-only log of ContributionRecords (02 §Provenance)."""

    def append(self, record: ContributionRecord) -> bytes:
        """Append and return the record's content hash. Sets record.prev_record_hash to the
        previous tail (a hash chain) so any rewrite of history is detectable.
        Raises ProvenanceError if the record's round_index is not strictly greater than the
        tail's (the append-only / monotone invariant) or if prev_record_hash does not match
        the current tail."""
        ...

    def verify_chain(self) -> bool:
        """Walk the chain: each record's prev_record_hash equals the prior record's content
        hash, round_index strictly increases. False (not silently) on any break; a structural
        corruption raises MerkleVerificationError."""
        ...
```

The `prev_record_hash` chain makes the ledger tamper-evident: rewriting any past round breaks every
subsequent link. In Phase 1 this is tamper-*evidence*; the chain plus the per-round commitments are what
the Phase-2 layer upgrades to a tamper-*proof* statement
([RFC-0006 §2](RFC-0006-verifiable-contribution.md)).

### 8. The provenance boundary (stated plainly)

A `DatasetCommitment` and its binding prove **origin**: that a released update was computed from data
under committed root `R_c`. They prove **neither**:

- **data quality** — `R_c` commits to whatever episodes the participant supplied; garbage commits as
  cleanly as gold. Quality is declared metadata ([RFC-0004 §6](RFC-0004-data-provenance.md)), not
  proven, and quality gating beyond provenance is out of scope for v0.1 (Open Questions).
- **honest computation** — the binding says nothing about whether the gradient was honestly derived from
  that data. Honest computation is the Phase-2 problem split across the aggregation STARK, the
  inner-step TEE attestation, and the inclusion-proof membership argument
  ([RFC-0006 §5](RFC-0006-verifiable-contribution.md)).

Stating this boundary is part of the design's integrity
([RFC-0006 §2](RFC-0006-verifiable-contribution.md)); the security model repeats it
([06 §Provenance](../spec/06-security.md)).

### 9. Failure modes

| Trigger | Detection | Error | System response |
|---|---|---|---|
| `commit_dataset` on an empty dataset | leaf-count check in `merkle_root` | `ProvenanceError` | reject the commit; a dataset with zero episodes cannot join a round |
| Released `Δ_c.dataset_root` ≠ participant's committed `R_c` | binding check at coordinator ingress (`verify_binding`) | `CommitmentMismatch` | reject the update; never swallowed; excluded from the sum (`INV-COMMIT-BINDING`, [RFC-0003 §9](RFC-0003-federated-protocol.md)) |
| Declared root is not a valid 32-byte digest | length check under the round `CommitmentScheme` | `MerkleVerificationError` | reject the `Update`/`Commitment` message at ingress |
| Inclusion proof structurally malformed (wrong sibling count, bad length) | `verify_inclusion` structural validation | `MerkleVerificationError` | reject the proof (Phase-2 path) |
| Inclusion proof well-formed but does not reconstruct the root | `verify_inclusion` returns `False` | (no raise) | caller decides; a Phase-2 verifier treats `False` as a failed claim |
| Ledger append with non-monotone `round_index` or wrong `prev_record_hash` | append-only invariant check | `ProvenanceError` | reject the append; the chain is not extended |
| On-disk `DatasetCommitment`/`ContributionRecord` with unknown/too-new `schema_version` | pydantic load + version gate ([RFC-0010](RFC-0010-artifact-checkpoint-format.md)) | `SchemaVersionMismatch` | refuse to load; run the migration if available |
| Two honest builds over the same episodes disagree on `R_c` | determinism conformance test (Testing Strategy) | (CI failure) | a defect in `canonical_episode_bytes` or `merkle_root`; blocks release |

`CommitmentMismatch` and `MerkleVerificationError` derive from `ProvenanceError`, which derives from
`LensembleError`; every one carries `.code` (a `LensembleErrorCode` value) and `.remediation`
([04 §Provenance](../spec/04-error-model.md)).

## Alternatives Considered

- **A flat hash of the whole dataset vs a Merkle tree.** What it is: `R_c = H(concat of all episode
  bytes)`. Why considered: simplest possible commitment, smallest code, and sufficient for Phase-1
  tamper-evidence and contribution accounting alone. Why rejected: a flat hash supports no inclusion
  proof and no incremental commitment, so it forecloses the Phase-2 provenance-binding membership
  argument ([RFC-0006 §3](RFC-0006-verifiable-contribution.md)) — a verifier could not argue that a
  specific episode is in the committed set without re-hashing the entire dataset, and a participant could
  not append episodes without recomputing the whole hash. The Merkle tree's `O(log n)` proofs and
  reusable internal nodes are exactly the proof-ready property we honor now to avoid rework later
  ([RFC-0006 §3](RFC-0006-verifiable-contribution.md)).
- **Insertion-order leaves vs sorted leaves.** What it is: build the tree in dataset enumeration order
  rather than sorting leaf digests. Why considered: avoids the sort and makes inclusion proofs index-
  stable. Why rejected: enumeration order is backend- and platform-dependent (`lance` index order ≠
  `hdf5` order ≠ `lerobot` order, [RFC-0004 §2](RFC-0004-data-provenance.md)), so two honest builds of
  the same dataset would produce different roots — breaking the determinism postcondition that makes
  `R_c` a reproducible commitment. Sorting makes `R_c` a commitment to the set, independent of read
  order, at the cost of a one-time sort.
- **Duplicating the odd node vs promoting it.** What it is: when a level has an odd count, the classic
  Bitcoin construction duplicates the last node (`H(N, last||last)`). Why considered: it keeps every
  level a perfect binary level and simplifies proof indexing. Why rejected: duplication is the source of
  Merkle malleability (CVE-2012-2459) — distinct leaf multisets can hash to the same root — which would
  undermine the very tamper-evidence the commitment exists for. Promotion (carry the odd node up
  unchanged) plus per-level domain separation removes the malleability at no security cost.
- **SHA-256 vs a STARK-friendly hash (Poseidon2) now.** What it is: build the tree directly over a
  prime-field hash whose circuit is cheap for the Phase-2 STARK ([RFC-0006](RFC-0006-verifiable-contribution.md)).
  Why considered: it would make the Phase-2 provenance-binding proof circuit far smaller, since SHA-256
  is expensive to prove in a Circle-STARK. Why rejected for Phase 1: SHA-256 is conservative,
  interoperable, and in the standard library ([09 §Dependencies](../spec/09-release-and-versioning.md));
  Phase-2 proofs are
  post-v1.0 ([RFC-0006](RFC-0006-verifiable-contribution.md)); and the `CommitmentScheme` record makes
  the hash a versioned, swappable choice, so committing to Poseidon2 prematurely would couple the
  Phase-1 data layer to an immature prover stack for no Phase-1 benefit. The migration is the shared Open
  Question.
- **No ledger / commitments only.** What it is: bind `Δ_c` to `R_c` per round but keep no durable log.
  Why considered: the binding alone supports per-round rejection of unattributed deltas. Why rejected:
  without an append-only, hash-chained record there is no audit substrate for contribution accounting
  ([RFC-0004 §5](RFC-0004-data-provenance.md)) and nothing for the Phase-2 layer to formalize; the
  ledger is the cheap durable artifact that makes "who contributed what, on what data" answerable after
  the fact.

## Drawbacks

- **Hashing cost over large datasets.** Computing leaf hashes touches every episode's frame tensors once
  per commitment epoch; for foundation-scale data (Stage E) this is non-trivial. Mitigation: commit per
  epoch, not per round; cache leaf hashes keyed by episode identity so unchanged episodes are not
  re-hashed; the Phase-2 incremental re-commitment (Open Questions) reuses internal nodes on append.
- **Proves origin only.** The provenance boundary (§8) is a genuine limitation, not a bug: the
  commitment cannot detect a participant that commits to honest-looking but adversarial or low-quality
  data. This is stated plainly rather than papered over; quality gating is deferred and honest
  computation is the Phase-2 problem.
- **SHA-256 is expensive to prove.** The conservative Phase-1 choice is exactly the wrong hash for a
  cheap Phase-2 circuit; the `CommitmentScheme` versioning is the planned escape, but a migration is real
  future work with a re-commitment cost (every participant re-hashes its dataset under the new scheme).
- **Cross-platform hash stability is fragile.** It depends on a fixed canonical serialization (§2); a
  silent change in dtype handling or frame ordering would split honest implementations. The determinism
  conformance test is the guard, but the canonicalization is shared surface area with the checkpoint hash
  ([RFC-0010](RFC-0010-artifact-checkpoint-format.md)) and must be changed only with a `scheme_version`
  bump.

## Migration / Rollout

The commitments are honored from day one, ahead of any proof
([RFC-0006 §3](RFC-0006-verifiable-contribution.md)):

- **v0.1 / Stage A.** `episode_leaf_hash`, `merkle_root`, `commit_dataset`/`DatasetCommitment`,
  `verify_binding`, and the `ContributionLedger` scaffold land with the data layer. Single-site Stage A
  exercises `commit_dataset` on the pooled dataset and records `ContributionRecord`s locally even though
  there is no federation yet — this is the cheap tamper-evidence
  ([RFC-0006 §3](RFC-0006-verifiable-contribution.md)).
- **v0.2 / Stage B.** The binding is exercised in the simulated federation: each simulated participant
  publishes `R_c` via the `Commitment` message ([RFC-0003 §8](RFC-0003-federated-protocol.md)) and the
  coordinator runs `verify_binding` at ingress; the ledger records the per-round roots and the committed
  global-model hash.
- **v0.3 / Stage C.** The `Commitment` message crosses a real network boundary; the ledger becomes a
  durable, networked audit log; residency enforcement over the wire guarantees only the 32-byte root and
  declared metadata leave a boundary, never episodes (`INV-RESIDENCY`).
- **Stage D (post-v1.0).** Migrate the `CommitmentScheme` hash to a STARK-friendly choice (Open
  Questions) and add the inclusion-proof-backed provenance-binding proof
  ([RFC-0006 §6](RFC-0006-verifiable-contribution.md), roadmap 2b). The `scheme_version` /
  `CommitmentScheme` carried on every commitment is what makes this a versioned migration, not a rewrite:
  old commitments remain verifiable under their recorded scheme.

The hash function is a versioned choice; a scheme change forces re-commitment (every participant rebuilds
`R_c` under the new scheme) but does not invalidate old records, which carry their own `CommitmentScheme`.

## Testing Strategy

CPU-runnable tests on tiny synthetic fixtures (no large downloads;
[07 §CI Gates](../spec/07-testing-strategy.md)). These realize the Phase-1 proof-readiness checks
([RFC-0006 §Testing Strategy](RFC-0006-verifiable-contribution.md)) for provenance:

- **Merkle root correctness.** Build the tree over known leaf sets (sizes 1, 2, 3, 4, 5, 8 — exercising
  the odd-node promotion path) and assert `R_c` matches independently hand-computed roots; assert the
  empty set raises `ProvenanceError`.
- **Inclusion-proof correctness.** For every leaf in a fixture, `prove_inclusion` then `verify_inclusion`
  returns `True`; a proof with a flipped `path_bit`, a corrupted sibling, or against a wrong root returns
  `False`; a structurally malformed proof raises `MerkleVerificationError`. Exercise a leaf whose path
  crosses a promoted odd node.
- **Hash determinism (cross-platform proxy).** `commit_dataset` over the same fixed-dtype episode set in
  two separate processes (and via the `lance` and `hdf5` backends) produces the byte-identical `R_c`;
  permute the dataset enumeration order and assert `R_c` is unchanged (leaf-sorting property).
- **Domain separation.** Assert a leaf preimage never collides with a node preimage by construction (a
  property test that no `EPISODE`/`LEAF`/`NODE`/`ROOT` digest equals another for the same payload).
- **Binding verification (`INV-COMMIT-BINDING`).** `verify_binding` raises `CommitmentMismatch` when a
  delta's `dataset_root` differs from the committed `R_c`, raises `MerkleVerificationError` on a
  short/malformed root, and passes on a match; assert the error is never swallowed
  ([RFC-0003 §9](RFC-0003-federated-protocol.md), [04 §Error-Handling Rules](../spec/04-error-model.md)).
- **Ledger append-only invariant.** `append` rejects a non-monotone `round_index` and a wrong
  `prev_record_hash` with `ProvenanceError`; `verify_chain` returns `False` on a rewritten past record
  and `True` on an intact chain.
- **Schema round-trip + migration.** A `DatasetCommitment`/`ContributionRecord` round-trips through JSON;
  an unknown/too-new `schema_version` raises `SchemaVersionMismatch`
  ([RFC-0010](RFC-0010-artifact-checkpoint-format.md)).
- **No-pickle / no-raw-data.** Assert the canonical episode serialization contains no pickle opcodes and
  that no metadata field carries raw observations beyond the hashed frames (`INV-RESIDENCY`,
  [05 §Redaction](../spec/05-observability.md)).

Numerical-tolerance policy: commitment hashing requires **exact byte equality** (no tolerance) — a hash
that is "close" is wrong ([07 §Numerical Tolerance](../spec/07-testing-strategy.md)).

## Open Questions

OPEN QUESTION: Migrate the Phase-1 commitment hash (SHA-256) to a STARK-friendly hash (e.g. Poseidon2)
to keep the Phase-2 provenance-binding circuit cheap. The `CommitmentScheme` record makes this a
versioned change. Owner @AbdelStark; resolution: Stage D (post-v1.0), shared with
[RFC-0006 §Open Questions](RFC-0006-verifiable-contribution.md) and the canonical-hash decision in
[09 §Dependencies](../spec/09-release-and-versioning.md).

OPEN QUESTION: Incremental re-commitment on dataset append. When a participant adds episodes between
rounds, recomputing `R_c` from scratch is wasteful; a Merkle tree admits `O(log n)` re-rooting if the
leaf-sorting and odd-node-promotion rules are made append-stable. The current sorted construction
re-sorts on every commit. Owner @AbdelStark; resolution: a follow-up in Stage D when incremental commits
become a cost concern; v0.1 commits per epoch from scratch.

OPEN QUESTION: Quality-gating policy beyond provenance. The commitment proves origin, not quality
([RFC-0004 §6](RFC-0004-data-provenance.md)); whether and how the federation weights or gates
contributions on declared (or attested) quality is unspecified. Owner @AbdelStark; resolution: a Stage-B
(v0.2) experiment on whether declared data-quality metadata correlates with contribution value, then a
follow-up RFC if a gating mechanism is warranted.

RISK: The cross-platform stability of `canonical_episode_bytes` (§2) is the load-bearing assumption for a
reproducible `R_c`; a silent dtype/endianness/ordering change splits honest implementations. Resolution
plan: pin the canonicalization behind `CommitmentScheme.scheme_version`, gate every change on a
`scheme_version` bump, and run the two-process two-backend determinism conformance test (Testing
Strategy) on every CI run; share the canonicalization decision with
[RFC-0010 §Open Questions](RFC-0010-artifact-checkpoint-format.md).

## References

- [RFC-0004 — Data, Sovereignty & Provenance](RFC-0004-data-provenance.md) (the data layer §2, the
  provenance-commitment requirements §4, contribution accounting §5, data-quality metadata §6).
- [RFC-0006 — Verifiable Contribution](RFC-0006-verifiable-contribution.md) (the Phase-2 proofs these
  commitments bind to §2, the proof-ready requirements §3, the roadmap §7).
- [RFC-0003 — Federated Training Protocol](RFC-0003-federated-protocol.md) (the `Commitment` message §8,
  the `PseudoGradient.dataset_root` binding §3, ingress rejection §9).
- [RFC-0010 — Checkpoint & Artifact Format](RFC-0010-artifact-checkpoint-format.md) (the global-model
  content hash recorded in the ledger; canonical-byte hashing, `INV-CHECKPOINT-HASH`).
- [RFC-0007 — WMCP Latent Contract & Embodiment Adapters](RFC-0007-wmcp-latent-contract.md) (the
  `wmcp_version` and `embodiment_id`/`ActionSpec` metadata on a `DatasetCommitment`).
- [RFC-0002 — The Latent Gauge & Frame-Anchored Aggregation](RFC-0002-gauge-and-aggregation.md) (the
  public-recomputation synergy that, like inclusion-proof verification, needs no prover).
- [03 — Data Model](../spec/03-data-model.md) (`Episode`, `DatasetCommitment`, `ContributionRecord`
  schemas) · [04 — Error Model](../spec/04-error-model.md) (`ProvenanceError`, `CommitmentMismatch`,
  `MerkleVerificationError`) · [06 — Security](../spec/06-security.md) (the provenance boundary) ·
  [07 — Testing Strategy](../spec/07-testing-strategy.md) (Merkle + binding tests).
- External: Merkle (1980) hash trees; the CVE-2012-2459 Merkle-duplication malleability bug (the reason
  for odd-node promotion + domain separation); Poseidon2 (the candidate STARK-friendly hash, Stage D);
  Stwo / Circle-STARK (the Phase-2 prover, [RFC-0006](RFC-0006-verifiable-contribution.md)).
