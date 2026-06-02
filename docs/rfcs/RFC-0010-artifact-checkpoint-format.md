# RFC-0010 — Checkpoint & Artifact Format

| | |
|---|---|
| **RFC** | 0010 |
| **Title** | Checkpoint & Artifact Format |
| **Slug** | artifact-checkpoint-format |
| **Status** | Accepted |
| **Track** | Standards |
| **Authors** | @AbdelStark |
| **Created** | 2026-06-02 |
| **Target milestone** | v0.1 (Stage A) |
| **Area** | artifacts |
| **Requires** | RFC-0001 |
| **Informs** | RFC-0003, RFC-0006, RFC-0009, RFC-0013, RFC-0014, RFC-0015 |

## Summary

This RFC specifies the on-disk format of a Lensemble model artifact: the bundle that holds the shared,
federated parameters $(\theta_t, \phi_t)$ — encoder $f_\theta$ and predictor $g_\phi$ — at a given
round $t$. The format is two files: a `safetensors` weight payload (tensors only, no pickle) plus a
sidecar JSON header validated by a pydantic v2 model carrying an integer `schema_version` and a
SHA-256 `content_hash` over a canonical byte serialization of the weights. The `content_hash` is the
single value that flows through the federated protocol — it is committed in `RoundClose`
([RFC-0003 §8](RFC-0003-federated-protocol.md#8-message-summary)) and is what `INV-CHECKPOINT-HASH` binds. A
`parent_hash` field chains each round's artifact to the previous one, giving the auditable model-version
history that the Phase-2 verifiable layer ([RFC-0006 §5](RFC-0006-verifiable-contribution.md#5-the-composed-trust-statement-per-roadmap-stage)) recomputes
over. Action heads $h_\psi^{(c)}$ are never serialized into a shared artifact (`INV-ACTIONHEAD-LOCAL`).
The stable data-model schema for this artifact (`CheckpointHeader`) is at
[03-data-model.md §10](../spec/03-data-model.md#10-modelartifact--checkpoint); this RFC owns the byte-level format, the canonicalization, the hashing
procedure, and the load/verify lifecycle. The artifact module is `lensemble.artifacts`
(`checkpoint.py`, `schema.py`, `hashing.py`), public surface per [conventions §1](../spec/conventions.md#1-repository-and-package-layout).

## Motivation

The artifact is the unit of trust in the federation. Three forces converge on its format:

1. **Aggregation reproducibility and proof-readiness.** The outer step is bitwise-deterministic
   (`INV-AGG-DETERMINISM`, [RFC-0001 §6](RFC-0001-architecture.md#6-trust-boundaries)). A verifier — public recomputation
   in Phase 1, a STARK circuit in Phase 2 — must be able to load the exact bytes the coordinator
   committed, recompute their hash, and get an identical value on a different OS, CPU architecture, and
   torch build. This forces a canonical, platform-stable byte serialization with no implementation
   freedom, and rules out any format whose bytes depend on object identity, dict ordering, or pickle
   protocol.

2. **Tamper-evidence and supply-chain safety.** A checkpoint crossing a boundary or sitting in shared
   storage must be detectably modified. A content hash recomputed on every load gives this for free,
   provided the serialization is deterministic. The same property forbids `pickle`/`torch.save`:
   loading a pickle executes arbitrary code, and the byte stream is non-deterministic, so it is unfit
   both as a security boundary and as a hash input.

3. **Schema evolution without breakage.** The header schema will change pre-1.0 ([conventions §10](../spec/conventions.md#10-versioning-and-schema-policy)). Readers must
   accept older artifacts via registered migrations and fail closed — not guess — on versions newer than
   they understand.

No existing torch checkpoint convention satisfies all three (the default `torch.save` fails 1 and 2),
so the format is specified here. The latent gauge result ([RFC-0002](RFC-0002-gauge-and-aggregation.md))
keeps aggregation a near-linear weighted average, which is what makes a single committed hash per round
a meaningful and cheap proving target ([RFC-0006 §3](RFC-0006-verifiable-contribution.md#3-phase-1-proof-ready-requirements-cheap-to-honor-now)); the artifact
format is the substrate that carries that commitment.

## Goals

- Specify the **two-file artifact**: a `safetensors` weight payload + a JSON `CheckpointHeader` sidecar,
  with every header field typed (the schema mirrors [03-data-model.md §10](../spec/03-data-model.md#10-modelartifact--checkpoint)).
- Specify a **canonical byte serialization** and the **SHA-256 hashing procedure** over it such that the
  `content_hash` is identical across OS, CPU architecture, and torch build (`INV-CHECKPOINT-HASH`).
- Define the **hash chain** (`parent_hash`) that links round artifacts into an auditable model history,
  serving the proof-ready requirement of committed model versions
  ([RFC-0006 §3](RFC-0006-verifiable-contribution.md#3-phase-1-proof-ready-requirements-cheap-to-honor-now)).
- Define **schema versioning + migration** (forward-compatible readers, ordered migration chain) and the
  two distinct failure errors: `SchemaVersionMismatch` and `CheckpointIntegrityError` ([conventions §6](../spec/conventions.md#6-error-taxonomy)).
- Enforce `INV-ACTIONHEAD-LOCAL` at the serialization boundary: per-embodiment action heads are never in
  a shared artifact; an attempt is a `ResidencyViolation`.
- Provide the `save`/`load`/`verify` API and its determinism, error-propagation, and
  fail-closed-on-tamper contract.

## Non-Goals

- This RFC does not specify episode/dataset commitments or Merkle construction (that is
  [RFC-0014](RFC-0014-provenance-commitments.md)); it only reuses the same canonical commitment hash
  (SHA-256, [conventions §11](../spec/conventions.md#11-external-dependencies)) so the two commitment surfaces agree.
- It does not specify the federated message wire format (`RoundOpen`/`Update`/`Commitment`/`RoundClose`
  are [RFC-0003 §8](RFC-0003-federated-protocol.md#8-message-summary)); it specifies only the artifact whose hash the
  `RoundClose` message carries.
- It does not specify the `RunManifest` ([RFC-0009](RFC-0009-configuration-reproducibility.md)) or the
  config-hash procedure; the artifact header merely references `config_hash` by value.
- It does not specify the STARK proof circuit or TEE attestation (Phase 2,
  [RFC-0006](RFC-0006-verifiable-contribution.md)); it specifies the proof-*ready* artifact disciplines
  that land in v0.1.
- Action-head local persistence (a participant saving its own $h_\psi^{(c)}$ to its private store) is a
  participant-runtime concern ([RFC-0013](RFC-0013-coordinator-runtime.md)); this RFC only forbids action
  heads from entering a *shared* artifact.

## Proposed Design

### 1. Artifact layout on disk

A model artifact is a directory (or a tar of one) containing exactly:

```
<artifact>/
  weights.safetensors        # tensors only; encoder + predictor param groups
  header.json                # CheckpointHeader (pydantic v2), schema-versioned
```

Large models MAY shard weights across `weights-00000-of-000NN.safetensors` files plus a
`weights.index.json` (the standard `safetensors` shard index mapping tensor name → shard file). When
sharded, the canonical hash (§4) is computed over the concatenation of shards in ascending shard index,
so a sharded and an unsharded artifact of the same weights produce the **same** `content_hash`. The
header is always a single `header.json`; it is the entry point a loader reads first.

Weights store only the **federated, shared** parameter groups — `encoder` ($\theta$) and `predictor`
($\phi$). Per-embodiment action heads $h_\psi^{(c)}$ are excluded by construction
(`INV-ACTIONHEAD-LOCAL`, §6). Master weights are fp32 ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)); the artifact stores the fp32 master
copy, not the bf16 compute view, so a reload is exact.

### 2. The `CheckpointHeader` schema

The header is the pydantic v2 model below (the canonical data-model definition is
[03-data-model.md §10](../spec/03-data-model.md#10-modelartifact--checkpoint); reproduced here as the format owner). It is JSON, UTF-8, with an integer
`schema_version` as the first-validated field.

```python
from datetime import datetime
from pydantic import BaseModel, ConfigDict

class CheckpointHeader(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int                  # on-disk schema version ([conventions §10](../spec/conventions.md#10-versioning-and-schema-policy)); validated before body
    content_hash: str                    # SHA-256 (lowercase hex, 64 chars) over canonical weight bytes
    parent_hash: str | None              # previous round's content_hash; None at round 0
    wmcp_version: str                    # pinned latent-contract version (INV-WMCP)
    round_index: int                     # the round t these params belong to (>= 0)
    config_hash: str                     # the RunManifest config content hash that produced them
    param_groups: tuple[str, ...]        # e.g. ("encoder", "predictor"); action heads NEVER included
    tensor_manifest: tuple["TensorEntry", ...]  # per-tensor name/dtype/shape/group, sorted by name
    weight_files: tuple[str, ...]        # ("weights.safetensors",) or ordered shard list
    created_at: datetime                 # UTC, RFC 3339

class TensorEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str                            # fully-qualified parameter name
    group: str                           # one of param_groups
    dtype: str                           # canonical dtype token, e.g. "float32", "bfloat16"
    shape: tuple[int, ...]               # tensor shape
```

| Field | Type | Domain | Meaning |
|---|---|---|---|
| `schema_version` | `int`, `>= 1` | version | header schema version; checked first (§7) |
| `content_hash` | `str` (64 hex) | hash | canonical SHA-256 of the weight bytes; the committed value |
| `parent_hash` | `str \| None` | hash | prior checkpoint's `content_hash`; `None` only at round 0 |
| `wmcp_version` | `str` | tag | pinned latent contract these params target (`INV-WMCP`) |
| `round_index` | `int`, `>= 0` | count | round $t$; `0` for the warm-start / Stage-A start |
| `config_hash` | `str` | hash | binds the artifact to the `RunManifest` that produced it |
| `param_groups` | `tuple[str,…]` | labels | shared groups stored: `encoder`/`predictor` only |
| `tensor_manifest` | `tuple[TensorEntry,…]` | — | every stored tensor's name/group/dtype/shape, name-sorted |
| `weight_files` | `tuple[str,…]` | filenames | one entry, or the ordered shard list |
| `created_at` | `datetime` | UTC RFC 3339 | artifact creation time (informational; not hashed) |

`tensor_manifest` is redundant with the `safetensors` file metadata by design: it lets a verifier check
the param-group disposition (and thus `INV-ACTIONHEAD-LOCAL`) and the dtype/shape contract **without
loading tensor data**, and it is part of the canonical hash input (§4) so the structural shape of the
artifact is itself tamper-evident.

### 3. Module responsibilities

- `lensemble.artifacts.schema` — the `CheckpointHeader` / `TensorEntry` pydantic models, the
  `SCHEMA_VERSION` constant, and the ordered migration chain (`migrate_vN_to_vN1`).
- `lensemble.artifacts.hashing` — the canonical byte serialization (§4) and `content_hash(...)` /
  `verify_hash(...)`. This is where `INV-CHECKPOINT-HASH` is enforced; it depends on nothing above L3
  ([conventions §1](../spec/conventions.md#1-repository-and-package-layout) / [RFC-0001 §3](RFC-0001-architecture.md#3-dependency-layering-no-cycles)).
- `lensemble.artifacts.checkpoint` — the public `save`/`load`/`verify` API (§5), gluing weights +
  header, calling hashing, and raising the typed errors of [conventions §6](../spec/conventions.md#6-error-taxonomy).

### 4. Canonical byte serialization and hashing (`INV-CHECKPOINT-HASH`)

The `content_hash` MUST be a deterministic function of the model parameters and the structural header
fields, independent of the host OS, CPU endianness/architecture, torch build, number of shards, and
Python dict iteration order. The procedure:

1. **Tensor normalization.** Each tensor is moved to CPU, made contiguous, and serialized as its raw
   element bytes in **little-endian** with its declared on-disk dtype (fp32 master weights; the dtype is
   recorded in `TensorEntry.dtype`). No device/layout/stride metadata enters the byte stream — only
   element bytes.
2. **Deterministic ordering.** Tensors are concatenated in ascending order of fully-qualified
   `TensorEntry.name` (byte-wise UTF-8 comparison), which is the same order `tensor_manifest` is sorted
   in. Sharding does not affect ordering: the manifest order, not the file layout, defines the hash
   input.
3. **Per-tensor framing.** Each tensor contributes a domain-separated frame:
   `len(name)‖name‖dtype_token‖rank‖shape_dims‖element_bytes`, all integers little-endian fixed-width,
   so that a rename or reshape changes the hash even if the raw bytes coincide.
4. **Structural header bind.** A trailing canonical frame binds the structural header fields that define
   *which* parameters these are: `schema_version`, `wmcp_version`, `round_index`, `parent_hash` (empty
   string when `None`), `param_groups` (sorted), and the ordered `weight_files`. `content_hash`,
   `config_hash`, and `created_at` are **excluded** from the hash input — `content_hash` because it is the
   output, `config_hash`/`created_at` because they are provenance pointers carried alongside, not part of
   the weight identity.
5. **Digest.** `content_hash = SHA-256(stream)` rendered as 64 lowercase hex characters.

The canonical commitment hash is **SHA-256** ([conventions §11](../spec/conventions.md#11-external-dependencies): conservative, interoperable, the same primitive
[RFC-0014](RFC-0014-provenance-commitments.md) uses for episode/Merkle leaves, so artifact and dataset
commitments share one algorithm). Domain separation distinguishes an artifact hash from a Merkle leaf
hash so the two surfaces never collide.

```python
def content_hash(weights: Mapping[str, "Tensor"], header_fields: "StructuralFields") -> str: ...
    # deterministic across OS/arch/torch build/shard count (INV-CHECKPOINT-HASH)
def verify_hash(artifact_dir: Path) -> None:  # raises CheckpointIntegrityError on mismatch
    ...
```

`INV-CHECKPOINT-HASH` is enforced here: the value returned by `content_hash` over the loaded weights and
structural fields MUST equal the header's `content_hash`, which MUST equal the hash committed in
`Commitment`/`RoundClose` ([RFC-0003 §8](RFC-0003-federated-protocol.md#8-message-summary)). A divergence at any of these
three equalities is a tamper/corruption signal and raises `CheckpointIntegrityError` (§7).

`RISK:` step 1/2/3 fix endianness, ordering, framing, and contiguity, but a residual risk is a torch or
`safetensors` version that changes the default storage layout of an exotic dtype (e.g. a sub-byte or
complex dtype) such that "raw element bytes" is ambiguous. Resolution plan: the artifact restricts stored
dtypes to `{float32, bfloat16, float16}` for v0.1 (the only dtypes the model uses, [conventions §9](../spec/conventions.md#9-determinism-dtype-device)) and a
cross-platform hash-stability test (Testing Strategy) is a release gate; a new dtype requires extending
the canonical-dtype-token table and re-pinning the test fixtures. Tracked in the Open Question on
canonical-byte ordering.

### 5. Save / load / verify lifecycle

```python
# lensemble.artifacts.checkpoint
def save_checkpoint(
    artifact_dir: Path,
    weights: Mapping[str, "Tensor"],   # encoder + predictor only
    *,
    wmcp_version: str,
    round_index: int,
    config_hash: str,
    parent_hash: str | None,
    param_groups: tuple[str, ...] = ("encoder", "predictor"),
    shard_size_bytes: int | None = None,
) -> str: ...                          # returns the content_hash (the value to commit)

def load_checkpoint(artifact_dir: Path) -> tuple[dict[str, "Tensor"], CheckpointHeader]: ...
    # validates schema_version, migrates if older, verifies hash, then loads tensors

def verify(artifact_dir: Path, expected_hash: str | None = None) -> CheckpointHeader: ...
    # header-only + hash check; if expected_hash is given it must equal header.content_hash
```

**Save** (coordinator on `COMMITTING`, [RFC-0013](RFC-0013-coordinator-runtime.md); single-site on a
checkpoint cadence): validate that `weights` contains only `param_groups` tensors and **no** action-head
parameters (`INV-ACTIONHEAD-LOCAL`, §6) — a violation raises `ResidencyViolation` (security-critical,
never swallowed) **before** any bytes are written. Write `safetensors` (optionally sharded), build the
`tensor_manifest`, compute `content_hash` (§4), write `header.json`, and return the hash. The returned
hash is exactly what the coordinator puts in `RoundClose`.

**Load** (any consumer — `evaluate`, the outer step reading the prior global model, the public verifier):
read `header.json` first; validate `schema_version` and migrate if older, fail closed if too new (§7);
recompute the canonical hash over the weight bytes and assert it equals `header.content_hash`
(`INV-CHECKPOINT-HASH`) — mismatch raises `CheckpointIntegrityError` and **no tensors are loaded into
memory**; only then deserialize the `safetensors` tensors. `safetensors` deserialization is mmap-able and
never executes code; a `pickle`/`torch.save` payload presented at this path is rejected (§7,
`CheckpointIntegrityError` with a "not a safetensors artifact" remediation) — the loader never falls back
to `torch.load`.

**Verify** is the header-and-hash-only path used by public recomputation
([RFC-0006 §4](RFC-0006-verifiable-contribution.md#4-public-recomputation-phase-1-free-in-scope-now), `lensemble.verify.recompute`) and by ingress checks:
it confirms integrity and, if given, that the artifact matches a committed `expected_hash`, without
materializing tensors for downstream compute.

### 6. `INV-ACTIONHEAD-LOCAL` at the serialization boundary

Per-embodiment action heads $h_\psi^{(c)}$ are local and never aggregated
([RFC-0001 §4](RFC-0001-architecture.md#4-federation-map)). The artifact format makes this a *serialization* invariant,
not only a federation-path invariant: `save_checkpoint` rejects any tensor whose param group is not in
the allowed shared set `{encoder, predictor}`. Because action-head parameters are private model state of
private embodiments, emitting one into a *shared* artifact is treated as a residency breach
(`ResidencyViolation`, fail-closed) — consistent with `INV-RESIDENCY` and the data-model rule that
`param_groups` never contains an action head ([03-data-model.md §10](../spec/03-data-model.md#10-modelartifact--checkpoint)). A participant persisting its
own action head to its own private store uses a separate, unshared local-state path
([RFC-0013](RFC-0013-coordinator-runtime.md)) that this RFC does not govern.

### 7. Failure modes and error propagation

All errors derive from `LensembleError` and carry `.code` (a `LensembleErrorCode`) and `.remediation`
([conventions §6](../spec/conventions.md#6-error-taxonomy)). The artifact errors are the `ArtifactError` subtree (`SchemaVersionMismatch`,
`CheckpointIntegrityError`) plus `ResidencyViolation` on the save path.

| Failure | Trigger | Detection | Error ([conventions §6](../spec/conventions.md#6-error-taxonomy)) | System response |
|---|---|---|---|---|
| **Schema too new / unknown** | header `schema_version` exceeds reader max | pydantic reads `schema_version` before the body | `SchemaVersionMismatch` (`SCHEMA_VERSION_MISMATCH`), carries `file_schema_version`, `reader_max_version` | **fail-closed**; load aborts; remediation names the upgrade path. Never silently coerces ([04-error-model.md §5.7](../spec/04-error-model.md#57-artifacts-lensembleartifacts)) |
| **Schema older (known)** | header `schema_version` < reader max | version compare on read | none (handled) | run the ordered migration chain `migrate_v(k)…→v(m)`, then proceed |
| **Hash mismatch / tamper** | recomputed canonical hash ≠ `header.content_hash` (or ≠ committed `RoundClose` hash) | hash recomputation on **every** load | `CheckpointIntegrityError` (`CHECKPOINT_INTEGRITY`), carries `expected_hash`, `got_hash` | **fail-closed**; tensors are not loaded; enforces `INV-CHECKPOINT-HASH`; never swallowed |
| **Not a safetensors artifact** | a `pickle`/`torch.save`/`npz` payload presented at the weight path | format sniff before deserialize | `CheckpointIntegrityError` (`CHECKPOINT_INTEGRITY`) with a "safetensors-only" remediation | **fail-closed**; the loader never executes a pickle; refuses load |
| **Action head in shared artifact** | a tensor outside `{encoder,predictor}` passed to `save_checkpoint` | param-group check before any write | `ResidencyViolation` (`RESIDENCY_VIOLATION`) | **fail-closed**; nothing written; security-critical, never caught-and-ignored (`INV-ACTIONHEAD-LOCAL`, `INV-RESIDENCY`) |
| **Malformed header** | header JSON fails pydantic validation (missing/extra field, bad hex length) | pydantic `extra="forbid"`, field validators | `ArtifactError` (`ARTIFACT_INVALID`) | **fail-closed**; load aborts with the validation detail |
| **Shard/index inconsistency** | `weight_files` / `weights.index.json` disagree with present files | manifest vs filesystem check on load | `CheckpointIntegrityError` (`CHECKPOINT_INTEGRITY`) | **fail-closed**; refuse load |

`SchemaVersionMismatch` (a *version* problem) and `CheckpointIntegrityError` (a *bytes* problem) are
deliberately distinct: the first is recoverable by upgrading the reader or migrating; the second is a
tamper/corruption signal with no in-band recovery. Validation happens at the load boundary, before any
tensor materialization, so a corrupt artifact never enters compute.

### 8. Determinism and concurrency

- **Determinism.** `content_hash` is a pure function of weights + structural fields (§4) and is required
  to be bitwise-reproducible across hosts; this is the artifact-side leg of `INV-AGG-DETERMINISM`
  ([RFC-0001 §6](RFC-0001-architecture.md#6-trust-boundaries)) and the proof-readiness requirement on committed model
  versions ([RFC-0006 §3](RFC-0006-verifiable-contribution.md#3-phase-1-proof-ready-requirements-cheap-to-honor-now)). `created_at` and any non-hashed field
  do not affect the hash, so two coordinators committing the same weights at different wall-clock times
  produce the same `content_hash`.
- **Concurrency.** Save is single-writer per `artifact_dir` (the coordinator owns commitment,
  [RFC-0013](RFC-0013-coordinator-runtime.md)); the directory is written to a temporary path and
  atomically renamed so a reader never observes a half-written artifact. Loads are concurrent and
  read-only (mmap), and the hash check makes a partially-written or truncated artifact a
  `CheckpointIntegrityError` rather than a silent partial read.

### 9. Proof-ready tie-in

This format satisfies two of the five Phase-1 proof-ready requirements
([RFC-0006 §3](RFC-0006-verifiable-contribution.md#3-phase-1-proof-ready-requirements-cheap-to-honor-now)) directly:

- **Committed model versions**: every $(\theta_t,\phi_t)$ is a hash-committed artifact; the
  `content_hash` is the commitment carried in `RoundClose`.
- **Reproducible / recomputable inputs**: the canonical hash lets a verifier load the exact
  committed bytes and recompute alignment on the public probe
  ([RFC-0002 §5](RFC-0002-gauge-and-aggregation.md#5-layer-3--procrustes-re-alignment-at-aggregation-backstop)) — the "free" public-recomputation row of the
  provable surface ([RFC-0006 §3](RFC-0006-verifiable-contribution.md#3-phase-1-proof-ready-requirements-cheap-to-honor-now)). `parent_hash` chains the
  artifacts into the model-version history a Phase-2 aggregation STARK proves over. No format change is
  needed to add the Phase-2 proofs.

## Alternatives Considered

**`safetensors` (chosen) vs `pickle` / `torch.save`.** `torch.save` is the default torch checkpoint
container and is convenient — it round-trips arbitrary Python objects and optimizer state. Rejected as the
artifact format because loading a pickle executes arbitrary code (a supply-chain and untrusted-artifact
hazard for a system whose whole point is exchanging artifacts across mutually-distrusting boundaries) and
because its byte stream is non-deterministic (object identity, protocol version, dict ordering), which
breaks `INV-CHECKPOINT-HASH` and the proof-ready determinism requirement. `safetensors` stores only
tensors with a small JSON header, is mmap-able, never executes code, and has a stable byte layout — it is
the format [conventions §11](../spec/conventions.md#11-external-dependencies) pins for exactly these reasons.

**`safetensors` vs `npz`.** `numpy`'s `.npz` is also pickle-free for plain arrays and zip-packaged.
Rejected because it forces a tensor → numpy round-trip (losing `bfloat16`, which numpy lacks natively),
its zip container adds compression and timestamp non-determinism that would have to be canonicalized away,
and it is not mmap-friendly for large models. `safetensors` is purpose-built for tensor weights at scale.

**Metadata in the `safetensors` header vs a sidecar `header.json`.** `safetensors` permits a small
string-keyed metadata map inside its own header. Considered for keeping the artifact a single file.
Rejected as the primary metadata home because the structured `CheckpointHeader` (nested `tensor_manifest`,
typed fields, `schema_version` + migrations) is awkward to express as a flat string map, and validating it
with pydantic v2 ([conventions §8](../spec/conventions.md#8-core-data-types)) is cleaner as standalone JSON. The sidecar is canonical; a small subset
(`content_hash`, `schema_version`) MAY be mirrored into the `safetensors` metadata as a convenience, but
the sidecar is authoritative and is what `load_checkpoint` reads first.

**Hashing the header file bytes vs hashing canonical weight bytes + structural fields (chosen).** A
simpler scheme hashes the literal `header.json` and `weights.safetensors` file bytes. Rejected because
file bytes depend on JSON key ordering, whitespace, shard count, and `safetensors` library version —
none of which change the model — so the hash would be unstable across platforms and library versions and
would not survive re-sharding. The canonical procedure (§4) hashes the *model identity*, not the file
encoding, which is what a cross-platform verifier needs.

**SHA-256 (chosen for Phase 1) vs a STARK-friendly hash (e.g. Poseidon2).** A Poseidon2-style hash would
make the Phase-2 aggregation/commitment circuit far cheaper to prove. Not adopted in Phase 1 because
SHA-256 is conservative, ubiquitous, hardware-accelerated, and interoperable, and Phase-1 verification is
public recomputation (no circuit). The migration is the shared Open Question with
[RFC-0006](RFC-0006-verifiable-contribution.md) and [RFC-0014](RFC-0014-provenance-commitments.md); the
`hash_algorithm` is a versioned choice so the change is a forward-compatible schema bump, not a rewrite.

## Drawbacks

- **`safetensors` stores only tensors**, so structured metadata needs a sidecar — two artifacts to keep
  in sync. This is also the safety win: the tensor payload cannot smuggle executable state, and the
  metadata is validated independently. The atomic-rename write (§8) keeps the pair consistent.
- **A custom canonical hash is a maintenance surface.** The byte procedure (§4) must be re-validated
  whenever a stored dtype or the `safetensors`/torch storage layout changes; the cross-platform
  hash-stability test is therefore a release gate, not an optional check.
- **No optimizer state in the shared artifact.** Inner-loop optimizer state (AdamW moments) and outer
  Nesterov momentum are not part of the shared artifact (they are local/coordinator state), so a
  shared artifact alone does not resume an inner training run; that is intentional (only $(\theta,\phi)$
  cross), but it means a participant rejoiner reconstructs optimizer state locally
  ([RFC-0013](RFC-0013-coordinator-runtime.md)).

## Migration / Rollout

- The format ships in **v0.1 (Stage A)**: `save`/`load`/`verify`, SHA-256 canonical hashing, the header
  schema, and the round-0 warm-start artifact whose `content_hash` equals the pinned warm-start hash
  (`INV-WARMSTART-T0`, [RFC-0001 §4](RFC-0001-architecture.md#4-federation-map)).
- **Hash chaining via `parent_hash`** is used from v0.2 (Stage B) when the federated round loop produces
  a sequence of committed global models.
- **Schema migration**: readers handle `schema_version <= SCHEMA_VERSION`; each bump adds one
  `migrate_vN_to_vN1` to the ordered chain and a round-trip migration test ([07-testing-strategy.md §210](../spec/07-testing-strategy.md#210-schema-round-trip-and-migration)).
  Pre-1.0 the schema may change with a bump and a migration; there is no removal of an old reader path
  within a major line ([09-release-and-versioning.md §4.1](../spec/09-release-and-versioning.md#41-artifact-schema_version-integer)). A `schema_version` bump is a
  Keep-a-Changelog `Changed` entry with the `area:artifacts` label.
- **Hash-algorithm migration** (SHA-256 → STARK-friendly) is a Stage-D / Phase-2 change carried by a
  versioned `hash_algorithm` field, coordinated with
  [RFC-0014](RFC-0014-provenance-commitments.md) so artifact and dataset commitments migrate together.

## Testing Strategy

The full pyramid is at ([07-testing-strategy.md](../spec/07-testing-strategy.md)); the artifact-owned tests:

- **Round-trip equality.** `save_checkpoint` then `load_checkpoint` returns tensors bitwise-equal to the
  inputs (fp32 master weights) and a header that re-validates; the returned `content_hash` equals the
  header's. Runs on the CPU fallback ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)).
- **Hash determinism across processes/platforms (release gate).** The same weights + structural fields
  produce an identical `content_hash` across two processes, and (in CI's matrix) across OS/arch, across a
  sharded vs unsharded layout, and across the pinned torch/`safetensors` builds — directly exercising
  `INV-CHECKPOINT-HASH` and the §4 `RISK`. Tiny synthetic weights only; no large downloads.
- **Tamper detection.** Flipping a byte in a weight file, reordering tensors, or editing the header's
  `content_hash` causes `load_checkpoint`/`verify` to raise `CheckpointIntegrityError` and to load no
  tensors (fail-closed). A mismatch against a supplied `expected_hash` (the committed `RoundClose` hash)
  also raises.
- **No-pickle assertion.** A `pickle`/`torch.save` payload presented at the weight path is rejected with
  `CheckpointIntegrityError` (safetensors-only remediation); the loader is asserted never to call
  `torch.load`/`pickle.load` (a property test over crafted payloads).
- **Action-head exclusion.** `save_checkpoint` given a tensor outside `{encoder,predictor}` raises
  `ResidencyViolation` before any write (`INV-ACTIONHEAD-LOCAL`), and the test asserts the error is not
  caught-and-ignored anywhere on the path.
- **Schema migration + round-trip.** A `schema_version = k` header loads into a version-`m` reader
  (`k < m`) via the migration chain and re-validates; a too-new version raises `SchemaVersionMismatch`
  with `file_schema_version`/`reader_max_version` populated.
- **Hash agreement with the commitment hash.** A property test asserts the artifact `content_hash`
  algorithm matches the SHA-256 primitive [RFC-0014](RFC-0014-provenance-commitments.md) uses, with
  distinct domain separation so an artifact hash can never equal a Merkle-leaf hash of the same bytes.

Numerical-tolerance policy ([07-testing-strategy.md](../spec/07-testing-strategy.md)): the hash and the fp32 master-weight
round-trip require **exact** equality (no atol/rtol); approximate tolerances apply only to compute paths,
not to the artifact bytes.

## Open Questions

OPEN QUESTION: The **exact canonical-byte ordering and framing** for guaranteed cross-platform hash
stability across all stored dtypes (the §4 `RISK`). The v0.1 scheme is little-endian element bytes,
name-sorted, domain-separated per-tensor framing, restricted to `{float32, bfloat16, float16}`. Owner
@AbdelStark; resolution path: the cross-platform hash-stability test is a v0.1 release gate, and any new
stored dtype extends the canonical-dtype-token table behind a `schema_version` bump (Stage A, hardened
through v1.0).

OPEN QUESTION: **Migrating the commitment hash from SHA-256 to a STARK-friendly hash** (e.g. Poseidon2)
to keep the Phase-2 proof circuit cheap. Shared with
[RFC-0006](RFC-0006-verifiable-contribution.md) and [RFC-0014](RFC-0014-provenance-commitments.md). Owner
@AbdelStark; resolution path: Stage D — carried by the versioned `hash_algorithm` field so the change is
a forward-compatible schema bump applied to artifact and dataset commitments together.

OPEN QUESTION: Whether a **subset of header fields** (`content_hash`, `schema_version`) should be mirrored
into the `safetensors` internal metadata map as a redundancy/self-description convenience, or kept solely
in the sidecar. Owner @AbdelStark; resolution path: decide during Stage A based on operator ergonomics;
the sidecar remains authoritative either way, so this does not affect the contract.

## References

- [RFC-0001 — Architecture & System Overview](RFC-0001-architecture.md): §4 federation map and
  `INV-ACTIONHEAD-LOCAL`; §6 trust boundaries and `INV-AGG-DETERMINISM`; §7 data-flow lifecycles (dataset commit).
- [RFC-0002 — The Latent Gauge & Frame-Anchored Aggregation](RFC-0002-gauge-and-aggregation.md): §5 the
  recomputable Procrustes alignment that a verifier checks against committed artifacts.
- [RFC-0003 — Federated Training Protocol](RFC-0003-federated-protocol.md): §8 the message table —
  `RoundClose` carries the artifact `content_hash`, `Commitment` carries the dataset root.
- [RFC-0006 — Verifiable Contribution](RFC-0006-verifiable-contribution.md): §2 the provable surface; §3
  the Phase-1 proof-ready requirements (committed model versions, reproducible outer step) this format
  satisfies.
- [RFC-0009 — Configuration, Run Manifest & Reproducibility](RFC-0009-configuration-reproducibility.md):
  the `config_hash` and `RunManifest` the header references.
- [RFC-0013 — Coordinator & Participant Runtime](RFC-0013-coordinator-runtime.md): the `COMMITTING` state
  that calls `save_checkpoint`; rejoiner local-state recovery.
- [RFC-0014 — Provenance Commitments & Merkle Scheme](RFC-0014-provenance-commitments.md): the shared
  SHA-256 commitment primitive and the STARK-friendly-hash migration.
- [RFC-0015 — Observability, Diagnostics & Telemetry](RFC-0015-observability-diagnostics.md): logging of
  artifact hashes/sizes (hashes and norms are permitted; raw tensors are not — `INV-RESIDENCY`).
- Spec: [03-data-model.md §10](../spec/03-data-model.md#10-modelartifact--checkpoint) (`CheckpointHeader`); [04-error-model.md §5.7](../spec/04-error-model.md#57-artifacts-lensembleartifacts) (artifact
  failure modes); [09-release-and-versioning.md §4.1](../spec/09-release-and-versioning.md#41-artifact-schema_version-integer) (artifact `schema_version` policy);
  [06-security.md](../spec/06-security.md) (no-pickle, supply-chain); [07-testing-strategy.md](../spec/07-testing-strategy.md).
- `safetensors` (`>=0.4`, [conventions §11](../spec/conventions.md#11-external-dependencies)) — safe, mmap-able, pickle-free tensor serialization.
- `pydantic` (`>=2,<3`, [conventions §11](../spec/conventions.md#11-external-dependencies)) — typed validation of the JSON header with an explicit `schema_version`.
- `hashlib` SHA-256 (stdlib, [conventions §11](../spec/conventions.md#11-external-dependencies)) — the Phase-1 canonical commitment hash.
- Stwo — the Circle-STARK prover that, in Phase 2, proves over the committed-artifact hash chain.
