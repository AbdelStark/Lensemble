# 03 — Data Model

This section is the canonical, typed reference for every core data type in Lensemble: its schema, its
units, the invariants it enforces, the validation that runs at its boundary, and the error that fires
on violation. Prose-only contracts are not permitted here — every type below is given as a Python
dataclass or pydantic v2 model sketch.

The data model is partitioned by *where the bytes live*:

1. **In-memory training/runtime objects** — `LatentState`, `ActionSpec`, `Episode`, `Transition`,
   `Window`, `PseudoGradient`, `GlobalState`, `RoundState`. These are not (all) serialized; some cross a
   trust boundary as messages (typed in [RFC-0003 §8](../rfcs/RFC-0003-federated-protocol.md#8-message-summary)).
2. **On-disk metadata** — `DatasetCommitment`, `ModelArtifact`/`Checkpoint` header, `RunManifest`,
   `EvalReport`, `FrameDriftReport`, `ContributionRecord`. These are pydantic v2 JSON documents carrying
   an integer `schema_version` (see [§14](#14-serialization-rules) and
   [§15](#15-schema-versioning-and-migration)).

Rationale for the schemas below is held in the owning RFCs; this section is the stable contract. The
authoritative WMCP latent contract is [RFC-0007](../rfcs/RFC-0007-wmcp-latent-contract.md); the
artifact/hashing detail is [RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md); commitment
construction is [RFC-0014](../rfcs/RFC-0014-provenance-commitments.md); the config/manifest detail is
[RFC-0009](../rfcs/RFC-0009-configuration-reproducibility.md).

Notation ($d$, $N$, $\theta$, $\phi$, $\Delta_c$, $R_c$, $s_t$, $f_{\text{ref}}$) is exactly as defined
in the canonical notation table; this section does not redefine symbols.

---

## 1. Type-layer conventions

- **In-memory tensors** are `torch.Tensor`. Their shape, dtype, and device contracts are stated per type.
  Default compute dtype is bf16 forward; fp32 master weights and statistic accumulation
  (see [09-release-and-versioning.md](09-release-and-versioning.md) for the numerical contract pointer
  and [RFC-0008 §7](../rfcs/RFC-0008-model-objective-numerics.md#7-numerical-contract)).
- **Runtime dataclasses** that hold tensors are plain `@dataclass` (frozen where they are values, not
  buffers). They are validated by an explicit `validate()` or by their constructor.
- **On-disk metadata** are `pydantic.BaseModel` subclasses with `model_config = ConfigDict(frozen=True,
  extra="forbid")`, serialized to canonical JSON, every one carrying `schema_version: int`.
- **Structured configuration** (`LensembleConfig`, its groups) are frozen dataclasses materialized via
  OmegaConf/Hydra; they are not pydantic (see [§14](#14-serialization-rules)).
- Every validation failure raises a typed error from the taxonomy in
  [04-error-model.md](04-error-model.md); each error carries `.code` (a `LensembleErrorCode`) and a
  human-actionable `.remediation` string. No validation path raises a bare `Exception`.

---

## 2. `LatentState` — the WMCP latent contract (summary)

The single value type that crosses the encoder→predictor boundary inside every participant and defines
what is comparable across silos. The full contract is normative in
[RFC-0007 §2](../rfcs/RFC-0007-wmcp-latent-contract.md#2-the-latentstate-contract); this is the summary the data model pins.

```python
@dataclass(frozen=True)
class LatentState:
    tokens: Tensor          # shape (N, d); per-clip latent tokens emitted by f_theta
    num_tokens: int          # N
    dim: int                 # d
    wmcp_version: str        # e.g. "wmcp-1.0.0"; the pinned contract version
    # dtype: bf16 in-flight, fp32 for cross-silo statistics; device: as produced
    # semantics: token i is the latent of spatiotemporal patch group i; order is contract-fixed
```

The authoritative `LatentState` schema is owned by
[RFC-0007 §2](../rfcs/RFC-0007-wmcp-latent-contract.md#2-the-latentstate-contract); this is the
data-model view of the same fields.

| Field | Type | Unit / domain | Meaning |
|---|---|---|---|
| `tokens` | `Tensor (N, d)` | dimensionless latent | the $N$ latent tokens of one clip; $N$, $d$ fixed by the contract |
| `num_tokens` | `int`, `> 0` | count | $N$, the number of latent tokens (matches `tokens.shape[0]`) |
| `dim` | `int`, `> 0` | count | $d$, the latent dimension (matches `tokens.shape[1]`) |
| `wmcp_version` | `str` | semver-like tag | the WMCP version this latent conforms to |

**Invariants.**
- `INV-WMCP` — every `LatentState` conforms to the pinned `wmcp_version`: `tokens.shape == (N, d)`, dtype
  in the contract-permitted set, token ordering as specified. Enforced by the conformance check in
  `lensemble.contracts`. Violation raises `ContractViolation`
  ([04-error-model.md](04-error-model.md)). This check runs at the encoder→predictor boundary and on any
  latent that is about to feed a cross-silo statistic (the probe embeddings $E_{\text{ref}}$, the
  drift diagnostic). It never runs on the wire — `LatentState` of *private* data never serializes
  outbound (`INV-RESIDENCY`, [§12](#12-residency-and-redaction-constraints-cross-cutting)).

**Validation rules.** Shape, dtype, finiteness (`torch.isfinite(tokens).all()`), and `wmcp_version`
equality against the round's pinned contract version. A `NaN`/`Inf` token, a wrong `N` or `d`, or a
version mismatch each raise `ContractViolation` with a remediation pointing at the encoder build or the
contract pin.

---

## 3. `ActionSpec` — per-embodiment action-space descriptor

Describes one embodiment's action space; it is the precondition for constructing that embodiment's
action head $h_\psi^{(c)}$. Full semantics in
[RFC-0007 §3](../rfcs/RFC-0007-wmcp-latent-contract.md#3-the-actionspec-descriptor).

```python
class ActionKind(StrEnum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"

@dataclass(frozen=True)
class ActionSpec:
    embodiment_id: str            # stable identifier, e.g. "unitree-go2", "franka-panda-7dof"
    kind: ActionKind              # continuous | discrete
    dim: int                      # action dimensionality (DoF for continuous; #actions for discrete)
    low: tuple[float, ...] | None  # per-dim lower bound (continuous only); len == dim
    high: tuple[float, ...] | None # per-dim upper bound (continuous only); len == dim
    units: tuple[str, ...]         # per-dim unit label, e.g. ("rad","rad",...,"N"); len == dim
    wmcp_version: str
```

| Field | Type | Unit / domain | Meaning |
|---|---|---|---|
| `embodiment_id` | `str` | identifier | which embodiment this spec describes |
| `kind` | `ActionKind` | enum | continuous vs discrete action space |
| `dim` | `int` | count, `> 0` | number of action dimensions |
| `low`/`high` | `tuple[float] \| None` | action units | per-dim bounds (continuous) |
| `units` | `tuple[str]` | label | physical unit per dimension |
| `wmcp_version` | `str` | tag | contract version the spec is validated against |

**Validation rules.** `dim > 0`; `len(units) == dim`; for `CONTINUOUS`, `low`/`high` are non-`None`,
`len(low) == len(high) == dim`, and `low[i] < high[i]` for all `i`; for `DISCRETE`, `low`/`high` are
`None`. Failure raises `ContractViolation` (`INV-WMCP`: every `ActionSpec` is validated before an action
head is constructed). The validation point is action-head construction in `lensemble.model.action_head`.

**Invariant tie-in.** `INV-ACTIONHEAD-LOCAL` — the head $h_\psi^{(c)}$ built from an `ActionSpec` is
local; it is never broadcast or aggregated. The `ActionSpec` itself may travel in declared quality
metadata, but the head's parameters $\psi$ never cross a boundary. Enforced by the broadcast/aggregate
path excluding $\psi$ ([RFC-0003 §3](../rfcs/RFC-0003-federated-protocol.md)); a $\psi$ tensor reaching
the outbound path is a `ResidencyViolation`.

---

## 4. `Transition` and `Episode` — the data layer

A `Transition` is the atomic learning tuple $(o_t, a_t, o_{t+1})$. An `Episode` is an ordered trajectory
of transitions plus declared metadata. These are the participant-local, residency-bound objects; they
are read through the `stable-worldmodel` data layer and never leave a boundary. Full data-layer spec:
[RFC-0004 §1](../rfcs/RFC-0004-data-provenance.md#1-per-participant-data-layer).

```python
@dataclass(frozen=True)
class Transition:
    obs_t: Tensor          # observation at t; modality-shaped (e.g. video clip C,T,H,W)
    action_t: Tensor       # action applied at t; shape (action_dim,) per the ActionSpec
    obs_tp1: Tensor        # observation at t+1; same modality-shape as obs_t
    # all three are RAW, private, residency-bound — never serialized outbound (INV-RESIDENCY)

@dataclass(frozen=True)
class Episode:
    episode_id: str               # participant-local stable id
    transitions: Sequence[Transition]
    embodiment_id: str             # must match an ActionSpec in scope
    modality: str                  # e.g. "rgb-video"
    action_spec: ActionSpec        # the embodiment action contract for this episode
    collection_meta: Mapping[str, str]  # declared collection conditions (RFC-0004 §7)
```

| Field | Type | Unit / domain | Meaning |
|---|---|---|---|
| `obs_t`, `obs_tp1` | `Tensor` | sensor units | raw observations bracketing one action |
| `action_t` | `Tensor (dim,)` | action units | action at step $t$; `dim == action_spec.dim` |
| `episode_id` | `str` | identifier | local episode key (used as a Merkle-leaf preimage component) |
| `embodiment_id` | `str` | identifier | the embodiment that produced the episode |
| `modality` | `str` | label | observation modality |
| `action_spec` | `ActionSpec` | — | the action contract this episode satisfies |
| `collection_meta` | `Mapping[str,str]` | — | declared, non-private data-quality metadata |

**Invariants.**
- `INV-RESIDENCY` — no `Transition`/`Episode` tensor (`obs_t`, `action_t`, `obs_tp1`) and no embedding
  derived from one is serialized into any outbound message or artifact. Enforced by
  `lensemble.data.residency`; a violating serialization attempt raises `ResidencyViolation`, which is
  **fail-closed and never caught-and-ignored** ([§12](#12-residency-and-redaction-constraints-cross-cutting),
  [04-error-model.md](04-error-model.md)).
- Provenance binding: each `Episode` contributes a leaf to its participant's dataset Merkle tree; the
  resulting root is $R_c$ ([§9](#9-datasetcommitment), `INV-COMMIT-BINDING`).

**Validation rules.** `action_t.shape == (action_spec.dim,)`; `obs_tp1` shares `obs_t`'s modality shape;
`embodiment_id == action_spec.embodiment_id`. Mismatch raises `ContractViolation`. Format/IO faults
during a read raise the data-layer error path (see [RFC-0004 §1](../rfcs/RFC-0004-data-provenance.md#1-per-participant-data-layer));
a probe-vs-private confusion raises `ProbeError`.

---

## 5. `Window` — the training sample

A fixed-length slice the loader yields for next-latent prediction. `num_steps` is a config constant, not
per-sample.

```python
@dataclass(frozen=True)
class Window:
    obs: Tensor            # shape (num_steps + 1, *modality_shape); o_t ... o_{t+num_steps}
    actions: Tensor        # shape (num_steps, action_dim); a_t ... a_{t+num_steps-1}
    num_steps: int         # fixed horizon; equals config data.num_steps
    embodiment_id: str
```

| Field | Type | Unit / domain | Meaning |
|---|---|---|---|
| `obs` | `Tensor (num_steps+1, …)` | sensor units | the observation sequence the predictor rolls over |
| `actions` | `Tensor (num_steps, dim)` | action units | the conditioning actions |
| `num_steps` | `int`, `> 0` | count | fixed window horizon (all windows in a run share it) |
| `embodiment_id` | `str` | identifier | embodiment of the source episode |

**Validation rules.** `obs.shape[0] == num_steps + 1`; `actions.shape[0] == num_steps`;
`actions.shape[1] == action_spec.dim` for the in-scope spec. Failure raises `ContractViolation`. A
`Window` is also residency-bound (`INV-RESIDENCY`): it is a private-data view and never serialized
outbound.

---

## 6. `PseudoGradient` — the one private object that *does* cross the boundary

The DiLoCo outer-loop delta $\Delta_c = (\theta_c^{\text{local}}, \phi_c^{\text{local}}) -
(\theta_t, \phi_t)$, after DP clip+noise, bound to the dataset root under which it was computed. It is the
only participant-derived object permitted across a trust boundary, and it crosses only under secure
aggregation ([RFC-0011](../rfcs/RFC-0011-secure-aggregation.md)) and DP
([RFC-0003 §4](../rfcs/RFC-0003-federated-protocol.md)).

```python
@dataclass(frozen=True)
class PseudoGradient:
    delta: Tensor          # flat fp32 vector; concat of (theta, phi) param-group deltas, fixed order
    l2_norm: float         # ||delta|| computed in fp32 BEFORE noising (post-clip)
    dataset_root: bytes    # the R_c (32-byte SHA-256) this delta is bound to (INV-COMMIT-BINDING)
    round_index: int       # the round t this delta targets
    clipped: bool          # whether the clip projection was applied (||.|| > C_clip)
    quantized: bool        # whether int8 pseudo-gradient quantization was applied on the wire
```

| Field | Type | Unit / domain | Meaning |
|---|---|---|---|
| `delta` | `Tensor` (flat fp32) | param-update units | the $H$-step local update treated as one gradient |
| `l2_norm` | `float`, `>= 0` | L2 | clip-time norm, recorded for the DP-bound check and logging |
| `dataset_root` | `bytes` (32) | hash | the $R_c$ this contribution binds to |
| `round_index` | `int`, `>= 0` | count | target round $t$ |
| `clipped` | `bool` | — | clip projection applied |
| `quantized` | `bool` | — | int8 wire quantization applied (orthogonal to the gauge) |

**Invariants.**
- `INV-DP-BOUND` — after clipping and *before* noising, $\lVert\Delta_c\rVert \le C_{\text{clip}}$.
  Enforced in `lensemble.privacy.dp`: the clip projection
  $\Delta_c \leftarrow \Delta_c \cdot \min(1, C_{\text{clip}}/\lVert\Delta_c\rVert)$ guarantees the bound;
  a post-clip check that finds `l2_norm > C_clip * (1 + tol)` raises `PrivacyBudgetExceeded`'s sibling
  path — concretely a `ConfigError`/assertion in DP if the clip is mis-wired (see
  [RFC-0012](../rfcs/RFC-0012-differential-privacy.md)).
- `INV-COMMIT-BINDING` — every released `PseudoGradient` carries exactly one `dataset_root` $R_c$.
  Enforced at emission and re-checked at aggregation ingress; a delta whose `dataset_root` does not match
  the participant's committed root raises `CommitmentMismatch`, which is **security-critical and never
  swallowed**.
- The `delta` is the only field built from private data; it leaves only post-DP and only masked. The raw
  per-step gradients and the local weights are never serialized (`INV-RESIDENCY`).

**Validation rules.** `delta.dtype == torch.float32`; `delta` is finite; `len(dataset_root) == 32`;
`round_index >= 0`; `l2_norm == ||delta||` recomputed within tolerance at clip time. Non-finite delta or
a wrong-length root raises `AggregationError` / `CommitmentMismatch` at ingress respectively.

---

## 7. `GlobalState` — the broadcast round state

The canonical global model reference plus the per-round public parameters every participant needs to run
a comparable local round. Constructed by the `Coordinator`, broadcast in the `RoundOpen` message
([RFC-0003 §8](../rfcs/RFC-0003-federated-protocol.md#8-message-summary)).

```python
@dataclass(frozen=True)
class GlobalState:
    theta_ref: ParamRef        # reference/hash of encoder params theta_t (NOT the raw tensors broadcast inline)
    phi_ref: ParamRef          # reference/hash of predictor params phi_t
    round_index: int           # t
    sketch_seed: int           # s_t; derives the shared SIGReg projection matrix A
    probe_hash: bytes          # 32-byte SHA-256 of the pinned public probe P content
    wmcp_version: str          # the contract all participants must conform to this round
    # ParamRef = a content hash + a fetch locator for the safetensors artifact (RFC-0010)
```

| Field | Type | Unit / domain | Meaning |
|---|---|---|---|
| `theta_ref`, `phi_ref` | `ParamRef` | hash+locator | the round-$t$ global encoder/predictor artifact references |
| `round_index` | `int`, `>= 0` | count | round $t$ |
| `sketch_seed` | `int` | seed | $s_t$; all participants derive the identical $A$ from it |
| `probe_hash` | `bytes` (32) | hash | content hash of the pinned probe $\mathcal{P}$ |
| `wmcp_version` | `str` | tag | the contract pinned for this round |

**Invariants.**
- `INV-SKETCH-CONSISTENCY` — all participants in round $t$ derive the identical projection matrix $A$
  from the broadcast `sketch_seed` $s_t$. Enforced by deriving $A$ deterministically from $s_t$ in
  `lensemble.model.sigreg`; a participant whose derived $A$ disagrees produces statistics that fail the
  determinism self-check and raise `NonDeterministicAggregation` at the outer step.
- `INV-PROBE-PIN` — `probe_hash` equals the hash committed in `RoundOpen`; landmark targets $t_i$ derive
  only from $f_{\text{ref}}$ (the round-0 encoder). A probe whose recomputed content hash differs raises
  `ProbeError` ([RFC-0004 §3](../rfcs/RFC-0004-data-provenance.md)).
- `INV-WARMSTART-T0` — at `round_index == 0` the `theta_ref` resolves to weights hash-identical to the
  pinned warm-start ($f_{\text{ref}}$). A round-0 encoder hash that differs raises
  `CheckpointIntegrityError`; the gauge is, by construction, closed at $t=0$.

**Validation rules.** `round_index >= 0`; `len(probe_hash) == 32`; `theta_ref`/`phi_ref` content hashes
resolve and match on fetch (`INV-CHECKPOINT-HASH`); `wmcp_version` is one the participant can satisfy
(conformance gate, else `ContractViolation`).

---

## 8. `RoundState` — the coordinator round lifecycle value

The coordinator's per-round bookkeeping; the state-machine semantics (transitions, triggers, failure
handling) are normative in [RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md). The data model pins the
shape.

```python
class RoundPhase(StrEnum):
    OPEN = "open"
    COLLECTING = "collecting"
    AGGREGATING = "aggregating"
    ALIGNING = "aligning"
    COMMITTING = "committing"
    CLOSED = "closed"
    ABORTED = "aborted"

@dataclass
class RoundState:
    round_index: int
    phase: RoundPhase
    global_state: GlobalState
    expected_participants: frozenset[str]
    received: dict[str, PseudoGradient]     # participant_id -> their committed, privatized delta
    commitments: dict[str, bytes]            # participant_id -> R_c
    aggregate: Tensor | None                 # set after AGGREGATING; the deterministic Sum_c Delta_c
    result_hash: bytes | None                # set after COMMITTING; the (theta_{t+1},phi_{t+1}) hash
```

| Field | Type | Domain | Meaning |
|---|---|---|---|
| `phase` | `RoundPhase` | enum | current lifecycle phase |
| `expected_participants` | `frozenset[str]` | ids | who was invited this round |
| `received` | `dict[str, PseudoGradient]` | — | deltas accepted into aggregation |
| `commitments` | `dict[str, bytes]` | id→32B | the $R_c$ each participant bound |
| `aggregate` | `Tensor \| None` | — | the bitwise-deterministic sum (`INV-AGG-DETERMINISM`) |
| `result_hash` | `bytes \| None` | 32B | content hash committed in `RoundClose` |

**Invariants.**
- `INV-AGG-DETERMINISM` — the transition `AGGREGATING → ALIGNING → COMMITTING` is a pure, bitwise-
  reproducible function of (committed deltas, round seed, prior global params): fixed reduction order,
  fp32/fp64 fixed-order summation, no atomics. A determinism self-check runs each outer step; failure
  raises `NonDeterministicAggregation` (abort the round, recompute). Enforced in
  `lensemble.aggregation` and `lensemble.federation.outer_optimizer`.
- `INV-COMMIT-BINDING` — every entry of `received[c]` has a `dataset_root` equal to `commitments[c]`; a
  mismatch rejects the update with `CommitmentMismatch` and the participant is excluded from the round.
- Fault tolerance: a round proceeds with whatever participants are present; if
  `len(received)` falls below the configured threshold the round transitions to `ABORTED` with
  `FaultToleranceExceeded` ([RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md)).

**Validation rules.** Only the transitions enumerated in `RoundPhase` are legal; an illegal transition
raises `RoundError`. `aggregate` is `None` until `AGGREGATING` completes; `result_hash` is `None` until
`COMMITTING` completes.

---

## 9. `DatasetCommitment`

The on-disk, JSON-serialized commitment binding a participant's dataset to a Merkle root $R_c$.
Construction (leaf/node hashing, domain separation, ordering, inclusion proofs) is normative in
[RFC-0014](../rfcs/RFC-0014-provenance-commitments.md).

```python
class DatasetCommitment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: int                 # on-disk schema version ([conventions §10](conventions.md#10-versioning-and-schema-policy))
    merkle_root: str                    # R_c as lowercase hex SHA-256 (Phase 1)
    episode_count: int                  # number of leaves committed
    hash_algorithm: Literal["sha256"]   # Phase-1 canonical; STARK-friendly migration is an open question
    wmcp_version: str                   # contract metadata for the committed data
    embodiment_ids: tuple[str, ...]     # embodiments present in the dataset
    created_at: datetime                # commitment time (UTC, RFC 3339)
```

| Field | Type | Domain | Meaning |
|---|---|---|---|
| `merkle_root` | `str` (64 hex) | hash | $R_c$ over the participant's episode leaves |
| `episode_count` | `int`, `>= 1` | count | number of committed episodes (leaves) |
| `hash_algorithm` | `Literal["sha256"]` | — | the Phase-1 canonical commitment hash |
| `wmcp_version` | `str` | tag | the latent-contract version the data targets |
| `embodiment_ids` | `tuple[str, …]` | ids | declared embodiments |

**Invariants.** `INV-COMMIT-BINDING` — this commitment is the root every `PseudoGradient.dataset_root`
must match. A `PseudoGradient` whose `dataset_root` does not match the participant's committed
`merkle_root` raises `CommitmentMismatch`; a malformed/inconsistent Merkle structure on verification
raises `MerkleVerificationError` (both security-critical, never swallowed).

**Validation rules.** `len(merkle_root) == 64` and hex; `episode_count >= 1`; `hash_algorithm` is the
pinned algorithm. An unknown/too-new `schema_version` raises `SchemaVersionMismatch`
([§15](#15-schema-versioning-and-migration)).

`OPEN QUESTION:` migrate the commitment hash from SHA-256 to a STARK-friendly hash (e.g. Poseidon2) to
keep the Phase-2 proof circuit cheap. Owner @AbdelStark; resolution in
[RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md) / Stage D. Until then `hash_algorithm` is a
versioned field so the migration is a forward-compatible schema change.

---

## 10. `ModelArtifact` / `Checkpoint`

The schema-versioned, hash-committed on-disk model artifact. Weights are stored as `safetensors`; a
sidecar JSON header carries metadata. Full format, canonicalization, and hashing are normative in
[RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md).

```python
class CheckpointHeader(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: int
    content_hash: str                   # SHA-256 (hex) over the canonical weight bytes (INV-CHECKPOINT-HASH)
    parent_hash: str | None             # previous round's content_hash; forms the hash chain
    wmcp_version: str
    round_index: int                    # the round t these params belong to
    config_hash: str                    # the RunManifest config content hash that produced them
    param_groups: tuple[str, ...]       # e.g. ("encoder","predictor"); action heads NEVER included
    created_at: datetime
# weights themselves: safetensors file(s), tensors only, no pickle
```

| Field | Type | Domain | Meaning |
|---|---|---|---|
| `content_hash` | `str` (64 hex) | hash | canonical hash of the weight bytes; the committed value |
| `parent_hash` | `str \| None` | hash | prior checkpoint's `content_hash` (round-0 has `None`) |
| `round_index` | `int`, `>= 0` | count | round $t$ |
| `config_hash` | `str` | hash | binds the artifact to the `RunManifest` that produced it |
| `param_groups` | `tuple[str,…]` | labels | which model parts are stored; encoder/predictor only |

**Invariants.**
- `INV-CHECKPOINT-HASH` — the artifact's recomputed `content_hash` equals the `Commitment`/`RoundClose`
  hash. Enforced on load in `lensemble.artifacts.hashing`: a recomputed hash that differs raises
  `CheckpointIntegrityError` (tamper/corruption; load fails closed).
- `INV-ACTIONHEAD-LOCAL` — `param_groups` never contains an action head; $h_\psi^{(c)}$ params are never
  serialized into a shared artifact. Including one is a `ResidencyViolation`.
- `INV-WARMSTART-T0` — the round-0 encoder artifact's `content_hash` equals the pinned warm-start hash.

**Validation rules.** `safetensors` only — a `pickle`/`torch.save` payload is rejected outright (no
arbitrary-code execution; see [06-security.md](06-security.md) pointer in
[RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md)). Unknown `schema_version` →
`SchemaVersionMismatch`; hash mismatch → `CheckpointIntegrityError`.

---

## 11. `RunManifest`

The reproducibility record emitted by every run (`train_local`, `Coordinator.run`,
`Participant.local_round`, `evaluate`). Full schema and seeding scheme in
[RFC-0009](../rfcs/RFC-0009-configuration-reproducibility.md).

```python
class RunManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: int
    config_hash: str                    # content hash of the resolved LensembleConfig (INV reproducibility)
    root_seed: int                      # the single root seed
    component_seeds: dict[str, int]     # python/numpy/torch/cuda derived seeds
    round_sketch_seeds: dict[int, int]  # t -> s_t = derive(root_seed, t)
    git_sha: str                        # repo commit producing this run
    env: dict[str, str]                 # python/torch/CUDA/driver versions, hardware tags
    dependency_versions: dict[str, str] # pinned versions (torch, numpy, stable-worldmodel, ...)
    probe_hash: str | None              # pinned public probe content hash (federated/eval runs)
    wmcp_version: str
    created_at: datetime
```

| Field | Type | Domain | Meaning |
|---|---|---|---|
| `config_hash` | `str` | hash | content hash of the resolved config tree |
| `root_seed` | `int` | seed | the one seed all others derive from |
| `component_seeds` | `dict[str,int]` | seeds | per-library seeds for reproducibility |
| `round_sketch_seeds` | `dict[int,int]` | seeds | $s_t = \mathrm{derive}(\text{root\_seed}, t)$ |
| `git_sha`, `env`, `dependency_versions` | — | — | the execution context to reproduce the run |
| `probe_hash` | `str \| None` | hash | the pinned probe (when one is in scope) |

**Invariants.** Reproducibility contract — same `LensembleConfig` + same `root_seed` ⇒ identical
`config_hash` and identical derived seeds; the aggregation path is bitwise-reproducible
(`INV-AGG-DETERMINISM`). A run that cannot reproduce its committed `config_hash` is a config/setup error
(`ConfigError`). `round_sketch_seeds` realize `INV-SKETCH-CONSISTENCY` across participants.

**Validation rules.** All hashes are well-formed; `root_seed` and every derived seed are present; unknown
`schema_version` → `SchemaVersionMismatch`. Validated at manifest load.

---

## 12. Residency and redaction constraints (cross-cutting)

`INV-RESIDENCY` governs the entire data model: no raw observation, action, or private-embedding tensor —
i.e. no field of `Transition`, `Episode`, `Window`, no `LatentState` of private data, and no action-head
parameter $\psi$ — is serialized into any outbound message or artifact crossing a trust boundary. The
only participant-derived object that crosses is a post-DP, masked `PseudoGradient.delta` bound to its
$R_c$.

- **Enforcement.** `lensemble.data.residency` guards every serialization/egress path. An attempt to emit
  a residency-bound tensor raises `ResidencyViolation`. This error is **fail-closed and never caught-and-
  ignored** ([conventions §6](conventions.md#6-error-taxonomy); [04-error-model.md](04-error-model.md)).
- **What may be logged/emitted.** Only hashes, L2 norms, tensor shapes, counts, and scalar metrics — never
  raw tensors or private embeddings. The redaction guard lives in `lensemble.observability.redaction`
  and is normative in [05-observability.md](05-observability.md) and
  [RFC-0015](../rfcs/RFC-0015-observability-diagnostics.md); it also fails closed.
- **Public exception.** Probe embeddings $E_{\text{ref}} = f_{\text{ref}}(\mathcal{P})$ are computed on
  the *public* probe $\mathcal{P}$ and are not residency-bound; they support the publicly-recomputable
  alignment and the frame-drift diagnostic. The probe is content-hash-pinned (`INV-PROBE-PIN`); a hash
  mismatch raises `ProbeError`.

---

## 13. Reporting types: `EvalReport`, `FrameDriftReport`, `ContributionRecord`

On-disk JSON reports produced by evaluation, the gauge diagnostic, and the contribution ledger.

### 13.1 `EvalReport`

The latent-MPC evaluation output. Metric definitions are normative in
[RFC-0005 §3–4](../rfcs/RFC-0005-evaluation.md).

```python
class EvalReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: int
    checkpoint_hash: str            # the ModelArtifact content_hash evaluated
    env_id: str                     # stable-worldmodel environment id
    planner: Literal["cem","icem","mppi"]
    success_rate: float             # in [0,1]; world.evaluate on held-out envs/factors
    planning_samples: int           # planner samples per action
    time_per_action_ms: float       # planning wall-cost per action, milliseconds
    effective_dim: float            # embedding-covariance effective dimension (collapse guard)
    probe_accuracy: float | None    # supporting linear/attentive probe accuracy, in [0,1]
    run_manifest_hash: str          # binds the report to its RunManifest
```

**Validation rules.** `0 <= success_rate <= 1`; `effective_dim > 0`; `planner` in the enum;
`probe_accuracy in [0,1]` when present. Out-of-range raises `EvaluationError`. Unknown `schema_version`
→ `SchemaVersionMismatch`.

### 13.2 `FrameDriftReport` — the headline empirical artifact

The per-round, per-participant-pair latent-frame-drift record. This is the central reproducible figure
of the paper; its emission contract is normative in
[RFC-0015](../rfcs/RFC-0015-observability-diagnostics.md) and the measurement in
[RFC-0002 §9](../rfcs/RFC-0002-gauge-and-aggregation.md#9-the-frame-drift-diagnostic-the-headline-measurement) /
[RFC-0005 §2](../rfcs/RFC-0005-evaluation.md#2-headline-diagnostic--latent-frame-drift). It MUST be deterministic given committed weights and the
pinned probe.

```python
class PairDrift(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    participant_a: str
    participant_b: str
    rotation_angle_deg: float        # mean inter-participant rotation angle on the probe P
    procrustes_residual: float       # the Procrustes residual ||Q* A - B||_F (RFC-0002 §5)

class FrameDriftReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: int
    round_index: int
    probe_hash: str                  # the pinned probe these embeddings came from (INV-PROBE-PIN)
    pairs: tuple[PairDrift, ...]     # all C-choose-2 participant pairs (O(C^2) per round)
    drift_from_global: dict[str, float]  # participant_id -> rotation angle vs the global model
```

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `rotation_angle_deg` | `float`, `>= 0` | degrees | mean rotation between two participant frames |
| `procrustes_residual` | `float`, `>= 0` | Frobenius | optimal-Procrustes residual on the probe |
| `drift_from_global` | `dict[str,float]` | degrees | each participant's drift from consensus |

**Invariants.** `INV-PROBE-PIN` — `probe_hash` matches the pinned probe; a mismatch raises `ProbeError`.
Determinism: recomputing the report from the committed weights + the pinned probe reproduces it bit-for-
bit (`recompute_alignment`, [RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md));
a non-reproducing recomputation surfaces as `NonDeterministicAggregation` on the alignment path. A
degenerate Procrustes SVD raises `DegenerateProcrustes` ([04-error-model.md](04-error-model.md)).

### 13.3 `ContributionRecord` — the audit substrate

One append-only ledger entry per round (the `ContributionLedger`,
[RFC-0014](../rfcs/RFC-0014-provenance-commitments.md); origin
[RFC-0004 §5](../rfcs/RFC-0004-data-provenance.md)). It proves data *origin*, not data quality or honest
computation — stated plainly here and in [06-security.md](06-security.md).

```python
class ContributionRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: int
    round_index: int
    participants: tuple[str, ...]        # contributing participant ids this round
    dataset_roots: dict[str, str]        # participant_id -> R_c (hex) bound this round
    global_model_hash: str               # resulting (theta_{t+1},phi_{t+1}) content_hash
    prev_record_hash: str | None         # append-only chain link
```

**Invariants.** `INV-COMMIT-BINDING` — every `dataset_roots[c]` is the root bound by participant $c$'s
accepted `PseudoGradient`; a mismatch is `CommitmentMismatch` and the record is not written.
`global_model_hash` equals the committed checkpoint hash (`INV-CHECKPOINT-HASH`). Append-only: a write
that does not chain `prev_record_hash` to the prior record's hash raises `ProvenanceError`.

---

## 14. Serialization rules

| Data class | Mechanism | Notes |
|---|---|---|
| Structured config (`LensembleConfig` + groups) | frozen dataclasses materialized via OmegaConf/Hydra | composition + `key=value` overrides; validated → `ConfigError` ([RFC-0009](../rfcs/RFC-0009-configuration-reproducibility.md)) |
| On-disk metadata (`DatasetCommitment`, `CheckpointHeader`, `RunManifest`, `EvalReport`, `FrameDriftReport`, `ContributionRecord`) | pydantic v2 → canonical JSON | every model carries integer `schema_version`; `extra="forbid"`, `frozen=True` |
| Tensors / model weights | `safetensors` | mmap-able, no `pickle`, deterministic byte layout — the basis of `INV-CHECKPOINT-HASH` |
| Episodes / trajectories | `stable-worldmodel` data layer | `lance` default (append-friendly indexed reads), `hdf5` portable single-file, `lerobot://<repo_id>` adapter ([RFC-0004 §1](../rfcs/RFC-0004-data-provenance.md#1-per-participant-data-layer)) |

`pickle`/`torch.save` for weights is prohibited (arbitrary-code execution, non-determinism); a pickle
payload on an artifact load is rejected with `ArtifactError`. Canonical commitment/checkpoint hash is
SHA-256 (Phase 1); the STARK-friendly migration is the open question in
[§9](#9-datasetcommitment).

JSON canonicalization (key ordering, number formatting, UTC RFC-3339 timestamps) is fixed so metadata
hashes are stable cross-platform; the exact canonicalization is normative in
[RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md).

`OPEN QUESTION:` the exact canonical-byte ordering for cross-platform checkpoint-hash stability. Owner
@AbdelStark; resolution in [RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md) / v0.1.

---

## 15. Schema versioning and migration

Every on-disk metadata document carries an explicit integer `schema_version`; the latent contract
carries `wmcp_version: str`. Policy ([conventions §10](conventions.md#10-versioning-and-schema-policy)):

- **Forward-compatible readers.** A reader accepts `schema_version <= current` and applies a registered
  migration function per version step. A document with `schema_version` greater than the reader's current
  version, or an unknown version, raises `SchemaVersionMismatch` (an `ArtifactError`) — never a silent
  best-effort parse.
- **Migrations are explicit and ordered.** Each version bump ships a `migrate_vN_to_vN+1` function;
  migrations chain. Pre-1.0, the config schema may change with a manifest `schema_version` bump and a
  deprecation note; at 1.0 the public surface freezes
  ([09-release-and-versioning.md](09-release-and-versioning.md)).
- **WMCP conformance gates on `wmcp_version`.** A `LatentState`/`ActionSpec` whose `wmcp_version` the
  runtime cannot satisfy raises `ContractViolation` and is rejected at the federation-join precondition
  ([RFC-0007](../rfcs/RFC-0007-wmcp-latent-contract.md)).
- **Hash-algorithm field is versioned.** `DatasetCommitment.hash_algorithm` and the checkpoint hash
  algorithm are explicit, so the Phase-2 STARK-friendly-hash migration is a forward-compatible schema
  change rather than a breaking reinterpretation.

`RISK:` reports with `O(C^2)` pair entries (`FrameDriftReport`) can grow large at high participant count;
a sampling policy for pair-drift at large $C$ is an open question owned by @AbdelStark, resolved in
[RFC-0015](../rfcs/RFC-0015-observability-diagnostics.md) (Stage B), and does not change the schema (sampled
pairs are a subset of `pairs`).

---

## 16. Type-to-RFC ownership map

| Type | Owning RFC | Enforced invariants |
|---|---|---|
| `LatentState`, `ActionSpec` | [RFC-0007](../rfcs/RFC-0007-wmcp-latent-contract.md) | `INV-WMCP`, `INV-ACTIONHEAD-LOCAL` |
| `Transition`, `Episode`, `Window` | [RFC-0004](../rfcs/RFC-0004-data-provenance.md) | `INV-RESIDENCY` |
| `PseudoGradient` | [RFC-0003](../rfcs/RFC-0003-federated-protocol.md), [RFC-0012](../rfcs/RFC-0012-differential-privacy.md) | `INV-DP-BOUND`, `INV-COMMIT-BINDING`, `INV-RESIDENCY` |
| `GlobalState` | [RFC-0003](../rfcs/RFC-0003-federated-protocol.md) | `INV-SKETCH-CONSISTENCY`, `INV-PROBE-PIN`, `INV-WARMSTART-T0` |
| `RoundState` | [RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md) | `INV-AGG-DETERMINISM`, `INV-COMMIT-BINDING` |
| `DatasetCommitment` | [RFC-0014](../rfcs/RFC-0014-provenance-commitments.md) | `INV-COMMIT-BINDING` |
| `ModelArtifact`/`Checkpoint` | [RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md) | `INV-CHECKPOINT-HASH`, `INV-ACTIONHEAD-LOCAL`, `INV-WARMSTART-T0` |
| `RunManifest` | [RFC-0009](../rfcs/RFC-0009-configuration-reproducibility.md) | reproducibility, `INV-SKETCH-CONSISTENCY` |
| `EvalReport` | [RFC-0005](../rfcs/RFC-0005-evaluation.md) | — |
| `FrameDriftReport` | [RFC-0015](../rfcs/RFC-0015-observability-diagnostics.md), [RFC-0002](../rfcs/RFC-0002-gauge-and-aggregation.md) | `INV-PROBE-PIN` |
| `ContributionRecord` | [RFC-0014](../rfcs/RFC-0014-provenance-commitments.md) | `INV-COMMIT-BINDING`, `INV-CHECKPOINT-HASH` |

Error taxonomy and the full failure-mode catalog: [04-error-model.md](04-error-model.md). Observability
and redaction: [05-observability.md](05-observability.md). Versioning and release policy:
[09-release-and-versioning.md](09-release-and-versioning.md).
