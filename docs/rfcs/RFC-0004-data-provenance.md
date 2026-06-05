# RFC-0004 — Data, Sovereignty & Provenance

| | |
|---|---|
| **RFC** | 0004 |
| **Title** | Data, Sovereignty & Provenance |
| **Slug** | data-provenance |
| **Status** | Accepted |
| **Track** | Standards |
| **Authors** | @AbdelStark |
| **Created** | 2026-06-02 |
| **Target milestone** | v0.1 (per-participant data layer, public probe, dataset commitments); v0.3 (residency enforcement over a real network boundary) |
| **Area** | data |
| **Requires** | [RFC-0001](RFC-0001-architecture.md), [RFC-0003](RFC-0003-federated-protocol.md) |
| **Defers to** | [RFC-0014](RFC-0014-provenance-commitments.md) (Merkle/commitment construction), [RFC-0007](RFC-0007-wmcp-latent-contract.md) (WMCP latent contract) |

## Summary

This RFC specifies the *data layer* of Lensemble: how a participant's sovereign interaction data is
stored and loaded for local training, how raw data is prevented from ever crossing a trust boundary,
how the shared public probe $\mathcal{P}$ is constituted and governed, and how each contribution is
committed to the data it was computed from. Three facts are load-bearing for the rest of the corpus and
are fixed here: (1) raw observations, actions, and private embeddings never leave a participant
boundary (`INV-RESIDENCY`); only the privatized pseudo-gradient $\Delta_c$ and the dataset commitment
$R_c$ cross ([RFC-0003 §3](RFC-0003-federated-protocol.md#3-the-pseudogradient-contract)). (2) The public probe $\mathcal{P}$ is the
one shared, hash-pinned artifact in an otherwise data-sovereign system; it is the substrate of the
frame anchor ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)) and of publicly-recomputable
alignment ([RFC-0006 §4](RFC-0006-verifiable-contribution.md#4-public-recomputation-phase-1-free-in-scope-now)), so changing it is a versioned
re-anchoring event (`INV-PROBE-PIN`). (3) Every released $\Delta_c$ is bound to exactly one dataset
Merkle root $R_c$ (`INV-COMMIT-BINDING`), which gives Phase-1 contribution accounting and tamper-
evidence and is the binding the Phase-2 proofs attest.

The cryptographic *construction* of $R_c$ (episode hashing, the Merkle scheme, inclusion proofs, the
`ContributionLedger`) is owned by [RFC-0014](RFC-0014-provenance-commitments.md) and only referenced
here. The shared latent/action contract (`LatentState`, `ActionSpec`, the embodiment head interface) is
owned by [RFC-0007](RFC-0007-wmcp-latent-contract.md) and summarized here as the precondition for
joining a federation. The stable data-type schemas this RFC describes are authored in
[03-data-model.md](../spec/03-data-model.md); the residency threat surface is detailed in [06-security.md](../spec/06-security.md).

## Motivation

A foundation-scale world model wants diverse embodied experience — robot fleets, manipulation labs,
driving stacks, egocentric video — but that data is siloed by IP, privacy, and safety and cannot be
pooled ([RFC-0001 §Motivation](RFC-0001-architecture.md#motivation)). Federated training is the access strategy
only if the data layer makes a hard guarantee: a participant's raw trajectories physically cannot be
serialized into anything that crosses its boundary. A federated protocol that merely *intends* not to
move data is not a sovereignty story; the guarantee must be enforced at the egress, fail-closed, and
testable. This RFC owns that guarantee (`INV-RESIDENCY`).

Two further problems are specific to *this* federation. First, the gauge fix
([RFC-0002](RFC-0002-gauge-and-aggregation.md)) requires a frame anchor, and the anchor requires a set
of inputs whose embeddings every participant can compare against a common reference. That set cannot be
any participant's private data (it would leak, and it would not be common). It must be a *public* probe,
fixed and content-hash-pinned, so that the reference frame is identical for everyone and recomputable by
anyone. The probe is therefore a first-class data artifact with its own governance, not an
implementation detail. Second, a federation that aggregates contributions must be able to say *which
data produced which update* — for contribution accounting in Phase 1 and for cryptographic provenance in
Phase 2 — without moving the data. A commitment (a Merkle root over hashed episodes) bound to each
released update is the mechanism, and committing from day one is the cheap Phase-1 discipline that makes
the Phase-2 proofs require no rework ([RFC-0006 §3](RFC-0006-verifiable-contribution.md#3-phase-1-proof-ready-requirements-cheap-to-honor-now)).

## Goals

- Specify the per-participant data layer: episode schema, the three supported formats and their
  trade-offs, the window loader, and the non-exportable residency flag.
- Specify the residency guarantee (`INV-RESIDENCY`): the egress mechanism that makes the training
  process refuse to emit raw observations/actions/private embeddings, fail-closed as `ResidencyViolation`,
  and where it is enforced (`lensemble.data.residency`).
- Specify the public probe $\mathcal{P}$: its requirements (public/licensed, fixed/versioned/hash-pinned,
  representative, landmarks $k \ge d$), its governance, and the `INV-PROBE-PIN` discipline that makes a
  probe change a versioned re-anchoring event.
- Define the provenance-commitment *contract* a participant honors — episode hashing into a Merkle root
  $R_c$, and binding each released $\Delta_c$ to exactly one $R_c$ (`INV-COMMIT-BINDING`) — and defer the
  construction to [RFC-0014](RFC-0014-provenance-commitments.md).
- Define the contribution-accounting substrate (the append-only ledger) at the level this RFC owns and
  defer its schema to [RFC-0014](RFC-0014-provenance-commitments.md).
- State the data-quality metadata each participant declares and the explicit boundary: provenance proves
  origin, not quality or honest computation.
- Summarize the WMCP precondition for joining a federation and defer the contract to
  [RFC-0007](RFC-0007-wmcp-latent-contract.md).

## Non-Goals

- The Merkle-tree construction, leaf/node domain separation, inclusion proofs, hash-function choice, and
  the `ContributionLedger` schema. Owned by [RFC-0014](RFC-0014-provenance-commitments.md).
- The WMCP `LatentState`/`ActionSpec` schemas and the embodiment-head interface. Owned by
  [RFC-0007](RFC-0007-wmcp-latent-contract.md).
- The pseudo-gradient privatization (clip+noise), secure aggregation, and the round lifecycle that
  consume $R_c$ and $\Delta_c$. Owned by [RFC-0003](RFC-0003-federated-protocol.md),
  [RFC-0011](RFC-0011-secure-aggregation.md), [RFC-0012](RFC-0012-differential-privacy.md).
- The frame-anchor *objective* and Procrustes backstop that consume $\mathcal{P}$. Owned by
  [RFC-0002](RFC-0002-gauge-and-aggregation.md).
- The Phase-2 proof system that attests the commitment binding. Owned by
  [RFC-0006](RFC-0006-verifiable-contribution.md).
- Data-quality *enforcement* beyond declared provenance metadata (weighting/gating policy is an Open
  Question, deferred past v0.1).
- The on-disk artifact/checkpoint format for model weights. Owned by
  [RFC-0010](RFC-0010-artifact-checkpoint-format.md).

## Proposed Design

### 1. Per-participant data layer

Each participant `c` holds a **local** dataset of interaction trajectories and trains against it only;
no participant ever reads another's data. The data layer reuses `stable-worldmodel`'s episode store
([conventions §11](../spec/conventions.md#11-external-dependencies)), wrapped by `lensemble.data`.

**Episode schema.** An `Episode` is an ordered sequence of `Transition` tuples $(o_t, a_t, o_{t+1})$ —
observation, action, next observation — plus per-episode metadata (modality, embodiment id, `ActionSpec`
reference, collection conditions; §6). A `Transition` is the atomic record; a `Window` is a fixed-length
contiguous slice of `num_steps` transitions, the unit the loader yields for next-embedding prediction
([RFC-0008](RFC-0008-model-objective-numerics.md)). The stable schemas for `Episode`, `Transition`, and
`Window` are authored in [03-data-model.md §4](../spec/03-data-model.md#4-transition-and-episode--the-data-layer); the shapes a loader emits:

```python
from dataclasses import dataclass
from pathlib import Path
from torch import Tensor

@dataclass(frozen=True)
class Transition:
    """A single (o_t, a_t, o_{t+1}) step. o_* are raw observations; a_t a raw action.
    These tensors are RESIDENT: they never cross a trust boundary (INV-RESIDENCY)."""
    obs: Tensor          # o_t   — raw observation (e.g. video frames), participant-local only
    action: Tensor       # a_t   — raw action in the embodiment's ActionSpec, participant-local only
    next_obs: Tensor     # o_{t+1}

@dataclass(frozen=True)
class Window:
    """A fixed-length contiguous slice of `num_steps` transitions; the training unit."""
    obs: Tensor          # (num_steps, *obs_shape)   — resident
    actions: Tensor      # (num_steps, action_dim)   — resident
    num_steps: int

class EpisodeDataset:
    """Participant-local store over `stable-worldmodel`'s data layer. Iterates Windows.
    Carries the residency flag; never exposes a method that serializes raw tensors across egress."""
    path: Path
    fmt: str             # "lance" | "hdf5" | "lerobot" | "lerobot-h5"
    exportable: bool     # MUST be False for sovereign data; the residency guard reads this
    def windows(self, num_steps: int) -> "Iterator[Window]": ...
    def commit(self) -> "DatasetCommitment": ...   # builds R_c (RFC-0014); raw episodes never emitted
```

**Formats.** Four storage backends are supported, selected per participant:

| Format | Backend ([conventions §11](../spec/conventions.md#11-external-dependencies)) | Use | Trade-off |
|---|---|---|---|
| `lance` (default) | `lance >= 0.10` | The reference store for new datasets | Append-friendly, fast indexed random reads of windows; columnar |
| `hdf5` (portable) | `h5py >= 3.10` | Single-file portability, archival, transfer between a participant's own machines | One self-contained file; slower random access than `lance` |
| `lerobot` (adapter) | `lerobot://<repo_id>` | Train directly on LeRobot-Hub robot datasets without re-ingest | Adapter over an external schema; read-only; conformance-checked on load |
| `lerobot-h5` (adapter) | `lerobot-h5://<path>` or `fmt="lerobot-h5"` | Train directly on local LeRobot-layout HDF5 exports, including HF-mounted datasets | Reads the de-facto `episode_index` / `observation/pixels_*` / `action` layout; read-only; conformance-checked on load |

The `lerobot://` adapter resolves a LeRobot-Hub `repo_id` to an `EpisodeDataset` view; it is read-only
and its records are validated against the `Episode` schema and the WMCP `ActionSpec`
([RFC-0007](RFC-0007-wmcp-latent-contract.md)) on load, raising `ContractViolation` on a mismatched
action space or latent-incompatible modality. New data adapters register through the same interface (the
extension point is documented in [02-public-api.md §5](../spec/02-public-api.md#5-extension-points)).
The `lerobot-h5://` adapter resolves a local LeRobot-layout HDF5 export to the same `EpisodeDataset`
contract, materializing resident pixel/action tensors inside the participant boundary; it is read-only
and intended for real-data claim smokes and HF Jobs mounts without a bespoke loader.

**Window loader.** The loader yields `Window`s of a fixed `num_steps` for next-embedding prediction. The
objective consumes $f_\theta(x_t)$ and the configured target branch for $f_\theta(x_{t+1})$ within the
window ([RFC-0008](RFC-0008-model-objective-numerics.md)); claim-grade LeWorldModel mode keeps that
target branch live, while the default proof-ready JEPA-family path may stop-gradient it. The loader does
not compute embeddings, it only materializes raw windows for the local trainer, which is inside the
trust boundary.

### 2. Residency: the sovereignty guarantee (`INV-RESIDENCY`)

Every local `EpisodeDataset` carries a non-exportable flag (`exportable = False` for sovereign data). The
training process MUST refuse to serialize any raw observation, raw action, or **private embedding**
$f_\theta(x)$ into any outbound message or artifact that crosses a trust boundary. Only the privatized
pseudo-gradient $\Delta_c$ and the dataset commitment $R_c$ leave
([RFC-0003 §8 message table](RFC-0003-federated-protocol.md#8-message-summary);
[RFC-0001 §4 federation map](RFC-0001-architecture.md#4-federation-map)).

`INV-RESIDENCY` is enforced by `lensemble.data.residency`: a guard that sits on the egress path of any
cross-boundary serialization. The guard inspects a payload before it is allowed to cross and **fails
closed** on any resident tensor.

```python
# lensemble/data/residency.py  (enforcement point for INV-RESIDENCY)
def guard_egress(payload: object) -> None:
    """Inspect an outbound, boundary-crossing payload. Raise if it carries resident data.

    Permitted to cross: PseudoGradient.delta (over θ, φ only; never action heads),
        DatasetCommitment (the root R_c + counts + WMCP metadata), coordination scalars/hashes
        (sketch seed s_t, probe hash, global-model hash), and redacted metrics
        (hashes, L2 norms, shapes, counts, scalars — see 05-observability §Redaction).
    Forbidden to cross: any raw observation/action tensor, any private embedding f_θ(x),
        any per-embodiment action-head parameter group h_ψ^(c) (INV-ACTIONHEAD-LOCAL).

    Raises:
        ResidencyViolation: a resident tensor was found on an egress payload (fail-closed,
            never caught-and-ignored; [conventions §6](../spec/conventions.md#6-error-taxonomy)).
    """
```

The guard is the single egress checkpoint; the federation layer routes all boundary-crossing payloads
(`RoundOpen` is inbound; `Update`, `Commitment` are outbound) through it
([RFC-0003 §8](RFC-0003-federated-protocol.md#8-message-summary)). `ResidencyViolation` is **security-critical**: per
[conventions §6](../spec/conventions.md#6-error-taxonomy) it is never caught-and-ignored, and per [04-error-model.md §5.1](../spec/04-error-model.md#51-data--residency-lensembledata-lensembledataresidency) its system response is
fail-closed (the round aborts; the participant does not silently degrade to sending less). The action-
head exclusion path (`INV-ACTIONHEAD-LOCAL`) is enforced at `PseudoGradient` construction and also raises
`ResidencyViolation` if an action-head parameter group reaches the released delta
([RFC-0003 §3](RFC-0003-federated-protocol.md#3-the-pseudogradient-contract)).

`INV-RESIDENCY` lands in scope as enforced code in **v0.3 (Stage C)**, when a real network boundary
exists; in **v0.1/v0.2** the guard is present and tested, but boundaries are simulated in-process
([RFC-0001 §8](RFC-0001-architecture.md#8-process--concurrency-model)), so the guard's egress test runs against the simulated egress.

### 3. The public probe set $\mathcal{P}$

The probe is the substrate for the frame anchor ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)) and
for publicly-recomputable alignment ([RFC-0006 §4](RFC-0006-verifiable-contribution.md#4-public-recomputation-phase-1-free-in-scope-now)). It is the one
shared, agreed artifact in an otherwise data-sovereign system — every participant computes embeddings of
the *same* probe so their frames are comparable against a common reference. Requirements:

- **Public & licensed for redistribution.** $\mathcal{P}$ contains no participant's private data. It is
  redistributable (data license CC-/CDLA-compatible, [09-release-and-versioning.md](../spec/09-release-and-versioning.md)) so any
  verifier can hold it and recompute alignment ([RFC-0006 §4](RFC-0006-verifiable-contribution.md#4-public-recomputation-phase-1-free-in-scope-now)).
- **Fixed & versioned, content-hash-pinned.** The probe's content hash is computed once and pinned. The
  `RoundOpen` broadcast carries this hash and the landmark hashes
  ([RFC-0003 §8](RFC-0003-federated-protocol.md#8-message-summary)); each participant checks the broadcast hash equals the
  pinned hash before training (`INV-PROBE-PIN`). A probe change is a versioned event: it redefines the
  reference frame and forces re-anchoring (§3.1).
- **Representative.** $\mathcal{P}$ spans the modalities/embodiments in the federation enough to anchor a
  meaningful frame. Under-coverage weakens the anchor (the pinned directions do not constrain the part of
  latent space the participants actually move in); over-size raises per-round alignment cost (the
  Procrustes backstop and the public recomputation scale with $|\mathcal{P}|$). The acceptance criterion
  for "representative enough" is an Open Question (§Open Questions).
- **Landmarks ($k \ge d$).** A designated subset of $k \ge d$ landmark points $p_i$ carry reference
  targets $t_i = f_{\text{ref}}(p_i)$ from the round-0 reference encoder $f_{\text{ref}}$
  ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix), Variant A). The condition $k \ge d$ is necessary:
  $k$ generic landmarks in general position pin all $d$ degrees of the $O(d)$ rotational gauge (with
  $k < d$ the anchor under-determines the frame and leaves a residual rotation free). Landmark targets
  derive **only** from $f_{\text{ref}}$ (the round-0 encoder), never from a later round's encoder
  (`INV-PROBE-PIN`).
- **Governance.** The probe is curated openly: the curator set and the update procedure are declared
  (maps to Project Tapestry's Data-Governance work group, [conventions §11](../spec/conventions.md#11-external-dependencies) / README). The probe is the single
  point at which the federation must agree on a shared artifact; its governance is therefore a named
  responsibility, not an implementation default.

```python
# lensemble/data/probe.py  (the probe contract; targets and pinning)
from dataclasses import dataclass
from torch import Tensor

@dataclass(frozen=True)
class PublicProbe:
    """The fixed, hash-pinned, public probe set P and its landmark targets.
    Public artifact: contains no resident data and may cross boundaries freely."""
    points: Tensor             # (P, *obs_shape) — the probe inputs p_i (public)
    landmark_idx: Tensor       # (k,) indices into points marking the k >= d landmarks
    landmark_targets: Tensor   # (k, N, d) — t_i = f_ref(p_i), derived ONLY from f_ref (INV-PROBE-PIN)
    content_hash: bytes        # SHA-256 over canonical bytes of points+landmark_idx ([conventions §11](../spec/conventions.md#11-external-dependencies))
    probe_version: int         # bumped on any content change; a re-anchoring event (§3.1)

def verify_probe_pin(probe: PublicProbe, broadcast_hash: bytes) -> None:
    """Check the RoundOpen-broadcast probe hash equals the pinned content hash (INV-PROBE-PIN).
    Raises:
        ProbeError: hash mismatch (re-anchoring required) or landmark under-coverage (k < d).
    """
```

CLI surface ([conventions §5](../spec/conventions.md#5-public-api-surface)): `lensemble probe build` (compute landmark targets from a pinned $f_{\text{ref}}$
and write a `PublicProbe`), `lensemble probe pin` (compute and freeze the content hash), `lensemble probe
verify` (check a held probe against a pinned hash). Each emits a `RunManifest` recording the probe
content-hash ([RFC-0009](RFC-0009-configuration-reproducibility.md)).

#### 3.1 Probe versioning (re-anchoring)

The probe redefines the reference frame, so changing its content is not a free edit. The procedure:
bump `probe_version`, recompute `content_hash`, and recompute landmark targets $t_i$ against the
*current* $f_{\text{ref}}$ — which means a probe change is a **re-anchoring event** and is coupled to the
warm-start choice ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)). A federation in flight does not
silently swap probes mid-run: a `RoundOpen` carrying a probe hash that differs from a participant's
pinned hash raises `ProbeError` and the round is rejected (`INV-PROBE-PIN`,
[RFC-0003 §9](RFC-0003-federated-protocol.md#9-failure-modes)). Probe versions are recorded in every `RunManifest`
([RFC-0009](RFC-0009-configuration-reproducibility.md)) so a run is reproducible against the exact probe
it used.

### 4. Provenance commitments (the bridge to Phase 2)

To make contributions attributable now and provable later, each participant commits its dataset and binds
its contribution to that commitment. This RFC owns the *contract*; the construction is owned by
[RFC-0014](RFC-0014-provenance-commitments.md).

- **Episode hashing.** Each participant content-hashes its episodes via a canonical byte serialization
  into domain-separated SHA-256 leaves (canonical Phase-1 hash is SHA-256, [conventions §11](../spec/conventions.md#11-external-dependencies)). The canonicalization
  and domain separation are specified in [RFC-0014 §Episode Hashing](RFC-0014-provenance-commitments.md).
- **Dataset Merkle root $R_c$.** Before contributing for a round, the participant commits a Merkle root
  $R_c$ over its episode-leaf hashes. The `commit_dataset(dataset) -> DatasetCommitment` API ([conventions §5](../spec/conventions.md#5-public-api-surface))
  produces a `DatasetCommitment` carrying $R_c$, the episode count, and WMCP metadata; raw episodes are
  never emitted (`INV-RESIDENCY`). The tree shape, node ordering, and inclusion proofs are specified in
  [RFC-0014 §Merkle Tree](RFC-0014-provenance-commitments.md).
- **Binding (`INV-COMMIT-BINDING`).** Every released $\Delta_c$ is associated with exactly one $R_c$ —
  the root under which it was computed. The binding travels as `PseudoGradient.dataset_root`
  ([RFC-0003 §3](RFC-0003-federated-protocol.md#3-the-pseudogradient-contract)) and the standalone `Commitment` message
  ([RFC-0003 §8](RFC-0003-federated-protocol.md#8-message-summary)). The coordinator validates the binding at ingress; a
  $\Delta_c$ with a missing or wrong $R_c$ is rejected with `CommitmentMismatch` (never swallowed,
  [conventions §6](../spec/conventions.md#6-error-taxonomy)). `INV-COMMIT-BINDING` is enforced where the commitment is checked, in
  `lensemble.provenance`/`lensemble.federation.coordinator`.

```python
# lensemble/provenance — the commitment carried across the boundary (full schema in RFC-0014 / 03-data-model)
from dataclasses import dataclass

@dataclass(frozen=True)
class DatasetCommitment:
    """The cross-boundary commitment to a participant's dataset. Public: carries no resident data."""
    root: bytes              # R_c — Merkle root over domain-separated episode-leaf SHA-256 hashes (RFC-0014)
    episode_count: int       # number of committed episodes
    wmcp_version: str        # the latent-contract version the episodes conform to (RFC-0007, INV-WMCP)
    schema_version: int      # on-disk schema version ([conventions §10](../spec/conventions.md#10-versioning-and-schema-policy)); SchemaVersionMismatch on unknown/too-new
```

In Phase 1 this enables **contribution accounting** (which committed dataset fed which round, §5) and
**tamper-evidence**: a participant cannot, after the fact, claim a different dataset produced an update
without changing $R_c$, which is bound and logged. In Phase 2
([RFC-0006 §2](RFC-0006-verifiable-contribution.md#2-the-provable-surface-and-what-stays-cheap)) the binding upgrades to a cryptographic claim:
*"this update was computed from data under committed root $R_c$."* This proves **provenance/origin**, not
data quality and not honest computation — stated plainly here and in [06-security.md §6](../spec/06-security.md#6-provenance-tamper-evidence-vs-cryptographic-proof). Committing from day one is the cheap Phase-1 discipline that makes the Phase-2 proofs need
no rework ([RFC-0006 §3](RFC-0006-verifiable-contribution.md#3-phase-1-proof-ready-requirements-cheap-to-honor-now)).

### 5. Contribution accounting

The federation maintains an append-only `ContributionLedger`: per round, the set of contributing
participants, their committed roots $R_c$, and the resulting global-model content hash. This supports
credit/governance (which data improved the model, in which round) and is the audit substrate the
verifiable layer formalizes ([RFC-0006 §3–§5](RFC-0006-verifiable-contribution.md#3-phase-1-proof-ready-requirements-cheap-to-honor-now)). The ledger record is
the `ContributionRecord` type ([conventions §8](../spec/conventions.md#8-core-data-types); full schema in [03-data-model.md §13.3](../spec/03-data-model.md#133-contributionrecord--the-audit-substrate) and
[RFC-0014 §ContributionLedger](RFC-0014-provenance-commitments.md)):

```python
@dataclass(frozen=True)
class ContributionRecord:
    """One round's audit entry in the append-only ContributionLedger (RFC-0014)."""
    round_index: int               # the round t
    participant_ids: tuple[str, ...]   # contributing participants (ids for accounting, not raw data)
    dataset_roots: tuple[bytes, ...]   # their committed R_c, aligned to participant_ids
    global_hash: bytes             # content hash of (θ_{t+1}, φ_{t+1}) committed this round (INV-CHECKPOINT-HASH)
```

The ledger is append-only (an entry is never mutated or deleted); the append-only invariant and the
on-disk format are specified in [RFC-0014 §ContributionLedger](RFC-0014-provenance-commitments.md). It
lands as enforced state in **v0.3 (Stage C)** alongside the real network boundary
([RFC-0001 §Migration](RFC-0001-architecture.md#migration--rollout)); commitments themselves land in **v0.1**.

### 6. Data-quality metadata and the WMCP precondition

**Data-quality metadata.** Each participant declares minimal, non-resident metadata about its dataset:
modality, embodiment id, the `ActionSpec` per WMCP ([RFC-0007](RFC-0007-wmcp-latent-contract.md)),
episode count, and collection conditions. This metadata is public (it crosses as part of
`DatasetCommitment` and the contribution record); it carries no raw data. The federation MAY weight or
gate contributions on declared quality, but **quality enforcement beyond declared provenance is out of
scope for v0.1** (the weighting/gating policy is an Open Question, §Open Questions). The integrity
boundary stated in §4 holds here too: declared metadata is *declared*, not verified — provenance proves
origin, not that the data is good.

**The WMCP precondition.** Heterogeneous embodiments can only federate into one model if they agree on
the latent interface. WMCP ([RFC-0007](RFC-0007-wmcp-latent-contract.md), WM-RFC-0001) defines that
interface: the shape/dtype/semantics of the `LatentState` every encoder emits and every predictor
consumes, and the `ActionSpec` + action-conditioning interface each per-embodiment head must satisfy.
This is the type-safety layer that makes cross-silo federation well-posed — the explicit analogue of the
fixed token vocabulary LLM federation gets for free ([RFC-0001 §1](RFC-0001-architecture.md#1-model)).
**Conformance to the pinned `wmcp_version` is a precondition for joining a Lensemble federation**
(`INV-WMCP`): a nonconforming `LatentState` raises `ContractViolation`, and an unvalidated `ActionSpec`
blocks action-head construction. This RFC only states the precondition; the contract is owned by
[RFC-0007](RFC-0007-wmcp-latent-contract.md).

### 7. Failure modes

Failure modes this RFC's design detects and handles (errors from [conventions §6](../spec/conventions.md#6-error-taxonomy); system response from
[04-error-model.md](../spec/04-error-model.md)):

| Trigger | Detection | Error | System response |
|---|---|---|---|
| Raw observation/action/private embedding about to cross a boundary | `data.residency.guard_egress` at egress | `ResidencyViolation` | Fail-closed; round aborts; never caught-and-ignored (`INV-RESIDENCY`) |
| Action-head parameter group reaches a released delta | param-group check at `PseudoGradient` construction ([RFC-0003 §3](RFC-0003-federated-protocol.md#3-the-pseudogradient-contract)) | `ResidencyViolation` | Reject; action heads stay local (`INV-ACTIONHEAD-LOCAL`) |
| `RoundOpen` probe hash ≠ pinned probe hash | `probe.verify_probe_pin` at participant ingress | `ProbeError` | Reject `RoundOpen`; re-anchoring required (`INV-PROBE-PIN`) |
| Probe landmark count $k < d$ (under-coverage) | probe build/verify check | `ProbeError` | Reject probe; anchor under-determined |
| `lerobot://`, `lerobot-h5://`, or local episode violates the latent/action contract | adapter/loader conformance check | `ContractViolation` | Reject dataset/record (`INV-WMCP`, [RFC-0007](RFC-0007-wmcp-latent-contract.md)) |
| Released $\Delta_c$ not bound to a valid $R_c$ | commitment-binding check at coordinator ingress | `CommitmentMismatch` | Reject the update; never swallowed (`INV-COMMIT-BINDING`) |
| A committed $R_c$ fails Merkle verification | Merkle check ([RFC-0014](RFC-0014-provenance-commitments.md)) | `MerkleVerificationError` | Reject the commitment |
| Unknown/too-new `DatasetCommitment.schema_version` | pydantic load validation ([conventions §10](../spec/conventions.md#10-versioning-and-schema-policy)) | `SchemaVersionMismatch` | Refuse load; explicit migration required |

`ResidencyViolation`, `CommitmentMismatch`, and `MerkleVerificationError` (where it carries a commitment-
binding decision) are security-critical and are never caught-and-ignored ([conventions §6](../spec/conventions.md#6-error-taxonomy)). All errors carry
`.code` and `.remediation`; error logging routes through [05-observability.md §1](../spec/05-observability.md#1-structured-logging).

## Alternatives Considered

- **Episode storage formats — `lance` (default) vs `hdf5` (portable) vs LeRobot adapters.**
  *What they are:* a columnar append-friendly store, a single-file portable store, a read-only adapter
  over LeRobot-Hub datasets, and a read-only local LeRobot-layout HDF5 adapter. *Why considered:*
  different participants have different needs — high-throughput local training (`lance`),
  archival/transfer on one participant's own machines (`hdf5`), training directly on existing public robot
  datasets without re-ingest (`lerobot://`), and training from HF-mounted local exports
  (`lerobot-h5://`). *Why all four are kept:* they serve disjoint needs; `lance` is the default for new
  datasets because of fast indexed random window reads, `hdf5` is the portable fallback, `lerobot://`
  avoids a costly re-ingest of an entire hub dataset, and `lerobot-h5` is the file-mount bridge for
  real-data claim smokes. None is rejected; the choice is per-participant config and the formats
  round-trip-test identically ([RFC-0009](RFC-0009-configuration-reproducibility.md)).
- **Probe sizing — small vs large $\mathcal{P}$.** *What it is:* the number of probe points and
  landmarks. *Why considered:* the probe trades anchor strength against per-round cost. *Why neither
  extreme is chosen:* under-coverage weakens the anchor (the pinned directions fail to constrain the
  latent subspace participants actually move in, so the frame can still drift in unconstrained
  directions); over-size raises the cost of the Procrustes backstop
  ([RFC-0002 §5](RFC-0002-gauge-and-aggregation.md#5-layer-3--procrustes-re-alignment-at-aggregation-backstop)) and the public recomputation
  ([RFC-0006 §4](RFC-0006-verifiable-contribution.md#4-public-recomputation-phase-1-free-in-scope-now)), both of which scale with $|\mathcal{P}|$. The
  size is set to the smallest probe that demonstrably holds the frame in the Stage-B sweep; the
  acceptance criterion is an Open Question.
- **Per-silo probe vs one shared probe.** *What it is:* each participant uses its own probe, vs a single
  federation-wide probe. *Why considered:* a per-silo probe could be tailored to local modality coverage.
  *Why rejected:* the anchor's entire purpose is to give every participant a *common* reference frame so
  their updates are comparable and averaging is valid ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix));
  per-silo probes re-introduce a distinct gauge per silo (the very problem the anchor solves) and break
  publicly-recomputable alignment (a verifier would need every silo's private probe). A single, public,
  hash-pinned probe is structural, not incidental.
- **Flat hash of the dataset vs a Merkle root for the commitment.** *What it is:* a single hash over all
  episodes vs a Merkle tree. *Why considered:* a flat hash is simpler and cheaper to compute. *Why a
  Merkle root is chosen (deferred to [RFC-0014](RFC-0014-provenance-commitments.md)):* the Merkle tree
  enables per-episode inclusion proofs and incremental re-commitment on dataset append, both of which the
  Phase-2 provenance binding ([RFC-0006 §2](RFC-0006-verifiable-contribution.md#2-the-provable-surface-and-what-stays-cheap)) needs; a flat hash
  cannot prove a single episode's membership without revealing the whole set.
- **Trusting the protocol to not move data vs an enforced egress guard.** *What it is:* relying on
  careful coding to never serialize raw data, vs a single fail-closed guard on the egress path. *Why
  considered:* a guard adds a checkpoint to every cross-boundary send. *Why the guard is chosen:*
  sovereignty is the project's premise; an intention is not a guarantee. A single enforced egress
  checkpoint that fails closed (`INV-RESIDENCY`) is testable and auditable, and the cost is negligible
  relative to a round.

## Drawbacks

- **Probe-curation governance burden.** The public probe is the one artifact the whole federation must
  agree on, and it must be curated openly with a declared curator set and update procedure. This is real
  coordination cost that a fully data-sovereign system would otherwise avoid; it is the irreducible price
  of a shared reference frame.
- **Provenance proves origin, not quality or honesty.** The commitment binds an update to a *committed
  dataset*; it does not attest that the data is good or that the gradient was computed honestly. A
  participant can commit a low-quality or adversarially-constructed dataset and the binding will faithfully
  record it. Quality-gating beyond declared metadata is out of v0.1 scope (Open Questions), and honest-
  computation guarantees are Phase 2 / TEE territory ([RFC-0006 §5](RFC-0006-verifiable-contribution.md#5-the-composed-trust-statement-per-roadmap-stage)).
- **The probe redefines the reference frame, so changing it is expensive.** A probe change is a versioned
  re-anchoring event (§3.1) coupled to the warm-start; it cannot happen mid-run, and it invalidates
  comparability with runs that used an earlier probe version. The probe is effectively a long-lived
  commitment of the federation.
- **Hashing cost over large datasets.** Episode hashing and Merkle-root construction scale with dataset
  size; for a large participant this is a non-trivial one-time (plus incremental-on-append) cost. The
  incremental re-commitment strategy is owned by [RFC-0014](RFC-0014-provenance-commitments.md); the
  hash-function migration to a STARK-friendly hash (an Open Question shared with RFC-0006/0014) would also
  affect this cost.

## Migration / Rollout

Mapped to the staged plan ([RFC-0001 §Migration](RFC-0001-architecture.md#migration--rollout), [conventions §12](../spec/conventions.md#12-milestones-and-stages) milestones via
[00-overview.md](../spec/00-overview.md)):

- **v0.1 / Stage A — data layer, probe, commitments (no real boundary).** The `EpisodeDataset` over
  `lance`/`hdf5`/`lerobot`, the `PublicProbe` build/pin/verify flow, `commit_dataset` producing
  `DatasetCommitment` with $R_c$, and the residency guard *present and tested against simulated egress*.
  Single-site training pools data locally ([RFC-0001 §7](RFC-0001-architecture.md#7-data-flow-lifecycles)); there is no
  cross-boundary traffic yet, so residency is exercised in test, not in deployment.
- **v0.2 / Stage B — commitments and probe inside the simulated federation.** The probe anchors the
  simulated round ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)); commitments bind simulated
  contributions; `INV-PROBE-PIN`, `INV-COMMIT-BINDING`, and `INV-SKETCH-CONSISTENCY` are exercised
  end-to-end in-process ([RFC-0003 §Migration](RFC-0003-federated-protocol.md#migration--rollout)).
- **v0.3 / Stage C — residency enforced over a real boundary; the ledger goes live.** The egress guard
  enforces `INV-RESIDENCY` on the wire; the `ContributionLedger` records real rounds. No data migration
  is required between stages: the schemas (`Episode`, `Window`, `DatasetCommitment`, `ContributionRecord`,
  `PublicProbe`) are stable from v0.1; Stage C swaps the transport, not the data contracts.

**Probe versioning** is the one in-band migration concern: a probe content change bumps `probe_version`
and forces re-anchoring (§3.1); it is a deliberate, recorded, federation-wide event and never an
incidental edit.

## Testing Strategy

CPU-runnable tests on tiny synthetic fixtures (no large downloads; cf. [07-testing-strategy.md §8](../spec/07-testing-strategy.md#8-ci-gates)):

- **Residency guard refuses to emit raw data (security-critical, `INV-RESIDENCY`).** Construct an
  outbound payload that embeds a raw observation, a raw action, and a private embedding $f_\theta(x)$ in
  turn; assert `guard_egress` raises `ResidencyViolation` for each and that the violation is never
  caught-and-ignored. Assert a valid payload (a `PseudoGradient.delta` over $\theta,\phi$ only, a
  `DatasetCommitment`, coordination scalars/hashes) passes. Cross-referenced from
  [RFC-0001 §Testing Strategy](RFC-0001-architecture.md#testing-strategy) and
  [06-security.md §3](../spec/06-security.md#3-residency-enforcement-inv-residency).
- **Action-head exclusion (`INV-ACTIONHEAD-LOCAL`).** Assert a released delta that includes an
  action-head parameter group raises `ResidencyViolation` at construction.
- **Merkle correctness and binding (contract; construction tested in [RFC-0014](RFC-0014-provenance-commitments.md)).**
  Assert `commit_dataset` over a known toy dataset yields a deterministic $R_c$; assert a $\Delta_c$
  released with a missing or wrong `dataset_root` is rejected with `CommitmentMismatch`
  (`INV-COMMIT-BINDING`); assert a tampered episode changes $R_c$.
- **Format round-trip (lance / hdf5 / LeRobot adapters).** Write a toy dataset in each format, read it back as
  `Window`s, and assert byte/tensor equality of the materialized windows across formats; assert the
  `lerobot://` and `lerobot-h5://` adapters validate `ActionSpec` and raise `ContractViolation` on a
  mismatched action space.
- **Probe-pin verification (`INV-PROBE-PIN`).** Assert `verify_probe_pin` accepts the pinned hash and
  raises `ProbeError` on a mismatched broadcast hash; assert a probe with $k < d$ landmarks raises
  `ProbeError`; assert landmark targets are derived only from $f_{\text{ref}}$ (changing a later-round
  encoder does not change `landmark_targets`).
- **Schema round-trip and version gating.** Round-trip `DatasetCommitment` and `ContributionRecord`
  through pydantic JSON; assert an unknown/too-new `schema_version` raises `SchemaVersionMismatch`
  ([conventions §10](../spec/conventions.md#10-versioning-and-schema-policy)).
- **Ledger append-only invariant.** Assert the `ContributionLedger` rejects mutation/deletion of an
  existing record (full test in [RFC-0014 §Testing](RFC-0014-provenance-commitments.md)).

These feed the ablation-ladder integration tests ([RFC-0005 §6](RFC-0005-evaluation.md#6-ablation-ladder-the-core-experiment),
[07-testing-strategy.md §3](../spec/07-testing-strategy.md#3-the-ablation-ladder-as-integration-tests)) only as fixtures; the residency and probe-pin tests are unit/property
tests that run on every CI invocation.

## Open Questions

OPEN QUESTION: The **probe-coverage acceptance criterion** — how representative must $\mathcal{P}$ be
(how many points, spanning which modalities/embodiments) before it demonstrably holds the frame, and what
metric certifies "representative enough"? Owner @AbdelStark; resolution path: the Stage-B (v0.2) probe-
sizing sweep, measured against the frame-drift diagnostic ([RFC-0005 §2](RFC-0005-evaluation.md#2-headline-diagnostic--latent-frame-drift),
[RFC-0015 §3](RFC-0015-observability-diagnostics.md#3-the-frame-drift-diagnostic-emission-contract-the-headline-artifact)) — the smallest probe whose
anchored runs hold drift flat is the acceptance threshold.

OPEN QUESTION: The **quality-gating policy beyond provenance** — whether and how the federation weights
or gates contributions on declared data-quality metadata (§6), given that provenance proves origin, not
quality. Owner @AbdelStark; resolution path: deferred past v0.1; a follow-up policy (candidate: a
weighting term in the outer step keyed to declared/measured quality) to be specified after the Stage-B
non-IID sweep ([RFC-0005 §7](RFC-0005-evaluation.md#7-non-iid-severity--scale-sweeps)) characterizes how quality heterogeneity affects the
recovered centralized−local gap.

OPEN QUESTION: Migrating the Phase-1 commitment hash from **SHA-256 to a STARK-friendly hash** (e.g.
Poseidon2) to keep the Phase-2 proof circuit cheap. Owner @AbdelStark; resolution path: shared with
[RFC-0014 §Open Questions](RFC-0014-provenance-commitments.md#open-questions) and
[RFC-0006 §Open Questions](RFC-0006-verifiable-contribution.md#open-questions); decided in Stage D (Phase 2). Episode
hashing and the probe content-hash use SHA-256 in Phase 1 ([conventions §11](../spec/conventions.md#11-external-dependencies)), and the hash function is a versioned
choice ([RFC-0014](RFC-0014-provenance-commitments.md)) so the migration does not invalidate the schema.

RISK: **Residency enforcement completeness.** The egress guard (§2) is only as strong as the set of
egress paths routed through it. A serialization path that bypasses `guard_egress` is a silent sovereignty
breach. Resolution plan: the federation layer is structured so all four boundary-crossing messages
([RFC-0003 §8](RFC-0003-federated-protocol.md#8-message-summary)) flow through the single guarded egress, and the
boundary-crossing redaction test ([RFC-0003 §Testing](RFC-0003-federated-protocol.md#testing-strategy),
[05-observability.md §5](../spec/05-observability.md#5-redaction-inv-residency)) asserts no raw tensor reaches any sink; a Stage-C security
review ([06-security.md](../spec/06-security.md)) audits the egress surface before the real boundary goes live.

## References

- [RFC-0001 — Architecture & System Overview](RFC-0001-architecture.md): the federation map (§4), trust
  boundaries (§6), the dataset-commit lifecycle (§7), and the staged plan.
- [RFC-0002 — The Latent Gauge & Frame-Anchored Aggregation](RFC-0002-gauge-and-aggregation.md): why the
  probe exists (§4 frame anchoring), and the Procrustes backstop (§5) that consumes $\mathcal{P}$.
- [RFC-0003 — Federated Training Protocol](RFC-0003-federated-protocol.md): the round lifecycle (§3) and
  message table (§8) that carry $\Delta_c$ and $R_c$; the residency egress on the wire.
- [RFC-0006 — Verifiable Contribution](RFC-0006-verifiable-contribution.md): the provable surface (§2) the
  commitment binding feeds, and the proof-ready requirements (§3) committing-from-day-one satisfies.
- [RFC-0007 — WMCP Latent Contract & Embodiment Adapters](RFC-0007-wmcp-latent-contract.md): the
  `LatentState`/`ActionSpec` contract that conformance gates on (`INV-WMCP`).
- [RFC-0009 — Configuration, Run Manifest & Reproducibility](RFC-0009-configuration-reproducibility.md):
  the `RunManifest` recording probe content-hash and seeds.
- [RFC-0010 — Checkpoint & Artifact Format](RFC-0010-artifact-checkpoint-format.md): the committed
  $(\theta_t,\phi_t)$ hash that appears in `ContributionRecord.global_hash`.
- [RFC-0014 — Provenance Commitments & Merkle Scheme](RFC-0014-provenance-commitments.md): the episode
  hashing, Merkle construction, inclusion proofs, and `ContributionLedger` this RFC defers to.
- [RFC-0015 — Observability, Diagnostics & Telemetry](RFC-0015-observability-diagnostics.md): the
  redaction guard and the frame-drift diagnostic against the probe.
- Spec: [00-overview.md](../spec/00-overview.md), [01-architecture.md §3](../spec/01-architecture.md#3-federation-map),
  [02-public-api.md §5](../spec/02-public-api.md#5-extension-points), [03-data-model.md](../spec/03-data-model.md),
  [04-error-model.md](../spec/04-error-model.md), [05-observability.md §5](../spec/05-observability.md#5-redaction-inv-residency), [06-security.md](../spec/06-security.md),
  [07-testing-strategy.md](../spec/07-testing-strategy.md), [09-release-and-versioning.md](../spec/09-release-and-versioning.md).
- External: `stable-worldmodel` (galilai-group) — the `lance`/`hdf5`/`lerobot` data layer ([conventions §11](../spec/conventions.md#11-external-dependencies));
  Project Tapestry (AI Alliance) — the data-governance framing for the public probe; WMCP (WM-RFC-0001) —
  the latent/action contract.
