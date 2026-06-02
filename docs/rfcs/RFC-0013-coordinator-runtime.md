# RFC-0013 — Coordinator & Participant Runtime

| | |
|---|---|
| **RFC** | 0013 |
| **Title** | Coordinator & Participant Runtime |
| **Slug** | coordinator-runtime |
| **Status** | Accepted |
| **Track** | Standards |
| **Authors** | @AbdelStark |
| **Created** | 2026-06-02 |
| **Target milestone** | v0.2 (single-process simulation); networked two-node in v0.3 |
| **Area** | `area:federation` |
| **Requires** | [RFC-0001](RFC-0001-architecture.md), [RFC-0003](RFC-0003-federated-protocol.md) |
| **Defers to** | [RFC-0011](RFC-0011-secure-aggregation.md) (secure aggregation), [RFC-0012](RFC-0012-differential-privacy.md) (DP accounting), [RFC-0002](RFC-0002-gauge-and-aggregation.md) (gauge correction), [RFC-0014](RFC-0014-provenance-commitments.md) (commitments) |

## Summary

This RFC specifies the *runtime* that executes the federated protocol of
[RFC-0003](RFC-0003-federated-protocol.md): the `Coordinator` and `Participant` classes, the
`RoundState` state machine that drives one outer round, the fault-tolerance and elasticity model that
lets a round proceed with whatever participants are present, the control-plane message transport, and
the concurrency model. [RFC-0003](RFC-0003-federated-protocol.md) owns the protocol *semantics* (what
crosses, the eight-step lifecycle, DP and secure-aggregation requirements, the message table); this RFC
owns the *operational machine* that realizes those semantics — the explicit states, the transitions,
the triggers, the timeouts, and the failure handling at each transition.

The runtime is the integration layer (`lensemble.federation`, layer L7 of
[RFC-0001 §3](RFC-0001-architecture.md#3-dependency-layering-no-cycles)). It composes the gauge ([RFC-0002](RFC-0002-gauge-and-aggregation.md)),
secure aggregation ([RFC-0011](RFC-0011-secure-aggregation.md)), DP
([RFC-0012](RFC-0012-differential-privacy.md)), provenance
([RFC-0014](RFC-0014-provenance-commitments.md)), and artifacts
([RFC-0010](RFC-0010-artifact-checkpoint-format.md)) into a single round loop. The public surface it
stabilizes — `Coordinator`, `Participant`, `RoundState` — is fixed by [conventions §5](../spec/conventions.md#5-public-api-surface) and consumed by the
federation public API ([02 §1.3](../spec/02-public-api.md#13-coordinator-and-participant)). The same machine runs in-process for
the Stage-B simulation (v0.2) and over a network boundary for the Stage-C sovereignty demonstration
(v0.3) by swapping only the transport, not the state machine.

## Motivation

[RFC-0003](RFC-0003-federated-protocol.md) defines what one round *means*; it does not define when a
round may advance, what happens when a participant times out mid-round, how a rejoining node recovers,
or how the determinism self-check ([RFC-0003 §7](RFC-0003-federated-protocol.md#7-determinism-concurrency-error-propagation),
`INV-AGG-DETERMINISM`) is wired into the abort path. Those are runtime concerns, and leaving them
implicit would make two implementations of the protocol mutually incompatible and would make the
fault-tolerance behavior untestable.

Three forces shape the runtime. First, **elasticity**: sovereign nodes have heterogeneous compute and
unreliable links, so the round must complete with a quorum rather than wait for every participant
(INTELLECT-1/PRIME elastic fault tolerance, [RFC-0003 §6](RFC-0003-federated-protocol.md#6-heterogeneity--fault-tolerance)). Second,
**determinism on the aggregation path**: the outer step is the proof-ready surface
([RFC-0006 §2](RFC-0006-verifiable-contribution.md#2-the-provable-surface-and-what-stays-cheap)), so the runtime must make the set of contributing
deltas, their order, and the round seed an explicit, recorded input to a pure function, not an artifact
of arrival timing. Third, **transport portability**: the contribution of Lensemble is the gauge and the
federation discipline, not the networking; the runtime must let the Stage-B in-process simulation and
the Stage-C networked deployment share one state machine so the experiments and the deployment validate
the same code path.

## Goals

- Specify the `RoundState` state machine —
  `OPEN → COLLECTING → AGGREGATING → ALIGNING → COMMITTING → CLOSED`, plus the `ABORTED` path — with
  each transition's trigger, precondition, and failure handling.
- Specify the `Coordinator` and `Participant` runtime classes with the signatures fixed by [conventions §5](../spec/conventions.md#5-public-api-surface)
  (`Coordinator.run(num_rounds: int) -> None`,
  `Participant.local_round(global_state: GlobalState, round_seed: int) -> PseudoGradient`).
- Specify fault tolerance and elasticity: quorum, per-participant timeouts, late/dropped reconciliation,
  rejoiner recovery from the latest committed checkpoint, and the `FaultToleranceExceeded` threshold.
- Reproduce the boundary-crossing control-plane message table from
  [RFC-0003 §8](RFC-0003-federated-protocol.md#8-message-summary) and bind each message to a state transition and a
  transport abstraction.
- Specify the concurrency model: the single coordinator round loop, concurrent participant processes,
  backpressure, and timeouts.
- Specify how the runtime wires the determinism self-check (`INV-AGG-DETERMINISM`) into the
  `AGGREGATING → ABORTED` recompute path, and how it enforces `INV-WARMSTART-T0` at join.
- State every runtime failure mode, the error raised from the [conventions §6](../spec/conventions.md#6-error-taxonomy) taxonomy, and the system's
  response (retry / abort / reject / fail-closed).

## Non-Goals

- The protocol semantics, the DP clip-then-noise ordering, the int8 quantization scheme, and the
  reference parameters. Owned by [RFC-0003](RFC-0003-federated-protocol.md).
- The secure-aggregation cryptographic protocol (pairwise masking, threshold secret-sharing, TEE
  backend). Owned by [RFC-0011](RFC-0011-secure-aggregation.md). The runtime calls it through an
  abstract aggregator interface.
- The `(ε,δ)` accountant and mechanism. Owned by [RFC-0012](RFC-0012-differential-privacy.md). The
  runtime queries the accountant at the privatize step and on budget exhaustion drives the round to
  `ABORTED`.
- The gauge correction (anchoring loss, Procrustes closed form, drift threshold). Owned by
  [RFC-0002](RFC-0002-gauge-and-aggregation.md). The runtime invokes `procrustes_align` /
  `frame_drift` in the `ALIGNING` state.
- The checkpoint/commitment byte formats. Owned by
  [RFC-0010](RFC-0010-artifact-checkpoint-format.md) and
  [RFC-0014](RFC-0014-provenance-commitments.md).
- Coordinator failover / a leaderless topology (post-v1.0; see Open Questions). Stage E
  own-pretraining and Stage D realized proofs are out of v1.0 scope
  ([00 §8](../spec/00-overview.md#8-v10-scope-boundary)).

## Proposed Design

### 1. Runtime classes

The runtime lives in `lensemble.federation`:
`coordinator.py` (the `Coordinator`), `participant.py` (the `Participant`), `round.py` (the
`RoundState` enum and the round-driver), and `outer_optimizer.py` (the Nesterov outer step). The class
signatures are fixed by [conventions §5](../spec/conventions.md#5-public-api-surface) and re-exported from `lensemble.federation`.

```python
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from collections.abc import Mapping, Sequence
from torch import Tensor

from lensemble.config import LensembleConfig
from lensemble.federation.round import GlobalState, PseudoGradient  # see 03 §GlobalState / §PseudoGradient


class RoundState(str, Enum):
    """Lifecycle of one outer round (the federation round state machine, §2)."""
    OPEN = "open"               # global state pinned; RoundOpen broadcast; awaiting joins/commitments
    COLLECTING = "collecting"   # participants run H inner steps; masked Updates arrive
    AGGREGATING = "aggregating" # secure-agg reveals Σ_c Δ_c; determinism self-check
    ALIGNING = "aligning"       # frame-drift measured on probe; Procrustes backstop if drift > τ
    COMMITTING = "committing"   # outer Nesterov step; hash-commit (θ_{t+1}, φ_{t+1})
    CLOSED = "closed"           # RoundClose emitted; ledger appended; ready for round t+1
    ABORTED = "aborted"         # quorum / determinism / privacy / commitment failure; round discarded


class Coordinator:
    """Orchestrates rounds, holds the canonical global model, runs the outer optimizer.

    Untrusted w.r.t. raw data; honest-but-curious in Phase 1, a proving target in Phase 2
    (RFC-0006). Single round loop; one Coordinator per federation.
    """
    def __init__(self, config: LensembleConfig, *, transport: "Transport") -> None: ...

    def run(self, num_rounds: int) -> None:
        """Drive `num_rounds` outer rounds through the RoundState machine (§2).

        Each round is OPEN→COLLECTING→AGGREGATING→ALIGNING→COMMITTING→CLOSED, or short-circuits to
        ABORTED. On CLOSED, the committed global hash advances and a ContributionRecord is appended
        to the ContributionLedger (RFC-0014). Emits a RunManifest (RFC-0009).
        """

    def round_state(self) -> RoundState: ...                 # current state (observability/test hook)
    def global_state(self) -> GlobalState: ...               # current canonical (θ_t, φ_t) refs + round


class Participant:
    """Holds sovereign data; runs the inner loop; emits a privatized, bound PseudoGradient.

    Trusts neither coordinator nor peers with raw data (INV-RESIDENCY).
    """
    def __init__(self, config: LensembleConfig, *, participant_id: str, transport: "Transport") -> None: ...

    def local_round(self, global_state: GlobalState, round_seed: int) -> PseudoGradient:
        """Run H inner AdamW steps on local data, form Δ_c, clip+noise, bind to R_c, return it.

        Preconditions: probe hash in `global_state` equals the pinned probe hash (INV-PROBE-PIN, else
        ProbeError); round-0 encoder is hash-identical to the warm-start (INV-WARMSTART-T0, else
        GaugeError); the sketch matrix A is derived from `round_seed` (INV-SKETCH-CONSISTENCY).
        Postconditions: returned PseudoGradient satisfies l2_norm <= C_clip (INV-DP-BOUND), covers
        only (θ, φ) param groups (INV-ACTIONHEAD-LOCAL), and carries exactly one dataset_root
        (INV-COMMIT-BINDING). Raw observations/actions/embeddings never appear (INV-RESIDENCY).
        """

    def join(self, coordinator_endpoint: str) -> GlobalState:
        """Register with the coordinator; recover the latest committed GlobalState (rejoiner path, §3)."""
```

`GlobalState` and `PseudoGradient` are defined in [03 §GlobalState](../spec/03-data-model.md#7-globalstate--the-broadcast-round-state) and
[03 §PseudoGradient](../spec/03-data-model.md#6-pseudogradient--the-one-private-object-that-does-cross-the-boundary); the protocol shape of `PseudoGradient` is reproduced in
[RFC-0003 §3](RFC-0003-federated-protocol.md#3-the-pseudogradient-contract). `GlobalState` carries the `(θ, φ)` refs, the round index,
the sketch seed, and the probe hash — exactly the payload `RoundOpen` broadcasts.

The `Coordinator` does not construct, broadcast, or aggregate any per-embodiment action head
$h_\psi^{(c)}$; the broadcast and aggregation payloads are materialized only over the encoder $\theta$
and predictor $\phi$ param groups (`INV-ACTIONHEAD-LOCAL`, enforced in `lensemble.federation` per
[RFC-0001 §4](RFC-0001-architecture.md#4-federation-map)). The outer Nesterov step
($(\theta_{t+1},\phi_{t+1}) = (\theta_t,\phi_t) - \eta_{\text{out}}\,\mathrm{Nesterov}(\tfrac1C\sum_c\Delta_c)$,
[RFC-0003 §2](RFC-0003-federated-protocol.md#2-round-structure-diloco-outer-loop)) lives in `outer_optimizer.py` and is the
bitwise-deterministic function guarded by `INV-AGG-DETERMINISM` (§4).

### 2. The `RoundState` state machine

One outer round is the eight-step lifecycle of [RFC-0003 §2](RFC-0003-federated-protocol.md#2-round-structure-diloco-outer-loop) realized as
six progress states plus a terminal `ABORTED`. The coordinator owns the state; participants observe it
only through the control-plane messages (§5). The states map onto the lifecycle steps so the two stay
consistent:

```text
                  quorum reached
   ┌────────┐    (≥ K joins,            ┌────────────┐  all Updates in OR
   │  OPEN  │──── Commitments bound)────►│ COLLECTING │── collect timeout w/ quorum ─┐
   └────────┘                            └────────────┘                              │
        │                                      │                                     ▼
        │ quorum not reached                   │ < K participants remain    ┌──────────────┐
        │ by open timeout                      │ (FaultToleranceExceeded)   │ AGGREGATING  │
        ▼                                      ▼                            └──────────────┘
   ┌─────────┐                          ┌─────────┐                                │
   │ ABORTED │◄─────────────────────────│ ABORTED │◄── below secure-agg threshold  │ Σ_c Δ_c revealed;
   └─────────┘   determinism self-check  └─────────┘    (SecureAggregationError)    │ determinism OK
        ▲        fails (NonDeterministic-      ▲                                     ▼
        │        Aggregation): re-sum;         │ self-check fails on retry   ┌──────────────┐
        │        on second failure ────────────┘                            │   ALIGNING   │
        │                                                                    └──────────────┘
        │ CommitmentMismatch / CheckpointIntegrityError                             │ drift measured;
        │ / PrivacyBudgetExceeded (fail-closed, no retry)                           │ backstop if τ
        │                                                                           ▼
        │                                                                    ┌──────────────┐
        └──────────────────────────────────────────────────────────────────│  COMMITTING  │
                                                                             └──────────────┘
                                                                                    │ outer step;
                                                                                    │ hash-commit
                                                                                    ▼
                                                                             ┌──────────────┐
                                                                             │    CLOSED    │
                                                                             └──────────────┘
```

In prose, each transition with its trigger, precondition, and failure handling:

- **(init) → `OPEN`.** The coordinator pins the canonical global state `(θ_t, φ_t)`, derives the round
  sketch seed `s_t = derive(root_seed, t)` ([conventions §9](../spec/conventions.md#9-determinism-dtype-device), [RFC-0009 §4](RFC-0009-configuration-reproducibility.md#4-seeding-scheme)),
  and broadcasts `RoundOpen` (the global hash, `s_t`, the probe hash, the landmark hashes, `H`).
  Participants validate the probe hash against the pinned hash; a mismatch raises `ProbeError` and the
  participant rejects the round (`INV-PROBE-PIN`); round-0 joins additionally validate
  `INV-WARMSTART-T0`. This is lifecycle step 1 (Broadcast).
- **`OPEN` → `COLLECTING`.** Trigger: a **quorum** of `K` participants have joined and each has bound a
  valid `Commitment` (its dataset Merkle root `R_c`, [RFC-0014](RFC-0014-provenance-commitments.md)).
  `K = max(min_participants, secure_agg_threshold)` (§3). Failure: if the quorum is not reached by the
  open timeout, the round transitions to `ABORTED` with `FaultToleranceExceeded`.
- **`COLLECTING` → `AGGREGATING`.** Participants run `H` inner AdamW steps (lifecycle step 2), form
  `Δ_c` (step 3), clip-and-noise (step 4, `INV-DP-BOUND`), and send a masked `Update`. Trigger: all
  expected `Update`s have arrived, OR the collect timeout fires while a quorum's worth of `Update`s are
  present (elastic completion, §3). Failure: if participant count drops below `K` (dropouts), the round
  transitions to `ABORTED` with `FaultToleranceExceeded`. A `PrivacyBudgetExceeded` reported by any
  participant's accountant before release ([RFC-0012](RFC-0012-differential-privacy.md)) drives the
  round to `ABORTED` fail-closed (training stops; no retry).
- **`AGGREGATING` → `ALIGNING`.** The secure aggregator reveals `Σ_c Δ_c` over the *set of contributing
  participants for this round* (lifecycle step 5; [RFC-0011](RFC-0011-secure-aggregation.md)). The
  coordinator runs the determinism self-check (§4): it re-sums the revealed contributions in the fixed
  canonical order and compares bitwise. Trigger: the revealed sum reproduces. Failure: below the
  secure-aggregation threshold → `ABORTED` with `SecureAggregationError`; a self-check mismatch →
  `NonDeterministicAggregation`, re-sum once, and on a second mismatch → `ABORTED` (never swallowed,
  `INV-AGG-DETERMINISM`).
- **`ALIGNING` → `COMMITTING`.** The coordinator measures frame drift on the public probe
  (`frame_drift`, [RFC-0002 §9](RFC-0002-gauge-and-aggregation.md#9-the-frame-drift-diagnostic-the-headline-measurement)) and, *only if* drift exceeds the
  configured threshold `τ`, folds in the hard Procrustes alignment `Q_c^*` (`procrustes_align`,
  [RFC-0002 §5](RFC-0002-gauge-and-aggregation.md#5-layer-3--procrustes-re-alignment-at-aggregation-backstop); lifecycle step 6). Trigger: drift measured and the
  backstop (if any) applied. Failure: a degenerate SVD raises `DegenerateProcrustes`; the runtime
  clamps/conditions per [RFC-0002 §5](RFC-0002-gauge-and-aggregation.md#5-layer-3--procrustes-re-alignment-at-aggregation-backstop) and proceeds, or aborts if
  conditioning fails. With Layer-2 anchoring active this backstop should rarely bind.
- **`COMMITTING` → `CLOSED`.** The coordinator applies the deterministic outer Nesterov step (lifecycle
  step 7), hash-commits `(θ_{t+1}, φ_{t+1})` ([RFC-0010](RFC-0010-artifact-checkpoint-format.md),
  `INV-CHECKPOINT-HASH`), appends a `ContributionRecord` to the `ContributionLedger`
  ([RFC-0014 §7](RFC-0014-provenance-commitments.md#7-the-contributionledger)), and emits `RoundClose`
  (lifecycle step 8). Failure: a `Δ_c` not bound to a valid `R_c` is rejected at ingress with
  `CommitmentMismatch` (never swallowed, `INV-COMMIT-BINDING`) and the round transitions to `ABORTED`; a
  checkpoint that fails its integrity hash raises `CheckpointIntegrityError` and aborts.
- **`CLOSED` → (next) `OPEN`.** The committed `(θ_{t+1}, φ_{t+1})` becomes round `t+1`'s pinned global
  state. Late or dropped participants reconcile here (§3).
- **any → `ABORTED`.** A discarded round. The canonical global state is unchanged from round `t` (the
  outer step is never half-applied; the commit in `COMMITTING` is the single atomic state advance). The
  coordinator either retries round `t` (transient: dropout-recoverable secure-agg failure, a single
  determinism re-sum) or stops the federation (fail-closed: `PrivacyBudgetExceeded`,
  `CommitmentMismatch`, repeated `NonDeterministicAggregation`). The retry-vs-stop decision per trigger
  is in §7.

A round is atomic at `COMMITTING`: either the outer step applies and the global hash advances
(`CLOSED`), or no state change occurs (`ABORTED`). There is no partial commit.

### 3. Fault tolerance & elasticity

The runtime follows the INTELLECT-1/PRIME elastic model ([RFC-0003 §6](RFC-0003-federated-protocol.md#6-heterogeneity--fault-tolerance)):
an outer step proceeds with whatever participants are present; the Nesterov outer optimizer is robust to
a varying participant count (a known DiLoCo property). The mechanics:

- **Quorum.** A round may leave `OPEN` only with `K = max(min_participants, secure_agg_threshold)`
  participants joined and committed. `min_participants` is a config floor (the federation's minimum
  meaningful breadth); `secure_agg_threshold` is the dropout threshold of the secure-aggregation scheme
  ([RFC-0011](RFC-0011-secure-aggregation.md)). Below `K`, the round `ABORTED`s with
  `FaultToleranceExceeded`.
- **Per-participant timeout.** Each participant has a `collect_timeout` (config) for the `COLLECTING`
  phase. A participant whose `Update` does not arrive by the timeout is treated as dropped for this
  round; the round completes elastically over the present set if it still satisfies the
  secure-aggregation threshold. The averaging denominator is the *actual contributing count* `C_t`, not
  the registered count: `(1/C_t) Σ_{c present} Δ_c`. `C_t` is recorded in the `ContributionRecord` and
  the `RunManifest`, so the deterministic outer step is reproducible (`INV-AGG-DETERMINISM`).
- **Late / dropped reconciliation.** A participant that missed round `t` does not retroactively
  contribute; it reconciles at round `t+1` by pulling the committed global state in the next `RoundOpen`.
  No back-application — this keeps each round's contributing set, and therefore the outer step, an
  explicit recorded input.
- **Rejoiner recovery.** `Participant.join` recovers the latest committed `GlobalState` from the
  coordinator (a hash-verified checkpoint, [RFC-0010](RFC-0010-artifact-checkpoint-format.md); tamper
  raises `CheckpointIntegrityError`). A rejoiner that has been absent across many rounds simply starts
  from the current global state; its stale local state is discarded. Round-0 rejoiners revalidate
  `INV-WARMSTART-T0`.
- **`FaultToleranceExceeded` threshold.** Raised when participants present fall below `K` at the
  `OPEN→COLLECTING` quorum check or during `COLLECTING`. The round `ABORTED`s; the coordinator waits for
  joins and retries round `t` (the global state did not advance). This is distinct from
  `SecureAggregationError`, which is raised by the aggregator when the *masking* threshold specifically
  is unmet ([RFC-0011](RFC-0011-secure-aggregation.md)); both are dropout conditions but at different
  layers, and the runtime treats both as round-`ABORTED`-then-retry.

### 4. Determinism wiring (`INV-AGG-DETERMINISM`)

The outer step is the proof-ready surface ([RFC-0006 §2](RFC-0006-verifiable-contribution.md#2-the-provable-surface-and-what-stays-cheap)), so the
runtime makes its inputs explicit and its computation reproducible. The runtime guarantees:

- The set of contributing deltas for round `t` is fixed at the `COLLECTING→AGGREGATING` transition (no
  delta arriving after the transition is admitted into the round; it reconciles next round).
- The summation order is the canonical order — participants sorted by `participant_id` (a total order),
  not arrival order — so the reduction `Σ_c Δ_c` is order-independent of the network
  ([RFC-0003 §7](RFC-0003-federated-protocol.md#7-determinism-concurrency-error-propagation): fixed reduction order, fp32/fp64, no atomics).
- The round seed `s_t`, the prior global params `(θ_t, φ_t)`, and the contributing set (with `C_t`) are
  recorded in the `RunManifest` ([RFC-0009](RFC-0009-configuration-reproducibility.md)), so the outer
  step is a pure function of recorded inputs.
- A determinism self-check runs at `AGGREGATING`: the coordinator re-sums in the canonical order and
  compares bitwise to the first reduction. A mismatch raises `NonDeterministicAggregation`; the runtime
  re-sums once, and on a second mismatch drives the round to `ABORTED` and surfaces the error (never
  swallowed). This is the abort-and-recompute response of
  [RFC-0003 §9](RFC-0003-federated-protocol.md#9-failure-modes) and [04 §5.4](../spec/04-error-model.md#54-aggregation-lensembleaggregation).

Inner-loop determinism is best-effort and seed-pinned, gated by `torch.use_deterministic_algorithms`
([conventions §9](../spec/conventions.md#9-determinism-dtype-device)); it is not on the aggregation critical path and a non-bitwise inner loop does not trip the
self-check.

### 5. Control-plane messages & transport

The four boundary-crossing messages are exactly those of
[RFC-0003 §8](RFC-0003-federated-protocol.md#8-message-summary), reproduced here with the state transition each drives and
the protection each carries. Raw observations, actions, and embeddings of private data appear in **no**
message (`INV-RESIDENCY`, fail-closed `ResidencyViolation`).

| Message | Direction | Drives transition | Contents | Protection |
|---|---|---|---|---|
| `RoundOpen` | coord → participant | enters `OPEN`; broadcast | `(θ_t, φ_t)` ref/hash, sketch seed `s_t`, probe hash, landmark hashes, `H` | integrity (hash) |
| `Commitment` | participant → coord | counts toward `OPEN→COLLECTING` quorum | dataset Merkle root `R_c` ([RFC-0014](RFC-0014-provenance-commitments.md)) | binding (`INV-COMMIT-BINDING`) |
| `Update` | participant → aggregator | counts toward `COLLECTING→AGGREGATING` | `Δ_c` (the `PseudoGradient.delta`), masked | DP (clip+noise) + secure-agg mask |
| `RoundClose` | coord → all | marks `CLOSED` | `(θ_{t+1}, φ_{t+1})` content hash | integrity (hash, `INV-CHECKPOINT-HASH`) |

The transport is abstracted behind a single interface so the same state machine runs in-process (Stage
B) and over a network (Stage C):

```python
from typing import Protocol

class Transport(Protocol):
    """Carries control-plane messages; pluggable per stage. No message carries raw data (INV-RESIDENCY)."""
    def send(self, peer_id: str, message: "ControlMessage") -> None: ...
    def recv(self, *, timeout_s: float) -> "ControlMessage | None": ...   # None on timeout
    def broadcast(self, message: "ControlMessage") -> None: ...
    def peers(self) -> Sequence[str]: ...
```

- **Stage B (v0.2) — `InProcessTransport`.** `C` simulated participants share one process; `send` enqueues
  onto an in-memory channel; timeouts are simulated wall-clock or step budgets. The full round lifecycle
  runs deterministically for the experiments and the ablation ladder
  ([RFC-0005 §6](RFC-0005-evaluation.md#6-ablation-ladder-the-core-experiment)).
- **Stage C (v0.3) — networked transport.** A request/response transport over the network boundary
  carrying the four messages. Ingress validation at the coordinator rejects malformed
  `Update`/`Commitment` messages with the typed error from [04 — Error Model](../spec/04-error-model.md).

Message payloads are validated at ingress (pydantic v2, [conventions §8](../spec/conventions.md#8-core-data-types)); a `Δ_c` not bound to a valid `R_c`
raises `CommitmentMismatch` and the update is rejected (`INV-COMMIT-BINDING`, never swallowed).

### 6. Concurrency model

- **Coordinator process** — one per federation ([RFC-0001 §8](RFC-0001-architecture.md#8-process--concurrency-model)). It runs a
  single round loop: it owns the canonical global model, drives the `RoundState` machine, runs the outer
  optimizer, and is the hash-commitment authority. The round loop is sequential across states within a
  round; rounds are sequential (round `t+1` does not open until round `t` reaches `CLOSED` or `ABORTED`).
- **Participant processes** — one per sovereign node. They run the inner loop (FSDP/TP ranks within the
  participant's trust domain, [RFC-0001 §5](RFC-0001-architecture.md#5-training-topology-two-level)) and emit a `PseudoGradient`. They
  never share an address space across a real boundary (Stage C); Stage B simulates them in one process.
- **Backpressure & timeouts.** The coordinator does not block indefinitely on any participant. The
  `OPEN` and `COLLECTING` phases each have a config timeout; on timeout the runtime evaluates the quorum
  / threshold and either completes elastically or `ABORTED`s. A slow participant cannot stall the
  federation past `collect_timeout`; its contribution is dropped for the round and reconciles next round
  (§3).
- **No nondeterministic concurrency on the aggregation path.** Concurrency is confined to message
  ingress and to the (already trust-domain-internal) inner loops. The reduction, the determinism
  self-check, the alignment, and the outer step run on the single coordinator thread in the canonical
  order (§4), so concurrency never enters `INV-AGG-DETERMINISM`'s function.

### 7. Failure modes

Each runtime failure mode, the error from the [conventions §6](../spec/conventions.md#6-error-taxonomy) taxonomy, and the system response. Security- and
correctness-critical errors (`ResidencyViolation`, `CommitmentMismatch`, `NonDeterministicAggregation`)
are never swallowed ([04 §7](../spec/04-error-model.md#7-error-handling-rules)).

| Trigger | Detected at | Error | State response |
|---|---|---|---|
| Quorum `< K` at open / dropouts during collect | `OPEN→COLLECTING` quorum check; `COLLECTING` monitor | `FaultToleranceExceeded` | `ABORTED`; global state unchanged; wait for joins, retry round `t` |
| Participants below secure-agg threshold | aggregator threshold ([RFC-0011](RFC-0011-secure-aggregation.md)) | `SecureAggregationError` | `ABORTED`; retry round `t` if dropouts recoverable, else stop |
| Revealed sum non-reproducible | determinism self-check (§4) | `NonDeterministicAggregation` | re-sum once; second failure → `ABORTED`; never swallowed |
| `(ε,δ)` budget spent | accountant pre-release ([RFC-0012](RFC-0012-differential-privacy.md)) | `PrivacyBudgetExceeded` | `ABORTED` fail-closed; stop training (no retry) |
| `Δ_c` not bound to a valid `R_c` | ingress binding check | `CommitmentMismatch` | reject the update; `ABORTED`; never swallowed (`INV-COMMIT-BINDING`) |
| Probe hash ≠ pinned probe hash | participant probe-pin check | `ProbeError` | participant rejects `RoundOpen`; re-anchor required (`INV-PROBE-PIN`) |
| Round-0 encoder ≠ warm-start | join handshake | `GaugeError` | reject join (`INV-WARMSTART-T0`) |
| Participant uses a different `A` from `s_t` | sketch-consistency check | `GaugeError` | reject contribution (`INV-SKETCH-CONSISTENCY`) |
| Degenerate Procrustes SVD | `ALIGNING` (`procrustes_align`) | `DegenerateProcrustes` | clamp/condition ([RFC-0002 §5](RFC-0002-gauge-and-aggregation.md#5-layer-3--procrustes-re-alignment-at-aggregation-backstop)); abort if conditioning fails |
| Committed checkpoint fails integrity hash | `COMMITTING` / rejoiner recovery | `CheckpointIntegrityError` | refuse the artifact; `ABORTED` (`INV-CHECKPOINT-HASH`) |
| Released delta includes raw data / private embedding | residency guard at participant egress | `ResidencyViolation` | fail-closed; never caught-and-ignored (`INV-RESIDENCY`) |
| Released delta includes an action-head param group | param-group check at `PseudoGradient` construction | `ResidencyViolation` | reject; action heads are local (`INV-ACTIONHEAD-LOCAL`) |
| Malformed `Update`/`Commitment` at ingress | message-schema validation | `RoundError` | reject message; do not advance state |

## Alternatives Considered

- **Centralized coordinator vs gossip / all-reduce topology.** What it is: a single coordinator owns the
  canonical model and drives the round, versus a leaderless peer-to-peer reduction. Why considered:
  gossip removes the Phase-1 single point of failure and trust
  ([RFC-0001 Drawbacks](RFC-0001-architecture.md#drawbacks)). Why the coordinator is chosen for Phase 1: a single
  coordinator gives a clean round boundary for the determinism self-check (`INV-AGG-DETERMINISM`), a
  single orchestration point for secure aggregation and commitment, and the natural proving target for
  Phase 2 ([RFC-0006](RFC-0006-verifiable-contribution.md)); gossip would scatter the aggregation
  reduction across peers and complicate both determinism and the proof. Coordinator failover toward a
  leaderless topology is a post-v1.0 Open Question.
- **gRPC vs HTTP/REST transport.** What it is: the concrete wire protocol behind the `Transport`
  abstraction for Stage C. Why considered: gRPC offers streaming and binary framing suited to large
  `Δ_c` payloads; HTTP/REST is simpler to operate and debug across organizational boundaries. Why
  deferred, not decided: the state machine is transport-agnostic (§5), so the choice is an
  implementation detail bound at Stage C; it is an Open Question, not a blocker.
- **Synchronous (round-barrier) vs asynchronous rounds.** What it is: a global round barrier where the
  outer step waits for the round's contributing set, versus participants pushing updates without a
  barrier. Why considered: async maximizes elasticity and removes straggler stall. Why synchronous is
  chosen ([RFC-0003 Alternatives](RFC-0003-federated-protocol.md#alternatives-considered)): a defined round boundary is what
  makes the determinism self-check and the probe alignment well-defined; the elastic-but-synchronous
  round (§3) recovers most of async's churn robustness without sacrificing `INV-AGG-DETERMINISM`.
  Revisit post-v1.0.
- **In-process simulation harness (Stage B) vs real networking from day one.** What it is: run `C`
  participants in one process behind `InProcessTransport`, versus building the networked transport
  first. Why considered: a real network exercises the true failure surface earlier. Why the in-process
  harness is chosen for Stage B: it makes the full round lifecycle and the ablation ladder deterministic
  and CPU-runnable for the paper's experiments without the operational cost of a network, while sharing
  the exact state machine that Stage C will drive over the wire — so the experiments validate the
  deployment code path, not a stand-in.

## Drawbacks

- **Coordinator is a single point of failure and trust in Phase 1.** It owns liveness (no coordinator,
  no round) and orchestration; it is honest-but-curious. Phase 2 makes the outer step provable
  ([RFC-0006](RFC-0006-verifiable-contribution.md)) but does not remove the liveness dependency.
  Mitigation path: coordinator failover is an Open Question (post-v1.0); secure aggregation
  ([RFC-0011](RFC-0011-secure-aggregation.md)) already hides individual `Δ_c` from a curious
  coordinator, so the confidentiality cost of centralization is bounded even in Phase 1.
- **Elastic averaging over a varying contributing set is a moving target.** Recording `C_t` keeps the
  outer step reproducible, but a round whose contributing set differs from the prior round's is not
  directly comparable; this is acceptable for DiLoCo's outer optimizer but complicates per-round
  analysis. Mitigation: `C_t` and the contributing `participant_id` set are recorded in every
  `ContributionRecord` and `RunManifest`.
- **Timeout tuning is a liveness/quality trade-off.** A short `collect_timeout` drops slow-but-honest
  participants and shrinks the round; a long one stalls the federation on a straggler. The thresholds are
  config, not hard-coded, and are characterized in Stage B.

## Migration / Rollout

The runtime rolls out along the staged plan ([RFC-0001 §Migration](RFC-0001-architecture.md#migration--rollout),
[00 §8](../spec/00-overview.md#8-v10-scope-boundary)):

- **v0.2 / Stage B — single-process simulation.** The complete `RoundState` machine, the
  `Coordinator`/`Participant` classes, the outer optimizer, fault tolerance, and the determinism
  self-check run in-process behind `InProcessTransport`. This is the harness the federated experiments
  and the ablation ladder ([RFC-0005 §6](RFC-0005-evaluation.md#6-ablation-ladder-the-core-experiment)) run on.
- **v0.3 / Stage C — networked two-node deployment.** The same state machine drives over the network
  boundary: swap `InProcessTransport` for the networked transport, enable real secure aggregation
  ([RFC-0011](RFC-0011-secure-aggregation.md)) and DP ([RFC-0012](RFC-0012-differential-privacy.md)),
  enforce residency over the wire, and exercise real churn / rejoiner recovery. The contracts
  (`Coordinator`, `Participant`, `RoundState`, the message table, `GlobalState`) are stable from v0.2;
  Stage C changes the transport and the aggregation backend, not the state machine
  ([RFC-0003 §Migration](RFC-0003-federated-protocol.md#migration--rollout)).

No data migration is required between stages. The `RoundState` enum and the runtime class signatures are
part of the public surface and follow the pre-1.0 / 1.0 freeze policy
([02 §3](../spec/02-public-api.md#3-stability--versioning-policy), [09 — Release & Versioning](../spec/09-release-and-versioning.md)).

## Testing Strategy

CPU-runnable tests on tiny synthetic fixtures (no large downloads,
[07 §8](../spec/07-testing-strategy.md#8-ci-gates)):

- **State-machine transition tests.** Drive a toy round through
  `OPEN→COLLECTING→AGGREGATING→ALIGNING→COMMITTING→CLOSED` and assert each transition's trigger and
  precondition; explicitly assert the `ABORTED` path on quorum failure, on a below-threshold dropout, on
  a forced determinism mismatch, on `PrivacyBudgetExceeded`, and on `CommitmentMismatch`. Assert that
  `ABORTED` leaves the canonical global hash unchanged (no partial commit).
- **Churn / elasticity simulation.** Simulate participant dropout above and below `K`: above completes
  elastically with the recorded `C_t`; below raises `FaultToleranceExceeded` and `ABORTED`s, then retries
  round `t` after joins. Assert the Nesterov outer step is stable under a varying participant count
  ([RFC-0003 §Testing](RFC-0003-federated-protocol.md#testing-strategy)).
- **Rejoiner recovery.** A participant absent for several rounds rejoins via `Participant.join`, recovers
  the latest committed `GlobalState`, validates the checkpoint hash (tamper → `CheckpointIntegrityError`),
  and contributes a well-formed `PseudoGradient` in the next round.
- **Determinism self-check wiring (`INV-AGG-DETERMINISM`).** Run the outer step twice on an identical
  contributing set and assert bitwise-identical `(θ_{t+1}, φ_{t+1})`; permute arrival order and assert
  the canonical-order reduction is unchanged; corrupt the reduction and assert the self-check raises
  `NonDeterministicAggregation` and the round `ABORTED`s after the single re-sum.
- **Message-schema validation.** Assert malformed `Update`/`Commitment` payloads are rejected at ingress
  with the typed error and the state does not advance; assert no raw observation/action/embedding appears
  in any message (`INV-RESIDENCY`; cf. [05 §5](../spec/05-observability.md#5-redaction-inv-residency)).
- **Timeout / backpressure behavior.** A participant that exceeds `collect_timeout` is dropped for the
  round; assert the federation does not stall and the dropped participant reconciles next round; assert
  the coordinator's round loop never blocks indefinitely on a single participant.
- **Round-lifecycle integration.** The end-to-end toy round of
  [RFC-0001 §Testing](RFC-0001-architecture.md#testing-strategy) (`RoundOpen` → local steps → clip+noise → simulated
  secure-agg → optional Procrustes backstop → deterministic outer step → hash commit → `RoundClose`)
  exercises this runtime; assert two identical-seed runs produce identical `RunManifest` aggregation
  hashes.

## Open Questions

OPEN QUESTION: The transport choice for the real network boundary (gRPC vs HTTP/REST) behind the
`Transport` abstraction (§5). The state machine is transport-agnostic, so this is bound at Stage C. Owner
@AbdelStark; resolution: Stage C (v0.3) implementation, shared with
[RFC-0003 §Open Questions](RFC-0003-federated-protocol.md#open-questions).

OPEN QUESTION: The maximum churn tolerance — how low `C_t` may fall (relative to `K` and the
secure-aggregation threshold) before the outer step is no longer meaningful, and whether `K` should
adapt to the federation's measured drop rate. Owner @AbdelStark; resolution: Stage-C (v0.3) churn
characterization, informed by the Stage-B elasticity simulation.

OPEN QUESTION: Coordinator failover / a leaderless (gossip) topology to remove the Phase-1 single point
of failure. Out of v1.0 scope. Owner @AbdelStark; resolution: a post-v1.0 follow-up RFC, reconsidering
the gossip alternative ([RFC-0003 Alternatives](RFC-0003-federated-protocol.md#alternatives-considered)) once the Phase-2
provable-coordinator story ([RFC-0006](RFC-0006-verifiable-contribution.md)) is in place.

RISK: The interaction between elastic completion (a varying contributing set `C_t`) and the gauge
backstop could compound — a round that drops several participants may both raise the per-round drift
(fewer anchored frames averaged) and shrink the effective batch the outer step sees, forcing the
Layer-3 Procrustes backstop ([RFC-0002 §5](RFC-0002-gauge-and-aggregation.md#5-layer-3--procrustes-re-alignment-at-aggregation-backstop)) to bind more often.
Resolution plan: the Stage-B simulation (§Migration) drives churn and drift jointly and reports both in
the frame-drift diagnostic ([RFC-0015 §3](RFC-0015-observability-diagnostics.md#3-the-frame-drift-diagnostic-emission-contract-the-headline-artifact));
if the coupling is severe, raise the quorum floor `K` so each round averages enough anchored frames.

## References

- [RFC-0001 — Architecture & System Overview](RFC-0001-architecture.md) (§5 two-level topology, §6 trust
  boundaries, §8 process/concurrency model, §Migration staged plan).
- [RFC-0003 — Federated Training Protocol](RFC-0003-federated-protocol.md) (§1 roles, §2 round lifecycle,
  §3 `PseudoGradient`, §6 heterogeneity & fault tolerance, §7 determinism/concurrency, §8 message table,
  §9 failure modes).
- [RFC-0002 — The Latent Gauge & Frame-Anchored Aggregation](RFC-0002-gauge-and-aggregation.md) (§5
  Procrustes backstop, §9 frame-drift diagnostic; the alignment invoked in `ALIGNING`).
- [RFC-0011 — Secure Aggregation Protocol](RFC-0011-secure-aggregation.md) (the aggregator interface,
  dropout threshold, `SecureAggregationError`).
- [RFC-0012 — Differential Privacy Accounting](RFC-0012-differential-privacy.md) (the accountant,
  `PrivacyBudgetExceeded`).
- [RFC-0010 — Checkpoint & Artifact Format](RFC-0010-artifact-checkpoint-format.md) (the committed
  global checkpoint, `INV-CHECKPOINT-HASH`, `CheckpointIntegrityError`).
- [RFC-0014 — Provenance Commitments & Merkle Scheme](RFC-0014-provenance-commitments.md) (the
  `Commitment`, `ContributionLedger`, `INV-COMMIT-BINDING`).
- [RFC-0009 — Configuration, Run Manifest & Reproducibility](RFC-0009-configuration-reproducibility.md)
  (seeding, the `RunManifest` recording the contributing set and `C_t`).
- [RFC-0006 — Verifiable Contribution](RFC-0006-verifiable-contribution.md) (§2 proof-ready outer step).
- [02 — Public API](../spec/02-public-api.md) · [03 — Data Model](../spec/03-data-model.md) ·
  [04 — Error Model](../spec/04-error-model.md) · [05 — Observability](../spec/05-observability.md) ·
  [07 — Testing Strategy](../spec/07-testing-strategy.md).
- External: DiLoCo / OpenDiLoCo / INTELLECT-1 / PRIME (elastic outer loop, fault tolerance over a
  varying participant count).
