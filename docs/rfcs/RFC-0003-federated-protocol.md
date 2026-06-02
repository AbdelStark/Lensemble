# RFC-0003 — Federated Training Protocol

| | |
|---|---|
| **RFC** | 0003 |
| **Title** | Federated Training Protocol |
| **Slug** | federated-protocol |
| **Status** | Accepted |
| **Track** | Standards |
| **Authors** | @AbdelStark |
| **Created** | 2026-06-02 |
| **Target milestone** | v0.2 (simulated federation, one cluster); real network boundary in v0.3 |
| **Area** | `area:federation` |
| **Requires** | [RFC-0001](RFC-0001-architecture.md), [RFC-0002](RFC-0002-gauge-and-aggregation.md) |
| **Defers to** | [RFC-0011](RFC-0011-secure-aggregation.md) (secure aggregation), [RFC-0012](RFC-0012-differential-privacy.md) (DP accounting), [RFC-0013](RFC-0013-coordinator-runtime.md) (runtime & state machine) |

## Summary

This RFC specifies the *operational* federated training protocol: how one outer round runs end to
end, what crosses a trust boundary and what never does, and the privacy and fault-tolerance machinery
that wraps each released update. The protocol is a DiLoCo outer loop — each participant runs `H` inner
optimization steps on its sovereign data, then emits a single pseudo-gradient `Δ_c` that is privatized,
securely aggregated, frame-aligned (backstop), and folded into the canonical global model by an outer
Nesterov step.

The aggregation *semantics* — frame alignment, anchoring, why naive weight-averaging is meaningless —
live in [RFC-0002](RFC-0002-gauge-and-aggregation.md); this RFC owns the *mechanics*. Three subsystems
that the round invokes are specified in their own RFCs and only referenced here: secure aggregation
([RFC-0011](RFC-0011-secure-aggregation.md)), differential-privacy accounting
([RFC-0012](RFC-0012-differential-privacy.md)), and the coordinator/participant runtime and round state
machine ([RFC-0013](RFC-0013-coordinator-runtime.md)). The contracts this RFC stabilizes are consumed
by the federation public API ([02 §Federation](../spec/02-public-api.md)) and the message-protection
table by the security model ([06 §Boundary-Crossing Messages](../spec/06-security.md)).

## Motivation

Lensemble must train one action-conditioned JEPA world model across mutually-distrusting participants
without raw trajectories ever leaving a participant's boundary
([01 §Trust Boundaries](../spec/01-architecture.md), `INV-RESIDENCY`). A synchronous per-step federated
scheme (FedAvg-style) would communicate the full parameter delta every optimizer step; for a
warm-started ViT-L (~300M, Stage A) scaling toward the 1.2B target (Stage E), per-step communication is
infeasible over a real network boundary and dominates wall time even on one cluster.

DiLoCo resolves the communication cost by communicating only every `H` inner steps, treating the
`H`-step local update as a single pseudo-gradient consumed by a momentum-bearing outer optimizer. But
DiLoCo was designed for objectives with a fixed output basis (next-token cross-entropy), where weight
averaging is approximately well-posed. The Lensemble objective has no fixed basis and is invariant
under `O(d)` rotation of the latent space ([RFC-0002 §2](RFC-0002-gauge-and-aggregation.md)); the
longer the horizon `H`, the further participant frames rotate apart before the outer step
([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md)). The protocol must therefore couple the DiLoCo
schedule to the gauge machinery: `H` is not a free communication knob but a control parameter traded
against frame drift. This RFC defines that coupling and the surrounding privacy/fault-tolerance
wrapper, leaving the gauge correction itself to RFC-0002.

## Goals

- Specify the per-round lifecycle (broadcast → local optimization → pseudo-gradient → privatize →
  secure-aggregate → backstop align → outer step → commit) with the exact data that crosses each step.
- Pin the inner/outer optimizer split: inner AdamW for `H` steps; outer Nesterov momentum over averaged
  pseudo-gradients.
- Define the `PseudoGradient` contract (`Δ_c`, its L2 norm, and its binding `dataset_root`) and state
  where it is constructed and validated.
- Define the differential-privacy clip-then-noise step at the protocol level (`INV-DP-BOUND`) and defer
  accounting to [RFC-0012](RFC-0012-differential-privacy.md).
- State the secure-aggregation requirement (coordinator learns only `Σ_c Δ_c`) and defer the protocol
  to [RFC-0011](RFC-0011-secure-aggregation.md).
- Specify heterogeneity handling (per-embodiment action heads stay local, `INV-ACTIONHEAD-LOCAL`) and
  fault tolerance / elasticity (an outer step proceeds with whatever participants are present).
- Define optional int8 pseudo-gradient quantization for communication efficiency, orthogonal to the
  gauge.
- Reproduce the boundary-crossing message table and its protection per message.
- State the determinism contract for the aggregation/outer-step path (`INV-AGG-DETERMINISM`) and how it
  is enforced.

## Non-Goals

- The secure-aggregation cryptographic protocol (pairwise masking, threshold secret-sharing, TEE
  backend). Owned by [RFC-0011](RFC-0011-secure-aggregation.md).
- The `(ε,δ)` accountant, mechanism analysis, and joint calibration experiment. Owned by
  [RFC-0012](RFC-0012-differential-privacy.md).
- The coordinator/participant runtime classes, the `RoundState` state machine, and the transport.
  Owned by [RFC-0013](RFC-0013-coordinator-runtime.md).
- The gauge correction (anchoring, Procrustes), proved and specified in
  [RFC-0002](RFC-0002-gauge-and-aggregation.md).
- The dataset-commitment construction (Merkle scheme), owned by
  [RFC-0014](RFC-0014-provenance-commitments.md).
- The model and objective internals, owned by [RFC-0008](RFC-0008-model-objective-numerics.md).
- Stage E own-foundation-scale federated pretraining and Stage D cryptographic proofs (out of v1.0
  scope per [00 §v1.0 Scope Boundary](../spec/00-overview.md)).

## Proposed Design

### 1. Roles

Three roles participate in a round. The runtime classes that realize them are specified in
[RFC-0013 §Runtime Classes](RFC-0013-coordinator-runtime.md); here only their protocol responsibilities
are fixed.

- **Participant** `c` — holds sovereign data; runs local training under intra-participant parallelism
  (the inner loop, [01 §Training Topology](../spec/01-architecture.md)); emits a `PseudoGradient`.
  Trusts neither the coordinator nor peers with raw data.
- **Coordinator** — orchestrates rounds, holds the canonical global model `(θ_t, φ_t)`, runs the outer
  optimizer, and commits each new global state. Untrusted with respect to raw data; honest-but-curious
  in Phase 1, a proving target in Phase 2 ([RFC-0006](RFC-0006-verifiable-contribution.md)).
- **Secure aggregator** — computes the masked sum `Σ_c Δ_c` without revealing any individual `Δ_c`. May
  be the coordinator under a secure-aggregation protocol, or a distinct party
  ([RFC-0011](RFC-0011-secure-aggregation.md)).

### 2. Round structure (DiLoCo outer loop)

Per outer round `t`, the protocol executes these steps. This is the operational expansion of the
algorithm sketch in [RFC-0002 §8](RFC-0002-gauge-and-aggregation.md); the two MUST stay consistent.

1. **Broadcast.** The coordinator sends `RoundOpen`: a reference/hash of the global parameters
   `(θ_t, φ_t)`, the round sketch seed `s_t`, the public-probe content hash, the landmark hashes, and
   the inner horizon `H`. Per-embodiment action heads `h_ψ^(c)` remain local and are never broadcast or
   aggregated (`INV-ACTIONHEAD-LOCAL`, enforced in `lensemble.federation`). The sketch seed `s_t` yields
   the identical SIGReg projection matrix `A` for every participant
   (`INV-SKETCH-CONSISTENCY`; [RFC-0002 §3](RFC-0002-gauge-and-aggregation.md)).
2. **Local optimization.** Each participant runs `H` inner steps with **AdamW** on the objective of
   [RFC-0002 §1](RFC-0002-gauge-and-aggregation.md) (prediction + `λ_sig·SIGReg_A` + `λ_anc·L_anchor`),
   over local data only. Raw data never leaves the boundary (`INV-RESIDENCY`).
3. **Pseudo-gradient.** Form `Δ_c = (θ_c^local, φ_c^local) − (θ_t, φ_t)`. DiLoCo treats the `H`-step
   local update as a single "gradient" for the outer optimizer.
4. **Privatize.** Clip and noise `Δ_c` per-participant (§3 below; `INV-DP-BOUND`).
5. **Secure-aggregate.** Compute `Σ_c Δ_c` without revealing any individual `Δ_c`
   ([RFC-0011](RFC-0011-secure-aggregation.md)).
6. **Backstop align.** Recompute the hard Procrustes alignment `Q_c^*` on the public probe and fold it
   in *only if* drift exceeds the configured threshold
   ([RFC-0002 §5](RFC-0002-gauge-and-aggregation.md)). With Layer-2 anchoring active this should
   rarely bind.
7. **Outer step.** Apply Nesterov momentum to the averaged pseudo-gradient:
   `(θ_{t+1}, φ_{t+1}) = (θ_t, φ_t) − η_out · Nesterov((1/C) Σ_c Δ_c)`. This path is bitwise
   deterministic (§7, `INV-AGG-DETERMINISM`).
8. **Commit.** Hash-commit `(θ_{t+1}, φ_{t+1})` ([RFC-0010](RFC-0010-artifact-checkpoint-format.md)
   format, `INV-CHECKPOINT-HASH`) and emit `RoundClose`.

**Sync frequency.** The protocol communicates every `H` inner steps. DiLoCo/INTELLECT use
`H ≈ 500` for LLMs; Lensemble starts smaller (`H ∈ [50, 500]`) while characterizing drift. `H` is tuned
against the frame-drift diagnostic ([RFC-0002 §9](RFC-0002-gauge-and-aggregation.md)): a larger `H`
cuts communication but lets participant frames rotate further apart before the outer step, raising the
anchoring burden. The communication-efficiency claim (`fed/comm_bytes` falls ~linearly in `H`) and how
it is measured are quantified in [08 §Communication Efficiency](../spec/08-performance-budget.md).

```text
round t
  coord ── RoundOpen(θ_t,φ_t hash, s_t, probe hash, H) ──► participants (broadcast)
                                                              │
  participant c:  H inner AdamW steps on local data  ◄────────┘  (raw data never leaves)
                  Δ_c = (θ_c,φ_c) − (θ_t,φ_t)
                  clip ‖Δ_c‖ ≤ C_clip ; add Gaussian noise        (RFC-0012)
                  ── Update(masked Δ_c) ──► secure aggregator       (RFC-0011)
  aggregator:     Σ_c Δ_c  (individual Δ_c hidden)
  coord:          backstop Procrustes-align on probe if drift > τ   (RFC-0002 §5)
                  (θ_{t+1},φ_{t+1}) = (θ_t,φ_t) − η_out·Nesterov(mean_c Δ_c)   [deterministic]
                  hash-commit (θ_{t+1},φ_{t+1})
                  ── RoundClose((θ_{t+1},φ_{t+1}) hash) ──► all
```

The diagram restates step-for-step the eight-step lifecycle above: the coordinator broadcasts the
global state and round coordination data; each participant trains locally and returns a privatized,
masked pseudo-gradient; the aggregator reveals only the sum; the coordinator applies the deterministic
outer step and commits.

### 3. The `PseudoGradient` contract

The unit that crosses the boundary is the `PseudoGradient`. It carries the flat delta, its L2 norm
(post-clip), and the dataset Merkle root it is bound to (`INV-COMMIT-BINDING`). Full schema lives in
[03 §PseudoGradient](../spec/03-data-model.md); the protocol-relevant shape:

```python
from dataclasses import dataclass
from torch import Tensor

@dataclass(frozen=True)
class PseudoGradient:
    """Participant c's H-step local update, post-DP, bound to one dataset commitment."""
    delta: Tensor            # flat fp32 vector over (θ, φ) params; encoder + predictor only,
                             #   never action heads (INV-ACTIONHEAD-LOCAL)
    l2_norm: float           # ‖delta‖ AFTER clipping; satisfies l2_norm <= C_clip (INV-DP-BOUND)
    dataset_root: bytes      # the Merkle root R_c this update is bound to (INV-COMMIT-BINDING)
    round_index: int         # the round t this Δ_c was produced for
    participant_id: str      # for logging/correlation only; redacted from any cross-boundary payload

# Coordinator / Participant runtime signatures (02 Public API; specified in RFC-0013).
# Participant.local_round(global_state: GlobalState, round_seed: int) -> PseudoGradient
# Coordinator.run(num_rounds: int) -> None
```

Construction (`Participant.local_round`) is in `lensemble.federation.participant`; the clip-then-noise
transform is applied before the `PseudoGradient` leaves the participant. The action-head exclusion is
enforced at construction: the flat `delta` is materialized only over the federated parameter groups
(encoder `θ`, predictor `φ`). Attempting to include an action-head parameter group in a released delta
is a contract violation raised at the residency boundary as `ResidencyViolation`
(`INV-ACTIONHEAD-LOCAL` / `INV-RESIDENCY`).

### 4. Differential privacy (protocol level)

Per-participant, before release, the protocol applies the Gaussian mechanism to the pseudo-gradient.
The mechanism, its `(ε,δ)` accounting, and the swappable accountant are specified in
[RFC-0012 §Mechanism](RFC-0012-differential-privacy.md); the protocol pins only the two operations and
their ordering:

- **Clip.** `Δ_c ← Δ_c · min(1, C_clip / ‖Δ_c‖)`. After this step,
  `‖Δ_c‖ ≤ C_clip` holds exactly (`INV-DP-BOUND`, enforced in `lensemble.privacy.dp`). This invariant
  is the precondition that makes the noise calibration sound and is asserted on the post-clip norm; a
  violation (a numerical drift past `C_clip`) raises `PrivacyBudgetExceeded` only if the accountant's
  budget is breached, but a clip-bound assertion failure is a defect, not a privacy event.
- **Noise.** Add `N(0, σ² C_clip² I)`, calibrated to a target `(ε,δ)` over the planned number of
  rounds. When the accountant reports the cumulative budget spent, the round refuses to release and
  raises `PrivacyBudgetExceeded`; training stops (the response is fail-closed, not degrade —
  [04 §Privacy](../spec/04-error-model.md)).

The privacy unit is the *participant's contribution to a round* (per-participant update DP), not
per-example DP-SGD in the inner loop ([RFC-0012](RFC-0012-differential-privacy.md) states this scope and
its limits).

**Interaction to tune (open).** DP noise on small predictor deltas interacts with SIGReg's variance and
with the anchor term. Joint calibration of `(σ, λ_sig, λ_anc, C_clip)` is a Stage-B experiment, not a
default (see Open Questions; shared with [RFC-0012](RFC-0012-differential-privacy.md)).

### 5. Secure aggregation (requirement)

The coordinator MUST learn only `Σ_c Δ_c`, never an individual `Δ_c` — an individual update leaks more
about a silo's data than the sum. The protocol requires a dropout-robust secure-aggregation scheme
(pairwise masking, Bonawitz-style, or a TEE-based aggregator); the construction, dropout-threshold
secret sharing, and the masked wire format are specified in [RFC-0011](RFC-0011-secure-aggregation.md).

Two protocol-level constraints this RFC imposes on that scheme:

- **Dropout robustness.** Participants may vanish mid-round (§6). The scheme MUST allow a round to
  complete when at least a configured threshold of participants remain; below threshold raises
  `SecureAggregationError` ([RFC-0011](RFC-0011-secure-aggregation.md)), and the round either retries or
  aborts via the runtime state machine ([RFC-0013](RFC-0013-coordinator-runtime.md)).
- **Determinism preservation.** Masking MUST cancel exactly so that the *revealed sum* is identical to
  the plaintext sum and introduces no nondeterminism on the aggregation path (`INV-AGG-DETERMINISM`,
  §7). A revealed sum that fails the determinism self-check raises `NonDeterministicAggregation`.

DP noise is added per-participant *before* masking ([RFC-0012](RFC-0012-differential-privacy.md) /
[RFC-0011 §DP Interaction](RFC-0011-secure-aggregation.md)).

### 6. Heterogeneity & fault tolerance

- **Embodiment heterogeneity** — handled at the model level. The shared encoder `f_θ` and predictor
  core `g_φ` federate; per-embodiment action encoders/heads `h_ψ^(c)` stay local
  ([01 §Federation Map](../spec/01-architecture.md), `INV-ACTIONHEAD-LOCAL`). The shared latent
  interface that makes this well-posed is the WMCP contract
  ([RFC-0007](RFC-0007-wmcp-latent-contract.md)).
- **Compute heterogeneity & churn** — INTELLECT-1/PRIME-style elasticity. An outer step proceeds with
  whatever participants are present; late or dropped participants reconcile at the next round; a
  rejoining participant recovers from the latest committed global checkpoint
  ([RFC-0010](RFC-0010-artifact-checkpoint-format.md)). The Nesterov outer optimizer is robust to a
  varying participant count (a known DiLoCo property). When too few participants remain to satisfy the
  secure-aggregation threshold or a configured minimum, the round raises `FaultToleranceExceeded`; the
  state-machine handling is in [RFC-0013](RFC-0013-coordinator-runtime.md).
- **Communication compression** — optional int8 quantization of the pseudo-gradient (per INTELLECT-1's
  int8 all-reduce) cuts outer-step bandwidth. It is orthogonal to the gauge machinery: quantization
  operates on the flat `Δ_c` after clipping and noising and before masking, and its round-trip error is
  bounded and tested (Testing Strategy). The quantization scheme MUST preserve `INV-AGG-DETERMINISM` on
  the dequantized sum.

### 7. Determinism, concurrency, error propagation

- **Aggregation determinism (`INV-AGG-DETERMINISM`).** The outer step is a pure, bitwise-reproducible
  function of (committed deltas, round seed, prior global params). Reductions use a fixed summation
  order in fp32 (or fp64), no atomics, no nondeterministic GPU reductions
  ([03 §Determinism](../spec/03-data-model.md)). A determinism self-check runs each outer step
  (re-summing in the fixed order and comparing); failure raises `NonDeterministicAggregation` and the
  round aborts and recomputes ([04 §Aggregation](../spec/04-error-model.md)). This is the proof-ready
  discipline that lets the outer step be publicly recomputed
  ([RFC-0006 §5](RFC-0006-verifiable-contribution.md)).
- **Inner-loop determinism** is best-effort and seed-pinned; full determinism is gated by a config flag
  (`torch.use_deterministic_algorithms`) and is not on the aggregation critical path.
- **Concurrency.** The coordinator runs a single round loop; participants run concurrently as separate
  processes (in-process in the Stage-B simulation, networked in Stage C). Backpressure and timeouts are
  the runtime's concern ([RFC-0013](RFC-0013-coordinator-runtime.md)).
- **Error propagation.** Ingress validation at the coordinator rejects malformed `Update`/`Commitment`
  messages with the typed error from [04 — Error Model](../spec/04-error-model.md); a `CommitmentMismatch` (a
  released `Δ_c` not bound to a valid `R_c`) is never swallowed and rejects the update
  ([RFC-0014](RFC-0014-provenance-commitments.md)).

### 8. Message summary

The four boundary-crossing messages and their protection. This table is reproduced (with the same
protections) by the security model ([06 §Boundary-Crossing Messages](../spec/06-security.md)) and the
runtime control plane ([RFC-0013 §Control-Plane Messages](RFC-0013-coordinator-runtime.md)).

| Message | Direction | Contents | Protection |
|---|---|---|---|
| `RoundOpen` | coord → participant | `(θ_t, φ_t)` ref/hash, sketch seed `s_t`, probe hash, landmark hashes, `H` | integrity (hash) |
| `Update` | participant → aggregator | `Δ_c` (the `PseudoGradient.delta`) | DP (clip+noise) + secure-agg mask |
| `Commitment` | participant → coord | dataset Merkle root `R_c` ([RFC-0014](RFC-0014-provenance-commitments.md)) | binding (`INV-COMMIT-BINDING`) |
| `RoundClose` | coord → all | `(θ_{t+1}, φ_{t+1})` content hash | integrity (hash, `INV-CHECKPOINT-HASH`) |

Raw observations, actions, and embeddings of private data appear in **no** message (`INV-RESIDENCY`,
enforced by `lensemble.data.residency`; a violation is fail-closed `ResidencyViolation`).

### 9. Failure modes

| Trigger | Detection | Error | System response |
|---|---|---|---|
| Released delta would include raw data or a private embedding | residency guard at participant egress | `ResidencyViolation` | fail-closed; never caught-and-ignored (`INV-RESIDENCY`) |
| Released delta includes an action-head param group | param-group check at `PseudoGradient` construction | `ResidencyViolation` | reject; action heads are local (`INV-ACTIONHEAD-LOCAL`) |
| Participants present below secure-agg threshold | aggregator threshold check | `SecureAggregationError` | round retries or aborts ([RFC-0013](RFC-0013-coordinator-runtime.md)) |
| Too few participants to run a round | coordinator quorum check | `FaultToleranceExceeded` | round aborted; reconcile next round |
| Revealed sum non-reproducible under self-check | per-outer-step determinism self-check | `NonDeterministicAggregation` | abort outer step and recompute; never swallowed (`INV-AGG-DETERMINISM`) |
| Post-clip norm exceeds `C_clip` | assertion on clipped norm | defect (assertion); privacy event only if budget breached | abort; clip path is a correctness bug (`INV-DP-BOUND`) |
| `(ε,δ)` budget spent | accountant query before release | `PrivacyBudgetExceeded` | stop training (fail-closed) |
| `Δ_c` not bound to a valid `R_c` | commitment binding check at ingress | `CommitmentMismatch` | reject the update (`INV-COMMIT-BINDING`); never swallowed |
| Broadcast probe hash ≠ pinned probe hash | participant probe-pin check | `ProbeError` | reject `RoundOpen`; re-anchor required (`INV-PROBE-PIN`) |
| Participant uses a different `A` from `s_t` | sketch-consistency check | `GaugeError` | reject contribution (`INV-SKETCH-CONSISTENCY`) |

## Alternatives Considered

- **DiLoCo (communicate every `H` steps).** What it is: an outer optimizer over `H`-step pseudo-
  gradients. Why considered: it is the only scheme whose communication is independent of inner-step
  count and that has demonstrated robustness to varying participant count at scale (OpenDiLoCo,
  INTELLECT-1). Why chosen: it makes the communication cost a tunable function of `H` rather than of the
  optimizer step count, which is what makes a real sovereign boundary (Stage C) viable.
- **Synchronous FedAvg (per-step communication).** What it is: average the parameter delta every inner
  step. Why considered: simplest and the textbook federated baseline; it is the *negative control* in
  the evaluation ([RFC-0005](RFC-0005-evaluation.md)). Why rejected as the protocol: per-step
  communication over a network boundary is infeasible at ViT-L scale and dominates wall time even in the
  one-cluster simulation; and naive averaging is anyway meaningless under the latent gauge
  ([RFC-0002 §2](RFC-0002-gauge-and-aggregation.md)).
- **Fully asynchronous rounds.** What it is: participants push updates without a global round barrier.
  Why considered: maximum elasticity and no straggler stall. Why rejected for Phase 1: it complicates
  the determinism contract (`INV-AGG-DETERMINISM`) and the gauge backstop (a probe alignment per round
  needs a defined round boundary); the elastic-but-synchronous DiLoCo round gives most of the churn
  robustness without giving up determinism. Revisit post-v1.0.
- **All-reduce / gossip topology.** What it is: peer-to-peer reduction with no central coordinator. Why
  considered: removes the Phase-1 single point of failure/trust. Why rejected for Phase 1: a coordinator
  simplifies the round state machine, secure-aggregation orchestration, and commitment, and is the
  natural proving target in Phase 2 ([RFC-0006](RFC-0006-verifiable-contribution.md)); gossip is
  reconsidered as a coordinator-failover path in [RFC-0013](RFC-0013-coordinator-runtime.md).
- **int8 pseudo-gradient quantization vs full precision.** What it is: quantize `Δ_c` to int8 before
  transport (INTELLECT-1's int8 all-reduce). Why considered: roughly 4× outer-step bandwidth reduction
  over fp32 with little quality loss in practice. Why offered as optional, not default: quantization
  error must be bounded and must not break `INV-AGG-DETERMINISM` on the dequantized sum; it is enabled
  by config once the round-trip error bound is validated (Testing Strategy), and it is orthogonal to the
  gauge.

## Drawbacks

- **Coordinator centralization (Phase 1).** The coordinator is a single point of failure and a single
  point of trust for orchestration and commitment. It is honest-but-curious in Phase 1; Phase 2 makes
  the outer step provable ([RFC-0006](RFC-0006-verifiable-contribution.md)) but does not remove the
  liveness dependency. Mitigation path: coordinator failover is an Open Question for
  [RFC-0013](RFC-0013-coordinator-runtime.md).
- **DiLoCo drift grows with `H`.** Longer inner horizons reduce communication but let participant frames
  rotate further apart before the outer step, directly raising the anchoring burden
  ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md)). The protocol does not eliminate this coupling;
  it exposes `H` as a control parameter to be characterized against the frame-drift diagnostic.
- **DP noise on small predictor deltas.** The predictor `g_φ` is compact; its delta is small, so a fixed
  noise multiplier `σ` degrades it more than the larger encoder delta. This interacts with SIGReg
  variance and the anchor term and is the joint-calibration risk above.

## Migration / Rollout

The protocol rolls out along the staged plan ([01 §Staged Plan](../spec/01-architecture.md),
[00 §v1.0 Scope Boundary](../spec/00-overview.md)):

- **v0.2 / Stage B — simulated federation on one cluster.** The full round lifecycle runs in-process
  (the simulation harness of [RFC-0013](RFC-0013-coordinator-runtime.md)): `C` simulated participants on
  a non-IID partition, DiLoCo + frame anchor (Layers 1–4), Procrustes backstop, simulated secure
  aggregation + DP. The `H` schedule starts small (`H ∈ [50, 500]`, smaller end first) while the
  frame-drift diagnostic is characterized, then is raised toward the communication-efficient regime once
  drift is shown controlled.
- **v0.3 / Stage C — two real sovereign nodes over a network boundary.** Real secure aggregation + DP,
  residency enforcement over the wire, fault tolerance/elasticity, and the contribution ledger. The
  message table (§8) becomes the on-the-wire control plane ([RFC-0013](RFC-0013-coordinator-runtime.md));
  int8 quantization may be enabled once its error bound is validated.

No data migration is required between stages: the contracts (`PseudoGradient`, the message table,
`GlobalState`) are stable from v0.2; Stage C swaps the transport and the secure-aggregation backend, not
the protocol semantics.

## Testing Strategy

Concrete, CPU-runnable tests on tiny synthetic fixtures (no large downloads; cf.
[07 §CI Gates](../spec/07-testing-strategy.md)):

- **Round state-machine transitions.** Drive a toy round through the lifecycle and assert each step's
  contract; assert the `ABORTED` path on quorum failure (detailed transitions in
  [RFC-0013 §Testing](RFC-0013-coordinator-runtime.md)).
- **`PseudoGradient` correctness.** Given known `(θ_t, φ_t)` and a deterministic toy local update,
  assert `Δ_c = (θ_c, φ_c) − (θ_t, φ_t)` exactly; assert the released delta covers only the encoder and
  predictor param groups and never an action-head group (`INV-ACTIONHEAD-LOCAL`).
- **DP clip bound (`INV-DP-BOUND`).** After clipping, assert `‖Δ_c‖ ≤ C_clip` for adversarial input
  norms above and below `C_clip`; assert clipping is deterministic.
- **Aggregation determinism (`INV-AGG-DETERMINISM`).** Run the outer step twice on identical inputs and
  assert bitwise-identical `(θ_{t+1}, φ_{t+1})`; corrupt the reduction order and assert the determinism
  self-check raises `NonDeterministicAggregation`.
- **Dropout-robustness simulation.** Simulate participant dropout above and below the secure-aggregation
  threshold; above-threshold completes, below-threshold raises `SecureAggregationError`; verify the
  Nesterov outer step is stable under a varying participant count.
- **int8 quantization round-trip error bound.** Quantize and dequantize a `Δ_c`; assert the L2 round-trip
  error is within the stated bound and that the dequantized sum still passes the determinism self-check.
- **Boundary-crossing redaction.** Assert no raw observation/action/private embedding appears in any
  `RoundOpen`/`Update`/`Commitment`/`RoundClose` payload (`INV-RESIDENCY`; cf.
  [05 §Redaction](../spec/05-observability.md)).
- **Commitment binding.** A delta released with a missing or wrong `R_c` is rejected with
  `CommitmentMismatch` ([RFC-0014](RFC-0014-provenance-commitments.md)).

The ablation-ladder rungs that exercise the protocol (naive FedAvg negative control vs anchored DiLoCo)
are realized as small-config integration tests per [07 §Ablation Ladder](../spec/07-testing-strategy.md)
and [RFC-0005 §6](RFC-0005-evaluation.md).

## Open Questions

OPEN QUESTION: The joint calibration of `(σ, λ_sig, λ_anc, C_clip)` — DP noise interacts with SIGReg
variance and the anchor term, and the sweet spot is not a default. Owner @AbdelStark; resolution: a
Stage-B (v0.2) sweep, shared with [RFC-0012 §Open Questions](RFC-0012-differential-privacy.md) and
[RFC-0002 §7](RFC-0002-gauge-and-aggregation.md).

OPEN QUESTION: The value of the inner horizon `H`. Start at the small end of `[50, 500]` while
characterizing drift, then raise toward the communication-efficient regime. Owner @AbdelStark;
resolution: Stage-B (v0.2) `H`-vs-drift characterization on the frame-drift diagnostic
([RFC-0005 §2](RFC-0005-evaluation.md), [08 §Communication Efficiency](../spec/08-performance-budget.md)).

OPEN QUESTION: The transport choice for the real network boundary (gRPC vs HTTP/REST) and the
coordinator failover story. Owner @AbdelStark; resolution: deferred to
[RFC-0013 §Open Questions](RFC-0013-coordinator-runtime.md), Stage C (v0.3).

RISK: The DiLoCo-drift × gauge coupling could be worse than anticipated at video-WM scale — a horizon
`H` large enough to be communication-efficient may rotate frames faster than a small `λ_anc` can pin
them, forcing the Layer-3 Procrustes backstop ([RFC-0002 §5](RFC-0002-gauge-and-aggregation.md)) to
bind frequently (raising per-round cost) or the Layer-4 distillation fallback
([RFC-0002 §6](RFC-0002-gauge-and-aggregation.md)) to engage. Resolution plan: the Stage-B `H`-vs-drift
characterization above is designed to surface this; the backstop and distillation layers are the
documented degrade, and Fork A ([RFC-0002 Fork A fallback](RFC-0002-gauge-and-aggregation.md#fork-a-fallback)) is the final safe-degrade
fallback if Fork-B gauge control is unstable at scale.

## References

- [RFC-0001 — Architecture & System Overview](RFC-0001-architecture.md) (federation map, two-level
  topology, trust boundaries, staged plan).
- [RFC-0002 — The Latent Gauge & Frame-Anchored Aggregation](RFC-0002-gauge-and-aggregation.md) (the
  aggregation semantics; §4 DiLoCo drift / frame anchoring; §5 Procrustes backstop; §8 per-round algorithm; Fork A fallback).
- [RFC-0011 — Secure Aggregation Protocol](RFC-0011-secure-aggregation.md) (pairwise masking / TEE,
  dropout robustness).
- [RFC-0012 — Differential Privacy Accounting](RFC-0012-differential-privacy.md) (the Gaussian mechanism,
  `(ε,δ)` accountant, privacy unit).
- [RFC-0013 — Coordinator & Participant Runtime](RFC-0013-coordinator-runtime.md) (runtime classes, round
  state machine, fault tolerance, control-plane transport).
- [RFC-0014 — Provenance Commitments & Merkle Scheme](RFC-0014-provenance-commitments.md) (the dataset
  Merkle root `R_c` and `Commitment` binding).
- [RFC-0006 — Verifiable Contribution](RFC-0006-verifiable-contribution.md) (the proof-ready disciplines:
  deterministic aggregation, committed model versions).
- [02 — Public API](../spec/02-public-api.md) · [03 — Data Model](../spec/03-data-model.md) ·
  [04 — Error Model](../spec/04-error-model.md) · [06 — Security](../spec/06-security.md) ·
  [08 — Performance Budget](../spec/08-performance-budget.md).
- External: DiLoCo / OpenDiLoCo / INTELLECT-1 / PRIME (inner/outer optimizer, int8 all-reduce, elastic
  fault tolerance); Bonawitz et al. practical secure aggregation; V-JEPA 2 (encoder warm-start).
