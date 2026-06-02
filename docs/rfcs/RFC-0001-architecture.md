# RFC-0001 — Architecture & System Overview

| | |
|---|---|
| **RFC** | 0001 |
| **Title** | Architecture & System Overview |
| **Slug** | architecture |
| **Status** | Accepted |
| **Track** | Standards |
| **Authors** | Abdelhamid Bakhta (@AbdelStark) |
| **Created** | 2026-06-02 |
| **Target milestone** | v0.1 (Stage A) |
| **Area** | core |
| **Requires** | — |
| **Informs** | RFC-0002, RFC-0003, RFC-0004, RFC-0005, RFC-0006, RFC-0007, RFC-0008, RFC-0009, RFC-0010, RFC-0011, RFC-0012, RFC-0013, RFC-0014, RFC-0015 |

## Summary

Lensemble trains a single **action-conditioned JEPA** world model **end-to-end** — encoder
$f_\theta$ and latent predictor $g_\phi$ co-trained, "Fork B" — across many mutually-distrusting
participants. Raw interaction data never leaves a participant's boundary; only model deltas
$\Delta_c$ cross, aggregated under privacy. This RFC specifies *what Lensemble builds*: the model
and its parts, which parts are federated versus local versus never-crossing, the two-level training
topology, the trust boundaries, the module map of the reference implementation, and the staged
rollout A–E mapped to milestones v0.1–v1.0. Mechanisms are deferred to focused RFCs: the latent
gauge to [RFC-0002](RFC-0002-gauge-and-aggregation.md), the wire protocol to
[RFC-0003](RFC-0003-federated-protocol.md), data and sovereignty to
[RFC-0004](RFC-0004-data-provenance.md), evaluation to [RFC-0005](RFC-0005-evaluation.md), and the
Phase-2 verifiable layer to [RFC-0006](RFC-0006-verifiable-contribution.md). This RFC holds the
rationale; the stable module-boundary contract lives in ([01-architecture.md](../spec/01-architecture.md)).

## Motivation

A foundation-scale world model wants diverse embodied experience — robot fleets, manipulation labs,
driving stacks, egocentric video — but that data is siloed by IP, privacy, and safety and cannot be
pooled. Federated training is the access strategy. The catch specific to JEPA: its self-supervised
objective is invariant under O(d) rotations of the latent space (the **latent gauge**, argued in
[RFC-0002 §2](RFC-0002-gauge-and-aggregation.md)), so independently-updated participants drift into
mutually-rotated coordinate frames and naive weight-averaging is meaningless — a failure mode that
anchored models (supervised nets, LLMs with a fixed vocabulary) never see.

The architecture must therefore make four things simultaneously true: (1) the shared backbone is
genuinely *one* model that averaging can combine; (2) the parts that legitimately differ per
participant (action spaces) stay local; (3) nothing that crosses a boundary leaks private data; and
(4) the aggregation path is disciplined enough that a Phase-2 proof of correct aggregation is cheap
to construct. No single existing system delivers all four for an end-to-end JEPA, so the
architecture is specified here as a coherent whole rather than assembled ad hoc.

## Goals

- Specify the model: encoder $f_\theta$, predictor $g_\phi$, per-embodiment action head
  $h_\psi^{(c)}$, the three-term objective, and latent-MPC planning, each typed against the WMCP
  latent contract ([RFC-0007](RFC-0007-wmcp-latent-contract.md)).
- Specify the **federation map**: for every component, whether it is federated, shared-per-round,
  local-personalized, or never-crosses, with the reason.
- Specify the **two-level training topology** — inner (intra-participant FSDP/TP, the only place
  large-model parallelism applies) and outer (inter-participant DiLoCo) — and make explicit that the
  standard distributed stack is the inner loop, not the contribution.
- Specify the **trust boundaries** and name exactly what crosses and what never does, enforcing
  `INV-RESIDENCY`.
- Specify the **module map** of the reference implementation ([conventions §1](../spec/conventions.md#1-repository-and-package-layout)) and its dependency layering,
  asserting no import cycles, as the stable reference for ([01-architecture.md](../spec/01-architecture.md)).
- Map the **staged plan A–E** to milestones v0.1/v0.2/v0.3/v1.0, with each stage gating the next.

## Non-Goals

- This RFC does not specify the gauge fix, the protocol wire format, secure aggregation, DP
  accounting, data formats, the eval suite, or the proof system; those are owned by their RFCs and
  only referenced here.
- Lensemble is not an LLM and not a frozen-encoder system by default (Fork A is the documented
  degrade, not the target). It is not an incentive or payment system; economic/on-chain mechanisms
  are out of scope (see [RFC-0006](RFC-0006-verifiable-contribution.md)).
- Stage E (own foundation-scale federated video pretraining from scratch) and the Stage-D
  realized proofs are out of v1.0 scope ([conventions §12](../spec/conventions.md#12-milestones-and-stages)); only the proof-*ready* disciplines are in scope.

## Proposed Design

### 1. Model

Lensemble trains an action-conditioned JEPA used as a latent world model for planning, following
the V-JEPA 2-AC shape but **co-training the encoder** (Fork B) rather than freezing it. Full model
and numerical contracts are in [RFC-0008](RFC-0008-model-objective-numerics.md); this section fixes
the responsibilities and types.

- **Encoder** $f_\theta:\text{video clip}\to\mathbb{R}^{N\times d}$ — a video Vision Transformer,
  **warm-started from released V-JEPA 2 weights**. Co-trained under SIGReg
  ([RFC-0002 §3](RFC-0002-gauge-and-aggregation.md)). The warm-start is also the gauge anchor at
  $t{=}0$ ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md)): the round-0 snapshot is $f_{\text{ref}}$.
  Its output is a WMCP `LatentState` of shape $(N, d)$ ([RFC-0007](RFC-0007-wmcp-latent-contract.md),
  `INV-WMCP`).
- **Latent predictor** $g_\phi$ — a compact transformer predicting future latents autoregressively,
  conditioned on an action embedding (the LeWM `ARPredictor` shape). It consumes and emits
  `LatentState`.
- **Action encoder / embodiment head** $h_\psi^{(c)}$ — **per-participant**, mapping that
  embodiment's action space into the shared latent-conditioning space. Never averaged or broadcast
  (`INV-ACTIONHEAD-LOCAL`), because action spaces genuinely differ: a quadruped is not a 7-DoF arm.
- **Objective** (per local step), stated exactly:
  $$\mathcal{L} = \lambda_{\text{pred}}\,\mathbb{E}\lVert g_\phi(f_\theta(x_t),a_t) - \text{sg}[f_\theta(x_{t+1})]\rVert^2 + \lambda_{\text{sig}}\,\mathrm{SIGReg}_A(f_\theta(x)) + \lambda_{\text{anc}}\,\mathcal{L}_{\text{anchor}}(f_\theta;\mathcal{P},\{t_i\}).$$
  Next-embedding prediction loss + SIGReg (collapse prevention during co-training,
  [RFC-0002 §3](RFC-0002-gauge-and-aggregation.md)) + the frame-anchor loss
  ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md)).
- **Planning / evaluation** — latent model-predictive control: CEM / iCEM / MPPI minimizing an
  $L_1$ goal-energy in latent space, exactly as `stable-worldmodel` provides
  ([RFC-0005](RFC-0005-evaluation.md)).

The shared latent interface — what every encoder emits and every predictor consumes — is the
**WMCP** contract ([RFC-0007](RFC-0007-wmcp-latent-contract.md)). It is what makes
heterogeneous-embodiment federation type-safe; it plays the role the fixed token vocabulary plays
for free in LLM federation. A nonconforming latent shape, dtype, or semantics raises
`ContractViolation` ([conventions §6](../spec/conventions.md#6-error-taxonomy)); an unvalidated `ActionSpec` blocks action-head construction
(`INV-WMCP`).

### 2. Module map (reference implementation)

The Python import root is `lensemble`. Each top-level module's responsibility, owning RFC, and
public/internal status (the full file tree is [conventions §1](../spec/conventions.md#1-repository-and-package-layout); the stable contract reference is
[01-architecture.md](../spec/01-architecture.md)):

| Module | Responsibility | Owning RFC | Surface |
|---|---|---|---|
| `errors` | Error taxonomy (`LensembleError` tree, `LensembleErrorCode`) | core | public |
| `config` | Frozen `LensembleConfig` tree, `RunManifest`, seeding | [RFC-0009](RFC-0009-configuration-reproducibility.md) | public |
| `contracts` | WMCP `LatentState`/`ActionSpec`, embodiment conformance | [RFC-0007](RFC-0007-wmcp-latent-contract.md) | public |
| `model` | `build_encoder`/`build_predictor`/`build_action_head`, `Objective`, SIGReg | [RFC-0008](RFC-0008-model-objective-numerics.md) | public |
| `gauge` | `frame_drift`, `procrustes_align`, anchor, drift | [RFC-0002](RFC-0002-gauge-and-aggregation.md) | public |
| `federation` | `Coordinator`, `Participant`, `RoundState`, outer optimizer | [RFC-0003](RFC-0003-federated-protocol.md), [RFC-0013](RFC-0013-coordinator-runtime.md) | public |
| `aggregation` | Secure aggregation (masking / TEE), deterministic summation | [RFC-0011](RFC-0011-secure-aggregation.md) | internal |
| `privacy` | DP clip+noise, $(\varepsilon,\delta)$ accountant | [RFC-0012](RFC-0012-differential-privacy.md) | internal |
| `data` | Dataset/loaders/adapters, residency guard, public probe | [RFC-0004](RFC-0004-data-provenance.md) | public |
| `provenance` | Episode hashing, Merkle, `ContributionLedger` | [RFC-0014](RFC-0014-provenance-commitments.md) | public |
| `eval` | Latent MPC `Planner`, `evaluate`, metrics | [RFC-0005](RFC-0005-evaluation.md) | public |
| `artifacts` | Checkpoint format, canonical hashing, schema versioning | [RFC-0010](RFC-0010-artifact-checkpoint-format.md) | public |
| `observability` | Structured logging, metrics emit, redaction guard | [RFC-0015](RFC-0015-observability-diagnostics.md) | public |
| `verify` | Phase-1 public recomputation; Phase-2 proof stubs | [RFC-0006](RFC-0006-verifiable-contribution.md) | public |
| `cli` | Typer CLI app | ([02-public-api.md](../spec/02-public-api.md)) | public |

Modules named `_internal` or prefixed `_` are private and unversioned ([conventions §10](../spec/conventions.md#10-versioning-and-schema-policy)). Public symbols are
re-exported from `lensemble/__init__.py` and frozen at 1.0 per ([09-release-and-versioning.md](../spec/09-release-and-versioning.md)).

### 3. Dependency layering (no cycles)

Allowed dependency direction, lowest layer first. The contract: a module may depend only on modules
at the same or a lower layer; back-edges are forbidden, so the import graph is a DAG.

```
  L0  errors                                  (depends on nothing)
  L1  config, observability                   (-> errors)
  L2  contracts                               (-> errors, config)
  L3  data, artifacts, provenance             (-> contracts, config, observability, errors)
  L4  model, gauge                            (-> contracts, artifacts, config, observability, errors)
  L5  aggregation, privacy                    (-> artifacts, provenance, config, observability, errors)
  L6  eval                                    (-> model, data, gauge, config, observability, errors)
  L7  federation                              (-> model, gauge, aggregation, privacy,
                                                  provenance, artifacts, data, config,
                                                  observability, errors)
  L8  verify                                  (-> federation, gauge, provenance, artifacts, ...)
  L9  cli                                     (-> everything public)
```

In prose: `errors` is foundational and depends on nothing. `config` and `observability` are
cross-cutting and may be imported by everything above them. `contracts` (WMCP) sits below the model
because both `model` and `gauge` are typed against it. `model` and `gauge` depend on `contracts`;
`gauge` additionally reads/writes `artifacts` for the reference encoder snapshot. `aggregation` and
`privacy` operate on serialized deltas and so depend on `artifacts` and `provenance`. `eval` depends
on `model` and `data`. `federation` is the integration layer and depends on `model`, `gauge`,
`aggregation`, `privacy`, `provenance`, `artifacts`, and `data`. `verify` sits on top of
`federation` (it recomputes alignment from committed artifacts and the public probe). Everything may
depend on `errors`, `config`, and `observability`. The absence of cycles is asserted by a
module-boundary import test (see Testing Strategy).

### 4. Federation map

For every component, whether it is federated, shared-per-round, local-personalized, or
never-crosses, and why. This table is the authoritative disposition; [RFC-0003](RFC-0003-federated-protocol.md)
and the spec at ([01-architecture.md](../spec/01-architecture.md)) reproduce it.

| Component | Disposition | Why | Invariant |
|---|---|---|---|
| Encoder backbone $f_\theta$ | **Federated** (gauge-controlled) | Shared physics; the point of the project | `INV-WARMSTART-T0` |
| Predictor core $g_\phi$ | **Federated** | Shared dynamics; frame-pinned so averaging is valid | — |
| Action encoder / heads $h_\psi^{(c)}$ | **Local — personalized** | Embodiment-specific action spaces | `INV-ACTIONHEAD-LOCAL` |
| SIGReg sketch matrix $A$ | **Shared per round** (broadcast seed $s_t$) | Objective consistency ([RFC-0002 §3](RFC-0002-gauge-and-aggregation.md)) | `INV-SKETCH-CONSISTENCY` |
| Public probe $\mathcal{P}$ + landmark targets $\{t_i\}$ | **Shared, fixed, hash-pinned** | The manufactured frame anchor ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md)) | `INV-PROBE-PIN` |
| Dataset Merkle root $R_c$ | **Crosses as a commitment** (not the data) | Provenance binding ([RFC-0014](RFC-0014-provenance-commitments.md)) | `INV-COMMIT-BINDING` |
| Pseudo-gradient $\Delta_c$ | **Crosses, DP-clipped + masked** | The only learning signal that leaves a boundary | `INV-DP-BOUND` |
| Raw trajectories (obs/actions) | **Never leaves the boundary** | Sovereignty | `INV-RESIDENCY` |
| Private embeddings $f_\theta(x)$ | **Never leaves the boundary** | Sovereignty | `INV-RESIDENCY` |

`INV-WARMSTART-T0` (every participant's round-0 encoder weights are hash-identical to the pinned
warm-start) closes the gauge at $t{=}0$ and is enforced by the federation handshake
([RFC-0003 §3](RFC-0003-federated-protocol.md)); a mismatch raises `GaugeError`.
`INV-SKETCH-CONSISTENCY` (all participants in round $t$ use the identical $A$ derived from $s_t$) is
enforced where the sketch is built in `lensemble.model.sigreg`. `INV-ACTIONHEAD-LOCAL` is enforced
in `lensemble.federation`: the broadcast and aggregation paths exclude $h_\psi^{(c)}$ parameters by
construction.

### 5. Training topology (two-level)

A two-level nesting; this is where the standard "distributed training" stack is the *inner* loop,
not the contribution.

- **Inner — intra-participant, for scale.** Within a participant, standard FSDP / tensor / context
  parallelism trains the warm-started 1.2B-class model. SIGReg projection statistics are reduced
  *within* this trust domain freely ([RFC-0002 §3](RFC-0002-gauge-and-aggregation.md); the
  reduce-within-trust-domain rule). This is the only place the large-model-parallelism playbook
  (INTELLECT-1 / PRIME) applies. Inner optimizer: AdamW. Inner determinism is best-effort,
  seed-pinned, and gated by a config flag ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)).
- **Outer — inter-participant, for sovereignty.** DiLoCo: each participant runs $H$ local steps,
  then an outer **Nesterov** step synchronizes pseudo-gradients
  $\Delta_c = (\theta_c^{\text{local}},\phi_c^{\text{local}}) - (\theta_t,\phi_t)$
  ([RFC-0003 §3](RFC-0003-federated-protocol.md)). Only $\Delta_c$ crosses the boundary, via secure
  aggregation ([RFC-0011](RFC-0011-secure-aggregation.md)) + DP
  ([RFC-0012](RFC-0012-differential-privacy.md)). The outer step is bitwise-deterministic given its
  inputs (`INV-AGG-DETERMINISM`, [conventions §9](../spec/conventions.md#9-determinism-dtype-device)): fixed reduction order, fp32/fp64 with fixed summation
  order, no atomics. A determinism self-check runs each outer step; failure raises
  `NonDeterministicAggregation` and aborts the step (see Testing Strategy and
  [RFC-0006 §3](RFC-0006-verifiable-contribution.md) for proof-readiness).

### 6. Trust boundaries

```
┌── Participant c (sovereign) ─────────────────────────┐
│  raw trajectories  ──►  local train (inner-parallel) │
│        │ (never leaves)         │                     │
│        ▼                        ▼                     │
│  Merkle commitment R_c     pseudo-gradient Δ_c        │
└───────────────│──────────────────│───────────────────┘
                │                  │  (DP-clipped + noised)
                ▼                  ▼
        ┌──────────── Coordinator / secure aggregator ─────────┐
        │  Σ_c Δ_c  (individual Δ_c never revealed)             │
        │  outer Nesterov step → θ^{global}_{t+1} (hash-committed)│
        │  frame re-alignment on public probe (recomputable)    │
        └───────────────────────────────────────────────────────┘
```

In prose: each participant is a sovereign trust domain. Raw trajectories and any private embedding
$f_\theta(x)$ stay inside it and never cross (`INV-RESIDENCY`, enforced by `lensemble.data.residency`;
an attempted crossing raises `ResidencyViolation`, which is fail-closed and never caught-and-ignored,
[conventions §6](../spec/conventions.md#6-error-taxonomy)). What crosses a boundary: model deltas $\Delta_c$ (privacy-protected, DP-clipped then
masked), dataset commitments $R_c$, and shared coordination state (the sketch seed $s_t$, the probe
hash, the global-model hash). The coordinator/aggregator learns only $\sum_c \Delta_c$, never an
individual $\Delta_c$ ([RFC-0011](RFC-0011-secure-aggregation.md)); for Phase 1 it is treated as
honest-but-curious and as a proving target in Phase 2
([RFC-0006](RFC-0006-verifiable-contribution.md)). The threat model in detail is at
([06-security.md](../spec/06-security.md)). The boundary-crossing message table is
[RFC-0003 §8](RFC-0003-federated-protocol.md).

### 7. Data-flow lifecycles

**(a) A federated training round, end-to-end** (per round $t$;
[RFC-0003 §3](RFC-0003-federated-protocol.md), runtime in
[RFC-0013](RFC-0013-coordinator-runtime.md)):

1. Coordinator emits `RoundOpen` broadcasting $(\theta_t,\phi_t)$ refs/hashes, the sketch seed $s_t$,
   the probe hash, and $H$. Action heads $h_\psi^{(c)}$ are not broadcast (`INV-ACTIONHEAD-LOCAL`).
2. Each participant validates the probe hash equals the committed hash (`INV-PROBE-PIN`; else
   `ProbeError`) and builds $A$ from $s_t$ (`INV-SKETCH-CONSISTENCY`).
3. Each participant runs $H$ inner AdamW steps on the objective over local data only.
4. Pseudo-gradient $\Delta_c = (\theta_c^{\text{local}},\phi_c^{\text{local}}) - (\theta_t,\phi_t)$.
5. Privatize: clip to $C_{\text{clip}}$ then add Gaussian noise (`INV-DP-BOUND`;
   [RFC-0012](RFC-0012-differential-privacy.md)). Budget exhaustion raises `PrivacyBudgetExceeded`
   and stops training.
6. Secure-aggregate: compute $\sum_c \Delta_c$ without revealing any individual $\Delta_c$
   ([RFC-0011](RFC-0011-secure-aggregation.md)); dropout below threshold raises
   `SecureAggregationError`.
7. Backstop align: Procrustes re-alignment on the public probe if drift exceeds threshold
   ([RFC-0002 §5](RFC-0002-gauge-and-aggregation.md); `FrameDriftExceeded` triggers the backstop,
   `DegenerateProcrustes` on a degenerate SVD).
8. Outer Nesterov step:
   $(\theta_{t+1},\phi_{t+1}) = (\theta_t,\phi_t) - \eta_{\text{out}}\,\mathrm{Nesterov}\big(\tfrac1C\sum_c\Delta_c\big)$;
   the determinism self-check guards `INV-AGG-DETERMINISM`.
9. Commit: hash-commit $(\theta_{t+1},\phi_{t+1})$ (`INV-CHECKPOINT-HASH`;
   [RFC-0010](RFC-0010-artifact-checkpoint-format.md)) and emit `RoundClose`. Each released
   $\Delta_c$ is bound to exactly one $R_c$ (`INV-COMMIT-BINDING`;
   [RFC-0014](RFC-0014-provenance-commitments.md)); a mismatch raises `CommitmentMismatch` and the
   update is rejected.

**(b) Single-site Stage-A training** (`train_local(config) -> RunResult`): the inner loop run
standalone — warm-start $f_\theta$, build $g_\phi$ and a single $h_\psi$, run the objective to
convergence on pooled data, evaluate via latent MPC. No outer loop, no boundaries, no DP/secure-agg.
This is the centralized upper bound ([RFC-0005](RFC-0005-evaluation.md)).

**(c) Evaluation (latent MPC)** (`evaluate(checkpoint, env_id, *, cfg) -> EvalReport`): load a
hash-verified checkpoint (`INV-CHECKPOINT-HASH`; `CheckpointIntegrityError` on tamper), encode the
goal and current observation, run CEM/iCEM/MPPI to minimize $L_1$ goal-energy in latent space,
execute, emit metrics (`eval/success_rate`, planning samples, time per action; [RFC-0015](RFC-0015-observability-diagnostics.md)).

**(d) Dataset commit** (`commit_dataset(dataset) -> DatasetCommitment`): canonically serialize each
episode, hash to domain-separated SHA-256 leaves, build a Merkle tree, produce the root $R_c$ with
episode count and WMCP metadata ([RFC-0014](RFC-0014-provenance-commitments.md)). Raw episodes are
never emitted (`INV-RESIDENCY`); only $R_c$ leaves the boundary.

### 8. Process / concurrency model

- **Coordinator process** — one per federation: owns the canonical global model, drives the round
  state machine ([RFC-0013](RFC-0013-coordinator-runtime.md)), runs the outer optimizer, and is the
  hash-commitment authority.
- **Participant processes** — one per sovereign node: own the local data, run the inner loop, emit a
  `PseudoGradient`. They never share an address space with each other or with the coordinator across
  a real boundary (Stage C); Stage B simulates them in one process for the experiments.
- **Inner parallel workers** — within a participant, FSDP/TP ranks for the large model. Determinism
  is best-effort and seed-pinned inside; only the outer step must be bitwise-deterministic.
- Concurrency on the outer loop is round-synchronous with elastic membership: an outer step proceeds
  with whatever participants are present, late/dropped participants reconcile next round, and too few
  remaining raises `FaultToleranceExceeded` ([RFC-0013](RFC-0013-coordinator-runtime.md)). Timeouts
  and backpressure are runtime concerns specified there.

### 9. Failure modes handled by the architecture

| Failure | Where detected | Error ([conventions §6](../spec/conventions.md#6-error-taxonomy)) | Architectural response |
|---|---|---|---|
| Raw data/embedding about to cross a boundary | `data.residency` | `ResidencyViolation` | Fail-closed; never caught-and-ignored |
| Round-0 encoder not equal to warm-start | federation handshake | `GaugeError` | Reject join (`INV-WARMSTART-T0`) |
| Frame drift exceeds threshold | `gauge.drift` | `FrameDriftExceeded` | Procrustes backstop ([RFC-0002 §5](RFC-0002-gauge-and-aggregation.md)) |
| Probe hash differs from committed | participant ingress | `ProbeError` | Reject round (`INV-PROBE-PIN`) |
| Outer step nondeterministic | outer self-check | `NonDeterministicAggregation` | Abort + recompute (never swallowed) |
| Update bound to wrong root | commitment check | `CommitmentMismatch` | Reject update (never swallowed) |
| DP budget exhausted | `privacy.accountant` | `PrivacyBudgetExceeded` | Stop training |
| Checkpoint tampered | artifact load | `CheckpointIntegrityError` | Refuse load (`INV-CHECKPOINT-HASH`) |

## Alternatives Considered

**Fork B (end-to-end, co-trained encoder) vs Fork A (frozen shared encoder, federate predictor
only).** Fork B co-trains $f_\theta$ and $g_\phi$; Fork A freezes a shared encoder and federates
only $g_\phi$. Fork B is considered the lead because the open scientific question — can an
end-to-end JEPA be federated at all, given the latent gauge — only exists when the encoder moves;
freezing it dissolves the gauge and the contribution with it
([RFC-0002 §7](RFC-0002-gauge-and-aggregation.md)). Fork A is *not* rejected: it is the documented
safe-degrade. If Fork-B gauge control proves unstable at scale, Fork A recovers a clean federation
and most of the sovereignty story, minus the end-to-end novelty. Both paths are supported and tested
at v1.0 ([conventions §12](../spec/conventions.md#12-milestones-and-stages)).

**Two-level topology (inner FSDP + outer DiLoCo) vs flat synchronous all-reduce.** A flat all-reduce
synchronizes every step across all participants. Considered for its simplicity and exact-averaging
semantics; rejected for the outer loop because per-step cross-boundary communication is infeasible at
foundation scale and incompatible with sovereign nodes on slow/unreliable links. DiLoCo communicates
every $H$ steps ([RFC-0003 §3](RFC-0003-federated-protocol.md)). All-reduce remains the *inner*-loop
mechanism, where it belongs.

**Warm-start from released V-JEPA 2 vs train-from-scratch.** Training the encoder from scratch was
considered for independence from a third-party release. Rejected for v0.1–v1.0: it incurs an
INTELLECT-class compute bill (Stage E, out of scope, [conventions §12](../spec/conventions.md#12-milestones-and-stages)) and, decisively, forfeits the
$t{=}0$ frame anchor that warm-start provides for free — `INV-WARMSTART-T0` makes the gauge closed at
round 0 only because every participant starts from byte-identical weights.

**Monolithic vs per-embodiment action heads.** A single shared action head was considered for
uniformity. Rejected because embodiments have genuinely different action spaces (a quadruped is not a
7-DoF arm); a shared head would either lose information or force a lowest-common-denominator action
representation. Per-embodiment heads $h_\psi^{(c)}$ stay local and are never aggregated
(`INV-ACTIONHEAD-LOCAL`); the shared WMCP latent-conditioning space
([RFC-0007](RFC-0007-wmcp-latent-contract.md)) is what they map into.

## Drawbacks

- **Dependence on a released warm-start.** The whole A–D path assumes a usable V-JEPA 2 release; its
  availability and license bound the project, and its representation choices seed the frame the
  anchor pins to.
- **Anchor strength is a central tuning knob.** `λ_anc` ([RFC-0002 §7](RFC-0002-gauge-and-aggregation.md))
  trades frame stability against quality: too strong clamps quality to the reference encoder, too
  weak lets the frame drift. One scalar carries a disproportionate share of the system's behavior.
- **Coordinator centralization in Phase 1.** The coordinator is a single point of failure and trust;
  Phase 2 makes correct aggregation provable ([RFC-0006](RFC-0006-verifiable-contribution.md)) but
  does not remove the central role in v1.0.

## Migration / Rollout

The staged plan A–E is the rollout; each stage gates the next, mapped to milestones ([conventions §12](../spec/conventions.md#12-milestones-and-stages)).

| Stage | Milestone | Goal | Compute |
|---|---|---|---|
| **A** | v0.1 | Single-site, warm-started, ViT-L/~300M end-to-end SIGReg + AC predictor on pooled robot data. Centralized upper bound; validate objective + MPC eval. Plus foundational scaffolding (package, config, data layer, WMCP, model+objective, eval harness, observability, artifacts, errors, CI, packaging). | Handful of GPUs, days |
| **B** | v0.2 | **Simulated federation** on one cluster: $C$ silos, non-IID partition, DiLoCo + frame anchor (Layers 1–4), Procrustes backstop, simulated secure aggregation + DP, the frame-drift diagnostic, the full ablation ladder and non-IID/scale sweeps. *The scientific core / the paper.* | Same hardware |
| **C** | v0.3 | **Two real sovereign nodes** over a network boundary: real secure aggregation + DP, residency enforcement, fault tolerance/elasticity, the contribution ledger. The sovereignty demonstration. | Two small clusters |
| **D** | (post-v1.0, Phase 2) | The realized **verifiable layer** ([RFC-0006](RFC-0006-verifiable-contribution.md)): aggregation STARK + provenance binding + TEE inner step. Out of v1.0 scope; only the proof-*ready* disciplines land earlier. | + prover |
| **E** | (post-v1.0) | **Scale** to V-JEPA-2 class (1.2B); optionally federated encoder pretraining from scratch. Out of v1.0 scope. | INTELLECT-class program |

v1.0 itself is the hardening milestone: frozen public API, complete docs + reproducibility package,
release automation, the Fork A fallback supported and tested, and the proof-ready guarantees
([RFC-0006 §3](RFC-0006-verifiable-contribution.md)) verified end-to-end ([conventions §12](../spec/conventions.md#12-milestones-and-stages)). Warm-starting
keeps A–C modest and runnable; Stage E is the expensive frontier, off the Phase-1 / paper critical
path. A stage may not begin until its predecessor's milestone exit criteria
([RFC-0005 §9](RFC-0005-evaluation.md)) are met.

## Testing Strategy

The full pyramid is at ([07-testing-strategy.md](../spec/07-testing-strategy.md)); the architecture-owned tests are:

- **Round-lifecycle integration test.** Wire the full round of §7(a) end-to-end on a toy CPU config
  (a few synthetic silos, a tiny model, a tiny probe): `RoundOpen` → local steps → clip+noise →
  simulated secure-agg → optional Procrustes backstop → deterministic outer step → hash commit →
  `RoundClose`. Asserts the global hash advances and that two identical-seed runs produce identical
  `RunManifest` aggregation hashes (`INV-AGG-DETERMINISM`). Runs on CPU ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)).
- **Module-boundary import tests.** A test that imports each module in isolation and asserts the
  dependency DAG of §3 has no cycles (for example via an import-graph check), so the layering is
  enforced mechanically rather than by convention.
- **Federation-map enforcement test.** Assert that the broadcast and aggregation payloads never
  include $h_\psi^{(c)}$ parameters (`INV-ACTIONHEAD-LOCAL`) and that round-0 encoder weights are
  hash-identical to the pinned warm-start (`INV-WARMSTART-T0`).
- **Residency guard test (security-critical).** Assert that attempting to serialize a raw
  observation/action/private embedding into an outbound message raises `ResidencyViolation` and is
  never caught-and-ignored (`INV-RESIDENCY`; cross-referenced from
  [RFC-0004](RFC-0004-data-provenance.md) and [06-security.md](../spec/06-security.md)).

CI runs these on the CPU fallback with tiny synthetic fixtures; no large downloads
([07-testing-strategy.md](../spec/07-testing-strategy.md)).

## Open Questions

OPEN QUESTION: The **personalization boundary** — how heterogeneous can participants' data and
embodiments be before a single global encoder $f_\theta$ stops being learnable and the
shared-backbone + per-embodiment-head split breaks down? The split is the hedge, but the threshold is
empirical. Owner @AbdelStark; resolution path: the non-IID severity sweep in Stage B
([RFC-0005 §7](RFC-0005-evaluation.md)), reporting the centralized−local gap recovered as a function
of heterogeneity.

RISK: **SIGReg at video-WM scale** is demonstrated only to ViT-H on images; co-training the encoder
at video scale is unproven and is precisely the regime that opens the gauge. Resolution plan: Stage A
(v0.1) de-risks the objective + MPC eval centrally before any federation
([RFC-0008](RFC-0008-model-objective-numerics.md)); if it fails to converge at ViT-L scale, fall
back to Fork A ([RFC-0002 §7](RFC-0002-gauge-and-aggregation.md)) and continue the sovereignty story
without the end-to-end claim.

RISK: **Anchor-strength instability.** `λ_anc` may have no setting that holds the frame without
clamping quality at video scale. Resolution plan: the `λ_anc` sweep in Stage B
([RFC-0002 §7](RFC-0002-gauge-and-aggregation.md)) produces drift-vs-quality curves; the Layer-3
Procrustes backstop ([RFC-0002 §5](RFC-0002-gauge-and-aggregation.md)) bounds the worst case if no
single $\lambda_{\text{anc}}$ suffices.

## References

- README (project thesis, contribution, ecosystem positioning, license stanza).
- [RFC-0002 — The Latent Gauge & Frame-Anchored Aggregation](RFC-0002-gauge-and-aggregation.md): the gauge, the anchor, Procrustes backstop, Fork A degrade.
- [RFC-0003 — Federated Training Protocol](RFC-0003-federated-protocol.md): round structure, DP, secure-agg pointer, message table, reference parameters.
- [RFC-0004 — Data, Sovereignty & Provenance](RFC-0004-data-provenance.md): data layer, public probe, residency.
- [RFC-0005 — Evaluation & Benchmark Protocol](RFC-0005-evaluation.md): claims, frame-drift diagnostic, ablation ladder, baselines, success criteria.
- [RFC-0006 — Verifiable Contribution](RFC-0006-verifiable-contribution.md): Phase-2 layer; the proof-ready requirements Phase 1 must satisfy.
- [RFC-0007 — WMCP Latent Contract & Embodiment Adapters](RFC-0007-wmcp-latent-contract.md): the shared latent interface.
- [RFC-0008 — Model, Objective & Numerical Contracts](RFC-0008-model-objective-numerics.md): encoder/predictor/objective and numerics.
- [RFC-0013 — Coordinator & Participant Runtime](RFC-0013-coordinator-runtime.md): the round state machine and fault tolerance.
- [RFC-0014 — Provenance Commitments & Merkle Scheme](RFC-0014-provenance-commitments.md): episode hashing, Merkle, ledger.
- Spec: ([00-overview.md](../spec/00-overview.md)), ([01-architecture.md](../spec/01-architecture.md)), ([02-public-api.md](../spec/02-public-api.md)), ([06-security.md](../spec/06-security.md)), ([07-testing-strategy.md](../spec/07-testing-strategy.md)), ([09-release-and-versioning.md](../spec/09-release-and-versioning.md)).
- V-JEPA 2 (Assran et al., 2025) — encoder warm-start, AC recipe, the $t{=}0$ frame anchor.
- LeJEPA / LeWorldModel (Balestriero & LeCun; Maes, Le Lidec et al., 2026) — SIGReg objective.
- DiLoCo / OpenDiLoCo / INTELLECT (Douillard et al.; Prime Intellect) — inner/outer optimizer, elastic fault tolerance, int8 all-reduce.
- stable-worldmodel (galilai-group) — data layer (`lance`/`hdf5`/`lerobot`), envs, latent-MPC eval.
- WMCP (WM-RFC-0001) — the latent/action contract.
- Project Tapestry (AI Alliance) — the sovereignty/governance framing for federated frontier models.
- Stwo — Circle-STARK prover for the Phase-2 aggregation-correctness proof.
- [RFC-0016 — Deployment, Vendoring & Topology](RFC-0016-deployment-vendoring-topology.md) — the
  Python-first stack decision, the `third_party/` vendoring of the reused ecosystem code, and the
  one-config-source deployment model (in-process simulation → Docker Compose → Kubernetes).
