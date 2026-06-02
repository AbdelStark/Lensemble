# 01 — Architecture

This section is the stable architectural reference for Lensemble: the module map, the dependency
layering, what is federated versus local, the two-level training topology, the trust boundaries, the
end-to-end data flows, the process model, and the staged rollout. It states *what exists and how it
fits together*. Rationale for each design choice lives in the RFCs cited inline; this document does not
restate rationale, it pins the contract.

Lensemble trains a single action-conditioned JEPA world model end-to-end (encoder $f_\theta$ AND
predictor $g_\phi$ co-trained — "Fork B") across many mutually-distrusting participants. Raw data never
leaves a participant boundary; only model deltas cross, aggregated under privacy. The scientific core
is the latent-gauge problem and its fix (the SIGReg-JEPA objective is invariant under $O(d)$ rotations
of the latent space, so independently-updated participants drift into mutually-rotated frames and naive
weight-averaging is meaningless). See [00 — Overview](00-overview.md) for the thesis and contribution,
and [RFC-0001 — Architecture & System Overview](../rfcs/RFC-0001-architecture.md) for the originating
rationale.

## 1. Module map

The Python import root is `lensemble`. Every top-level module owns one subsystem, is specified by
exactly one RFC, and has a defined public/internal split. The layout below is canonical ([conventions §1](conventions.md#1-repository-and-package-layout)); the
reference implementation MUST NOT introduce additional top-level modules without a new RFC. Anything
under a module named `_internal` or prefixed with `_` is private and unversioned (see
[02 — Public API §2](02-public-api.md) for the stability policy).

| Module | Responsibility | Specified by | Surface |
|---|---|---|---|
| `lensemble/__init__.py` | Public re-exports and `__version__` (SemVer). | [conventions §5](conventions.md#5-public-api-surface) | public |
| `errors.py` | Error taxonomy: `LensembleError` base, `LensembleErrorCode` enum, every typed error carrying `.code` and `.remediation`. | [04 — Error Model](04-error-model.md) | public |
| `cli.py` | Typer CLI app (`train`, `federate`, `eval`, `probe`, `commit`, `drift`, `verify`, `doctor`); every command emits a `RunManifest`. | [02 — Public API §3](02-public-api.md) | public |
| `contracts/` | WMCP latent contract (`LatentState`, `ActionSpec`) and embodiment conformance checks. | [RFC-0007](../rfcs/RFC-0007-wmcp-latent-contract.md) | public |
| `model/` | Encoder $f_\theta$, predictor $g_\phi$, per-embodiment action heads $h_\psi^{(c)}$, the three-term `Objective`, the SIGReg statistic. Files: `encoder.py`, `predictor.py`, `action_head.py`, `objective.py`, `sigreg.py`. | [RFC-0008](../rfcs/RFC-0008-model-objective-numerics.md) | public |
| `gauge/` | Frame anchoring, Procrustes alignment, frame-drift diagnostics. Files: `anchor.py`, `procrustes.py`, `drift.py`. | [RFC-0002](../rfcs/RFC-0002-gauge-and-aggregation.md) | public |
| `federation/` | DiLoCo outer loop, round state machine, roles. Files: `coordinator.py`, `participant.py`, `round.py`, `outer_optimizer.py`. | [RFC-0003](../rfcs/RFC-0003-federated-protocol.md), [RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md) | public |
| `aggregation/` | Secure aggregation (pairwise masking / TEE) and the deterministic summation. Files: `secure_agg.py`, `masking.py`. | [RFC-0011](../rfcs/RFC-0011-secure-aggregation.md) | internal |
| `privacy/` | DP clip+noise mechanism and the $(\varepsilon,\delta)$ accountant. Files: `dp.py`, `accountant.py`. | [RFC-0012](../rfcs/RFC-0012-differential-privacy.md) | public (accountant), internal (mechanism) |
| `data/` | Data layer, loaders, embodiment adapters, residency enforcement, public probe. Files: `dataset.py`, `loaders.py`, `adapters/`, `residency.py`, `probe.py`. | [RFC-0004](../rfcs/RFC-0004-data-provenance.md) | public |
| `provenance/` | Episode hashing, Merkle tree, contribution ledger. Files: `merkle.py`, `commit.py`, `ledger.py`. | [RFC-0014](../rfcs/RFC-0014-provenance-commitments.md) | public |
| `eval/` | Latent MPC planner, eval harness, metrics. Files: `mpc.py`, `harness.py`, `metrics.py`. | [RFC-0005](../rfcs/RFC-0005-evaluation.md) | public |
| `config/` | Structured config schema, run manifest, seeding. Files: `schema.py`, `manifest.py`, `seed.py`. | [RFC-0009](../rfcs/RFC-0009-configuration-reproducibility.md) | public |
| `artifacts/` | Checkpoint/artifact format, hashing, schema versioning. Files: `checkpoint.py`, `schema.py`, `hashing.py`. | [RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md) | public |
| `observability/` | Structured logging, metric emission, redaction guard. Files: `logging.py`, `metrics.py`, `redaction.py`. | [RFC-0015](../rfcs/RFC-0015-observability-diagnostics.md) | public |
| `verify/` | Phase-2 verifiable layer + Phase-1 public recomputation. Files: `recompute.py`, `stark.py`. | [RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md) | public (`recompute`), stub (`stark`) |

`tests/` holds the pytest suite (unit, property, integration, ml); `docs/` holds this corpus;
`configs/` holds the Hydra config groups. These are not import modules.

The model split is fixed ([conventions §2](conventions.md#2-mathematical-notation), [RFC-0008 — Proposed Design](../rfcs/RFC-0008-model-objective-numerics.md#proposed-design)):

- Encoder $f_\theta:\text{video clip}\to\mathbb{R}^{N\times d}$ — a video Vision Transformer warm-started
  from released V-JEPA 2 weights, co-trained under SIGReg. The round-0 snapshot is $f_{\text{ref}}$,
  frozen and used as the gauge anchor (`INV-WARMSTART-T0`).
- Predictor $g_\phi$ — a compact transformer predicting future latents autoregressively, conditioned on
  an action embedding (the LeWM `ARPredictor` shape).
- Action head $h_\psi^{(c)}$ — per-participant, mapping that embodiment's action space into the shared
  latent-conditioning space. Never broadcast or aggregated (`INV-ACTIONHEAD-LOCAL`).

The shared latent interface that every encoder emits and every predictor consumes is the WMCP
`LatentState` contract ([RFC-0007](../rfcs/RFC-0007-wmcp-latent-contract.md)). It makes
heterogeneous-embodiment federation type-safe — the explicit analogue of the fixed token vocabulary
that LLM federation gets for free.

## 2. Dependency layering

The modules form an acyclic dependency graph in four bands. A lower band never imports a higher one.
`core` here means the always-importable foundation: `errors.py`, `config/`, and `observability/` (these
carry no domain dependencies and may be imported by anything). The acyclicity is enforced by a
module-boundary import test (see [07 — Testing Strategy](07-testing-strategy.md) and
[RFC-0001 — Testing Strategy](../rfcs/RFC-0001-architecture.md#testing-strategy)).

```
Band 0  core            errors.py   config/   observability/
                            ▲          ▲           ▲
Band 1  contracts        contracts/  artifacts/  provenance/   privacy/
        + leaf io           ▲          ▲           ▲              ▲
Band 2  model/gauge      model/  ─────┘            │              │
        + aggregation    gauge/  ─────┘            │              │
                         aggregation/ ─────────────┘              │
                            ▲                                     │
Band 3  orchestration    federation/ ── imports ──► model, gauge, aggregation,
                            ▲                        privacy, provenance, artifacts
                         eval/ ── imports ──► model, data
                            ▲
                         cli.py / verify/ ── top-level entry points
```

In prose, the allowed dependency edges are:

- `model/` and `gauge/` depend on `contracts/` (they produce and align `LatentState` instances).
- `aggregation/` depends on `contracts/` and `artifacts/` only for the delta and commitment types it
  transports; it depends on `privacy/` for the clip+noise step ordering ([RFC-0011](../rfcs/RFC-0011-secure-aggregation.md)).
- `federation/` (the orchestrator) depends on `model/`, `gauge/`, `aggregation/`, `privacy/`,
  `provenance/`, and `artifacts/`. It is the only module that composes the full round.
- `eval/` depends on `model/` (to load and run the world model) and `data/` (to source clips and the
  probe); it does NOT depend on `federation/`.
- `data/` depends on `contracts/` (loaders emit WMCP-conformant windows) and `provenance/` (commit on
  ingest).
- `verify/` depends on `gauge/`, `provenance/`, and `artifacts/` for public recomputation; the
  Phase-2 `stark.py` path is a stub raising `NotImplementedError`.
- Everything may depend on Band 0: `errors.py`, `config/`, `observability/`.

There are no back-edges: `contracts/` imports nothing from `model/`, `data/`, or `federation/`;
`model/` imports nothing from `federation/` or `eval/`. The orchestration band is the apex. The
boundary-import test imports each module in isolation and asserts no cycle and no upward edge; a
violation is a build failure, not a runtime error.

## 3. Federation map

Which components are federated, local, shared-per-round, or never-crossing is fixed by the model split
and the gauge mechanism. This table expands [RFC-0001 §4](../rfcs/RFC-0001-architecture.md#4-federation-map) and is the
authoritative disposition; the invariants named are enforced as cited.

| Component | Disposition | Enforced by / invariant | Why |
|---|---|---|---|
| Encoder backbone $f_\theta$ | Federated (gauge-controlled) | `INV-WARMSTART-T0` at $t{=}0$; frame anchor each round | Shared physics; the point of the project |
| Predictor core $g_\phi$ | Federated | aggregated alongside $f_\theta$ in $\Delta_c$ | Shared dynamics; frame-pinned so averaging is valid |
| Action head $h_\psi^{(c)}$ | Local — personalized | `INV-ACTIONHEAD-LOCAL` ([RFC-0007](../rfcs/RFC-0007-wmcp-latent-contract.md)) | Embodiment action spaces differ (quadruped ≠ 7-DoF arm) |
| SIGReg sketch matrix $A$ | Shared per round (broadcast seed $s_t$) | `INV-SKETCH-CONSISTENCY` ([RFC-0002 §3](../rfcs/RFC-0002-gauge-and-aggregation.md#3-layer-1--shared-sketch-matrix-objective-consistency-not-the-gauge-fix)) | Objective consistency across silos in a round |
| Public probe $\mathcal{P}$ + landmarks $\{t_i\}$ | Shared, fixed, hash-pinned | `INV-PROBE-PIN` ([RFC-0004](../rfcs/RFC-0004-data-provenance.md)) | The manufactured frame anchor |
| Reference encoder $f_{\text{ref}}$ | Shared, frozen | `INV-WARMSTART-T0` | Defines round-0 frame; landmark targets $t_i=f_{\text{ref}}(p_i)$ |
| Pseudo-gradient $\Delta_c$ | Crosses boundary (DP + secure-agg) | `INV-DP-BOUND`, `INV-COMMIT-BINDING` | The only learning signal that leaves a participant |
| Dataset Merkle root $R_c$ | Crosses boundary | `INV-COMMIT-BINDING` ([RFC-0014](../rfcs/RFC-0014-provenance-commitments.md)) | Binds a contribution to its data, not the data itself |
| Global params $(\theta_t,\phi_t)$ hash | Shared coordination state | `INV-CHECKPOINT-HASH` ([RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md)) | Reproducible, committed round state |
| Raw trajectories (obs/actions) | Never leaves the boundary | `INV-RESIDENCY` ([RFC-0004](../rfcs/RFC-0004-data-provenance.md)) | Sovereignty (fail-closed) |
| Private-data embeddings | Never leaves the boundary | `INV-RESIDENCY` | Sovereignty (fail-closed) |

`INV-RESIDENCY` is enforced in `lensemble.data.residency`: any attempt to serialize a raw observation,
action, or private-data embedding into an outbound message or artifact raises `ResidencyViolation`,
which is fail-closed and never caught-and-ignored ([04 — Error Model](04-error-model.md),
[06 — Security](06-security.md)). The probe $\mathcal{P}$ is public/licensed data and its embeddings
$E_{\text{ref}}=f_{\text{ref}}(\mathcal{P})$ are not private, so they may cross for alignment.

## 4. Training topology

Training nests two loops. The standard distributed-training stack is the *inner* loop and is not the
contribution; the federated outer loop and its gauge control are. See
[RFC-0001 §5](../rfcs/RFC-0001-architecture.md#5-training-topology-two-level) and [RFC-0003 §2](../rfcs/RFC-0003-federated-protocol.md#2-round-structure-diloco-outer-loop).

- Inner — intra-participant, for scale. Within a participant, standard FSDP2 / tensor / context
  parallelism (torch `>=2.4`) trains the warm-started model (ViT-L/~300M at Stage A, toward the 1.2B
  target). The inner optimizer is AdamW. SIGReg projection statistics are reduced *within* this trust
  domain freely — the reduce-within-trust-domain rule ([RFC-0008](../rfcs/RFC-0008-model-objective-numerics.md)).
  This is the only place the large-model-parallelism playbook applies; it is off-the-shelf.
- Outer — inter-participant, for sovereignty. DiLoCo: each participant runs $H$ inner steps, then the
  coordinator applies an outer Nesterov step on the averaged pseudo-gradient
  $(\theta_{t+1},\phi_{t+1}) = (\theta_t,\phi_t) - \eta_{\text{out}}\,\mathrm{Nesterov}\big(\tfrac{1}{C}\sum_c\Delta_c\big)$,
  where $\Delta_c = (\theta_c^{\text{local}},\phi_c^{\text{local}}) - (\theta_t,\phi_t)$. Only $\Delta_c$
  crosses the boundary, after DP clip+noise and secure aggregation. Sync frequency is every $H$ inner
  steps; $H\in[50,500]$ (smaller while characterizing drift). $H$ is tuned against frame drift
  ([RFC-0002 §4](../rfcs/RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)): larger $H$ is cheaper communication but
  more drift to anchor.

The outer step is on the bitwise-deterministic aggregation path (`INV-AGG-DETERMINISM`, enforced in
`lensemble.aggregation` / `lensemble.federation.outer_optimizer`): fixed reduction order, fp32 with
fixed summation order, no atomics. A determinism self-check runs each outer step; failure raises
`NonDeterministicAggregation` and aborts the round ([conventions §9](conventions.md#9-determinism-dtype-device), [04 — Error Model](04-error-model.md)). The
inner loop is best-effort deterministic and seed-pinned; full inner determinism is gated by a config
flag (`torch.use_deterministic_algorithms`).

## 5. Trust boundaries

Each participant is a sovereign trust domain. The coordinator/aggregator is a separate domain, treated
as honest-but-curious in Phase 1 and as a proving target in Phase 2 ([RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md)).
This diagram reproduces [RFC-0001 §6](../rfcs/RFC-0001-architecture.md#6-trust-boundaries); the prose below says the same.

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

What crosses a boundary: model deltas $\Delta_c$ (DP-protected and secure-aggregated, so the
coordinator sees only $\sum_c\Delta_c$), dataset commitments $R_c$, and shared coordination state (the
sketch seed $s_t$, the probe hash, the global-model hash). What never crosses: raw observations,
actions, or embeddings of private data — enforced by `INV-RESIDENCY`. The frame re-alignment the
coordinator performs uses only the public probe and the committed weights, so it is publicly
recomputable ([RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md), via
`lensemble.verify.recompute_alignment`). The boundary-crossing message set and per-message protection
are tabulated in [06 — Security](06-security.md) and [RFC-0003 §8](../rfcs/RFC-0003-federated-protocol.md#8-message-summary).

## 6. Data flow lifecycles

### 6.1 Federated training round (Stage B/C)

End-to-end, per round $t$, mapping to [RFC-0003 §2](../rfcs/RFC-0003-federated-protocol.md#2-round-structure-diloco-outer-loop):

1. Coordinator broadcasts `RoundOpen`: the global parameters $(\theta_t,\phi_t)$ reference/hash, the
   round sketch seed $s_t$, the probe hash, and $H$. Action heads $h_\psi^{(c)}$ are not broadcast
   (`INV-ACTIONHEAD-LOCAL`).
2. Each participant verifies the probe hash against the pinned probe (`INV-PROBE-PIN`; mismatch raises
   `ProbeError`) and derives the identical sketch matrix $A$ from $s_t$ (`INV-SKETCH-CONSISTENCY`).
3. Local optimization: $H$ inner AdamW steps on the objective
   $\mathcal{L} = \lambda_{\text{pred}}\,\mathbb{E}\lVert g_\phi(f_\theta(x_t),a_t) - \text{sg}[f_\theta(x_{t+1})]\rVert^2 + \lambda_{\text{sig}}\,\mathrm{SIGReg}_A(f_\theta(x)) + \lambda_{\text{anc}}\,\mathcal{L}_{\text{anchor}}(f_\theta;\mathcal{P},\{t_i\})$,
   over local data only.
4. Form the pseudo-gradient $\Delta_c = (\theta_c^{\text{local}},\phi_c^{\text{local}}) - (\theta_t,\phi_t)$.
5. Privatize: clip $\Delta_c \leftarrow \Delta_c\cdot\min(1, C_{\text{clip}}/\lVert\Delta_c\rVert)$
   (`INV-DP-BOUND`, post-clip $\lVert\Delta_c\rVert \le C_{\text{clip}}$), then add
   $\mathcal{N}(0,\sigma^2 C_{\text{clip}}^2 I)$ ([RFC-0012](../rfcs/RFC-0012-differential-privacy.md)).
   The accountant updates cumulative $(\varepsilon,\delta)$; exhaustion raises `PrivacyBudgetExceeded`
   and stops training.
6. Each participant emits `Commitment` carrying its dataset Merkle root $R_c$, binding $\Delta_c$ to
   exactly one $R_c$ (`INV-COMMIT-BINDING`; mismatch raises `CommitmentMismatch`, update rejected).
7. Secure-aggregate: compute $\sum_c\Delta_c$ without revealing any individual $\Delta_c$
   ([RFC-0011](../rfcs/RFC-0011-secure-aggregation.md)). Dropout below threshold raises
   `SecureAggregationError`; participant churn below the minimum raises `FaultToleranceExceeded`.
8. Backstop align: if frame drift on the public probe exceeds the threshold, apply the Procrustes
   re-alignment $Q^\star$ ([RFC-0002 §5](../rfcs/RFC-0002-gauge-and-aggregation.md#5-layer-3--procrustes-re-alignment-at-aggregation-backstop));
   `FrameDriftExceeded` triggers it, a degenerate SVD raises `DegenerateProcrustes`.
9. Outer step: deterministic Nesterov update producing $(\theta_{t+1},\phi_{t+1})$ (`INV-AGG-DETERMINISM`;
   self-check failure raises `NonDeterministicAggregation`, aborts and recomputes).
10. Commit: hash-commit $(\theta_{t+1},\phi_{t+1})$ (`INV-CHECKPOINT-HASH`); emit `RoundClose` with the
    hash; append a `ContributionRecord` to the ledger ([RFC-0014](../rfcs/RFC-0014-provenance-commitments.md)).

```
RoundOpen → [verify probe + derive A] → H inner steps → Δ_c → clip+noise
  → Commitment(R_c) → secure-agg Σ_c Δ_c → (drift? Procrustes Q*) → Nesterov outer step
  → hash-commit θ_{t+1} → RoundClose → ledger append
```

### 6.2 Single-site Stage-A training

1. Load `LensembleConfig` (Hydra structured config); validate (raises `ConfigError`); emit a
   `RunManifest` (config hash, seeds, env, versions, git SHA).
2. Build the WMCP-conformant encoder (warm-started from V-JEPA 2), predictor, and a single action head;
   snapshot $f_{\text{ref}}$.
3. Train end-to-end on pooled robot data with the three-term objective. This is the centralized upper
   bound (no federation, no DP, no secure aggregation). Public entry point `train_local(config) -> RunResult`
   ([02 — Public API](02-public-api.md)).
4. Periodically checkpoint via the schema-versioned, hash-committed artifact format
   ([RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md)).
5. Evaluate via latent MPC (§6.3) to validate the objective before federation.

### 6.3 Evaluation (latent MPC)

1. Load a committed checkpoint (verify `INV-CHECKPOINT-HASH`; tamper raises `CheckpointIntegrityError`).
2. Encode the goal and current observation to latent states via $f_\theta$.
3. The `Planner` runs latent model-predictive control — CEM / iCEM / MPPI minimizing an $L_1$
   goal-energy in latent space, rolling $g_\phi$ forward under candidate action sequences, exactly as
   `stable-worldmodel` provides ([RFC-0005](../rfcs/RFC-0005-evaluation.md)).
4. The eval harness records `eval/success_rate`, `eval/planning_samples`, `eval/time_per_action_ms` and
   produces an `EvalReport`. Public entry point `evaluate(checkpoint, env_id, *, cfg) -> EvalReport`.

### 6.4 Dataset commit

1. On ingest, each episode is canonically serialized and hashed to a domain-separated SHA-256 leaf.
2. Leaves are assembled into a Merkle tree with defined ordering; the root is $R_c$
   ([RFC-0014](../rfcs/RFC-0014-provenance-commitments.md)).
3. `commit_dataset(dataset) -> DatasetCommitment` returns the root $R_c$, episode count, and WMCP
   metadata. The commitment binds future pseudo-gradients to this data (`INV-COMMIT-BINDING`); a
   verification failure raises `MerkleVerificationError`.
4. This runs from day one for Phase-1 tamper-evidence and is the bridge to Phase-2 proofs
   ([RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md)).

## 7. Process and concurrency model

- Coordinator process. One per federation. Runs the round state machine
  `OPEN → COLLECTING → AGGREGATING → ALIGNING → COMMITTING → CLOSED` (with an `ABORTED` path), holds the
  canonical global model, runs the outer optimizer on the deterministic path, and emits the contribution
  ledger ([RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md)). Public entry point
  `Coordinator.run(num_rounds: int) -> None`.
- Participant processes. One per sovereign node. Each runs `Participant.local_round(global_state, round_seed) -> PseudoGradient`,
  performing the $H$ inner steps and emitting a privatized, commitment-bound delta. Backpressure and
  timeouts on the collection step are handled by the coordinator; a participant that misses the window is
  reconciled next round (elasticity, [RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md)).
- Inner parallel workers. Within a participant, FSDP2 / tensor / context parallelism distributes the
  large model across local devices. This is intra-trust-domain; no Lensemble boundary is crossed here.
- Simulation harness. In Stage B the coordinator and participants run in one process (in-process round
  simulation) so the gauge science is studied on one cluster; in Stage C they run as separate networked
  processes over a real boundary ([RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md)).

Device contract ([conventions §9](conventions.md#9-determinism-dtype-device)): CUDA primary, with a CPU fallback path that runs the small CI configs; tests
must pass on CPU. Compute dtype is bf16 forward with fp32 master weights and loss/statistic
accumulation; the aggregation path is fp32 with fixed summation order (or fp64).

## 8. Staged plan mapped to milestones

The A–E staging from [RFC-0001 — Migration / Rollout](../rfcs/RFC-0001-architecture.md#migration--rollout) maps onto the release milestones
([conventions §12](conventions.md#12-milestones-and-stages)). Each stage gates the next.

| Stage | Goal | Milestone |
|---|---|---|
| A | Single-site, warm-started, ViT-L/~300M end-to-end SIGReg + AC predictor on pooled robot data; latent-MPC eval. The centralized upper bound, plus foundational scaffolding (package skeleton, config, data layer, WMCP contract, model+objective, eval harness, observability, artifact format, error taxonomy, CI, packaging). | v0.1 |
| B | Simulated federation on one cluster: DiLoCo outer loop, frame anchor (Layers 1–4), Procrustes backstop, simulated secure aggregation + DP, the frame-drift diagnostic, the full ablation ladder and non-IID/scale sweeps. The scientific core / the paper. | v0.2 |
| C | Two real sovereign nodes over a network boundary: real secure aggregation + DP, residency enforcement, fault tolerance/elasticity, contribution ledger. The sovereignty demonstration. | v0.3 |
| (hardening) | Frozen public API, complete docs + reproducibility package, release automation, Fork A fallback supported and tested, proof-ready guarantees verified end-to-end. | v1.0 |
| D | The actual STARK/TEE verifiable layer (Phase 2) beyond the proof-ready disciplines. | Out of v1.0 scope (future work) |
| E | Own foundation-scale federated video pretraining (toward 1.2B from scratch). | Out of v1.0 scope (future work) |

Proof-*ready* engineering disciplines (deterministic aggregation, hash commitments, Merkle roots,
pinned probe, public recomputation) are in scope for v0.1–v1.0; the actual proofs (Stage D) and
own-pretrain (Stage E) are future work captured in the tracker, not implementable v1.0 issues ([conventions §12](conventions.md#12-milestones-and-stages),
[RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md)).

Fork B (end-to-end, encoder co-trained) is the target architecture. Fork A (frozen shared encoder,
federate the predictor only) is the documented safe-degrade fallback: it dissolves the latent gauge
entirely at the cost of the end-to-end novelty, and it must be supported and tested by v1.0
([RFC-0002](../rfcs/RFC-0002-gauge-and-aggregation.md), [RFC-0008](../rfcs/RFC-0008-model-objective-numerics.md)).

## 9. Open questions

OPEN QUESTION: The personalization boundary — how heterogeneous the participating embodiments can be
before a single global encoder $f_\theta$ stops being viable and the shared-backbone +
per-embodiment-head split breaks. The split is the hedge; the boundary is empirical. Owner @AbdelStark;
resolution path: the non-IID severity sweep in Stage B ([RFC-0001](../rfcs/RFC-0001-architecture.md),
[RFC-0005](../rfcs/RFC-0005-evaluation.md)).

RISK: The architecture depends on a released V-JEPA 2 warm-start for both initialization and the $t{=}0$
frame anchor (`INV-WARMSTART-T0`). If the warm-start is unavailable or unsuitable at the target scale,
both the gauge closure and the modest-compute staging assumption weaken. Resolution plan: Stage A
validates the objective and eval on the warm-start before any federation work; if warm-start quality is
insufficient, the Fork A fallback (frozen encoder) still recovers a clean federation
([RFC-0002](../rfcs/RFC-0002-gauge-and-aggregation.md)).

## 10. References

- [00 — Overview](00-overview.md) — thesis, contribution, success criteria.
- [02 — Public API](02-public-api.md) — the public surface this architecture exposes.
- [03 — Data Model](03-data-model.md) — schemas for the types named here.
- [04 — Error Model](04-error-model.md) — the error taxonomy referenced throughout.
- [06 — Security](06-security.md) — the threat model and boundary-crossing message table.
- [07 — Testing Strategy](07-testing-strategy.md) — the module-boundary and round-lifecycle tests.
- [RFC-0001 — Architecture & System Overview](../rfcs/RFC-0001-architecture.md) — originating rationale.
- [RFC-0002 — The Latent Gauge & Frame-Anchored Aggregation](../rfcs/RFC-0002-gauge-and-aggregation.md).
- [RFC-0003 — Federated Training Protocol](../rfcs/RFC-0003-federated-protocol.md).
- [RFC-0004 — Data, Sovereignty & Provenance](../rfcs/RFC-0004-data-provenance.md).
- [RFC-0005 — Evaluation & Benchmark Protocol](../rfcs/RFC-0005-evaluation.md).
- [RFC-0006 — Verifiable Contribution](../rfcs/RFC-0006-verifiable-contribution.md).
- [RFC-0007 — WMCP Latent Contract & Embodiment Adapters](../rfcs/RFC-0007-wmcp-latent-contract.md).
- [RFC-0008 — Model, Objective & Numerical Contracts](../rfcs/RFC-0008-model-objective-numerics.md).
- [RFC-0009 — Configuration, Run Manifest & Reproducibility](../rfcs/RFC-0009-configuration-reproducibility.md).
- [RFC-0010 — Checkpoint & Artifact Format](../rfcs/RFC-0010-artifact-checkpoint-format.md).
- [RFC-0011 — Secure Aggregation Protocol](../rfcs/RFC-0011-secure-aggregation.md).
- [RFC-0012 — Differential Privacy Accounting](../rfcs/RFC-0012-differential-privacy.md).
- [RFC-0013 — Coordinator & Participant Runtime](../rfcs/RFC-0013-coordinator-runtime.md).
- [RFC-0014 — Provenance Commitments & Merkle Scheme](../rfcs/RFC-0014-provenance-commitments.md).
- [RFC-0015 — Observability, Diagnostics & Telemetry](../rfcs/RFC-0015-observability-diagnostics.md).
- External: V-JEPA 2 (encoder warm-start, AC recipe); LeJEPA / LeWM (SIGReg objective);
  stable-worldmodel (data layer, envs, latent-MPC eval); WMCP (latent contract); DiLoCo / INTELLECT /
  PRIME (inner/outer optimizer, elastic fault tolerance); Stwo (Phase-2 prover). Constraints in [conventions §11](conventions.md#11-external-dependencies).
