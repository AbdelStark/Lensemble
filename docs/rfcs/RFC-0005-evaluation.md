# RFC-0005 — Evaluation & Benchmark Protocol

| | |
|---|---|
| **RFC** | 0005 |
| **Title** | Evaluation & Benchmark Protocol |
| **Slug** | evaluation |
| **Status** | Accepted |
| **Track** | Standards |
| **Authors** | @AbdelStark |
| **Created** | 2026-06-02 |
| **Target milestone** | v0.2 (Stage B) |
| **Area** | eval |
| **Requires** | RFC-0001, RFC-0002 |
| **Informs** | RFC-0008, RFC-0009, RFC-0015 |

## Summary

This RFC defines how Lensemble proves its claims. It is the backbone of the paper. The central
result is not "we trained distributed" (solved prior art) but "we measured and controlled the latent
gauge under federation, and federation closes the gap to centralized without moving data." It
specifies four falsifiable claims (§1); the headline empirical artifact, the **latent frame-drift
diagnostic** (the first measurement of latent frame-drift under federated self-supervision, §2); the
primary downstream metric, **planning success via latent MPC** (§3); the supporting metrics (§4); the
four bracketing baselines and the headline gap-recovery fraction (§5); the **ablation ladder** that
maps one-to-one onto the gauge layers of [RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix) (§6); the
non-IID and scale sweeps (§7); the reproducibility discipline (§8); and the success criteria (§9).
The metric implementations live in `lensemble.eval` (`metrics.py`, `mpc.py`, `harness.py`); the
emission contract is owned by [RFC-0015](RFC-0015-observability-diagnostics.md) and the stable type
contract for `EvalReport`/`FrameDriftReport` by [03-data-model.md §13](../spec/03-data-model.md#13-reporting-types-evalreport-framedriftreport-contributionrecord). The public entry points
are `evaluate(...)`, `Planner`, and `frame_drift(...)` ([conventions §5](../spec/conventions.md#5-public-api-surface), [02-public-api.md §1](../spec/02-public-api.md#1-public-python-surface)).

## Motivation

Federated and decentralized training of supervised models and LLMs is established prior art; merely
demonstrating that Lensemble trains across silos proves nothing new. The novel, contestable claim is
that an *end-to-end* JEPA world model — encoder and predictor co-trained, "Fork B" — can be
federated at all, given the $O(d)$ latent gauge argued in
[RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix), and that frame anchoring controls that gauge well
enough to recover most of the centralized–local quality gap without moving raw data. Two things must
therefore be measured that no prior evaluation measures:

1. **The gauge itself.** Whether independently-updated participants drift into mutually-rotated
   coordinate frames, and whether the anchor holds the frame pinned, is directly observable as a
   rotation between encoder outputs on a shared public probe. This measurement — the frame-drift
   diagnostic — is the headline artifact and a standalone contribution.
2. **That controlling the gauge buys downstream capability.** Frame stability is necessary but not
   sufficient; the model must plan. Planning success via latent MPC is the metric that ultimately
   matters; representation-probe accuracy and effective dimension support it and guard against silent
   partial collapse that success rate alone can mask.

An evaluation that reported only training curves, or only success rate, would leave the central
scientific question unanswered and the central failure mode (the gauge) invisible. This RFC fixes
exactly what is measured, against what baselines, under what reproducibility discipline, so the
paper's claims are falsifiable and the reference implementation's centralized/local/federated triple
is reproducible end-to-end.

## Goals

- State four **falsifiable claims** and the metric that decides each.
- Specify the **frame-drift diagnostic** precisely enough that the central figure is reproducible
  from committed weights plus the public probe alone, deterministically.
- Specify **planning success** via latent MPC (`evaluate(...)`, `Planner`) as the primary downstream
  metric, on held-out environments and held-out factors of variation.
- Specify the **supporting metrics**: representation-probe accuracy, effective dimension (collapse
  guard), communication bytes/rounds.
- Specify the **four baselines** and the headline quantity: the fraction of the centralized−local
  gap recovered by anchored federation, with no data moved.
- Specify the **ablation ladder** mapping one-to-one onto [RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)
  Layers 1–4, plus the $\lambda_{\text{anc}}$ sweep.
- Specify the **non-IID severity** and **participant-count / scale** sweeps.
- Fix the **reproducibility** discipline (Hydra configs, pinned probe hash, per-round sketch seeds,
  released checkpoints) so the triple reproduces.
- State the **success criteria**, both paper-grade and engineering-grade.

## Non-Goals

- This RFC does not implement the gauge mechanisms (anchor, Procrustes, sketch) — those are
  [RFC-0002](RFC-0002-gauge-and-aggregation.md) — nor the model/objective
  ([RFC-0008](RFC-0008-model-objective-numerics.md)); it only consumes them.
- It does not specify the metric *emission* schema, log records, or sinks; those are owned by
  [RFC-0015 §3-4](RFC-0015-observability-diagnostics.md). This RFC specifies *what* is computed and
  *what it means*; RFC-0015 specifies *how it is emitted and made reproducible from logs*.
- It does not specify the config system or `RunManifest` schema — that is
  [RFC-0009](RFC-0009-configuration-reproducibility.md) — only which fields a reported run must pin.
- It does not select the final environment suite or the exact gap-recovery fraction to claim; those
  are Open Questions resolved in Stage B.
- Stage-D realized proofs and Stage-E own-pretraining are out of v1.0 scope ([conventions §12](../spec/conventions.md#12-milestones-and-stages)); the scale
  sweep here repeats key rungs at increasing encoder size, it does not perform Stage-E pretraining.

## Proposed Design

### 1. Claims to test

Each claim is falsifiable and is decided by a named metric. Failure of Claim 1 or 2 falsifies the
gauge thesis of [RFC-0002](RFC-0002-gauge-and-aggregation.md); failure of Claim 3 falsifies the
practical value of anchored federation and triggers the Fork A degrade
([RFC-0002 Fork A fallback](RFC-0002-gauge-and-aggregation.md#fork-a-fallback)).

| # | Claim | Decided by | Direction that confirms |
|---|---|---|---|
| 1 | Naive end-to-end `FedAvg` of a JEPA degrades / collapses under non-IID silos (the gauge in action). | frame drift (§2) + MPC success (§3) + effective dim (§4) | Naive run measurably worse on all three. |
| 2 | Frame anchoring ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)) holds the latent frame pinned across participants over training. | frame drift (§2) | Anchored drift stays flat/low where naive diverges. |
| 3 | Anchored federation closes most of the centralized–local gap on downstream planning, *without moving data*. | MPC success (§3), as fraction of the centralized−local gap (§5) | Recovered fraction ≥ a stated threshold (Open Question), data residency held (`INV-RESIDENCY`). |
| 4 | Robustness across non-IID severity and participant count $C$. | §2-4 metrics under the §7 sweeps | Claims 1–3 hold across the swept range. |

### 2. Headline diagnostic — latent frame drift

The novel measurement at the center of the paper. On the fixed public probe $\mathcal{P}$
([RFC-0004 §3](RFC-0004-data-provenance.md#3-the-public-probe-set-mathcalp)), after each round $t$ compute, for each ordered pair of
participants $(c,c')$, the optimal Procrustes rotation between their encoder outputs on the probe and
report a rotation magnitude. The closed form is exactly that of
[RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix) Variant B: from the SVD
$f_{\theta_{c'}}(\mathcal{P})^\top f_{\theta_c}(\mathcal{P}) = U\Sigma V^\top$, the optimal
$Q^\star_{c,c'} = VU^\top \in O(d)$, and the reported magnitudes are:

- the **mean inter-participant rotation angle** (degrees) — derived from $Q^\star_{c,c'}$ via the
  geodesic angle $\theta_{\text{rot}} = \arccos\big(\tfrac{\operatorname{tr}(Q^\star)-(d-2c_-)}{\ldots}\big)$,
  reported in practice as the Frobenius **Procrustes residual** $\lVert Q^\star_{c,c'} - I\rVert_F$
  (the same quantity the anchor minimizes), averaged over all pairs; and
- the same quantity against the **global model** $\theta_t$ (drift from consensus): for each $c$,
  the residual $\lVert Q^\star_{c,\text{global}} - I\rVert_F$ on $\mathcal{P}$.

The public function is `frame_drift(embeddings: Mapping[str, Tensor]) -> FrameDriftReport` ([conventions §5](../spec/conventions.md#5-public-api-surface),
contracted in [02-public-api.md §1.6](../spec/02-public-api.md#16-frame_drift-and-procrustes_align)); each key is a participant id (and the reserved key
`"global"`), each value is that participant's probe embeddings $f_{\theta_c}(\mathcal{P})$ of shape
$(\lvert\mathcal{P}\rvert\cdot N, d)$. The pairwise alignment is computed by `procrustes_align(source,
target) -> tuple[Tensor, float]` returning $(Q^\star, \text{residual})$ ([conventions §5](../spec/conventions.md#5-public-api-surface)).

**Expected figure.** Naive `FedAvg` curves diverge (frames rotate apart); the anchored configuration
stays flat/low. To our knowledge this is the first measurement of latent frame-drift under federated
self-supervision; it stands on its own as a contribution.

**Determinism and reproducibility (load-bearing).** The diagnostic MUST be a deterministic function
of (committed encoder weights $\theta_c$, the pinned public probe $\mathcal{P}$). Inputs derive only
from hash-committed checkpoints ([RFC-0010](RFC-0010-artifact-checkpoint-format.md),
`INV-CHECKPOINT-HASH`) and the hash-pinned probe (`INV-PROBE-PIN`); the probe content hash MUST equal
the hash committed in `RoundOpen` or `frame_drift` refuses to run with `ProbeError`. The Procrustes
SVD path follows the conditioning rule of
[RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix): near-degenerate singular values are clamped, and
a degenerate decomposition raises `DegenerateProcrustes` rather than emitting a meaningless angle.
Because the diagnostic depends only on public-probe data plus committed weights, it is also publicly
recomputable ([RFC-0002 §5](RFC-0002-gauge-and-aggregation.md#5-layer-3--procrustes-re-alignment-at-aggregation-backstop),
[RFC-0006 §4](RFC-0006-verifiable-contribution.md#4-public-recomputation-phase-1-free-in-scope-now)) and needs no ZK proof. The per-round,
per-participant-pair emission record schema and the JSONL sink are owned by
[RFC-0015 §3](RFC-0015-observability-diagnostics.md#3-the-frame-drift-diagnostic-emission-contract-the-headline-artifact); this section fixes the quantity and its
determinism, so the headline figure is fully reproducible from logs alone.

`FrameDriftReport` (full schema in [03-data-model.md §13.2](../spec/03-data-model.md#132-framedriftreport--the-headline-empirical-artifact)) carries at minimum: `round_index:
int`, `pairwise_residual: dict[tuple[str, str], float]`, `pairwise_angle_deg: dict[tuple[str, str],
float]`, `drift_from_global: dict[str, float]`, `probe_hash: str`, `effective_dim: dict[str, float]`
(§4). Emitted metrics: `gauge/drift_angle_deg`, `gauge/procrustes_residual`, `gauge/effective_dim`
([RFC-0015 §3](RFC-0015-observability-diagnostics.md#3-the-frame-drift-diagnostic-emission-contract-the-headline-artifact)).

### 3. Downstream metric — planning success

The thing that ultimately matters. Via `stable-worldmodel`, the trained model is used as the
cost / world model for **latent model-predictive control**:

- The planner is one of CEM / iCEM / MPPI, minimizing an $L_1$ **goal-energy** in latent space
  between the predicted future latent and a goal-image latent. The planner is the `Planner` class
  ([conventions §5](../spec/conventions.md#5-public-api-surface)); the planner family is a config choice (§"Alternatives Considered").
- Report **success rate** via `stable-worldmodel`'s `world.evaluate` on **held-out** environments and
  **held-out factors of variation**, with goal-image specification.
- Report **planning cost** — number of sampled action sequences and wall time per action — for parity
  with baselines.

The public entry point is `evaluate(checkpoint: Path, env_id: str, *, cfg) -> EvalReport` ([conventions §5](../spec/conventions.md#5-public-api-surface)).
Its contract (full statement in [02-public-api.md §1.5](../spec/02-public-api.md#15-evaluate-and-planner)):

```python
def evaluate(checkpoint: Path, env_id: str, *, cfg: LensembleConfig) -> EvalReport: ...
```

- **Preconditions.** `checkpoint` is a hash-verified `Checkpoint`
  ([RFC-0010](RFC-0010-artifact-checkpoint-format.md)); a tampered or schema-mismatched artifact
  raises `CheckpointIntegrityError` / `SchemaVersionMismatch` and `evaluate` refuses to load
  (`INV-CHECKPOINT-HASH`). `env_id` resolves to a `stable-worldmodel` environment held out from
  training partitions. `cfg.eval` fixes the planner family, horizon, sample count, seed, and the
  held-out factor set.
- **Postconditions.** Returns an `EvalReport` (schema in [03-data-model.md §13.1](../spec/03-data-model.md#131-evalreport)) carrying:
  `success_rate: float`, `n_episodes: int`, `planning_samples: int`, `time_per_action_ms: float`,
  `env_id: str`, `held_out_factors: list[str]`, `goal_spec_hash: str`, `seed: int`, `config_hash:
  str`. The report is reproducible under a fixed seed ([conventions §9](../spec/conventions.md#9-determinism-dtype-device); inner determinism is best-effort,
  seed-pinned).
- **Errors.** `EvaluationError` for an unresolvable `env_id` or planner misconfiguration;
  `CheckpointIntegrityError` on a tampered checkpoint; `ConfigError` on an invalid `cfg.eval`.
- **Determinism.** Best-effort, seed-pinned ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)); the report records `seed` and `config_hash` so
  a run is reproducible. The metric value is not required to be bitwise-identical across hardware
  (the aggregation path is the only bitwise-deterministic surface, `INV-AGG-DETERMINISM`), but the
  same seed + config + checkpoint on the same device class reproduces the reported `success_rate`
  within the §"numerical-tolerance" band.

Emitted metrics: `eval/success_rate`, `eval/planning_samples`, `eval/time_per_action_ms`
([RFC-0015 §3](RFC-0015-observability-diagnostics.md#3-the-frame-drift-diagnostic-emission-contract-the-headline-artifact)).

### 4. Supporting metrics

- **Representation quality** — linear / attentive probe accuracy on a held-out downstream task on the
  *frozen* learned encoder. Supports planning success; planning is primary, probe accuracy is
  corroborating evidence that the representation is usable (§"Alternatives Considered").
- **Collapse** — **effective dimension** of the embedding covariance, computed from the eigenspectrum
  of $\operatorname{Cov}(f_\theta(x))$ as the participation ratio
  $(\sum_i \sigma_i)^2 / \sum_i \sigma_i^2$ (or the stable-rank surrogate), where $\sigma_i$ are the
  covariance eigenvalues. Guards against silent partial collapse that success rate alone might mask:
  averaging across mutually-rotated frames can re-introduce the low-rank solutions SIGReg prevented
  locally ([RFC-0002 §2.1](RFC-0002-gauge-and-aggregation.md#21-three-failures-that-compound-it), collapse re-entry). Emitted as
  `gauge/effective_dim` per participant and for the global model.
- **Communication** — total bytes and rounds, to contextualize against centralized cost and support
  the DiLoCo efficiency claim (communicate every $H$ steps;
  [RFC-0003](RFC-0003-federated-protocol.md)). Emitted as `fed/comm_bytes`, `fed/round_seconds`,
  `fed/participants`, `fed/quant_ratio` ([RFC-0015 §3](RFC-0015-observability-diagnostics.md#3-the-frame-drift-diagnostic-emission-contract-the-headline-artifact));
  the byte accountant and the budget reasoning live in [08-performance-budget.md §4](../spec/08-performance-budget.md#4-communication-budget--the-diloco-efficiency-claim).

Each metric has a unit and a unit test (§"Testing Strategy"). The metric implementations are in
`lensemble.eval.metrics`; effective-dimension shares the eigendecomposition discipline of the gauge
diagnostics (conditioning, fp32 accumulation, [conventions §9](../spec/conventions.md#9-determinism-dtype-device)).

### 5. Baselines

The four baselines bracket the result. The headline claim (Claim 3) is quantified as the **fraction
of the centralized − local gap recovered** by anchored federation: if $S$ denotes MPC success rate,
the recovered fraction is

$$\rho = \frac{S_{\text{anchored-fed}} - S_{\text{local-only}}}{S_{\text{centralized-pooled}} - S_{\text{local-only}}} \in [0, 1].$$

| Baseline | Role | Training |
|---|---|---|
| **Centralized-pooled** (all silo data in one place, end-to-end) | **Upper bound** — what federation aspires to | `train_local(config)` on pooled data; no outer loop, no boundaries ([RFC-0001 §7(b)](RFC-0001-architecture.md#7-data-flow-lifecycles)). |
| **Local-only** (each silo trains alone, no sync) | **Lower bound** — value of federating at all | One `train_local` per silo; report best / mean per the §7 protocol. |
| **Naive end-to-end `FedAvg`** (no gauge control) | **Negative control** — the collapse/divergence the design fixes | DiLoCo outer loop with $\lambda_{\text{anc}}=0$ and no Procrustes backstop. |
| **Fork A** (frozen shared encoder, federate predictor only) | Reference point; the safe degrade ([RFC-0002 Fork A fallback](RFC-0002-gauge-and-aggregation.md#fork-a-fallback)) | Encoder frozen at warm-start; federate $g_\phi$ only; the gauge dissolves. |

All baselines share the same warm-start, public probe, environment suite, seeds, and reporting
discipline so $\rho$ is well-defined. Centralized-pooled and local-only move no model deltas across a
federation boundary by construction; for them `INV-RESIDENCY` is enforced only at the
dataset-commit / eval boundary.

### 6. Ablation ladder (the core experiment)

Each rung adds one mechanism from [RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix); the ladder is the
gauge fix realized additively, so it *is* the rollout of RFC-0002. Report all three metric families
(frame drift §2, MPC success §3, collapse / effective dim §4) at each rung:

1. **Naive end-to-end `FedAvg`** — the negative control (no gauge control).
2. **+ shared sketch matrix** $A$ ([RFC-0002 §3](RFC-0002-gauge-and-aggregation.md#3-layer-1--shared-sketch-matrix-objective-consistency-not-the-gauge-fix),
   `INV-SKETCH-CONSISTENCY`). Objective consistency, not the gauge fix; the frame still drifts.
3. **+ Procrustes align-then-average** at aggregation
   ([RFC-0002 §5](RFC-0002-gauge-and-aggregation.md#5-layer-3--procrustes-re-alignment-at-aggregation-backstop)). The backstop alone.
4. **+ frame-anchor loss** — landmark (Variant A) or rotational (Variant B)
   ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)). **← expected recommended configuration.**
5. **+ function-space distillation** ([RFC-0002 §6](RFC-0002-gauge-and-aggregation.md#6-layer-4--function-space-distillation-fallback--heterogeneity)) — the
   heterogeneity / instability fallback.

Also sweep the central knob $\lambda_{\text{anc}}$ ([RFC-0002 §7](RFC-0002-gauge-and-aggregation.md#7-the-central-hyperparameter)):
frame drift and MPC success vs $\lambda_{\text{anc}}$, to locate the "pin frame, not content" sweet
spot — too high clamps the encoder to the reference frame and its quality, too low lets the frame
drift. The sweep is driven by Hydra config composition ([conventions §11](../spec/conventions.md#11-external-dependencies);
[RFC-0009](RFC-0009-configuration-reproducibility.md)); each rung and each $\lambda_{\text{anc}}$
value is a config group override, and a small-config version of each rung is a CPU integration test
(§"Testing Strategy").

### 7. Non-IID severity & scale sweeps

- **Non-IID severity.** Partition a multi-environment / multi-factor-of-variation corpus across silos
  by factor-of-variation or by embodiment; `stable-worldmodel`'s factors-of-variation give
  *controlled, reproducible* heterogeneity. Sweep from near-IID to strongly non-IID; report
  degradation curves for each ladder rung. This sweep also resolves the personalization-boundary Open
  Question of [RFC-0001](RFC-0001-architecture.md) (how heterogeneous before one global encoder
  breaks).
- **Participant count $C$ and inner horizon $H$.** Vary $C$ and $H$; report effect on convergence and
  frame drift. Longer $H$ rotates frames further apart before the outer step
  ([RFC-0002 §2.1](RFC-0002-gauge-and-aggregation.md#21-three-failures-that-compound-it), DiLoCo drift), so this sweep validates DiLoCo
  robustness in the JEPA setting and characterizes the $H$ schedule
  ([RFC-0003](RFC-0003-federated-protocol.md)).
- **Scale.** Repeat the key ladder rungs at increasing encoder size from ViT-L/~300M toward
  V-JEPA-2-class to show the recipe holds ([conventions §12](../spec/conventions.md#12-milestones-and-stages): Stage A validates ViT-L centrally; the scale
  steps are within v0.2–v1.0; Stage E own-pretraining is out of scope). The cost model for scaling is
  in ([08-performance-budget.md](../spec/08-performance-budget.md)).

Note that pairwise drift is $O(C^2)$ per round; at large $C$ the diagnostic samples participant pairs
rather than enumerating all of them ([RFC-0015](RFC-0015-observability-diagnostics.md) Open
Question).

### 8. Reproducibility & reporting

Every reported run pins ([conventions §9](../spec/conventions.md#9-determinism-dtype-device); schema in
[RFC-0009](RFC-0009-configuration-reproducibility.md)):

- Hydra configs, the root seed and derived component seeds, the per-round sketch seeds
  $s_t = \mathrm{derive}(\text{root\_seed}, t)$ (`INV-SKETCH-CONSISTENCY`), and the pinned probe
  content hash (`INV-PROBE-PIN`).
- A `RunManifest` (config content-hash, seeds, git SHA, environment, pinned dependency versions,
  probe hash) emitted by every run.
- Reported alongside results: hardware, number of rounds, $H$, communication bytes, the DP
  $(\varepsilon,\delta)$ budget ([RFC-0012](RFC-0012-differential-privacy.md)), and the $\lambda$
  settings ($\lambda_{\text{pred}}, \lambda_{\text{sig}}, \lambda_{\text{anc}}$).

Release checkpoints + Hydra configs in the reference repo ([RFC-0001 Migration / Rollout](RFC-0001-architecture.md#migration--rollout),
[09-release-and-versioning.md §5](../spec/09-release-and-versioning.md#5-release-process)) so the **centralized / local / federated triple is
reproducible end to end**. The frame-drift figure is reproducible from committed weights + the pinned
probe alone (§2); the planning numbers from released checkpoints + configs + seeds (§3).

### 9. Success criteria

**Paper-grade.**

- Claim 1 demonstrated — naive `FedAvg` measurably worse on frame drift (§2) and MPC success (§3).
- Claim 2 demonstrated — anchored frame drift flat where naive diverges (§2).
- Claim 3 quantified — recovered gap fraction $\rho$ (§5) $\ge$ a stated threshold (Open Question),
  with no data moved (`INV-RESIDENCY` held).
- Claim 4 — Claims 1–3 hold across at least one non-IID severity sweep and one scale step (§7).

**Engineering-grade.** The reference implementation reproduces the **centralized / local / federated
triple** end-to-end from released checkpoints + Hydra configs + pinned seeds (§8): `train_local` for
the centralized upper bound and each local-only silo; the `Coordinator`/`Participant` federation run
([RFC-0013](RFC-0013-coordinator-runtime.md)) for the anchored and naive configurations; `evaluate`
for the MPC numbers; `frame_drift` for the headline figure. A new contributor reproduces every
figure from the released artifacts without contacting the author.

### Data flow & lifecycle

Evaluation is read-only with respect to training state. A `Checkpoint` produced by a round commit
([RFC-0001 §7(a) step 9](RFC-0001-architecture.md#7-data-flow-lifecycles)) is loaded (hash-verified), the encoder embeds the
pinned probe and the eval-environment goal/observation, the metrics are computed, and an `EvalReport`
/ `FrameDriftReport` is emitted to the metrics JSONL and structured log
([RFC-0015](RFC-0015-observability-diagnostics.md)). No raw observation/action/private embedding is
ever serialized into a report or log (`INV-RESIDENCY`,
[05-observability.md §5](../spec/05-observability.md#5-redaction-inv-residency)): reports carry only scalar metrics, hashes, shapes, and counts; the
redaction guard fails closed on any attempt to emit a raw tensor.

### Concurrency & determinism

- The **frame-drift diagnostic** is bitwise-reproducible given committed weights + the pinned probe
  (no random sampling on its core path; pair sampling at large $C$ is seeded and recorded). The
  Procrustes SVD uses the conditioning of [RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix).
- **Planning success** is best-effort deterministic and seed-pinned ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)); the report records the
  seed and config hash. The planner's stochastic search (CEM/iCEM/MPPI) is seeded from the run seed.
- Effective dimension and probe accuracy use fp32 accumulation ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)).
- Evaluation runs MUST pass on the CPU fallback for the tiny CI configs ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)).

### Failure modes handled

| Failure | Detection | Error ([conventions §6](../spec/conventions.md#6-error-taxonomy)) | Response |
|---|---|---|---|
| Probe hash differs from committed hash | `frame_drift` / eval ingress | `ProbeError` | Refuse to compute the diagnostic; the figure is invalid (`INV-PROBE-PIN`). |
| Degenerate Procrustes SVD (near-zero singular values) | `procrustes_align` | `DegenerateProcrustes` | Clamp/condition per [RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix); raise rather than emit a meaningless angle. |
| Tampered / corrupt checkpoint | `evaluate` artifact load | `CheckpointIntegrityError` | Refuse to load (`INV-CHECKPOINT-HASH`). |
| Checkpoint schema too new / unknown | artifact load | `SchemaVersionMismatch` | Refuse to load ([RFC-0010](RFC-0010-artifact-checkpoint-format.md)). |
| Unresolvable `env_id` / invalid planner config | `evaluate` | `EvaluationError` / `ConfigError` | Fail before running; remediation in `.remediation`. |
| Attempt to emit a raw obs/action/embedding into a report | `observability.redaction` | `ResidencyViolation` | Fail-closed; never caught-and-ignored (`INV-RESIDENCY`). |

## Alternatives Considered

**MPC planning success vs representation-probe accuracy as the PRIMARY metric.** Probe accuracy
(linear/attentive probe on the frozen encoder) is cheaper and standard in SSL evaluation. It was
considered as the headline downstream metric. Rejected as primary because planning is the thing that
ultimately matters for a world model — a representation can probe well yet plan poorly, and the
project's premise is embodied control, not classification. Probe accuracy is retained as a
*supporting* metric (§4): corroborating evidence that the representation is usable, and a cheaper
signal during sweeps.

**CEM vs iCEM vs MPPI for the planner.** All three are sampling-based latent-MPC planners that
`stable-worldmodel` supports. CEM (cross-entropy method) is the simplest; iCEM adds colored-noise
sampling and reuse for sample efficiency; MPPI (model-predictive path integral) uses a softmax-
weighted update. The planner family is a *config choice*, not a fixed decision, so the same model can
be evaluated under each and reported for parity with baselines; iCEM is the expected default for its
sample efficiency. None is rejected: the comparison itself is part of the planning-cost reporting
(§3).

**Reporting the Procrustes residual vs the mean rotation angle for drift.** The Frobenius residual
$\lVert Q^\star - I\rVert_F$ is exactly the quantity Variant B's anchor minimizes
([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)), so it is the most faithful measure of what the
mechanism controls and is numerically stable. The mean geodesic rotation angle (degrees) is more
interpretable for a figure. Decision: emit both (`gauge/procrustes_residual` and
`gauge/drift_angle_deg`); the residual is the canonical machine-checked quantity and the angle is the
figure label. Neither is rejected.

**Enumerate vs sample participant pairs for the $O(C^2)$ drift diagnostic.** Enumerating all pairs is
exact but quadratic in $C$. At large $C$ the diagnostic samples a seeded subset of pairs; the sample
is recorded in the manifest so the figure remains reproducible. Full enumeration is the default at the
$C$ used for the paper's central figure; sampling is the documented degrade for large-$C$ sweeps
([RFC-0015](RFC-0015-observability-diagnostics.md) Open Question).

## Drawbacks

- **Stage-B evidence is simulated federation on one cluster** ([conventions §12](../spec/conventions.md#12-milestones-and-stages)), not a real cross-network
  deployment. The frame-drift result and the gap-recovery fraction are established in simulation;
  Stage C ([RFC-0013](RFC-0013-coordinator-runtime.md)) demonstrates real sovereign nodes but is not
  where the paper's central numbers come from. This is stated plainly so the claim is not overread.
- **Success rate can mask partial collapse.** A model can retain some planning success while its
  representation silently loses rank. This is exactly why effective dimension (§4) is a first-class
  metric and is reported at every ladder rung; it is a guard, but a metric that fails to detect a
  particular collapse mode is a residual risk.
- **The gap-recovery fraction depends on the chosen non-IID severity.** $\rho$ (§5) is not a single
  number; it is a curve over the §7 severity sweep. Reporting a single headline $\rho$ requires
  fixing a severity, which is an Open Question; the honest report is the curve.
- **The diagnostic is only as good as the probe.** If the public probe under-covers the data manifold
  ([RFC-0004](RFC-0004-data-provenance.md) Open Question), the measured drift may understate the true
  frame divergence on private data. The probe-coverage acceptance criterion is owned by RFC-0004.

## Migration / Rollout

The evaluation suite lands with the simulated federation in **Stage B (v0.2)** ([conventions §12](../spec/conventions.md#12-milestones-and-stages)); it is the
scientific core / the paper. The rollout within Stage B follows the ablation ladder (§6), which *is*
the additive rollout of [RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix): rung 1 (naive) establishes
the negative control, each subsequent rung adds one mechanism, and the recommended configuration is
rung 4. The **scale steps repeat the key ablation rungs at increasing encoder size** (§7), from the
Stage-A ViT-L baseline toward V-JEPA-2-class within v0.2–v1.0; Stage-E own-pretraining is out of v1.0
scope. The metric *scaffold* (the `eval` module, `evaluate`, `frame_drift`, the metric
implementations and their unit tests) lands earlier with the Stage-A single-site eval
([RFC-0001 §7(c)](RFC-0001-architecture.md#7-data-flow-lifecycles)) so the centralized upper bound is measured before
federation exists; the federation-specific diagnostics activate in Stage B.

## Testing Strategy

The full pyramid is at [07-testing-strategy.md](../spec/07-testing-strategy.md); the eval-owned tests are unit tests of each
metric implementation plus a harness integration test, all runnable on the CPU fallback with tiny
synthetic fixtures ([conventions §9](../spec/conventions.md#9-determinism-dtype-device); no large downloads in CI).

- **Frame-drift metric.** Unit test on synthetically rotated silos: embed a fixed probe, apply a
  known $Q\in O(d)$ to one silo's embeddings, assert `frame_drift` recovers the corresponding residual
  / angle within tolerance and reports near-zero for an unrotated pair. Mirrors the RFC-0002
  diagnostic test ([RFC-0002 Testing Strategy](RFC-0002-gauge-and-aggregation.md#testing-strategy)): naive diverges,
  anchored holds flat.
- **Procrustes correctness.** `procrustes_align` closed form $Q^\star=VU^\top$ matches a brute-force
  / numerical-optimization search on small $d$ within tolerance; a degenerate (rank-deficient) input
  raises `DegenerateProcrustes`.
- **Effective-dimension metric.** On a synthetic Gaussian with a known covariance spectrum, the
  participation-ratio effective dimension matches the analytic value within tolerance; a rank-1 input
  reports effective dimension near 1 (collapse detected).
- **Success-rate metric.** On a toy `stable-worldmodel` env with a deterministic stub world model,
  `evaluate` returns a known success rate; assert `EvalReport` field types and that the seed/config
  hash are recorded.
- **Communication-bytes metric.** The byte accountant reports the expected serialized
  pseudo-gradient size for a tiny model, with and without int8 quantization
  ([RFC-0003](RFC-0003-federated-protocol.md)); cross-checked against
  [08-performance-budget.md §4](../spec/08-performance-budget.md#4-communication-budget--the-diloco-efficiency-claim).
- **Eval-harness integration on a toy env.** Wire `harness.py` end-to-end: load a tiny hash-verified
  checkpoint, run a few-step latent-MPC episode on a toy env, emit an `EvalReport` and a
  `FrameDriftReport`, assert both serialize/round-trip (pydantic v2, [conventions §8](../spec/conventions.md#8-core-data-types)) and that no raw tensor
  reaches a sink (redaction test, cross-referenced from [RFC-0015](RFC-0015-observability-diagnostics.md)).
- **Determinism and pinned probe.** Two runs with identical seeds and the same pinned probe hash
  produce identical `frame_drift` outputs (bitwise on the diagnostic core path) and `EvalReport`
  `success_rate` within the numerical-tolerance band; a probe hash mismatch raises `ProbeError`.
- **Ablation-ladder smoke tests.** A small-config version of each ladder rung (§6) runs on CPU and
  asserts the expected qualitative ordering (naive worst on drift; anchored flat), serving as the
  regression guard for the central experiment ([07-testing-strategy.md §3](../spec/07-testing-strategy.md#3-the-ablation-ladder-as-integration-tests)).

**Numerical-tolerance policy.** The frame-drift diagnostic core path requires bitwise equality across
repeated runs on the same device (it is deterministic). Procrustes residual, effective dimension, and
Gaussian-fixture metrics use `atol=1e-5, rtol=1e-4` in fp32. Planning success rate is compared within
a stated band (a small absolute tolerance on the rate) under fixed seed/config/device, reflecting its
best-effort determinism ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)). These tolerances are restated in [07-testing-strategy.md §6](../spec/07-testing-strategy.md#6-numerical-tolerance-policy).

## Open Questions

OPEN QUESTION: **The environment suite for Stage B.** Which `stable-worldmodel` environments and which
factors-of-variation partitions constitute the official Stage-B benchmark (the non-IID partition axes,
the held-out env/factor split, the goal-image specification). Owner @AbdelStark; resolution path:
fixed at the start of Stage B (v0.2) and pinned in the released Hydra configs (§8), with the choice
recorded in the paper's reproducibility appendix.

OPEN QUESTION: **The stated fraction of the centralized−local gap to claim recovered.** The threshold
on $\rho$ (§5) that Claim 3 must clear, and the non-IID severity at which the headline $\rho$ is
reported. Owner @AbdelStark; resolution path: the Stage B (v0.2) non-IID severity sweep (§7) produces
the full $\rho$-vs-severity curve; the headline threshold and severity are fixed from that curve
before the claim is asserted.

RISK: **Effective dimension may not catch every collapse mode.** A representation can lose
task-relevant structure while retaining nominal rank, or collapse along directions the participation
ratio under-weights. Resolution plan: report effective dimension alongside probe accuracy and MPC
success at every ladder rung (§4, §6), so three independent signals must agree; if a collapse mode
slips all three in Stage B, add a targeted probe (e.g. an anisotropy / per-direction-variance metric)
before the paper's claims are finalized.

RISK: **Simulated-federation evidence may not transfer to real sovereign nodes.** The paper's central
numbers come from Stage-B simulation on one cluster (Drawbacks); network-induced staleness, DP noise
on real deltas, and real participant churn ([RFC-0013](RFC-0013-coordinator-runtime.md)) could shift
the frame-drift and gap-recovery results. Resolution plan: Stage C (v0.3) re-runs the headline
diagnostic and the gap-recovery measurement on two real nodes and reports any delta from the
simulated result; the simulated and real numbers are reported side by side.

## References

- README (project thesis, contribution, ecosystem positioning).
- [RFC-0001 — Architecture & System Overview](RFC-0001-architecture.md): the model, the federation
  map, the round lifecycle (§7), the centralized / single-site / eval data flows, the staged plan.
- [RFC-0002 — The Latent Gauge & Frame-Anchored Aggregation](RFC-0002-gauge-and-aggregation.md): the
  gauge, the anchor variants, the Procrustes closed form $Q^\star=VU^\top$, the layers that the
  ablation ladder (§6) realizes, the $\lambda_{\text{anc}}$ trade-off, the frame-drift pointer (§9
  there).
- [RFC-0003 — Federated Training Protocol](RFC-0003-federated-protocol.md): rounds, DiLoCo $H$,
  communication, int8 quantization, DP pointer.
- [RFC-0008 — Model, Objective & Numerical Contracts](RFC-0008-model-objective-numerics.md): the
  encoder/predictor/objective and the numerical contract the eval consumes.
- [RFC-0009 — Configuration, Run Manifest & Reproducibility](RFC-0009-configuration-reproducibility.md):
  Hydra config composition for the ladder and sweeps, the `RunManifest`, seeding.
- [RFC-0010 — Checkpoint & Artifact Format](RFC-0010-artifact-checkpoint-format.md): hash-verified
  checkpoints the eval loads (`INV-CHECKPOINT-HASH`).
- [RFC-0012 — Differential Privacy Accounting](RFC-0012-differential-privacy.md): the reported
  $(\varepsilon,\delta)$ budget.
- [RFC-0013 — Coordinator & Participant Runtime](RFC-0013-coordinator-runtime.md): the federation runs
  that produce the evaluated checkpoints; Stage-C real-node re-runs.
- [RFC-0015 — Observability, Diagnostics & Telemetry](RFC-0015-observability-diagnostics.md): the
  metric names, the frame-drift emission record schema, the metrics JSONL sink, the redaction guard.
- Spec: [02-public-api.md](../spec/02-public-api.md) (`evaluate`/`Planner`/`frame_drift`/`procrustes_align` contracts),
  [03-data-model.md](../spec/03-data-model.md) (`EvalReport`/`FrameDriftReport` schemas), [05-observability.md](../spec/05-observability.md)
  (redaction), [07-testing-strategy.md](../spec/07-testing-strategy.md) (the test pyramid and tolerances),
  [08-performance-budget.md](../spec/08-performance-budget.md) (planning-cost and communication budgets).
- stable-worldmodel (galilai-group) — `world.evaluate`, the latent-MPC planners (CEM / iCEM / MPPI),
  the standardized environments and factors-of-variation.
- V-JEPA 2 (Assran et al., 2025) — the warm-start and the action-conditioned latent-MPC recipe that
  the planning metric follows.
- LeJEPA / LeWorldModel (Balestriero & LeCun; Maes, Le Lidec et al., 2026) — the SIGReg objective
  whose collapse behavior the effective-dimension metric guards.
- DiLoCo / OpenDiLoCo / INTELLECT (Douillard et al.; Prime Intellect) — the inner/outer optimizer the
  communication metric contextualizes.
