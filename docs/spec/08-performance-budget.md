# 08 — Performance Budget

This section states the latency, throughput, memory, and communication **budgets** for Lensemble,
the **profiling plan** that measures them, and the **CI perf smoke** that guards against regression.

A budget is a target the system commits to validate at a named Stage ([conventions §12](conventions.md#12-milestones-and-stages)), not a measured
guarantee. Every numeric target below is annotated with the Stage that validates it and is a planning
band, not a benchmark result. None of these numbers is a measured datum; they are engineering
budgets derived from the model shapes (ViT-L/~300M, 1.2B target — [RFC-0001 §2](../rfcs/RFC-0001-architecture.md)),
the DiLoCo communication schedule ([RFC-0003 §2](../rfcs/RFC-0003-federated-protocol.md)), and the
numerical contract ([conventions §9](conventions.md#9-determinism-dtype-device)). Where the source RFCs give no number, the budget is left as an
`OPEN QUESTION:` with a measuring Stage rather than fabricated.

Rationale for budgets lives in [RFC-0001 §4](../rfcs/RFC-0001-architecture.md) (training topology) and
[RFC-0003 §2, §6, §8](../rfcs/RFC-0003-federated-protocol.md) (round structure, compression, reference
parameters). Stable contracts for the determinism this section depends on live in [conventions §9](conventions.md#9-determinism-dtype-device).

## 1. Budget philosophy and how to read this section

A budget here has four parts, always stated together:

| Field | Meaning |
|---|---|
| **Metric** | The canonical observability metric that measures it (see [05 — Observability](05-observability.md), `loss/*`, `gauge/*`, `fed/*`, `dp/*`, `eval/*`). |
| **Budget band** | A range, not a point. The lower edge is "acceptable"; the upper edge is "needs investigation". |
| **Validating Stage** | The milestone ([conventions §12](conventions.md#12-milestones-and-stages)) at which the budget is measured and either confirmed or revised. |
| **Failure response** | What happens when a measurement exceeds the band: revise the budget (documented in the run's `RunManifest`), investigate a regression, or raise an error if an invariant is touched. |

Budgets are not enforced as runtime assertions except where they coincide with a named invariant
([conventions §7](conventions.md#7-named-invariants)) — for example, the determinism self-check on the aggregation path raises
`NonDeterministicAggregation` ([04 — Error Model](04-error-model.md)), independent of any wall-time
budget. A wall-time overrun is a regression signal, not an error.

RISK: hardware drift. Budget bands assume "a single modern data-center GPU" (the class used in
[RFC-0001 §Migration / Rollout](../rfcs/RFC-0001-architecture.md#migration--rollout) Stage A, "handful of GPUs, days"). The corpus does not
pin a specific accelerator SKU, so absolute throughput numbers will move with hardware.
Resolution plan: every reported figure is recorded with the hardware string in the `RunManifest`
(hardware/env fields, [RFC-0009 §Proposed Design](../rfcs/RFC-0009-configuration-reproducibility.md)),
so budgets are compared only within a fixed hardware class. Validating Stage: A.

## 2. Compute budgets — single-site (Stage A)

Stage A is the centralized upper bound: single-site, warm-started ViT-L/~300M end-to-end SIGReg + AC
predictor on pooled robot data, with latent-MPC evaluation ([RFC-0001 §Migration / Rollout](../rfcs/RFC-0001-architecture.md#migration--rollout),
[conventions §12](conventions.md#12-milestones-and-stages) v0.1). The compute budgets below are the inner-loop budgets; they are the same inner loop that
runs inside each participant during federation (the inner/outer split of
[RFC-0001 §4](../rfcs/RFC-0001-architecture.md)).

| Budget | Metric | Band | Validating Stage | Failure response |
|---|---|---|---|---|
| Training throughput, ViT-L/~300M, bf16 forward / fp32 accumulation | derived from `fed/round_seconds` and inner step count; raw step time logged | tens to low-hundreds of clip-samples/s on one modern GPU | A | revise band per hardware class; investigate if below lower edge |
| Encoder forward latency $f_\theta(\text{clip})$ | step-time component (logged scalar) | single-clip forward in the tens-of-ms range on GPU; CPU-fallback path far slower, used only for CI | A | CPU path is for correctness, not throughput; see §7 |
| Predictor step $g_\phi$ (one autoregressive latent step) | step-time component | small relative to encoder forward (compact transformer, [RFC-0001 §2](../rfcs/RFC-0001-architecture.md)) | A | profile if it dominates the step (§6) |
| Objective overhead (SIGReg + anchor) | `loss/sigreg`, `loss/anchor` emission cadence; profiler span | SIGReg with sketch dim 64 and ~17 Epps–Pulley knots ([RFC-0003 §8](../rfcs/RFC-0003-federated-protocol.md)) is a low-rank projection plus a per-direction univariate statistic — small relative to the ViT forward | A | if SIGReg dominates, reduce knots/sketch dim within the LeJEPA defaults |

These numbers are bands, not guarantees. The Stage-A run records actual throughput against the
hardware string; the band is then tightened in the `RunManifest` for that hardware class.

OPEN QUESTION: the precise Stage-A throughput band for ViT-L/~300M. The source RFCs commit only to
"handful of GPUs, days" ([RFC-0001 §Migration / Rollout](../rfcs/RFC-0001-architecture.md#migration--rollout)); no samples/s figure is
published. Owner @AbdelStark. Resolution path: measured during Stage A (v0.1), recorded in the run
manifest, and the band above replaced with the measured value ± a tolerance.

## 3. Memory budgets — ViT-L vs the 1.2B target

The compute dtype contract ([conventions §9](conventions.md#9-determinism-dtype-device)) is bf16 forward with fp32 master weights and fp32 loss/statistic
accumulation. Memory budgets follow from this contract and the parameter counts in
[RFC-0001 §2](../rfcs/RFC-0001-architecture.md#2-module-map-reference-implementation) and the staged plan ([RFC-0001 §Migration / Rollout](../rfcs/RFC-0001-architecture.md#migration--rollout)).

| Configuration | Dominant memory terms | Budget posture | Validating Stage |
|---|---|---|---|
| ViT-L/~300M (Stage A, and per-participant inner model in Stage B) | fp32 master weights + bf16 working copy + AdamW optimizer state (two moments, fp32) + activations | fits a single modern GPU without model parallelism; this is the regime where the inner loop is data-parallel only | A, B |
| 1.2B target (V-JEPA-2 class, [RFC-0001 §Migration / Rollout](../rfcs/RFC-0001-architecture.md#migration--rollout) Stage E) | same terms scaled ~4x in parameters; optimizer state grows with parameters | requires intra-participant FSDP / tensor / context parallelism ([RFC-0001 §4](../rfcs/RFC-0001-architecture.md)) — the only place large-model parallelism applies | E (out of v1.0 scope, [conventions §12](conventions.md#12-milestones-and-stages)) |

The inner master-weights + optimizer-state footprint for AdamW is approximately
`params * (4 fp32 master + 2 bf16 working + 8 AdamW moments)` bytes before activations; for ViT-L this
is single-GPU-resident, for the 1.2B target it is the reason FSDP2 (torch `>=2.4,<3`, [conventions §11](conventions.md#11-external-dependencies)) is
required. Activation memory is set by clip length, token count `N`, and latent dimension `d` ([conventions §2](conventions.md#2-mathematical-notation)),
and is the term most sensitive to the configured `Window.num_steps` ([03 — Data Model](03-data-model.md)).

Note on what does NOT enter the shared-artifact memory budget: per-embodiment action heads
$h_\psi^{(c)}$ are local and never aggregated (`INV-ACTIONHEAD-LOCAL`, [conventions §7](conventions.md#7-named-invariants), enforced in
`lensemble.federation`); they add to a participant's local memory but never to the cross-boundary
communication budget of §4.

OPEN QUESTION: the FSDP2 sharding configuration and activation-checkpointing policy for the 1.2B
target. Owner @AbdelStark. Resolution path: Stage E is out of v1.0 scope ([conventions §12](conventions.md#12-milestones-and-stages)); captured as future
work in the tracker, not as an implementable v1.0 issue. The v1.0 budget covers ViT-L only.

## 4. Communication budget — the DiLoCo efficiency claim

The communication-efficiency claim is the DiLoCo property: a participant communicates only every $H$
inner steps, not every step ([RFC-0003 §2](../rfcs/RFC-0003-federated-protocol.md)). This is what makes
sovereign federation tractable against the cost of moving data or gradients every step.

### 4.1 What crosses the boundary

Per round, per participant, exactly one pseudo-gradient $\Delta_c$ crosses
($\Delta_c = (\theta_c^{\text{local}},\phi_c^{\text{local}}) - (\theta_t,\phi_t)$, [conventions §2](conventions.md#2-mathematical-notation),
[RFC-0003 §2 step 3](../rfcs/RFC-0003-federated-protocol.md)). The federation map
([RFC-0001 §3](../rfcs/RFC-0001-architecture.md)) fixes which parameters are in $\Delta_c$: the encoder
backbone $f_\theta$ and the predictor core $g_\phi$ (both federated), and nothing else. Action heads do
not cross (`INV-ACTIONHEAD-LOCAL`); raw observations, actions, and private embeddings never cross
(`INV-RESIDENCY`, [conventions §7](conventions.md#7-named-invariants), enforced in `lensemble.data.residency`; a violation raises
`ResidencyViolation`, fail-closed — [04 — Error Model](04-error-model.md),
[06 — Security](06-security.md)).

### 4.2 Bytes per round

| Quantity | Metric | Budget | Validating Stage |
|---|---|---|---|
| Per-participant bytes per round, full precision | `fed/comm_bytes` | `|θ_federated| + |φ_federated|` parameters at the wire dtype; for ViT-L this is the encoder+predictor delta, full fp32 ≈ `4 * num_federated_params` bytes | B |
| Per-participant bytes per round, int8 quantized | `fed/comm_bytes`, `fed/quant_ratio` | ~`1 * num_federated_params` bytes + scale metadata; ~4x reduction vs fp32, per INTELLECT-1 int8 all-reduce ([RFC-0003 §6](../rfcs/RFC-0003-federated-protocol.md)) | B |
| Total communication over a run | `fed/comm_bytes` summed over rounds; `fed/round_seconds` for rounds count | `bytes_per_round * num_rounds * C`; rounds scale as `total_inner_steps / H` | B |
| Quantization ratio | `fed/quant_ratio` | target ~0.25 (int8 vs fp32) with a bounded round-trip error ([RFC-0003 testing](../rfcs/RFC-0003-federated-protocol.md)) | B |

The headline efficiency framing: against a hypothetical centralized step-synchronous baseline that
would communicate every inner step, DiLoCo communicates once per $H$ steps, so the communication budget
is reduced by a factor of approximately $H$ (modulo the per-round overhead of secure aggregation and
the outer step). With $H \in [50,500]$ ([RFC-0003 §8](../rfcs/RFC-0003-federated-protocol.md)) this is a
50x–500x reduction in synchronization frequency. The number is reported, not asserted: it is computed
from `fed/comm_bytes` (total bytes moved) and the rounds count, and contextualized against the
centralized pooled cost in the evaluation communication metric
([RFC-0005 §4](../rfcs/RFC-0005-evaluation.md)).

### 4.3 How it is measured

The communication budget is measured by an explicit **comms accountant** (§6), not inferred from
network counters. The accountant sums the serialized byte length of every $\Delta_c$ that the masking
layer ([RFC-0011 secure aggregation](../rfcs/RFC-0011-secure-aggregation.md)) accepts for transmission,
keyed by round and participant, and emits `fed/comm_bytes`. Masking adds per-pair mask material; the
accountant counts the on-wire bytes including mask overhead so the budget reflects real bandwidth, not
just payload.

RISK: the $H$-vs-drift coupling. Larger $H$ reduces communication but increases frame drift that the
gauge machinery must absorb ([RFC-0003 §2](../rfcs/RFC-0003-federated-protocol.md),
[RFC-0002 §4](../rfcs/RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)). The communication budget cannot be
optimized in isolation from the drift budget of §5. Resolution plan: the $H$ schedule starts small
while drift is characterized (Stage B), and the joint $(H, \lambda_{\text{anc}})$ operating point is an
ablation output ([RFC-0005 §6](../rfcs/RFC-0005-evaluation.md)). Validating Stage: B.

## 5. Aggregation, outer-step, and gauge budgets

### 5.1 Outer step wall time

The outer step is the Nesterov update on the averaged pseudo-gradient
([RFC-0003 §2 step 7](../rfcs/RFC-0003-federated-protocol.md), [conventions §2](conventions.md#2-mathematical-notation)). It is a fixed-cost reduction over
$C$ deltas plus a momentum update over the federated parameter set.

| Budget | Metric | Band | Validating Stage | Failure response |
|---|---|---|---|---|
| Outer-step wall time | component of `fed/round_seconds` | small relative to a round's inner compute (one reduction + one optimizer step over `num_federated_params`); should not dominate `fed/round_seconds` | B | if it dominates, profile the reduction (§6); never trade determinism for speed (`INV-AGG-DETERMINISM`, [conventions §9](conventions.md#9-determinism-dtype-device)) |

`INV-AGG-DETERMINISM` ([conventions §7](conventions.md#7-named-invariants), [conventions §9](conventions.md#9-determinism-dtype-device)) constrains the outer step: it MUST be a pure, bitwise-reproducible
function of (committed deltas, round seed, prior global params), with fixed reduction order and fp32
(or fp64) summation, no atomics. The aggregation path therefore cannot use nondeterministic fast
reductions to meet a wall-time budget. A determinism self-check runs each outer step
([RFC-0003 §7](../rfcs/RFC-0003-federated-protocol.md#7-determinism-concurrency-error-propagation), [conventions §9](conventions.md#9-determinism-dtype-device)); on failure it raises
`NonDeterministicAggregation` ([04 — Error Model](04-error-model.md)), and the round aborts and
recomputes — wall time is subordinate to determinism, which is proof-readiness
([RFC-0006 §5](../rfcs/RFC-0006-verifiable-contribution.md)).

### 5.2 Gauge-diagnostic cost

The frame-drift diagnostic is the headline empirical artifact
([RFC-0002 §9](../rfcs/RFC-0002-gauge-and-aggregation.md#9-the-frame-drift-diagnostic-the-headline-measurement),
[RFC-0005 §2](../rfcs/RFC-0005-evaluation.md)). It computes, per round, for each participant pair, the
optimal Procrustes rotation between probe embeddings on the public probe $\mathcal{P}$.

| Budget | Metric | Band | Validating Stage | Failure response |
|---|---|---|---|---|
| Per-round diagnostic cost | derived from `gauge/drift_angle_deg`, `gauge/procrustes_residual` emission | one encoder forward over the probe set per participant ($f_\theta(\mathcal{P})$) plus an $O(C^2)$ set of $d \times d$ Procrustes SVDs | B | sample pairs when $C^2$ is large (see RISK) |
| Effective-dim metric | `gauge/effective_dim` | one covariance eigenspectrum over probe embeddings per participant; cheap relative to the encoder forward | B | emitted every round to catch silent collapse ([RFC-0005 §4](../rfcs/RFC-0005-evaluation.md)) |

The probe forward dominates the diagnostic cost; the Procrustes SVDs operate on $d \times d$ matrices
([conventions §2](conventions.md#2-mathematical-notation)) and are negligible per pair. The cost driver is the pair count.

RISK: $O(C^2)$ pair-wise drift at large participant count $C$. Per-round all-pairs drift is quadratic in
$C$. Resolution plan: sample a fixed subset of pairs, or report drift against the global model only
(linear in $C$), as specified in the diagnostic emission contract
([RFC-0015 §Proposed Design](../rfcs/RFC-0015-observability-diagnostics.md)). The effective-dim metric
remains linear in $C$. Owner @AbdelStark, Stage B.

The effective-dim metric (`gauge/effective_dim`) is the silent-collapse guard: SIGReg prevents collapse
during co-training, but a success-rate metric alone can mask partial collapse
([RFC-0005 §4](../rfcs/RFC-0005-evaluation.md)). Emitting effective dimension every round is the
profiling-plan tripwire for representation collapse (§6).

## 6. Profiling plan

Three instruments, each tied to a budget above:

1. **Compute profiler — torch profiler.** Profiles the inner loop (encoder forward, predictor step,
   SIGReg/anchor overhead, backward) on a representative Stage-A config. Output identifies the dominant
   span so a throughput-budget overrun (§2) routes to the right kernel. Run ad hoc during Stage A
   bring-up and on any throughput regression; not run in CI (§7).
2. **Comms accountant — explicit byte counter.** The communication budget (§4) is measured by counting
   serialized $\Delta_c$ bytes including mask overhead, keyed by round and participant, emitted as
   `fed/comm_bytes` and `fed/quant_ratio` ([05 — Observability](05-observability.md),
   [RFC-0015](../rfcs/RFC-0015-observability-diagnostics.md)). This is authoritative for the DiLoCo
   efficiency claim because it counts payload deterministically rather than depending on network-stack
   counters.
3. **Collapse tripwire — effective-dim metric.** `gauge/effective_dim`
   ([RFC-0005 §4](../rfcs/RFC-0005-evaluation.md)) is emitted every round to catch silent representation
   collapse that throughput and loss curves would not reveal. A sustained drop in effective dimension is
   a correctness signal, investigated before any performance tuning.

All three feed the metrics JSONL sink ([05 — Observability](05-observability.md),
[RFC-0015](../rfcs/RFC-0015-observability-diagnostics.md)); none emits raw observations, actions, or
embeddings of private data — only norms, shapes, counts, hashes, and scalar metrics, per the redaction
rule (`INV-RESIDENCY`, [conventions §7](conventions.md#7-named-invariants)). A profiling instrument that attempted to dump activations of private
data across a boundary would hit the residency guard and raise `ResidencyViolation`
([06 — Security](06-security.md)).

## 7. CI performance smoke

CI runs a tiny synthetic config on the CPU fallback path ([conventions §9](conventions.md#9-determinism-dtype-device): "Tests must pass on CPU"; no large
downloads in CI — [07 — Testing Strategy](07-testing-strategy.md)). Its purpose is regression detection,
not absolute-throughput measurement.

| CI smoke check | What it asserts | Failure response |
|---|---|---|
| Wall-time ceiling on a tiny end-to-end round | a fixed toy config completes one inner+outer cycle within a generous CPU wall-time ceiling | a large overrun fails the job, flagging an algorithmic regression (e.g. an accidental per-step recomputation) |
| Comms-accountant smoke | `fed/comm_bytes` for the toy config equals the expected `bytes_per_round` for the toy parameter count (full precision and int8) | a mismatch fails the job, catching a serialization or quantization regression |
| Determinism check on the outer step | two runs of the toy aggregation produce bitwise-identical global params (`INV-AGG-DETERMINISM`) | a mismatch raises `NonDeterministicAggregation` and fails the job ([04 — Error Model](04-error-model.md)) |
| int8 quant round-trip error bound | the int8 quantized $\Delta_c$ reconstructs within the documented error bound ([RFC-0003 testing](../rfcs/RFC-0003-federated-protocol.md)) | an exceeded bound fails the job |

The wall-time ceiling is deliberately generous (CPU, tiny model): it catches order-of-magnitude
regressions, not micro-optimizations. Absolute throughput budgets (§2) are validated on GPU during
Stage A/B and recorded in the `RunManifest`, never in CI. This split keeps CI fast and download-free
while still trapping the regressions that matter — runaway compute, runaway communication, and any
break in aggregation determinism.

OPEN QUESTION: the exact CPU wall-time ceiling for the CI smoke. It must be loose enough to survive
CI-runner variance yet tight enough to catch a 10x regression. Owner @AbdelStark. Resolution path: set
empirically from the first green CI run's measured time plus a multiplier, recorded in the CI config;
revisit at v0.1 ([conventions §12](conventions.md#12-milestones-and-stages)). This couples to the CI gate list in
[07 — Testing Strategy](07-testing-strategy.md) and the release gates in
[09 — Release & Versioning](09-release-and-versioning.md).

## 8. Scaling expectations: ViT-L toward 1.2B

What grows from ViT-L/~300M to the 1.2B target ([RFC-0001 §Migration / Rollout](../rfcs/RFC-0001-architecture.md#migration--rollout), Stage E,
out of v1.0 scope per [conventions §12](conventions.md#12-milestones-and-stages)):

- **Compute-bound terms** (grow ~linearly with parameters and ~quadratically with token count `N`):
  encoder forward/backward, optimizer state, activation memory. These are absorbed by intra-participant
  parallelism (FSDP2 / tensor / context — [RFC-0001 §4](../rfcs/RFC-0001-architecture.md)); this is the
  standard distributed stack and the only place it applies. It is the inner loop, not the contribution.
- **Communication-bound terms** (grow ~linearly with the federated parameter count): the per-round
  $\Delta_c$ payload (§4). This is the term DiLoCo's every-$H$-steps schedule and int8 quantization
  exist to contain. As the model scales, the federation stays communication-bound on the outer loop, so
  the int8 quantization budget (`fed/quant_ratio` ~0.25) becomes more load-bearing, and $H$ may need to
  grow — which re-couples to the drift budget (§5, the $H$-vs-drift RISK).
- **Diagnostic cost** (grows with probe size and `d`, and $O(C^2)$ in participants): the per-pair
  Procrustes diagnostic (§5.2). The probe forward scales with the encoder, and `d` scales the SVD cost
  modestly.

The scaling story is intentionally bounded for v1.0: the v1.0 hardening milestone ([conventions §12](conventions.md#12-milestones-and-stages)) covers
ViT-L, Fork A fallback, and proof-ready guarantees, not the 1.2B own-pretrain. Stage E scaling is
captured as future work, not an implementable v1.0 issue.

OPEN QUESTION: the communication budget at 1.2B scale and the $H$ value that balances it against drift.
Owner @AbdelStark. Resolution path: the $(H, C)$ convergence-and-drift sweep is a Stage-B experiment
([RFC-0005 §7](../rfcs/RFC-0005-evaluation.md)); the scale step extends the key rungs at increasing
encoder size ([RFC-0005 §7](../rfcs/RFC-0005-evaluation.md)). The 1.2B point is Stage E (post-v1.0).

## 9. Budget-to-Stage summary

| Budget area | Primary metric(s) | Validating Stage ([conventions §12](conventions.md#12-milestones-and-stages)) |
|---|---|---|
| Single-site compute throughput / latency (§2) | `fed/round_seconds`, step-time scalars | A (v0.1) |
| Memory, ViT-L (§3) | (resident memory, recorded in manifest) | A, B (v0.1, v0.2) |
| Memory, 1.2B target (§3) | (resident memory) | E (post-v1.0, future work) |
| Communication bytes / round and DiLoCo efficiency (§4) | `fed/comm_bytes`, `fed/quant_ratio` | B (v0.2) |
| Outer-step wall time + determinism (§5.1) | `fed/round_seconds`, determinism self-check | B (v0.2) |
| Gauge-diagnostic + effective-dim cost (§5.2) | `gauge/procrustes_residual`, `gauge/drift_angle_deg`, `gauge/effective_dim` | B (v0.2) |
| CI perf smoke (§7) | toy `fed/comm_bytes`, wall-time ceiling, determinism check | A (v0.1), enforced thereafter |

Every band in this section is revised against measured data at its validating Stage and the measured
value recorded in the `RunManifest` ([RFC-0009](../rfcs/RFC-0009-configuration-reproducibility.md)). No
band is treated as a guarantee before the Stage that measures it.

## References

- Architecture, training topology, staged plan, model sizes: [RFC-0001 §2, §4](../rfcs/RFC-0001-architecture.md) and the staged plan ([RFC-0001 §Migration / Rollout](../rfcs/RFC-0001-architecture.md#migration--rollout))
- Round structure, DiLoCo $H$ schedule, int8 quantization, reference parameters: [RFC-0003 §2, §6, §8](../rfcs/RFC-0003-federated-protocol.md)
- Determinism, dtype, device contract: [conventions §9](conventions.md#9-determinism-dtype-device)
- Milestones and Stage mapping: [conventions §12](conventions.md#12-milestones-and-stages)
- Gauge diagnostic (headline measurement): [RFC-0002 §9](../rfcs/RFC-0002-gauge-and-aggregation.md#9-the-frame-drift-diagnostic-the-headline-measurement), [RFC-0005 §2, §4](../rfcs/RFC-0005-evaluation.md)
- Communication metric for the efficiency claim: [RFC-0005 §4](../rfcs/RFC-0005-evaluation.md)
- Observability metric taxonomy and emission contract: [05 — Observability](05-observability.md), [RFC-0015](../rfcs/RFC-0015-observability-diagnostics.md)
- Error model (`NonDeterministicAggregation`, `ResidencyViolation`): [04 — Error Model](04-error-model.md)
- Security and residency: [06 — Security](06-security.md)
- Testing strategy and CI gates: [07 — Testing Strategy](07-testing-strategy.md)
- Release and versioning gates: [09 — Release & Versioning](09-release-and-versioning.md)
- Configuration and `RunManifest` for hardware/env recording: [RFC-0009](../rfcs/RFC-0009-configuration-reproducibility.md)
- Secure aggregation (mask overhead in the comms budget): [RFC-0011](../rfcs/RFC-0011-secure-aggregation.md)
