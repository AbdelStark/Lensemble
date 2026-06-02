# 10 — Glossary

Canonical terms for the Lensemble corpus. Each entry is defined exactly as the rest of the corpus
uses it, with a parenthetical link to the RFC or spec section that owns the term. Notation follows
[RFC-0002 §3-4](../rfcs/RFC-0002-gauge-and-aggregation.md) and the canonical symbol table; for the
authoritative type signatures see [03 — Data Model](03-data-model.md), for the error names see
[04 — Error Model](04-error-model.md), and for invariant IDs (`INV-*`) see the named-invariant set
threaded through both.

Terms are grouped by topic. Within each group they appear in the order a reader meets them when
building up the concept. A symbol index closes the document.

---

## Architecture & model

**JEPA (Joint-Embedding Predictive Architecture).** A self-supervised architecture that predicts in a
learned *latent* space rather than reconstructing raw pixels: an encoder maps an observation to an
embedding, and a predictor maps a context embedding (plus, here, an action) to the embedding of a
future observation. The training target is another embedding produced by the same network, not an
external label. Lensemble trains an action-conditioned JEPA and uses it as a latent world model for
planning ([RFC-0001 §2](../rfcs/RFC-0001-architecture.md)).

**V-JEPA 2.** The released foundation-scale video JEPA (Assran et al., 2025; 1.2B parameters,
pretrained on >1M hours of video) and its action-conditioned-predictor plus latent-MPC recipe.
Lensemble warm-starts its encoder from released V-JEPA 2 weights; that warm-start also serves as the
gauge anchor at round 0 ([RFC-0001 §2](../rfcs/RFC-0001-architecture.md);
[RFC-0008 design](../rfcs/RFC-0008-model-objective-numerics.md)).

**Action-conditioned predictor.** The latent predictor $g_\phi$: a compact transformer that predicts
future latents autoregressively, conditioned on an action embedding produced by the per-embodiment
action head. It follows the LeWM `ARPredictor` shape and consumes a WMCP `LatentState`
([RFC-0001 §2](../rfcs/RFC-0001-architecture.md); detailed in
[RFC-0008 design](../rfcs/RFC-0008-model-objective-numerics.md)).

**Encoder ($f_\theta$).** The video Vision Transformer mapping a video clip to $\mathbb{R}^{N\times d}$,
warm-started from V-JEPA 2 and co-trained under the objective (Fork B). It emits a WMCP-conformant
`LatentState` ([RFC-0001 §2](../rfcs/RFC-0001-architecture.md)).

**Fork A / Fork B.** The two federation regimes. **Fork B** (the target) co-trains encoder *and*
predictor end-to-end; this is the hard regime that opens the latent gauge and is the lead
contribution. **Fork A** (the documented safe-degrade fallback) freezes the warm-started shared
encoder and federates only the predictor; the frozen encoder is itself a shared frame, so the gauge
problem dissolves and the anchoring layers become unnecessary, at the cost of the end-to-end novelty
([RFC-0001 §7](../rfcs/RFC-0001-architecture.md);
[RFC-0002 Fork A fallback](../rfcs/RFC-0002-gauge-and-aggregation.md#fork-a-fallback)).

**Warm-start.** Initializing every participant's encoder from the same released V-JEPA 2 weights.
It buys foundation-scale credibility without an INTELLECT-class pretraining bill and, critically,
closes the gauge at $t{=}0$: round-0 encoder weights are hash-identical across participants
(`INV-WARMSTART-T0`). The round-0 encoder snapshot is frozen as the anchoring reference $f_{\text{ref}}$
([RFC-0002 §4](../rfcs/RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)).

**WMCP (World-Model Contract Protocol; WM-RFC-0001).** The shared latent/action interface — the
shape, dtype, and semantics every encoder must emit and every predictor must consume, plus the
action-conditioning interface every per-embodiment head must satisfy. It is the type-safety layer
that makes heterogeneous-embodiment federation well-posed: the explicit analogue of the fixed token
vocabulary that LLM federation gets for free. Conformance is a precondition for joining a federation
(`INV-WMCP`) ([RFC-0004 §6](../rfcs/RFC-0004-data-provenance.md);
[RFC-0007](../rfcs/RFC-0007-wmcp-latent-contract.md)).

**Latent state (`LatentState`).** The WMCP-governed encoder output: a tensor of shape $(N, d)$ with a
pinned dtype and a `wmcp_version` string, carrying the per-clip latent tokens. Every conforming
encoder emits it and every conforming predictor consumes it; a shape, dtype, or semantics mismatch
raises `ContractViolation` ([RFC-0007 design](../rfcs/RFC-0007-wmcp-latent-contract.md);
schema in [03 — Data Model](03-data-model.md)).

**Action head (action encoder, embodiment head; $h_\psi^{(c)}$).** The per-participant module mapping
participant $c$'s embodiment-specific action space into the shared latent-conditioning space, validated
against that embodiment's `ActionSpec`. Action heads are local: never broadcast and never aggregated,
because action spaces genuinely differ across embodiments (`INV-ACTIONHEAD-LOCAL`)
([RFC-0001 §2-3](../rfcs/RFC-0001-architecture.md);
[RFC-0007 design](../rfcs/RFC-0007-wmcp-latent-contract.md)).

**Embodiment.** A physical or simulated agent class with a specific action space and morphology
(for example a quadruped versus a 7-DoF arm). Different embodiments share the encoder and predictor
core but keep their own action head and `ActionSpec`; the WMCP contract is what lets them co-train one
model ([RFC-0001 §2](../rfcs/RFC-0001-architecture.md);
[RFC-0007](../rfcs/RFC-0007-wmcp-latent-contract.md)).

## The objective

**SIGReg (Sketched Isotropic Gaussian Regularization).** The LeJEPA / LeWM regularizer that pushes the
embedding marginal toward an isotropic Gaussian $\mathcal{N}(0, I_d)$ by projecting embeddings onto
random 1-D directions and matching each univariate marginal to a standard Gaussian via the Epps-Pulley
characteristic-function statistic. SIGReg removes the EMA target, stop-gradient, and teacher-student
machinery, which is exactly what makes it federation-friendly — there is no momentum-encoder state to
reconcile across participants. It is also the source of the latent gauge, because an isotropic Gaussian
target is itself basis-free ([RFC-0002 §1](../rfcs/RFC-0002-gauge-and-aggregation.md#1-background-the-objective-verbatim-from-rfc-0008);
[RFC-0008 design](../rfcs/RFC-0008-model-objective-numerics.md)).

**Cramér-Wold / Epps-Pulley characteristic-function statistic.** The statistical tools SIGReg rests on.
The *Cramér-Wold* device reduces matching a high-dimensional distribution to its target to matching all
of its 1-D projections; SIGReg samples those projections via the random sketch matrix. The *Epps-Pulley*
statistic then tests, per projected direction, whether the univariate marginal matches a standard
Gaussian by comparing empirical and target characteristic functions over a fixed set of integration
knots (reference default ~17 knots) ([RFC-0002 §1](../rfcs/RFC-0002-gauge-and-aggregation.md#1-background-the-objective-verbatim-from-rfc-0008);
defaults in [RFC-0003 §2](../rfcs/RFC-0003-federated-protocol.md#2-round-structure-diloco-outer-loop)).

**Sketch / random-projection matrix ($A$).** The matrix that projects $d$-dimensional embeddings onto a
batch of random 1-D directions for the SIGReg statistic (reference sketch dimension 64). All
participants in round $t$ derive the identical $A$ from the broadcast round sketch seed $s_t$, so every
participant minimizes the same regularizer rather than $C$ different empirical objectives
(`INV-SKETCH-CONSISTENCY`). A shared sketch fixes objective *consistency* but does not close the gauge
([RFC-0002 §3](../rfcs/RFC-0002-gauge-and-aggregation.md#3-layer-1--shared-sketch-matrix-objective-consistency-not-the-gauge-fix)).

**Function-space distillation (Layer 4).** The heterogeneity / instability fallback: rather than average
weights, aggregate *behaviors* — each participant emits predictions on the public probe, the coordinator
forms a frame-aligned consensus, and a global student distills it. It is gauge-invariant by construction
(it compares functions on shared inputs) and admits participants with different encoder sizes, at a
higher per-round cost; held in reserve ([RFC-0002 §6](../rfcs/RFC-0002-gauge-and-aggregation.md#6-layer-4--function-space-distillation-fallback--heterogeneity)).

## The latent gauge & frame anchoring

**Latent gauge.** The $O(d)$ rotational symmetry of the SIGReg-JEPA objective: the entire objective is
invariant under a global orthogonal rotation of the latent space, so nothing pins a shared coordinate
frame. Independently-updated participants therefore drift into mutually-rotated frames, and averaging
their weight matrices in parameter space averages features expressed in different bases — producing
neither participant's representation. This is the irreducible, JEPA-specific obstruction Lensemble
exists to control ([RFC-0002 §2](../rfcs/RFC-0002-gauge-and-aggregation.md#2-the-od-gauge-formally-preserved-verbatim)).

**$O(d)$ rotation symmetry.** The formal statement of the gauge. For any orthogonal $Q \in O(d)$, the
transform $f_\theta \mapsto Q f_\theta,\ g_\phi \mapsto Q g_\phi Q^{\top}$ leaves the objective
unchanged: the prediction loss depends only on inner products and distances, which $Q$ preserves, and
$\mathcal{N}(0, I)$ is invariant under $O(d)$, so every projected SIGReg marginal of $Qz$ is still
standard normal. The $O(d)$ latent rotation is the *extra* symmetry that anchored models (supervised
nets, LLMs with a fixed vocabulary) never see; the neuron-permutation symmetries present in any network
are a separate, lesser matter handled by the same alignment machinery
([RFC-0002 §2](../rfcs/RFC-0002-gauge-and-aggregation.md#2-the-od-gauge-formally-preserved-verbatim)).

**Frame anchor.** The mechanism that manufactures the shared coordinate frame the objective otherwise
lacks, so weight-averaging becomes valid again (Layer 2). It combines a shared warm-start (the gauge is
closed at $t{=}0$) with a light public-probe anchor loss that re-pins the frame during fine-tuning. The
probe is public data, so nothing private crosses a boundary; the same construction also keeps the
eventual proof-of-contribution circuit cheap ([RFC-0002 §4](../rfcs/RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)).

**Landmark anchoring (Variant A).** The recommended default anchor: choose $k \ge d$ generic landmark
points on the probe with fixed absolute targets $t_i = f_{\text{ref}}(p_i)$ from the round-0 encoder,
and penalize $\tfrac1k\sum_i \lVert f_\theta(p_i) - t_i\rVert^2$. Because $k \ge d$ generic absolute
constraints admit only $Q = I$ as the satisfying orthogonal map, this pins the frame while leaving all
other points free — "pin the frame, not the content." It needs no differentiable SVD, which is why it
is the safe starting point ([RFC-0002 §4](../rfcs/RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)).

**Rotational-drift penalty (Variant B).** The principled alternative anchor: decompose probe drift into a
*frame* part and a *content* part, compute the optimal Procrustes rotation $Q^\star$ between the probe
embeddings and $E_{\text{ref}}$, and penalize only the rotation, $\lVert Q^\star - I\rVert_F^2$.
Gradients flow through the differentiable SVD into the encoder, pushing the frame toward the reference
while leaving the post-alignment content residual unpenalized. It is the cleaner statement of "pin the
frame, not the content" but risks instability near degenerate singular values, which must be clamped or
conditioned ([RFC-0002 §4](../rfcs/RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)).

**Public probe ($\mathcal{P}$).** The small, fixed, content-hash-pinned set of public (licensed-for-
redistribution) data points that is the substrate for the frame anchor and for publicly-recomputable
alignment. It must be representative enough to anchor a meaningful frame (under-coverage weakens the
anchor; over-size raises per-round alignment cost), is governed openly, and includes the designated
$k \ge d$ landmark subset. Because it redefines the reference frame, changing it is a versioned
re-anchoring event; the probe hash equals the hash committed in `RoundOpen` (`INV-PROBE-PIN`)
([RFC-0004 §3](../rfcs/RFC-0004-data-provenance.md#3-the-public-probe-set-mathcalp)).

**Procrustes alignment.** The closed-form computation of the optimal orthogonal rotation $Q^\star$
aligning one set of probe embeddings to a reference: from the SVD $E_{\text{ref}}^\top f_\theta(\mathcal{P})
= U\Sigma V^\top$, the solution is $Q^\star = V U^\top$. It appears in three places — inside Variant B's
anchor loss, as the Layer-3 aggregation backstop (re-aligning each participant before averaging when
drift exceeds threshold), and as the per-pair frame-drift diagnostic. Because it is a deterministic
function of public-probe data and committed weights, it is publicly recomputable and needs no proof. A
near-degenerate or ill-conditioned SVD raises `DegenerateProcrustes`; the public API exposes it as
`procrustes_align(source, target) -> (Q*, residual)`
([RFC-0002 §5](../rfcs/RFC-0002-gauge-and-aggregation.md#5-layer-3--procrustes-re-alignment-at-aggregation-backstop)).

**Frame drift.** The empirical quantity at the center of the paper: the inter-participant Procrustes
residual (or mean rotation angle) on the public probe over training, computed per participant pair and
also against the global model. Naive `FedAvg` curves diverge as frames rotate apart; the anchored
configuration holds them flat. This is, to our knowledge, the first measurement of latent frame-drift
under federated self-supervision. When it exceeds the configured threshold the Layer-3 backstop fires;
beyond a hard limit the system raises `FrameDriftExceeded` ([RFC-0002 §9](../rfcs/RFC-0002-gauge-and-aggregation.md#9-the-frame-drift-diagnostic-the-headline-measurement);
[RFC-0005 §2](../rfcs/RFC-0005-evaluation.md#2-headline-diagnostic--latent-frame-drift); emission contract
in [RFC-0015 design](../rfcs/RFC-0015-observability-diagnostics.md)).

## Federated training

**DiLoCo.** The low-communication outer/inner optimizer Lensemble builds on (Douillard et al.;
OpenDiLoCo / INTELLECT, Prime Intellect). Each participant runs $H$ local inner steps, then a single
outer step synchronizes the resulting pseudo-gradients; communication happens only every $H$ steps
rather than every step. This is the engineering substrate for sovereignty, not the contribution
([RFC-0001 §4](../rfcs/RFC-0001-architecture.md);
[RFC-0003 §2](../rfcs/RFC-0003-federated-protocol.md#2-round-structure-diloco-outer-loop)).

**Inner / outer loop.** The two-level training topology. The **inner loop** is intra-participant
optimization (standard FSDP / tensor / context parallelism with AdamW) over local data only — the only
place the large-model-parallelism playbook applies, and not the contribution. The **outer loop** is the
inter-participant DiLoCo step that aggregates pseudo-gradients across the federation
([RFC-0001 §4](../rfcs/RFC-0001-architecture.md);
[RFC-0003 §2](../rfcs/RFC-0003-federated-protocol.md#2-round-structure-diloco-outer-loop)).

**Inner horizon ($H$).** The number of local inner steps each participant runs between outer
synchronizations (reference range $[50, 500]$; DiLoCo/INTELLECT use $H \approx 500$ for LLMs). Larger
$H$ means cheaper communication but more frame drift to anchor, so it is started small while
characterizing drift and tuned against the gauge ([RFC-0003 §2](../rfcs/RFC-0003-federated-protocol.md#2-round-structure-diloco-outer-loop)).

**Pseudo-gradient ($\Delta_c$).** Participant $c$'s contribution for a round: the difference between its
post-inner-loop parameters and the round's global parameters,
$\Delta_c = (\theta_c^{\text{local}}, \phi_c^{\text{local}}) - (\theta_t, \phi_t)$. DiLoCo treats the
whole $H$-step local update as a single "gradient." Only $\Delta_c$ (clipped, noised, and masked)
crosses the boundary; it carries its own L2 norm and is bound to exactly one dataset Merkle root
(`INV-COMMIT-BINDING`) ([RFC-0003 §3](../rfcs/RFC-0003-federated-protocol.md#3-the-pseudogradient-contract);
type in [03 — Data Model](03-data-model.md)).

**Outer Nesterov step.** The outer-loop parameter update: Nesterov momentum applied to the averaged
pseudo-gradient, $(\theta_{t+1}, \phi_{t+1}) = (\theta_t, \phi_t) - \eta_{\text{out}}\,
\mathrm{Nesterov}(\tfrac1C \sum_c \Delta_c)$. This step is on the aggregation path and MUST be a pure,
bitwise-reproducible function of the committed deltas, round seed, and prior global parameters
(`INV-AGG-DETERMINISM`); a non-deterministic reduction raises `NonDeterministicAggregation`
([RFC-0003 §2](../rfcs/RFC-0003-federated-protocol.md#2-round-structure-diloco-outer-loop)).

**Coordinator.** The role that orchestrates rounds, holds the canonical global model, and runs the outer
optimizer. It is untrusted with respect to raw data: treated as honest-but-curious in Phase 1 and as a
proving target in Phase 2. In Phase 1 it is a single point of trust and failure; Phase 2 makes its
aggregation provable ([RFC-0003 §1](../rfcs/RFC-0003-federated-protocol.md#1-roles);
runtime in [RFC-0013 design](../rfcs/RFC-0013-coordinator-runtime.md)).

**Participant.** The role that holds sovereign data, runs the local inner loop with intra-participant
parallelism, and emits pseudo-gradients. Raw data never leaves a participant boundary
(`INV-RESIDENCY`); only the privatized, masked $\Delta_c$ and the dataset commitment cross
([RFC-0003 §1](../rfcs/RFC-0003-federated-protocol.md#1-roles);
runtime in [RFC-0013 design](../rfcs/RFC-0013-coordinator-runtime.md)).

**Round.** One outer iteration of the protocol, progressing through the state machine
`OPEN -> COLLECTING -> AGGREGATING -> ALIGNING -> COMMITTING -> CLOSED` (with an `ABORTED` path):
broadcast of global parameters / sketch seed / probe hash, local optimization, privatized
pseudo-gradient release, secure aggregation, optional Procrustes backstop, the outer Nesterov step, and
the hash-commit of the new global model. Bounded by the `RoundOpen` and `RoundClose` messages; round
faults raise `RoundError` / `FaultToleranceExceeded`
([RFC-0003 §2, §7](../rfcs/RFC-0003-federated-protocol.md#2-round-structure-diloco-outer-loop);
state machine in [RFC-0013 design](../rfcs/RFC-0013-coordinator-runtime.md)).

## Privacy & secure aggregation

**Secure aggregation.** The protocol that lets the coordinator learn only the sum $\sum_c \Delta_c$ and
never an individual $\Delta_c$ (an individual update leaks more about a silo's data than the sum). The
reference design is pairwise-mask secure aggregation (Bonawitz-style) with threshold secret sharing for
dropout robustness; a TEE-based aggregator is a supported alternative. Falling below the dropout
threshold raises `SecureAggregationError`; masking must not introduce nondeterminism in the revealed sum
([RFC-0003 §5](../rfcs/RFC-0003-federated-protocol.md#5-secure-aggregation-requirement);
[RFC-0011](../rfcs/RFC-0011-secure-aggregation.md)).

**Differential privacy (DP).** The per-participant guarantee on a released pseudo-gradient: clip
$\Delta_c$ to a fixed norm, then add calibrated Gaussian noise, so a participant's contribution to a
round is privacy-bounded. The unit of privacy is a participant's per-round contribution (update-level
DP), not per-example DP-SGD inside the inner loop. The mechanism interacts with SIGReg variance and the
anchor term; joint calibration is a Stage-B experiment, not a default
([RFC-0003 §4](../rfcs/RFC-0003-federated-protocol.md#4-differential-privacy-protocol-level);
[RFC-0012](../rfcs/RFC-0012-differential-privacy.md)).

**$(\varepsilon, \delta)$.** The differential-privacy budget: $\varepsilon$ bounds the privacy loss and
$\delta$ the probability of exceeding it, accumulated over the planned number of rounds by the
accountant. When the budget is spent, training stops and the system raises `PrivacyBudgetExceeded`
([RFC-0003 §4](../rfcs/RFC-0003-federated-protocol.md#4-differential-privacy-protocol-level);
accounting in [RFC-0012 design](../rfcs/RFC-0012-differential-privacy.md)).

**Clip norm ($C_{\text{clip}}$).** The fixed L2 bound to which each pseudo-gradient is clipped before
noising: $\Delta_c \leftarrow \Delta_c \cdot \min(1, C_{\text{clip}}/\lVert\Delta_c\rVert)$. After
clipping and before noise, $\lVert\Delta_c\rVert \le C_{\text{clip}}$ is the enforced bound
(`INV-DP-BOUND`); it sets the sensitivity the noise is calibrated against
([RFC-0003 §4](../rfcs/RFC-0003-federated-protocol.md#4-differential-privacy-protocol-level)).

**Noise multiplier ($\sigma$).** The scale of the Gaussian noise relative to the clip norm: noise
$\mathcal{N}(0, \sigma^2 C_{\text{clip}}^2 I)$ is added to the clipped pseudo-gradient, with $\sigma$
calibrated to the target $(\varepsilon, \delta)$ over the planned rounds. It is jointly tuned with
$\lambda_{\text{sig}}$, $\lambda_{\text{anc}}$, and $C_{\text{clip}}$ in Stage B
([RFC-0003 §4](../rfcs/RFC-0003-federated-protocol.md#4-differential-privacy-protocol-level)).

**Honest-but-curious.** The Phase-1 trust assumption on the coordinator and aggregator: they follow the
protocol faithfully but may try to infer information from what they legitimately observe. Secure
aggregation and DP protect raw data under this model. Phase 2 strengthens the model to detect a
*malicious* coordinator and misreporting participants
([RFC-0006 §1](../rfcs/RFC-0006-verifiable-contribution.md#1-trust-model);
threat model in [06 — Security](06-security.md)).

## Data & provenance

**Sovereign data.** The premise: each participant's raw interaction trajectories are siloed by IP,
privacy, or safety and cannot be pooled. Federated training is the access strategy; the data stays put
and only model deltas move ([README thesis](../../README.md);
[RFC-0004 §1](../rfcs/RFC-0004-data-provenance.md#1-per-participant-data-layer)).

**Data residency.** The enforced property that raw observations, actions, and embeddings of private data
never cross a trust boundary — only pseudo-gradients leave. Every local dataset carries a non-exportable
flag and the training process must refuse to emit raw data or its embeddings (`INV-RESIDENCY`). A breach
raises `ResidencyViolation`, which is security-critical and fail-closed: never caught and ignored
([RFC-0004 §2](../rfcs/RFC-0004-data-provenance.md#2-residency-the-sovereignty-guarantee-inv-residency); enforcement in
[06 — Security](06-security.md), redaction in [05 — Observability](05-observability.md)).

**Merkle root ($R_c$).** The root of the binary Merkle tree a participant builds over the content hashes
of its episodes, committed before an epoch via the `Commitment` message. It identifies the dataset that
fed a contribution without revealing the data, and supports inclusion proofs for Phase 2. The canonical
hash is SHA-256 in Phase 1 ([RFC-0004 §4](../rfcs/RFC-0004-data-provenance.md#4-provenance-commitments-the-bridge-to-phase-2);
construction in [RFC-0014 design](../rfcs/RFC-0014-provenance-commitments.md)).

**Dataset commitment (`DatasetCommitment`).** The committed object binding a participant's data to a
round: the Merkle root $R_c$, the episode count, and WMCP metadata. Every released $\Delta_c$ is bound
to exactly one $R_c$ (`INV-COMMIT-BINDING`); a contributed update bound to the wrong root is rejected
with `CommitmentMismatch`. Produced by the public API `commit_dataset(dataset) -> DatasetCommitment`
([RFC-0004 §4](../rfcs/RFC-0004-data-provenance.md#4-provenance-commitments-the-bridge-to-phase-2);
[RFC-0014 design](../rfcs/RFC-0014-provenance-commitments.md)).

**Contribution accounting.** The append-only record of which committed dataset fed which round: per
round, the set of contributing participants, their committed roots, and the resulting global-model hash
(a `ContributionRecord` in the `ContributionLedger`). In Phase 1 it provides credit/governance tracking
and tamper-evidence; in Phase 2 it is the audit substrate the verifiable layer formalizes
([RFC-0004 §5](../rfcs/RFC-0004-data-provenance.md#5-contribution-accounting);
ledger in [RFC-0014 design](../rfcs/RFC-0014-provenance-commitments.md)).

## Evaluation & planning

**Latent MPC (model-predictive control).** The planning and evaluation procedure: use the trained model
as the world model and minimize an $L_1$ goal-energy in latent space over candidate action sequences,
re-planning each step, exactly as `stable-worldmodel` provides. Downstream planning success via latent
MPC is the metric that ultimately matters; representation-probe accuracy is supporting evidence
([RFC-0001 §2](../rfcs/RFC-0001-architecture.md);
[RFC-0005 §3](../rfcs/RFC-0005-evaluation.md#3-downstream-metric--planning-success)).

**CEM / iCEM / MPPI.** The sampling-based planners used inside latent MPC — the Cross-Entropy Method,
its improved variant, and Model-Predictive Path Integral control. They search candidate action
sequences against the latent goal-energy; the planner choice is an evaluation knob, not part of the
training contribution ([RFC-0005 §3](../rfcs/RFC-0005-evaluation.md#3-downstream-metric--planning-success)).

**Goal-energy.** The $L_1$ distance in latent space between a predicted future state and the encoded
goal specification (goal-image), which the latent-MPC planner minimizes to choose actions
([RFC-0005 §3](../rfcs/RFC-0005-evaluation.md#3-downstream-metric--planning-success)).

**Effective dimension.** The metric guarding against representation collapse: the effective rank of the
embedding covariance (from its eigenspectrum), emitted as `gauge/effective_dim`. It catches silent
*partial* collapse that a downstream success rate alone might mask
([RFC-0005 §4](../rfcs/RFC-0005-evaluation.md#4-supporting-metrics); emission in
[RFC-0015 design](../rfcs/RFC-0015-observability-diagnostics.md)).

**Representation collapse.** The failure mode where the encoder maps inputs into a low-rank or constant
subspace, destroying useful structure. SIGReg prevents it locally by enforcing an isotropic marginal;
averaging across mutually-rotated frames can re-introduce it ("collapse re-entry"), which the
effective-dimension metric and the frame anchor together guard against
([RFC-0002 §1, §2.1](../rfcs/RFC-0002-gauge-and-aggregation.md#21-three-failures-that-compound-it);
[RFC-0005 §4](../rfcs/RFC-0005-evaluation.md#4-supporting-metrics)).

**Ablation ladder.** The core experiment: a sequence of configurations each adding one mechanism —
(1) naive end-to-end `FedAvg`; (2) + shared sketch matrix; (3) + Procrustes align-then-average;
(4) + frame-anchor loss (the expected recommended configuration); (5) + function-space distillation.
All three metric families (frame drift, MPC success, collapse) are reported at each rung; the layers
are additive, so the ladder *is* the rollout plan for the gauge machinery
([RFC-0005 §6](../rfcs/RFC-0005-evaluation.md#6-ablation-ladder-the-core-experiment)).

**Non-IID.** Heterogeneity across silos: each participant's data is partitioned by factor-of-variation or
by embodiment so its empirical embedding marginal differs from the others'. Non-IID marginals are one of
the failures that compound the gauge problem (SIGReg constrains a different marginal per participant).
Stage B sweeps from near-IID to strongly non-IID using controlled factors of variation
([RFC-0002 §2.1](../rfcs/RFC-0002-gauge-and-aggregation.md#21-three-failures-that-compound-it);
[RFC-0005 §7](../rfcs/RFC-0005-evaluation.md#7-non-iid-severity--scale-sweeps)).

**Factors of variation.** The controlled axes of `stable-worldmodel` (environment, viewpoint, and other
generative factors) used to partition data across silos for *reproducible* non-IID heterogeneity and to
hold out factors at evaluation time ([RFC-0005 §7](../rfcs/RFC-0005-evaluation.md#7-non-iid-severity--scale-sweeps)).

## Verifiability (Phase 2)

**Proof-ready.** The discipline of building Phase 1 so the Phase-2 proofs require no rework. The required
properties are: deterministic aggregation (`INV-AGG-DETERMINISM`), hash-committed model versions
(`INV-CHECKPOINT-HASH`), episode hashing and Merkle roots from day one, a content-hash-pinned public
probe (`INV-PROBE-PIN`), and a reproducible outer step. These are inexpensive engineering disciplines in
scope for v0.1-v1.0, distinct from the actual proofs (Phase 2 / Stage D, out of v1.0 scope)
([RFC-0006 §3](../rfcs/RFC-0006-verifiable-contribution.md#3-phase-1-proof-ready-requirements-cheap-to-honor-now)).

**Function-space distillation.** *(See "The objective" group.)* Listed here as well because its
gauge-invariance is what makes behavior-level aggregation an option when weight-space aggregation is
unstable ([RFC-0002 §6](../rfcs/RFC-0002-gauge-and-aggregation.md#6-layer-4--function-space-distillation-fallback--heterogeneity)).

**TEE (Trusted Execution Environment).** A hardware-isolated enclave whose attestation is the pragmatic
proxy for proving that an inner training step executed faithfully. It is a weaker trust assumption than a
SNARK but far cheaper than proving training in zero knowledge, so Lensemble reserves it for the heavy,
data-touching inner step while proving the cheap, near-linear aggregation
([RFC-0006 §2, §5](../rfcs/RFC-0006-verifiable-contribution.md#2-the-provable-surface-and-what-stays-cheap)).

**Circle-STARK / Stwo.** Stwo is the Circle-STARK prover used in Phase 2 to prove that aggregation was
computed correctly over the submitted deltas. The proof is tractable precisely *because* the anchored
frame keeps aggregation a plain weighted-average / Nesterov step — a near-linear operation with a small
circuit. This synergy (the gauge fix is also the verifiability enabler) is the structural leverage of the
design ([RFC-0006 §2, §4](../rfcs/RFC-0006-verifiable-contribution.md#2-the-provable-surface-and-what-stays-cheap)).

## Symbol index

| Symbol | Term | Group |
|---|---|---|
| $d$ | latent embedding dimension | model |
| $N$ | latent tokens per clip | model |
| $f_\theta$ | encoder | model |
| $g_\phi$ | action-conditioned predictor | model |
| $h_\psi^{(c)}$ | per-participant action head | model |
| $f_{\text{ref}}$ | round-0 reference encoder (warm-start) | frame anchoring |
| $C$ | participant count ($c$ = participant index) | federation |
| $H$ | inner horizon | federation |
| $A$ | SIGReg sketch / random-projection matrix | objective |
| $s_t$ | round sketch seed | objective |
| $\mathcal{P}$ | public probe set ($p_i$ its points) | frame anchoring |
| $t_i$ | landmark target $= f_{\text{ref}}(p_i)$ (Variant A) | frame anchoring |
| $E_{\text{ref}}$ | reference probe embeddings $= f_{\text{ref}}(\mathcal{P})$ | frame anchoring |
| $Q \in O(d)$ | gauge rotation; $Q^\star$ optimal Procrustes rotation | latent gauge |
| $\lambda_{\text{sig}}, \lambda_{\text{anc}}, \lambda_{\text{pred}}$ | SIGReg / anchor / prediction loss weights | objective |
| $\Delta_c$ | pseudo-gradient of participant $c$ | federation |
| $\eta_{\text{out}}$ | outer-optimizer (Nesterov) learning rate | federation |
| $R_c$ | dataset Merkle root committed by participant $c$ | provenance |
| $(\varepsilon, \delta)$ | differential-privacy budget | privacy |
| $\sigma$ | DP noise multiplier | privacy |
| $C_{\text{clip}}$ | DP clip norm | privacy |

## References

- [00 — Overview](00-overview.md), [01 — Architecture](01-architecture.md),
  [03 — Data Model](03-data-model.md), [04 — Error Model](04-error-model.md),
  [05 — Observability](05-observability.md), [06 — Security](06-security.md).
- [RFC-0001 Architecture & System Overview](../rfcs/RFC-0001-architecture.md);
  [RFC-0002 The Latent Gauge & Frame-Anchored Aggregation](../rfcs/RFC-0002-gauge-and-aggregation.md);
  [RFC-0003 Federated Training Protocol](../rfcs/RFC-0003-federated-protocol.md);
  [RFC-0004 Data, Sovereignty & Provenance](../rfcs/RFC-0004-data-provenance.md);
  [RFC-0005 Evaluation & Benchmark Protocol](../rfcs/RFC-0005-evaluation.md);
  [RFC-0006 Verifiable Contribution](../rfcs/RFC-0006-verifiable-contribution.md).
- [RFC-0007 WMCP Latent Contract](../rfcs/RFC-0007-wmcp-latent-contract.md);
  [RFC-0008 Model, Objective & Numerics](../rfcs/RFC-0008-model-objective-numerics.md);
  [RFC-0011 Secure Aggregation](../rfcs/RFC-0011-secure-aggregation.md);
  [RFC-0012 Differential Privacy](../rfcs/RFC-0012-differential-privacy.md);
  [RFC-0013 Coordinator & Participant Runtime](../rfcs/RFC-0013-coordinator-runtime.md);
  [RFC-0014 Provenance Commitments](../rfcs/RFC-0014-provenance-commitments.md);
  [RFC-0015 Observability & Diagnostics](../rfcs/RFC-0015-observability-diagnostics.md).
- External: V-JEPA 2 (Assran et al., 2025); LeJEPA / LeWorldModel (Balestriero & LeCun; Maes,
  Le Lidec et al., 2026); stable-worldmodel (galilai-group); DiLoCo / OpenDiLoCo / INTELLECT
  (Douillard et al.; Prime Intellect); Bonawitz et al. (secure aggregation); Project Tapestry
  (AI Alliance); Stwo (Circle-STARK prover).
