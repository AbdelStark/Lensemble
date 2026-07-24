# 00 — Overview

This document is the entry point to the Lensemble specification corpus. It states what the project is,
what ships, and what does not. It is normative for scope and success criteria; rationale for individual
mechanisms lives in the referenced RFCs, and stable contracts live in the sibling spec sections. A new
contributor should be able to read this document and the index at the end and know which document to open
next without messaging the author.

## 1. Thesis

Lensemble is a research stack for executing and evaluating **joint encoder-and-predictor training of
an action-conditioned Joint-Embedding Predictive Architecture (JEPA) world model across
participants**. Each participant keeps its raw interaction data inside its boundary and releases model
deltas $\Delta_c$ plus protocol metadata. The implemented coordinator/participant stack can execute
this "Fork B" training path; that software capability is not evidence of a useful, gauge-stable
full-model result. Secure aggregation and effective differential privacy are normative Stage-C target
protections, with cryptographic contribution proofs deferred to a later research phase.

> **Current implementation boundary.** Raw observations, actions, private embeddings, and local
> action-head weights remain resident. The current Phase-3 coordinator nevertheless receives each
> released pseudo-gradient in plaintext and commits the outer step from those individual values.
> Simulated and software-TEE aggregation are post-commit equivalence checks
> (`secure_sum_consumed=false`), while masking reports an explicit fallback. Configured clipping and
> deterministic replay noise execute, but the replay seed is reconstructible and each report creates a
> fresh one-round accountant (`effective_dp=false`, `status=deterministic_replay_only`); this is neither
> claim-grade DP nor cumulative budget enforcement. Secure-sum-only visibility and a persistent
> pre-release accountant remain target contracts below, not claims about the live optimizer path.
> The checked-in full-model evidence also predates correction of the outer-update direction and the
> target-bound public-probe contract. It is retained as audit history and does not validate the
> corrected runtime; [issue #335](https://github.com/AbdelStark/Lensemble/issues/335) tracks the rerun.

The name: *l'ensemble* — "the whole / together" in French — and the ML *ensemble* (many models acting as
one), with **lens** in front: the perception encoder at the heart of a visual world model.

### 1.1 The one-paragraph thesis

A foundation-scale world model wants diverse embodied experience — robot fleets, manipulation labs,
driving stacks, egocentric video — but that data is siloed by IP, privacy, and safety, and cannot be
pooled. Federated training is the access strategy. The catch is specific to JEPA: its self-supervised
objective is **invariant under rotations of the latent space** (the SIGReg-JEPA loss $\mathcal{L}$ is
unchanged under the gauge transform $f_\theta \mapsto Qf_\theta,\ g_\phi \mapsto Qg_\phi Q^\top$ for any
$Q \in O(d)$). Independently-updated participants can therefore drift into mutually-rotated coordinate
frames, and naive weight-averaging can combine incompatible parameterizations — a failure mode that
fixed-output bases in supervised networks and LLMs constrain. Lensemble implements a shared encoder
warm-start (the round-0 reference $f_{\text{ref}}$, which closes the gauge at $t{=}0$ per
`INV-WARMSTART-T0`) plus a light public-probe **frame anchor** to study whether that control makes
aggregation stable. Constraints intended to support a possible future cryptographic layer are
specified separately in [RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md); the proofs themselves
are deferred.

<a id="12-the-unique-claim"></a>

### 1.2 The core research hypothesis

The core claim is a falsifiable research hypothesis: joint JEPA co-training exposes a latent-frame
compatibility problem at the federation boundary, and explicit anchoring may control it. The mechanism
and failure modes are specified in [RFC-0002](../rfcs/RFC-0002-gauge-and-aggregation.md); the
frame-drift diagnostic in
[RFC-0005 §2](../rfcs/RFC-0005-evaluation.md#2-headline-diagnostic--latent-frame-drift) is how the
hypothesis is tested. This corpus does not claim priority over all prior work, nor treat the existing
pre-correction evidence as validation of the corrected runtime.

## 2. Contribution

Three research directions, in priority order. The first two define the current implementation and
evaluation program; the third is deferred.

1. **Federated joint JEPA co-training.** The implementation can update both encoder and predictor
   instead of restricting federation to a frozen-encoder predictor. It provides a frame-anchoring
   mechanism ([RFC-0002 §4](../rfcs/RFC-0002-gauge-and-aggregation.md)) and a latent-frame diagnostic
   ([RFC-0005 §2](../rfcs/RFC-0005-evaluation.md#2-headline-diagnostic--latent-frame-drift)).
2. **Sovereign embodied data.** The federation premise applied to physical AI: interaction trajectories
   that cannot be moved. Raw data stays inside a participant boundary (`INV-RESIDENCY`); released model
   deltas cross. Secure aggregation ([RFC-0011](../rfcs/RFC-0011-secure-aggregation.md)) and effective
   differential privacy ([RFC-0012](../rfcs/RFC-0012-differential-privacy.md)) are the target protections
   for those deltas, not properties of the current plaintext coordinator path.
3. **Verifiable contribution (deferred).** RFC-0006 sketches future cryptographic attestation of
   aggregation and provenance. Current deterministic and hash-bound seams make that direction
   inspectable, but do not prove honest computation or guarantee no-rework integration
   ([RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md)).

Stated plainly: the project studies what happens when the encoder as well as the predictor moves at a
federated boundary. Whether the implemented gauge controls yield a useful, magnitude-stable, and
gauge-stable full-model result after the corrected outer update remains an open empirical question.

## 3. Where it sits in the ecosystem

Lensemble composes named ecosystem dependencies rather than reinventing them; its focus is the latent
gauge question and the federation discipline layered on top. Version constraints and rationale are in
[conventions §11](conventions.md#11-external-dependencies) as reproduced in [09 — Release & Versioning §8](09-release-and-versioning.md#8-supported-runtimes-and-support-window).

- **V-JEPA 2** (Assran et al., 2025) — the warm-start encoder and the action-conditioned-predictor +
  latent-MPC recipe. Released weights become the round-0 reference $f_{\text{ref}}$ and therefore the
  gauge anchor at $t{=}0$. See [RFC-0008](../rfcs/RFC-0008-model-objective-numerics.md).
- **LeJEPA / LeWorldModel** (Balestriero & LeCun; Maes, Le Lidec et al., 2026) — the objective: SIGReg
  (random-projection sketch + characteristic-function matching of each marginal to $\mathcal{N}(0,1)$),
  which removes EMA / stop-gradient teacher–student state and so is federation-friendly.
  See [RFC-0008](../rfcs/RFC-0008-model-objective-numerics.md).
- **stable-worldmodel** (galilai-group) — the data layer (`lance` default, `hdf5` portable, `lerobot://`
  adapter), standardized environments, and the latent-MPC evaluation harness. See
  [RFC-0004](../rfcs/RFC-0004-data-provenance.md) and
  [RFC-0005](../rfcs/RFC-0005-evaluation.md).
- **WMCP** (WM-RFC-0001) — the shared latent/action contract that makes heterogeneous-embodiment
  federation well-posed; the explicit analogue of the fixed token vocabulary LLM federation gets for free.
  See [RFC-0007](../rfcs/RFC-0007-wmcp-latent-contract.md).
- **DiLoCo / OpenDiLoCo / INTELLECT** (Douillard et al.; Prime Intellect) — the low-communication
  inner/outer optimizer (communicate every $H$ inner steps) and the elastic fault-tolerance engineering.
  See [RFC-0003](../rfcs/RFC-0003-federated-protocol.md) and
  [RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md).
- **Stwo** (Circle-STARK prover) — the Phase-2 aggregation-correctness proof. See
  [RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md).

## 4. Goals

Each goal is testable; the verification path is named in parentheses. The corresponding tests are
enumerated in [07 — Testing Strategy](07-testing-strategy.md).

- **G1 — End-to-end JEPA co-training.** Train encoder $f_\theta$ and predictor $g_\phi$ jointly under the
  three-term objective $\mathcal{L} = \lambda_{\text{pred}}\,\mathbb{E}\lVert g_\phi(f_\theta(x_t),a_t) -
  \text{sg}[f_\theta(x_{t+1})]\rVert^2 + \lambda_{\text{sig}}\,\mathrm{SIGReg}_A(f_\theta(x)) +
  \lambda_{\text{anc}}\,\mathcal{L}_{\text{anchor}}(f_\theta;\mathcal{P},\{t_i\})$, warm-started from
  V-JEPA 2 (verification: Stage A centralized run + latent-MPC eval,
  [RFC-0005 §3](../rfcs/RFC-0005-evaluation.md#3-downstream-metric--planning-success)).
- **G2 — Close the latent gauge under federation.** Hold the inter-participant frame pinned over training
  where naive aggregation diverges (verification: the frame-drift diagnostic,
  [RFC-0002 §9](../rfcs/RFC-0002-gauge-and-aggregation.md#9-the-frame-drift-diagnostic-the-headline-measurement) and
  [RFC-0005 §2](../rfcs/RFC-0005-evaluation.md#2-headline-diagnostic--latent-frame-drift); enforced by
  `INV-WARMSTART-T0`, `INV-PROBE-PIN`).
- **G3 — Sovereign data residency.** No raw observation, action, or private embedding crosses a trust
  boundary (verification: residency-guard test, `INV-RESIDENCY` enforced in `lensemble.data.residency`,
  raising `ResidencyViolation`; see [06 — Security](06-security.md)).
- **G4 — Privacy on released deltas (target).** Each released $\Delta_c$ is clipped and noised with
  participant-secret entropy under a persistent, pre-release $(\varepsilon,\delta)$ budget. The current
  replay-noise and one-round report path does not satisfy this goal (verification: `INV-DP-BOUND`,
  [RFC-0012](../rfcs/RFC-0012-differential-privacy.md)).
- **G5 — Deterministic, reproducible aggregation.** The outer step is a bitwise-reproducible function of
  its inputs (verification: `INV-AGG-DETERMINISM`, the determinism self-check that raises
  `NonDeterministicAggregation`; [RFC-0009](../rfcs/RFC-0009-configuration-reproducibility.md)).
- **G6 — Preserve future proof-integration seams.** Deterministic aggregation, hash-committed model versions
  (`INV-CHECKPOINT-HASH`), episode hashing + Merkle commitments (`INV-COMMIT-BINDING`), a pinned public
  probe (`INV-PROBE-PIN`), and public recomputation of committed-model diagnostics remain explicit,
  testable interfaces. These properties do not prove participant-local execution or guarantee that a
  future proof layer will require no rework (verification: `recompute_alignment` test,
  [RFC-0006 §5](../rfcs/RFC-0006-verifiable-contribution.md)).
- **G7 — Produce a claim-grade centralized/local/federated comparison.** A corrected run must reproduce
  the centralized-pooled upper bound, local-only lower bound, and anchored-federation condition from
  released checkpoints and configs. Existing full-model comparisons predate the outer-update fix and
  cannot satisfy this goal (verification:
  [RFC-0005 §8](../rfcs/RFC-0005-evaluation.md#8-reproducibility--reporting)).

## 5. Non-Goals

These are explicit exclusions. They bound the work and prevent scope drift.

- **Not an LLM.** Lensemble is a visual world model over embodied interaction; there is no token vocabulary
  and no language modeling objective. The fixed-vocabulary anchoring LLM federation enjoys for free is
  exactly what we must manufacture (the frame anchor).
- **Not frozen-encoder by default.** The target is Fork B (co-trained encoder). Fork A (frozen shared
  encoder, federate the predictor only) is the documented safe-degrade fallback, not the default
  (see [§6](#6-fork-b-target-vs-fork-a-safe-degrade)).
- **Not a data-quality or honesty oracle.** Provenance commitments prove data *origin*, not data *quality*
  and not honest computation ([RFC-0014](../rfcs/RFC-0014-provenance-commitments.md);
  [06 — Security](06-security.md)).
- **Stage E (own foundation-scale federated video pretraining) is out of v1.0 scope.** Lensemble
  warm-starts from released V-JEPA 2 weights; training a 1.2B-class encoder from scratch over a federation
  is the expensive frontier, captured as future work, not an implementable v1.0 deliverable
  ([conventions §12](conventions.md#12-milestones-and-stages); [§8](#8-v10-scope-boundary)).
- **Stage D actual STARK/TEE proofs are out of v1.0 scope.** The proof-oriented engineering disciplines
  (deterministic aggregation, hash commitments, Merkle roots, pinned probe, public recomputation) ARE in
  scope for v0.1–v1.0 as auditable interfaces; the cryptographic proofs themselves are Phase-2 / post-v1.0
  ([RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md); [§8](#8-v10-scope-boundary)).

## 6. Fork B target vs Fork A safe-degrade

Lensemble has two named training regimes. The choice is whether the encoder is co-trained or frozen.

- **Fork B — end-to-end (the target).** Encoder $f_\theta$ and predictor $g_\phi$ are co-trained. This is
  the scientifically harder regime: co-training the encoder is precisely what opens the latent gauge (the SIGReg
  objective's $O(d)$ rotation invariance acts on a *moving* encoder), so the gauge problem and its fix
  ([RFC-0002](../rfcs/RFC-0002-gauge-and-aggregation.md)) define the target
  research question. The current corrected runtime has not yet produced the
  evidence needed to answer it.
- **Fork A — frozen shared encoder (the documented safe-degrade).** If Fork-B gauge control proves
  unstable at scale, freeze the shared encoder at the warm-start and federate only the predictor $g_\phi$.
  A frozen encoder dissolves the gauge entirely (no $O(d)$ symmetry to drift through) and recovers a clean
  federation path, at the cost of no longer testing the moving-encoder gauge question. Fork A is a
  supported, tested path at v1.0 ([conventions §12](conventions.md#12-milestones-and-stages)), invoked as a fallback, never the default
  ([RFC-0002](../rfcs/RFC-0002-gauge-and-aggregation.md), Alternatives Considered;
  [RFC-0008](../rfcs/RFC-0008-model-objective-numerics.md)).

RISK: Fork-B gauge control at video-world-model scale is unproven. SIGReg is demonstrated only to ViT-H on
images; the frame anchor's central knob $\lambda_{\text{anc}}$ trades quality against drift. Resolution
plan: Stage A de-risks the objective centrally before federation; Stage B sweeps $\lambda_{\text{anc}}$ and
the ablation ladder; Fork A is the explicit degrade if anchoring does not hold. Owner @AbdelStark; tracked
in [RFC-0002](../rfcs/RFC-0002-gauge-and-aggregation.md), Open Questions.

## 7. Success criteria

These are acceptance gates for a future claim-grade result, not statements about current evidence.
The checked-in full-model runs predate the outer-update correction and target-bound probe contract, so
they cannot satisfy either the scientific or engineering gates below. Both sets of gates must hold
before the result is promoted.

### 7.1 Paper-grade (claims 1–3 plus sweeps)

From [RFC-0005 §9](../rfcs/RFC-0005-evaluation.md#9-success-criteria):

- **Claim 1 gate.** A corrected matched run must show whether naive end-to-end FedAvg degrades under
  non-IID silos on both frame drift and a downstream metric. This is the negative control.
- **Claim 2 gate.** A corrected matched run must show whether frame anchoring keeps inter-participant
  frame drift below the naive-FedAvg control
  ([RFC-0002 §4](../rfcs/RFC-0002-gauge-and-aggregation.md)).
- **Claim 3 gate.** A corrected matched run must quantify any fraction of the centralized−local gap
  recovered on downstream planning success, with no data moved. (OPEN QUESTION: the minimum fraction is
  set in Stage B; owner @AbdelStark, resolved in
  [RFC-0005](../rfcs/RFC-0005-evaluation.md), Open Questions.)
- **Sweeps.** Any promoted claims must hold across at least one non-IID severity sweep and one scale step
  ([RFC-0005 §7](../rfcs/RFC-0005-evaluation.md#7-non-iid-severity--scale-sweeps)).

Claim 4 (robustness across non-IID severity and participant count $C$, inner horizon $H$) is the sweep
dimension that supports claims 1–3 rather than a standalone gate.

### 7.2 Engineering-grade

A claim-grade release must reproduce the centralized/local/federated comparison through the same
documented execution and artifact path:

- The **centralized-pooled** upper bound (all silo data in one place, end-to-end), the **local-only** lower
  bound, and the **anchored-federation** configuration all run from released checkpoints + Hydra configs
  ([RFC-0005 §5, §8](../rfcs/RFC-0005-evaluation.md#5-baselines)).
- Every run emits a `RunManifest` (config hash, seeds, environment, dependency versions, git SHA, probe
  hash) such that the same config + same seed reproduces the manifest's config and seed hashes, and the
  aggregation path is bitwise-reproducible (`INV-AGG-DETERMINISM`;
  [RFC-0009](../rfcs/RFC-0009-configuration-reproducibility.md)).
- The named invariants `INV-RESIDENCY`, `INV-WARMSTART-T0`, `INV-SKETCH-CONSISTENCY`,
  `INV-AGG-DETERMINISM`, `INV-PROBE-PIN`, `INV-COMMIT-BINDING`, `INV-CHECKPOINT-HASH`, `INV-DP-BOUND`,
  `INV-WMCP`, and `INV-ACTIONHEAD-LOCAL` are each enforced and tested (see
  [04 — Error Model](04-error-model.md) for the failure response of each and
  [07 — Testing Strategy](07-testing-strategy.md) for the test).

## 8. v1.0 scope boundary

Milestones map to stages per [conventions §12](conventions.md#12-milestones-and-stages). The table
defines target contents; it does not certify that a milestone's empirical claims have been achieved.

| Milestone | Stage | Content |
|---|---|---|
| **v0.1** | A | Single-site, warm-started, ViT-L/~300M end-to-end SIGReg + AC predictor on pooled robot data; latent-MPC eval (the centralized upper bound). Plus foundational scaffolding: package skeleton, config system, data layer, WMCP contract, model + objective, eval harness, observability, artifact format, error taxonomy, CI, packaging. |
| **v0.2** | B | Simulated federation on one cluster: DiLoCo outer loop, frame anchor (Layers 1–4), Procrustes backstop, aggregation/DP mechanism plumbing with explicit effectiveness status, the frame-drift diagnostic, the full ablation ladder and non-IID / scale sweeps. The scientific core / the paper. |
| **v0.3** | C | Two real sovereign nodes over a network boundary: real secure aggregation + DP, residency enforcement, fault tolerance / elasticity, contribution ledger. The sovereignty demonstration. |
| **v1.0** | — | Hardening: frozen public API, complete docs + reproducibility package, release automation, Fork A fallback path supported and tested, and proof-integration seams exercised without claiming a cryptographic proof or no-rework guarantee ([RFC-0006 §5](../rfcs/RFC-0006-verifiable-contribution.md)). |

The target boundary at v1.0 includes the rows above. Beyond it lie two out-of-scope items, captured as future work
in the tracker and **not** filed as implementable v1.0 issues ([conventions §12](conventions.md#12-milestones-and-stages)):

- **Stage D** — the actual STARK / TEE verifiable layer (Phase 2), beyond the proof-oriented disciplines that
  v0.1–v1.0 deliver.
- **Stage E** — own foundation-scale federated video pretraining (training the 1.2B-class encoder from
  scratch over a federation rather than warm-starting from released V-JEPA 2 weights).

Proof-oriented engineering disciplines — deterministic aggregation, hash commitments, Merkle roots,
the pinned probe, and public recomputation — are in scope for v0.1–v1.0. They keep interfaces explicit;
they do not prove that Stage D can be added without changes.

## 9. Corpus index

The corpus is two layers: **spec sections** (`docs/spec/`) hold the stable, normative contracts; **RFCs**
(`docs/rfcs/`) hold the rationale, design, and alternatives. Spec sections cite RFCs for rationale; RFCs
cite spec sections for the stable contract. The top-level `SPEC.md` (authored by the lead) is the corpus
front matter.

### 9.1 Specification sections

| Section | Title | Owns |
|---|---|---|
| [00 — Overview](00-overview.md) | this document | Thesis, goals, non-goals, success criteria, scope |
| [01 — Architecture](01-architecture.md) | System architecture | Module map, dependency layering, federation map, topology, trust boundaries, data flow |
| [02 — Public API](02-public-api.md) | Public API surface | Python surface, CLI, stability/versioning behavior, extension points |
| [03 — Data Model](03-data-model.md) | Types & schemas | Schemas for every core type, invariants, serialization, schema versioning |
| [04 — Error Model](04-error-model.md) | Errors & recovery | Exception hierarchy, error codes, failure-mode catalog, recovery |
| [05 — Observability](05-observability.md) | Logging, metrics, tracing | Log/metric taxonomy, frame-drift diagnostic emission, redaction, sinks |
| [06 — Security](06-security.md) | Threat model & trust | Threat model, trust boundaries, residency, secure-agg + DP target contracts, secrets |
| [07 — Testing Strategy](07-testing-strategy.md) | Tests & CI gates | Test pyramid, ML-specific tests, ablation-ladder tests, CI gates |
| [08 — Performance Budget](08-performance-budget.md) | Budgets & profiling | Throughput/latency/memory/comms budgets, profiling plan, scaling |
| [09 — Release & Versioning](09-release-and-versioning.md) | SemVer & releases | SemVer policy, deprecation, changelog, artifact release, licenses |
| [10 — Glossary](10-glossary.md) | Canonical terms | Definitions for every term used across the corpus |

### 9.2 RFCs

| RFC | Title | Status | Specifies |
|---|---|---|---|
| [RFC-0001](../rfcs/RFC-0001-architecture.md) | Architecture & System Overview | Accepted | What we build: model, federation map, topology, trust boundaries, staged plan |
| [RFC-0002](../rfcs/RFC-0002-gauge-and-aggregation.md) | The Latent Gauge & Frame-Anchored Aggregation | Accepted | The scientific core: the gauge problem and its frame-anchored fix |
| [RFC-0003](../rfcs/RFC-0003-federated-protocol.md) | Federated Training Protocol | Accepted | Rounds, DiLoCo outer loop, DP clip+noise, secure-agg pointer, fault tolerance |
| [RFC-0004](../rfcs/RFC-0004-data-provenance.md) | Data, Sovereignty & Provenance | Accepted | Per-participant data layer, public probe, provenance commitments |
| [RFC-0005](../rfcs/RFC-0005-evaluation.md) | Evaluation & Benchmark Protocol | Accepted | Claims, frame-drift diagnostic, ablation ladder, baselines, metrics |
| [RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md) | Verifiable Contribution | Draft · Phase 2 (Deferred) | Deferred crypto layer and the current interfaces it may consume |
| [RFC-0007](../rfcs/RFC-0007-wmcp-latent-contract.md) | WMCP Latent Contract & Embodiment Adapters | Accepted | `LatentState` / `ActionSpec` contract; per-embodiment action heads |
| [RFC-0008](../rfcs/RFC-0008-model-objective-numerics.md) | Model, Objective & Numerical Contracts | Accepted | Encoder, predictor, SIGReg objective, numerical contracts |
| [RFC-0009](../rfcs/RFC-0009-configuration-reproducibility.md) | Configuration, Run Manifest & Reproducibility | Accepted | Hydra configs, `RunManifest`, seeding, reproducibility contracts |
| [RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md) | Checkpoint & Artifact Format | Accepted | Schema-versioned, hash-committed on-disk artifacts |
| [RFC-0011](../rfcs/RFC-0011-secure-aggregation.md) | Secure Aggregation Protocol | Accepted | Target pairwise-mask / TEE aggregation; current runtime reports post-commit cross-check or fallback |
| [RFC-0012](../rfcs/RFC-0012-differential-privacy.md) | Differential Privacy Accounting | Accepted | Per-participant update DP; $(\varepsilon,\delta)$ accountant |
| [RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md) | Coordinator & Participant Runtime | Accepted | Round state machine, fault tolerance, control plane |
| [RFC-0014](../rfcs/RFC-0014-provenance-commitments.md) | Provenance Commitments & Merkle Scheme | Accepted | Episode hashing, Merkle root $R_c$, contribution ledger |
| [RFC-0015](../rfcs/RFC-0015-observability-diagnostics.md) | Observability, Diagnostics & Telemetry | Accepted | Instrumentation contract incl. the frame-drift diagnostic |

Read order for the paper: [RFC-0002](../rfcs/RFC-0002-gauge-and-aggregation.md) →
[RFC-0005](../rfcs/RFC-0005-evaluation.md) → [RFC-0001](../rfcs/RFC-0001-architecture.md). Read order to
build: [RFC-0001](../rfcs/RFC-0001-architecture.md) → [RFC-0003](../rfcs/RFC-0003-federated-protocol.md) →
[RFC-0004](../rfcs/RFC-0004-data-provenance.md) → [RFC-0002](../rfcs/RFC-0002-gauge-and-aggregation.md) →
[RFC-0005](../rfcs/RFC-0005-evaluation.md), then [RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md).

## 10. Open Questions

- OPEN QUESTION: the exact fraction of the centralized−local planning-success gap that anchored federation
  must recover to count as Claim 3 satisfied. Owner @AbdelStark; resolved in Stage B via
  [RFC-0005](../rfcs/RFC-0005-evaluation.md), Open Questions.
- OPEN QUESTION: the personalization boundary — how heterogeneous participants may be before one global
  encoder breaks and the shared-backbone + per-embodiment-head split (`INV-ACTIONHEAD-LOCAL`) no longer
  suffices. Owner @AbdelStark; resolved in Stage B via
  [RFC-0001](../rfcs/RFC-0001-architecture.md), Open Questions.

## 11. References

- Lensemble README (project framing, thesis, contribution, ecosystem placement, license stanza).
- [RFC-0001 — Architecture & System Overview](../rfcs/RFC-0001-architecture.md) (model, federation map,
  topology, trust boundaries, staged plan A–E).
- [RFC-0002 — The Latent Gauge & Frame-Anchored Aggregation](../rfcs/RFC-0002-gauge-and-aggregation.md)
  (the gauge problem and its fix).
- [RFC-0005 — Evaluation & Benchmark Protocol](../rfcs/RFC-0005-evaluation.md) (claims, diagnostics,
  baselines, success criteria).
- [RFC-0006 — Verifiable Contribution](../rfcs/RFC-0006-verifiable-contribution.md) (Phase-2 proofs and
  Phase-1 proof-readiness).
- V-JEPA 2 (Assran et al., 2025); LeJEPA / LeWorldModel (Balestriero & LeCun; Maes, Le Lidec et al., 2026);
  stable-worldmodel (galilai-group); WMCP (WM-RFC-0001); DiLoCo / OpenDiLoCo / INTELLECT (Douillard et al.;
  Prime Intellect); Stwo (Circle-STARK prover).
