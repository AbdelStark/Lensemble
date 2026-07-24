# Lensemble — Specification

Research stack for reproducible federated training and evaluation of action-conditioned JEPA world models.

- Version: v0.1 (specification corpus)
- Author: Abdelhamid Bakhta ([@AbdelStark](https://github.com/AbdelStark))
- Last updated: 2026-07-24
- License: code Apache-2.0, docs CC-BY-4.0, data CDLA-Permissive-2.0 (proposed; see [09-release-and-versioning.md](docs/spec/09-release-and-versioning.md))

This is the entry point to the Lensemble specification. It is an index and executive summary; the
normative detail lives in [`docs/spec/`](docs/spec/) (the stable contract) and [`docs/rfcs/`](docs/rfcs/)
(the decision records). The reference implementation now lives in [`lensemble/`](lensemble/), with
claim-gated evidence and roadmap status tracked alongside the spec corpus.

## Executive summary

Lensemble is a research stack for executing and evaluating federated,
action-conditioned Joint-Embedding Predictive Architecture (JEPA) world-model
training while participant trajectories remain local. The implementation can
co-train and aggregate the encoder and predictor (the "Fork B" path), and it
includes differential-privacy, secure-aggregation, artifact, provenance, and
evaluation contracts.

That implemented execution path is not an achieved useful and gauge-stable
scientific result. The checked-in full-model evidence predates correction of an
outer-step sign defect and must be rerun before it supports new claims. The
corrected full-model research program remains open in
[#335](https://github.com/AbdelStark/Lensemble/issues/335). The browser
`real-lewm-tworooms` mode is a separate, narrower result: federated adapter
continuation on a frozen checkpoint, not full-model federated training.

The main research question is the **latent gauge** problem. The SIGReg-JEPA objective is invariant under
orthogonal rotations $Q \in O(d)$ of the latent space, so independently-updated participants converge to
mutually-rotated coordinate frames and make naive weight averaging unreliable.
Lensemble provides shared warm starts, public-probe frame anchors, alignment
diagnostics, and explicit controls for studying that failure mode. Whether those
mechanisms yield a useful, magnitude-stable, and gauge-stable full-model result
after the corrected outer step is an empirical question, not a settled claim.
Cryptographic contribution proofs remain deferred future work.

Full statement of thesis, goals, non-goals, and success criteria: [00-overview.md](docs/spec/00-overview.md).

## How to read this corpus

- For the research rationale: [RFC-0002](docs/rfcs/RFC-0002-gauge-and-aggregation.md) (the gauge and
  proposed controls) → [RFC-0005](docs/rfcs/RFC-0005-evaluation.md) (how results are evaluated) →
  [RFC-0001](docs/rfcs/RFC-0001-architecture.md) (the system).
- To build: [RFC-0001](docs/rfcs/RFC-0001-architecture.md) →
  [RFC-0003](docs/rfcs/RFC-0003-federated-protocol.md) →
  [RFC-0004](docs/rfcs/RFC-0004-data-provenance.md) →
  [RFC-0002](docs/rfcs/RFC-0002-gauge-and-aggregation.md) →
  [RFC-0005](docs/rfcs/RFC-0005-evaluation.md), then the subsystem RFCs (0007–0017), then
  [RFC-0006](docs/rfcs/RFC-0006-verifiable-contribution.md).
- For conventions, notation, named invariants, and the type/API/error contracts shared across the
  corpus: [conventions.md](docs/spec/conventions.md).

## Specification sections

| Section | Contents |
|---|---|
| [00 — Overview](docs/spec/00-overview.md) | Thesis, contribution, goals, non-goals, success criteria, v1.0 scope |
| [01 — Architecture](docs/spec/01-architecture.md) | Module map, dependency layering, federation map, topology, trust boundaries, data flow |
| [02 — Public API](docs/spec/02-public-api.md) | Public Python surface, stability policy, CLI, extension points |
| [03 — Data Model](docs/spec/03-data-model.md) | Core types, schemas, invariants, serialization, schema versioning |
| [04 — Error Model](docs/spec/04-error-model.md) | Error taxonomy, failure-mode catalog, recovery, handling rules |
| [05 — Observability](docs/spec/05-observability.md) | Structured logging, metric taxonomy, the frame-drift diagnostic, redaction |
| [06 — Security](docs/spec/06-security.md) | Threat model, trust boundaries, residency, secure aggregation, secrets |
| [07 — Testing Strategy](docs/spec/07-testing-strategy.md) | Test pyramid, ML-specific tests, the ablation ladder as tests, CI gates |
| [08 — Performance Budget](docs/spec/08-performance-budget.md) | Throughput/latency/memory/communication budgets, profiling plan |
| [09 — Release & Versioning](docs/spec/09-release-and-versioning.md) | SemVer, deprecation, changelog, license, contributor workflow |
| [10 — Glossary](docs/spec/10-glossary.md) | Canonical terms |
| [Conventions & Contracts](docs/spec/conventions.md) | Notation, invariants, naming, the shared API/type/error contracts |

## RFC index

| RFC | Title | Status | Area |
|---|---|---|---|
| [0001](docs/rfcs/RFC-0001-architecture.md) | Architecture & System Overview | Accepted | core |
| [0002](docs/rfcs/RFC-0002-gauge-and-aggregation.md) | The Latent Gauge & Frame-Anchored Aggregation | Accepted | gauge |
| [0003](docs/rfcs/RFC-0003-federated-protocol.md) | Federated Training Protocol | Accepted | federation |
| [0004](docs/rfcs/RFC-0004-data-provenance.md) | Data, Sovereignty & Provenance | Accepted | data |
| [0005](docs/rfcs/RFC-0005-evaluation.md) | Evaluation & Benchmark Protocol | Accepted | eval |
| [0006](docs/rfcs/RFC-0006-verifiable-contribution.md) | Verifiable Contribution | Draft · Phase 2 (Deferred) | verify |
| [0007](docs/rfcs/RFC-0007-wmcp-latent-contract.md) | WMCP Latent Contract & Embodiment Adapters | Accepted | contracts |
| [0008](docs/rfcs/RFC-0008-model-objective-numerics.md) | Model, Objective & Numerical Contracts | Accepted | model |
| [0009](docs/rfcs/RFC-0009-configuration-reproducibility.md) | Configuration, Run Manifest & Reproducibility | Accepted | config |
| [0010](docs/rfcs/RFC-0010-artifact-checkpoint-format.md) | Checkpoint & Artifact Format | Accepted | artifacts |
| [0011](docs/rfcs/RFC-0011-secure-aggregation.md) | Secure Aggregation Protocol | Accepted | aggregation |
| [0012](docs/rfcs/RFC-0012-differential-privacy.md) | Differential Privacy Accounting | Accepted | privacy |
| [0013](docs/rfcs/RFC-0013-coordinator-runtime.md) | Coordinator & Participant Runtime | Accepted | federation |
| [0014](docs/rfcs/RFC-0014-provenance-commitments.md) | Provenance Commitments & Merkle Scheme | Accepted | provenance |
| [0015](docs/rfcs/RFC-0015-observability-diagnostics.md) | Observability, Diagnostics & Telemetry | Accepted | observability |
| [0016](docs/rfcs/RFC-0016-deployment-vendoring-topology.md) | Deployment, Vendoring & Topology | Accepted | core |
| [0017](docs/rfcs/RFC-0017-dynamic-env-ungameable-metrics.md) | Dynamic Env & Ungameable Ground-Truth Metrics | Draft | eval |

## Scope and status

- **Phase 1** (this corpus, milestones `v0.1`–`v1.0`): the reference implementation,
  reproducible experiments, and artifact/provenance disciplines used to study
  full-model federation. Current full-model results require a corrected rerun
  before supporting new scientific claims ([#335](https://github.com/AbdelStark/Lensemble/issues/335)).
- **Phase 2** ([RFC-0006](docs/rfcs/RFC-0006-verifiable-contribution.md), Stage D, post-`v1.0`):
  the cryptographic verifiable-contribution layer.
- **Fork B** (encoder + predictor co-trained) is the target; **Fork A** (frozen shared encoder, federate
  the predictor only) is the documented safe-degrade fallback
  ([RFC-0002](docs/rfcs/RFC-0002-gauge-and-aggregation.md)).
- **Browser federated demo** ([#294](https://github.com/AbdelStark/Lensemble/issues/294),
  [#303](https://github.com/AbdelStark/Lensemble/issues/303),
  [BROWSER_FEDERATED_DEMO.md](docs/roadmap/BROWSER_FEDERATED_DEMO.md)) is an educational
  orchestration surface: QR joins, WebSocket primary transport, REST polling fallback, participant
  liveness, bounded tiny browser update vectors, coordinator-style aggregation, model revision
  publication, inference artifact attachment, and residency-safe evidence export. It is not a
  production browser-training claim and not evidence that the current dynamic-env federation materially
  beats local-only.
- **Tapestry-like LeWM demo** ([#314](https://github.com/AbdelStark/Lensemble/issues/314),
  [TAPESTRY_LEWM.md](docs/roadmap/TAPESTRY_LEWM.md)) is an optional demonstration above that substrate: a
  `real-lewm-tworooms` mode with checkpoint-backed LeWorldModel browser inference, browser-local bounded
  adapter continuation, federated adapter-delta aggregation, and claim-bounded evidence. It is a
  Tapestry-like adaptation demo around a real TwoRooms checkpoint, not full from-scratch browser LeWM
  pretraining and not a paper-scale benchmark claim.
- Milestones map to the staged plan A–E:
  [conventions §12](docs/spec/conventions.md#12-milestones-and-stages). Stage E (own foundation-scale
  federated pretraining) and the Stage-D proofs are out of the v1.0 scope and tracked as future work.

Open questions carried by the corpus are listed in each document's `Open Questions` section and
summarized in [00-overview.md](docs/spec/00-overview.md); each carries an owner and a resolution path.
