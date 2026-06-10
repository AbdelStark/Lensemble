# Lensemble — Specification

Federated, end-to-end JEPA world models — trained across sovereign data, verifiable by construction.

- Version: v0.1 (specification corpus)
- Author: Abdelhamid Bakhta ([@AbdelStark](https://github.com/AbdelStark))
- Date: 2026-06-02
- License: code Apache-2.0, docs CC-BY-4.0, data CDLA-Permissive-2.0 (proposed; see [09-release-and-versioning.md](docs/spec/09-release-and-versioning.md))

This is the entry point to the Lensemble specification. It is an index and executive summary; the
normative detail lives in [`docs/spec/`](docs/spec/) (the stable contract) and [`docs/rfcs/`](docs/rfcs/)
(the decision records). The reference implementation now lives in [`lensemble/`](lensemble/), with
claim-gated evidence and roadmap status tracked alongside the spec corpus.

## Executive summary

Lensemble trains a single action-conditioned Joint-Embedding Predictive Architecture (JEPA) world model
**end-to-end across many mutually-distrusting participants**, where each participant's raw interaction
data never leaves its boundary. Encoder and predictor are co-trained (the hard regime, "Fork B"); only
model deltas cross the network, aggregated under differential privacy and secure aggregation, with a
Phase-2 roadmap to cryptographic proof of each participant's contribution.

The scientific core is the **latent gauge** problem. The SIGReg-JEPA objective is invariant under
orthogonal rotations $Q \in O(d)$ of the latent space, so independently-updated participants converge to
mutually-rotated coordinate frames and naive weight-averaging is meaningless — a failure mode that
anchored models (supervised nets, LLMs with a fixed vocabulary) never see. Lensemble closes the gauge
with a shared encoder warm-start and a light public-probe **frame anchor**, which makes weight-averaging
valid again and, structurally, keeps the eventual proof-of-contribution circuit cheap (the anchored frame
keeps aggregation a near-linear operation). The federated end-to-end JEPA result and the first measurement
of latent frame-drift under federation are the lead contributions; verifiable contribution is the Phase-2
differentiator. To our knowledge no prior work federates an end-to-end JEPA world model, nor
measures/controls the latent gauge under federation.

Full statement of thesis, goals, non-goals, and success criteria: [00-overview.md](docs/spec/00-overview.md).

## How to read this corpus

- For the science (the paper): [RFC-0002](docs/rfcs/RFC-0002-gauge-and-aggregation.md) (the gauge and
  its fix) → [RFC-0005](docs/rfcs/RFC-0005-evaluation.md) (how the claims are proved) →
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

- **Phase 1** (this corpus, milestones `v0.1`–`v1.0`): the federated end-to-end JEPA science and a
  reference implementation. Ships **proof-ready** so Phase 2 needs no rework.
- **Phase 2** ([RFC-0006](docs/rfcs/RFC-0006-verifiable-contribution.md), Stage D, post-`v1.0`):
  the cryptographic verifiable-contribution layer.
- **Fork B** (encoder + predictor co-trained) is the target; **Fork A** (frozen shared encoder, federate
  the predictor only) is the documented safe-degrade fallback
  ([RFC-0002](docs/rfcs/RFC-0002-gauge-and-aggregation.md)).
- **Browser federated demo** ([#294](https://github.com/AbdelStark/Lensemble/issues/294),
  [BROWSER_FEDERATED_DEMO.md](docs/roadmap/BROWSER_FEDERATED_DEMO.md)) is a local educational
  orchestration surface: QR joins, backend lifecycle API, browser-surrogate update metadata,
  coordinator-style aggregation, inference artifact attachment, and residency-safe evidence export. It
  is not a production browser-training claim and not evidence that the current dynamic-env federation
  materially beats local-only.
- Milestones map to the staged plan A–E:
  [conventions §12](docs/spec/conventions.md#12-milestones-and-stages). Stage E (own foundation-scale
  federated pretraining) and the Stage-D proofs are out of the v1.0 scope and tracked as future work.

Open questions carried by the corpus are listed in each document's `Open Questions` section and
summarized in [00-overview.md](docs/spec/00-overview.md); each carries an owner and a resolution path.
