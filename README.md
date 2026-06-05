# Lensemble

**Federated, end-to-end JEPA world models — trained across sovereign data, verifiable by construction.**

*Specification corpus · v0.1 (Draft) · June 2026*
Author: Abdelhamid Bakhta ([@AbdelStark](https://github.com/AbdelStark))

[![ci](https://github.com/AbdelStark/Lensemble/actions/workflows/ci.yml/badge.svg)](https://github.com/AbdelStark/Lensemble/actions/workflows/ci.yml)

---

## What Lensemble is

Lensemble trains a single **action-conditioned Joint-Embedding Predictive Architecture (JEPA) world model end-to-end across many mutually-distrusting participants**, where each participant's raw interaction data never leaves its boundary. Encoder *and* predictor are co-trained (this is the hard regime — see below); only model deltas cross the network, aggregated under privacy guarantees, with a roadmap to **cryptographic proof of each participant's contribution**.

The name: *l'ensemble* — "the whole / together" in French, and the ML *ensemble* (many models acting as one) — with **lens** sitting in front, the perception encoder at the heart of a visual world model. Many sovereign learners, trained *ensemble*, into one model.

## The one-paragraph thesis

A foundation-scale world model wants diverse embodied experience — robot fleets, manipulation labs, driving stacks, egocentric video — but that data is siloed by IP, privacy, and safety and cannot be pooled. Federated training is the access strategy. The catch specific to JEPA: its self-supervised objective is **invariant under rotations of the latent space**, so independently-updated participants drift into mutually-rotated coordinate frames and naive weight-averaging is meaningless — a failure mode that anchored models (supervised nets, LLMs with a fixed vocabulary) never see. Lensemble closes that gauge with a shared encoder warm-start and a light public-probe **frame anchor**, which makes weight-averaging valid again *and* keeps the eventual proof-of-contribution circuit cheap. The science (federated end-to-end JEPA + the gauge result) is the lead contribution; verifiable contribution is the Phase-2 differentiator.

## Contribution (the part nobody occupies)

1. **Federated end-to-end JEPA** — not LLM, not frozen-encoder. The latent-gauge problem, its solution, and the first measurement of latent frame-drift under federated self-supervision.
2. **Sovereign embodied data** — the federation premise applied to physical AI: trajectories that cannot be moved.
3. **Verifiable contribution** — succinct cryptographic attestation of correct aggregation and data provenance (Phase 2). Existing decentralized-training efforts rely on redundancy/economics; we move to cryptographic attestation.

To our knowledge no prior work federates an end-to-end JEPA world model, nor measures/controls the latent gauge under federation; this corpus specifies both, and the verifiable layer on top.

The canonical entry point is [SPEC.md](SPEC.md): an index and executive summary over the corpus.
The normative spec sections live in [`docs/spec/`](docs/spec/) and the decision records in
[`docs/rfcs/`](docs/rfcs/). Shared notation, named invariants, and the API/type/error contracts are in
[`docs/spec/conventions.md`](docs/spec/conventions.md).

## RFC index

| RFC | Title | Status | Role |
|---|---|---|---|
| [0001](docs/rfcs/RFC-0001-architecture.md) | Architecture & System Overview | Accepted | What we build: model, federation map, topology, trust boundaries |
| [0002](docs/rfcs/RFC-0002-gauge-and-aggregation.md) | The Latent Gauge & Frame-Anchored Aggregation | Accepted | **Scientific core**: the gauge problem and its solution |
| [0003](docs/rfcs/RFC-0003-federated-protocol.md) | Federated Training Protocol | Accepted | How the network runs: rounds, DiLoCo, secure aggregation, DP, fault tolerance |
| [0004](docs/rfcs/RFC-0004-data-provenance.md) | Data, Sovereignty & Provenance | Accepted | Per-silo data, the public probe, Merkle commitments |
| [0005](docs/rfcs/RFC-0005-evaluation.md) | Evaluation & Benchmark Protocol | Accepted | How we prove it: diagnostics, ablation ladder, baselines, metrics |
| [0006](docs/rfcs/RFC-0006-verifiable-contribution.md) | Verifiable Contribution | Draft · **Phase 2** | The crypto layer; what Phase 1 must satisfy to stay proof-ready |
| [0007](docs/rfcs/RFC-0007-wmcp-latent-contract.md) | WMCP Latent Contract & Embodiment Adapters | Accepted | The shared latent/action contract for heterogeneous embodiments |
| [0008](docs/rfcs/RFC-0008-model-objective-numerics.md) | Model, Objective & Numerical Contracts | Accepted | Encoder, predictor, SIGReg + anchor objective, numerical contract |
| [0009](docs/rfcs/RFC-0009-configuration-reproducibility.md) | Configuration, Run Manifest & Reproducibility | Accepted | Hydra configs, seeding, run manifests, reproducibility |
| [0010](docs/rfcs/RFC-0010-artifact-checkpoint-format.md) | Checkpoint & Artifact Format | Accepted | Schema-versioned, hash-committed model artifacts |
| [0011](docs/rfcs/RFC-0011-secure-aggregation.md) | Secure Aggregation Protocol | Accepted | Coordinator learns only the sum; dropout robustness |
| [0012](docs/rfcs/RFC-0012-differential-privacy.md) | Differential Privacy Accounting | Accepted | Per-participant clip+noise and (ε,δ) accounting |
| [0013](docs/rfcs/RFC-0013-coordinator-runtime.md) | Coordinator & Participant Runtime | Accepted | Round state machine, fault tolerance, control plane |
| [0014](docs/rfcs/RFC-0014-provenance-commitments.md) | Provenance Commitments & Merkle Scheme | Accepted | Episode hashing, Merkle roots, contribution ledger |
| [0015](docs/rfcs/RFC-0015-observability-diagnostics.md) | Observability, Diagnostics & Telemetry | Accepted | Logging, metrics, the frame-drift diagnostic, redaction |
| [0016](docs/rfcs/RFC-0016-deployment-vendoring-topology.md) | Deployment, Vendoring & Topology | Accepted | Python-first stack, third_party vendoring, IaC (Compose/Kubernetes) |

Read order for the paper: 0002 → 0005 → 0001. Read order to build: 0001 → 0003 → 0004 → 0002 → 0005, then the subsystem RFCs 0007–0016, then 0006.

## Where it sits in the ecosystem

- **V-JEPA 2** (Assran et al., 2025) — the warm-start encoder (1.2B; pretrained on >1M h video) and the action-conditioned-predictor + latent-MPC recipe.
- **LeJEPA / LeWorldModel** (Balestriero & LeCun; Maes, Le Lidec et al., 2026) — the objective: SIGReg (random-projection + characteristic-function matching to $\mathcal{N}(0,I)$), which removes EMA/stop-gradient/teacher–student.
- **stable-worldmodel** (galilai-group) — data collection, training scaffold, and model-predictive-control evaluation across standardized environments; `lance`/`hdf5`/`lerobot` data layer.
- **WMCP (WM-RFC-0001)** — the shared latent/action contract that makes heterogeneous-embodiment federation well-posed.
- **DiLoCo / OpenDiLoCo / INTELLECT** (Douillard et al.; Prime Intellect) — the low-communication inner/outer optimizer and the decentralized-training engineering we build on.
- **Project Tapestry** (AI Alliance) — the sovereignty/governance framing for federated frontier models; Lensemble is the JEPA-world-model instance of that premise.
- **Stwo** — Circle-STARK prover for the Phase-2 aggregation-correctness proof.

## Implementation Status

Lensemble now has an operational claim-MVP path in addition to the design corpus:

- **Federated LeWorldModel-style training path:** claim mode sets
  `objective.target_stop_gradient=false`, keeping the `f_theta(o_{t+1})` target branch live while using
  prediction MSE + SIGReg + the public-probe frame anchor.
- **Two-silo LeRobot-H5 federation:** default `Participant` hooks consume `cfg.data.data_source`,
  produce dataset Merkle roots, release encoder/predictor pseudo-gradients only, and close a
  `Coordinator` round.
- **Published HF evidence bundle:** final HF Job
  [`6a229653e52fdd2a02ed9125`](https://huggingface.co/jobs/abdelstark/6a229653e52fdd2a02ed9125) ran
  the federated launcher from merge `e76b680` on `cpu-basic` and published private smoke datasets
  `abdelstark/lensemble-claim-mvp-silo0` / `abdelstark/lensemble-claim-mvp-silo1` plus checkpoint/report
  artifacts in `abdelstark/lensemble-claim-mvp-checkpoint`.

The published `claim_mvp_report.json` records `round_state=closed`, `publication.pushed=true`,
`blocker=None`, final global checkpoint hash
`cf1c99a7e94ca610daa3bfc00c99d9ee68e9e34a302a96d848508e88edf4c0d5`, distinct participant dataset
roots, and scalar metrics: `val_pred`, `val_sigreg`, `effective_rank`, `frame_drift_deg`, and
`run_manifest_hash`. This is still a **claim MVP**, not a full paper-scale result: the current HF evidence
uses tiny smoke silos. The next empirical evidence tier is tracked in
[#200](https://github.com/AbdelStark/Lensemble/issues/200) and
[`docs/roadmap/PHASE2.md`](docs/roadmap/PHASE2.md).

Phase 2 data refs have started landing: the public
[`abdelstark/lensemble-phase2-so100-silos`](https://huggingface.co/datasets/abdelstark/lensemble-phase2-so100-silos)
dataset contains two deterministic SO-100 LeRobot-H5 participant silos, their
split manifest, and the smoke report with Merkle roots and window counts. The
remaining Phase 2 gates are GPU multi-round jobs, downstream evaluation,
baselines/curves, and the final model-card evidence bundle.

## Working assumptions

Assumptions, all overridable:

- **Goal**: a research paper plus an open reference implementation; the corpus is written to be scientifically self-contained.
- **Fork B (end-to-end)** is the target; Fork A (frozen shared encoder) is the documented fallback if gauge control proves unstable at scale ([RFC-0002, Fork A fallback](docs/rfcs/RFC-0002-gauge-and-aggregation.md#fork-a-fallback)).
- **Phase 2 has two tracks**: the empirical scale/evaluation stream in
  [#200](https://github.com/AbdelStark/Lensemble/issues/200), and the cryptographic contribution-proof
  layer in [RFC-0006](docs/rfcs/RFC-0006-verifiable-contribution.md). Phase 1 stays proof-ready
  ([RFC-0006 §3](docs/rfcs/RFC-0006-verifiable-contribution.md#3-phase-1-proof-ready-requirements-cheap-to-honor-now))
  so the proof layer should not require reworking artifact contracts.
- **Warm-start from released V-JEPA 2** — foundation-scale credibility without an INTELLECT-class pretraining bill, and a shared frame at $t{=}0$.
- **License**: code Apache-2.0, docs CC-BY-4.0, data CDLA-Permissive-2.0 — matching ecosystem norms (see [License](#license)).

## Repo layout

```
Lensemble/
├── README.md                          # this file
├── SPEC.md                            # corpus entry point: index + executive summary
└── docs/
    ├── spec/                          # normative spec sections
    │   ├── 00-overview.md … 10-glossary.md
    │   └── conventions.md             # notation, invariants, shared contracts
    ├── rfcs/                          # RFC-0001 … RFC-0016 (decision records)
    └── roadmap/                       # implementation tracker (filed alongside issues)
```

A reference implementation (`lensemble/`, Python; warm-starting V-JEPA 2, wrapping stable-worldmodel for data + MPC eval) follows the staged plan in [RFC-0001](docs/rfcs/RFC-0001-architecture.md) (Migration / Rollout) and the milestones in [conventions §12](docs/spec/conventions.md#12-milestones-and-stages).

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) covers the dev setup and the blocking CI gates (lint, type-check,
the CPU test suite, and the coverage thresholds) that must be green before any pull request merges.

## Documentation site

The spec/RFC corpus and the generated API reference render as a browsable site: `pip install -e ".[docs]"`
then `mkdocs serve` (or `mkdocs build`). The site indexes the [specification sections](docs/spec/00-overview.md)
and [RFCs](docs/rfcs/RFC-0001-architecture.md); the API reference covers the 1.0-frozen public surface.

## License

Lensemble splits its license by asset class (09 §7), and every release ships each file:

- **Code** — the `lensemble/` package, tests, build tooling, and CLI — is [Apache License 2.0](LICENSE)
  (SPDX `Apache-2.0`).
- **Docs** — `docs/spec/`, `docs/rfcs/`, and this README — are [CC-BY-4.0](LICENSE-docs).
- **Data** — released datasets, the public probe `P`, and the landmark targets — are
  [CDLA-Permissive-2.0](LICENSE-data) (proposed; ratification is gated on the first data release).

Raw participant trajectories are never released or licensed: they never cross a trust boundary
(`INV-RESIDENCY`).
