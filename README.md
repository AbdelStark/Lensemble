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
| [0017](docs/rfcs/RFC-0017-dynamic-env-ungameable-metrics.md) | Dynamic Env & Ungameable Ground-Truth Metrics | Draft | Synthetic control env and binding `state_probe_r2` usefulness gate |

Read order for the paper: 0002 → 0005 → 0001. Read order to build: 0001 → 0003 → 0004 → 0002 → 0005, then the subsystem RFCs 0007–0017, then 0006.

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
first GPU-backed Phase 2 HF Job
[`6a22ba68e6aa50b87b9ebef7`](https://huggingface.co/jobs/abdelstark/6a22ba68e6aa50b87b9ebef7)
ran three closed federated rounds from pinned commit
`4b446a558882f25e47ee6410a4c32982bbf33477` on `t4-small` and published
checkpoint/report artifacts to
[`abdelstark/lensemble-phase2-so100-checkpoint`](https://huggingface.co/abdelstark/lensemble-phase2-so100-checkpoint)
at revision `da52ef380ac87317c89e87f048d65bae65c16b9e`. The report records
schema v2 round metrics, final global hash
`8f1494fd9e57b7496daf96e379a3de1457a435080b81b9e0ea1d20a52f4827c4`,
participant dataset roots, `val_pred=1.513671025633812`,
`val_sigreg=0.15686095133423805`, `effective_rank=1.5215493440628052`,
`frame_drift_deg=10.538757949205232`, and `publication.blocker=None`. This is
still compact engineering evidence (`latent_dim=96`, `depth=4`,
`inner_horizon=1`).

The first Phase 2 downstream eval report is also published:
[`6a22c9e3ece949d7b3dca25a`](https://huggingface.co/jobs/abdelstark/6a22c9e3ece949d7b3dca25a)
ran `scripts/phase2_eval_checkpoint.py` from commit
`b57aed3da3b6250dce540da25b0bd65c391e68f4` against that checkpoint and pushed
`reports/phase2_downstream_eval_report.json` to the model repo revision
`021a461eb789700209fcb49e99bb9bcc5d84bfe5`. A checked-in copy lives at
[`docs/evidence/phase2_downstream_eval_report.json`](docs/evidence/phase2_downstream_eval_report.json).
It records `success_rate=0.5`, `effective_dim=1.0000066342911489`,
`planning_samples=1`, `planner_iterations=1`, and a `synthetic://toy` task
boundary.

The Phase 2 baseline/curve table is generated at
[`docs/evidence/phase2_baselines_curves_report.json`](docs/evidence/phase2_baselines_curves_report.json)
by [`scripts/phase2_curves_report.py`](scripts/phase2_curves_report.py). It
binds every curve point to source-report, config, checkpoint/global-model, and
run-manifest hashes. Current coverage is partial: anchored training scalars,
per-round update norms, a matched `lambda_anc=0` naive-FedAvg control, and
downstream eval rows are present. Missing local-only, centralized/pooled, and
Fork-A controls are explicitly marked blocked until matched public runs exist.
The final Phase 2 evidence bundle is checked in at
[`docs/evidence/phase2_evidence_bundle.json`](docs/evidence/phase2_evidence_bundle.json)
with the generated model card at
[`docs/evidence/phase2_model_card.md`](docs/evidence/phase2_model_card.md). The
checkpoint repo
[`abdelstark/lensemble-phase2-so100-checkpoint`](https://huggingface.co/abdelstark/lensemble-phase2-so100-checkpoint)
now contains `README.md`, `reports/phase2_evidence_bundle.json`,
`reports/phase2_model_card.md`, and `reports/phase2_baselines_curves_report.json`
at revision `eaf13136b42cde324758a191c98e377636ded7f8`.

Phase 3 is now completed on real HF Jobs GPU compute. The headline run
[`6a26885bece949d7b3dcb715`](https://huggingface.co/jobs/abdelstark/6a26885bece949d7b3dcb715)
ran an anchored federation on an `h200` HF Job from pinned commit `056f7407`:
ten closed federated rounds with four participants, all `0`-dropped, at
`latent_dim=256` and `num_tokens=196`, with per-round `secure_sum` secure
aggregation and DP accounting `(ε≈5.30, δ=1e-5, rdp, noise_multiplier=1.0,
clip_norm=0.5)`. It recorded final global hash
`bb31c0922de639cb9220c4cc5fc35d79aec719eb6fcedb09159bdff8cfb8fd43`, config hash
`27f2c77c9d47a7d053c01ab65f8d43aad79463b27d882f2d85ec28bc062cb2b2`, run-manifest
SHA-256 `21819c9b936468ffc38f943b4ce13ec2ac150d328410f503fa73d9014e040c9d`, and
per-round `effective_rank` ≈36–47 of 256 — the public-probe frame anchor holds
representational rank under DP federation. The four published training silos plus
a held-out split live in
[`abdelstark/lensemble-phase3-so100-silos`](https://huggingface.co/datasets/abdelstark/lensemble-phase3-so100-silos)
at revision `15f71911432b300dfdf41c998e27492e8c986be4`, and the checkpoint,
manifests, ledger, report, and pinned probe are published to
[`abdelstark/lensemble-phase3-consortium-checkpoint`](https://huggingface.co/abdelstark/lensemble-phase3-consortium-checkpoint)
at immutable revision `828e210cba4870b2be4ab573a5f0dd4ee30bae29`
(`publication.status: hf_jobs_release`).

Matched DP-off control probes (each an `a10g-large` job, six rounds,
`latent_dim=256`) make the gauge finding concrete on real SO-100 data: the
anchored probe (`abdelstark/lensemble-phase3-consortium-anchored-probe` @
`567755d2`) reduces inter-participant latent frame-drift to **48.97°** at
round 0, versus **180°** for naive FedAvg (`…-naive-fedavg` @ `1aace225`). Fork-A
frozen-encoder (`…-fork-a` @ `148e4217`) is the 0° safe-degrade with constant
`effective_rank` 2.39, and local-only silos (`…-local-only` @ `a696da17`) train
healthily (`effective_rank` ≈120) but diverge maximally (180°) without a shared
frame. The downstream eval report records real held-out SO-100 latent metrics
beyond the `synthetic://toy` boundary (final-round `effective_rank` ≈35.8/256),
and the evidence bundle is `published` at revision `828e210c` with four completed
controls, zero blocked controls, and nine artifact checks all `exists:true`.

This is consortium-engineering plus real federated-training evidence, **not** a
cryptographic proof of honest participant computation (RFC-0006 remains out of
scope) and not a paper-scale robotics result. The honest residual limitations:
at four participants × ~8.4M parameters the meaningful-DP regime is
gradient-noise-dominated, so the gauge contrast rests on the round-0 measurement
and the DP-off probes; the federated global representation collapses over rounds
at the default outer step with a random-init warm-start (real V-JEPA-2 weights
unvendored, [#96](https://github.com/AbdelStark/Lensemble/issues/96)), so
sustained non-collapsing federated training is a documented follow-up; and
closed-loop physical SO-100 task-success remains blocked pending stable-worldmodel
([#96](https://github.com/AbdelStark/Lensemble/issues/96)). Full detail is in
[`docs/roadmap/PHASE3.md`](docs/roadmap/PHASE3.md).

## Dynamic-Env Demo Status (#273)

RFC-0017 replaces the SO-100 proxy-usefulness story with a resident
ground-truth synthetic control env, `kinematic://swipe-dot`, and a single
binding metric: held-out `state_probe_r2`. The current dynamic-env run is a
useful educational systems demo of the Tapestry-like idea applied to JEPA world
models: sovereign synthetic participants, federated scratch training, control
baselines, DP/secure-aggregation observability, artifact contracts, and a
browser ONNX inference surface.

It is **not** a binding benchmark win. The federated checkpoint reaches
`state_probe_r2=0.8885337114`, beating random (`0.8082002401`) and naive-FedAvg
(`0.5502954721`), but local-only reaches `0.8838405609`. The federated margin
over local-only is only `0.0046931505`, below RFC-0017's required absolute
margin of `0.05`. The final dynamic-env benchmark bundle/model card is therefore
not published as a success artifact; the code path correctly rejects the
overclaim. Continue to frame this slice as a clean concept/demo path, not as
evidence that federated training materially outperforms local-only.

## Browser Federated Demo App (#294)

The browser federated demo app is now implemented as an educational systems
surface for the same dynamic-env framing. Run it locally with:

```bash
uv run lensemble demo federated --port 8765
```

The app supports a frontend-only simulator plus a local backend API mode. In
backend mode a host can create a run, configure max participants/quorum/rounds,
share a QR join URL, watch participant lifecycle state, start/abort/drop timed
out participants, collect browser-surrogate update metadata, close a
coordinator-style round, attach checkpoint-like and inference artifact metadata,
run the swipe-dot browser inference panel, and export a residency-safe evidence
JSON bundle.

The browser-local learner is deliberately a Web Worker surrogate over resident
synthetic swipe-dot samples. It submits only a versioned `browser-update/1`
metadata artifact: shape, sample count, norm, hash, runtime, source, run id,
participant id, and round. It does not upload raw observations, actions, state
labels, latents, tensors, or model weights. This proves the orchestration and
artifact contract, not production-grade browser training and not a benchmark
win. Full usage and architecture notes are in
[`docs/roadmap/BROWSER_FEDERATED_DEMO.md`](docs/roadmap/BROWSER_FEDERATED_DEMO.md).

## MVP Benchmarks / Results (#259)

The MVP ([#259](https://github.com/AbdelStark/Lensemble/issues/259)) is now corrected as a gauge-only
SO-100 result. With the M1 fixes — the frame anchor strengthened and pinned to the **fixed round-0
reference** (not the drifting global), the live Procrustes backstop wired into the coordinator over the
encoder terminal frame + predictor, and a tamed DiLoCo outer step — the anchored run reduces the
naive-FedAvg gauge failure. It does **not** prove downstream usefulness. Held-out magnitude collapse
(`~7.5e-6` latent variance; `thoughts/collapse_fix_probe.py`) and the central ceiling probe
(`thoughts/central_ceiling_probe.py`) show the SO-100 checkpoint is not a downstream-useful world model.
Three real HF Jobs `a10g-large` runs from scratch (`latent_dim=256`, `depth=8`, 224px, four SO-100 silos +
held-out silo4), relaxed-DP probe regime, simulated secure-aggregation:

| control | effective_rank (held-out) | val_pred (held-out) | frame_drift_deg | verdict |
|---|---|---|---|---|
| local-only (per-silo) | ~105 (healthy) | ~0.025 | 180 (inter-silo) | silos learn alone; gauges diverge |
| naive-FedAvg | 1.1 → ~1 (collapse) | 3 → **203 776** (explode) | **180** every round | catastrophic collapse |
| **anchored (M1)** | 2.6 → **14.8** (held, grows) | 1.4 → **22.2** (bounded) | 7–124 (controlled) | gauge held; downstream usefulness not shown |

The anchored federation **prevents the gauge collapse** (the #259 root cause): `effective_rank` holds and
grows where naive collapses to ~1, drift is controlled where naive is pinned at 180°, and `val_pred` stays
~4 orders of magnitude below naive. That is not a binding usefulness metric: `effective_rank` is
scale-invariant, `skill_vs_identity` is gameable, and the held-out latent magnitude collapse invalidates
the old “dramatically more usable” reading. In plain text: skill_vs_identity is gameable; effective_rank is
scale-invariant. The checked-in inference report is a proxy audit with
latent-MPC `success_rate=0.0`, reported as a negative result. Artifacts (checkpoint, per-round metrics,
ledger, benchmark report, inference report, corrected model card) are published to
[`abdelstark/lensemble-phase3-converged-checkpoint`](https://huggingface.co/abdelstark/lensemble-phase3-converged-checkpoint)
at immutable revision `a6f5a961…` (anchored run `3c2258ce…`).

Honest boundary: convergence is demonstrated in the **gauge sense** only. Held-out magnitude collapse
(`~7.5e-6`; `thoughts/collapse_fix_probe.py`), the central ceiling probe
(`thoughts/central_ceiling_probe.py`), gameable `skill_vs_identity`, and scale-invariant `effective_rank`
make the SO-100 downstream usefulness claim invalid. The dynamic-env RFC-0017 pivot now provides the
right ground-truth `state_probe_r2` demo path, but the current published run still misses the local-only
margin gate. Latent-space only; closed-loop physical task-success stays gated on
[#96](https://github.com/AbdelStark/Lensemble/issues/96).

## Working assumptions

Assumptions, all overridable:

- **Goal**: a research paper plus an open reference implementation; the corpus is written to be scientifically self-contained.
- **Fork B (end-to-end)** is the target; Fork A (frozen shared encoder) is the documented fallback if gauge control proves unstable at scale ([RFC-0002, Fork A fallback](docs/rfcs/RFC-0002-gauge-and-aggregation.md#fork-a-fallback)).
- **Phase 2 has two tracks**: the empirical scale/evaluation stream in
  [#200](https://github.com/AbdelStark/Lensemble/issues/200), and the cryptographic contribution-proof
  layer in [RFC-0006](docs/rfcs/RFC-0006-verifiable-contribution.md). Phase 1 stays proof-ready
  ([RFC-0006 §3](docs/rfcs/RFC-0006-verifiable-contribution.md#3-phase-1-proof-ready-requirements-cheap-to-honor-now))
  so the proof layer should not require reworking artifact contracts.
- **Phase 3 is the final operational consortium-training stream** in
  [#220](https://github.com/AbdelStark/Lensemble/issues/220) and
  [`docs/roadmap/PHASE3.md`](docs/roadmap/PHASE3.md): governed multi-party training with separate
  participant agents, a networked coordinator, secure aggregation and DP runtime controls, downstream
  evaluation, lifecycle reporting, and a final evidence bundle. It explicitly excludes the provenance
  ledger and RFC-0006 cryptographic-proof implementation.
- **Warm-start from released V-JEPA 2** — foundation-scale credibility without an INTELLECT-class pretraining bill, and a shared frame at $t{=}0$.
- **License**: code Apache-2.0, docs CC-BY-4.0, data CDLA-Permissive-2.0 — matching ecosystem norms (see [License](#license)).

## Repo layout

```
Lensemble/
├── AGENTS.md                          # coding-agent context and claim discipline
├── README.md                          # this file
├── SPEC.md                            # corpus entry point: index + executive summary
└── docs/
    ├── spec/                          # normative spec sections
    │   ├── 00-overview.md … 10-glossary.md
    │   └── conventions.md             # notation, invariants, shared contracts
    ├── rfcs/                          # RFC-0001 … RFC-0017 (decision records)
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
