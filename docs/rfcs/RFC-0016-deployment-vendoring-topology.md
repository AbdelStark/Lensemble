# RFC-0016 — Deployment, Vendoring & Topology

| | |
|---|---|
| **RFC** | 0016 |
| **Title** | Deployment, Vendoring & Topology |
| **Slug** | deployment-vendoring-topology |
| **Status** | Accepted |
| **Track** | Standards |
| **Authors** | @AbdelStark |
| **Created** | 2026-06-02 |
| **Target milestone** | Decisions: `v0.1`; rollout across `v0.1`–`v0.3` |
| **Area** | core |
| **Requires** | [RFC-0001](RFC-0001-architecture.md), [RFC-0009](RFC-0009-configuration-reproducibility.md), [RFC-0011](RFC-0011-secure-aggregation.md), [RFC-0013](RFC-0013-coordinator-runtime.md) |
| **Informs** | [RFC-0003](RFC-0003-federated-protocol.md), [RFC-0005](RFC-0005-evaluation.md), [RFC-0010](RFC-0010-artifact-checkpoint-format.md) |

## Summary

This RFC ratifies four foundation decisions that the rest of the corpus assumes but did not record: the
implementation is **Python-first** (including distributed-training orchestration), with a documented Rust
contingency; the reused ecosystem code (`stable-worldmodel`, `stable-pretraining`) is **vendored into a
`third_party/` tree** so it can be modified in place to ship a proof of concept quickly; the project runs
on a **single deployment substrate model** — one configuration source rendered to an in-process
simulation, Docker Compose, and Kubernetes — over the single [RFC-0013](RFC-0013-coordinator-runtime.md)
`Transport` seam; and the **infrastructure is defined as code**. It is the decision record behind issue
[#96](https://github.com/AbdelStark/Lensemble/issues/96). External-ecosystem facts cited here are
research findings and are marked *confirm against upstream* where not yet pinned in-tree.

## Motivation

The corpus specifies *what* Lensemble builds and *how the science works*, but leaves the engineering
substrate implicit: which language the orchestration is written in, how the heavily-reused
`stable-worldmodel` data/eval layer is integrated, how a contributor runs `N` nodes on a laptop, and how
the same system reaches a production cluster. Re-litigating these per subsystem is slow and produces
drift. The proof-of-concept goal demands the ability to **modify** the reused code in place now (a fork
is not used yet) while staying clean enough to contribute upstream later. These are cross-cutting
choices that span [RFC-0001](RFC-0001-architecture.md) (topology, layout, dependencies),
[RFC-0013](RFC-0013-coordinator-runtime.md) (the runtime and its transport), and
[RFC-0009](RFC-0009-configuration-reproducibility.md) (the config model), so they are recorded once here
rather than amended into three Accepted RFCs in place (see Alternatives Considered).

## Goals

- Lock Python as the implementation language for the whole stack, including outer-loop orchestration,
  with named, measurable conditions under which a partial Rust port is reconsidered.
- Define a `third_party/` vendoring policy that keeps vendored code modifiable, auditable against
  upstream, license-clean, and isolated from the public dependency DAG.
- Add `stable-pretraining` as a declared dependency and reconcile the dependency pins the reused code
  requires against [conventions §11](../spec/conventions.md#11-external-dependencies).
- Define a one-config-source deployment model: `LensembleConfig` renders to the in-process simulation,
  Docker Compose, and Kubernetes, all driving the identical RFC-0013 round state machine.
- Keep every choice consistent with the named invariants (`INV-AGG-DETERMINISM`, `INV-RESIDENCY`, the
  no-cycle layering DAG) and with the staged plan ([conventions §12](../spec/conventions.md#12-milestones-and-stages)).

## Non-Goals

- Implementing the runtime, transport, or secure aggregation themselves — those stay with their issues
  and RFCs ([RFC-0013](RFC-0013-coordinator-runtime.md), [RFC-0011](RFC-0011-secure-aggregation.md)).
- Writing the `pyproject.toml` (issue #71) or the Hydra config schema (issue #34); this RFC specifies the
  decisions and amendments they apply.
- Building a Rust kernel — recorded only as a triggered contingency.
- Mandating a specific cloud or Kubernetes distribution; this RFC fixes the model and the tooling
  classes, not a vendor.
- The Stage-D verifiable layer; only the Phase-1 proof-ready disciplines constrain the substrate.

## Proposed Design

### 1. Language and frameworks: Python-first

The implementation is Python (`>=3.11`, [conventions §11](../spec/conventions.md#11-external-dependencies)),
end to end, including the orchestration of distributed training.

- **Inner loop (intra-participant), for scale.** PyTorch FSDP2 — the `fully_shard` / DTensor /
  `MixedPrecisionPolicy` (bf16 compute, fp32 reduce) surface in `torch>=2.4` (the exact symbol paths are
  pinned by issue #71) — launched with `torchrun`. This is the only place the large-model-parallelism
  playbook applies ([RFC-0001 §5](RFC-0001-architecture.md#5-training-topology-two-level)).
- **Outer loop (inter-participant), for sovereignty.** A Python DiLoCo implementation (inner AdamW, outer
  Nesterov SGD over pseudo-gradients) driven by the
  [RFC-0013](RFC-0013-coordinator-runtime.md) `Coordinator`/`Participant`/`RoundState` machine behind the
  `Transport` Protocol. Rationale: DiLoCo communicates only every `H` inner steps
  ([RFC-0003 §2](RFC-0003-federated-protocol.md#2-round-structure-diloco-outer-loop)), so the control
  plane is I/O-bound and off Python's hot path; OpenDiLoCo demonstrated this regime at billion-parameter
  scale with single-digit communication overhead (*research finding; confirm against upstream*).

**Risks to monitor** (named budget bands; relate to
[08 — Performance Budget](../spec/08-performance-budget.md) and the metrics of
[RFC-0015](RFC-0015-observability-diagnostics.md)). The GIL can serialize work only if the coordinator
does heavy Python-side tensor work on the aggregation critical path; this is low-risk because
[RFC-0013 §4](RFC-0013-coordinator-runtime.md#4-determinism-wiring-inv-agg-determinism) keeps the
reduction single-threaded in fixed canonical order anyway (`INV-AGG-DETERMINISM`). The other watch item
is per-round `Δ_c` serialization/masking cost as the encoder scales from ViT-L toward 1.2B.

- `fed/round_overhead_ms` (non-compute round wall time), serialize+deserialize time per round,
  `fed/comm_bytes` (already a CI perf smoke), coordinator Python-thread utilization, and scheduling
  jitter vs the contributing-set size. `OPEN QUESTION:` the breach thresholds are set by measurement at
  Stage A/B, not asserted here (owner @AbdelStark).
- **Mitigation ladder, applied in order before any language change:** int8/FP16 pseudo-gradient
  quantization ([RFC-0003 §6](RFC-0003-federated-protocol.md#6-heterogeneity--fault-tolerance)) →
  free-threaded CPython (PEP 703/779; non-default, ~5–10% single-thread cost — *research finding*) →
  vectorize / native extension → Rust.

**Rust/Burn contingency (deferred, triggered).** A partial, kernel-level Rust port is reconsidered only
when (1) a monitored Python path on the aggregation/serialization critical path is the *proven*
bottleneck after the ladder above is exhausted, or (2) the Phase-2 proof surface
([RFC-0006 §2](RFC-0006-verifiable-contribution.md#2-the-provable-surface-and-what-stays-cheap)) wants a
small, auditable, memory-safe reduction kernel. Burn (Rust, swappable backends, loads `safetensors`/
PyTorch weights — *confirm against upstream*) is the candidate so [RFC-0010](RFC-0010-artifact-checkpoint-format.md)
artifacts survive the seam; tinygrad is recorded as a philosophy/edge idea, not a near-term option. A
wholesale inner-loop rewrite is out of scope: it forfeits FSDP2, the V-JEPA 2 warm-start
(`INV-WARMSTART-T0`), and the `stable-worldmodel` data/eval layer for no proven benefit.

**Tapestry: concepts, not primitives.** Project Tapestry (AI Alliance) informs the sovereignty/governance
framing and the rigid world-model training-regime vocabulary only; Lensemble keeps its own objective and
model authority ([RFC-0002](RFC-0002-gauge-and-aggregation.md)/[RFC-0008](RFC-0008-model-objective-numerics.md)),
since Tapestry targets a frontier LLM with no latent-gauge concern. `RISK:` the corpus previously assumed
Tapestry is C++; research indicates its product code is Python (torch + numpy) with Proposed-status ADRs
(*confirm against upstream*). Either way the Python-first decision stands on its own merits.

### 2. Vendoring: `third_party/` as modifiable, license-clean subtrees

Add a `third_party/` tree at the repo root holding each reused project at a recorded upstream commit, so
Lensemble can modify the code in place to ship the PoC fast. This is the ratified target (a fork is not
used now); a pinned PyPI/VCS dependency ("treat them as libraries") is the **fallback**, used only where
a constraint blocks committing source. This is a layout addition relative to
[conventions §1](../spec/conventions.md#1-repository-and-package-layout) and
[RFC-0001 §2](RFC-0001-architecture.md#2-module-map-reference-implementation), ratified by this RFC;
`third_party/` is a separately-licensed subtree and is **outside** the `lensemble` import DAG of
[RFC-0001 §3](RFC-0001-architecture.md#3-dependency-layering-no-cycles).

```
third_party/
  stable_worldmodel/        # vendored at a recorded SHA; data layer + envs + latent-MPC eval
    UPSTREAM.md             # source URL, vendored SHA + date, license SPDX + in-tree LICENSE path, mod log, sync procedure
    patches/*.patch         # local modifications, applied over a pristine snapshot (tree stays byte-identical to upstream)
  stable_pretraining/       # vendored at a recorded SHA; pretraining scaffold
    UPSTREAM.md  patches/*.patch
```

- **Both projects are vendored now.** `stable-pretraining` ships a real MIT `LICENSE` (*confirm at vendor
  time*). `stable-worldmodel`'s repository was missing a `LICENSE` file despite an MIT classifier in its
  metadata; the maintainers have **confirmed the license is MIT and that the missing file is a packaging
  mistake they will correct**, and confirmed it is usable today — so it is vendored now, with the
  `UPSTREAM.md` manifest recording the MIT SPDX, the maintainer confirmation (date + source), and a note
  that the in-tree `LICENSE` file is to be synced from upstream once published. No upstream LICENSE issue
  is opened by Lensemble (the maintainers own the fix).
- **Modifications live as a patch series** (`third_party/<project>/patches/*.patch`) over a pristine
  snapshot, never edited in place, so the vendored tree stays byte-identical to upstream at the recorded
  SHA, re-syncing is a clean re-clone + re-apply, and the diff against upstream is always inspectable
  (consistent with [conventions §9](../spec/conventions.md#9-determinism-dtype-device) and
  [RFC-0006 §3](RFC-0006-verifiable-contribution.md#3-phase-1-proof-ready-requirements-cheap-to-honor-now)).
- **Import boundary / shim.** A thin internal adapter (the exact module path is for issue #2 to fix —
  e.g. an adapter under `lensemble.data` for the episode/format readers of
  [conventions §8](../spec/conventions.md#8-core-data-types) and under `lensemble.eval` for the
  `world.evaluate` + planner factory of [RFC-0005 §3](RFC-0005-evaluation.md#3-downstream-metric--planning-success))
  imports from `stable_worldmodel`/`stable_pretraining` and re-exposes only the surface Lensemble depends
  on. No `third_party` symbol is re-exported from `lensemble.__init__`, confining upstream API churn to
  one module and preserving the no-cycle DAG.
- **Upstream-contribution path.** Keep an upstream remote; re-vendor by bumping the recorded SHA and
  re-applying patches; contribute fixes back so the local patch set shrinks over time. The procedure is
  documented in `UPSTREAM.md` so a contributor re-syncs without messaging the author
  ([conventions §13](../spec/conventions.md#13-authoring-conventions)).

### 3. Dependencies reconciled

[conventions §11](../spec/conventions.md#11-external-dependencies) and the
[09 — Release & Versioning](../spec/09-release-and-versioning.md) dependency table are amended (each
*to confirm against upstream*):

| Dependency | Constraint | Reason |
|---|---|---|
| stable-pretraining | pinned (vendored SHA) | pretraining scaffold reused alongside `stable-worldmodel`; previously undeclared |

`RISK:` the following are unverified research leads, reconciled when the vendored SHAs are confirmed —
do not pin until verified at vendor time: `stable-worldmodel` may require `lancedb>=0.30.0` /
`pylance>=4.0.0` at runtime against the looser `lance>=0.10` currently pinned; and its `lerobot://`
adapter may require Python `>=3.12` against the current `>=3.11` floor. Decision: keep the `>=3.11` floor
and treat the `lerobot` adapter as **out of Stage-A scope** (the `lance`/`hdf5` paths cover Stage A,
[RFC-0004 §1](RFC-0004-data-provenance.md#1-per-participant-data-layer)); revisit the floor only if
`lerobot` enters scope. Candidate vendored SHAs are research leads recorded in `UPSTREAM.md`, not pinned
here.

### 4. Deployment: one config source, three substrates, one transport

`LensembleConfig` ([RFC-0009 §2](RFC-0009-configuration-reproducibility.md#2-structured-configuration-tree))
is the single source of truth; every substrate is a *renderer* over it (participant count → replicas,
`FederationConfig.transport` → service wiring). All three substrates drive the identical RFC-0013 round
state machine over the one `Transport` Protocol
([RFC-0013 §5](RFC-0013-coordinator-runtime.md#5-control-plane-messages--transport)); this RFC extends
that seam with Compose and Kubernetes materializations, it does not invent a new one.

- **Layer 0 — in-process simulation (default; Stage B / `v0.2`).** An event-driven `InProcessTransport`
  that is deterministic, CPU-runnable, and fast. It is the canonical test/CI harness and the substrate
  the paper's ablation ladder runs on; it guards `INV-AGG-DETERMINISM` on CPU. The multi-process/
  container path is a **superset**, not a replacement — the same `Coordinator`/`Participant`/`RoundState`
  code runs over a real transport.
- **Layer 1 — Docker Compose (local multi-node).** One `compose.yaml` using `profiles` to select subsets
  (a `coordinator` service + a `participant` service scaled via `deploy.replicas`), a `gpu` profile
  (GPU passthrough via `deploy.resources.reservations.devices`, `capabilities: ['gpu']`) and a `cpu`
  profile for CI/laptops, and `healthcheck`/`depends_on` so participants join only after the coordinator
  is ready (`transport='network'` over loopback). Generated from `LensembleConfig.federation`, not
  hand-maintained.
- **Layer 2 — Kubernetes (production; Stage C+).** A coordinator Deployment/Job + a participant set
  (StatefulSet or a JobSet replicated group). The two-level topology of
  [RFC-0001 §5](RFC-0001-architecture.md#5-training-topology-two-level) maps onto K8s primitives as
  **inner = gang-scheduled** (a participant's FSDP/TP pod group: Volcano/Kueue all-or-nothing admission
  is correct there; Kubeflow Trainer v2 (`TrainJob`/`TrainingRuntime` on JobSet) and KubeRay
  (`RayJob`/`RayCluster`) are candidate inner-loop runtimes, evaluated not mandated) and **outer =
  elastic, NOT gang-scheduled** across participants, because a round proceeds with a quorum
  ([RFC-0013 §3](RFC-0013-coordinator-runtime.md#3-fault-tolerance--elasticity)). The TEE-attested
  aggregator backend ([RFC-0011 §5](RFC-0011-secure-aggregation.md#5-backend-b--tee-attested-aggregator))
  implies a confidential-compute node pool — a documented Stage-C requirement.
- **IaC tooling.** Helm packages the coordinator+participant deployment (versioned chart, rollback);
  Kustomize overlays per environment (local kind/minikube, staging, sovereign-prod). Cloud provisioning
  uses Pulumi-in-Python (Python-first; typed Kubernetes resources) with Terraform documented as the
  alternative. Every IaC artifact is parameterized from the same `LensembleConfig`, so
  config → simulation/Compose/Kubernetes is one pipeline.

### 5. Invariants the substrate must preserve

- `INV-AGG-DETERMINISM` — no transport may introduce nondeterministic reduction; the coordinator sums in
  canonical participant order regardless of arrival order
  ([RFC-0013 §4](RFC-0013-coordinator-runtime.md#4-determinism-wiring-inv-agg-determinism)). The
  determinism self-check ([RFC-0009](RFC-0009-configuration-reproducibility.md)) is the backstop.
- `INV-RESIDENCY` — moving from one process to many containers/pods makes raw-data egress a real risk;
  [RFC-0009 §3](RFC-0009-configuration-reproducibility.md#3-load-validation-and-override-semantics)
  forces `data.residency_enforced=True` when `transport=='network'`, and `ResidencyViolation` is
  fail-closed. K8s overlays must not co-locate participant data volumes across trust domains.
- The no-cycle import DAG ([RFC-0001 §3](RFC-0001-architecture.md#3-dependency-layering-no-cycles)) —
  vendored `third_party` code stays outside it, reached only through the shim.

## Alternatives Considered

- **Depend-not-vendor (use as libraries).** Cleaner dependency hygiene, but it forfeits in-place
  modifiability, which is the stated PoC requirement (the reused code needs local changes now). Kept as
  the fallback for any project a constraint blocks from source-vendoring.
- **Fork the upstream repos.** Rejected for now: a fork is heavier to track against the canonical repo
  and the decision is explicitly to integrate directly while staying clean; the patch-series-over-pinned-
  SHA approach gives the same auditability without a separate fork to maintain.
- **Rust-first (Burn/tinygrad) implementation.** Rejected: it forfeits the PyTorch/V-JEPA 2/
  stable-worldmodel ecosystem the project is built on, for a perf concern that is unproven at PoC scale.
  Retained as a triggered, partial-port contingency.
- **Ray as the outer-loop orchestrator.** Powerful (cluster scheduling, fault recovery) but a heavy
  dependency, and its lineage-based recovery and actor scheduling are not verified to preserve
  `INV-AGG-DETERMINISM`. Deferred: adopt only if a hand-rolled coordinator outgrows its scope, and only
  after verifying the determinism contract. KubeRay remains a candidate *inner-loop* runtime.
- **Gang-schedule everything on Kubernetes.** Rejected for the outer loop: all-or-nothing admission
  contradicts the elastic quorum-completion contract ([RFC-0013 §3](RFC-0013-coordinator-runtime.md#3-fault-tolerance--elasticity)).
  Gang scheduling is correct only inside a participant's inner pod group.
- **Terraform instead of Pulumi.** Broader provider catalog and registry maturity, but HCL is a separate
  language; Pulumi-in-Python keeps IaC in the project's language and can read existing Terraform state for
  incremental adoption. Terraform documented as the alternative.
- **Amend RFC-0001 + RFC-0013 in place instead of a new RFC.** Rejected: the decisions span two Accepted
  RFCs and introduce dependency/vendoring concerns that belong to neither cleanly; a standalone decision
  record is easier to review and supersede than scattered in-place edits.

## Drawbacks

- Vendoring adds maintenance: the patch series must be re-applied on each upstream bump, and a stale
  vendored tree can mask upstream fixes. Mitigated by the drift CI guard and a shrinking patch set as
  fixes land upstream.
- All-Python orchestration carries the monitored perf risk of §1; the mitigation ladder is the hedge.
- Supporting three substrates is surface area. Mitigated by staging strictly (the paper needs only the
  in-process simulation) and by the single-config-source model so the substrates share one code path.

## Migration / Rollout

Staged to [conventions §12](../spec/conventions.md#12-milestones-and-stages). `v0.1`: ratify these
decisions; add `third_party/` + the vendoring manifests + the import shim; amend the dependency table;
land the Compose `cpu` profile for local dev. `v0.2`: the in-process simulation is the only substrate the
scientific core needs; Compose is a developer convenience. `v0.3` (Stage C): the networked transport
(issue #45), the Kubernetes manifests (Helm + Kustomize), and the confidential-compute node pool for the
TEE backend. The Rust contingency is out of all v1.0 milestones unless a trigger condition fires.

## Testing Strategy

- **Vendoring-drift guard** (CI, relate to #65/#66): a test asserts each vendored tree equals its
  recorded upstream SHA with `patches/*.patch` applied — no silent local drift. CPU-only, tiny fixtures
  ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)).
- **License-presence guard**: CI fails if `third_party/<project>/` source is present without an in-tree
  `LICENSE` at the recorded SHA (enforces the license discipline rather than relying on review).
- **No-leak / layering test** (owned with #2): asserts no `stable_worldmodel`/`stable_pretraining` symbol
  is re-exported from `lensemble.__init__` and the [RFC-0001 §3](RFC-0001-architecture.md#3-dependency-layering-no-cycles)
  no-cycle DAG holds.
- **Substrate-parity test**: the same federated round produces the identical committed global-model hash
  under `InProcessTransport` and under the Compose loopback transport (transport portability does not
  perturb `INV-AGG-DETERMINISM`).
- **Compose smoke**: `docker compose --profile cpu up` brings up `N` participants + 1 coordinator on
  loopback and completes one round; `helm template` / `kustomize build` render without error.
- **Config-render test**: `LensembleConfig.federation` with `participant_count=N` renders a Compose file
  with `N` participant replicas and a K8s manifest set with the matching topology.

## Open Questions

- `OPEN QUESTION:` the numeric monitoring thresholds (`fed/round_overhead_ms` etc.) that trigger each
  mitigation step — set by measurement at Stage A/B (owner @AbdelStark; resolves in
  [08 — Performance Budget](../spec/08-performance-budget.md)).
- `OPEN QUESTION:` whether Ray (KubeRay/Ray Train) is adopted as the inner-loop runtime, contingent on
  verifying it preserves `INV-AGG-DETERMINISM` (owner @AbdelStark; Stage C).
- `OPEN QUESTION:` the inner-loop runtime choice on Kubernetes — bare `torchrun` vs Kubeflow Trainer v2
  vs KubeRay (owner @AbdelStark; Stage C).
- `OPEN QUESTION:` reconciling the `lance`/`pylance` pins and the `lerobot` Python-`>=3.12` requirement
  against the `>=3.11` floor, once the vendored SHA is confirmed (owner @AbdelStark; Stage A).
- The control-plane transport (gRPC vs HTTP) is deferred to
  [RFC-0013 Open Questions](RFC-0013-coordinator-runtime.md#open-questions).

## References

- Corpus: [RFC-0001](RFC-0001-architecture.md) (topology, layout, dependency layering),
  [RFC-0013](RFC-0013-coordinator-runtime.md) (`Transport`, elasticity, determinism),
  [RFC-0009](RFC-0009-configuration-reproducibility.md) (the config model + residency validation),
  [RFC-0011](RFC-0011-secure-aggregation.md) (aggregator backends),
  [RFC-0003 §6](RFC-0003-federated-protocol.md#6-heterogeneity--fault-tolerance) (quantization),
  [conventions §1/§9/§11/§12](../spec/conventions.md#1-repository-and-package-layout).
- Issue [#96](https://github.com/AbdelStark/Lensemble/issues/96) — the foundation issue this RFC records.
- Prior art (research findings; *confirm against upstream*): V-JEPA 2; `stable-worldmodel` /
  `stable-pretraining` (galilai-group); LeJEPA/LeWM (SIGReg); DiLoCo / OpenDiLoCo / INTELLECT (Prime
  Intellect); Project Tapestry (AI Alliance); PyTorch FSDP2; Kubeflow Trainer v2; Volcano; KubeRay;
  Helm / Kustomize; Pulumi; Burn; tinygrad; free-threaded CPython (PEP 703/779).
