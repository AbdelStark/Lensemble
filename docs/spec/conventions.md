# Conventions & Contracts

The normative reference for naming, notation, type contracts, invariants, milestones, and the
RFC/spec index used across the Lensemble specification corpus. Every spec section and RFC conforms to
it: same module names, same notation, same type names, same API signatures, same milestone names. When
another document cites "conventions §N", it refers to a numbered section here. This document is the
single home for the canonical mathematical notation (§2) and the named invariants (§7); other documents
reference them rather than restating them.

Section numbers are stable and cited corpus-wide; do not renumber them.

---

## 0. Project identity

- Name: Lensemble. One line: "Federated, end-to-end JEPA world models — trained across sovereign data,
  verifiable by construction."
- Etymology: *l'ensemble* (the whole / together) + the ML *ensemble* (many models acting as one), with
  *lens* (the perception encoder) in front.
- Thesis: train a single action-conditioned JEPA world model end-to-end (encoder and predictor
  co-trained — "Fork B") across many mutually-distrusting participants; raw data never leaves a
  boundary; only model deltas cross, aggregated under privacy, with a Phase-2 roadmap to cryptographic
  proof of contribution.
- Scientific core (the lead contribution): the latent gauge problem and its fix. The SIGReg-JEPA
  objective is invariant under O(d) rotations of the latent space, so independently-updated participants
  drift into mutually-rotated coordinate frames and naive weight-averaging is meaningless. Lensemble
  closes the gauge with a shared warm-start plus a light public-probe frame anchor, which also keeps the
  eventual proof-of-contribution circuit cheap.
- Verifiable contribution is the Phase-2 differentiator, not Phase-1 scope. Phase 1 ships "proof-ready"
  so Phase 2 needs no rework.
- Fork B (end-to-end) is the target; Fork A (frozen shared encoder, federate predictor only) is the
  documented safe-degrade fallback.

Corpus style: no marketing language, no hype, no emojis, no tool/vendor branding in normative prose
beyond the named ecosystem dependencies (§11). Active voice. Dense, signal-rich, honest over
overclaiming.

---

## 1. Repository and package layout

The Python import root is `lensemble`. Reference implementation layout:

```
lensemble/
  __init__.py             # public re-exports; __version__
  errors.py               # error taxonomy (§6)
  cli.py                  # Typer CLI app (§5)
  contracts/              # WMCP latent contract and embodiment conformance     [RFC-0007]
  model/                  # encoder, predictor, action heads, objective         [RFC-0008]
    encoder.py  predictor.py  action_head.py  objective.py  sigreg.py
  gauge/                  # frame anchoring, Procrustes alignment, drift         [RFC-0002]
    anchor.py  procrustes.py  drift.py
  federation/             # DiLoCo outer loop, round state machine, roles        [RFC-0003, RFC-0013]
    coordinator.py  participant.py  round.py  outer_optimizer.py
  aggregation/            # secure aggregation (masking / TEE), summation        [RFC-0011]
    secure_agg.py  masking.py
  privacy/                # DP clip+noise, (eps,delta) accountant                [RFC-0012]
    dp.py  accountant.py
  data/                   # data layer, window loader, adapters, residency, probe [RFC-0004]
    dataset.py  adapters/  residency.py  probe.py
  provenance/             # episode hashing, Merkle, contribution ledger         [RFC-0014]
    merkle.py  commit.py  ledger.py
  eval/                   # latent MPC planner, eval harness, metrics            [RFC-0005]
    mpc.py  harness.py  metrics.py
  config/                 # structured config schema, run manifest, seeding      [RFC-0009]
    schema.py  manifest.py  seed.py
  artifacts/              # checkpoint/artifact format, hashing, schema version   [RFC-0010]
    checkpoint.py  schema.py  hashing.py
  observability/          # structured logging, metrics emit, redaction          [RFC-0015]
    logging.py  metrics.py  redaction.py
  verify/                 # Phase-2 verifiable layer and public recomputation    [RFC-0006]
    recompute.py  stark.py
tests/                    # pytest suite (unit, property, integration, ml)
docs/                     # this specification corpus
configs/                  # Hydra config groups
third_party/              # vendored, modifiable upstream subtrees (outside the lensemble import DAG)  [RFC-0016]
  stable_worldmodel/      # data layer, envs, latent-MPC eval (vendored at a recorded SHA + patch series)
  stable_pretraining/     # pretraining scaffold (vendored at a recorded SHA + patch series)
deploy/                   # infrastructure-as-code: compose.yaml, Helm chart, Kustomize overlays  [RFC-0016]
```

Area taxonomy (one per major subsystem; used for issue labels): `core`, `contracts`, `model`, `gauge`,
`federation`, `aggregation`, `privacy`, `data`, `provenance`, `eval`, `config`, `artifacts`,
`observability`, `verify`, `cli`, `docs`, `ci`, `packaging`. Here `core` denotes the shared
foundation — `errors.py`, `config/`, and `observability/` — that other subsystems may depend on.

---

## 2. Mathematical notation

| Symbol | Meaning |
|---|---|
| $d$ | latent embedding dimension |
| $N$ | number of latent tokens emitted per clip by the encoder |
| $f_\theta$ | encoder (video ViT), parameters $\theta$ |
| $g_\phi$ | latent predictor, parameters $\phi$ |
| $h_\psi^{(c)}$ | per-participant action encoder / embodiment head, params $\psi$, participant $c$ |
| $f_{\text{ref}}$ | round-0 reference encoder (the warm-start), frozen for anchoring |
| $C$ | participant count; $c$ is the participant index |
| $H$ | inner horizon (local steps per outer round) |
| $A$ | SIGReg random projection (sketch) matrix; $s_t$ is the round sketch seed |
| $\mathcal{P}$ | public probe set; $p_i$ its points |
| $t_i$ | landmark target $= f_{\text{ref}}(p_i)$ (Variant A) |
| $E_{\text{ref}}$ | reference probe embeddings $= f_{\text{ref}}(\mathcal{P})$ |
| $Q \in O(d)$ | a gauge (orthogonal) rotation; $Q^\star$ the optimal Procrustes rotation |
| $\lambda_{\text{sig}}, \lambda_{\text{anc}}, \lambda_{\text{pred}}$ | SIGReg / anchor / prediction loss weights |
| $\Delta_c$ | pseudo-gradient of participant $c$ $= (\theta_c^{\text{local}},\phi_c^{\text{local}}) - (\theta_t,\phi_t)$ |
| $\eta_{\text{out}}$ | outer-optimizer learning rate (Nesterov) |
| $R_c$ | dataset Merkle root committed by participant $c$ |
| $d_{\text{cond}}$ | conditioning-embedding dimension (the code field name is `cond_dim`) |
| $(\varepsilon,\delta)$ | differential-privacy budget; $\sigma$ noise multiplier; $C_{\text{clip}}$ clip norm |

The per-local-step objective:
$$\mathcal{L} = \lambda_{\text{pred}}\,\mathbb{E}\lVert g_\phi(f_\theta(x_t),a_t) - \text{sg}[f_\theta(x_{t+1})]\rVert^2 + \lambda_{\text{sig}}\,\mathrm{SIGReg}_A(f_\theta(x)) + \lambda_{\text{anc}}\,\mathcal{L}_{\text{anchor}}(f_\theta;\mathcal{P},\{t_i\}).$$

The gauge transform $f_\theta \mapsto Qf_\theta,\ g_\phi \mapsto Qg_\phi Q^\top$ leaves $\mathcal{L}$ invariant.

---

## 3. Corpus structure and cross-reference conventions

Entry point: [`SPEC.md`](../../SPEC.md) (index + executive summary). Detail lives in `docs/spec/`
(the stable contract) and `docs/rfcs/` (the decision records).

| File | Title |
|---|---|
| [00-overview.md](00-overview.md) | Overview: thesis, goals, non-goals, success criteria |
| [01-architecture.md](01-architecture.md) | System architecture, module boundaries, data flow |
| [02-public-api.md](02-public-api.md) | Public API surface, contracts, versioning policy, CLI |
| [03-data-model.md](03-data-model.md) | Types, schemas, invariants, schema versioning |
| [04-error-model.md](04-error-model.md) | Error taxonomy, failure modes, recovery |
| [05-observability.md](05-observability.md) | Logging, metrics, tracing, redaction rules |
| [06-security.md](06-security.md) | Threat model, trust boundaries, secrets handling |
| [07-testing-strategy.md](07-testing-strategy.md) | Test pyramid, property/integration/ML tests, CI gates |
| [08-performance-budget.md](08-performance-budget.md) | Latency/throughput/memory budgets, profiling plan |
| [09-release-and-versioning.md](09-release-and-versioning.md) | SemVer, deprecation, changelog, license discipline |
| [10-glossary.md](10-glossary.md) | Canonical terms |
| [conventions.md](conventions.md) | This document |

Cross-reference convention:

- Link to another corpus document by relative path. From a `docs/spec/` file to an RFC use
  `../rfcs/RFC-NNNN-<slug>.md`; from a `docs/rfcs/` file to a spec section use `../spec/NN-<slug>.md`;
  to a sibling RFC use `RFC-NNNN-<slug>.md`.
- Append a `#anchor` fragment only when it has been checked against the target file's actual headings;
  a stale fragment is worse than none. Otherwise link the file and name the section number or title in
  the link text. Anchors within the same document are always safe to use.
- Never write "see above" or "see below"; cite the section.
- Spec sections cite RFCs for rationale; RFCs cite spec sections for the stable contract.

---

## 4. RFC index

Files live at `docs/rfcs/RFC-NNNN-<slug>.md`.

| RFC | Title | Slug | Status | Area |
|---|---|---|---|---|
| [0001](../rfcs/RFC-0001-architecture.md) | Architecture & System Overview | architecture | Accepted | core |
| [0002](../rfcs/RFC-0002-gauge-and-aggregation.md) | The Latent Gauge & Frame-Anchored Aggregation | gauge-and-aggregation | Accepted | gauge |
| [0003](../rfcs/RFC-0003-federated-protocol.md) | Federated Training Protocol | federated-protocol | Accepted | federation |
| [0004](../rfcs/RFC-0004-data-provenance.md) | Data, Sovereignty & Provenance | data-provenance | Accepted | data |
| [0005](../rfcs/RFC-0005-evaluation.md) | Evaluation & Benchmark Protocol | evaluation | Accepted | eval |
| [0006](../rfcs/RFC-0006-verifiable-contribution.md) | Verifiable Contribution | verifiable-contribution | Draft · Phase 2 (Deferred) | verify |
| [0007](../rfcs/RFC-0007-wmcp-latent-contract.md) | WMCP Latent Contract & Embodiment Adapters | wmcp-latent-contract | Accepted | contracts |
| [0008](../rfcs/RFC-0008-model-objective-numerics.md) | Model, Objective & Numerical Contracts | model-objective-numerics | Accepted | model |
| [0009](../rfcs/RFC-0009-configuration-reproducibility.md) | Configuration, Run Manifest & Reproducibility | configuration-reproducibility | Accepted | config |
| [0010](../rfcs/RFC-0010-artifact-checkpoint-format.md) | Checkpoint & Artifact Format | artifact-checkpoint-format | Accepted | artifacts |
| [0011](../rfcs/RFC-0011-secure-aggregation.md) | Secure Aggregation Protocol | secure-aggregation | Accepted | aggregation |
| [0012](../rfcs/RFC-0012-differential-privacy.md) | Differential Privacy Accounting | differential-privacy | Accepted | privacy |
| [0013](../rfcs/RFC-0013-coordinator-runtime.md) | Coordinator & Participant Runtime | coordinator-runtime | Accepted | federation |
| [0014](../rfcs/RFC-0014-provenance-commitments.md) | Provenance Commitments & Merkle Scheme | provenance-commitments | Accepted | provenance |
| [0015](../rfcs/RFC-0015-observability-diagnostics.md) | Observability, Diagnostics & Telemetry | observability-diagnostics | Accepted | observability |
| [0016](../rfcs/RFC-0016-deployment-vendoring-topology.md) | Deployment, Vendoring & Topology | deployment-vendoring-topology | Accepted | core |

Every RFC follows this section order: Summary, Motivation, Goals, Non-Goals, Proposed Design,
Alternatives Considered, Drawbacks, Migration / Rollout, Testing Strategy, Open Questions, References.
The header table carries: RFC, Title, Slug, Status, Track, Authors, Created, Target milestone, Area,
Requires. RFC lifecycle: Draft → Accepted → Superseded by RFC-MMMM.

---

## 5. Public API surface

Python ≥ 3.11. Public top-level (`lensemble/__init__.py` re-exports); the authoritative contract is in
[02-public-api.md](02-public-api.md).

```python
__version__: str  # SemVer

from lensemble.config import LensembleConfig, RunManifest, load
def train_local(config: LensembleConfig) -> "RunResult": ...

from lensemble.federation import Coordinator, Participant, RoundState
from lensemble.model import build_encoder, build_predictor, build_action_head, Objective
from lensemble.eval import evaluate, Planner
from lensemble.gauge import frame_drift, procrustes_align
from lensemble.provenance import commit_dataset, DatasetCommitment, ContributionLedger
from lensemble.verify import recompute_alignment
```

CLI (`lensemble`, Typer): `train`, `federate coordinator|participant`, `eval`, `probe build|pin|verify`,
`commit dataset`, `drift`, `verify recompute|prove`, `doctor`. Every command accepts `--config` and
Hydra-style `key=value` overrides and emits a `RunManifest`. Names under a `_internal` module or
prefixed `_` are private and unversioned. Pre-1.0 the public surface may change across minors with a
deprecation note; at 1.0 the names above are frozen under SemVer (§10).

---

## 6. Error taxonomy

Base `LensembleError(Exception)`; every error carries `.code` (a `LensembleErrorCode` enum value) and
`.remediation: str`. The authoritative catalog is in [04-error-model.md](04-error-model.md).

- `ConfigError` — invalid/inconsistent configuration.
- `ContractViolation` — WMCP nonconformance (latent shape/dtype/semantics or `ActionSpec` mismatch).
- `ResidencyViolation` — attempt to emit raw observation/action/private-embedding across a boundary
  (security-critical; never caught-and-ignored).
- `GaugeError` → `FrameDriftExceeded`, `DegenerateProcrustes`.
- `AggregationError` → `SecureAggregationError`, `NonDeterministicAggregation`.
- `PrivacyBudgetExceeded`.
- `ProvenanceError` → `CommitmentMismatch`, `MerkleVerificationError`.
- `ArtifactError` → `SchemaVersionMismatch`, `CheckpointIntegrityError`.
- `RoundError` → `FaultToleranceExceeded`.
- `ProbeError` — probe hash mismatch / under-coverage.
- `EvaluationError`.

Never use a bare `except`; never swallow `ResidencyViolation`, `CommitmentMismatch`, or
`NonDeterministicAggregation`; validate at boundaries (config load, message ingress, artifact load,
dataset ingest) and raise typed errors with remediation.

---

## 7. Named invariants

Referenced by these identifiers corpus-wide. Each is stated, where enforced, in the owning document.

- `INV-RESIDENCY`: no raw observation/action/private-embedding tensor is serialized into any outbound
  message or artifact that crosses a trust boundary. Enforced by `lensemble.data.residency`.
- `INV-WARMSTART-T0`: at round 0 every participant's encoder weights are hash-identical to the pinned
  warm-start (the gauge is closed at $t{=}0$).
- `INV-SKETCH-CONSISTENCY`: all participants in round $t$ use the identical projection matrix $A$
  derived from the broadcast seed $s_t$.
- `INV-AGG-DETERMINISM`: the outer step is a pure, bitwise-reproducible function of (committed deltas,
  round seed, prior global params). No nondeterministic reductions on the aggregation path.
- `INV-PROBE-PIN`: the probe content hash equals the hash committed in `RoundOpen`; landmark targets
  derive only from $f_{\text{ref}}$ (round-0 encoder).
- `INV-COMMIT-BINDING`: every released $\Delta_c$ is bound to exactly one dataset Merkle root $R_c$.
- `INV-CHECKPOINT-HASH`: every committed $(\theta_t,\phi_t)$ artifact's content hash equals the
  `Commitment`/`RoundClose` hash.
- `INV-DP-BOUND`: after clipping and before noising, $\lVert\Delta_c\rVert \le C_{\text{clip}}$.
- `INV-WMCP`: every `LatentState` conforms to the pinned `wmcp_version`; every `ActionSpec` is
  validated before an action head is constructed.
- `INV-ACTIONHEAD-LOCAL`: per-embodiment action heads $h_\psi^{(c)}$ are never broadcast or aggregated.

---

## 8. Core data types

`LatentState` (WMCP) · `ActionSpec` · `Episode` · `Transition` (= $(o_t,a_t,o_{t+1})$) · `Window`
(fixed `num_steps`) · `PseudoGradient` (flat delta + L2 norm + bound `dataset_root`) · `GlobalState`
($\theta,\phi$ refs + round index + sketch seed + probe hash) · `RoundState` · `DatasetCommitment`
(Merkle root $R_c$ + episode count + WMCP metadata) · `ModelArtifact`/`Checkpoint` (schema-versioned,
hash-committed) · `RunManifest` (config hash, seeds, env, versions, git SHA) · `EvalReport` ·
`FrameDriftReport` · `ContributionRecord` (round, participants, roots, global hash). The authoritative
schemas are in [03-data-model.md](03-data-model.md); `LatentState`/`ActionSpec` are owned by
[RFC-0007](../rfcs/RFC-0007-wmcp-latent-contract.md). The canonical WMCP version literal is
`"wmcp-1.0.0"`.

Serialization: structured configs are frozen dataclasses validated via OmegaConf (Hydra). On-disk
metadata (manifests, commitments, artifact headers, reports) are JSON validated by pydantic v2 models
with an explicit integer `schema_version`. Tensors/weights use `safetensors`. Episodes use the
`stable-worldmodel` data layer (`lance` default, `hdf5` portable, `lerobot://` adapter).

---

## 9. Determinism, dtype, device

- Compute dtype default bf16 forward; fp32 master weights and loss/statistic accumulation.
- The aggregation/outer-step path is bitwise-deterministic given its inputs (`INV-AGG-DETERMINISM`,
  proof-readiness per [RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md)): fixed reduction order,
  fp32 with fixed summation order (or fp64), no atomics. A determinism self-check runs each outer step;
  failure raises `NonDeterministicAggregation`.
- Inner training determinism is best-effort and seed-pinned; full determinism is gated by a config flag
  (`torch.use_deterministic_algorithms`).
- Device: CUDA primary; a CPU fallback path runs the small CI configs. Tests must pass on CPU.
- Seeding: one root seed derives component seeds (python/numpy/torch/cuda) and per-round sketch seeds
  $s_t = \mathrm{derive}(\text{root\_seed}, t)$ deterministically. All seeds recorded in `RunManifest`.

---

## 10. Versioning and schema policy

- Package SemVer. Pre-1.0: minor tracks milestone (0.1, 0.2, 0.3); 1.0 freezes the public API of §5.
- `schema_version: int` on every on-disk artifact; forward-compatible readers, explicit migrations,
  `SchemaVersionMismatch` on unknown/too-new versions.
- `wmcp_version: str` on the latent contract (canonical literal `"wmcp-1.0.0"`); conformance gates on it.
- Deprecation: pre-1.0 deprecate for one minor then remove; post-1.0 keep two minors. A changelog entry
  (Keep a Changelog style) is required for every user-visible change.
- RFC lifecycle: Draft → Accepted → Superseded by RFC-MMMM.

The authoritative policy is in [09-release-and-versioning.md](09-release-and-versioning.md).

---

## 11. External dependencies

| Dependency | Constraint | Reason |
|---|---|---|
| Python | `>=3.11` | structural typing, perf, `tomllib` |
| torch | `>=2.4,<3` | autograd, FSDP2, differentiable SVD, deterministic algorithms |
| numpy | `>=1.26` | array/stat ops |
| safetensors | `>=0.4` | safe, mmap-able weight serialization (no pickle) |
| V-JEPA 2 weights | pinned release | encoder warm-start + AC recipe; the $t{=}0$ frame anchor |
| LeJEPA / LeWM (SIGReg) | pinned | the objective (random-projection + characteristic-function Gaussianity) |
| stable-worldmodel | pinned (vendored, [RFC-0016](../rfcs/RFC-0016-deployment-vendoring-topology.md)) | data layer (`lance`/`hdf5`/`lerobot`), envs, latent-MPC eval |
| stable-pretraining | pinned (vendored, [RFC-0016](../rfcs/RFC-0016-deployment-vendoring-topology.md)) | pretraining scaffold reused alongside stable-worldmodel |
| pylance | `>=0.10` | the Lance columnar format (import name `lance`) — append-friendly fast indexed episode reads (default format). The bare `lance` PyPI name is an unrelated typosquat; depend on `pylance`. |
| h5py | `>=3.10` | portable single-file episode format |
| hydra-core / omegaconf | `>=1.3` / `>=2.3` | structured config groups + overrides |
| pydantic | `>=2,<3` | typed validation of on-disk metadata schemas |
| typer / rich | recent | CLI + console rendering |
| opacus (or a vendored RDP/PRV accountant) | pinned | DP accounting reference; abstracted behind `privacy.accountant` |
| blake3 / hashlib | recent / stdlib | hashing; canonical commitment hash = SHA-256 (Phase 1) |
| pytest / hypothesis / pytest-benchmark | recent | unit/property/perf tests |
| ruff / pyright | recent | lint + type-check CI gates |
| Stwo (Circle-STARK prover) | Phase-2 only | aggregation-correctness proof ([RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md)) |

The Phase-1 canonical commitment hash is SHA-256 (conservative, interoperable). Migrating to a
STARK-friendly hash (e.g. Poseidon2) for Phase 2 to keep the proof circuit cheap is an open question;
owner `@AbdelStark`, resolution in [RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md) / Stage D.

---

## 12. Milestones and stages

| Milestone | Stage | Content |
|---|---|---|
| `v0.1` | A | Single-site, warm-started, ViT-L/~300M end-to-end SIGReg + AC predictor on pooled robot data; latent-MPC eval (centralized upper bound). Plus foundational scaffolding: package skeleton, config system, data layer, WMCP contract, model+objective, eval harness, observability, artifact format, error taxonomy, CI, packaging. |
| `v0.2` | B | Simulated federation on one cluster: DiLoCo outer loop, frame anchor (Layers 1–4), Procrustes backstop, simulated secure aggregation + DP, the frame-drift diagnostic, the full ablation ladder and non-IID/scale sweeps. The scientific core. |
| `v0.3` | C | Two real sovereign nodes over a network boundary: real secure aggregation + DP, residency enforcement, fault tolerance/elasticity, contribution ledger. The sovereignty demonstration. |
| `v1.0` | — | Hardening: frozen public API, complete docs + reproducibility package, release automation, Fork A fallback supported and tested, proof-ready guarantees verified end-to-end. |

Out of the v1.0 scope (tracked as future work, not filed as implementable issues): Stage D (the actual
STARK/TEE verifiable layer — Phase 2) beyond the proof-ready disciplines, and Stage E (own
foundation-scale federated video pretraining). The proof-ready engineering disciplines (deterministic
aggregation, hash commitments, Merkle roots, pinned probe, public recomputation) are in scope for
v0.1–v1.0.

---

## 13. Authoring conventions

- Every contract has a type signature or schema, not prose alone.
- Every invariant is named (§7) and stated where enforced.
- Every failure mode is enumerated with the system's response (which error from §6, what recovery).
- Every external dependency is named with a version constraint and reason (§11).
- No `TBD`: open items are written `OPEN QUESTION:` with an owner and a resolution path (a Stage,
  milestone, or follow-up RFC). Shaky sections are marked `RISK:` with a resolution plan.
- ASCII diagrams are accompanied by prose that says the same thing.
- A new contributor must be able to implement from a document without messaging the author.
