# 09 — Release & Versioning

This section is the normative policy for how Lensemble names versions, freezes its public surface,
deprecates symbols, records changes, versions on-disk artifacts and contracts, builds and publishes
the package, releases the research-artifact bundle that makes the centralized/local/federated triple
reproducible, governs contributions and RFC lifecycle, and licenses code, docs, and data. Rationale
for individual mechanisms lives in the RFCs cited inline; this section holds the stable rules a
maintainer applies on every release.

Scope boundary: the milestone-to-stage mapping (v0.1 → Stage A, v0.2 → Stage B, v0.3 → Stage C,
v1.0 → hardening) is the canonical schedule from the [conventions document](conventions.md). Stage D (the STARK/TEE
verifiable layer) and Stage E (own foundation-scale federated pretraining) are out of v1.0 scope and
are tracked as future work, not as implementable release content. The proof-*ready* engineering
disciplines that Phase 2 builds on — deterministic aggregation, hash commitments, Merkle roots, the
pinned probe, public recomputation — ARE in scope for v0.1–v1.0 and are versioned here.

## 1. Semantic versioning policy

Lensemble follows Semantic Versioning 2.0.0 for the Python package version string exported as
`lensemble.__version__: str`. The version is the single source of truth for compatibility promises on
the public API surface defined in [02 — Public API](02-public-api.md) and reproduced from the [conventions document](conventions.md)'s
public-API contract.

| Component | Format | Bump rule |
|---|---|---|
| MAJOR | `X` | Incompatible change to the frozen public API (only at and after 1.0). |
| MINOR | `Y` | Backward-compatible feature addition; pre-1.0 a minor tracks a milestone. |
| PATCH | `Z` | Backward-compatible bug fix; no public-API change. |

Pre-1.0 minor-tracks-milestone schedule (canonical, from the [conventions §12](conventions.md#12-milestones-and-stages) milestone table):

| Version | Milestone | Stage | Released content |
|---|---|---|---|
| `0.1.z` | v0.1 | A | Single-site warm-started end-to-end SIGReg + AC predictor; latent-MPC eval; full foundational scaffolding (package skeleton, config system, data layer, WMCP contract, model + objective, eval harness, observability, artifact format, error taxonomy, CI, packaging). |
| `0.2.z` | v0.2 | B | Simulated federation on one cluster: DiLoCo outer loop, frame anchor (Layers 1–4), Procrustes backstop, simulated secure aggregation + DP, the frame-drift diagnostic, the ablation ladder, non-IID and scale sweeps. |
| `0.3.z` | v0.3 | C | Two real sovereign nodes over a network boundary: real secure aggregation + DP, residency enforcement, fault tolerance / elasticity, contribution ledger. |
| `1.0.z` | v1.0 | — | Hardening: frozen public API, complete docs + reproducibility package, release automation, Fork A fallback supported and tested, proof-ready guarantees verified end-to-end (RFC-0006 §3). |

Pre-1.0 (`0.y.z`) semantics: under SemVer, `0.y` makes no compatibility promise on `y` bumps. Lensemble
narrows this with the explicit policy below — a `0.y` minor MAY change the public surface, but only with
a recorded deprecation (§2) and a changelog entry (§3). The version string is set in exactly one place,
`pyproject.toml`'s `[project].version`, and re-exported by `lensemble/__init__.py` (read at build time;
no second hand-maintained literal). A mismatch between the two is a release-blocking CI failure (§5).

1.0 is the freeze point: at 1.0 every name listed in the [conventions document](conventions.md) public-API surface — `train_local`,
`Coordinator`, `Participant`, `RoundState`, `build_encoder`, `build_predictor`, `build_action_head`,
`Objective`, `evaluate`, `Planner`, `frame_drift`, `procrustes_align`, `commit_dataset`,
`DatasetCommitment`, `ContributionLedger`, `recompute_alignment`, `LensembleConfig`, `RunManifest`, and
`__version__` — is frozen under the MAJOR-bump rule. See [02 — Public API §stability](02-public-api.md).

Rationale for tracking milestones in minors during 0.y is in ([RFC-0001](../rfcs/RFC-0001-architecture.md))
(staged plan A–E) and ([RFC-0009](../rfcs/RFC-0009-configuration-reproducibility.md)).

## 2. Public-API stability and deprecation policy

Public versus private surface (the rule the version contract is enforced against):

- Public, versioned: every symbol re-exported from `lensemble/__init__.py` and the documented submodule
  entry points in [02 — Public API](02-public-api.md), plus the CLI commands `train`,
  `federate coordinator|participant`, `eval`, `probe build|pin|verify`, `commit dataset`, `drift`,
  `verify recompute|prove`, `doctor`.
- Private, unversioned: any module named `_internal` or any name prefixed with a single underscore.
  These may change in any release, including a PATCH, without notice. Importing a private symbol is
  unsupported and gives no compatibility guarantee.

Deprecation windows:

| Phase | Window before removal | Mechanism |
|---|---|---|
| Pre-1.0 (`0.y`) | one minor (deprecate in `0.y`, remove no earlier than `0.(y+1)`) | `DeprecationWarning` + changelog `Deprecated` entry |
| Post-1.0 (`X.y`) | two minors (deprecate in `X.y`, remove no earlier than `X.(y+2)`) | `DeprecationWarning` + changelog `Deprecated` entry; removal only at the next MAJOR if the window straddles a major |

Deprecation-warning mechanism (contract):

```python
import warnings

def _deprecated(symbol: str, *, since: str, removal: str, replacement: str | None) -> None:
    """Emit a stacklevel-correct DeprecationWarning. Called at the top of a deprecated
    public callable, or via a decorator on a deprecated class/function.

    since:       version that introduced the deprecation, e.g. "0.2.0".
    removal:     earliest version the symbol may be removed, e.g. "0.3.0".
    replacement: dotted path of the successor, or None if removed outright.
    """
    msg = f"{symbol} is deprecated since {since} and may be removed in {removal}."
    if replacement is not None:
        msg += f" Use {replacement} instead."
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
```

Rules:

- A `DeprecationWarning` carries the introducing version, the earliest removal version, and the
  replacement (or its absence). The warning text is also the body of the changelog `Deprecated` entry.
- Deprecating a public symbol requires a changelog `Deprecated` entry in the same release (§3).
- Removing a symbol requires that its deprecation window has fully elapsed AND a changelog `Removed`
  entry. Removal before the window elapses is a release-blocking review failure.
- A behavioral change to a public symbol that is not a bug fix (a contract change in
  [02 — Public API](02-public-api.md), e.g. a precondition tightened or a postcondition altered) is a
  MAJOR bump post-1.0 and a documented deprecation-then-change pre-1.0.
- Renaming a CLI command or removing a flag follows the same window; the old name emits a
  `DeprecationWarning` to stderr and is aliased to the new one until removal.

Contract-bearing constants travel with the version surface even though they are not callables:
`wmcp_version` (§4) and the artifact `schema_version` (§4) each have their own compatibility rule; a
public-API stability promise does NOT cover an on-disk-format change, which is governed by §4.

## 3. Changelog discipline

Lensemble keeps a top-level `CHANGELOG.md` in the Keep-a-Changelog format. Every user-visible change
requires a changelog entry in the pull request that introduces it; a release-blocking CI check (§5)
fails a PR that touches public surface, CLI, on-disk schema, or documented behavior without adding an
entry under `## [Unreleased]`.

Categories (the only permitted headings under a version):

| Category | Use |
|---|---|
| `Added` | New public symbol, CLI command/flag, config group, or metric. |
| `Changed` | Backward-compatible change to existing documented behavior. |
| `Deprecated` | A public symbol/flag scheduled for removal (carries `since`/`removal`/replacement, §2). |
| `Removed` | A symbol/flag removed after its deprecation window elapsed. |
| `Fixed` | A bug fix with no public-API change. |
| `Security` | A fix to a residency, secrets-handling, or supply-chain issue (see [06 — Security](06-security.md)). |

Discipline rules:

- The `## [Unreleased]` block accumulates entries between releases. At release time the maintainer
  retitles it `## [X.Y.Z] - YYYY-MM-DD` and opens a fresh `## [Unreleased]`.
- Entries are imperative and reference the affected symbol or subsystem area label (`area:gauge`,
  `area:federation`, etc., from the [conventions document](conventions.md) area taxonomy) and, where applicable, the RFC and the invariant
  by its `INV-*` id (§7 of the [conventions document](conventions.md)). Example: "Changed: `procrustes_align` now raises
  `DegenerateProcrustes` instead of returning NaN on a rank-deficient target (area:gauge, RFC-0002)."
- A change that alters a versioned invariant's enforcement point or a schema version MUST name the
  invariant and the version bump in the entry.
- The changelog is the human-readable counterpart to the machine-readable `RunManifest` and artifact
  headers; the two never disagree about which versions a release carries.

## 4. Artifact, contract, and RFC versioning

Three version axes exist below the package SemVer, each with its own rule. They are recorded together
in every `RunManifest` and artifact header so a reader can reconstruct exactly which formats a release
produced. Schemas for these objects are defined in [03 — Data Model](03-data-model.md).

### 4.1 Artifact `schema_version` (integer)

Every on-disk metadata object — `RunManifest`, `DatasetCommitment`, the checkpoint header, `EvalReport`,
`FrameDriftReport`, `ContributionRecord` — is a pydantic v2 model carrying an explicit integer field
`schema_version: int`. The construction and hashing of the checkpoint/artifact header is specified in
([RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md)); the metadata schemas are catalogued in
[03 — Data Model](03-data-model.md).

Policy (canonical, from the [conventions document](conventions.md) versioning rule):

- Readers are forward-compatible within a major package line: a reader accepts any `schema_version`
  less than or equal to the version it was built for, applying a registered migration function per
  intervening bump.
- An unknown or too-new `schema_version` is fail-closed: the loader raises `SchemaVersionMismatch`
  (subclass of `ArtifactError`, [conventions §6](conventions.md#6-error-taxonomy)) with a remediation naming the maximum supported version. The
  loader never silently coerces or guesses.
- A migration is a pure function `migrate_vN_to_vN1(doc: dict) -> dict` registered in an ordered
  migration chain; loading a `schema_version = k` document into a version-`m` reader (`k < m`) applies
  `migrate_v(k)_to_v(k+1) ∘ … ∘ migrate_v(m-1)_to_v(m)`. Each migration step is covered by a round-trip
  test (see [07 — Testing Strategy](07-testing-strategy.md)).
- A `schema_version` bump is a changelog `Changed` (or `Added`) entry and, if it changes a reader's
  acceptance set, is called out as such. Pre-1.0 the schema may change with a bump and a migration;
  there is no removal of an old reader path within a major line.

`CheckpointIntegrityError` (also `ArtifactError`) fires when the recomputed content hash of a checkpoint
does not equal its committed `content_hash` (`INV-CHECKPOINT-HASH`, enforced in `artifacts.hashing` /
`artifacts.checkpoint`); this is a tamper/corruption signal, distinct from a version mismatch. Tensors
and weights are serialized with `safetensors` only — never `pickle`/`torch.save` — both for the
no-arbitrary-code-execution property and because it keeps the content hash deterministic across
platforms (`INV-CHECKPOINT-HASH`; see [RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md) and
[06 — Security](06-security.md)).

### 4.2 WMCP `wmcp_version` (string)

The shared latent contract carries `wmcp_version: str`, recorded on every `LatentState` and pinned in
each `RunManifest`, checkpoint header, and `DatasetCommitment` ([conventions §8](conventions.md#8-core-data-types)). Conformance gates on it
(`INV-WMCP`, enforced in `contracts/`): a `LatentState` whose `wmcp_version` does not match the pinned
version, or whose shape/dtype/semantics do not conform, raises `ContractViolation` ([conventions §6](conventions.md#6-error-taxonomy)). The full
contract and its version-gating semantics are in ([RFC-0007](../rfcs/RFC-0007-wmcp-latent-contract.md)).

Policy: `wmcp_version` is an independent string version (a contract is not the package). A participant
MAY only join a federation round whose advertised `wmcp_version` it conforms to; a mismatch is rejected
at the join boundary with `ContractViolation`, before any model state is exchanged. Contract extension
is additive-optional within a version; a breaking contract change bumps `wmcp_version` and is a package
MINOR (pre-1.0) or a documented MAJOR concern (post-1.0). A `wmcp_version` change always appears in the
changelog with the affected `area:contracts` label.

### 4.3 RFC lifecycle

RFCs in `docs/rfcs/RFC-NNNN-<slug>.md` carry a `Status` of `Draft → Accepted → Superseded by RFC-MMMM`.
The current index, numbers, slugs, and statuses are the canonical set in the [conventions document](conventions.md) RFC index (RFC-0001..0015;
RFC-0006 is `Draft · Phase 2 (Deferred)`, the rest `Accepted`). Rules:

- An `Accepted` RFC's normative content changes only via a follow-up RFC that supersedes it; the old RFC
  is then marked `Superseded by RFC-MMMM` and retained (never deleted) for the audit trail.
- A spec section under `docs/spec/` cites RFCs for rationale; an RFC cites spec sections for the stable
  contract. A normative change to a contract therefore lands as both an RFC change and the corresponding
  spec-section edit, in the same release, with one changelog entry.
- The RFC `Target milestone` field ties the RFC to a release line (e.g. RFC-0011 secure aggregation is
  `v0.2` simulated → `v0.3` real); shipping the milestone is what moves the contract from specified to
  implemented, not the RFC status alone.

## 5. Release process

### 5.1 Build and package

Build configuration lives in `pyproject.toml` using the PEP 517 / PEP 621 metadata layout. The package
is a pure-Python source-and-wheel build (no compiled extensions in the reference implementation; the
`jepa-rs`/`lewm-rs` verifiable reference path and the Phase-2 Stwo prover are out-of-tree and out of
v1.0 scope). The supported runtimes and dependency constraints (§8) are declared in
`[project].requires-python` and `[project.dependencies]` with the exact constraints from the [conventions document](conventions.md)
dependency table; optional groups (`[project.optional-dependencies]`) carry the swappable backends
(DP accountant, tensorboard/wandb observability adapters, the Phase-2 verify extras as
NotImplementedError stubs).

### 5.2 Release checklist (release-blocking gates)

A tagged release is cut only after all gates pass. These reuse the CI gates defined in
[07 — Testing Strategy](07-testing-strategy.md) and are listed here in release order:

1. `ruff` lint clean and `pyright` type-check clean.
2. Unit + property (hypothesis) suite green on the CPU fallback path ([conventions §9](conventions.md#9-determinism-dtype-device) mandates tests pass on CPU).
3. The aggregation-determinism self-check green: the outer step is bitwise-reproducible given its inputs
   (`INV-AGG-DETERMINISM`, enforced in `aggregation`/`federation.outer_optimizer`; failure raises
   `NonDeterministicAggregation`). This gate is also a Phase-2 proof-readiness prerequisite (RFC-0006 §3).
4. The reproducibility gate: two runs of the canonical small config under the same root seed produce
   identical `RunManifest` config and seed hashes (same-seed ⇒ same manifest hash; see
   [RFC-0009](../rfcs/RFC-0009-configuration-reproducibility.md)).
5. The public-recomputation gate: `recompute_alignment` reproduces the coordinator's frame alignment from
   the public probe + committed weights (Phase-1 proof-readiness, RFC-0006 §3; the probe is pinned per
   `INV-PROBE-PIN`).
6. Coverage threshold met; docs link-check passes (every relative cross-reference resolves, every
   in-document `#anchor` resolves).
7. `CHANGELOG.md` has a non-empty release block; `pyproject.toml` `[project].version`,
   `lensemble.__version__`, and the changelog version agree.
8. A `RunManifest` from the canonical config validates against its `schema_version` and round-trips.

A failure of any security-critical gate — the residency guard test (`INV-RESIDENCY`), the
commitment-binding test (`INV-COMMIT-BINDING`), or the aggregation-determinism check — is fail-closed:
the release is blocked, never waived. These map to the never-swallowed errors in
[04 — Error Model](04-error-model.md) (`ResidencyViolation`, `CommitmentMismatch`,
`NonDeterministicAggregation`).

### 5.3 Tag, build, publish

1. Branch a `release/X.Y.Z` branch; finalize the changelog block and the version in `pyproject.toml`.
2. Run the §5.2 checklist; on green, create an annotated git tag `vX.Y.Z` on the release commit.
3. Build the source distribution and wheel via the PEP 517 frontend; verify the wheel installs into a
   clean environment and `lensemble.__version__` equals the tag.
4. Publish the wheel + sdist to PyPI. The published artifact set is immutable; a regression after publish
   is fixed by a new PATCH release, never by re-uploading a version.
5. Cut a GitHub release referencing the tag and the changelog block, and attach the research-artifact
   manifest (§6) so the published code is paired with the checkpoints/configs that reproduce its claims.

RISK: PyPI publication is irreversible per version and a bad release cannot be un-published, only
yanked. Resolution plan: §5.2 gates run on the exact release commit, and the post-publish smoke install
in step 3 runs before the upload, not after; a yanked version's slot is never reused. Owner @AbdelStark;
release-automation hardening lands in v1.0.

### 5.4 Research-artifact release

The scientific claims of [00 — Overview](00-overview.md) and ([RFC-0005](../rfcs/RFC-0005-evaluation.md)) are only
credible if the centralized / local / federated triple is reproducible end to end. Each release that
carries new empirical results ships a research-artifact bundle alongside the package (RFC-0005 §8):

- The Hydra config groups (`configs/`) for every reported run: the centralized-pooled upper bound, the
  local-only lower bound, the naive-FedAvg negative control, the Fork A reference, and each rung of the
  ablation ladder.
- The `safetensors` checkpoints for the released models, each with its schema-versioned, hash-committed
  header (`schema_version`, `content_hash`, `wmcp_version`, `round_index`, `parent_hash`, `config_hash`;
  see [RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md)).
- The pinned public-probe content hash and the per-round sketch seeds `s_t`, so the SIGReg sketch matrix
  `A` is identical to the released run (`INV-SKETCH-CONSISTENCY`) and the frame-drift diagnostic recomputes
  deterministically from logs + committed weights + probe (`INV-PROBE-PIN`; see
  ([RFC-0015](../rfcs/RFC-0015-observability-diagnostics.md)) and [05 — Observability](05-observability.md)).
- For each run, the emitted `RunManifest` (config hash, root + derived seeds, per-round sketch seeds, git
  SHA, environment — python/torch/CUDA/driver — pinned dependency versions, probe content hash, hardware).
- The reported settings: hardware, rounds, inner horizon `H`, communication bytes, DP `(ε, δ)`, and the
  `λ` weights (`λ_pred`, `λ_sig`, `λ_anc`).

The bundle is large-binary content and is NOT committed to the source tree (the `.gitignore` excludes
checkpoints); it is hosted as an external release artifact and pinned by content hash from the GitHub
release and the paper. CI never downloads it: tests use tiny synthetic fixtures and the CPU smoke config
only (see [07 — Testing Strategy](07-testing-strategy.md) and [08 — Performance Budget](08-performance-budget.md)).

## 6. Contributor workflow and RFC process

### 6.1 Contribution flow

- Work happens on a branch, never on `main`. A change opens a pull request that names the affected
  subsystem area label(s) (the [conventions document](conventions.md) area taxonomy: `area:core`, `area:contracts`, `area:model`,
  `area:gauge`, `area:federation`, `area:aggregation`, `area:privacy`, `area:data`, `area:provenance`,
  `area:eval`, `area:config`, `area:artifacts`, `area:observability`, `area:verify`, `area:cli`,
  `area:docs`, `area:ci`, `area:packaging`).
- A PR that touches public surface, CLI, on-disk schema, or documented behavior MUST include the
  changelog entry (§3) and, where it changes a contract, the matching RFC + spec-section edit (§4.3).
- A PR that touches a versioned invariant's enforcement names the `INV-*` id in the description and in
  the test it adds or updates (see [04 — Error Model](04-error-model.md) and
  [07 — Testing Strategy](07-testing-strategy.md)).
- All §5.2 CI gates run on every PR, not only at release; a PR cannot merge red. Security-critical gates
  (residency, commitment-binding, aggregation determinism) are non-waivable on PRs as well as releases.

### 6.2 RFC process

A normative change to a contract, an invariant, a wire message, or a milestone goes through an RFC:

1. Open a `Draft` RFC at `docs/rfcs/RFC-NNNN-<slug>.md` following the mandatory template section order
   (Summary, Motivation, Goals, Non-Goals, Proposed Design, Alternatives Considered, Drawbacks,
   Migration / Rollout, Testing Strategy, Open Questions, References) with a header carrying Status,
   Authors, Created, and Target milestone.
2. On acceptance the Status moves to `Accepted` and the implementing spec sections are updated to cite it.
3. A later RFC that replaces it marks it `Superseded by RFC-MMMM`; the superseded RFC is retained.

`OPEN QUESTION:` items in an RFC carry an owner and a target resolution (a Stage, a milestone, or a
follow-up RFC); they are tracked to closure, not left as `TBD`. The same discipline applies to this
section (see §10).

### 6.3 Required repository documents

The following repository documents are required for a 1.0 release and SHOULD exist from v0.1 onward:

| File | Purpose | License of the file's content |
|---|---|---|
| `LICENSE` (code) | Apache License 2.0 over the source tree. | Apache-2.0 |
| `LICENSE-docs` | CC-BY-4.0 over `docs/` (the spec corpus and RFCs). | CC-BY-4.0 |
| `LICENSE-data` | CDLA-Permissive-2.0 over released datasets/probe (proposed; §7). | CDLA-Permissive-2.0 |
| `SECURITY.md` | Vulnerability-disclosure contact and process; references the residency and secrets-handling guarantees of [06 — Security](06-security.md). | CC-BY-4.0 |
| `CONTRIBUTING.md` | The §6.1 flow, the §6.2 RFC process, the changelog requirement, and the CI gates. | CC-BY-4.0 |
| `CODE_OF_CONDUCT.md` | Community standards (Contributor Covenant-style). | CC-BY-4.0 |

`SECURITY.md` states the no-secrets-in-artifacts-or-logs rule and that there is no `pickle` in the
supply chain (safetensors only), consistent with [06 — Security](06-security.md). A security fix is
released as promptly as the §5.2 gates allow and recorded under the changelog `Security` category.

## 7. License discipline

Lensemble uses a three-way licensing split matching ecosystem norms (proposed in the README; carried
here as the normative policy until ratified):

| Asset class | License | Covers |
|---|---|---|
| Code | Apache-2.0 | The `lensemble/` package, tests, build tooling, CLI. |
| Documentation | CC-BY-4.0 | `docs/spec/` and `docs/rfcs/` (this corpus), `README.md`. |
| Data | CDLA-Permissive-2.0 | Released datasets, the public probe `P`, and landmark targets. |

Rules:

- Each asset class carries its own license file (§6.3); a release MUST NOT ship an asset whose license
  is unstated.
- Released data and the public probe carry CDLA-Permissive-2.0; the probe is public/licensed by
  construction (it is what makes alignment publicly recomputable — RFC-0006 §4, RFC-0004), so its
  license is compatible with redistribution. Raw participant trajectories are NEVER released or licensed
  here: they never cross a trust boundary (`INV-RESIDENCY`, enforced in `data.residency`; a violation is
  `ResidencyViolation`, fail-closed — see [06 — Security](06-security.md)).
- Third-party warm-start weights (the pinned V-JEPA 2 release) and pinned dependencies retain their
  upstream licenses; Lensemble redistributes derived checkpoints only under terms compatible with those
  upstreams. A checkpoint warm-started from a third-party release records the warm-start provenance in
  its artifact header.

OPEN QUESTION: the data license (CDLA-Permissive-2.0) is proposed, not ratified, and its compatibility
with the upstream licenses of any redistributed embodied datasets must be confirmed per dataset before a
data release. Owner @AbdelStark; resolution path: per-dataset license review gated on the first
research-artifact data release (Stage B / v0.2), tracked in ([RFC-0004](../rfcs/RFC-0004-data-provenance.md)).

## 8. Supported runtimes and support window

Supported runtimes and the dependency constraints they are built against (canonical, from the [conventions document](conventions.md)
dependency table; reasons abbreviated):

| Dependency | Constraint | Reason |
|---|---|---|
| Python | `>=3.11` | structural typing, performance, `tomllib` |
| torch | `>=2.4,<3` | autograd, FSDP2, differentiable SVD, deterministic algorithms |
| numpy | `>=1.26` | array / statistic ops |
| safetensors | `>=0.4` | safe, mmap-able weight serialization (no pickle) |
| lance | `>=0.10` | append-friendly indexed episode reads (default format) |
| h5py | `>=3.10` | portable single-file episode format |
| hydra-core / omegaconf | `>=1.3` / `>=2.3` | structured config groups + overrides |
| pydantic | `>=2,<3` | typed validation of on-disk metadata schemas |
| typer / rich | recent | CLI + console rendering |
| opacus (or vendored RDP/PRV accountant) | pinned | DP accounting reference, behind `privacy.accountant` |
| blake3 / hashlib | recent / stdlib | hashing; canonical commitment hash = SHA-256 (Phase 1) |
| pytest / hypothesis / pytest-benchmark | recent | unit / property / perf tests |
| ruff / pyright | recent | lint + type-check CI gates |
| V-JEPA 2 weights | pinned release | encoder warm-start + AC recipe; the `t=0` frame anchor |
| LeJEPA / LeWM (SIGReg) | pinned | the objective (random-projection + characteristic-function Gaussianity) |
| stable-worldmodel | pinned (vendored, [RFC-0016](../rfcs/RFC-0016-deployment-vendoring-topology.md)) | data layer (`lance`/`hdf5`/`lerobot`), envs, latent-MPC eval |
| stable-pretraining | pinned (vendored, [RFC-0016](../rfcs/RFC-0016-deployment-vendoring-topology.md)) | pretraining scaffold reused alongside stable-worldmodel |
| Stwo (Circle-STARK prover) | Phase-2 only | aggregation-correctness proof — out of v1.0 scope |

Support-window policy:

- Python and torch: each MINOR release supports the constraint range above. A bump to the minimum
  supported Python or to the torch upper/lower bound is a public-compatibility-relevant change recorded
  in the changelog `Changed` category; it is permitted on a package MINOR pre-1.0 and is treated as a
  MAJOR-relevant change post-1.0 if it drops a previously-supported runtime.
- A release tests against the CPU fallback path on the minimum supported Python, so the support claim is
  exercised by CI rather than asserted ([conventions §9](conventions.md#9-determinism-dtype-device): tests must pass on CPU; CUDA is the primary device but
  not required to run the small CI configs).
- Pinned-by-content dependencies (the V-JEPA 2 warm-start, LeJEPA/LeWM, stable-worldmodel) are part of
  the reproducibility contract: their pinned versions are recorded in each `RunManifest`, and changing a
  pin is a `Changed` changelog entry because it can change reproduced numbers.

The canonical Phase-1 commitment hash is SHA-256 (conservative, interoperable; used for episode leaf
hashing, the Merkle root `R_c`, and the checkpoint `content_hash` — see
[RFC-0014](../rfcs/RFC-0014-provenance-commitments.md) and [RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md)).

OPEN QUESTION: migrate the Phase-1 commitment hash from SHA-256 to a STARK-friendly hash (e.g.
Poseidon2) to keep the Phase-2 proof circuit cheap. This is a versioned hash-function choice — a hash
change is a `schema_version` bump on every artifact whose `content_hash` it computes, with a migration
path. Owner @AbdelStark; resolution path: Stage D, governed by ([RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md))
and tracked in ([RFC-0014](../rfcs/RFC-0014-provenance-commitments.md)).

## 9. References

- Public-API surface, CLI, and stability promise: [02 — Public API](02-public-api.md).
- Core types and schema catalogue: [03 — Data Model](03-data-model.md).
- Error taxonomy and fail-closed rules: [04 — Error Model](04-error-model.md).
- Observability, the frame-drift diagnostic, and redaction: [05 — Observability](05-observability.md),
  ([RFC-0015](../rfcs/RFC-0015-observability-diagnostics.md)).
- Threat model, residency, secrets, and supply chain: [06 — Security](06-security.md).
- CI gates and the test pyramid: [07 — Testing Strategy](07-testing-strategy.md).
- Performance budgets and the CI perf smoke: [08 — Performance Budget](08-performance-budget.md).
- Staged plan A–E and milestone mapping: ([RFC-0001](../rfcs/RFC-0001-architecture.md)).
- Reproducibility, `RunManifest`, seeding, and config schema versioning:
  ([RFC-0009](../rfcs/RFC-0009-configuration-reproducibility.md)), ([RFC-0005](../rfcs/RFC-0005-evaluation.md)).
- Checkpoint/artifact format and hashing: ([RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md)).
- WMCP contract and `wmcp_version` gating: ([RFC-0007](../rfcs/RFC-0007-wmcp-latent-contract.md)).
- Provenance commitments and the SHA-256 → Poseidon2 question:
  ([RFC-0014](../rfcs/RFC-0014-provenance-commitments.md)).
- Proof-readiness requirements that the release gates enforce now:
  ([RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md)).
- License stanza (proposed code Apache-2.0 / docs CC-BY-4.0 / data CDLA-Permissive-2.0): `README.md`.

## 10. Open Questions

OPEN QUESTION: ratify the data license (CDLA-Permissive-2.0) and confirm per-dataset compatibility before
the first data release. Owner @AbdelStark; resolution path: Stage B / v0.2 research-artifact data release,
tracked in ([RFC-0004](../rfcs/RFC-0004-data-provenance.md)). (Cross-referenced from §7.)

OPEN QUESTION: STARK-friendly commitment-hash migration (SHA-256 → Poseidon2) and the artifact
`schema_version` bump + migration it forces. Owner @AbdelStark; resolution path: Stage D, governed by
([RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md)). (Cross-referenced from §8.)

OPEN QUESTION: the exact 1.0 freeze contents if any pre-1.0 deprecation window is still open at the 1.0
boundary — whether a still-deprecated symbol is removed at 1.0 or carried two further minors. Owner
@AbdelStark; resolution path: v1.0 hardening milestone, decided when the v0.3 deprecation set is final.
