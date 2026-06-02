# 07 — Testing Strategy

This document specifies how Lensemble is tested: the layered test pyramid, the machine-learning-specific
tests that constitute the differentiating verification layer, the ablation ladder realized as
CPU-runnable integration tests, the numerical-tolerance policy, and the continuous-integration gates. It
is the stable contract for what every change must satisfy before it merges. RFCs hold the rationale for
each subsystem; this section names the tests and the properties they pin, and binds each ML test to a
named invariant ([conventions §7](conventions.md#7-named-invariants)) and the error it forces ([conventions §6](conventions.md#6-error-taxonomy), [04 — Error Model](04-error-model.md)).

Two non-negotiable principles govern the entire suite:

1. **Every named invariant `INV-*` has at least one test that fails when the invariant is violated.** The
   invariant id appears in the test docstring. Security-critical invariants (`INV-RESIDENCY`,
   `INV-COMMIT-BINDING`, `INV-AGG-DETERMINISM`) have property tests, not just example tests.
2. **The full unit + property + integration suite runs on CPU with tiny synthetic fixtures.** No CI job
   downloads a model checkpoint, a dataset, or a probe. CUDA-only paths are exercised in a separate,
   non-blocking nightly job (see [§8](#8-ci-gates)).

---

## 1. The test pyramid

The suite is organized into six layers under `tests/`. Proportions are guidance for where effort and
test count concentrate, not hard quotas. The mapping to the package layout ([conventions §1](conventions.md#1-repository-and-package-layout)) is given so a
contributor knows where a new test belongs.

```
                   ┌───────────────────────┐
   end-to-end  ~3% │  full round lifecycle  │  tests/e2e/
                   ├───────────────────────┤
   ml-specific ~17%│ gauge / sigreg / dp /  │  tests/ml/
                   │ merkle / determinism   │
                   ├───────────────────────┤
   integration ~20%│ round SM, eval harness,│  tests/integration/
                   │ ablation ladder rungs  │
                   ├───────────────────────┤
   property    ~20%│ hypothesis invariants  │  tests/property/
                   ├───────────────────────┤
   unit        ~40%│ pure functions, schemas│  tests/unit/
                   └───────────────────────┘
   regression  (cross-cutting)  golden values   tests/regression/
```

| Layer | Owns | Directory | Runtime budget (full layer, CPU) |
|---|---|---|---|
| Unit | Pure functions, dataclass/pydantic validation, error construction, hashing, canonicalization | `tests/unit/` | < 60 s |
| Property | Invariant-preserving randomized inputs via `hypothesis` ([conventions §11](conventions.md#11-external-dependencies)) | `tests/property/` | < 120 s |
| Integration | Multi-module flows wired without network: round state machine, eval harness on a toy env, ablation-ladder rungs | `tests/integration/` | < 300 s |
| ML-specific | The mathematical contracts: gauge invariance, anchor pinning, Procrustes, SIGReg, DP, determinism, Merkle, reproducibility | `tests/ml/` | < 300 s |
| Regression | Golden-value snapshots (committed hashes, manifest hashes, drift curves on a fixed fixture) that must not change silently | `tests/regression/` | included above |
| End-to-end | One full simulated federation round through `Coordinator`/`Participant` on a synthetic two-participant fixture | `tests/e2e/` | < 120 s |

Layer boundaries: unit tests touch the filesystem only under `tmp_path` and build no `torch.nn.Module`
larger than a 2-layer toy; integration tests spawn no network socket (the coordinator runtime runs
in-process, [RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md)); ML tests depend on no pretrained
warm-start — a 2-layer linear "encoder" with output dim `d=8` exercises the gauge mathematics.

---

## 2. ML-specific tests (the differentiating layer)

These tests verify the mathematical contracts that make Lensemble correct, not merely runnable. Each is
listed with its file, the property it checks, the invariant it pins ([conventions §7](conventions.md#7-named-invariants)), and the error it forces on
violation ([conventions §6](conventions.md#6-error-taxonomy)). Tolerances are defined in [§6](#6-numerical-tolerance-policy). All run on CPU with
`d` small (typically `d=8` or `d=16`).

### 2.1 Gauge invariance of the objective

`tests/ml/test_gauge_invariance.py::test_objective_invariant_under_random_rotation`

Property: the full objective $\mathcal{L} = \lambda_{\text{pred}}\lVert g_\phi(f_\theta(x_t),a_t) -
\text{sg}[f_\theta(x_{t+1})]\rVert^2 + \lambda_{\text{sig}}\mathrm{SIGReg}_A(f_\theta(x)) +
\lambda_{\text{anc}}\mathcal{L}_{\text{anchor}}$ is unchanged under the gauge transform $f_\theta \mapsto
Qf_\theta,\ g_\phi \mapsto Qg_\phi Q^\top$ for a random $Q\in O(d)$ ([RFC-0002 §4](../rfcs/RFC-0002-gauge-and-aggregation.md)).
The anchor term ($\lambda_{\text{anc}}>0$) is the only term that breaks the symmetry by design, so this
test runs in two modes: (a) prediction + SIGReg only ($\lambda_{\text{anc}}=0$) must be invariant to
within `rtol_loss`; (b) with the anchor active the loss must NOT be invariant (a regression-guard against
an anchor that silently does nothing).

```python
def test_objective_invariant_under_random_rotation(rng) -> None:
    """INV: gauge symmetry. Objective with lambda_anc=0 is invariant under Q in O(d)."""
    Q = random_orthogonal(d=8, rng=rng)          # Haar-distributed via QR of a Gaussian
    base = objective(f, g, batch, lambda_anc=0.0)
    rot  = objective(rotate_encoder(f, Q), conjugate_predictor(g, Q), batch, lambda_anc=0.0)
    assert math.isclose(base.total, rot.total, rel_tol=RTOL_LOSS, abs_tol=ATOL_LOSS)
```

Property variant: `tests/property/test_gauge_property.py` draws many random $Q$ via `hypothesis` and
asserts invariance over the distribution (catches non-generic $Q$ that an example test would miss).

### 2.2 The anchor pins the frame

`tests/ml/test_anchor.py::test_landmark_anchor_recovers_identity`

Property (Variant A, [RFC-0002 §4](../rfcs/RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)): with $k \ge d$ generic
landmark points and fixed targets $t_i = f_{\text{ref}}(p_i)$, the only orthogonal map satisfying all
landmark constraints is $Q\approx I$. Construct $f_{\text{ref}}$, apply a known rotation $Q_0$ to produce a
drifted encoder, minimize $\mathcal{L}_{\text{anchor}}$ over an orthogonal correction, and assert the
recovered map drives the probe Procrustes rotation back to identity (rotation angle `< ANGLE_TOL_DEG`).
A companion negative case uses $k < d$ landmarks and asserts the frame is NOT pinned (the constraint is
under-determined), guarding the documented $k \ge d$ requirement ([RFC-0004 §3](../rfcs/RFC-0004-data-provenance.md#3-the-public-probe-set-mathcalp)).
Enforces the precondition behind `INV-PROBE-PIN`: landmark targets derive only from $f_{\text{ref}}$.

### 2.3 Procrustes closed-form correctness

`tests/ml/test_procrustes.py::test_procrustes_matches_brute_force`

Property: `procrustes_align(source, target) -> (Q*, residual)` ([conventions §5](conventions.md#5-public-api-surface))
returns the closed-form solution $Q^\star = VU^\top$ from the SVD $E_{\text{ref}}^\top f_\theta(\mathcal{P})
= U\Sigma V^\top$. Verify against a small-angle brute-force search in $d=3$: the closed-form residual must
be `<=` every sampled candidate's residual within `RTOL_PROC`, and $Q^\star$ must be orthogonal
($\lVert Q^{\star\top}Q^\star - I\rVert_F < ATOL_ORTHO$) with $\det Q^\star = +1$ for the proper-rotation
case. Degenerate-input case: near-equal singular values trigger the conditioning path and must raise
`DegenerateProcrustes` ([conventions §6](conventions.md#6-error-taxonomy)) rather than return a `NaN`/non-orthogonal matrix —
see [04 — Error Model §5.3](04-error-model.md#53-gauge-lensemblegauge).

### 2.4 SIGReg statistic correctness

`tests/ml/test_sigreg.py::test_sigreg_statistic_against_known_samples`

Property ([RFC-0002 §3](../rfcs/RFC-0002-gauge-and-aggregation.md#3-layer-1--shared-sketch-matrix-objective-consistency-not-the-gauge-fix), [RFC-0008 §6](../rfcs/RFC-0008-model-objective-numerics.md#6-sigreg-algorithm)):
the Epps–Pulley characteristic-function statistic (Cramér–Wold), projected onto the shared sketch matrix
$A$ (sketch dim 64, ~17 integration knots — the LeJEPA defaults), is near zero for a large standard-normal
sample and clearly positive for a known non-normal sample (e.g. a uniform or bimodal mixture). Two
asserts: `stat(N(0,1) sample) < SIGREG_NULL_TOL` and `stat(non_normal) > SIGREG_SIGNAL_FLOOR`. The shared
$A$ derives from the round sketch seed $s_t$; a second assert checks two participants given the same $s_t$
compute the identical $A$ — this is `INV-SKETCH-CONSISTENCY`, enforced in `model.sigreg` /
`federation.round`, with mismatch surfacing as a `ConfigError` at round open.

### 2.5 Aggregation determinism (bitwise)

`tests/ml/test_aggregation_determinism.py::test_outer_step_is_bitwise_reproducible`

Property: `INV-AGG-DETERMINISM`. Given fixed (committed deltas, round seed, prior global params), the
outer Nesterov step ([RFC-0003 §2](../rfcs/RFC-0003-federated-protocol.md#2-round-structure-diloco-outer-loop)) is a pure, bitwise-identical
function. Run the aggregation path twice in the same process and once in a fresh subprocess; assert
byte-equality of the resulting flat tensor (`torch.equal` AND identical `safetensors` content hash). A
second assert shuffles the input delta order and confirms the fixed reduction order yields the SAME result
(no order-dependence). The self-check that runs each real outer step ([conventions §9](conventions.md#9-determinism-dtype-device)) is exercised by injecting a
deliberately non-deterministic reduction and asserting it raises `NonDeterministicAggregation` ([conventions §6](conventions.md#6-error-taxonomy)),
which aborts the round and triggers recompute — never silently averages.
This determinism is also the Phase-1 proof-ready requirement of
[RFC-0006 §3](../rfcs/RFC-0006-verifiable-contribution.md#3-phase-1-proof-ready-requirements-cheap-to-honor-now).

### 2.6 DP clip bound

`tests/ml/test_dp.py::test_clip_enforces_bound`

Property: `INV-DP-BOUND`. After clipping ($\Delta_c \leftarrow \Delta_c\cdot\min(1, C_{\text{clip}}/
\lVert\Delta_c\rVert)$) and before noising, $\lVert\Delta_c\rVert \le C_{\text{clip}}(1 + \text{rtol})$
for every input, including the boundary case $\lVert\Delta_c\rVert = C_{\text{clip}}$ and a zero vector
(no division by zero). `hypothesis` draws delta vectors across magnitudes. A companion test asserts the
Gaussian noise is calibrated to $\sigma C_{\text{clip}}$ (empirical std over many draws within `RTOL_DP`)
and that clipping itself is deterministic given the input. Budget exhaustion is tested in
`tests/integration/test_dp_accountant.py`: the accountant raises `PrivacyBudgetExceeded` ([conventions §6](conventions.md#6-error-taxonomy)) when the
planned $(\varepsilon,\delta)$ is spent, stopping training rather than continuing —
[RFC-0012 §Testing Strategy](../rfcs/RFC-0012-differential-privacy.md#testing-strategy).

### 2.7 Residency guard refuses to emit raw data

`tests/ml/test_residency.py::test_residency_guard_fails_closed`

Property: `INV-RESIDENCY`, security-critical. The guard in `data.residency` MUST refuse to serialize a raw
observation, action, or private-data embedding into any outbound message or artifact. The test attempts to
place each forbidden tensor type into an `Update` payload and an outbound artifact header and asserts each
attempt raises `ResidencyViolation` ([conventions §6](conventions.md#6-error-taxonomy)). `ResidencyViolation` is fail-closed and is on the
never-swallow list — a paired test asserts no `except` in the egress path catches it. A permitted-payload
case confirms hashes, L2 norms, shapes, and scalar counts pass the guard (so the guard is not vacuously
rejecting everything). Cross-referenced by [05 — Observability §5](05-observability.md#5-redaction-inv-residency) and
[06 — Security §3](06-security.md#3-residency-enforcement-inv-residency).

### 2.8 Merkle root and inclusion proofs

`tests/ml/test_merkle.py::test_merkle_root_and_inclusion`

Property: the dataset Merkle root $R_c$ over domain-separated episode-leaf hashes is correct and stable,
and every committed episode has a verifying inclusion proof ([RFC-0014](../rfcs/RFC-0014-provenance-commitments.md)).
Asserts: (a) the root is invariant to recomputation (deterministic); (b) a valid inclusion proof verifies;
(c) a tampered leaf or a proof for a non-member episode fails verification with `MerkleVerificationError`
([conventions §6](conventions.md#6-error-taxonomy)); (d) leaf and node domain separation differ so a leaf hash can never be confused with an internal
node. Binding is tested separately: a `PseudoGradient` bound to the wrong `dataset_root` is rejected with
`CommitmentMismatch` (`INV-COMMIT-BINDING`, never swallowed), in
`tests/integration/test_commit_binding.py`.

### 2.9 Checkpoint hash round-trip and tamper detection

`tests/ml/test_checkpoint.py::test_checkpoint_hash_roundtrip_and_tamper`

Property: `INV-CHECKPOINT-HASH`. A `Checkpoint` written and re-read yields identical tensors (`safetensors`,
[conventions §8](conventions.md#8-core-data-types)/§11) and an identical SHA-256 `content_hash` over the canonical byte serialization, across two
processes (cross-platform stability of the canonicalization, [RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md)).
A flipped byte in the weight file or a mutated header field causes the recomputed hash to differ from the
committed hash and raises `CheckpointIntegrityError` ([conventions §6](conventions.md#6-error-taxonomy)). A separate assert confirms the loader
refuses any non-`safetensors` (e.g. pickle) payload — no arbitrary-code-execution path
([06 — Security §7](06-security.md#7-secrets-handling-and-supply-chain)).

### 2.10 Schema round-trip and migration

`tests/ml/test_schema.py::test_schema_roundtrip_and_migration`

Property: every on-disk pydantic v2 model (`RunManifest`, `DatasetCommitment`, `ModelArtifact` header,
`EvalReport`, `FrameDriftReport`, `ContributionRecord`; [conventions §8](conventions.md#8-core-data-types)) round-trips JSON without loss and carries an
integer `schema_version`. A reader given an older `schema_version` migrates it via the registered migration
function; a reader given an unknown/too-new version raises `SchemaVersionMismatch` ([conventions §6](conventions.md#6-error-taxonomy), [conventions §10](conventions.md#10-versioning-and-schema-policy)). The
test parametrizes over each schema and over (current, current-1, current+1) versions.

### 2.11 Reproducibility — same seed, same manifest hash

`tests/ml/test_reproducibility.py::test_same_seed_same_manifest_hash`

Property: same `LensembleConfig` + same root seed ⇒ identical `RunManifest` config-hash and seed-derivation,
and per-round sketch seeds $s_t = \mathrm{derive}(\text{root\_seed}, t)$ are deterministic ([conventions §9](conventions.md#9-determinism-dtype-device),
[RFC-0009](../rfcs/RFC-0009-configuration-reproducibility.md)). Two runs of a tiny single-site
`train_local` config produce identical config/seed hashes in their manifests. With the determinism flag
(`torch.use_deterministic_algorithms`) set, the aggregation-path outputs are bitwise identical across the
two runs (reusing the §2.5 assertion). Anchors `INV-WARMSTART-T0` at the unit level: a round-0 encoder load
test asserts every participant's encoder weights are hash-identical to the pinned warm-start before any
local step.

### 2.12 WMCP conformance

`tests/ml/test_wmcp.py::test_latent_state_and_actionspec_conformance`

Property: `INV-WMCP`. Every `LatentState` conforms to the pinned `wmcp_version` — shape `(N, d)`, dtype,
and semantics; a malformed latent (wrong rank, wrong `d`, wrong dtype) raises `ContractViolation` ([conventions §6](conventions.md#6-error-taxonomy),
[RFC-0007](../rfcs/RFC-0007-wmcp-latent-contract.md)). Every `ActionSpec` is validated before an action head
is constructed; an invalid spec raises `ContractViolation`. A two-embodiment fixture (a quadruped and a
7-DoF arm) confirms both conform to one shared latent contract while their action heads differ. Paired
assert for `INV-ACTIONHEAD-LOCAL`: an attempt to include an action head $h_\psi^{(c)}$ in an outbound
aggregation payload is rejected (it is never broadcast or aggregated).

---

## 3. The ablation ladder as integration tests

The ablation ladder ([RFC-0005 §6](../rfcs/RFC-0005-evaluation.md#6-ablation-ladder-the-core-experiment)) is the core scientific experiment. Each
rung is also a CPU-runnable integration test on a tiny synthetic federation (two to four toy participants,
`d` small, a synthetic probe), wired through `Coordinator`/`Participant` in-process. The tests assert the
qualitative ordering the science predicts, not paper-grade numbers — they are smoke-and-monotonicity guards
that the mechanisms compose, runnable in CI.

`tests/integration/test_ablation_ladder.py`:

| Rung | Mechanism added | Test assertion (synthetic, CPU) |
|---|---|---|
| 1 | Naive end-to-end FedAvg (no gauge control) | On synthetically rotated silos, the frame-drift diagnostic INCREASES over rounds (the negative control diverges). |
| 2 | + shared sketch matrix $A$ ([RFC-0002 §3](../rfcs/RFC-0002-gauge-and-aggregation.md#3-layer-1--shared-sketch-matrix-objective-consistency-not-the-gauge-fix)) | All participants compute the identical $A$ from $s_t$ (`INV-SKETCH-CONSISTENCY`); drift behavior unchanged (Layer 1 does NOT close the gauge — an explicit assert that drift is still high). |
| 3 | + Procrustes align-then-average ([RFC-0002 §5](../rfcs/RFC-0002-gauge-and-aggregation.md#5-layer-3--procrustes-re-alignment-at-aggregation-backstop)) | Post-alignment inter-participant residual drops at aggregation; the backstop fires only when drift exceeds threshold. |
| 4 | + frame-anchor loss, Variant A ([RFC-0002 §4](../rfcs/RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)) | Frame drift stays flat/low across rounds where rung 1 diverged. This is the expected recommended configuration. |
| 5 | + function-space distillation ([RFC-0002 §6](../rfcs/RFC-0002-gauge-and-aggregation.md#6-layer-4--function-space-distillation-fallback--heterogeneity)) | Aggregating behaviors on the probe produces a gauge-invariant consensus; drift is low by construction. |

`tests/integration/test_lambda_anc_sweep.py` realizes the $\lambda_{\text{anc}}$ sweep
([RFC-0002 §7](../rfcs/RFC-0002-gauge-and-aggregation.md#7-the-central-hyperparameter)) on the toy federation: it asserts the
monotone trade-off direction — very large $\lambda_{\text{anc}}$ pins the frame hard (drift near zero) at
the cost of representational freedom, very small $\lambda_{\text{anc}}$ lets drift grow — so the
drift-vs-quality curve has the predicted shape. The full quantitative sweep is a Stage-B experiment, not a
CI test; see [RFC-0005 §6](../rfcs/RFC-0005-evaluation.md#6-ablation-ladder-the-core-experiment).

`tests/integration/test_eval_harness.py`: the latent-MPC eval harness ([RFC-0005 §3](../rfcs/RFC-0005-evaluation.md#3-downstream-metric--planning-success))
runs end-to-end on a toy `stable-worldmodel`-style synthetic env with a CEM planner over a 2-step horizon;
asserts `evaluate(...)` returns a well-formed `EvalReport` with a finite `success_rate`, deterministic under
a fixed seed and pinned probe hash. The individual metric implementations (frame-drift, effective
dimension, success rate, communication bytes) each get a unit test in `tests/unit/test_metrics.py`
([05 — Observability §2](05-observability.md#2-metric-taxonomy)).

`tests/integration/test_non_iid_partition.py`: confirms the factor-of-variation partition produces
controlled, reproducible heterogeneity across silos (same seed ⇒ same partition) so the non-IID sweep
([RFC-0005 §7](../rfcs/RFC-0005-evaluation.md#7-non-iid-severity--scale-sweeps)) is deterministic.

---

## 4. End-to-end and round-lifecycle tests

`tests/e2e/test_full_round.py` wires one complete simulated federation round through the public surface
([conventions §5](conventions.md#5-public-api-surface)): `Coordinator.run(num_rounds=1)` over two in-process `Participant`s on a synthetic fixture, with
DP, simulated secure aggregation, the frame anchor, and a hash-committed result. It asserts the round state
machine traverses `OPEN → COLLECTING → AGGREGATING → ALIGNING → COMMITTING → CLOSED`
([RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md)), that `INV-RESIDENCY` holds across the round (no raw
tensor in any message — reuses the §2.7 guard), and that the committed global hash matches the
`RoundClose` hash (`INV-CHECKPOINT-HASH`).

`tests/integration/test_secure_agg.py` ([RFC-0011 §Testing Strategy](../rfcs/RFC-0011-secure-aggregation.md#testing-strategy)):
mask cancellation correctness (the revealed sum equals the plaintext sum within exact integer/`RTOL_AGG`
bound), dropout recovery above the threshold succeeds and below it raises `SecureAggregationError` ([conventions §6](conventions.md#6-error-taxonomy)),
and a no-individual-leak property check. Masking must not perturb the determinism of the revealed sum
(`INV-AGG-DETERMINISM`), asserted by composing with §2.5.

`tests/integration/test_recompute_alignment.py` is the Phase-1 proof-readiness test
([RFC-0006 §4](../rfcs/RFC-0006-verifiable-contribution.md#4-public-recomputation-phase-1-free-in-scope-now)): `recompute_alignment` ([conventions §5](conventions.md#5-public-api-surface)) reproduces the
coordinator's frame alignment from the public probe + committed weights alone, bit-for-bit, with no access
to private data. This makes the alignment publicly recomputable and needs no proof
([RFC-0002 §5](../rfcs/RFC-0002-gauge-and-aggregation.md#5-layer-3--procrustes-re-alignment-at-aggregation-backstop)). A `ProbeError` is raised if the probe content
hash differs from the hash committed in `RoundOpen` (`INV-PROBE-PIN`).

---

## 5. Module-boundary and import tests

`tests/unit/test_module_boundaries.py` asserts the dependency layering of
[01 — Architecture](01-architecture.md) holds and that there are no import cycles. It walks the import
graph of the `lensemble` package and fails if a forbidden edge appears, e.g. `contracts` importing
`federation`, or `model`/`gauge` importing anything outside `{contracts, config, errors, observability}`.
This keeps the documented layering ([RFC-0001 §Testing Strategy](../rfcs/RFC-0001-architecture.md#testing-strategy)) machine-
enforced rather than aspirational. A companion assert confirms the public re-exports in
`lensemble/__init__.py` match the surface in [02 — Public API](02-public-api.md) (no accidental
public-symbol drift) and that nothing prefixed `_` or under `_internal` is re-exported.

---

## 6. Numerical-tolerance policy

ML tests compare floating-point quantities; the policy below states where exact equality is required and
where approximate equality applies. Tolerances are named constants in `tests/conftest.py` so they are set
in one place and cited by id, not scattered as magic numbers.

| Quantity | Comparison | Tolerance (constant) | Rationale |
|---|---|---|---|
| Aggregation / outer-step output | **Exact** (`torch.equal`, identical content hash) | none — bitwise | `INV-AGG-DETERMINISM`; the proof-ready path ([conventions §9](conventions.md#9-determinism-dtype-device)) must be reproducible bit-for-bit |
| Checkpoint / Merkle / manifest hashes | **Exact** (string equality) | none | `INV-CHECKPOINT-HASH`, `INV-COMMIT-BINDING`; hashes are discrete |
| Schema round-trip | **Exact** (structural equality) | none | JSON is lossless for the typed fields |
| SIGReg null statistic on `N(0,1)` | Approximate, upper-bounded | `SIGREG_NULL_TOL = 5e-2` | finite-sample statistic; loose by design |
| SIGReg signal floor on non-normal | Approximate, lower-bounded | `SIGREG_SIGNAL_FLOOR = 1e-1` | must be clearly separated from the null |
| Objective gauge-invariance loss | Relative | `RTOL_LOSS = 1e-5`, `ATOL_LOSS = 1e-6` (fp32) | fp32 accumulation ([conventions §9](conventions.md#9-determinism-dtype-device)); rotation is exact in real arithmetic, error is roundoff |
| Procrustes residual / orthogonality | Relative / Frobenius | `RTOL_PROC = 1e-5`, `ATOL_ORTHO = 1e-5` | SVD roundoff |
| Recovered rotation angle (anchor) | Absolute, degrees | `ANGLE_TOL_DEG = 1.0` | optimization residual on a toy problem |
| DP clip bound | Relative, upper-bounded | `RTOL_DP = 1e-6` | `INV-DP-BOUND`; clip is a scalar multiply |
| DP empirical noise std | Relative | `RTOL_DP_STD = 1e-1` | sampling variance over a finite draw |
| bf16 forward vs fp32 reference | Relative | `RTOL_BF16 = 1e-2`, `ATOL_BF16 = 1e-2` | bf16 has ~3 decimal digits; compute dtype default ([conventions §9](conventions.md#9-determinism-dtype-device)) |
| Secure-agg revealed sum | Relative (or exact for integer masks) | `RTOL_AGG = 1e-6` | float masking roundoff; integer masking is exact |

Rule: any test that compares the aggregation path, a hash, or a schema uses exact equality. Approximate
equality is permitted only for genuinely continuous quantities, and the tolerance must be a named constant
with the rationale above. New tolerances require an entry in this table.

`RISK:` the cross-platform stability of `content_hash` ([conventions §8](conventions.md#8-core-data-types), [RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md))
depends on a fixed canonical byte ordering of the `safetensors` payload. If the canonicalization is
under-specified, §2.9 may pass on the development platform and fail on another. Resolution plan: pin the
exact canonical-byte ordering in RFC-0010 and run §2.9 on both an x86-64 and an arm64 CI runner in the
nightly job before the v0.1 release gate. Owner `@AbdelStark`, resolved in Stage A.

---

## 7. Test fixtures and data discipline

- **No large downloads.** Tests never fetch the V-JEPA 2 warm-start, a real dataset, or a real probe
  ([conventions §11](conventions.md#11-external-dependencies)). A `tiny_warmstart` fixture is a 2-layer linear encoder with `d=8`; a `synthetic_probe` is a
  small deterministic tensor with $k \ge d$ landmarks; a `toy_env` is an in-memory `stable-worldmodel`-style
  environment with a closed-form transition.
- **Determinism in fixtures.** Every fixture takes the test's seeded `rng`; none calls a global RNG, so
  the property layer is reproducible on failure (`hypothesis` derandomizes on replay).
- **Episode fixtures** cover all three formats round-trip — `lance` (default), `hdf5` (portable), and a
  `lerobot://` adapter stub — in `tests/integration/test_data_formats.py`
  ([RFC-0004](../rfcs/RFC-0004-data-provenance.md)). The matrix skips a variant whose optional dependency is
  absent, but at least one format always runs.
- **Redaction fixtures** for [05 — Observability](05-observability.md): a test sink that records everything
  written; `tests/ml/test_redaction.py` asserts no raw observation/action/embedding tensor ever reaches the
  sink (`INV-RESIDENCY`), only hashes, norms, shapes, counts, and scalar metrics. The redaction guard fails
  closed.

---

## 8. CI gates

Every push and pull request runs the blocking gate below on CPU. The pipeline configuration and the
relationship to release tagging are owned by [09 — Release & Versioning](09-release-and-versioning.md);
this section states what must be green to merge.

Blocking gates (ordered; fail fast):

1. **Lint** — `ruff check` and `ruff format --check` ([conventions §11](conventions.md#11-external-dependencies)). Zero findings.
2. **Type-check** — `pyright` ([conventions §11](conventions.md#11-external-dependencies)) on `lensemble/` and `tests/`. Zero errors.
3. **Unit + property** — `pytest tests/unit tests/property` on CPU. All pass; `hypothesis` runs with a
   fixed profile (deadline disabled, derandomized for replay).
4. **Integration + ML + e2e** — `pytest tests/integration tests/ml tests/e2e` on CPU. All pass.
5. **Determinism check** — a dedicated job that runs §2.5 in two processes and asserts bitwise equality of
   the aggregation path. Any drift fails the build (`INV-AGG-DETERMINISM`).
6. **Coverage threshold** — line coverage `>= 85%` overall and `100%` on `data.residency`,
   `aggregation.secure_agg` (the reveal path), `provenance.commit`, and `privacy.dp` (the security-critical
   modules). Below threshold fails the build.
7. **Docs link-check** — every relative cross-reference in `docs/` (the convention of [conventions §3](conventions.md#3-corpus-structure-and-cross-reference-conventions)) resolves to an
   existing file and, where a `#anchor` is used, to an existing heading. Broken links fail the build.

Non-blocking nightly jobs:

- **CUDA suite** — the GPU-only paths (FSDP/TP inner loop, differentiable SVD on CUDA,
  `torch.use_deterministic_algorithms` on GPU). Failures open an issue; they do not block merges, because
  CI runners are CPU-only and tests MUST pass on CPU ([conventions §9](conventions.md#9-determinism-dtype-device)).
- **Cross-platform hash check** — §2.9 on x86-64 and arm64 runners (see the `RISK:` in [§6](#6-numerical-tolerance-policy)).
- **Performance smoke** — the tiny-config wall-time ceiling that guards against perf regressions; the budget
  and its measurement live in [08 — Performance Budget](08-performance-budget.md).

What CI explicitly does NOT do: download any model/dataset/probe; require a GPU for a blocking gate; run the
full ablation sweep or non-IID/scale sweeps (those are Stage-B research runs,
[RFC-0005 §6–7](../rfcs/RFC-0005-evaluation.md), reproduced from the released configs, not CI gates).

---

## 9. Coverage of invariants (traceability matrix)

Every invariant in [conventions §7](conventions.md#7-named-invariants) maps to at least one test. This matrix is the audit substrate: if an invariant
lacks a row, the suite is incomplete.

| Invariant | Test(s) | Error on violation |
|---|---|---|
| `INV-RESIDENCY` | §2.7, §7 redaction, §4 e2e | `ResidencyViolation` (fail-closed) |
| `INV-WARMSTART-T0` | §2.11 round-0 load | (assertion; mismatch is a `ConfigError` at load) |
| `INV-SKETCH-CONSISTENCY` | §2.4, ladder rung 2 | `ConfigError` at round open |
| `INV-AGG-DETERMINISM` | §2.5, gate 5, §4 secure-agg | `NonDeterministicAggregation` |
| `INV-PROBE-PIN` | §2.2, §4 recompute | `ProbeError` |
| `INV-COMMIT-BINDING` | §2.8 binding | `CommitmentMismatch` |
| `INV-CHECKPOINT-HASH` | §2.9, §4 e2e | `CheckpointIntegrityError` |
| `INV-DP-BOUND` | §2.6 | (assertion; budget exhaustion is `PrivacyBudgetExceeded`) |
| `INV-WMCP` | §2.12 | `ContractViolation` |
| `INV-ACTIONHEAD-LOCAL` | §2.12 paired | `ResidencyViolation` / `ContractViolation` on outbound attempt |

---

## 10. Open questions

`OPEN QUESTION:` the coverage threshold for the non-security modules (currently `>= 85%`) may be too low for
the gauge and aggregation math, where a missed branch can silently degrade the science rather than crash.
Owner `@AbdelStark`; resolution: revisit per-module thresholds after the Stage-B ablation suite stabilizes
(milestone v0.2), raising `gauge` and `aggregation` toward 100% if the math paths prove brittle.

`OPEN QUESTION:` whether the ablation-ladder integration tests ([§3](#3-the-ablation-ladder-as-integration-tests))
should assert quantitative drift thresholds on the synthetic fixture, or remain qualitative
monotonicity/ordering checks. Quantitative thresholds risk flakiness from optimizer noise on a toy problem;
qualitative checks risk passing while the real mechanism is subtly wrong. Owner `@AbdelStark`; resolution:
decide after the Stage-B frame-drift diagnostic ([RFC-0005 §2](../rfcs/RFC-0005-evaluation.md#2-headline-diagnostic--latent-frame-drift)) is
characterized on real silos (milestone v0.2), then back-port a calibrated threshold into the synthetic test.

`OPEN QUESTION:` the drift threshold that fires the Layer-3 Procrustes backstop in the ladder rung-3 test
inherits the same open value as [RFC-0002 §Open Questions](../rfcs/RFC-0002-gauge-and-aggregation.md#open-questions). Until
that value is set in Stage B, the rung-3 test uses a fixture-local threshold and asserts only the
fire/no-fire behavior, not the production value. Owner `@AbdelStark`; resolution: adopt the Stage-B value
once RFC-0002 fixes it (milestone v0.2).

---

## 11. References

- Conventions document: [§6](conventions.md#6-error-taxonomy) (error taxonomy), [§7](conventions.md#7-named-invariants) (invariants), [§9](conventions.md#9-determinism-dtype-device) (determinism/dtype/device), [§11](conventions.md#11-external-dependencies)
  (dependencies: `pytest`, `hypothesis`, `pytest-benchmark`, `ruff`, `pyright`), [§12](conventions.md#12-milestones-and-stages) (milestones).
- RFCs (each RFC's Testing Strategy section is the source for the corresponding tests above):
  [RFC-0001](../rfcs/RFC-0001-architecture.md) (module-boundary, round lifecycle),
  [RFC-0002](../rfcs/RFC-0002-gauge-and-aggregation.md) (gauge invariance, anchor, Procrustes, $\lambda_{\text{anc}}$ sweep, drift diagnostic),
  [RFC-0003](../rfcs/RFC-0003-federated-protocol.md) (state machine, pseudo-gradient, int8 round-trip),
  [RFC-0004](../rfcs/RFC-0004-data-provenance.md) (residency, format round-trip, probe pin),
  [RFC-0005](../rfcs/RFC-0005-evaluation.md) (ablation ladder, baselines, metrics, reproducibility),
  [RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md) (proof-readiness: determinism, recompute, binding, pinned probe),
  [RFC-0007](../rfcs/RFC-0007-wmcp-latent-contract.md), [RFC-0008](../rfcs/RFC-0008-model-objective-numerics.md),
  [RFC-0009](../rfcs/RFC-0009-configuration-reproducibility.md), [RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md),
  [RFC-0011](../rfcs/RFC-0011-secure-aggregation.md), [RFC-0012](../rfcs/RFC-0012-differential-privacy.md),
  [RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md), [RFC-0014](../rfcs/RFC-0014-provenance-commitments.md),
  [RFC-0015](../rfcs/RFC-0015-observability-diagnostics.md).
- Sibling spec sections: [01 — Architecture](01-architecture.md), [02 — Public API](02-public-api.md),
  [03 — Data Model](03-data-model.md), [04 — Error Model](04-error-model.md),
  [05 — Observability](05-observability.md), [06 — Security](06-security.md),
  [08 — Performance Budget](08-performance-budget.md),
  [09 — Release & Versioning](09-release-and-versioning.md).
