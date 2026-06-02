# 04 — Error Model

This section specifies the canonical exception hierarchy, the error-code enumeration, the per-subsystem
failure-mode catalog, and the error-handling discipline for the `lensemble` package. It is the stable
contract that every subsystem raises into and that every caller catches against. Rationale for the
underlying mechanisms lives in the RFCs; this document specifies *what fails, how it is detected, which
typed error fires, what fields it carries, and how the system responds*.

Every error in Lensemble is a typed `LensembleError` carrying a machine-readable `.code`
(`LensembleErrorCode`) and a human-readable `.remediation`. Prose alone is never a contract: every entry
below names a class, a code, the carried fields, the triggering invariant where one applies, and the
recovery posture (retry, fail-closed, or backstop). Error logging and the redaction rules that govern
*what may appear in an error message* are specified in [05 — Observability](05-observability.md);
security-critical errors that must never be swallowed are reconciled against the threat model in
[06 — Security](06-security.md).

## 1. Design principles

1. **Typed, not stringly.** Callers branch on `LensembleError` subclasses and on `.code`, never on
   message substrings. The class hierarchy is the public contract (frozen at 1.0 per
   [02 — Public API §stability](02-public-api.md) and the versioning policy in
   [conventions §10](conventions.md#10-versioning-and-schema-policy)).
2. **Validate at boundaries.** The four ingress boundaries — config load, federation message ingress,
   artifact load, dataset ingest — validate eagerly and raise a typed error with remediation rather than
   letting a malformed value propagate into compute. A boundary failure is always actionable.
3. **Fail closed on security-critical violations.** `ResidencyViolation`, `CommitmentMismatch`, and
   `NonDeterministicAggregation` are never caught-and-ignored, never downgraded to a warning, and never
   retried blindly. They abort the operation that raised them.
4. **No bare `except`.** Handlers name the exception type(s) they intend to handle. A handler that cannot
   classify a failure re-raises. Catch-all clauses are permitted only at a process-supervisory top level
   that logs and exits non-zero — never on the aggregation, residency, or provenance path.
5. **Remediation is mandatory.** Every raised `LensembleError` populates `.remediation` with a concrete
   next action (a config key to fix, a CLI command to run, a milestone/RFC to consult). A bare raise with
   an empty remediation is a defect.

## 2. Exception hierarchy

`LensembleError` is the single base. All subsystem errors derive from it (directly or through a category
parent). Standard library exceptions (`ValueError`, `OSError`, `RuntimeError`) are wrapped into a
`LensembleError` at the boundary where they are first observed, so callers never have to catch raw stdlib
types from `lensemble` public APIs.

```
Exception
└── LensembleError                      # base; carries .code, .remediation
    ├── ConfigError                     # invalid / inconsistent configuration
    ├── ContractViolation               # WMCP nonconformance (latent shape/dtype/semantics, ActionSpec)
    ├── ResidencyViolation              # raw obs/action/private-embedding crosses a boundary  [SECURITY-CRITICAL]
    ├── GaugeError                      # latent-frame / alignment failures
    │   ├── FrameDriftExceeded          # inter-participant frame drift over threshold
    │   └── DegenerateProcrustes        # SVD ill-conditioned in Procrustes alignment
    ├── AggregationError                # outer-step / secure-sum failures
    │   ├── SecureAggregationError      # masked-sum protocol failure (dropout below threshold, etc.)
    │   └── NonDeterministicAggregation # aggregation path is not bitwise-reproducible  [SECURITY-CRITICAL]
    ├── PrivacyBudgetExceeded           # planned (eps,delta) budget spent
    ├── ProvenanceError                 # dataset-commitment / Merkle failures
    │   ├── CommitmentMismatch          # Δ_c bound to wrong / no dataset root  [SECURITY-CRITICAL]
    │   └── MerkleVerificationError     # Merkle root or inclusion proof does not verify
    ├── ArtifactError                   # checkpoint / artifact failures
    │   ├── SchemaVersionMismatch       # on-disk schema_version unknown / too new
    │   └── CheckpointIntegrityError    # content hash mismatch (tamper / corruption)
    ├── RoundError                      # round lifecycle failures
    │   └── FaultToleranceExceeded      # too few participants remain for a valid round
    ├── ProbeError                      # probe hash mismatch / under-coverage
    └── EvaluationError                 # latent-MPC / metric / harness failures
```

Base class contract (see [03 — Data Model](03-data-model.md) for the dataclass conventions; this is the
exception, not an on-disk type, so it is a plain class not a pydantic model):

```python
class LensembleError(Exception):
    """Base for all Lensemble errors. Always carries a code and a remediation."""
    code: "LensembleErrorCode"
    remediation: str

    def __init__(self, message: str, *, code: "LensembleErrorCode", remediation: str) -> None:
        super().__init__(message)
        self.code = code
        self.remediation = remediation
```

`GaugeError`, `AggregationError`, `ProvenanceError`, `ArtifactError`, and `RoundError` are category
parents: a caller may catch the parent to handle "any gauge failure" uniformly, or a leaf to branch on
the specific cause. Leaf classes set a more specific `.code` than their parent would.

## 3. Error reference table

Recoverability legend: **retry** = the operation may be re-attempted (possibly next round) without
operator intervention; **fail-closed** = abort and do not proceed (security or correctness critical);
**backstop** = a defined fallback path handles it automatically; **operator** = requires a human decision
or config change before progress.

| Class | Code | Raised when | Carried fields (beyond `code`, `remediation`) | Recovery |
|---|---|---|---|---|
| `ConfigError` | `CONFIG_INVALID` | Config fails OmegaConf/dataclass validation at load, or two keys are mutually inconsistent | `key`, `value`, `expected` | operator |
| `ContractViolation` | `WMCP_CONTRACT_VIOLATION` | A `LatentState` or `ActionSpec` does not conform to the pinned `wmcp_version` | `wmcp_version`, `expected_shape`, `got_shape`, `field` | fail-closed |
| `ResidencyViolation` | `RESIDENCY_VIOLATION` | The residency guard intercepts raw obs/action/private-embedding headed across a boundary | `tensor_role`, `boundary`, `dataset_id` (no tensor data) | fail-closed |
| `FrameDriftExceeded` | `FRAME_DRIFT_EXCEEDED` | Inter-participant Procrustes residual / rotation angle on `P` exceeds the configured threshold | `participant_pair`, `drift_angle_deg`, `threshold_deg`, `round` | backstop |
| `DegenerateProcrustes` | `PROCRUSTES_DEGENERATE` | The Procrustes SVD has near-zero / near-tied singular values past the condition tolerance | `min_singular_value`, `condition_number`, `tol` | backstop |
| `SecureAggregationError` | `SECURE_AGG_FAILED` | Mask reconstruction fails or surviving participants fall below the secure-agg threshold | `round`, `present`, `threshold`, `cause` | retry |
| `NonDeterministicAggregation` | `AGG_NONDETERMINISTIC` | The per-step determinism self-check detects a non-bitwise-reproducible outer step | `round`, `expected_hash`, `got_hash` | fail-closed |
| `PrivacyBudgetExceeded` | `DP_BUDGET_EXCEEDED` | The accountant reports cumulative `(ε,δ)` has reached/exceeded the configured budget | `epsilon_spent`, `epsilon_budget`, `delta`, `round` | operator |
| `CommitmentMismatch` | `COMMITMENT_MISMATCH` | A released `Δ_c` is bound to no root, the wrong root, or a root that does not match the committed `R_c` | `participant_id`, `expected_root`, `got_root`, `round` | fail-closed |
| `MerkleVerificationError` | `MERKLE_VERIFY_FAILED` | A Merkle root or inclusion proof does not verify against the recomputed tree | `expected_root`, `got_root`, `leaf_index` | fail-closed |
| `SchemaVersionMismatch` | `SCHEMA_VERSION_MISMATCH` | An on-disk `schema_version` is unknown or newer than the reader supports | `field`, `file_schema_version`, `reader_max_version` | operator |
| `CheckpointIntegrityError` | `CHECKPOINT_INTEGRITY` | A checkpoint's recomputed content hash differs from its header / committed hash | `path`, `expected_hash`, `got_hash` | fail-closed |
| `FaultToleranceExceeded` | `FAULT_TOLERANCE_EXCEEDED` | Surviving participants in a round fall below the minimum for a valid outer step | `round`, `present`, `minimum` | operator |
| `ProbeError` | `PROBE_INVALID` | Probe content hash ≠ committed hash, or probe coverage / landmark count is insufficient (`k < d`) | `expected_hash`, `got_hash`, `num_landmarks`, `d` | fail-closed |
| `EvaluationError` | `EVALUATION_FAILED` | Latent-MPC planning, metric computation, or the eval harness fails (env unavailable, planner non-convergence flagged as error) | `env_id`, `phase`, `cause` | retry |
| `AggregationError` (parent) | `AGGREGATION_FAILED` | Generic aggregation-path failure not covered by a leaf | `round`, `cause` | retry |
| `GaugeError` (parent) | `GAUGE_FAILED` | Generic gauge failure not covered by a leaf | `round`, `cause` | backstop |
| `ProvenanceError` (parent) | `PROVENANCE_FAILED` | Generic provenance failure not covered by a leaf | `cause` | fail-closed |
| `RoundError` (parent) | `ROUND_FAILED` | Round lifecycle failure not covered by a leaf (bad transition, timeout without churn cause) | `round`, `from_state`, `to_state` | retry |
| `LensembleError` (base) | `INTERNAL` | An unclassified internal failure, or a wrapped stdlib exception at a boundary | `wrapped` (the original type name) | operator |

No fail-closed error carries any raw observation, action, or private-embedding tensor in its fields or
message; carried fields are scalars, hashes, shapes, counts, and identifiers only (see
[05 — Observability §redaction](05-observability.md) and `INV-RESIDENCY`).

## 4. `LensembleErrorCode` enum

`LensembleErrorCode` is a stable, string-valued enum in `lensemble/errors.py`. Codes are append-only
across pre-1.0 minors and frozen at 1.0; a removed code is reserved, never reused. Codes are emitted into
structured logs and the CLI exit-code mapping (§7).

```python
from enum import Enum

class LensembleErrorCode(str, Enum):
    # core / config
    INTERNAL = "internal"
    CONFIG_INVALID = "config_invalid"
    # contracts (WMCP)
    WMCP_CONTRACT_VIOLATION = "wmcp_contract_violation"
    # residency (security-critical)
    RESIDENCY_VIOLATION = "residency_violation"
    # gauge
    GAUGE_FAILED = "gauge_failed"
    FRAME_DRIFT_EXCEEDED = "frame_drift_exceeded"
    PROCRUSTES_DEGENERATE = "procrustes_degenerate"
    # aggregation
    AGGREGATION_FAILED = "aggregation_failed"
    SECURE_AGG_FAILED = "secure_agg_failed"
    AGG_NONDETERMINISTIC = "agg_nondeterministic"          # security-critical
    # privacy
    DP_BUDGET_EXCEEDED = "dp_budget_exceeded"
    # provenance
    PROVENANCE_FAILED = "provenance_failed"
    COMMITMENT_MISMATCH = "commitment_mismatch"            # security-critical
    MERKLE_VERIFY_FAILED = "merkle_verify_failed"
    # artifacts
    SCHEMA_VERSION_MISMATCH = "schema_version_mismatch"
    CHECKPOINT_INTEGRITY = "checkpoint_integrity"
    # round lifecycle
    ROUND_FAILED = "round_failed"
    FAULT_TOLERANCE_EXCEEDED = "fault_tolerance_exceeded"
    # probe
    PROBE_INVALID = "probe_invalid"
    # evaluation
    EVALUATION_FAILED = "evaluation_failed"
```

## 5. Failure-mode catalog by subsystem

Each entry states the **trigger**, the **detection** point (where the guard runs), the **error** raised,
and the **system response** including the invariant enforced ([conventions §7](conventions.md#7-named-invariants)). Subsystems map 1:1 to the module
taxonomy in [01 — Architecture](01-architecture.md).

### 5.1 Data & residency (`lensemble.data`, `lensemble.data.residency`)

- **Residency breach.** *Trigger:* the training/serialization path attempts to place a raw observation,
  action, or a private-data embedding into an outbound message or artifact that crosses a trust boundary
  (the boundaries of [01 — Architecture](01-architecture.md) and
  [RFC-0001 §6](../rfcs/RFC-0001-architecture.md#6-trust-boundaries)). *Detection:* the egress guard in
  `lensemble.data.residency` inspects every tensor about to be serialized into a `PseudoGradient`,
  `Update` message, checkpoint, log, or metric, and matches its declared role against the per-dataset
  non-exportable flag ([RFC-0004 §2](../rfcs/RFC-0004-data-provenance.md#2-residency-the-sovereignty-guarantee-inv-residency)). *Error:* `ResidencyViolation`
  (`RESIDENCY_VIOLATION`). *Response:* **fail-closed** — the operation aborts, the partially-formed
  message/artifact is discarded, the round transitions to `ABORTED`, and the violation is logged with the
  tensor *role* only (never the tensor). Enforces **INV-RESIDENCY**. This error is never caught-and-
  ignored anywhere in the codebase.

- **Dataset ingest schema failure.** *Trigger:* an episode store (`lance`/`hdf5`/`lerobot://`) yields
  records whose `(o_t, a_t, o_{t+1})` shape or dtype does not match the loader contract, or a `Window`
  with the wrong `num_steps`. *Detection:* the loader validates each `Window` at read time. *Error:*
  `ConfigError` (`CONFIG_INVALID`) for a misconfigured loader, or `ContractViolation`
  (`WMCP_CONTRACT_VIOLATION`) when the produced `LatentState` would violate the WMCP shape/dtype contract.
  *Response:* **operator** — ingest stops; remediation names the offending record and field.

### 5.2 Contracts / WMCP (`lensemble.contracts`)

- **Latent nonconformance.** *Trigger:* an encoder emits a `LatentState` whose shape `(N, d)`, dtype, or
  `wmcp_version` does not match the pinned contract, or a predictor receives one that does not.
  *Detection:* the conformance check at the encoder→predictor and the message-ingress boundaries.
  *Error:* `ContractViolation` (`WMCP_CONTRACT_VIOLATION`) with `expected_shape`/`got_shape`. *Response:*
  **fail-closed** — the model is not constructed / the message is rejected. Enforces **INV-WMCP**. Detail
  in [RFC-0007](../rfcs/RFC-0007-wmcp-latent-contract.md).

- **ActionSpec mismatch.** *Trigger:* an `ActionSpec` (dimensionality, bounds, discrete/continuous,
  embodiment id, units) fails validation, or an action head is built against a spec that does not match
  the embodiment's declared spec. *Detection:* `ActionSpec` is validated before any action head
  `h_ψ^(c)` is constructed. *Error:* `ContractViolation` (`WMCP_CONTRACT_VIOLATION`). *Response:*
  **fail-closed** — head construction fails. Enforces **INV-WMCP**; per-embodiment heads are local and
  never aggregated (**INV-ACTIONHEAD-LOCAL**), so a malformed head cannot poison the shared model.

### 5.3 Gauge (`lensemble.gauge`)

- **Frame drift beyond threshold.** *Trigger:* immediately before an outer step, the inter-participant
  Procrustes residual / mean rotation angle on the public probe `P` exceeds the configured threshold —
  the gauge is re-opening during federated fine-tuning
  ([RFC-0002 §4](../rfcs/RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)). *Detection:* `frame_drift(...)`
  computed each round before aggregation (the headline diagnostic of
  [RFC-0005 §2](../rfcs/RFC-0005-evaluation.md#2-headline-diagnostic--latent-frame-drift)). *Error:* `FrameDriftExceeded` (`FRAME_DRIFT_EXCEEDED`)
  with `drift_angle_deg`, `threshold_deg`, `participant_pair`. *Response:* **backstop** — the Layer-3
  Procrustes re-alignment fires: each participant's terminal linear map is aligned to the consensus frame
  on `P` (and the predictor I/O conjugated) before averaging
  ([RFC-0002 §5](../rfcs/RFC-0002-gauge-and-aggregation.md#5-layer-3--procrustes-re-alignment-at-aggregation-backstop)). The error is informational-but-typed: it
  is raised, logged at WARNING, and handled by the backstop within the same round; it does not abort.
  With Layer-2 anchoring active this should rarely fire ([conventions §12](conventions.md#12-milestones-and-stages), Stage B characterizes the threshold).

- **Degenerate Procrustes SVD.** *Trigger:* the SVD `E_ref^⊤ f_θ(P) = U Σ V^⊤` used to compute
  `Q* = V U^⊤` (Variant B, [RFC-0002 §4](../rfcs/RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)) has near-zero or
  near-tied singular values, making `Q*` ill-defined and its gradient unstable. *Detection:* condition-
  number / minimum-singular-value check against a tolerance after the SVD, before forming `Q*`. *Error:*
  `DegenerateProcrustes` (`PROCRUSTES_DEGENERATE`) with `min_singular_value`, `condition_number`, `tol`.
  *Response:* **backstop** — clamp/condition the singular values (floor them at the tolerance) and proceed
  with the conditioned `Q*`; if the alignment was a backstop for `FrameDriftExceeded`, fall back to the
  Variant-A landmark constraint for that round. Logged at WARNING.

### 5.4 Aggregation (`lensemble.aggregation`)

- **Secure-aggregation dropout below threshold.** *Trigger:* participants vanish mid-round so that the
  surviving set cannot reconstruct the masked sum, or pairwise-mask reconstruction otherwise fails
  ([RFC-0003 §5](../rfcs/RFC-0003-federated-protocol.md#5-secure-aggregation-requirement),
  [RFC-0011](../rfcs/RFC-0011-secure-aggregation.md)). *Detection:* the secure-agg protocol checks the
  surviving count against the secret-sharing threshold before revealing the sum. *Error:*
  `SecureAggregationError` (`SECURE_AGG_FAILED`) with `present`, `threshold`. *Response:* **retry** — the
  round is abandoned and re-attempted; if survivors are also below the round minimum the
  `FaultToleranceExceeded` path (§5.8) takes over. The masked sum is never partially revealed.

- **Nondeterministic aggregation.** *Trigger:* the outer step produces a result that is not a bitwise-
  reproducible function of (committed deltas, round seed, prior global params) — e.g. an atomic reduction,
  a non-fixed summation order, or device nondeterminism leaked onto the aggregation path. *Detection:* the
  per-outer-step determinism self-check recomputes the reduction under the fixed order and compares the
  content hash ([conventions §9](conventions.md#9-determinism-dtype-device), [RFC-0006 §3](../rfcs/RFC-0006-verifiable-contribution.md#3-phase-1-proof-ready-requirements-cheap-to-honor-now)). *Error:*
  `NonDeterministicAggregation` (`AGG_NONDETERMINISTIC`) with `expected_hash`, `got_hash`. *Response:*
  **fail-closed** — abort the outer step, do not commit the global model, log, and recompute under the
  deterministic reduction path; if it recurs, the run halts for operator inspection. Enforces
  **INV-AGG-DETERMINISM**. This error is security-critical (it would silently break Phase-2 proof-
  readiness) and is never swallowed.

### 5.5 Privacy (`lensemble.privacy`)

- **DP budget exhausted.** *Trigger:* the `(ε,δ)` accountant
  ([RFC-0012](../rfcs/RFC-0012-differential-privacy.md)) reports cumulative spend has reached the
  configured budget over the planned rounds. *Detection:* the accountant is queried before each round
  releases pseudo-gradients. *Error:* `PrivacyBudgetExceeded` (`DP_BUDGET_EXCEEDED`) with `epsilon_spent`,
  `epsilon_budget`. *Response:* **operator / stop training** — no further `Update` messages are released;
  the run stops cleanly at the last committed global model. Remediation: increase the budget (a policy
  decision) and resume from the committed checkpoint, or accept the trained model.

- **Clip-bound violation (defensive).** *Trigger:* a `PseudoGradient` is presented for release whose L2
  norm exceeds `C_clip` after the clip step — a programming error in the clip path. *Detection:* a post-
  clip assertion that `‖Δ_c‖ ≤ C_clip` ([conventions §9](conventions.md#9-determinism-dtype-device), [RFC-0012](../rfcs/RFC-0012-differential-privacy.md)).
  *Error:* `LensembleError` base (`INTERNAL`) — this should be unreachable in correct code, so it is
  classified internal, not a DP-policy error. *Response:* **operator** — abort release; enforces
  **INV-DP-BOUND**. The clip is applied before noising; this guard ensures the privacy analysis holds.

### 5.6 Provenance (`lensemble.provenance`)

- **Commitment mismatch.** *Trigger:* a released `Δ_c` is bound to no dataset root, to a root that does
  not match the `R_c` the participant committed in its `Commitment` message, or to more than one root
  ([RFC-0003 §7](../rfcs/RFC-0003-federated-protocol.md#7-determinism-concurrency-error-propagation),
  [RFC-0014](../rfcs/RFC-0014-provenance-commitments.md)). *Detection:* the coordinator checks the
  binding at `Update` ingress. *Error:* `CommitmentMismatch` (`COMMITMENT_MISMATCH`) with
  `expected_root`, `got_root`. *Response:* **fail-closed** — the update is rejected and excluded from the
  round (the round proceeds with the remaining valid updates, subject to §5.8). Enforces
  **INV-COMMIT-BINDING**. Security-critical; never swallowed.

- **Merkle verification failure.** *Trigger:* a recomputed dataset Merkle root, or an inclusion proof for
  an episode, does not verify ([RFC-0014](../rfcs/RFC-0014-provenance-commitments.md)). *Detection:*
  verification at commitment ingest and during any audit/recompute. *Error:* `MerkleVerificationError`
  (`MERKLE_VERIFY_FAILED`). *Response:* **fail-closed** — the commitment is rejected; the associated
  update cannot be bound and is excluded.

### 5.7 Artifacts (`lensemble.artifacts`)

- **Schema/version mismatch.** *Trigger:* an on-disk artifact (manifest, commitment, checkpoint header,
  report) carries an integer `schema_version` the reader does not recognize or that is newer than the
  reader's maximum ([conventions §10](conventions.md#10-versioning-and-schema-policy), [03 — Data Model](03-data-model.md),
  [RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md)). *Detection:* the pydantic v2 reader checks
  `schema_version` first, before parsing the body. *Error:* `SchemaVersionMismatch`
  (`SCHEMA_VERSION_MISMATCH`) with `file_schema_version`, `reader_max_version`. *Response:* **operator** —
  for an older known version, the registered migration function runs and the load proceeds; for an unknown
  or too-new version, the load fails with remediation pointing to the upgrade path. Forward-compatible
  readers tolerate unknown *optional* fields but never an unknown *version*.

- **Checkpoint integrity / tamper.** *Trigger:* a checkpoint's recomputed SHA-256 content hash over its
  canonical bytes differs from the hash in its header or from the hash committed in `Commitment` /
  `RoundClose` ([RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md),
  [RFC-0006 §3](../rfcs/RFC-0006-verifiable-contribution.md#3-phase-1-proof-ready-requirements-cheap-to-honor-now)). *Detection:* hash recomputation on every
  load. *Error:* `CheckpointIntegrityError` (`CHECKPOINT_INTEGRITY`) with `expected_hash`, `got_hash`.
  *Response:* **fail-closed** — the checkpoint is not loaded. Enforces **INV-CHECKPOINT-HASH**. Loading
  uses `safetensors` only; `pickle`/`torch.save` artifacts are rejected at this boundary (arbitrary-code-
  execution and nondeterminism risk, [RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md)).

### 5.8 Round lifecycle (`lensemble.federation`, `lensemble.federation.round`)

- **Participant churn beyond tolerance.** *Trigger:* across `COLLECTING`, participants drop until the
  surviving set is below the configured minimum for a valid outer step
  ([RFC-0003 §6](../rfcs/RFC-0003-federated-protocol.md#6-heterogeneity--fault-tolerance),
  [RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md)). *Detection:* the coordinator's collect loop
  compares the present count against the minimum at the `COLLECTING → AGGREGATING` transition. *Error:*
  `FaultToleranceExceeded` (`FAULT_TOLERANCE_EXCEEDED`) with `present`, `minimum`. *Response:*
  **operator** — the round transitions to `ABORTED` rather than aggregate too few participants;
  the prior committed global model stands; late/dropped participants reconcile next round (DiLoCo
  elasticity). Below this threshold the design deliberately does not silently proceed.

- **Invalid state transition / round timeout.** *Trigger:* the round state machine
  (`OPEN → COLLECTING → AGGREGATING → ALIGNING → COMMITTING → CLOSED`, with `ABORTED`,
  [RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md)) is asked for an undefined transition, or a phase
  exceeds its timeout without a churn cause. *Detection:* the state machine guards every transition.
  *Error:* `RoundError` (`ROUND_FAILED`) with `from_state`, `to_state`. *Response:* **retry** — the round
  is aborted and re-opened; a recurring transition fault halts for operator inspection.

### 5.9 Probe (`lensemble.data.probe`)

- **Probe hash mismatch.** *Trigger:* the public probe `P` a participant uses does not hash to the value
  committed in `RoundOpen`, or landmark targets are not derived from `f_ref`
  ([RFC-0004 §3](../rfcs/RFC-0004-data-provenance.md#3-the-public-probe-set-mathcalp),
  [RFC-0002 §4](../rfcs/RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)). *Detection:* the participant verifies the
  probe content hash against `RoundOpen` before computing anchor targets; the coordinator re-verifies at
  alignment. *Error:* `ProbeError` (`PROBE_INVALID`) with `expected_hash`, `got_hash`. *Response:*
  **fail-closed** — the participant refuses the round until the pinned probe is present. Enforces
  **INV-PROBE-PIN**.

- **Probe under-coverage.** *Trigger:* the landmark count `k < d`, so the `k` generic absolute
  constraints do not pin a unique frame (Variant A,
  [RFC-0002 §4](../rfcs/RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)). *Detection:* a precondition check at
  `probe build`/`probe pin`. *Error:* `ProbeError` (`PROBE_INVALID`) with `num_landmarks`, `d`.
  *Response:* **fail-closed** — the probe is rejected at build time, before any round depends on it.

### 5.10 Evaluation (`lensemble.eval`)

- **Eval harness / planning failure.** *Trigger:* the latent-MPC planner cannot run (env id unavailable,
  goal spec missing), a metric computation fails, or the harness cannot load a checkpoint
  ([RFC-0005](../rfcs/RFC-0005-evaluation.md)). *Detection:* the harness validates `env_id`, goal spec,
  and checkpoint before planning. *Error:* `EvaluationError` (`EVALUATION_FAILED`) with `env_id`, `phase`;
  a missing checkpoint surfaces as `CheckpointIntegrityError`/`SchemaVersionMismatch` from §5.7, not as a
  generic eval error. *Response:* **retry** — eval is side-effect-free, so a failed evaluation is logged
  and re-attempted; it never mutates training state.

### 5.11 Configuration (`lensemble.config`)

- **Invalid or inconsistent config.** *Trigger:* a value fails OmegaConf/frozen-dataclass validation, an
  override `key=value` targets an unknown key, or two keys are mutually inconsistent (e.g. a deterministic-
  aggregation flag disabled while a verifiable-contribution path is requested)
  ([RFC-0009](../rfcs/RFC-0009-configuration-reproducibility.md)). *Detection:* validation at config load,
  before any model or data is touched. *Error:* `ConfigError` (`CONFIG_INVALID`) with `key`, `value`,
  `expected`. *Response:* **operator** — the run does not start; remediation names the key and the
  expected form.

## 6. Invariant → enforcing site → error map

This table is the cross-index from each named invariant ([conventions §7](conventions.md#7-named-invariants)) to where it is enforced and which error
fires on violation. Every invariant in the corpus that this section touches appears here.

| Invariant | Enforced in | Error on violation | §5 entry |
|---|---|---|---|
| `INV-RESIDENCY` | `lensemble.data.residency` egress guard | `ResidencyViolation` | 5.1 |
| `INV-WMCP` | `lensemble.contracts` conformance check | `ContractViolation` | 5.2 |
| `INV-ACTIONHEAD-LOCAL` | `lensemble.model.action_head` / federation broadcast filter | `ContractViolation` (if an attempt to aggregate a head occurs) | 5.2 |
| `INV-WARMSTART-T0` | round-0 warm-start load (`lensemble.model.encoder`) | `CheckpointIntegrityError` (hash ≠ pinned warm-start) | 5.7 |
| `INV-SKETCH-CONSISTENCY` | `RoundOpen` ingress; sketch derivation from `s_t` | `ConfigError` / `RoundError` (seed mismatch at ingress) | 5.8 |
| `INV-PROBE-PIN` | `lensemble.data.probe` hash check | `ProbeError` | 5.9 |
| `INV-COMMIT-BINDING` | `lensemble.provenance` binding check at `Update` ingress | `CommitmentMismatch` | 5.6 |
| `INV-CHECKPOINT-HASH` | `lensemble.artifacts.checkpoint` load-time hash | `CheckpointIntegrityError` | 5.7 |
| `INV-DP-BOUND` | `lensemble.privacy.dp` post-clip assertion | `LensembleError`/`INTERNAL` (defensive) | 5.5 |
| `INV-AGG-DETERMINISM` | `lensemble.aggregation` per-step self-check | `NonDeterministicAggregation` | 5.4 |

`INV-WARMSTART-T0` and `INV-SKETCH-CONSISTENCY` do not have dedicated error classes; their violations
surface through the integrity/config/round errors noted above, which is sufficient because both are
detected at a boundary (warm-start load; `RoundOpen` ingress) where a typed boundary error already fires.

## 7. Error-handling rules

These rules are normative for all code under `lensemble/` and are checked in review and (where mechanizable)
by lint.

1. **No bare `except` and no broad `except Exception` on the hot paths.** Handlers name the type(s) they
   handle. A broad catch is permitted only at the CLI/process supervisory top level, where it logs the
   error (with `.code` and `.remediation`), maps it to an exit code, and exits non-zero. The aggregation,
   residency, provenance, and probe paths never use a broad catch.

2. **Never swallow security-critical errors.** `ResidencyViolation`, `CommitmentMismatch`, and
   `NonDeterministicAggregation` are fail-closed: they are never caught-and-ignored, never downgraded to a
   log-and-continue, and never retried without first removing the cause. Catching one to *augment context
   and re-raise* is allowed; catching one to suppress it is a defect ([conventions §6](conventions.md#6-error-taxonomy)).

3. **Validate at boundaries.** The four ingress boundaries — config load
   ([RFC-0009](../rfcs/RFC-0009-configuration-reproducibility.md)), federation message ingress
   ([RFC-0003 §7](../rfcs/RFC-0003-federated-protocol.md#7-determinism-concurrency-error-propagation)), artifact load
   ([RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md)), and dataset ingest
   ([RFC-0004 §2](../rfcs/RFC-0004-data-provenance.md#2-residency-the-sovereignty-guarantee-inv-residency)) — raise a typed `LensembleError` with remediation.
   Interior code may then assume validated inputs and use defensive `INTERNAL` assertions only for
   "cannot happen" conditions.

4. **Wrap stdlib exceptions at the boundary.** A `ValueError`/`OSError`/`RuntimeError` surfacing from a
   dependency at a boundary is caught there and re-raised as the appropriate `LensembleError` subclass with
   `wrapped` set to the original type name, so public APIs raise only `LensembleError`.

5. **Every error carries `.code` and `.remediation`.** Raising a `LensembleError` without a concrete,
   actionable remediation is a defect. Codes come from `LensembleErrorCode` (§4) only.

6. **Recovery posture is explicit.** A handler chooses exactly one of: retry (re-attempt the operation or
   defer to the next round), backstop (invoke the defined fallback — Procrustes re-alignment, SVD
   conditioning), fail-closed (abort, do not proceed), or operator (stop and surface). The posture for
   each error is fixed in §3 and §5; handlers do not override a fail-closed error into a retry.

### 7.1 Logging and CLI mapping

Errors are logged as structured records carrying `code`, `remediation`, the round/participant correlation
ids, and the redacted scalar fields from §3 — never raw tensors (`INV-RESIDENCY`, redaction contract in
[05 — Observability §redaction](05-observability.md)). The CLI ([02 — Public API §CLI](02-public-api.md))
maps each error category to a stable non-zero exit code so scripts and CI can branch on failure class:
config/validation errors, security-critical fail-closed errors, transient/retryable errors, and internal
errors each occupy a distinct exit-code band. The exact code-to-exit-code table is owned by
[02 — Public API](02-public-api.md) and referenced, not duplicated, here.

## 8. Open questions

OPEN QUESTION: The numeric `FrameDriftExceeded` threshold (the `threshold_deg` that fires the Layer-3
backstop) is not yet fixed. Owner `@AbdelStark`; resolution path: the `λ_anc` and drift sweep in Stage B
([RFC-0002 §7](../rfcs/RFC-0002-gauge-and-aggregation.md#7-the-central-hyperparameter),
[RFC-0005 §6](../rfcs/RFC-0005-evaluation.md#6-ablation-ladder-the-core-experiment)), milestone v0.2.

OPEN QUESTION: The minimum-participant count for `FaultToleranceExceeded` and the secure-agg dropout
threshold for `SecureAggregationError` are deployment-policy values, not yet pinned. Owner `@AbdelStark`;
resolution path: Stage C real-node deployment ([RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md),
[RFC-0011](../rfcs/RFC-0011-secure-aggregation.md)), milestone v0.3.

OPEN QUESTION: Whether the per-clip `INV-DP-BOUND` defensive guard should raise a dedicated
`PrivacyError` leaf rather than reusing `INTERNAL`. Owner `@AbdelStark`; resolution path: follow-up to
[RFC-0012](../rfcs/RFC-0012-differential-privacy.md) once the clip path is implemented and the failure is
confirmed unreachable in correct code, milestone v0.2.

RISK: The set of stdlib exceptions that can surface from `torch`, `lance`, and `h5py` at the artifact and
dataset boundaries is large and version-dependent ([conventions §11](conventions.md#11-external-dependencies) dependency constraints). The boundary-wrapping
rule (§7 rule 4) is the mitigation, but exhaustively mapping each dependency exception to a `LensembleError`
subclass is incremental. Resolution plan: maintain a boundary-wrap table per dependency, extended as new
exception types are observed in the test suite and Stage-A/B runs; tracked under `area:core`.

## 9. References

- [conventions document](conventions.md): [§6](conventions.md#6-error-taxonomy) (error taxonomy),
  [§7](conventions.md#7-named-invariants) (named invariants),
  [§9](conventions.md#9-determinism-dtype-device) (determinism/dtype contract),
  [§10](conventions.md#10-versioning-and-schema-policy) (versioning).
- [01 — Architecture](01-architecture.md) — module taxonomy and trust boundaries.
- [02 — Public API](02-public-api.md) — public surface, CLI exit-code mapping.
- [03 — Data Model](03-data-model.md) — typed schemas the validation guards check against.
- [05 — Observability](05-observability.md) — structured error logging and the redaction contract.
- [06 — Security](06-security.md) — threat model behind the fail-closed posture.
- [RFC-0001 — Architecture & System Overview](../rfcs/RFC-0001-architecture.md) — trust boundaries, model.
- [RFC-0002 — The Latent Gauge & Frame-Anchored Aggregation](../rfcs/RFC-0002-gauge-and-aggregation.md) —
  frame drift, Procrustes, anchoring.
- [RFC-0003 — Federated Training Protocol](../rfcs/RFC-0003-federated-protocol.md) — round structure, DP,
  secure-agg pointer, message table, fault tolerance.
- [RFC-0004 — Data, Sovereignty & Provenance](../rfcs/RFC-0004-data-provenance.md) — residency, probe,
  commitments.
- [RFC-0005 — Evaluation & Benchmark Protocol](../rfcs/RFC-0005-evaluation.md) — frame-drift diagnostic,
  metrics.
- [RFC-0006 — Verifiable Contribution](../rfcs/RFC-0006-verifiable-contribution.md) — proof-ready
  determinism and commitment requirements.
- [RFC-0007 — WMCP Latent Contract & Embodiment Adapters](../rfcs/RFC-0007-wmcp-latent-contract.md) —
  `LatentState`/`ActionSpec` conformance.
- [RFC-0009 — Configuration, Run Manifest & Reproducibility](../rfcs/RFC-0009-configuration-reproducibility.md) —
  config validation.
- [RFC-0010 — Checkpoint & Artifact Format](../rfcs/RFC-0010-artifact-checkpoint-format.md) — hashing,
  schema versioning, tamper detection.
- [RFC-0011 — Secure Aggregation Protocol](../rfcs/RFC-0011-secure-aggregation.md) — dropout threshold.
- [RFC-0012 — Differential Privacy Accounting](../rfcs/RFC-0012-differential-privacy.md) — clip/noise,
  budget.
- [RFC-0013 — Coordinator & Participant Runtime](../rfcs/RFC-0013-coordinator-runtime.md) — round state
  machine, fault tolerance.
- [RFC-0014 — Provenance Commitments & Merkle Scheme](../rfcs/RFC-0014-provenance-commitments.md) — Merkle
  verification, binding.
