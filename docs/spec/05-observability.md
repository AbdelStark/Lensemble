# 05 — Observability

Stable contract for how Lensemble emits observability data: structured logs, metrics, traces, and the
headline frame-drift diagnostic, plus the redaction discipline that keeps every emission path safe under
the data-residency invariant. Rationale and the full instrumentation design live in
[RFC-0015](../rfcs/RFC-0015-observability-diagnostics.md); the diagnostic is specified scientifically in
[RFC-0002 §9](../rfcs/RFC-0002-gauge-and-aggregation.md#9-the-frame-drift-diagnostic-the-headline-measurement) and consumed
by [RFC-0005 §2](../rfcs/RFC-0005-evaluation.md#2-headline-diagnostic--latent-frame-drift).

The reference implementation lives in `lensemble/observability/`: `logging.py` (log records),
`metrics.py` (metric taxonomy and emission), `redaction.py` (the residency guard on every sink) — all
public-stable surfaces under [02 — Public API](02-public-api.md).

Governing rule: **observability must never widen the trust boundary.** Anything emitted to a log line,
metric, diagnostic, or trace is by construction allowed to leave a sovereign participant's process, so the
redaction guard (`observability.redaction`, [#5-redaction-inv-residency](#5-redaction-inv-residency))
sits in front of every sink and fails closed.

## 1. Structured logging

### 1.1 Record schema

Every log emission is a single JSON object on one line (newline-delimited JSON, "JSONL"). The reference
implementation uses the stdlib `logging` module with a JSON formatter; `structlog` is a permitted drop-in
that emits the identical schema. The schema is validated as a pydantic v2 model with an explicit integer
`schema_version` (consistent with the on-disk metadata policy of [conventions §10](conventions.md#10-versioning-and-schema-policy) and
[03 — Data Model](03-data-model.md)).

```python
from pydantic import BaseModel, Field
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

class LogRecord(BaseModel):
    schema_version: int = 1
    ts: str                       # RFC-3339 UTC timestamp, microsecond precision
    level: LogLevel               # see §1.2
    event: str                    # canonical event name, dotted, e.g. "round.aggregating"
    round: int | None = None      # outer-round index t; None outside a round
    participant_id: str | None = None   # stable participant id; None for coordinator-global events
    correlation_id: str           # ties a record to one round/participant context (§4)
    code: str | None = None       # LensembleErrorCode value when the record reports an error
    payload: dict[str, "RedactedScalar"] = Field(default_factory=dict)  # redaction-cleared (§5)
```

`RedactedScalar` is the union of types the redaction guard permits in `payload`: `int`, `float`, `str`
(bounded length), `bool`, and nested `dict`/`list` of the same. Raw tensors and any private observation,
action, or embedding are categorically excluded (enforced in §5).

Required on every record: `schema_version`, `ts`, `level`, `event`, `correlation_id`. The triple
`(round, participant_id, correlation_id)` is the reconstruction key for tracing (§4). `participant_id` is
the stable, non-secret identity assigned by the coordinator at federation join (see
[06 — Security](06-security.md)); never a raw network address or a key.

### 1.2 Levels and their meaning

| Level | Meaning | Examples |
|---|---|---|
| `DEBUG` | Step-level detail, off by default in federation runs. | inner-step loss, sketch seed derivation |
| `INFO` | Normal lifecycle events. | round opened/closed, participant joined, checkpoint committed |
| `WARNING` | Recoverable degradation; a backstop fired. | Procrustes backstop engaged, participant dropout above zero but below threshold, near-degenerate SVD clamped |
| `ERROR` | A typed `LensembleError` raised and handled. | `SecureAggregationError`, `SchemaVersionMismatch`, `FrameDriftExceeded` |
| `CRITICAL` | Fail-closed security event; the run aborts. | `ResidencyViolation`, `CommitmentMismatch`, `NonDeterministicAggregation` |

Error and critical records carry `code` set to the `LensembleErrorCode` enum value
([04 — Error Model](04-error-model.md)). The class-to-level mapping is fixed: the three security-critical,
never-swallowed errors (`ResidencyViolation`, `CommitmentMismatch`, `NonDeterministicAggregation`) log at
`CRITICAL` before the process fails closed; all other typed errors log at `ERROR`. This gives a
one-to-one correspondence between a raised typed error and a log record, so the failure-mode catalog in
[04 — Error Model](04-error-model.md) is auditable from logs alone.

### 1.3 Canonical lifecycle events

Event names are dotted, lower-case, and stable. The round-lifecycle events mirror the `RoundState`
state machine of [RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md):

`round.open`, `round.collecting`, `round.aggregating`, `round.aligning`, `round.committing`,
`round.close`, `round.aborted`; plus `participant.join`, `participant.local_round_start`,
`participant.pseudo_gradient_emitted`, `participant.dropout`, `checkpoint.committed`,
`probe.pinned`, `probe.verified`, `dp.budget_step`, `gauge.backstop_fired`,
`eval.episode_done`. Each event is documented with its required `payload` keys in
[RFC-0015 §2](../rfcs/RFC-0015-observability-diagnostics.md).

## 2. Metric taxonomy

Metrics are scalar time series. Each metric has a canonical name, a unit, an emission cadence, and a set
of dimension keys (always a subset of `{round, participant_id}`). Metric records share the redaction guard
and the schema_version discipline of §1; the on-disk metrics record is:

```python
class MetricRecord(BaseModel):
    schema_version: int = 1
    ts: str                       # RFC-3339 UTC
    name: str                     # canonical metric name, see tables below
    value: float
    unit: str                     # e.g. "scalar", "deg", "bytes", "s", "ms", "ratio", "count"
    round: int | None = None
    participant_id: str | None = None
    correlation_id: str
```

All metric values are derived statistics (losses, norms, angles, counts, ratios): a scalar reduction over
private data carries no recoverable raw datum, so metrics are redaction-safe by construction. The guard
(§5) still inspects every record, because a malformed instrumentation call could attempt a non-scalar;
such an attempt fails closed.

### 2.1 Training metrics (`loss/*`, `grad_norm`)

Emitted at inner-step cadence on the participant (or single site in Stage A). The three loss terms map
exactly onto the objective of [conventions §2](conventions.md#2-mathematical-notation) / [RFC-0008](../rfcs/RFC-0008-model-objective-numerics.md), which
returns per-term scalars precisely so they can be logged here.

| Metric | Unit | Cadence | Dimensions | Meaning |
|---|---|---|---|---|
| `loss/pred` | scalar | per inner step | `participant_id` | prediction term $\lambda_{\text{pred}}\,\mathbb{E}\lVert g_\phi(f_\theta(x_t),a_t)-\text{sg}[f_\theta(x_{t+1})]\rVert^2$ (weighted) |
| `loss/sigreg` | scalar | per inner step | `participant_id` | $\lambda_{\text{sig}}\,\mathrm{SIGReg}_A(f_\theta(x))$ |
| `loss/anchor` | scalar | per inner step | `participant_id` | $\lambda_{\text{anc}}\,\mathcal{L}_{\text{anchor}}(f_\theta;\mathcal{P},\{t_i\})$ |
| `grad_norm` | scalar | per inner step | `participant_id` | L2 norm of the inner-loop gradient (pre-clip) |

### 2.2 Gauge metrics (`gauge/*`)

The instrumentation of the scientific core
([RFC-0002](../rfcs/RFC-0002-gauge-and-aggregation.md)). The per-pair drift series is the headline
diagnostic and has its own emission contract in §3; the scalars below are the per-round summaries.

| Metric | Unit | Cadence | Dimensions | Meaning |
|---|---|---|---|---|
| `gauge/drift_angle_deg` | deg | per round, per pair | `round`, `participant_id` (pair encoded, §3) | mean inter-participant rotation angle on the probe $\mathcal{P}$ |
| `gauge/procrustes_residual` | scalar | per round, per pair | `round`, `participant_id` | post-alignment Frobenius residual $\lVert f_{\theta_c}(\mathcal{P})Q^\star - f_{\theta_{c'}}(\mathcal{P})\rVert_F$ |
| `gauge/effective_dim` | scalar | per round, per participant | `round`, `participant_id` | effective dimension of the embedding covariance on $\mathcal{P}$; guards against silent partial collapse (RFC-0005 §4) |

`gauge/effective_dim` is the collapse sentinel: success-rate alone can mask partial representation
collapse, so the eigenspectrum-derived effective dimension is emitted every round (definition in
[RFC-0005 §4](../rfcs/RFC-0005-evaluation.md#4-supporting-metrics)).

### 2.3 Federation metrics (`fed/*`)

Emitted by the coordinator once per outer round.

| Metric | Unit | Cadence | Dimensions | Meaning |
|---|---|---|---|---|
| `fed/round_seconds` | s | per round | `round` | wall time of the outer round (open → close) |
| `fed/participants` | count | per round | `round` | number of participants contributing to this round's aggregate |
| `fed/comm_bytes` | bytes | per round | `round`, `participant_id` | bytes of the pseudo-gradient $\Delta_c$ that crossed the boundary; summed over participants gives the round's communication cost |
| `fed/quant_ratio` | ratio | per round | `participant_id` | compression ratio of int8 pseudo-gradient quantization vs fp32 (RFC-0003 §6); 1.0 when quantization is off |

`fed/comm_bytes` and `fed/round_seconds` are the accounting basis for the DiLoCo communication-efficiency
claim (communicate every $H$ steps); the comms accountant that produces `fed/comm_bytes` is specified in
[08 — Performance Budget](08-performance-budget.md).

### 2.4 Privacy metrics (`dp/*`)

Emitted on the differential-privacy path
([RFC-0012](../rfcs/RFC-0012-differential-privacy.md)).

| Metric | Unit | Cadence | Dimensions | Meaning |
|---|---|---|---|---|
| `dp/epsilon_cumulative` | scalar | per round | `round` | cumulative $\varepsilon$ spent across rounds so far, from the accountant; the budget that triggers `PrivacyBudgetExceeded` |
| `dp/clip_fraction` | ratio | per round | `participant_id` | fraction of participant updates whose norm exceeded $C_{\text{clip}}$ and were clipped (`INV-DP-BOUND`) |

`dp/epsilon_cumulative` is the monotone series the accountant compares against the configured budget; the
round at which it crosses the budget is the round `PrivacyBudgetExceeded` is raised and training stops
(see [04 — Error Model](04-error-model.md)). `δ` is a fixed configuration value recorded in the
`RunManifest`, not a per-round metric.

### 2.5 Evaluation metrics (`eval/*`)

Emitted by the eval harness ([RFC-0005](../rfcs/RFC-0005-evaluation.md)) during latent-MPC rollout.

| Metric | Unit | Cadence | Dimensions | Meaning |
|---|---|---|---|---|
| `eval/success_rate` | ratio | per eval run | (env encoded in `event` payload) | fraction of held-out episodes solved, from `world.evaluate` |
| `eval/planning_samples` | count | per action | — | CEM/iCEM/MPPI samples per planning step |
| `eval/time_per_action_ms` | ms | per action | — | planner wall time per action, for baseline parity |

## 3. Frame-drift diagnostic emission contract

The headline empirical artifact of the paper. Its scientific definition is in
[RFC-0002 §9](../rfcs/RFC-0002-gauge-and-aggregation.md#9-the-frame-drift-diagnostic-the-headline-measurement) and
[RFC-0005 §2](../rfcs/RFC-0005-evaluation.md#2-headline-diagnostic--latent-frame-drift); its emission
schema is fixed here so the central figure (naive `FedAvg` frames rotate apart; the anchored configuration
stays flat) is **reproducible from logs alone** without rerunning training.

### 3.1 What is computed

After each outer round $t$, on the fixed public probe $\mathcal{P}$ (content-hash pinned, `INV-PROBE-PIN`),
for every unordered participant pair $(c, c')$ and for every participant against the global model, compute
the optimal Procrustes rotation $Q^\star = VU^\top$ (closed form from the SVD; see
[RFC-0002 §4](../rfcs/RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix))
between the two encoders' probe embeddings, then report both the mean rotation angle (degrees) and the
post-alignment Frobenius residual.

### 3.2 Emission record

```python
class FrameDriftRecord(BaseModel):
    schema_version: int = 1
    ts: str
    round: int                    # outer-round index t
    pair: tuple[str, str]         # (participant_id_c, participant_id_c'); sorted; or
                                  # ("global", participant_id) for drift-from-consensus
    rotation_angle_deg: float     # mean rotation angle of Q* on P, in degrees
    procrustes_residual: float    # post-alignment Frobenius residual on P
    probe_hash: str               # SHA-256 of the probe content; MUST equal the pinned round-open hash
    weights_hash_c: str           # content hash of participant c's committed encoder (INV-CHECKPOINT-HASH)
    weights_hash_c_prime: str     # content hash of c' (or the global model) encoder
    correlation_id: str
```

The per-round, per-pair records are written to a dedicated diagnostic JSONL sink (a metrics-stream
specialization, §6) and the two scalars are also mirrored into the `gauge/drift_angle_deg` and
`gauge/procrustes_residual` metric series of §2.2. The full `FrameDriftReport` aggregate type (the
per-run rollup consumed by the figure-generation code) is defined in
[03 — Data Model](03-data-model.md) and produced by `lensemble.gauge.frame_drift`.

### 3.3 Reproducibility requirement

The diagnostic MUST be a deterministic, pure function of (the pinned public probe $\mathcal{P}$, the
committed encoder weights of the two parties). This is the same recomputability property that makes
Layer-3 alignment publicly checkable without a zero-knowledge proof
([RFC-0002 §5](../rfcs/RFC-0002-gauge-and-aggregation.md#5-layer-3--procrustes-re-alignment-at-aggregation-backstop),
[RFC-0006 §3](../rfcs/RFC-0006-verifiable-contribution.md)). Two consequences are recorded directly in
the `FrameDriftRecord`:

- `probe_hash` MUST equal the probe content hash committed at round open (`INV-PROBE-PIN`, enforced in
  `lensemble.data.probe`); a mismatch raises `ProbeError` and the diagnostic for that round is rejected
  (see [04 — Error Model](04-error-model.md)).
- `weights_hash_c` / `weights_hash_c_prime` MUST equal the committed checkpoint hashes
  (`INV-CHECKPOINT-HASH`, enforced in `lensemble.artifacts.hashing`); this binds the figure to exactly the
  weights that produced it, so anyone with the public probe and the committed weights recomputes the same
  numbers. This recomputation is the Phase-1 proof-readiness test in
  [RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md) (public alignment recomputation via
  `recompute_alignment`).

Because the diagnostic computes over the **public** probe and **committed weight hashes** — never over
raw private observations — it crosses the trust boundary safely by construction (the probe is public data,
RFC-0002 §4). The redaction guard (§5) still clears every `FrameDriftRecord` before it reaches a sink.

### 3.4 Cost note

The per-pair computation is $O(C^2)$ Procrustes solves per round. This is acceptable at Stage-B
participant counts. RISK: at large $C$ this dominates per-round diagnostic cost. Resolution plan: sample a
fixed random subset of pairs (seeded from the round sketch seed $s_t$ so the sample is itself
reproducible) above a configured $C$ threshold, and always retain every participant-vs-global pair. See
the open question in
[RFC-0015](../rfcs/RFC-0015-observability-diagnostics.md) (owner @AbdelStark, resolution in Stage B
sweeps).

## 4. Tracing and round reconstruction

Lensemble does not depend on a distributed-tracing backend; correlation is achieved with deterministic
correlation IDs so a round can be reconstructed offline from the log and metric JSONL files alone.

- **Round correlation id.** When the coordinator opens round $t$, it derives `correlation_id =
  "round-{t}"` and stamps it on every coordinator record for that round (`round.open` …
  `round.close`/`round.aborted`).
- **Participant correlation id.** A participant's local-round records carry
  `correlation_id = "round-{t}/p-{participant_id}"`, so the participant's inner-loop loss series, its
  `participant.pseudo_gradient_emitted` event, and its DP `dp/clip_fraction` metric all join on the same
  key.
- **Reconstruction.** To rebuild round $t$: filter all log and metric records whose `correlation_id`
  starts with `round-{t}`. The result is the complete causal trace of the round: which participants
  contributed (`fed/participants`, `participant.join`), the inner-loop trajectories per participant, the
  aggregation and alignment events (`round.aggregating`, `round.aligning`, `gauge.backstop_fired`), the
  committed checkpoint (`checkpoint.committed` with its content hash), and the frame-drift records (§3).

Correlation IDs are derived deterministically from `(round, participant_id)`; they contain no secret and
no raw datum, so they are themselves redaction-safe. The round index $t$ and participant ids are the same
values recorded in the `ContributionRecord` ledger entry
([RFC-0014](../rfcs/RFC-0014-provenance-commitments.md)), so a log-reconstructed round can be
cross-checked against the audit ledger.

## 5. Redaction (`INV-RESIDENCY`)

**This is the security-critical core of observability.** `INV-RESIDENCY` ([conventions §7](conventions.md#7-named-invariants)) states that no raw
observation, action, or private-embedding tensor is serialized into any outbound message or artifact that
crosses a trust boundary. Logs, metrics, traces, and diagnostics are all such artifacts: they are written
to disk and may be shipped off-host. Therefore every emission passes through the redaction guard in
`lensemble.observability.redaction`, which sits in front of every sink and **fails closed**.

### 5.1 The guard contract

```python
class RedactionViolation:  # internal marker, raised as ResidencyViolation
    ...

def redact(record: dict) -> dict:
    """Clear a log/metric/diagnostic record for emission.

    Returns the record unchanged if every field is residency-safe.
    Raises ResidencyViolation (LensembleErrorCode.RESIDENCY_VIOLATION) — never returns a
    partially-cleared record — if any field carries a disallowed type or value.
    """
    ...
```

The guard classifies every value reaching a sink:

| Class | Examples | Disposition |
|---|---|---|
| **Permitted** | content hashes (SHA-256 hex), L2 norms, tensor shapes (tuple of ints), counts, scalar metrics, rotation angles, residuals, round/participant ids, correlation ids, error codes | passes through |
| **Forbidden** | any `torch.Tensor`/`numpy.ndarray` of raw observations, actions, or private embeddings; raw byte buffers of episode data; any field whose type is not in the `RedactedScalar` union | raises `ResidencyViolation`, fails closed |

The permitted set is exactly hashes, norms, shapes, counts, and scalar metrics. Any tensor is forbidden,
because a tensor of probe-or-private embeddings is indistinguishable to the guard from raw data. To log a
tensor-derived quantity, the caller reduces it to a permitted scalar (a norm, count, or hash) *before* the
emission call; the guard rejects the tensor itself.

### 5.2 Why fail-closed, and why never swallowed

`ResidencyViolation` is one of the three errors that must never be caught-and-ignored ([conventions §6](conventions.md#6-error-taxonomy), [conventions §7](conventions.md#7-named-invariants);
[04 — Error Model](04-error-model.md)). A redaction failure means instrumentation tried to emit private
data across a boundary; continuing would breach sovereignty silently, which is the worst failure mode the
system has. So the guard raises, the violation logs at `CRITICAL` with `code = RESIDENCY_VIOLATION`, and
the process aborts the run. There is no degraded mode that drops the offending field and continues, because
a partially-cleared emission cannot be proven safe.

Consistent with the [conventions §6](conventions.md#6-error-taxonomy) rule "validate at boundaries", the guard runs at the sink write (the boundary),
not optimistically at the call site. A buggy or malicious instrumentation call elsewhere in the codebase
therefore cannot exfiltrate private data through a log line: the sink-side guard is the last line of
defense.

### 5.3 Relationship to the frame-drift diagnostic

The frame-drift diagnostic (§3) computes over the public probe and committed weight hashes, so it is
residency-safe by construction, yet it still passes through the guard — deliberately, since it is the
most-shipped artifact in the system. The guard verifies a `FrameDriftRecord` carries only scalars and
hashes (rotation angle, residual, probe hash, weight hashes) and never an embedding tensor, before write.

## 6. Sinks

The canonical, reproducibility-grade sinks are plain files; richer backends are optional adapters that
never replace the files.

| Sink | Format | Content | Status |
|---|---|---|---|
| Structured log file | JSONL of `LogRecord` (§1) | lifecycle events, warnings, errors, criticals | **canonical** |
| Metrics file | JSONL of `MetricRecord` (§2) | all scalar metric series | **canonical** |
| Frame-drift diagnostic file | JSONL of `FrameDriftRecord` (§3) | the headline diagnostic, per round per pair | **canonical** |
| TensorBoard | event files | metrics mirror | optional adapter |
| Weights & Biases | run stream | metrics + events mirror | optional adapter |

The files are canonical because the headline figure and every reported number must be reproducible from a
released artifact bundle ([conventions §13](conventions.md#13-authoring-conventions); [09 — Release & Versioning](09-release-and-versioning.md) ships the
JSONL files alongside checkpoints and Hydra configs). TensorBoard and Weights & Biases are write-through
mirrors selected by configuration; they receive **already-redacted** records, so the residency guarantee
holds for third-party sinks too. An unavailable adapter is non-load-bearing: the canonical files are still
written and the run proceeds.

Every sink record carries `schema_version`, so a reader handles forward-compatible schema evolution and
raises `SchemaVersionMismatch` on an unknown or too-new version ([conventions §10](conventions.md#10-versioning-and-schema-policy);
[04 — Error Model](04-error-model.md)).

OPEN QUESTION: the default optional metrics backend (TensorBoard vs Weights & Biases vs neither) for the
reference research runs. Owner @AbdelStark; resolution path: Stage B, recorded in
[RFC-0015](../rfcs/RFC-0015-observability-diagnostics.md) Open Questions.

## 7. Invariants enforced in this section

| Invariant | Where enforced here | Error on violation |
|---|---|---|
| `INV-RESIDENCY` | `observability.redaction.redact`, in front of every sink (§5); fails closed | `ResidencyViolation` (logged `CRITICAL`, run aborts) |
| `INV-PROBE-PIN` | `FrameDriftRecord.probe_hash` checked against the round-open commitment (§3.3) | `ProbeError` (diagnostic for the round rejected) |
| `INV-CHECKPOINT-HASH` | `FrameDriftRecord.weights_hash_*` checked against committed checkpoint hashes (§3.3) | `CheckpointIntegrityError` |

These are the observability-side enforcement points. The same invariants are enforced at their primary
sites — residency in `lensemble.data.residency`, probe pinning in `lensemble.data.probe`, checkpoint
hashing in `lensemble.artifacts.hashing` — and are cross-referenced from
[06 — Security](06-security.md) and [04 — Error Model](04-error-model.md).

## 8. References

- [RFC-0015 — Observability, Diagnostics & Telemetry](../rfcs/RFC-0015-observability-diagnostics.md) — the
  instrumentation design and rationale this section contracts against.
- [RFC-0002 — The Latent Gauge & Frame-Anchored Aggregation](../rfcs/RFC-0002-gauge-and-aggregation.md) —
  the frame-drift diagnostic definition (§9) and the publicly-recomputable alignment (§5).
- [RFC-0005 — Evaluation & Benchmark Protocol](../rfcs/RFC-0005-evaluation.md) — the headline diagnostic
  (§2), supporting metrics including effective dimension (§4), and reproducibility/reporting (§8).
- [RFC-0006 — Verifiable Contribution](../rfcs/RFC-0006-verifiable-contribution.md) — the public
  recomputation that the diagnostic's determinism enables.
- [RFC-0012 — Differential Privacy Accounting](../rfcs/RFC-0012-differential-privacy.md) — source of the
  `dp/*` metrics and `PrivacyBudgetExceeded`.
- [RFC-0013 — Coordinator & Participant Runtime](../rfcs/RFC-0013-coordinator-runtime.md) — the
  `RoundState` machine the lifecycle events mirror.
- [RFC-0014 — Provenance Commitments & Merkle Scheme](../rfcs/RFC-0014-provenance-commitments.md) — the
  `ContributionRecord` ledger a reconstructed round cross-checks against.
- [03 — Data Model](03-data-model.md), [04 — Error Model](04-error-model.md),
  [06 — Security](06-security.md), [08 — Performance Budget](08-performance-budget.md),
  [09 — Release & Versioning](09-release-and-versioning.md) — sibling spec sections referenced above.
