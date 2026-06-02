# RFC-0015 — Observability, Diagnostics & Telemetry

| | |
|---|---|
| **RFC** | 0015 |
| **Title** | Observability, Diagnostics & Telemetry |
| **Slug** | observability-diagnostics |
| **Status** | Accepted |
| **Track** | Standards |
| **Authors** | @AbdelStark |
| **Created** | 2026-06-02 |
| **Target milestone** | v0.1 (Stage A — logging/metrics scaffold) → v0.2 (Stage B — the frame-drift diagnostic) |
| **Area** | observability |
| **Requires** | RFC-0002 (the frame-drift diagnostic), RFC-0005 (evaluation metrics), RFC-0004 (residency), RFC-0009 (run manifest, correlation seed), RFC-0010 (committed-weights hash) |

> This RFC specifies the instrumentation contract of Lensemble: the structured log record, the metric
> taxonomy, the redaction guard that enforces `INV-RESIDENCY` on every emission path, and — the headline
> empirical artifact — the per-round, per-participant-pair frame-drift diagnostic record from which the
> central figure of the paper is reproducible from logs alone. It owns the `lensemble.observability`
> subsystem (`logging.py`, `metrics.py`, `redaction.py`).

## Summary

Lensemble's empirical claims (RFC-0005 §1 claims 1–3) and its sovereignty guarantee (`INV-RESIDENCY`)
both live or die on observability. The headline result — that naive end-to-end `FedAvg` of a JEPA
diverges in latent frame while the anchored design holds it pinned — is a *measurement*, not a side
effect, and must be reproducible from committed weights and the public probe alone. At the same time,
every byte that leaves a participant through a log line or a metric sample is a potential residency
breach: a logged raw observation, action, or private embedding would defeat the entire trust model.

This RFC defines four contracts. (1) A **structured JSON log record** with a fixed required-field set
and level semantics, emitted via the stdlib `logging` module behind a JSON formatter. (2) A **metric
taxonomy** with canonical names (`loss/*`, `gauge/*`, `fed/*`, `dp/*`, `eval/*`), units, and cadence,
emitted as a metrics JSONL stream. (3) The **frame-drift diagnostic emission contract**: the per-round,
per-pair record carrying the Procrustes residual and mean rotation angle on the public probe
$\mathcal{P}$, plus drift-from-global, in a versioned schema, deterministic given committed weights +
the pinned probe (RFC-0002 §9, RFC-0005 §2). (4) The **redaction guard** in `observability.redaction`
that fails closed (`ResidencyViolation`) before any raw tensor or private datum reaches a sink.

It enforces `INV-RESIDENCY` on the emission path and consumes `INV-PROBE-PIN`, `INV-CHECKPOINT-HASH`,
and `INV-AGG-DETERMINISM` so that the diagnostic record is verifiable and reproducible.

## Motivation

The central scientific contribution is a quantity that must be plotted: latent frame drift over rounds
(RFC-0002 §9). If that quantity is recomputed differently by the coordinator, a reviewer, and the
Phase-2 public-recomputation path (RFC-0006 §4), the figure is an artifact of the plotting code, not a
result. The diagnostic therefore needs a *pinned emission schema* and a determinism guarantee — it is
the input `recompute_alignment` reproduces, so its schema and hash dependencies (committed weights, probe
hash) are load-bearing for verifiability.

Observability is also the enforcement surface for sovereignty. `INV-RESIDENCY` ([conventions §7](../spec/conventions.md#7-named-invariants)) forbids any raw
observation/action/private-embedding tensor from crossing a trust boundary, and logs and metrics *are* a
boundary crossing — written to disk, shipped to dashboards, read by humans. A naive
`logger.debug("batch=%s", batch)` is a residency breach. The redaction guard makes that breach
structurally impossible: a tensor or private datum is rejected, and only derived statistics (hashes, L2
norms, shapes, counts, scalar metrics) pass.

Federation multiplies the diagnostic surface: a round is a distributed event across $C$ participant
processes and one coordinator, so reconstructing "what happened in round $t$" from interleaved logs
requires correlation identifiers. This RFC defines that correlation scheme.

## Goals

- Define the structured JSON **log record schema** (required fields, level semantics) and name the
  reference implementation (`logging` + JSON formatter), so every subsystem logs the same shape.
- Define the **metric taxonomy**: canonical metric names, unit, and cadence for `loss/*`, `gauge/*`,
  `fed/*`, `dp/*`, `eval/*`, with the emission point for each (the producing subsystem).
- Define the **frame-drift diagnostic emission contract** precisely enough that the headline figure
  (RFC-0002 §9, RFC-0005 §2) is reproducible from the metrics stream and committed weights alone;
  guarantee it is deterministic given committed weights + the pinned probe.
- Define **tracing**: round/participant correlation IDs and the procedure to reconstruct a round.
- Define the **redaction guard** (`observability.redaction`): the allow-list of emittable types, the
  reject path, and that it raises `ResidencyViolation` and fails closed (`INV-RESIDENCY`).
- Define the **sinks**: the canonical structured log file + metrics JSONL, and optional adapters
  (tensorboard / wandb) that route through the same redaction guard.
- State every failure mode of the emission path with the error raised ([conventions §6](../spec/conventions.md#6-error-taxonomy)) and the system response.

## Non-Goals

- The frame-drift *algorithm* (`frame_drift`, `procrustes_align`, the rotation-angle derivation) is
  owned by [RFC-0002 §9](RFC-0002-gauge-and-aggregation.md); this RFC owns only its *emission*: the
  record schema, cadence, and reproducibility guarantee.
- The *meaning* of each metric in the evaluation argument — what success rate or effective dimension
  *proves* — is owned by [RFC-0005](RFC-0005-evaluation.md); this RFC owns the metric name, unit, and
  cadence, not the experiment.
- The residency-enforcement mechanism on the *training/wire* path (refusing to serialize raw tensors
  into a `PseudoGradient` or a message) is owned by [RFC-0004 §2](RFC-0004-data-provenance.md) and
  `lensemble.data.residency`. This RFC enforces `INV-RESIDENCY` only on the *observability* path.
- The `RunManifest` schema (config hash, seeds, env, git SHA) is owned by
  [RFC-0009](RFC-0009-configuration-reproducibility.md); this RFC consumes the root seed (to derive
  correlation IDs) and references the manifest from log records, it does not define it.
- Choosing the default external metrics backend (tensorboard vs wandb vs none) and the large-$C$
  pair-sampling policy are Open Questions, not normative defaults of this RFC.

## Proposed Design

The subsystem is three modules with a strict dependency rule: every emission — log line, metric sample,
diagnostic record, adapter write — passes through `observability.redaction` before it reaches a sink.
There is no emission path that bypasses redaction; that is the structural enforcement of `INV-RESIDENCY`.

```
                  subsystem call sites (model / gauge / federation / privacy / eval / ...)
                                         |
                  emit_log() / emit_metric() / emit_diagnostic()
                                         |
                                         v
                         observability.redaction  (allow-list guard)   <-- INV-RESIDENCY, fail-closed
                                         |  (raises ResidencyViolation on a raw tensor / private datum)
                          +--------------+--------------+
                          v                             v
                 observability.logging         observability.metrics
                 (JSON log record)             (metric sample / diagnostic record)
                          |                             |
                          v                             v
              <run_dir>/lensemble.log.jsonl   <run_dir>/metrics.jsonl
                                         |
                                         v
                       optional adapters: tensorboard / wandb (read the same redacted stream)
```

In prose: subsystems never write to a file or dashboard directly. They call `emit_log`, `emit_metric`,
or `emit_diagnostic`; each call routes through the redaction guard, which inspects every field value
against an allow-list and raises `ResidencyViolation` if any field is a tensor or private datum.
Surviving records are written to the canonical log file and metrics JSONL. Optional tensorboard/wandb
adapters subscribe to the already-redacted metric stream, so they cannot reintroduce raw data.

### 1. Structured logging (`observability.logging`)

The reference implementation uses the Python stdlib `logging` module with a JSON formatter (a
`logging.Formatter` subclass serializing the record's structured `extra` payload to one JSON line),
chosen over `structlog` to avoid an extra runtime dependency and keep CPU-only CI light ([conventions §11](../spec/conventions.md#11-external-dependencies) lists no
logging dependency; `structlog` is the rejected alternative below). Decision (chosen under [conventions §1](../spec/conventions.md#1-repository-and-package-layout)
underspecification): one JSON object per line (JSONL), UTF-8, to `<run_dir>/lensemble.log.jsonl`; each
record is flat, with nested data under `payload`.

```python
from enum import Enum
from typing import Any, Mapping

class LogLevel(str, Enum):
    DEBUG = "DEBUG"      # development tracing; never emitted in CI gate runs by default
    INFO = "INFO"        # normal lifecycle events (round open/close, commit, eval start)
    WARN = "WARN"        # recovered/degraded conditions (backstop fired, participant dropped, SVD clamped)
    ERROR = "ERROR"      # round-fatal or run-fatal conditions (an aborted round, a raised LensembleError)

# Required fields on EVERY log record. Validated by the JSON formatter before write.
# (pydantic v2 model in observability.logging; schema_version per conventions section 10.)
class LogRecord:
    schema_version: int          # integer schema version ([conventions §10](../spec/conventions.md#10-versioning-and-schema-policy)); >=1
    timestamp: str               # RFC 3339 UTC, microsecond precision, e.g. "2026-06-02T11:04:05.123456Z"
    level: LogLevel
    event: str                   # canonical event name, dotted: "round.open", "agg.outer_step", "eval.episode_done"
    logger: str                  # producing module, e.g. "lensemble.gauge.drift"
    correlation_id: str          # ties a round across processes (see Tracing); hex of derive(root_seed, round)
    round: int | None            # outer round index; None for single-site / pre-round events
    participant_id: str | None   # stable participant id; None on the coordinator's own events
    code: str | None             # LensembleErrorCode value when level == ERROR and an error was raised
    payload: Mapping[str, Any]   # redacted structured detail; ONLY emittable types (see Redaction)
```

`event` is a closed, dotted per-subsystem vocabulary (e.g. `round.open`, `round.collecting`,
`agg.aligned`, `agg.outer_step`, `dp.budget_check`, `eval.episode_done`, `commit.checkpoint`); a new name
is additive, a rename is a `schema_version` bump ([conventions §10](../spec/conventions.md#10-versioning-and-schema-policy)). Level semantics are normative: `WARN` marks
*recovered* degradations (the Layer-3 Procrustes backstop fired and succeeded, a participant dropped but
the round met the fault-tolerance threshold, an SVD was clamped); `ERROR` marks round- or run-fatal
conditions and always carries a `code` from `LensembleErrorCode` ([conventions §6](../spec/conventions.md#6-error-taxonomy),
[04 — Error Model](../spec/04-error-model.md)).

```python
def emit_log(
    level: LogLevel,
    event: str,
    *,
    round: int | None = None,
    participant_id: str | None = None,
    code: str | None = None,
    **payload: Any,
) -> None:
    """Emit one structured log record through the redaction guard.

    Pre:  every value in `payload` is an emittable type (Redaction allow-list).
    Post: one JSON line appended to the structured log sink; correlation_id derived from
          (root_seed, round) when `round` is not None (see Tracing).
    Raises: ResidencyViolation (fail-closed) if any payload value is a tensor or private datum;
            the record is NOT written. Never swallowed ([conventions §6](../spec/conventions.md#6-error-taxonomy)).
    """
```

### 2. Metric taxonomy (`observability.metrics`)

Metrics are emitted as one JSON object per line to `<run_dir>/metrics.jsonl`. A metric sample is:

```python
class MetricSample:
    schema_version: int          # [conventions §10](../spec/conventions.md#10-versioning-and-schema-policy)
    name: str                    # canonical metric name (table below)
    value: float                 # scalar; NaN/Inf are rejected by the guard (see Failure Modes)
    unit: str                    # canonical unit string from the table
    round: int | None            # outer round for federated metrics; step index folded into payload
    step: int | None             # inner step for training metrics; None for per-round metrics
    participant_id: str | None
    correlation_id: str
    timestamp: str               # RFC 3339 UTC

def emit_metric(
    name: str, value: float, *, unit: str,
    round: int | None = None, step: int | None = None,
    participant_id: str | None = None,
) -> None:
    """Emit one scalar metric sample through the redaction guard.

    Pre:  `value` is a finite Python float (no tensor; no NaN/Inf — see Failure Modes);
          `name` and `unit` match the canonical taxonomy below.
    Raises: ResidencyViolation if `value` is a tensor; EvaluationError if `value` is non-finite
            and the metric is on the eval path; otherwise ConfigError on an unknown metric name
            under strict mode (default off pre-1.0). Never swallowed.
    """
```

Canonical metric names, unit, cadence, and emission point (the producing subsystem). These names are the
contract; the observability spec ([05 — Observability §metrics](../spec/05-observability.md)) reproduces
the same names.

| Metric name | Unit | Cadence | Emission point (subsystem) |
|---|---|---|---|
| `loss/pred` | dimensionless (mean sq. latent error) | per inner step | `model.objective` ([RFC-0008](RFC-0008-model-objective-numerics.md)) |
| `loss/sigreg` | dimensionless (Epps–Pulley statistic) | per inner step | `model.sigreg` |
| `loss/anchor` | dimensionless (unweighted $\mathcal{L}_{\text{anchor}}$) | per inner step | `gauge.anchor` ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md)) |
| `grad_norm` | L2 norm | per inner step | `model` (inner optimizer) |
| `gauge/drift_angle_deg` | degrees | per round per participant pair | `gauge.drift` ([RFC-0002 §9](RFC-0002-gauge-and-aggregation.md)) |
| `gauge/procrustes_residual` | Frobenius norm $\lVert SQ^\star - T\rVert_F$ | per round per pair (and on backstop) | `gauge.procrustes` |
| `gauge/effective_dim` | count (participation-ratio of the covariance eigenspectrum) | per round per participant | `gauge.drift` / `eval.metrics` ([RFC-0005 §4](RFC-0005-evaluation.md)) |
| `fed/round_seconds` | seconds | per round | `federation.coordinator` ([RFC-0013](RFC-0013-coordinator-runtime.md)) |
| `fed/participants` | count | per round | `federation.coordinator` |
| `fed/comm_bytes` | bytes | per round (and cumulative) | `federation` comms accountant ([RFC-0003 §6](RFC-0003-federated-protocol.md)) |
| `fed/quant_ratio` | dimensionless (full-precision bytes / quantized bytes) | per round | `federation` (int8 pseudo-gradient quantization) |
| `dp/epsilon_cumulative` | $\varepsilon$ (privacy loss) | per round | `privacy.accountant` ([RFC-0012](RFC-0012-differential-privacy.md)) |
| `dp/clip_fraction` | fraction in [0,1] | per round per participant | `privacy.dp` (`INV-DP-BOUND`) |
| `eval/success_rate` | fraction in [0,1] | per eval run | `eval.harness` ([RFC-0005 §3](RFC-0005-evaluation.md)) |
| `eval/planning_samples` | count | per eval run | `eval.mpc` (CEM/iCEM/MPPI) |
| `eval/time_per_action_ms` | milliseconds | per eval run | `eval.mpc` |

Units are part of the contract so that a downstream reader (or the Phase-2 verifier) never has to guess
whether `gauge/drift_angle_deg` is radians or degrees. `gauge/effective_dim` is the collapse guard of
[RFC-0005 §4](RFC-0005-evaluation.md): a silent partial collapse drops the participation ratio of the
embedding-covariance eigenspectrum even while `eval/success_rate` looks healthy, so it is emitted every
round per participant.

### 3. The frame-drift diagnostic emission contract (the headline artifact)

This is the central empirical artifact (RFC-0002 §9, RFC-0005 §2). The *computation* is `frame_drift`
(owned by [RFC-0002 §9](RFC-0002-gauge-and-aggregation.md)), which returns a `FrameDriftReport`
(`lensemble.gauge`, schema authored in [03 — Data Model](../spec/03-data-model.md)). This RFC owns its
*emission*: the on-disk record so the headline figure is reproducible from logs alone.

```python
# pydantic v2 model in observability.metrics; written to metrics.jsonl alongside scalar samples,
# tagged record_kind == "frame_drift" so a reader can filter the diagnostic stream.
class FrameDriftRecord:
    schema_version: int                                  # [conventions §10](../spec/conventions.md#10-versioning-and-schema-policy)
    record_kind: str = "frame_drift"
    round_index: int
    probe_hash: str                                      # the pinned probe; INV-PROBE-PIN
    global_checkpoint_hash: str                          # committed (theta_t, phi_t); INV-CHECKPOINT-HASH
    participant_checkpoint_hash: Mapping[str, str]       # participant_id -> committed local checkpoint hash
    # one entry per UNORDERED pair (c, c') with a canonical ordering c < c' (deduplicated):
    pairwise_angle_deg: list["PairAngle"]                # (c, c', mean rotation angle in degrees)
    pairwise_residual: list["PairResidual"]              # (c, c', Procrustes residual ||SQ*-T||_F)
    drift_from_global_deg: Mapping[str, float]           # participant_id -> angle to the global model on P
    pair_sampling: str                                   # "full" or "sampled:<scheme>"  (large-C policy)
    timestamp: str                                       # RFC 3339 UTC

class PairAngle:
    c: str; c_prime: str; angle_deg: float               # c < c' lexicographically (canonical order)

class PairResidual:
    c: str; c_prime: str; residual: float
```

Reproducibility contract (this is the verifiability tie-in):

1. **Determinism.** A `FrameDriftRecord` is a pure function of (the committed checkpoint hashes it names,
   the pinned probe identified by `probe_hash`). Given the same committed weights and the same probe,
   `frame_drift` produces an identical report (RFC-0002 §9 determinism note), so the record is
   byte-reproducible up to the float formatting fixed below. This rides on `INV-AGG-DETERMINISM`
   ([RFC-0003 §7](RFC-0003-federated-protocol.md)): the Procrustes SVDs upcast to fp32/fp64 with a fixed
   reduction order, no atomics.
2. **Float formatting.** All floats in the diagnostic record are serialized with `repr(float)` (shortest
   round-trippable decimal) so the JSONL is byte-stable across platforms for identical fp64 values. This
   is the canonicalization that makes the *file* reproducible, not merely the *values*.
3. **Pin binding.** The record names the `probe_hash` and every checkpoint hash it depends on, so a
   reviewer or the Phase-2 public recomputation (`recompute_alignment`,
   [RFC-0006 §4](RFC-0006-verifiable-contribution.md)) can recompute the diagnostic from public probe +
   committed weights and check it against the emitted record. The diagnostic needs no ZK proof — it is
   publicly recomputable by construction (RFC-0002 §5).
4. **Canonical pair ordering.** Pairs are stored once with `c < c'` lexicographically; this both halves
   the volume and removes order ambiguity, so two runs of the diagnostic emit the same record.

The headline figure (naive `FedAvg` diverges, anchored holds flat) is produced by reading the
`pairwise_angle_deg` (or `pairwise_residual`) series across `round_index` from `metrics.jsonl` and
plotting the mean over pairs per round, one curve per configuration — no state beyond the metrics stream,
the "reproducible from logs alone" guarantee of RFC-0005 §2. Emission cadence: one `FrameDriftRecord` per
outer round, written after `agg.aligned` and before `commit.checkpoint` (so the committed hashes it
references already exist). The scalar `gauge/drift_angle_deg` / `gauge/procrustes_residual` samples
(table §2) are the per-pair flattening of the same numbers for ad-hoc dashboards; the `FrameDriftRecord`
is the authoritative, schema-versioned artifact.

### 4. Tracing — round/participant correlation

A federated round is a distributed event. To reconstruct it from interleaved logs across the coordinator
and $C$ participants, every record carries a `correlation_id` derived deterministically from the run's
root seed and the round index:

```python
def correlation_id(root_seed: int, round_index: int) -> str:
    """Deterministic per-round correlation id, identical across the coordinator and all participants.

    Derived as the hex of derive(root_seed, round_index) using the same derive() that produces the
    per-round sketch seed s_t ([conventions §9](../spec/conventions.md#9-determinism-dtype-device), RFC-0009). Because every process shares root_seed (from the
    pinned RunManifest), they independently compute the SAME correlation_id for the same round without
    any extra coordination message.
    """
```

Reconstruction procedure (normative): to reconstruct round $t$, filter all log records and metric samples
across all participant and coordinator sinks where `correlation_id == correlation_id(root_seed, t)`. The
resulting set is the complete, ordered (by `timestamp`) trace of the round: `round.open` from the
coordinator, `round.collecting` and per-step training metrics from each participant, the
`PseudoGradient` commit events, `agg.aligned`, the `FrameDriftRecord`, `agg.outer_step`, and
`commit.checkpoint`. Because `correlation_id` is derived (not assigned by the coordinator), a participant
tags its records with the correct id even if it never received an explicit round-id message — which is
what makes the trace reconstructable under churn ([RFC-0013](RFC-0013-coordinator-runtime.md)).

### 5. Redaction (`observability.redaction`) — security-critical, `INV-RESIDENCY`

> **`INV-RESIDENCY`** ([conventions §7](../spec/conventions.md#7-named-invariants)): no raw observation/action/private-embedding tensor is serialized into any
> outbound message or artifact that crosses a trust boundary. On the observability path this is enforced
> by `observability.redaction`: every value in a log `payload`, every metric `value`, and every field of
> a diagnostic record passes the allow-list guard before any sink write. A disallowed value raises
> `ResidencyViolation` ([conventions §6](../spec/conventions.md#6-error-taxonomy)) and the record is *not written* — fail-closed. `ResidencyViolation` is
> never caught-and-ignored ([conventions §6](../spec/conventions.md#6-error-taxonomy)).

The guard is an allow-list, not a block-list, because a block-list of "things that look like private
data" is unwinnable. Only the following types are emittable:

```python
EmittableScalar = bool | int | float | str           # finite floats only; NaN/Inf rejected on metric path
# Permitted derived statistics — NEVER the underlying tensor:
#   - content hashes (hex str): checkpoint hash, probe hash, dataset Merkle root R_c, episode hash
#   - L2 / Frobenius norms (float): grad_norm, pseudo-gradient norm, Procrustes residual
#   - shapes (tuple[int, ...]) and dtypes (str): e.g. (N, d), "bfloat16"
#   - counts (int): episode counts, participant counts, sample counts
#   - scalar metrics (float): every value in the §2 taxonomy

def redact(value: object, *, field: str) -> EmittableScalar | tuple[int, ...]:
    """Return `value` iff it is on the emittable allow-list; otherwise fail closed.

    Rejects (raises ResidencyViolation, remediation in the message):
      - torch.Tensor / numpy.ndarray of any shape (raw or derived embeddings, observations, actions)
      - bytes / bytearray that are not a hex-encoded hash of the expected length
      - any object exposing a tensor-like buffer / __array__ / __torch_function__
      - mappings/sequences are recursed; a single disallowed leaf fails the whole record
    Allows: EmittableScalar, hex-hash strings, shape tuples, dtype strings.
    """
```

Concrete rule (normative): to log "what an embedding looked like", log its **shape, dtype, and L2 norm
and/or content hash** — never the embedding; to log "what data a participant trained on", log the
**dataset Merkle root $R_c$ and episode count** ([RFC-0014](RFC-0014-provenance-commitments.md)) — never
an episode. The public probe $\mathcal{P}$ is public ([RFC-0004 §3](RFC-0004-data-provenance.md)), so
probe *hashes* are loggable; probe *embeddings* feed `frame_drift` internally but only the derived
scalars (angles, residuals) and the probe *hash* are emitted. The guard runs on the participant process
*before* anything ships to the coordinator's aggregated sink, so a buggy call site cannot leak across the
boundary.

### 6. Sinks

Canonical (always written, the reproducibility substrate): a structured log file
`<run_dir>/lensemble.log.jsonl` and a metrics stream `<run_dir>/metrics.jsonl` (scalar samples and
`FrameDriftRecord`s, distinguished by `record_kind`). Both are append-only JSONL, referenced by the
`RunManifest` ([RFC-0009](RFC-0009-configuration-reproducibility.md)) so an artifact release (RFC-0005
§8) ships them. Optional adapters (tensorboard, wandb) are thin subscribers to the already-redacted metric
stream: a dashboard may sample or drop, but the canonical file retains every emitted sample so the
headline figure is reproducible offline. Adapters MUST consume the redacted stream and MUST NOT receive
raw call-site arguments. Adapter availability is gated on the optional dependency; absence degrades to
canonical-file-only with an `INFO` log, never an error.

### 7. Concurrency, determinism, error propagation

- **Per-process sinks.** Each participant process and the coordinator write their own
  `lensemble.log.jsonl` / `metrics.jsonl` under their own `<run_dir>`; correlation IDs (§4) stitch them.
  There is no cross-process shared log handle, so there is no lock contention and no nondeterministic
  interleaving *within* a process. Within a process, emission is synchronous and ordered.
- **Diagnostic determinism.** The `FrameDriftRecord` inherits `INV-AGG-DETERMINISM` from `frame_drift`
  (RFC-0002 §9): fixed reduction order, fp32/fp64 SVDs, no atomics, plus the `repr(float)` float
  canonicalization of §3. The diagnostic is on the aggregation path's reproducibility contract; a
  determinism self-check mismatch on that path raises `NonDeterministicAggregation`
  ([RFC-0003 §7](RFC-0003-federated-protocol.md)), which is never swallowed.
- **Emission failures are non-fatal except residency.** A sink write failure (disk full, permission)
  raises at the call site and is logged at `ERROR`, but it does not corrupt training state — the metrics
  stream is observational. The *one* exception is `ResidencyViolation`: it is security-critical, it fails
  closed (the record is dropped, not partially written), and it propagates ([conventions §6](../spec/conventions.md#6-error-taxonomy), never swallowed).

### 8. Failure modes and system response

| Failure mode | Trigger | Detection | Error ([conventions §6](../spec/conventions.md#6-error-taxonomy)) | System response |
|---|---|---|---|---|
| Raw tensor in a log/metric payload | A call site passes an embedding/observation/action tensor | Allow-list check in `redact()` | `ResidencyViolation` | Fail-closed: drop the record, never write it, propagate the error (never swallowed); remediation "emit shape/dtype/L2-norm/hash, never the tensor (INV-RESIDENCY)" |
| Private datum disguised as bytes | `bytes` payload that is not a valid hex hash | Length/format check in `redact()` | `ResidencyViolation` | Fail-closed as above; remediation "hash the datum (RFC-0014) and emit the hex digest" |
| Non-finite metric value | `NaN`/`Inf` reaches `emit_metric` (e.g. a diverged loss) | Finiteness check on the metric path | `EvaluationError` on the eval path; logged `ERROR` with `code` elsewhere | Reject the sample; raise on the eval path (a non-finite success rate is a test failure); on the training path log `ERROR` and let the inner-loop guard decide (training metrics are observational) |
| Unknown metric name | A name outside the §2 taxonomy under strict mode | Name lookup in `emit_metric` | `ConfigError` | Strict mode (opt-in pre-1.0, default at 1.0): reject with remediation "register the metric name in the taxonomy"; lenient mode: emit and log `WARN` |
| Diagnostic non-determinism | `FrameDriftRecord` differs across recomputation on identical committed weights + probe | Determinism self-check on the aggregation path | `NonDeterministicAggregation` (`AggregationError`) | Never swallowed; abort and recompute with fixed reduction order ([RFC-0003 §7](RFC-0003-federated-protocol.md)); a non-reproducible headline figure is treated as a correctness bug |
| Probe-hash mismatch in a diagnostic | `FrameDriftRecord.probe_hash` ≠ the pinned `GlobalState.probe_hash` | Hash comparison when the record is assembled | `ProbeError` | Refuse to emit the diagnostic (a drift number against the wrong probe is meaningless); remediation "re-pin the probe (RFC-0004 §3, INV-PROBE-PIN)" |
| Missing committed checkpoint hash | A referenced participant checkpoint has no committed hash | Lookup against the committed artifacts | `CheckpointIntegrityError` (`ArtifactError`) | Refuse to emit the diagnostic for that participant; log `ERROR`; the round's diagnostic is incomplete and flagged, not silently partial ([RFC-0010](RFC-0010-artifact-checkpoint-format.md), `INV-CHECKPOINT-HASH`) |
| Sink write failure | Disk full / permission denied on the JSONL sink | OS error at write | re-raised as `LensembleError` (logged `ERROR`) | Non-fatal to training (metrics are observational); raise at the call site, do not corrupt the partial line — write whole-line-or-nothing |
| Log/metric volume blowup | Per-pair $O(C^2)$ diagnostic at large $C$, or long-horizon per-step metrics | Volume monitor / config cadence | not an error (Open Question) | Apply the configured `pair_sampling` policy and per-step metric cadence; record the sampling scheme in `FrameDriftRecord.pair_sampling` so the figure is honest about sampling |

Error-handling rules ([conventions §6](../spec/conventions.md#6-error-taxonomy)): never a bare `except`; `ResidencyViolation` and `NonDeterministicAggregation`
are never swallowed; every emitted error carries `.code` (a `LensembleErrorCode`) and a `.remediation`
string. The redaction guard is the boundary validator for the observability path
([04 — Error Model](../spec/04-error-model.md)).

## Alternatives Considered

- **`structlog` vs stdlib `logging` + a JSON formatter.** *What:* `structlog` offers bound context
  processors and a cleaner correlation/round binding. *Why considered:* it is the idiomatic
  structured-logging choice and would make the `correlation_id`/`round` binding a processor rather than
  explicit kwargs. *Why rejected:* it adds a runtime dependency not in [conventions §11](../spec/conventions.md#11-external-dependencies), and the CPU-only CI
  configs ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)) value a zero-extra-dependency path; the stdlib `logging.Formatter` subclass meets the
  contract. It is a clean drop-in behind the same `emit_log` facade if ergonomics demand (Open Question).
- **Dashboard (tensorboard/wandb) vs JSONL file as the canonical sink.** *What:* make a dashboard the
  source of truth. *Why considered:* dashboards are where humans read metrics; wandb gives free
  experiment tracking. *Why rejected as canonical:* both sample/downsample and depend on an external
  service, breaking the "reproducible from logs alone" guarantee (RFC-0005 §2) and the Phase-2 public
  recomputation (RFC-0006 §4), and breaking the self-contained artifact release (RFC-0005 §8). The JSONL
  file is canonical and byte-stable; tensorboard/wandb are adapters on the redacted stream.
- **Block-list vs allow-list redaction.** *What:* reject "things that look like raw data" (large or
  image-shaped tensors). *Why considered:* less call-site friction. *Why rejected:* a block-list silently
  passes the unanticipated case, and for a security-critical invariant (`INV-RESIDENCY`) a false negative
  is a leak — the catastrophic failure. The allow-list fails toward over-rejection (forcing a derived
  statistic), the safe direction.
- **In-line emission vs a typed `FrameDriftRecord` facade.** *What:* let `gauge.drift` write diagnostic
  numbers directly to the log. *Why considered:* fewer layers. *Why rejected:* the headline artifact needs
  a pinned, schema-versioned, hash-bound record for reproducibility and Phase-2 recomputation; an ad-hoc
  log line has no schema, float canonicalization, or probe/checkpoint-hash binding. The `FrameDriftRecord`
  is the contract; the scalar `gauge/*` samples are the convenience flattening.

## Drawbacks

- **Metric volume at long horizons.** Per-inner-step training metrics (`loss/*`, `grad_norm`) over a long
  inner horizon $H$ across many rounds produce a large `metrics.jsonl`. Mitigation: a configurable
  per-step emission cadence (e.g. emit every $k$ steps); the per-round federated/gauge metrics are
  low-volume and always emitted.
- **$O(C^2)$ diagnostic cost.** The frame-drift diagnostic is quadratic in participant count per round
  (RFC-0002 §9 / §Drawbacks), both in Procrustes compute and in record size. At large $C$ this is real
  cost. Mitigation: the `pair_sampling` policy (Open Question) records the scheme in the diagnostic so the
  figure stays honest; `drift_from_global_deg` is only $O(C)$ and is always full.
- **Allow-list friction.** Every logged field must be an emittable type; a developer must first decide
  which derived statistic answers the question, which can push toward ad-hoc `print` (bypassing the
  guard). Mitigation: documented patterns (shape+dtype+norm+hash) and a `DEBUG`-level inspection helper
  that still routes through the guard (see RISK below).
- **Per-process sinks complicate aggregate views.** Each process writes its own JSONL, so a whole-run view
  requires stitching by `correlation_id` (§4). Mitigation: the deterministic scheme makes stitching
  mechanical; an optional collector can merge per-process files post hoc.

## Migration / Rollout

- **`v0.1` (Stage A).** The logging and metrics scaffold ships with the single-site training path: the
  JSON log record, the metric taxonomy for `loss/*` and `grad_norm`, `gauge/effective_dim` and
  `eval/*` (single-site eval), the redaction guard (live from day one — `INV-RESIDENCY` is not optional),
  the canonical JSONL sinks, and the `RunManifest` references. No federated metrics yet.
- **`v0.2` (Stage B).** The federated metrics (`fed/*`, `dp/*`) and the **frame-drift diagnostic emission
  contract** (`FrameDriftRecord`) land with simulated federation; this is when the headline figure is
  first produced. Correlation-ID tracing (§4) is needed once there are multiple participant processes.
- **`v0.3` (Stage C).** Over the real-network boundary the redaction guard's role intensifies (real data,
  real boundary): the same allow-list applies, now enforcing across a genuine trust boundary, and the
  per-process sinks become per-node. No schema change — the contract is transport-agnostic.
- **`v1.0`.** Strict metric-name mode becomes the default (unknown names raise `ConfigError`); the log,
  metric, and diagnostic `schema_version`s are frozen with the public API ([conventions §10](../spec/conventions.md#10-versioning-and-schema-policy)); optional adapters
  are documented in the reproducibility package.

Schema evolution: each of the log record, metric sample, and `FrameDriftRecord` carries an integer
`schema_version` ([conventions §10](../spec/conventions.md#10-versioning-and-schema-policy)). Adding a field is additive; renaming or removing one is a version bump with a
forward-compatible reader and a migration note. A reader rejecting an unknown/too-new version raises
`SchemaVersionMismatch` ([RFC-0010](RFC-0010-artifact-checkpoint-format.md)).

## Testing Strategy

ML- and contract-specific tests (the differentiating layers of
[07 — Testing Strategy](../spec/07-testing-strategy.md)), each runnable on CPU with tiny synthetic
fixtures and no large downloads:

- **Redaction fails closed on raw data.** Assert `emit_log`, `emit_metric`, and the diagnostic facade
  each raise `ResidencyViolation` when handed a `torch.Tensor` / `numpy.ndarray` (raw embedding,
  observation, action) and that *no* record is written (inspect the sink is unchanged). Property test
  (hypothesis): for arbitrary nested payloads, if any leaf is a tensor-like the whole record is rejected;
  if all leaves are emittable types, it is written. This is the `INV-RESIDENCY` enforcement test.
- **Allow-list completeness.** Assert hashes (hex str), L2/Frobenius norms (float), shapes (tuple),
  dtypes (str), and finite scalars pass; assert `NaN`/`Inf` on the metric path is rejected; assert
  non-hex `bytes` are rejected.
- **Diagnostic reproducibility.** Construct fixed synthetic committed weights and a pinned probe; call the
  diagnostic facade twice and assert the two `FrameDriftRecord`s are byte-identical (the `repr(float)`
  canonicalization + `INV-AGG-DETERMINISM`). Then recompute the drift independently from the committed
  weights + probe (the public-recomputation path,
  [RFC-0006 §4](RFC-0006-verifiable-contribution.md)) and assert it matches the emitted record. This is
  the "headline figure reproducible from logs alone" test.
- **Diagnostic on synthetically rotated silos (figure proxy).** Reuse the RFC-0002 §Testing fixture
  ($C$ encoders = warm-start composed with known rotations $Q_c$); assert the emitted per-round
  `pairwise_angle_deg` series for naive averaging *increases* over rounds while the anchored
  configuration stays flat/low — the unit-scale proxy for the headline figure, asserted directly on the
  emitted records.
- **Correlation-ID determinism.** Assert `correlation_id(root_seed, t)` is identical across two
  independently-seeded "processes" with the same `root_seed`, and that a full round trace is recoverable
  by filtering on it (a synthetic multi-process log fixture).
- **Probe-hash / checkpoint-hash binding.** Assert the diagnostic facade raises `ProbeError` when the
  record's probe hash ≠ the pinned probe hash, and `CheckpointIntegrityError` when a referenced
  participant checkpoint hash is missing — the diagnostic is never silently partial.
- **Schema round-trip + version.** pydantic round-trip (write → read → equality) for `LogRecord`,
  `MetricSample`, `FrameDriftRecord`; assert a too-new `schema_version` raises `SchemaVersionMismatch`;
  assert a migration function upgrades an old record.
- **Metric taxonomy conformance.** Assert every name emitted by the subsystems in the §2 table matches a
  canonical name+unit; under strict mode an unknown name raises `ConfigError`.
- **Sink robustness.** Assert a simulated write failure raises and logs `ERROR` without corrupting prior
  lines (whole-line-or-nothing), and that `ResidencyViolation` drops the offending record without
  partial write.

Numerical tolerance ([07 — Testing Strategy](../spec/07-testing-strategy.md)): the diagnostic
reproducibility test requires *bitwise* equality of the emitted JSONL (the canonicalization is the point);
the synthetically-rotated-silos figure-proxy test uses `atol/rtol` appropriate to fp32 on the *trend*
(monotone-increasing vs flat), not on absolute angles.

## Open Questions

OPEN QUESTION: The default external metrics backend (tensorboard vs wandb vs none) for the reference
configs. The canonical JSONL sink is fixed and not in question; this is only which adapter ships enabled
by default. Owner @AbdelStark. Resolution: Stage B, decided alongside the experiment-tracking workflow of
[RFC-0005 §8](RFC-0005-evaluation.md), milestone v0.2.

OPEN QUESTION: The pair-sampling policy for the $O(C^2)$ frame-drift diagnostic at large participant count
$C$ — which pairs to sample, how to keep the headline figure unbiased under sampling, and how to record
the scheme in `FrameDriftRecord.pair_sampling`. Owner @AbdelStark. Resolution: the $C$/$H$ scale sweep of
[RFC-0005 §7](RFC-0005-evaluation.md); set a full-enumeration ceiling and a sampling scheme above it,
milestone v0.2 (shared with [RFC-0002 §Drawbacks](RFC-0002-gauge-and-aggregation.md)).

OPEN QUESTION: Whether to migrate the logging implementation from stdlib `logging` to `structlog` if the
explicit-context ergonomics (`round`/`participant_id`/`correlation_id` kwargs on every `emit_log`) prove a
maintenance burden. The `emit_log` facade is the stable contract regardless. Owner @AbdelStark.
Resolution: a follow-up decision after Stage B usage, milestone v0.3.

OPEN QUESTION: The per-inner-step metric emission cadence (every step vs every $k$ steps) at long inner
horizons $H$, to bound `metrics.jsonl` volume without losing the convergence signal. Owner @AbdelStark.
Resolution: Stage B, tuned against the $H$ schedule of
[RFC-0003](RFC-0003-federated-protocol.md), milestone v0.2.

RISK: The redaction allow-list could push a frustrated developer to bypass it with `print`/`torch.save`
during debugging, defeating `INV-RESIDENCY` on a private node. Resolution plan: provide a guarded
`DEBUG`-level inspection helper that emits shape/dtype/norm/hash for any tensor (so the common debugging
need is met *inside* the guard), document it prominently, and add a CI lint that flags bare `print(` and
`torch.save(` of training tensors in `lensemble/` ([07 — Testing Strategy](../spec/07-testing-strategy.md),
[06 — Security](../spec/06-security.md)).

## References

- Internal: [RFC-0002 — Gauge & Aggregation](RFC-0002-gauge-and-aggregation.md) (§9 `frame_drift` /
  `FrameDriftReport`, §5 public recomputation, §Drawbacks the $O(C^2)$ cost);
  [RFC-0005 — Evaluation](RFC-0005-evaluation.md) (§2 headline diagnostic, §3 planning success, §4
  supporting metrics incl. effective dimension, §8 reproducibility);
  [RFC-0003 — Federated Protocol](RFC-0003-federated-protocol.md) (§7 deterministic aggregation /
  `INV-AGG-DETERMINISM`, §6 communication bytes and int8 quantization);
  [RFC-0004 — Data & Provenance](RFC-0004-data-provenance.md) (§3 the public probe / `INV-PROBE-PIN`, §2
  the residency mechanism on the training/wire path);
  [RFC-0006 — Verifiable Contribution](RFC-0006-verifiable-contribution.md) (§4 public recomputation, §3
  proof-readiness); [RFC-0008 — Model, Objective & Numerics](RFC-0008-model-objective-numerics.md) (the
  `loss/*` emission points); [RFC-0009 — Configuration & Reproducibility](RFC-0009-configuration-reproducibility.md)
  (the `RunManifest`, root seed, `derive()` shared by correlation id and sketch seed);
  [RFC-0010 — Artifact & Checkpoint Format](RFC-0010-artifact-checkpoint-format.md) (committed-weights
  hash, `INV-CHECKPOINT-HASH`, `SchemaVersionMismatch`);
  [RFC-0012 — Differential Privacy](RFC-0012-differential-privacy.md) (the `dp/*` emission points);
  [RFC-0013 — Coordinator & Participant Runtime](RFC-0013-coordinator-runtime.md) (round lifecycle events,
  churn under which correlation tracing holds);
  [RFC-0014 — Provenance Commitments](RFC-0014-provenance-commitments.md) (Merkle root $R_c$ / episode
  hashes as loggable statistics). Spec: [05 — Observability](../spec/05-observability.md) (the stable
  contract this RFC backs), [03 — Data Model](../spec/03-data-model.md) (`FrameDriftReport`),
  [04 — Error Model](../spec/04-error-model.md) (`ResidencyViolation`, `NonDeterministicAggregation`,
  `ProbeError`, `SchemaVersionMismatch`, `CheckpointIntegrityError`),
  [07 — Testing Strategy](../spec/07-testing-strategy.md), [06 — Security](../spec/06-security.md).
- External ([conventions §11](../spec/conventions.md#11-external-dependencies)): Python stdlib `logging` + a JSON formatter (chosen reference impl); `pydantic` v2
  (typed validation of the log/metric/diagnostic schemas); optional `tensorboard` / `wandb` adapters
  (experiment tracking, not the canonical sink); the orthogonal-Procrustes literature underlying the
  frame-drift quantity (consumed via RFC-0002 §9).
