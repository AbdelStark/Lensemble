# RFC-0007 — WMCP Latent Contract & Embodiment Adapters

| | |
|---|---|
| **RFC** | 0007 |
| **Title** | WMCP Latent Contract & Embodiment Adapters |
| **Slug** | wmcp-latent-contract |
| **Status** | Accepted |
| **Track** | Standards |
| **Authors** | Abdelhamid Bakhta (@AbdelStark) |
| **Created** | 2026-06-02 |
| **Target milestone** | v0.1 (Stage A) |
| **Area** | contracts |
| **Requires** | RFC-0001 |
| **Informs** | RFC-0002, RFC-0003, RFC-0004, RFC-0008, RFC-0013, RFC-0014 |

## Summary

WMCP (the World-Model Contract Protocol, originally WM-RFC-0001) is the shared latent interface that
makes heterogeneous-embodiment federation well-posed. It fixes, for the whole federation, the shape,
dtype, and semantics of the latent state every encoder $f_\theta$ emits and every predictor $g_\phi$
consumes, plus the action-conditioning interface every per-embodiment head $h_\psi^{(c)}$ must
satisfy. It is the explicit analogue of the fixed token vocabulary LLM federation gets for free:
without it, participants training "the same" model on different embodiments exchange incomparable
objects, and weight averaging ([RFC-0003 §3](RFC-0003-federated-protocol.md)) is ill-typed before the
latent gauge ([RFC-0002 §2](RFC-0002-gauge-and-aggregation.md#2-the-od-gauge-formally-preserved-verbatim))
even enters. This RFC specifies the `LatentState` contract, the `ActionSpec` descriptor, the
action-head interface, the conformance checks gating all three, and `wmcp_version` versioning plus the
federation-join gate. It owns `lensemble.contracts` and the invariants `INV-WMCP` and
`INV-ACTIONHEAD-LOCAL` ([conventions §7](../spec/conventions.md#7-named-invariants)). Consumers are deferred to their RFCs: the producing model to
[RFC-0008](RFC-0008-model-objective-numerics.md), data ingest and the public probe to
[RFC-0004](RFC-0004-data-provenance.md), the version-gating handshake to
[RFC-0013](RFC-0013-coordinator-runtime.md).

## Motivation

Lensemble federates one shared backbone across participants whose embodiments genuinely differ — a
quadruped, a 7-DoF arm, a driving stack — and whose action spaces therefore differ in
dimensionality, type, and bounds ([RFC-0001 §1](RFC-0001-architecture.md), `INV-ACTIONHEAD-LOCAL`).
For $f_\theta$ and $g_\phi$ to be a *single* model that averaging can combine, every participant must
emit and consume latent states that are shape-compatible, dtype-compatible, and semantically the same
kind of object: a per-clip set of $N$ $d$-dimensional latent tokens in the gauge-controlled frame. If
participant A emits $(N, d)$ patch-token latents in bf16 and B emits a pooled $(d,)$ vector in fp32,
their pseudo-gradients $\Delta_c$ ([RFC-0003 §3](RFC-0003-federated-protocol.md)) are not summable,
and the frame anchor ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix))
has nothing common to pin.

In a supervised net or an LLM this agreement is implicit — a fixed label set or token vocabulary makes
"the same output space" automatic, one reason such systems never see the latent gauge
([RFC-0001 Motivation](RFC-0001-architecture.md)). A self-supervised JEPA has no external anchor, so
the contract must be explicit and enforced at every boundary. WMCP is that contract, and conformance
is a *precondition for joining a federation* ([RFC-0004 §6](RFC-0004-data-provenance.md)): a
non-conforming participant is rejected at the handshake rather than allowed to corrupt the global
model. The per-embodiment difference is handled by a deliberate seam — the latent space is shared and
contract-pinned; the action space is local. Each embodiment owns a head $h_\psi^{(c)}$ mapping its
private action into the shared latent-conditioning space; the contract pins that head's *output* (so
$g_\phi$ can consume it) and leaves its *input* free to differ. Heads never cross a boundary
(`INV-ACTIONHEAD-LOCAL`).

## Goals

- Specify the `LatentState` contract: tensor shape $(N, d)$, dtype, `wmcp_version`, and the semantics
  every conforming encoder emits and every predictor consumes, enforced by `INV-WMCP`.
- Specify the conformance check that validates a `LatentState` and raises `ContractViolation` ([conventions §6](../spec/conventions.md#6-error-taxonomy))
  on any shape, dtype, or semantic mismatch, with a remediation string.
- Specify `ActionSpec`: the per-embodiment action-space descriptor (dimensionality, bounds, discrete
  vs continuous, embodiment id, units) and its validation, run before any action head is constructed.
- Specify the action-head interface $h_\psi^{(c)}: \text{ActionSpec} \to$ conditioning embedding in
  the shared latent-conditioning space — the required method signatures a conforming head implements,
  marked local and never-aggregated (`INV-ACTIONHEAD-LOCAL`).
- Specify `wmcp_version` semantics and the federation-join conformance gate that preconditions
  participation.
- Own `lensemble.contracts` ([conventions §1](../spec/conventions.md#1-repository-and-package-layout)) as the lowest typed layer above `errors`/`config`: both `model`
  and `gauge` are typed against it ([RFC-0001 §3](RFC-0001-architecture.md)).

## Non-Goals

- This RFC does not implement the encoder, predictor, or any concrete action head; it specifies the
  interfaces they satisfy. Implementations live in
  [RFC-0008](RFC-0008-model-objective-numerics.md).
- It does not specify the public probe set $\mathcal{P}$, residency enforcement, or provenance
  commitments; those are [RFC-0004](RFC-0004-data-provenance.md),
  [RFC-0014](RFC-0014-provenance-commitments.md). The contract is referenced by them (the probe must
  itself yield conforming latents; commitments carry WMCP metadata).
- It does not specify the federation transport or the handshake state machine; it specifies *what*
  the handshake checks (`wmcp_version` equality), and [RFC-0013](RFC-0013-coordinator-runtime.md)
  specifies *how* and *when*.
- It does not define a latent normalization or whitening scheme; the contract pins shape/dtype/frame
  semantics, and the objective's collapse control is SIGReg
  ([RFC-0008](RFC-0008-model-objective-numerics.md)).

## Proposed Design

### 1. Module placement and surface

`lensemble.contracts` ([conventions §1](../spec/conventions.md#1-repository-and-package-layout), area `contracts`) holds `LatentState`, `ActionSpec`, the action-head
abstract base, the conformance checks, and the `wmcp_version` constant. Its layer is L2 in the
dependency DAG ([RFC-0001 §3](RFC-0001-architecture.md)): it depends only on `errors` and `config`,
and both `model` and `gauge` depend on it. The public symbols are re-exported as documented in
[02-public-api.md](../spec/02-public-api.md); the conformance functions are public so participants can self-check
before a round. Nothing in `contracts` imports `torch` for *type definition* beyond the tensor type
alias; the runtime conformance checks do touch tensor attributes (shape, dtype, finiteness).

### 2. The `LatentState` contract

A `LatentState` is the canonical per-clip latent object. It carries the tensor plus the metadata that
makes conformance checkable without inspecting the producing model.

```python
# lensemble/contracts/latent.py
from __future__ import annotations
from dataclasses import dataclass
import torch

# The pinned contract version. A federation agrees on exactly one (INV-WMCP, §6).
WMCP_VERSION: str = "wmcp-1.0.0"

@dataclass(frozen=True, slots=True)
class LatentState:
    """A per-clip set of N latent tokens of dimension d ([conventions §2](../spec/conventions.md#2-mathematical-notation), §8)."""
    tokens: torch.Tensor      # shape (N, d) for one clip, or (B, N, d) batched
    num_tokens: int           # N: latent tokens emitted per clip by the encoder
    dim: int                  # d: latent embedding dimension
    wmcp_version: str         # MUST equal WMCP_VERSION at the gate (INV-WMCP)

    @property
    def is_batched(self) -> bool:
        return self.tokens.ndim == 3
```

Contract clauses, all enforced by `check_latent_state` (§4):

- **Shape.** `tokens` is rank-2 `(N, d)` for a single clip or rank-3 `(B, N, d)` batched. `num_tokens`
  equals `N` (the trailing-but-one axis); `dim` equals `d` (the last axis). $N$ is the number of
  latent tokens the encoder emits per clip and $d$ is the latent embedding dimension, exactly as in
  [conventions §2](../spec/conventions.md#2-mathematical-notation). Both are fixed for a given `wmcp_version` across the federation. Whether `N` may vary across
  embodiments is an Open Question (§10); for v0.1 it is fixed.
- **Dtype.** `tokens.dtype` is a floating type in the permitted set `{bf16, fp16, fp32}`. The compute
  default is bf16 forward with fp32 accumulation ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)); the contract permits any of the three so a
  CPU-fallback test path may use fp32 ([RFC-0008](RFC-0008-model-objective-numerics.md)). Integer or
  complex dtypes fail conformance.
- **Finiteness.** `tokens` contains no `NaN`/`Inf`. A non-finite latent indicates a numerical fault
  upstream and is rejected here rather than allowed to poison aggregation
  (`INV-AGG-DETERMINISM` depends on clean inputs, [RFC-0003 §7](RFC-0003-federated-protocol.md#7-determinism-concurrency-error-propagation)).
- **Semantics.** The tokens live in the *shared, gauge-controlled* latent frame: the coordinate basis
  fixed at $t{=}0$ by the warm-start ($f_{\text{ref}}$, `INV-WARMSTART-T0`) and held by the frame
  anchor ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)).
  The contract cannot *verify* the frame at the type level (a rotated frame still has shape $(N,d)$);
  it pins the *declared* frame via `wmcp_version` and defers measurement of actual drift to the gauge
  diagnostic ([RFC-0002 §9](RFC-0002-gauge-and-aggregation.md#9-the-frame-drift-diagnostic-the-headline-measurement),
  [RFC-0015](RFC-0015-observability-diagnostics.md)). The separation is deliberate: the contract is a
  cheap, local, total check; gauge alignment is a federation-wide, probe-based measurement.
- **Version.** `wmcp_version` MUST equal `WMCP_VERSION` at every boundary check (`INV-WMCP`).

`LatentState` is the type produced by `Encoder.__call__(clip: Tensor) -> LatentState`
([conventions §5](../spec/conventions.md#5-public-api-surface)) and consumed by the predictor; both the encoder and predictor are typed against it
([RFC-0008](RFC-0008-model-objective-numerics.md)).

### 3. The `ActionSpec` descriptor

`ActionSpec` describes one embodiment's action space. It is the input to action-head construction and
the declared metadata carried in a `DatasetCommitment` ([RFC-0004 §6](RFC-0004-data-provenance.md#6-data-quality-metadata-and-the-wmcp-precondition),
[RFC-0014](RFC-0014-provenance-commitments.md), [conventions §8](../spec/conventions.md#8-core-data-types)).

```python
# lensemble/contracts/action.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class ActionKind(str, Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"

@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Per-embodiment action-space descriptor ([conventions §8](../spec/conventions.md#8-core-data-types)). Local; declared at join."""
    embodiment_id: str            # stable id, e.g. "so101-arm-7dof"
    kind: ActionKind              # continuous | discrete
    dim: int                      # action dimensionality (>0)
    low: tuple[float, ...] | None # per-dim lower bounds; len==dim if continuous
    high: tuple[float, ...] | None# per-dim upper bounds; len==dim if continuous
    num_classes: int | None       # per-dim category counts if discrete; else None
    units: tuple[str, ...]        # per-dim unit label, len==dim (e.g. "rad", "m/s")
    wmcp_version: str             # MUST equal WMCP_VERSION (INV-WMCP)
```

Validation rules, enforced by `validate_action_spec` (§4), failing with `ContractViolation`:

- `dim > 0`; `len(units) == dim`.
- If `kind == CONTINUOUS`: `low` and `high` are present, `len(low) == len(high) == dim`, and
  `low[i] < high[i]` for every `i`; `num_classes is None`.
- If `kind == DISCRETE`: `num_classes` is present with `len(num_classes) == dim` and every entry
  `>= 2`; `low`/`high` are `None`.
- `embodiment_id` is non-empty and matches `^[a-z0-9][a-z0-9._-]*$` (used as a stable key and a label;
  no whitespace, so it is safe in log records and file names, see [05-observability.md](../spec/05-observability.md)).
- `wmcp_version == WMCP_VERSION`.

`ActionSpec` is frozen and hashable; its content hash is part of the embodiment metadata declared at
join and recorded in the `RunManifest` ([RFC-0009](RFC-0009-configuration-reproducibility.md)).

### 4. Conformance checks

Two total functions, both pure (no I/O, no mutation), raising `ContractViolation` with a remediation
string on any failure. They are the only sanctioned validation path; callers MUST NOT re-implement ad
hoc shape checks.

```python
# lensemble/contracts/conformance.py
from lensemble.contracts.latent import LatentState
from lensemble.contracts.action import ActionSpec

def check_latent_state(
    state: LatentState,
    *,
    expected_dim: int | None = None,
    expected_num_tokens: int | None = None,
) -> None:
    """Validate a LatentState against the WMCP contract (INV-WMCP).

    Raises ContractViolation (code WMCP_LATENT_NONCONFORMANT) on shape rank,
    (num_tokens, dim) mismatch, disallowed dtype, non-finite values, or a
    wmcp_version mismatch. No-op return on success.
    """

def validate_action_spec(spec: ActionSpec) -> None:
    """Validate an ActionSpec before action-head construction (INV-WMCP).

    Raises ContractViolation (code WMCP_ACTIONSPEC_INVALID) on any rule in §3.
    No-op return on success.
    """
```

`check_latent_state` checks, in order: `wmcp_version == WMCP_VERSION`; rank in `{2, 3}`; the trailing
axis equals `state.dim` and (for the chosen rank) the tokens axis equals `state.num_tokens`; the
optional `expected_dim`/`expected_num_tokens` match when supplied (used by the predictor to assert it
is being fed the size it was built for); dtype in `{bf16, fp16, fp32}`; `torch.isfinite(tokens).all()`.
The first failing clause determines the `.remediation` string (for example, "expected dtype in
{bfloat16, float16, float32}, got int64; cast the encoder output before emitting a LatentState").

`INV-WMCP` ([conventions §7](../spec/conventions.md#7-named-invariants)) is "every `LatentState` conforms to the pinned `wmcp_version`; every `ActionSpec`
is validated before an action head is constructed." It is enforced here: `check_latent_state` is
called at every contract boundary (encoder output, predictor input, probe ingest, any
boundary-crossing message that references a latent shape), and `validate_action_spec` is the first
statement of `build_action_head` ([RFC-0008](RFC-0008-model-objective-numerics.md)). Violation raises
`ContractViolation`, which is a hard reject — there is no recovery path that quietly reshapes data,
because a silent reshape would mask a real model bug or a real interface disagreement.

### 5. The action-head interface

The action head $h_\psi^{(c)}$ maps a raw, embodiment-specific action into the shared
latent-conditioning space the predictor consumes. The contract pins its *output* (so $g_\phi$ consumes
any conforming head's output) and its *construction* (built from a validated `ActionSpec`), leaving
its *input* free to match the embodiment.

```python
# lensemble/contracts/action_head.py
from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from lensemble.contracts.action import ActionSpec

class ActionHead(ABC):
    """Per-embodiment conditioning map (INV-ACTIONHEAD-LOCAL). LOCAL: never
    broadcast, never aggregated, never written to a shared artifact."""

    spec: ActionSpec
    cond_dim: int  # dimensionality of the shared latent-conditioning space

    @abstractmethod
    def __init__(self, spec: ActionSpec, *, cond_dim: int) -> None:
        """MUST call validate_action_spec(spec) before allocating parameters."""

    @abstractmethod
    def encode(self, action: torch.Tensor) -> torch.Tensor:
        """Map a raw action batch of shape (B, spec.dim) -- or (B,) discrete
        indices per dim -- into a conditioning embedding of shape
        (B, cond_dim) in the shared latent-conditioning space the predictor
        g_phi consumes. Output dtype follows the compute dtype ([conventions §9](../spec/conventions.md#9-determinism-dtype-device))."""

    @abstractmethod
    def state_dict_local(self) -> dict[str, torch.Tensor]:
        """Return this head's parameters for LOCAL checkpointing only.
        These parameters MUST NOT be placed in any shared ModelArtifact or
        PseudoGradient (INV-ACTIONHEAD-LOCAL, [conventions §7](../spec/conventions.md#7-named-invariants))."""
```

Contract clauses:

- `encode`'s output has last dimension `cond_dim`, the single federation-fixed conditioning
  dimensionality the predictor is built against; `cond_dim` is a shared constant (it travels with the
  model config, [RFC-0009](RFC-0009-configuration-reproducibility.md)) while the *input* `spec.dim` is
  per-embodiment. This is the seam that lets a quadruped and a 7-DoF arm condition the same $g_\phi$:
  different `spec.dim`, identical `cond_dim`.
- `__init__` MUST call `validate_action_spec(self.spec)` before allocating parameters; building a head
  from an unvalidated spec is the `INV-WMCP` violation this clause forecloses.
- The head's parameters $\psi$ are *local*. `INV-ACTIONHEAD-LOCAL` ([conventions §7](../spec/conventions.md#7-named-invariants)) is enforced in two places:
  federation broadcast/aggregation payloads exclude any `ActionHead` parameters by construction
  ([RFC-0001 §4](RFC-0001-architecture.md), in `lensemble.federation`), and the shared `ModelArtifact`
  param-group manifest contains only encoder/predictor groups
  ([RFC-0010](RFC-0010-artifact-checkpoint-format.md)). The `state_dict_local` name is deliberate:
  there is no `state_dict` a shared serializer could pick up by convention.
- The conditioning embedding crosses no trust boundary — it is consumed by the local predictor inside
  the participant's domain; only the pseudo-gradient over *shared* (encoder/predictor) parameters
  leaves ([RFC-0001 §6](RFC-0001-architecture.md), `INV-RESIDENCY`).

A conforming reference head (`build_action_head`, [conventions §5](../spec/conventions.md#5-public-api-surface)) is specified in
[RFC-0008](RFC-0008-model-objective-numerics.md); this RFC owns only the interface above.

### 6. Versioning and the federation-join gate

`wmcp_version` is a SemVer string ([conventions §10](../spec/conventions.md#10-versioning-and-schema-policy)) pinned for the whole federation for the life of a run.
The gate:

- **Encode-side.** Every `LatentState` and `ActionSpec` produced inside a participant carries
  `wmcp_version = WMCP_VERSION` of the participant's installed `lensemble.contracts`.
- **Join-side.** At the federation handshake ([RFC-0013](RFC-0013-coordinator-runtime.md)), the
  coordinator's `RoundOpen`/handshake advertises the federation's `wmcp_version`; a participant whose
  `WMCP_VERSION` differs is refused before it can contribute. The check is exact-equality on the full
  SemVer string for v0.1 (no minor-compatible negotiation yet); loosening this to "compatible minor"
  is part of the contract-extension Open Question (§10).
- **Extension.** The contract is extended by *additive optional fields* on `LatentState`/`ActionSpec`
  with a minor `wmcp_version` bump; removing or retyping a field is a major bump. Readers handle
  `wmcp_version` they recognize and reject unknown majors with `ContractViolation`. This mirrors the
  on-disk `schema_version` discipline ([conventions §10](../spec/conventions.md#10-versioning-and-schema-policy), [RFC-0010](RFC-0010-artifact-checkpoint-format.md))
  but for the in-memory latent contract.

The gate is the operational meaning of "conformance is a precondition for joining a federation"
([RFC-0004 §6](RFC-0004-data-provenance.md)): a non-conforming participant is detected at join, not
after it has corrupted an aggregation.

### 7. Failure modes and system response

| Failure | Where detected | Error ([conventions §6](../spec/conventions.md#6-error-taxonomy)) | `.code` | System response |
|---|---|---|---|---|
| Latent rank not in {2,3}, or `(num_tokens,dim)` mismatch | `check_latent_state` (encoder out, predictor in, probe ingest) | `ContractViolation` | `WMCP_LATENT_NONCONFORMANT` | Hard reject; no reshape. Remediation names the expected shape |
| Latent dtype not in {bf16,fp16,fp32} | `check_latent_state` | `ContractViolation` | `WMCP_LATENT_NONCONFORMANT` | Hard reject; remediation says to cast before emitting |
| Latent contains NaN/Inf | `check_latent_state` | `ContractViolation` | `WMCP_LATENT_NONFINITE` | Hard reject; surfaces an upstream numerical fault ([RFC-0008](RFC-0008-model-objective-numerics.md)) |
| `ActionSpec` invalid (dim/bounds/units/regex) | `validate_action_spec` (start of `build_action_head`) | `ContractViolation` | `WMCP_ACTIONSPEC_INVALID` | Block action-head construction (`INV-WMCP`) |
| `wmcp_version` on object != `WMCP_VERSION` | both checks | `ContractViolation` | `WMCP_VERSION_MISMATCH` | Reject object |
| Participant `WMCP_VERSION` != federation version | join handshake ([RFC-0013](RFC-0013-coordinator-runtime.md)) | `ContractViolation` | `WMCP_VERSION_MISMATCH` | Refuse join before any contribution |
| Action-head params found in a shared payload/artifact | federation payload build; artifact write ([RFC-0010](RFC-0010-artifact-checkpoint-format.md)) | `ContractViolation` | `WMCP_ACTIONHEAD_LEAK` | Reject the payload/artifact (`INV-ACTIONHEAD-LOCAL`); this is a programming error, fail-closed |

`ContractViolation` carries `.code: LensembleErrorCode` and `.remediation: str` ([conventions §6](../spec/conventions.md#6-error-taxonomy)). It is never
caught-and-ignored on a boundary path; a contract failure means the federation is exchanging
incomparable objects and must stop, not paper over the mismatch.

### 8. Determinism and concurrency

The conformance checks are pure and side-effect-free; they impose no ordering and add no
nondeterminism, so they are safe to call from every inner-loop worker
([RFC-0001 §8](RFC-0001-architecture.md)) and on the aggregation path, where a non-finite or
mis-shaped latent would otherwise threaten `INV-AGG-DETERMINISM`
([RFC-0003 §7](RFC-0003-federated-protocol.md#7-determinism-concurrency-error-propagation)). `check_latent_state`'s `isfinite().all()` reduction
yields an order-independent boolean and does not enter the bitwise-deterministic summation path. The
checks run identically on CPU and CUDA ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)), exercised on the CPU fallback in CI
([07-testing-strategy.md](../spec/07-testing-strategy.md)).

## Alternatives Considered

**Per-embodiment latent spaces with a learned translation layer.** Each silo trains in its own latent
space; a learned map translates between spaces for aggregation. Considered because it removes the
up-front agreement burden and lets each embodiment optimize its own representation. Rejected: it
defeats the premise of a single shared backbone (no one model to average) and re-introduces a latent
gauge *per silo* (each space has its own $O(d)$ freedom,
[RFC-0002 §2](RFC-0002-gauge-and-aggregation.md#2-the-od-gauge-formally-preserved-verbatim)) plus a
new translation-network gauge on top — multiplying the problem Lensemble exists to solve.

**A fixed action vocabulary (shared discrete action token set).** Quantize every embodiment's actions
into one shared discrete vocabulary, giving a single shared action head. Considered for making actions
a fixed-vocabulary problem like tokens. Rejected: embodiments have continuous, dimensionally different
action spaces — a quadruped's joint-velocity vector is not commensurable with a 7-DoF arm's
end-effector pose — so a shared vocabulary either loses information through coarse quantization or
balloons to a lowest-common-denominator union. The contract instead shares the conditioning *output*
dimension `cond_dim` and lets each head own the mapping from its native action space.

**A single shared action head across all embodiments.** One head, federated like the encoder and
predictor. Considered for uniformity. Rejected: action spaces differ in `dim`, `kind`, and `units`, so
a shared head's input layer is undefined across embodiments; forcing it requires the rejected union.
Local heads (`INV-ACTIONHEAD-LOCAL`) are correct and cheaper — they cross no boundary, adding nothing
to communication or the proof surface ([RFC-0006 §2](RFC-0006-verifiable-contribution.md#2-the-provable-surface-and-what-stays-cheap)).

**Validate latents only at the federation boundary, not at encoder output.** Run conformance once,
where deltas cross, to save per-step cost in an inner loop that runs the encoder millions of times.
Rejected as the default: a mis-shaped or non-finite latent caught only at the boundary has already
wasted an inner horizon $H$ of compute and corrupted the pseudo-gradient. The check is $O(Nd)$
(finiteness) and $O(1)$ (shape), negligible against a ViT-L forward; the policy checks at encoder
output, predictor input, and the boundary. A config flag may downgrade the inner-loop check to sampled
(every $k$ steps), but the boundary check is mandatory and unconditional.

## Drawbacks

- **A rigid contract constrains new embodiments.** Any embodiment that does not fit fixed $(N, d)$ and
  a `cond_dim`-output head cannot join without a contract change. This is intentional rigidity (it is
  what makes federation well-posed), but it raises the bar for onboarding a genuinely novel modality.
- **Version bumps ripple.** A `wmcp_version` change forces every participant to upgrade in lockstep
  before they can rejoin (the join gate is exact-equality in v0.1, §6). Additive-optional extension
  (§6) limits the blast radius, but a major bump is a coordinated event across all sovereign nodes.
- **The contract cannot type-check the frame.** A latent in a *rotated* gauge frame still satisfies
  every clause of `check_latent_state` (same shape, dtype, finiteness, version). The contract pins the
  declared frame; actual frame agreement is only ever *measured*, by the gauge diagnostic
  ([RFC-0002 §9](RFC-0002-gauge-and-aggregation.md#9-the-frame-drift-diagnostic-the-headline-measurement)).
  A reader must not mistake conformance for frame alignment. RISK: a participant could conform while
  silently drifting; resolution plan: the per-round frame-drift diagnostic
  ([RFC-0015](RFC-0015-observability-diagnostics.md)) and the Procrustes backstop
  ([RFC-0002 §5](RFC-0002-gauge-and-aggregation.md#5-layer-3--procrustes-re-alignment-at-aggregation-backstop))
  catch and correct drift that the contract structurally cannot see.

## Migration / Rollout

- **v0.1 (Stage A).** `lensemble.contracts` ships with `LatentState`, `ActionSpec`, the conformance
  checks, the `ActionHead` ABC, and `WMCP_VERSION = "wmcp-1.0.0"`. Stage-A single-site training
  ([RFC-0001 §7(b)](RFC-0001-architecture.md)) exercises the contract end-to-end with a single
  embodiment, proving the encoder emits conforming latents and the predictor consumes them before any
  federation exists.
- **v0.2 (Stage B).** Simulated federation exercises the multi-embodiment seam: at least two
  `ActionSpec`s (a quadruped and a 7-DoF arm, the test fixture of the Testing Strategy) conform under
  one `wmcp_version`, and the join gate refuses a deliberately mismatched version.
- **Versioning.** `wmcp_version` follows SemVer ([conventions §10](../spec/conventions.md#10-versioning-and-schema-policy)). Pre-1.0 the contract may gain
  additive-optional fields with a minor bump and one minor's deprecation for any field removal;
  at 1.0 the contract surface (the dataclasses and the check signatures of §2–§5) freezes under the
  public-API policy ([conventions §10](../spec/conventions.md#10-versioning-and-schema-policy), [09-release-and-versioning.md](../spec/09-release-and-versioning.md)). Extension is additive-optional
  (§6); a major bump is a coordinated federation-wide upgrade.

## Testing Strategy

The full pyramid is at [07-testing-strategy.md](../spec/07-testing-strategy.md); the contract-owned tests:

- **Latent conformance, valid and invalid.** Construct a conforming `(N, d)` and `(B, N, d)`
  `LatentState` and assert `check_latent_state` returns. Then assert it raises `ContractViolation`
  with the correct `.code` for each invalid case: wrong rank, `(num_tokens, dim)` mismatch, an integer
  dtype, a `NaN`-containing tensor, and a wrong `wmcp_version`. Verify each `.remediation` is non-empty
  and names the expected value.
- **`ActionSpec` validation.** Property tests (hypothesis) over generated specs: a valid continuous
  spec and a valid discrete spec validate; `dim <= 0`, mismatched `len(units)`, `low[i] >= high[i]`,
  `num_classes < 2`, a continuous spec carrying `num_classes`, and a bad `embodiment_id` each raise
  `ContractViolation` (`WMCP_ACTIONSPEC_INVALID`).
- **Action-head output shape.** A reference `ActionHead` built from a `(dim=7)` continuous spec and a
  `(dim=12)` continuous spec both produce `encode(action)` outputs of shape `(B, cond_dim)` for the
  same `cond_dim` — the multi-embodiment seam (§5).
- **Action-head locality (`INV-ACTIONHEAD-LOCAL`).** Assert that a constructed federation
  broadcast/aggregation payload contains no action-head parameter keys and that a written
  `ModelArtifact` param-group manifest lists only encoder/predictor groups; a deliberately injected
  action-head key into a shared payload raises `ContractViolation` (`WMCP_ACTIONHEAD_LEAK`). Cross-checked
  from [RFC-0001 §4](RFC-0001-architecture.md) and [RFC-0010](RFC-0010-artifact-checkpoint-format.md).
- **Version gating.** Assert exact-equality join gate: a participant with a mismatched `WMCP_VERSION`
  is refused at the simulated handshake ([RFC-0013](RFC-0013-coordinator-runtime.md)); a matching one
  is admitted.
- **Two-embodiment fixture.** A shared CPU fixture providing a quadruped `ActionSpec` and a 7-DoF arm
  `ActionSpec` that both conform under one `wmcp_version`, reused by federation and model tests so the
  heterogeneous case is exercised wherever a single embodiment would otherwise be assumed. Runs on the
  CPU fallback ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)); no large downloads ([07-testing-strategy.md](../spec/07-testing-strategy.md)).

## Open Questions

OPEN QUESTION: **Contract-extension policy.** Exactly which fields may be added additive-optional under
a minor `wmcp_version` bump, and whether the join gate should relax from exact-equality to
"compatible-minor" negotiation once more than one minor exists. Owner @AbdelStark; resolution path:
a follow-up minor RFC superseding §6 once a concrete second contract version is needed (no earlier than
Stage C, when real heterogeneous nodes appear).

OPEN QUESTION: **May `N` (the latent-token count) vary across embodiments?** v0.1 fixes $N$
federation-wide for shape compatibility of the shared backbone. A modality with a different native
token count would need either a fixed adapter to the common $N$ or a contract that admits a
per-embodiment $N$ with a pooling/padding convention the predictor tolerates. Owner @AbdelStark;
resolution path: Stage B non-IID/scale experiments ([RFC-0005 §7](RFC-0005-evaluation.md#7-non-iid-severity--scale-sweeps)) inform
whether a single $N$ holds across the target embodiment set; if not, a contract extension (the first
Open Question) addresses it.

## References

- [RFC-0001 — Architecture & System Overview](RFC-0001-architecture.md): the model, the federation map
  (the disposition of $f_\theta$, $g_\phi$, $h_\psi^{(c)}$), the trust boundaries, the module DAG.
- [RFC-0002 — The Latent Gauge & Frame-Anchored Aggregation](RFC-0002-gauge-and-aggregation.md): the
  $O(d)$ gauge the contract's frame semantics defer to; the frame anchor and the drift diagnostic.
- [RFC-0003 — Federated Training Protocol](RFC-0003-federated-protocol.md): the pseudo-gradient
  $\Delta_c$ the contract makes comparable; `INV-AGG-DETERMINISM`.
- [RFC-0004 — Data, Sovereignty & Provenance](RFC-0004-data-provenance.md): §6, the WMCP pointer,
  "conformance is a precondition for joining", and the declared `ActionSpec` metadata.
- [RFC-0008 — Model, Objective & Numerical Contracts](RFC-0008-model-objective-numerics.md): the
  encoder/predictor that produce/consume `LatentState`, the reference `build_action_head`, the numerics.
- [RFC-0009 — Configuration, Run Manifest & Reproducibility](RFC-0009-configuration-reproducibility.md):
  `cond_dim` and `wmcp_version` recorded in the `RunManifest`.
- [RFC-0010 — Checkpoint & Artifact Format](RFC-0010-artifact-checkpoint-format.md): shared artifacts
  carry only encoder/predictor param groups (`INV-ACTIONHEAD-LOCAL`).
- [RFC-0013 — Coordinator & Participant Runtime](RFC-0013-coordinator-runtime.md): the join handshake
  that gates on `wmcp_version`.
- [RFC-0015 — Observability, Diagnostics & Telemetry](RFC-0015-observability-diagnostics.md): the
  frame-drift diagnostic that measures the frame the contract cannot type-check.
- Spec: ([02-public-api.md](../spec/02-public-api.md)) (the public contract surface), ([03-data-model.md](../spec/03-data-model.md))
  (`LatentState`/`ActionSpec` schemas), ([04-error-model.md](../spec/04-error-model.md)) (`ContractViolation`),
  ([05-observability.md](../spec/05-observability.md)) (safe-to-log fields), ([07-testing-strategy.md](../spec/07-testing-strategy.md)).
- WMCP (WM-RFC-0001) — the originating latent/action contract this RFC realizes for Lensemble.
- V-JEPA 2 (Assran et al., 2025) — the warm-start that fixes the $t{=}0$ frame the contract's latent
  semantics reference.
