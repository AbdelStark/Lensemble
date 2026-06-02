# RFC-0008 — Model, Objective & Numerical Contracts

| | |
|---|---|
| **RFC** | 0008 |
| **Title** | Model, Objective & Numerical Contracts |
| **Slug** | model-objective-numerics |
| **Status** | Accepted |
| **Track** | Standards |
| **Authors** | @AbdelStark |
| **Created** | 2026-06-02 |
| **Target milestone** | v0.1 (Stage A) |
| **Area** | model |
| **Requires** | RFC-0001 (architecture), RFC-0007 (WMCP latent contract) |
| **Informs** | RFC-0002 (gauge & aggregation), RFC-0005 (evaluation), RFC-0009 (config & reproducibility), RFC-0010 (artifacts), RFC-0015 (observability) |

## Summary

This RFC specifies the Lensemble model and its training objective with explicit numerical contracts.
It owns the `lensemble.model` subsystem (`encoder.py`, `predictor.py`, `action_head.py`,
`objective.py`, `sigreg.py`) and the public constructors `build_encoder`, `build_predictor`,
`build_action_head`, and the `Objective` class ([conventions §5](../spec/conventions.md#5-public-api-surface)). The model is an action-conditioned JEPA used
as a latent world model: an encoder $f_\theta$ (a video Vision Transformer warm-started from released
V-JEPA 2 weights, co-trained — Fork B), a compact action-conditioned latent predictor $g_\phi$ (the
LeWM `ARPredictor` shape), and a per-embodiment action head $h_\psi^{(c)}$ (contract owned by
[RFC-0007](RFC-0007-wmcp-latent-contract.md)). The training loss is the three-term objective fixed in
[conventions §2](../spec/conventions.md#2-mathematical-notation): a next-embedding prediction loss, the SIGReg anti-collapse regularizer, and the frame-anchor
loss that manufactures the missing latent frame ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)).

This RFC fixes: the warm-start loading path and the round-0 snapshot $f_{\text{ref}}$ that closes the
gauge (`INV-WARMSTART-T0`); the stop-gradient on the prediction target $\text{sg}[f_\theta(x_{t+1})]$;
the SIGReg algorithm (random-projection sketch $A$, the Epps–Pulley characteristic-function statistic,
Cramér–Wold reduction, reference defaults) and the **reduce-within-trust-domain** rule for projection
statistics (`INV-RESIDENCY`); and the numerical contract (bf16 forward, fp32 accumulation,
deterministic-mode flag, CUDA-primary with CPU fallback). Every encoder output conforms to the WMCP
`LatentState` (`INV-WMCP`); the per-embodiment action head is never aggregated
(`INV-ACTIONHEAD-LOCAL`). The objective returns per-term scalars for the metric stream of
[RFC-0015](RFC-0015-observability-diagnostics.md).

## Motivation

Two corpus claims rest on this module being correct and disciplined. First, the scientific core
([RFC-0002](RFC-0002-gauge-and-aggregation.md)) is *about* this objective: SIGReg's $O(d)$ invariance
is what opens the latent gauge, and the anchor term is the only term in $\mathcal{L}$ that is not
$O(d)$-invariant by construction. A gauge fix can only be specified against a precisely defined
objective, so the objective must be pinned here rather than described loosely. Second, the federation
protocol ([RFC-0003 §3](RFC-0003-federated-protocol.md#3-the-pseudogradient-contract)) averages
weight-space deltas of exactly these parameters; the choice of objective is what makes that average
well-posed.

The objective choice is itself load-bearing for federation. SIGReg (LeJEPA / LeWM) matches each
projected embedding marginal to a standard Gaussian via a characteristic-function statistic; it removes
the EMA target, stop-gradient teacher, and teacher–student bookkeeping that other JEPA-family
objectives carry. That removal is not a convenience — momentum-encoder state is *additional mutable
state per participant that would have to be reconciled across the federation boundary*, and there is no
clean averaging rule for a momentum buffer that diverged across silos for $H$ steps. SIGReg's
statelessness is precisely the property that makes the objective federation-friendly. The cost is that
SIGReg is the source of the gauge, which is why this RFC and [RFC-0002](RFC-0002-gauge-and-aggregation.md)
are co-designed.

Numerical discipline is a hard requirement, not a style preference. The aggregation/outer-step path
must be bitwise-reproducible (`INV-AGG-DETERMINISM`, [conventions §9](../spec/conventions.md#9-determinism-dtype-device)) so that the Phase-2 proof of correct
aggregation is cheap and so that the public alignment recomputation
([RFC-0006 §4](RFC-0006-verifiable-contribution.md#4-public-recomputation-phase-1-free-in-scope-now)) matches the coordinator bit-for-bit. The forward
pass need not be bitwise-deterministic, but the statistics it feeds into the objective, and the
warm-start it loads, must be specified down to dtype and reduction order, or two participants will
silently compute different objectives and the gauge analysis will not hold.

## Goals

- Specify the encoder $f_\theta$: a video ViT warm-started from released V-JEPA 2 weights, co-trained
  (Fork B), emitting a WMCP `LatentState` of shape $(N, d)$. Fix the warm-start loading path and the
  $f_{\text{ref}}$ snapshot taken at round 0 (`INV-WARMSTART-T0`, `INV-WMCP`).
- Specify the predictor $g_\phi$: a compact action-conditioned transformer predicting future latents
  autoregressively (the LeWM `ARPredictor` shape), with the stop-gradient on the target
  $\text{sg}[f_\theta(x_{t+1})]$ stated as a contract.
- Specify the `Objective` class: the three weighted terms ([conventions §2](../spec/conventions.md#2-mathematical-notation)), returning per-term scalars for
  logging ([RFC-0015](RFC-0015-observability-diagnostics.md)).
- Specify SIGReg in implementation detail: the shared sketch $A$
  ([RFC-0002 §3](RFC-0002-gauge-and-aggregation.md#3-layer-1--shared-sketch-matrix-objective-consistency-not-the-gauge-fix)),
  the Epps–Pulley statistic, Cramér–Wold reduction, reference defaults, and the reduce-within-trust-domain
  rule for projection statistics.
- Specify the numerical contract ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)): bf16 forward / fp32 accumulation; the deterministic-mode
  flag; CUDA-primary with a CPU fallback that CI exercises; normalization; where exact reductions are
  required.
- Reference (not duplicate) the WMCP `LatentState`/`ActionSpec`/action-head contract owned by
  [RFC-0007](RFC-0007-wmcp-latent-contract.md), and enforce `INV-WMCP` and `INV-ACTIONHEAD-LOCAL` at
  this module's boundaries.

## Non-Goals

- This RFC does not specify the latent gauge fix, the anchor variants, the Procrustes backstop, or the
  frame-drift diagnostic; those are owned by [RFC-0002](RFC-0002-gauge-and-aggregation.md). This RFC
  defines $\mathcal{L}_{\text{anchor}}$ only as a term the `Objective` evaluates by delegating to
  `lensemble.gauge`.
- It does not define the WMCP `LatentState`/`ActionSpec` schemas or the action-head conformance
  contract; those are [RFC-0007](RFC-0007-wmcp-latent-contract.md). This RFC consumes them.
- It does not define the federation round, the DiLoCo schedule, the outer optimizer, secure
  aggregation, or DP; those are [RFC-0003](RFC-0003-federated-protocol.md),
  [RFC-0011](RFC-0011-secure-aggregation.md), and [RFC-0012](RFC-0012-differential-privacy.md). This
  RFC owns only the inner-step loss and forward/backward numerics.
- It does not define the latent-MPC planner or the eval metrics; those are
  [RFC-0005](RFC-0005-evaluation.md). Stage A uses them to validate the objective centrally.
- It does not fix production values of $\lambda_{\text{sig}}$, the SIGReg sketch dimension, or the
  Epps–Pulley knot count at video scale; those are Stage A/B empirical tasks (Open Questions).

## Proposed Design

### 1. Module map and construction surface

`lensemble.model` is an L4 module ([conventions §1](../spec/conventions.md#1-repository-and-package-layout), [RFC-0001 §3](RFC-0001-architecture.md#3-dependency-layering-no-cycles)):
it depends on `contracts` (WMCP), `artifacts` (warm-start load / `f_ref` snapshot), `config`,
`observability`, and `errors`. It depends on `gauge` only at the `Objective` boundary, which receives a
pre-constructed anchor callable to avoid a cycle (`gauge` is also L4; the anchor object is injected by
the caller in `federation`/`train_local`, not imported by `model`).

| File | Responsibility | Public symbols |
|---|---|---|
| `encoder.py` | Video ViT, warm-start load, `f_ref` snapshot | `build_encoder`, `Encoder` |
| `predictor.py` | Action-conditioned autoregressive latent predictor | `build_predictor`, `Predictor` |
| `action_head.py` | Per-embodiment action head construction (contract: RFC-0007) | `build_action_head` |
| `objective.py` | The three-term loss; per-term scalar outputs | `Objective`, `LossTerms` |
| `sigreg.py` | Sketch matrix, Epps–Pulley statistic, Cramér–Wold reduction | `build_sketch`, `sigreg_statistic` |

Public constructors take the relevant `LensembleConfig` sub-config ([conventions §5](../spec/conventions.md#5-public-api-surface),
[RFC-0009](RFC-0009-configuration-reproducibility.md)) and raise `ConfigError` on an invalid or
inconsistent model config (for example $d$ inconsistent with the WMCP-pinned dimension, or a sketch
dimension exceeding $d$).

### 2. Encoder `f_\theta`

```python
from pathlib import Path
import torch
from torch import Tensor
from lensemble.contracts import LatentState   # RFC-0007 (INV-WMCP)

class Encoder(torch.nn.Module):
    """Video ViT encoder f_theta. Warm-started from V-JEPA 2; co-trained (Fork B).

    __call__: clip -> LatentState of shape (N, d), conforming to wmcp_version.
    """
    wmcp_version: str
    d: int          # latent embedding dimension ([conventions §2](../spec/conventions.md#2-mathematical-notation))
    num_tokens: int # N, latent tokens emitted per clip ([conventions §2](../spec/conventions.md#2-mathematical-notation))

    def forward(self, clip: Tensor) -> LatentState: ...
        # clip: (B, T, C, Hpx, Wpx); returns LatentState carrying (B, N, d) bf16 forward.

def build_encoder(cfg) -> Encoder:
    """Construct an Encoder per the model config.

    Pre:  cfg.model.d == LatentState.d for the pinned wmcp_version (else ContractViolation).
    Post: an Encoder whose forward emits a conformant LatentState (INV-WMCP).
    Raises: ConfigError (inconsistent dims/patching), ContractViolation (WMCP mismatch).
    """

def load_warmstart(encoder: Encoder, checkpoint: Path, *, expected_hash: str) -> None:
    """Load pinned V-JEPA 2 warm-start weights into `encoder`.

    Pre:  the checkpoint content hash equals `expected_hash` (the pinned warm-start hash;
          RFC-0010 INV-CHECKPOINT-HASH). Mismatch => CheckpointIntegrityError.
    Post: encoder weights are byte-identical across all participants (INV-WARMSTART-T0).
    """

def snapshot_reference(encoder: Encoder) -> "ReferenceEncoder":
    """Freeze the round-0 encoder as f_ref (used for anchor targets and E_ref).

    Post: a frozen, eval-mode encoder; its content hash equals the pinned warm-start hash at
          round 0 (INV-WARMSTART-T0). f_ref is never trained and never broadcast.
    """
```

The encoder is the V-JEPA 2 video ViT shape ([RFC-0001 §1](RFC-0001-architecture.md#1-model)). It is
**co-trained** under SIGReg — this is Fork B, the lead contribution ([conventions §0](../spec/conventions.md#0-project-identity)); the frozen-encoder
variant (Fork A) is the documented degrade
([RFC-0002 §7 Fork A fallback](RFC-0002-gauge-and-aggregation.md#fork-a-fallback)). Warm-start loading
is the only path that establishes `INV-WARMSTART-T0`: every participant calls `load_warmstart` with the
same `expected_hash`, so round-0 encoder weights are hash-identical across participants and the gauge is
closed at $t{=}0$ ([RFC-0002 §4(a)](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)).
The coordinator verifies this at round-0 admission via the reported `Checkpoint.content_hash`
([RFC-0010](RFC-0010-artifact-checkpoint-format.md), [RFC-0003 §1](RFC-0003-federated-protocol.md#1-roles));
a mismatch raises `GaugeError` (`INV-WARMSTART-T0`). `f_{\text{ref}}` is the round-0 snapshot used to
produce $E_{\text{ref}} = f_{\text{ref}}(\mathcal{P})$ and the Variant-A landmark targets
$t_i = f_{\text{ref}}(p_i)$ (`INV-PROBE-PIN`); targets derive only from $f_{\text{ref}}$, never from a
later checkpoint, which is enforced in `lensemble.gauge.anchor`
([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)).

The encoder output is a WMCP `LatentState` of shape $(N, d)$ (`INV-WMCP`,
[RFC-0007](RFC-0007-wmcp-latent-contract.md)). A nonconforming shape/dtype/semantics raises
`ContractViolation` ([conventions §6](../spec/conventions.md#6-error-taxonomy)) at the contract boundary, not inside the forward.

### 3. Predictor `g_\phi`

```python
class Predictor(torch.nn.Module):
    """Compact action-conditioned latent predictor (LeWM ARPredictor shape).

    Predicts the next latent given the current latent and an action conditioning embedding;
    rolled out autoregressively for multi-step horizons. Consumes and emits LatentState.
    """
    d: int

    def forward(self, latent: LatentState, action_embedding: Tensor) -> LatentState: ...
        # latent: (B, N, d); action_embedding: (B, cond_dim); returns predicted (B, N, d).

def build_predictor(cfg) -> Predictor:
    """Construct g_phi per the model config. Raises ConfigError on inconsistent dims."""
```

The predictor is conditioned on an action embedding produced by the per-embodiment action head
$h_\psi^{(c)}$ ([RFC-0007](RFC-0007-wmcp-latent-contract.md)), which maps that embodiment's
`ActionSpec`-typed action into the shared latent-conditioning space. The conditioning embedding lives
in the shared space so that $g_\phi$ is one federated model across embodiments
([RFC-0001 §4](RFC-0001-architecture.md#4-federation-map)); the head that produces it is local
(`INV-ACTIONHEAD-LOCAL`).

The prediction loss compares the predicted next latent to the encoder's embedding of the actual next
observation, **stop-gradient on the target** ([conventions §2](../spec/conventions.md#2-mathematical-notation)):

$$\mathcal{L}_{\text{pred}} = \mathbb{E}\,\lVert g_\phi(f_\theta(x_t),a_t) - \text{sg}[f_\theta(x_{t+1})]\rVert^2 .$$

The stop-gradient is a contract: gradients flow into $g_\phi$ and into $f_\theta$ *through the input
branch* $f_\theta(x_t)$, but not through the target branch $f_\theta(x_{t+1})$. With SIGReg preventing
collapse, no EMA/teacher target is needed (the property that makes the objective federation-friendly;
see §6 and Alternatives Considered). The implementation applies `Tensor.detach()` to the target
embedding before the residual; a gradient finite-difference test asserts the target branch carries zero
gradient (Testing Strategy).

### 4. Action head `h_\psi^{(c)}`

Per-embodiment action heads $h_\psi^{(c)}$ map an `ActionSpec`-typed action space into the shared
latent-conditioning space. Their full interface — the `ActionSpec` descriptor, the conformance check,
and the required method signatures — is owned by [RFC-0007](RFC-0007-wmcp-latent-contract.md). This RFC
fixes only:

```python
def build_action_head(action_spec, cfg) -> "ActionHead":
    """Construct a per-embodiment action head h_psi^(c) for a validated ActionSpec.

    Pre:  action_spec validated against the WMCP contract (INV-WMCP); else ContractViolation.
    Post: a head whose output is a (B, cond_dim) conditioning embedding in the shared space.
    Local-only: the constructed head is never serialized into a broadcast or aggregation
                payload (INV-ACTIONHEAD-LOCAL), enforced in lensemble.federation.
    """
```

`build_action_head` requires a validated `ActionSpec`; constructing a head from an unvalidated spec is a
`ContractViolation` (`INV-WMCP`). The resulting parameters $\psi$ are excluded from every pseudo-gradient
$\Delta_c$ by construction (`INV-ACTIONHEAD-LOCAL`, enforced in
[RFC-0003 §3](RFC-0003-federated-protocol.md#3-the-pseudogradient-contract) /
[RFC-0001 §4](RFC-0001-architecture.md#4-federation-map)); this module exposes the partition so the
federation layer can assert it.

### 5. The `Objective`

The objective is the three weighted terms, stated exactly as in [conventions §2](../spec/conventions.md#2-mathematical-notation) (and verbatim in
[RFC-0002 §1](RFC-0002-gauge-and-aggregation.md#1-background-the-objective-verbatim-from-rfc-0008)):

$$\mathcal{L} = \lambda_{\text{pred}}\,\mathbb{E}\lVert g_\phi(f_\theta(x_t),a_t) - \text{sg}[f_\theta(x_{t+1})]\rVert^2 + \lambda_{\text{sig}}\,\mathrm{SIGReg}_A(f_\theta(x)) + \lambda_{\text{anc}}\,\mathcal{L}_{\text{anchor}}(f_\theta;\mathcal{P},\{t_i\}).$$

The gauge transform $f_\theta \mapsto Qf_\theta,\ g_\phi \mapsto Qg_\phi Q^\top$ leaves the first two
terms invariant and breaks only the third ([conventions §2](../spec/conventions.md#2-mathematical-notation),
[RFC-0002 §2](RFC-0002-gauge-and-aggregation.md#2-the-od-gauge-formally-preserved-verbatim)).

```python
from dataclasses import dataclass
from typing import Protocol
import torch
from torch import Tensor

@dataclass(frozen=True)
class LossTerms:
    """Per-term scalars for logging (RFC-0015 metric names in parentheses)."""
    pred: Tensor     # fp32 0-dim   (loss/pred)
    sigreg: Tensor   # fp32 0-dim   (loss/sigreg)
    anchor: Tensor   # fp32 0-dim   (loss/anchor)
    total: Tensor    # fp32 0-dim, the weighted sum that .backward() is called on

class AnchorTerm(Protocol):
    """Injected by the caller (gauge.FrameAnchor.loss); avoids a model->gauge import cycle."""
    def __call__(self, encoder: "Encoder") -> Tensor: ...   # unweighted L_anchor, fp32 0-dim

class Objective:
    """The three-term SIGReg-JEPA + frame-anchor loss.

    Constructed per round from the broadcast sketch seed s_t and the injected anchor term.
    `lambda_anc == 0.0` yields the bare LeJEPA objective (used by the gauge-invariance test,
    RFC-0002 Testing Strategy, and by Fork A where the encoder is frozen).
    """
    def __init__(
        self,
        *,
        lambda_pred: float,
        lambda_sig: float,
        lambda_anc: float,
        sketch_seed: int,          # s_t; INV-SKETCH-CONSISTENCY (RFC-0002 §3)
        sketch_dim: int = 64,
        ep_knots: int = 17,        # Epps-Pulley integration knots (§6)
        anchor: AnchorTerm | None = None,
    ) -> None: ...

    def __call__(
        self,
        encoder: "Encoder",
        predictor: "Predictor",
        window,                    # Window of Transitions (o_t, a_t, o_{t+1}); [conventions §8](../spec/conventions.md#8-core-data-types)
        action_embedding: Tensor,
    ) -> LossTerms:
        """Compute the three terms and the weighted total.

        Post: every LossTerms field is an fp32 0-dim tensor; `total` requires grad and is the
              value .backward() is called on. SIGReg uses the sketch A derived from sketch_seed
              (INV-SKETCH-CONSISTENCY). The target embedding f(x_{t+1}) is stop-gradiented.
        Raises: ContractViolation if encoder output is non-conformant (INV-WMCP);
                GaugeError('FrameDriftExceeded') if the landmark anchor is under-determined
                (k < d), surfaced from the injected anchor (RFC-0002 §4).
        """
```

The objective is constructed per round so that the sketch seed $s_t$ is fixed for the round
(`INV-SKETCH-CONSISTENCY`); the anchor term is injected (Variant A landmark or Variant B rotational,
[RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix))
rather than imported, keeping the module DAG acyclic
([RFC-0001 §3](RFC-0001-architecture.md#3-dependency-layering-no-cycles)). Returning per-term scalars
(not just the total) is required so the metric stream can emit `loss/pred`, `loss/sigreg`, `loss/anchor`,
and `grad_norm` per step ([RFC-0015](RFC-0015-observability-diagnostics.md)); the terms are scalar
metrics, not raw embeddings, so emitting them does not engage `INV-RESIDENCY`.

### 6. SIGReg algorithm

SIGReg pushes the embedding marginal toward $\mathcal{N}(0,I_d)$ by projecting embeddings onto random
1-D directions and matching each projected (univariate) marginal to a standard Gaussian via the
Epps–Pulley characteristic-function statistic. The Cramér–Wold argument justifies reducing the
$d$-dimensional Gaussianity test to a finite set of 1-D projections: a distribution is standard normal
iff all of its 1-D projections are.

**(a) The sketch matrix $A$.** Project $f_\theta(x) \in \mathbb{R}^{B \times N \times d}$ onto
`sketch_dim` random directions using the shared projection matrix
$A \in \mathbb{R}^{d \times \text{sketch\_dim}}$, built deterministically from the broadcast round
sketch seed $s_t$. The construction is the one fixed in
[RFC-0002 §3](RFC-0002-gauge-and-aggregation.md#3-layer-1--shared-sketch-matrix-objective-consistency-not-the-gauge-fix):

```python
def build_sketch(seed: int, d: int, sketch_dim: int = 64) -> Tensor:
    """Deterministic SIGReg projection matrix A of shape (d, sketch_dim).

    Pre:  seed == GlobalState.sketch_seed for the current round (INV-SKETCH-CONSISTENCY).
    Post: bitwise-identical across participants for identical (seed, d, sketch_dim); columns
          drawn i.i.d. from a fixed generator seeded by `seed` and L2-normalized to unit
          directions (so each projection is a coordinate of a random rotation, matching the
          O(d) symmetry argument of RFC-0002 §2).
    """
```

All participants in round $t$ use the identical $A$ derived from $s_t$ (`INV-SKETCH-CONSISTENCY`); a
seed mismatch is detected at participant ingress and raises `GaugeError`
([RFC-0002 §3](RFC-0002-gauge-and-aggregation.md#3-layer-1--shared-sketch-matrix-objective-consistency-not-the-gauge-fix)).
A shared sketch gives *objective consistency* — every participant minimizes the identical regularizer
— but does **not** close the gauge: matching an isotropic target along shared directions remains
$O(d)$-invariant ([RFC-0002 §3](RFC-0002-gauge-and-aggregation.md#3-layer-1--shared-sketch-matrix-objective-consistency-not-the-gauge-fix)).
The frame fix is the anchor term, not the sketch.

**(b) The Epps–Pulley statistic.** For each projected direction $j$, the projected sample
$u^{(j)} = (Az)_{\cdot,j}$ is standardized and compared to the standard-normal characteristic function
$\hat\varphi(t) = e^{-t^2/2}$ via a weighted $L_2$ distance between the empirical and target
characteristic functions, evaluated at `ep_knots` integration knots (reference default 17):

```python
def sigreg_statistic(embeddings: Tensor, sketch: Tensor, *, ep_knots: int = 17) -> Tensor:
    """Mean Epps-Pulley characteristic-function distance to N(0,1) over projected directions.

    Input:  embeddings (M, d) flattened over (B, N); sketch A (d, sketch_dim).
    Output: a 0-dim fp32 tensor; ~0 for a standard-normal sample, large for non-normal.
    Numerics: projection and standardization in fp32 (statistic accumulation, [conventions §9](../spec/conventions.md#9-determinism-dtype-device));
              the per-knot reduction order is fixed so the statistic is reproducible given
              identical inputs.
    """
```

**(c) Reference defaults.** Sketch dimension 64; Epps–Pulley integration knots ~17. These are the
LeJEPA-scale defaults; their values at video-world-model scale are an Open Question (Stage A/B). The
statistic is dimensionless (a distance), the loss weight $\lambda_{\text{sig}}$ scales it.

**(d) Reduce-within-trust-domain rule (`INV-RESIDENCY`).** SIGReg's projection statistics are computed
per-sample and then reduced (summed/averaged) across the batch. This batch reduction may be performed
**freely within a participant's inner-parallel group** — the FSDP/TP ranks of one sovereign node are
the same trust domain ([RFC-0001 §5](RFC-0001-architecture.md#5-training-topology-two-level)). The
reduction MUST NOT cross a participant boundary: neither raw projected values $Az$, the embeddings
$f_\theta(x)$, nor any private-batch statistic is serialized into an outbound message. Only the inner
gradient (folded into $\Delta_c$, then DP-clipped and masked) leaves the boundary. An attempt to emit a
projection statistic or embedding across a boundary raises `ResidencyViolation` ([conventions §6](../spec/conventions.md#6-error-taxonomy)), which is
fail-closed and never caught-and-ignored ([RFC-0015 redaction](RFC-0015-observability-diagnostics.md),
[RFC-0001 §6](RFC-0001-architecture.md#6-trust-boundaries)). Scalar SIGReg *loss values* may be logged
(they are aggregate statistics, not data); raw projected samples may not.

### 7. Numerical contract

| Concern | Contract | Enforced |
|---|---|---|
| Forward compute dtype | bf16 forward by default | `Encoder.forward`, `Predictor.forward` |
| Master weights | fp32 master weights | optimizer state (inner AdamW) |
| Loss / statistic accumulation | fp32 (or fp64 where configured) | `Objective.__call__`, `sigreg_statistic` |
| Determinism (inner) | best-effort, seed-pinned; full determinism gated by `torch.use_deterministic_algorithms` via a config flag | `train_local` / inner loop |
| Determinism (objective values) | given identical inputs, seed, and sketch, per-term scalars reproduce within fp32 tolerance | Testing Strategy §bf16/fp32 |
| Device | CUDA primary; CPU fallback for the small CI configs (tests pass on CPU) | `build_*` device placement |
| Normalization | embeddings standardized per projected direction before the Epps–Pulley statistic; SIGReg targets $\mathcal{N}(0,I)$, so no separate output LayerNorm is assumed on $f_\theta$'s final latent | `sigreg_statistic` |

The inner training forward/backward is **not** required to be bitwise-deterministic; inner determinism
is best-effort and seed-pinned ([conventions §9](../spec/conventions.md#9-determinism-dtype-device), [RFC-0001 §5](RFC-0001-architecture.md#5-training-topology-two-level)).
The bitwise-determinism requirement (`INV-AGG-DETERMINISM`) applies to the aggregation/outer-step path,
which is owned by [RFC-0003 §7](RFC-0003-federated-protocol.md#7-determinism-concurrency-error-propagation)
and [RFC-0002 §11](RFC-0002-gauge-and-aggregation.md#11-concurrency-determinism-dtype), not to this
module's inner loop. What this module guarantees is narrower and sufficient: the warm-start load is
byte-identical across participants (`INV-WARMSTART-T0`), the sketch is bitwise-identical given the seed
(`INV-SKETCH-CONSISTENCY`), and the per-term loss scalars are reproducible within fp32 tolerance given
identical inputs. Seeds are derived from one root seed and recorded in the `RunManifest`
([RFC-0009](RFC-0009-configuration-reproducibility.md), [conventions §9](../spec/conventions.md#9-determinism-dtype-device)); the per-round sketch seed is
$s_t = \mathrm{derive}(\text{root\_seed}, t)$.

### 8. Data flow (one inner step)

1. Draw a `Window` of `Transition`s $(o_t, a_t, o_{t+1})$ from local data ([conventions §8](../spec/conventions.md#8-core-data-types)); raw data stays in
   the boundary (`INV-RESIDENCY`).
2. Encode: $z_t = f_\theta(x_t)$, $z_{t+1} = f_\theta(x_{t+1})$ — each a `LatentState` $(B,N,d)$
   (`INV-WMCP`).
3. Condition: $a_{\text{emb}} = h_\psi^{(c)}(a_t)$ from the local action head.
4. Predict: $\hat z_{t+1} = g_\phi(z_t, a_{\text{emb}})$.
5. Prediction term: $\lVert \hat z_{t+1} - \text{sg}[z_{t+1}]\rVert^2$ (target detached, §3).
6. SIGReg term: $\mathrm{SIGReg}_A(z_t)$ via the shared sketch $A$ from $s_t$ (§6); batch reduction
   within the trust domain only (`INV-RESIDENCY`).
7. Anchor term: the injected `AnchorTerm(encoder)` evaluating $\mathcal{L}_{\text{anchor}}$ on the
   pinned probe $\mathcal{P}$ against $f_{\text{ref}}$ targets (`INV-PROBE-PIN`,
   [RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)).
8. Total $= \lambda_{\text{pred}}\cdot(5) + \lambda_{\text{sig}}\cdot(6) + \lambda_{\text{anc}}\cdot(7)$;
   `backward`; inner AdamW step. Emit `LossTerms` to the metric stream
   ([RFC-0015](RFC-0015-observability-diagnostics.md)).

After $H$ such steps the pseudo-gradient $\Delta_c$ is formed by the federation layer
([RFC-0003 §3](RFC-0003-federated-protocol.md#3-the-pseudogradient-contract)); action-head parameters
$\psi$ are excluded (`INV-ACTIONHEAD-LOCAL`).

### 9. Failure modes and system response

| Failure mode | Trigger | Detection | Error ([conventions §6](../spec/conventions.md#6-error-taxonomy)) | System response |
|---|---|---|---|---|
| Non-conformant latent | Encoder emits wrong shape/dtype/semantics | WMCP conformance check on `Encoder.forward` output | `ContractViolation` | Fail-closed; refuse to proceed (`INV-WMCP`); remediation "emit a LatentState of shape (N,d) for the pinned wmcp_version (RFC-0007)" |
| Warm-start hash mismatch | Loaded checkpoint hash ≠ pinned warm-start hash | `load_warmstart` hash compare | `CheckpointIntegrityError` (`ArtifactError`) | Refuse load ([RFC-0010](RFC-0010-artifact-checkpoint-format.md)); upstream this also fails round-0 admission as `GaugeError` (`INV-WARMSTART-T0`) |
| Action head from unvalidated spec | `build_action_head` called before `ActionSpec` validation | precondition check | `ContractViolation` | Fail-closed; remediation "validate the ActionSpec against the WMCP contract (RFC-0007) before constructing a head" |
| Sketch seed mismatch | Participant builds $A$ from the wrong seed | seed compare at ingress | `GaugeError` | Reject the round contribution (`INV-SKETCH-CONSISTENCY`); remediation "reconstruct A from the RoundOpen seed (RFC-0002 §3)" |
| Under-determined landmark anchor | $k < d$ generic landmarks (Variant A) | shape check in injected anchor | `FrameDriftExceeded` (`GaugeError`) | Fail-closed at construction; remediation "increase probe landmark count to k >= d (RFC-0004 §3, RFC-0002 §4)" |
| Embedding/projection about to cross a boundary | Reduce or serialize a private statistic across a boundary | residency guard at the egress path | `ResidencyViolation` | Fail-closed; never caught-and-ignored (`INV-RESIDENCY`); only $\Delta_c$ (DP-clipped, masked) leaves |
| Invalid model config | $d$/sketch-dim/patching inconsistent | `build_*` validation | `ConfigError` | Fail-closed at construction; remediation names the offending field |

Error-handling rules ([conventions §6](../spec/conventions.md#6-error-taxonomy)): never a bare `except`; `ResidencyViolation` is never swallowed; every
error carries `.code` (a `LensembleErrorCode`) and a `.remediation` string. WMCP, warm-start, sketch,
and residency failures are fail-closed at the boundary (config load, contract check, artifact load).

## Alternatives Considered

**SIGReg vs VICReg / Barlow-Twins style variance–covariance regularizers.** *What:* VICReg and
Barlow-Twins prevent collapse with explicit variance/covariance (or cross-correlation) penalties on the
embedding batch statistics. *Why considered:* both are stateless (no momentum encoder), which is the
same federation-friendly property SIGReg has, and both are well established. *Why SIGReg:* SIGReg's
characteristic-function Gaussianity test gives a single principled target distribution
($\mathcal{N}(0,I)$) via the Cramér–Wold reduction, and its random-projection structure is exactly the
sketch $A$ that the gauge analysis is built on
([RFC-0002 §3](RFC-0002-gauge-and-aggregation.md#3-layer-1--shared-sketch-matrix-objective-consistency-not-the-gauge-fix)).
SIGReg also has the precise $O(d)$-invariance that makes the gauge problem statable as a theorem rather
than an empirical drift. VICReg/Barlow are not rejected as wrong; they are not chosen because they do
not give the clean sketch-shared, isotropic-target structure the rest of the corpus depends on. The
covariance-decorrelation term of Barlow/VICReg is itself $O(d)$-covariant in a way that does not pin a
frame, so it would face the same gauge.

**EMA-target / teacher–student JEPA (V-JEPA / I-JEPA / BYOL-style).** *What:* a momentum (EMA) encoder
produces the prediction target; an asymmetric predictor and stop-gradient prevent collapse without an
explicit regularizer. *Why considered:* it is the dominant JEPA recipe and what V-JEPA 2 itself uses.
*Why rejected for Fork B federation:* the momentum encoder is *additional mutable per-participant state*
that crosses the federation boundary badly. After $H$ inner steps, each silo's EMA buffer has diverged;
there is no principled averaging rule for momentum buffers in mutually-rotated frames (the gauge applies
to the EMA encoder too), and reconciling them re-introduces exactly the frame problem on a second set of
weights. SIGReg removes the EMA target, the stop-gradient teacher, and the teacher–student machinery,
leaving only the stateless encoder/predictor pair plus a regularizer
([RFC-0002 §1](RFC-0002-gauge-and-aggregation.md#1-background-the-objective-verbatim-from-rfc-0008));
that statelessness is the property that makes the objective federation-friendly. (The stop-gradient on
the *target embedding* in §3 is retained; what is removed is the *separate momentum-encoder* that
produces it.)

**Frozen encoder = Fork A.** *What:* freeze the warm-started encoder, co-train only the predictor; the
frozen shared encoder is a shared frame, so the gauge dissolves and the anchor term is unnecessary
([RFC-0002 §7 Fork A fallback](RFC-0002-gauge-and-aggregation.md#fork-a-fallback)). *Why considered:* it
is the clean safe-degrade and removes the hardest part of the problem. *Why not the lead:* it sacrifices
the end-to-end (Fork B) novelty that is the project's core claim ([conventions §0](../spec/conventions.md#0-project-identity)). Fork A is supported and tested
by v1.0 ([conventions §12](../spec/conventions.md#12-milestones-and-stages)); under Fork A the `Objective` runs with `lambda_anc = 0.0` and the encoder in eval
mode (no encoder gradient), which is why the `Objective` signature admits `lambda_anc = 0.0` and a
`None` anchor.

**Per-step encoder target vs averaged/EMA target embedding.** *What:* whether the stop-gradient target
$\text{sg}[f_\theta(x_{t+1})]$ uses the current per-step encoder or a time-averaged encoder. *Why
considered:* an averaged target can stabilize the prediction loss. *Why deferred:* an averaged target
re-introduces a second set of encoder state to reconcile across the boundary (the EMA objection above),
so the default is the per-step current encoder; whether an averaged target helps at video scale without
re-opening a state-reconciliation problem is an Open Question (Stage A/B).

## Drawbacks

- **SIGReg is demonstrated only to ViT-H on images.** Its behavior co-training a video-world-model
  encoder at ViT-L (~300M, Stage A) and toward the 1.2B target (Stage E) is unproven. This is the
  central risk; Stage A (v0.1) de-risks the objective and the latent-MPC eval centrally before any
  federation ([RFC-0005](RFC-0005-evaluation.md), Migration / Rollout). RISK: if SIGReg fails to
  converge at ViT-L scale, the project falls back to Fork A
  ([RFC-0002 §7](RFC-0002-gauge-and-aggregation.md#fork-a-fallback)) and continues the sovereignty story
  without the end-to-end claim; resolution path: the Stage-A convergence gate.
- **Co-training the encoder is the hard regime that opens the gauge.** This is by design (it is the
  contribution), but it means this module cannot be validated in isolation from
  [RFC-0002](RFC-0002-gauge-and-aggregation.md); the gauge-invariance test (Testing Strategy) is the
  link.
- **Reference defaults are image-scale.** Sketch dimension 64 and ~17 Epps–Pulley knots come from
  LeJEPA at image scale; the SIGReg variance these control may be inadequate at video scale, where
  $N$ (tokens per clip) is much larger. Mitigation: treat them as swept hyperparameters in Stage A/B
  (Open Questions), not fixed constants.
- **The stop-gradient and reduce-within-trust-domain rules are easy to get subtly wrong.** A missing
  `detach` silently changes the objective; a batch reduction that leaks across a boundary silently
  violates `INV-RESIDENCY`. Mitigation: both are covered by explicit tests (gradient finite-difference
  for the stop-gradient, a residency-guard egress test, Testing Strategy).

## Migration / Rollout

The model and objective ship in **v0.1 (Stage A)** as the foundational scaffolding: warm-started
$f_\theta$, $g_\phi$, a single action head $h_\psi$, the three-term `Objective` (with Layer-1 shared
sketch active for objective consistency even single-site, and `lambda_anc` available), validated
centrally on pooled robot data and evaluated via latent MPC ([RFC-0005](RFC-0005-evaluation.md)). This
is the centralized upper bound and the `train_local(config) -> RunResult` path
([RFC-0001 §7(b)](RFC-0001-architecture.md#7-data-flow-lifecycles)); no outer loop, no boundaries, no
DP/secure-agg.

- **Stage A (v0.1):** validate the objective + MPC eval before federation. Layer-1 shared sketch is on;
  the anchor term is wired but exercised in earnest in Stage B. This stage gates everything: the SIGReg
  convergence result at ViT-L is the exit criterion for moving the encoder under federation.
- **Stage B (v0.2):** the same `Objective` is what the simulated federation minimizes per inner step;
  $\lambda_{\text{anc}}$ becomes a swept knob ([RFC-0002 §7](RFC-0002-gauge-and-aggregation.md#7-the-central-hyperparameter),
  [RFC-0005 §6](RFC-0005-evaluation.md)). No change to this module's contracts.
- **Scale (Stage E, post-v1.0):** repeat at increasing encoder size; the sketch dimension and knot count
  are re-tuned. Out of v1.0 scope ([conventions §12](../spec/conventions.md#12-milestones-and-stages)).

Pre-1.0 the model config schema may change with a manifest `schema_version` bump
([RFC-0009](RFC-0009-configuration-reproducibility.md), [conventions §10](../spec/conventions.md#10-versioning-and-schema-policy)); the public constructors `build_encoder`,
`build_predictor`, `build_action_head`, and `Objective` are frozen at 1.0 ([conventions §5](../spec/conventions.md#5-public-api-surface)).

## Testing Strategy

The full pyramid is at ([07-testing-strategy.md](../spec/07-testing-strategy.md)); this module owns the following, each runnable
on the CPU fallback with tiny synthetic fixtures (no large downloads, [conventions §9](../spec/conventions.md#9-determinism-dtype-device)):

- **SIGReg statistic correctness.** Assert `sigreg_statistic` is near zero on a known standard-normal
  sample and large on a known non-normal sample (for example a heavy-tailed or low-rank sample). This is
  the unit owned here; it is referenced by [RFC-0002 Testing Strategy](RFC-0002-gauge-and-aggregation.md#testing-strategy)
  because Layer 1 depends on it.
- **Gradient finite-difference check.** Verify the analytic gradient of each loss term against a
  finite-difference estimate on a tiny model, within fp32 tolerance. Includes the stop-gradient
  assertion: gradient through the target branch $f_\theta(x_{t+1})$ is zero (the `detach` contract, §3).
- **AC-predictor output shape.** Assert `Predictor.forward` emits a `LatentState` of shape $(B,N,d)$
  given a $(B,N,d)$ latent and a $(B,d_{\text{cond}})$ (code field: `cond_dim`) conditioning embedding, and that an
  autoregressive rollout preserves the shape over the horizon.
- **Warm-start load test.** Assert `load_warmstart` refuses a checkpoint whose hash differs from
  `expected_hash` (`CheckpointIntegrityError`) and that, on success, the encoder weights are
  byte-identical to the pinned reference, and `snapshot_reference` produces an `f_ref` whose content
  hash equals the pinned warm-start hash (`INV-WARMSTART-T0`).
- **bf16 / fp32 numerical-tolerance test.** Run a forward+loss in bf16 forward / fp32 accumulation and
  assert the per-term scalars match an fp32-only reference within a stated `atol/rtol`; assert
  statistic accumulation is in fp32.
- **Determinism of a forward+backward under the deterministic flag.** With
  `torch.use_deterministic_algorithms` enabled via the config flag and a fixed seed, assert two runs
  produce the same per-term `LossTerms` within the deterministic-mode tolerance (best-effort inner
  determinism, [conventions §9](../spec/conventions.md#9-determinism-dtype-device)). Note this is the *inner* loop; the bitwise outer-step determinism check belongs
  to [RFC-0003](RFC-0003-federated-protocol.md#7-determinism-concurrency-error-propagation).
- **Sketch consistency.** Assert `build_sketch` is bitwise-identical for identical $(\text{seed}, d,
  \text{sketch\_dim})$ across processes, and that a seed mismatch is the condition the federation layer
  rejects (`INV-SKETCH-CONSISTENCY`).
- **WMCP conformance and residency.** Assert a non-conformant encoder output raises `ContractViolation`
  (`INV-WMCP`), and that attempting to serialize an embedding or a projection statistic onto an
  outbound message raises `ResidencyViolation` and is never caught-and-ignored (`INV-RESIDENCY`;
  cross-referenced from [06-security.md](../spec/06-security.md) and [05-observability.md](../spec/05-observability.md)).

Numerical-tolerance policy: exact (bitwise) equality is **not** required for the inner forward/backward;
the SIGReg, gradient, and bf16/fp32 tests use `atol/rtol` appropriate to fp32 (the policy is set in
[07-testing-strategy.md](../spec/07-testing-strategy.md)). Bitwise equality is required only on the aggregation/outer-step path,
which this module does not own.

## Open Questions

OPEN QUESTION: The $\lambda_{\text{sig}}$ schedule — whether SIGReg's weight should be fixed or annealed
over training, and how it trades against the prediction term during encoder co-training. Owner
@AbdelStark. Resolution: Stage A characterization on pooled data, then re-checked under federation in
Stage B ([RFC-0005 §6](RFC-0005-evaluation.md)), milestone v0.1 → v0.2.

OPEN QUESTION: The SIGReg sketch dimension and Epps–Pulley knot count at video-world-model scale — the
image-scale defaults (sketch 64, ~17 knots) may under-control variance when $N$ (latent tokens per clip)
is large. Owner @AbdelStark. Resolution: the Stage-A/B hyperparameter sweep against collapse
(`gauge/effective_dim`, [RFC-0005 §4](RFC-0005-evaluation.md)) and MPC success, milestone v0.1 → v0.2.

OPEN QUESTION: Whether the stop-gradient target $\text{sg}[f_\theta(x_{t+1})]$ should use the current
per-step encoder or a time-averaged encoder. The per-step encoder is the default to avoid re-introducing
reconcilable state across the federation boundary; an averaged target may stabilize the prediction loss
at video scale. Owner @AbdelStark. Resolution: Stage A ablation, conditional on not re-opening a
state-reconciliation problem under federation (the EMA objection, Alternatives Considered), milestone
v0.1 → v0.2.

RISK: SIGReg co-training a video-WM encoder at ViT-L and toward 1.2B is unproven (demonstrated only to
ViT-H on images). Resolution plan: Stage A (v0.1) de-risks the objective + MPC eval centrally before any
federation; the convergence result is the Stage-A exit gate. If it fails at ViT-L scale, fall back to
Fork A ([RFC-0002 §7](RFC-0002-gauge-and-aggregation.md#fork-a-fallback)) and continue the sovereignty
story without the end-to-end claim.

## References

- Internal: [RFC-0001 — Architecture](RFC-0001-architecture.md) (§1 model responsibilities and types,
  §3 dependency layering, §4 federation map, §5 two-level topology, §6 trust boundaries);
  [RFC-0002 — The Latent Gauge & Frame-Anchored Aggregation](RFC-0002-gauge-and-aggregation.md) (§1 the
  objective verbatim, §2 the $O(d)$ invariance, §3 the shared sketch / `INV-SKETCH-CONSISTENCY`, §4 the
  frame anchor / `INV-WARMSTART-T0` / `INV-PROBE-PIN`, §7 Fork A fallback, §11 numerics);
  [RFC-0003 — Federated Training Protocol](RFC-0003-federated-protocol.md) (§3 the `PseudoGradient`
  contract, §7 outer-step determinism); [RFC-0005 — Evaluation](RFC-0005-evaluation.md) (the Stage-A
  central validation, latent-MPC eval, the ablation ladder, collapse metric);
  [RFC-0006 — Verifiable Contribution](RFC-0006-verifiable-contribution.md) (§4 the public-recomputation
  property the aggregation determinism supports); [RFC-0007 — WMCP Latent Contract](RFC-0007-wmcp-latent-contract.md)
  (the `LatentState`/`ActionSpec`/action-head contract this RFC consumes);
  [RFC-0009 — Configuration & Reproducibility](RFC-0009-configuration-reproducibility.md) (the model
  config groups, seeding, `RunManifest`); [RFC-0010 — Artifact & Checkpoint Format](RFC-0010-artifact-checkpoint-format.md)
  (warm-start hash, `CheckpointIntegrityError`); [RFC-0012 — Differential Privacy](RFC-0012-differential-privacy.md)
  (the DP that protects $\Delta_c$ after the inner loop); [RFC-0015 — Observability & Diagnostics](RFC-0015-observability-diagnostics.md)
  (the `loss/*` and `grad_norm` metric emission and the redaction guard). Spec:
  ([02-public-api.md](../spec/02-public-api.md)) (the model construction surface), ([03-data-model.md](../spec/03-data-model.md))
  (`LatentState`, `Window`, `Transition`), ([05-observability.md](../spec/05-observability.md)) (loss-term metrics, redaction),
  ([06-security.md](../spec/06-security.md)) (residency), ([07-testing-strategy.md](../spec/07-testing-strategy.md)) (ML-specific tests and
  tolerance policy).
- External ([conventions §11](../spec/conventions.md#11-external-dependencies)): LeJEPA / LeWorldModel (Balestriero & LeCun; Maes, Le Lidec et al., 2026) — the
  SIGReg objective (random-projection sketch + characteristic-function Gaussianity, Cramér–Wold /
  Epps–Pulley) and the `ARPredictor` shape; V-JEPA 2 (Assran et al., 2025) — the video-ViT encoder
  warm-start, the action-conditioned predictor recipe, and the $t{=}0$ frame anchor
  ($f_{\text{ref}}$); the Cramér–Wold theorem and the Epps–Pulley characteristic-function normality test
  (the 1-D-projection reduction and the goodness-of-fit statistic); torch (`>=2.4,<3`) — autograd, bf16
  forward / fp32 accumulation, deterministic algorithms; safetensors (`>=0.4`) — the warm-start weight
  serialization (no pickle); stable-worldmodel (pinned) — the latent-MPC eval that validates the
  objective in Stage A.
