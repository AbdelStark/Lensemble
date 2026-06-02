# RFC-0002 — The Latent Gauge & Frame-Anchored Aggregation

| | |
|---|---|
| **RFC** | 0002 |
| **Title** | The Latent Gauge & Frame-Anchored Aggregation |
| **Slug** | gauge-and-aggregation |
| **Status** | Accepted |
| **Track** | Standards |
| **Authors** | @AbdelStark |
| **Created** | 2026-06-02 |
| **Target milestone** | v0.2 (Stage B) |
| **Area** | gauge |
| **Requires** | RFC-0001 (architecture), RFC-0008 (model & objective), RFC-0004 (probe & provenance) |

> This RFC is the scientific core of Lensemble. It states the one problem that makes federated
> end-to-end JEPA genuinely hard, proves why the standard model-merging fix does not apply, and
> specifies the solution: a shared warm-start plus a light public-probe frame anchor. The same
> construction keeps the eventual proof-of-contribution circuit cheap (RFC-0006).

## Summary

The SIGReg-JEPA objective ([RFC-0008 §5](RFC-0008-model-objective-numerics.md#5-the-objective)) is invariant under a global orthogonal rotation
$Q\in O(d)$ of the $d$-dimensional latent space. Independently-updated participants therefore drift
into mutually-rotated coordinate frames, and naive weight-averaging (`FedAvg` / DiLoCo outer step)
combines weight matrices expressing features in incompatible bases — producing neither participant's
representation. This is the **latent gauge** problem: a JEPA-specific obstruction that LLM and
supervised federation do not face, because a fixed output basis (vocabulary, class labels) pins their
frame for free.

Lensemble closes the gauge with four additive layers: (1) a **shared sketch matrix** for objective
consistency; (2) a **shared warm-start plus a public-probe frame anchor** that pins the frame at
$t{=}0$ and holds it during fine-tuning — the gauge fix; (3) a **Procrustes re-alignment backstop**
at aggregation; (4) **function-space distillation** as a heterogeneity/instability fallback. The
layers are additive and constitute the ablation ladder of [RFC-0005 §6](RFC-0005-evaluation.md). The
anchored frame keeps the outer step a near-linear weighted average, which is the property that makes
aggregation cheap to prove in Phase 2 ([RFC-0006 §3](RFC-0006-verifiable-contribution.md)).

This RFC owns the `lensemble.gauge` subsystem (`anchor.py`, `procrustes.py`, `drift.py`) and the
public functions `frame_drift` and `procrustes_align` ([conventions §5](../spec/conventions.md#5-public-api-surface)). It enforces invariants
`INV-WARMSTART-T0`, `INV-SKETCH-CONSISTENCY`, and `INV-PROBE-PIN`, and raises `GaugeError`
(`FrameDriftExceeded`, `DegenerateProcrustes`) on the failure modes enumerated below.

## Motivation

Federated and averaged training of supervised models and LLMs is approximately well-posed because the
loss depends on a **fixed output basis** — class labels, or a softmax over a fixed vocabulary. That
basis breaks the symmetry of the representation and pins a shared coordinate frame, so client weight
matrices are comparable and `FedAvg` / DiLoCo outer-averaging behaves.

End-to-end JEPA has **no fixed output basis**. The prediction target is another embedding produced by
the same network, and the regularizer's target — an isotropic Gaussian — is itself basis-free.
Nothing pins the frame. Co-training the encoder (Fork B, the lead contribution per [conventions §0](../spec/conventions.md#0-project-identity)) is exactly
the regime that opens the gauge; freezing the encoder (Fork A) dissolves it but forfeits the
end-to-end novelty.

Without a gauge fix, the project's central claim — that federation closes the centralized–local gap
without moving data — is untestable, because the aggregation step that is supposed to deliver that
gain is meaningless. The frame-drift diagnostic of this RFC (defined below) is also a standalone
contribution: to our knowledge it is the first measurement of latent frame-drift under federated
self-supervision.

## Goals

- Define the $O(d)$ latent gauge formally and prove the SIGReg-JEPA objective is invariant under it.
- Distinguish the gauge from the neuron-permutation symmetry that ordinary model-merging handles, and
  show why permutation alignment alone is insufficient.
- Specify a layered gauge-control mechanism (Layers 1–4) with typed contracts for each layer.
- Provide two anchoring variants — landmark (A, the safe default) and rotational-drift penalty (B) —
  with their loss forms, the Procrustes closed form, and their failure handling.
- Specify the per-outer-round algorithm wiring the layers into the federation protocol of
  [RFC-0003](RFC-0003-federated-protocol.md).
- Define the frame-drift diagnostic precisely enough that the headline figure is reproducible from
  committed weights and the public probe alone (emission contract in
  [RFC-0015](RFC-0015-observability-diagnostics.md)).
- Enforce `INV-WARMSTART-T0`, `INV-SKETCH-CONSISTENCY`, `INV-PROBE-PIN` and name the errors that fire.

## Non-Goals

- The federation wire protocol, DiLoCo schedule, secure aggregation, and DP are owned by
  [RFC-0003](RFC-0003-federated-protocol.md), [RFC-0011](RFC-0011-secure-aggregation.md), and
  [RFC-0012](RFC-0012-differential-privacy.md); this RFC references them, it does not define them.
- The encoder/predictor/SIGReg implementation and numerical contracts are owned by
  [RFC-0008](RFC-0008-model-objective-numerics.md); this RFC consumes the `Objective` and the latent
  contract.
- The public probe construction, sizing, licensing, and content-hash pin are owned by
  [RFC-0004 §3](RFC-0004-data-provenance.md); this RFC consumes a pinned probe and its landmark targets.
- The proof system, STARK circuit, and TEE attestation are [RFC-0006](RFC-0006-verifiable-contribution.md)
  (Phase 2, deferred); this RFC only guarantees the public-recomputability property the proof layer relies on.
- Choosing the production value of $\lambda_{\text{anc}}$ and the drift threshold is a Stage-B empirical
  task (Open Questions), not a normative default of this RFC.

## Proposed Design

### 1. Background: the objective (verbatim from RFC-0008)

Per LeJEPA / LeWM, each participant minimizes the objective ([conventions §2](../spec/conventions.md#2-mathematical-notation); stated identically in
[RFC-0008 §5](RFC-0008-model-objective-numerics.md#5-the-objective)):

$$\mathcal{L} = \lambda_{\text{pred}}\,\underbrace{\mathbb{E}\,\lVert g_\phi(f_\theta(x_t),\,a_t) - \text{sg}[f_\theta(x_{t+1})]\rVert^2}_{\text{next-embedding prediction}} \;+\; \lambda_{\text{sig}}\,\mathrm{SIGReg}_A\big(f_\theta(x)\big) \;+\; \lambda_{\text{anc}}\,\mathcal{L}_{\text{anchor}}(f_\theta;\mathcal{P},\{t_i\}).$$

**SIGReg** pushes the embedding marginal toward $\mathcal{N}(0,I_d)$ by projecting embeddings onto
random 1-D directions (the sketch matrix $A$) and matching each univariate marginal to a standard
Gaussian via the Epps–Pulley characteristic-function statistic (Cramér–Wold). SIGReg is what removes
the EMA target, stop-gradient, and teacher–student machinery — which is exactly why it is
federation-friendly (no momentum-encoder state to reconcile across the boundary). It is also the
source of the gauge. The third term, $\mathcal{L}_{\text{anchor}}$, is the contribution of this RFC;
it is absent from the bare LeJEPA objective and is what manufactures the missing frame.

### 2. The $O(d)$ gauge, formally (preserved verbatim)

Let $Q \in O(d)$. Transform the encoder and conjugate the predictor:

$$f_\theta \;\mapsto\; Q f_\theta, \qquad g_\phi \;\mapsto\; Q\, g_\phi\, Q^{\top}.$$

Then:

- **Prediction loss is unchanged** — it depends only on inner products / Euclidean distances, which
  $Q$ preserves: $\lVert Q\hat{z} - Q z\rVert = \lVert \hat z - z\rVert$.
- **SIGReg is unchanged** — $\mathcal{N}(0,I)$ is invariant under $O(d)$; every projected marginal of
  $Qz$ is still standard normal.

So the **entire objective (the first two terms) is invariant under a global $O(d)$ rotation**.
Independently-updated participants therefore converge to frames related by arbitrary $Q_c$, and

$$\bar\theta=\tfrac{1}{C}\textstyle\sum_c \theta_c$$

averages weight matrices expressing features in mutually-rotated bases — producing neither
participant's representation. This is the irreducible, JEPA-specific obstruction. Neuron-permutation
symmetries exist in any network and are handled by the same alignment machinery; the $O(d)$ latent
rotation is the *extra* one that anchored models never see. The third term
$\mathcal{L}_{\text{anchor}}$ breaks this invariance by tying the frame to fixed absolute targets — it
is the only term in $\mathcal{L}$ that is *not* $O(d)$-invariant, by construction.

#### 2.1 Three failures that compound it

- **Non-IID marginals** — each silo's empirical embedding distribution differs, so SIGReg constrains a
  different marginal per participant.
- **Collapse re-entry** — averaging across frames can re-introduce the low-rank solutions SIGReg
  prevented locally (detected by the effective-dimension metric, `gauge/effective_dim`,
  [RFC-0005 §4](RFC-0005-evaluation.md)).
- **DiLoCo drift** — the longer the inner horizon $H$, the further frames rotate apart before the
  outer step. The interaction of $H$ with drift is an Open Question shared with
  [RFC-0003](RFC-0003-federated-protocol.md).

### 3. Layer 1 — shared sketch matrix (objective consistency, not the gauge fix)

All participants use the **same** random projection (sketch) matrix $A$ each round, derived
deterministically from the broadcast round sketch seed $s_t = \mathrm{derive}(\text{root\_seed}, t)$
([conventions §9](../spec/conventions.md#9-determinism-dtype-device)). This ensures everyone minimizes the *identical* regularizer rather than $C$ different
empirical objectives, and it reduces SIGReg's small-batch variance (which falls as batch grows; the
projection statistics may be reduced freely *within* a participant's inner-parallel group — same trust
domain — per RFC-0008).

> **`INV-SKETCH-CONSISTENCY`** ([conventions §7](../spec/conventions.md#7-named-invariants)): all participants in round $t$ use the identical projection
> matrix $A$ derived from the broadcast seed $s_t$. Enforced at participant ingress: a participant
> validates that the `GlobalState.sketch_seed` it received matches the seed pinned in the round's
> `RoundOpen` message before constructing $A$. A mismatch raises `GaugeError` with remediation
> "re-fetch RoundOpen and reconstruct A from the pinned seed".

**Important:** a shared sketch does **not** close the gauge. Matching an isotropic target along shared
directions is still rotation-invariant — rotating every embedding by a single $Q$ rotates the
projection directions' images identically and leaves each univariate marginal standard normal. Layer 1
fixes consistency; the frame needs Layer 2.

Contract:

```python
def build_sketch(seed: int, d: int, sketch_dim: int = 64) -> Tensor:
    """Deterministic SIGReg projection matrix A of shape (d, sketch_dim).

    Pre:  seed == GlobalState.sketch_seed for the current round (INV-SKETCH-CONSISTENCY).
    Post: bitwise-identical across participants for identical (seed, d, sketch_dim).
          Columns drawn i.i.d. from a fixed generator seeded by `seed`.
    """
```

### 4. Layer 2 — frame anchoring on a public probe (the gauge fix)

Two ingredients manufacture the anchor the objective otherwise lacks. The probe is **public** data
([RFC-0004 §3](RFC-0004-data-provenance.md)), so nothing private leaves a boundary; `INV-RESIDENCY` is
not at stake here.

**(a) Shared warm-start.** Initialize every participant from the same released V-JEPA 2 encoder
($f_{\text{ref}}$). At round 0 the gauge is **closed**: every participant's encoder is in the identical
frame, so it can only re-open *during* federated fine-tuning. This is most of the battle.

> **`INV-WARMSTART-T0`** ([conventions §7](../spec/conventions.md#7-named-invariants)): at round 0 every participant's encoder weights are hash-identical to
> the pinned warm-start. Enforced by the coordinator at round-0 admission: it compares each
> participant's reported encoder content hash (the `Checkpoint.content_hash` of
> [RFC-0010](RFC-0010-artifact-checkpoint-format.md)) against the pinned warm-start hash. A mismatch
> raises `GaugeError` with remediation "reload the pinned warm-start checkpoint before joining round 0".

**(b) A public probe with fixed targets.** Maintain a small fixed public probe set $\mathcal{P}$ with
points $p_i$ (RFC-0004 §3). Fix reference embeddings $E_{\text{ref}} = f_{\text{ref}}(\mathcal{P})$
from the round-0 encoder $f_{\text{ref}}$. Two anchoring variants:

- **Variant A — landmark anchoring (recommended to start; simple, no differentiable SVD).** Choose
  $k \ge d$ generic landmark points with fixed absolute targets $t_i = f_{\text{ref}}(p_i)$. Add
  $$\mathcal{L}_{\text{anchor}} = \tfrac{1}{k}\textstyle\sum_{i=1}^{k}\lVert f_\theta(p_i)-t_i\rVert^2 .$$
  Because $k\ge d$ generic absolute constraints admit only $Q=I$ as the satisfying orthogonal map, this
  **pins the frame** while leaving the representation of all other (probe and private) points free —
  *pin the frame, not the content*. (The objective above carries the $\lambda_{\text{anc}}$ weight;
  $\mathcal{L}_{\text{anchor}}$ here is the unweighted term.)

- **Variant B — rotational-drift penalty (principled; fuller frame control).** Decompose the probe
  drift into a *frame* part and a *content* part. Compute the optimal Procrustes rotation
  $Q^\star=\arg\min_{Q\in O(d)}\lVert f_\theta(\mathcal{P})Q-E_{\text{ref}}\rVert_F$ (closed form:
  $Q^\star=VU^\top$ from the SVD $E_{\text{ref}}^\top f_\theta(\mathcal{P})=U\Sigma V^\top$). Penalize
  **only the rotation**:
  $$\mathcal{L}_{\text{anchor}} = \lVert Q^\star - I\rVert_F^2 .$$
  Gradients flow through the (differentiable) SVD into $f_\theta$, pushing the frame toward the
  reference while the content drift (the post-alignment residual) stays unpenalized. Mind
  near-degenerate singular values; clamp/condition the SVD (see Failure Modes, `DegenerateProcrustes`).

Variant A is the safe default; Variant B is the cleaner statement of "pin frame, not content" when
fuller control is wanted, at the cost of a differentiable SVD on the backward path.

> **`INV-PROBE-PIN`** ([conventions §7](../spec/conventions.md#7-named-invariants)): the probe content hash equals the hash committed in `RoundOpen`;
> landmark targets derive only from $f_{\text{ref}}$ (the round-0 encoder), never from a later
> checkpoint. Enforced in `anchor.py` when targets are loaded: the loader recomputes the probe content
> hash and compares it to `GlobalState.probe_hash`; a mismatch raises `ProbeError`
> ([RFC-0004 §3](RFC-0004-data-provenance.md)). Using any $f_t$ with $t>0$ as the target source is a
> programming error caught by the type of the targets carried in the round state.

Anchor contract:

```python
from typing import Literal
import torch
from torch import Tensor

AnchorVariant = Literal["landmark", "rotational"]

class FrameAnchor:
    """Layer-2 frame anchor. Owns the reference targets and computes L_anchor.

    Constructed once per run from the pinned probe and f_ref; reused every round.
    """
    def __init__(
        self,
        probe: Tensor,            # (k, *clip_shape) public probe points P
        ref_embeddings: Tensor,   # E_ref = f_ref(P), shape (k, d); INV-PROBE-PIN
        variant: AnchorVariant = "landmark",
        *,
        probe_hash: str,          # must equal GlobalState.probe_hash
    ) -> None: ...

    def loss(self, encoder: "Encoder") -> Tensor:
        """Unweighted L_anchor(f_theta; P, t_i) as a 0-dim fp32 tensor.

        Variant 'landmark': mean squared error to absolute targets t_i (k >= d required).
        Variant 'rotational': || Q* - I ||_F^2 with Q* = procrustes_align(f(P), E_ref).
        Raises GaugeError('FrameDriftExceeded') if k < d for the landmark variant
        (under-determined frame). Raises GaugeError('DegenerateProcrustes') if the
        rotational variant SVD is ill-conditioned beyond the clamp tolerance.
        """
```

### 5. Layer 3 — Procrustes re-alignment at aggregation (backstop)

Immediately before each outer step, recompute the hard alignment $Q_c^\star$ on $\mathcal{P}$ for each
participant and fold it into the encoder's terminal linear map (and conjugate the predictor I/O,
$g_\phi \mapsto Q_c^\star g_\phi Q_c^{\star\top}$) before averaging. With Layer 2 active this should
rarely bind; it fires only when the measured drift exceeds a configured threshold (Open Question
below). **Verifiability bonus:** alignment is a deterministic function of *public-probe data +
committed weights*, so it is publicly recomputable — anyone can recompute and check it, and it needs
**no** ZK proof ([RFC-0006 §3](RFC-0006-verifiable-contribution.md)). This is realized by
`recompute_alignment` in [RFC-0006](RFC-0006-verifiable-contribution.md).

Procrustes closed form (the same map used by Variant B and by the diagnostic):

```python
def procrustes_align(source: Tensor, target: Tensor) -> tuple[Tensor, float]:
    """Optimal orthogonal alignment of `source` onto `target` ([conventions §5](../spec/conventions.md#5-public-api-surface)).

    Given source S, target T of shape (k, d), solve
        Q* = argmin_{Q in O(d)} || S Q - T ||_F
    via the SVD of M = T^T S = U Sigma V^T, returning Q* = V U^T and the residual
    || S Q* - T ||_F.

    Determinism: fixed reduction order; bf16 inputs are upcast to fp32 (or fp64 when
      configured) before the SVD, so the outer/aggregation path stays bitwise-reproducible
      (INV-AGG-DETERMINISM, RFC-0003 §7).
    Failure: raises GaugeError('DegenerateProcrustes') when the smallest singular value of
      M falls below the condition tolerance (the rotation is not well-defined); the caller
      clamps/conditions and re-tries, or skips the backstop for that participant and logs
      gauge/procrustes_residual at WARN (RFC-0015).
    """
```

The fold-in is a pure linear operation applied to the participant's released delta *before* it enters
the deterministic outer reduction; it does not read any private state and is recomputable by a verifier.

### 6. Layer 4 — function-space distillation (fallback / heterogeneity)

If weight-space averaging is unstable at scale, or to admit participants with *different* encoder
sizes, aggregate **behaviors** instead of weights: each participant emits predictions on $\mathcal{P}$;
the coordinator forms a consensus (after frame alignment via Layer 3); a global student distills it.
This is gauge-invariant by construction — it compares functions on shared inputs, never weights — at a
higher per-round cost (an extra distillation pass over $\mathcal{P}$). It is held in reserve and is the
top rung of the ablation ladder ([RFC-0005 §6](RFC-0005-evaluation.md)).

```python
def distill_consensus(
    probe_predictions: Mapping[str, Tensor],  # participant_id -> f_c(P), each (k, d)
    *,
    align: bool = True,                        # Layer-3 align before averaging
) -> Tensor:
    """Frame-aligned consensus target on the probe for global-student distillation.

    Returns the consensus embeddings (k, d). When `align`, each participant's probe
    embeddings are Procrustes-aligned to a reference (the round-0 E_ref) before the mean,
    so the consensus is gauge-invariant. Pure function of public-probe outputs only;
    no private data crosses (INV-RESIDENCY not at stake — probe is public).
    """
```

### 7. The central hyperparameter

$\lambda_{\text{anc}}$ trades frame stability against representational freedom:

- too high → the encoder is clamped to the reference frame *and* its quality;
- too low → the frame drifts and averaging degrades.

Characterize it in Stage B ([RFC-0005 §6](RFC-0005-evaluation.md)) by sweeping $\lambda_{\text{anc}}$
against (i) the frame-drift diagnostic and (ii) downstream MPC success. Hypothesis: **warm-start + a
small $\lambda_{\text{anc}}$ keeps frames pinned cheaply**, so Layers 3–4 rarely fire. The sweep
harness is part of the Testing Strategy below.

### 8. The training algorithm (per outer round, preserved verbatim)

```
Given: global θ, φ; participants c = 1..C; public probe P, landmark targets {t_i};
       sketch seed s_t (→ projection matrix A); inner horizon H.
1. Broadcast θ, φ, s_t to all participants. Action heads h^(c) stay local.   # INV-ACTIONHEAD-LOCAL
2. Each participant c, in parallel (inner FSDP/TP):
     for H steps:  minimize  L_pred + λ_sig·SIGReg_A(f) + λ_anc·L_anchor(f; P, t_i)
                   on local data via AdamW.   # raw data never leaves (INV-RESIDENCY)
     form pseudo-gradient Δ_c = (θ_c, φ_c) − (θ, φ)
     DP: clip ‖Δ_c‖, add Gaussian noise               # RFC-0012 (INV-DP-BOUND)
3. Secure-aggregate Σ_c Δ_c  (individual Δ_c hidden).  # RFC-0011
4. (Backstop) Procrustes-align participants on P if drift exceeds threshold.  # Layer 3
5. Outer Nesterov step:  (θ, φ) ← (θ, φ) − η_out · OuterOpt( mean_c Δ_c )     # INV-AGG-DETERMINISM
6. Hash-commit θ^{global}_{t+1}.                        # RFC-0010 (INV-CHECKPOINT-HASH)
```

Step ordering is load-bearing for verifiability: DP (step 2) precedes secure aggregation (step 3), and
the Procrustes backstop (step 4) and outer step (step 5) operate only on public-probe data and the
revealed sum, so the entire aggregation path is a deterministic, publicly recomputable function
(`INV-AGG-DETERMINISM`, [RFC-0003 §7](RFC-0003-federated-protocol.md#7-determinism-concurrency-error-propagation)).

### 9. The frame-drift diagnostic (the headline measurement)

The headline empirical artifact is the **frame-drift diagnostic**: the inter-participant Procrustes
residual (or mean rotation angle) on $\mathcal{P}$ over training. Naive `FedAvg` visibly diverges
(frames rotate apart); the anchored design holds it pinned. This is itself novel — to our knowledge the
first measurement of latent frame-drift under federated JEPA — and stands as a contribution
independently of the planning result. The full evaluation protocol is
[RFC-0005 §2](RFC-0005-evaluation.md); the per-round, per-pair emission schema is
[RFC-0015](RFC-0015-observability-diagnostics.md).

```python
@dataclass(frozen=True)
class FrameDriftReport:
    round_index: int
    pairwise_angle_deg: Mapping[tuple[str, str], float]   # (c, c') -> mean rotation angle (deg)
    pairwise_residual: Mapping[tuple[str, str], float]    # (c, c') -> Procrustes residual ||SQ*-T||_F
    drift_from_global_deg: Mapping[str, float]            # c -> angle to the global model on P
    probe_hash: str                                       # the pinned probe; INV-PROBE-PIN

def frame_drift(embeddings: Mapping[str, Tensor]) -> FrameDriftReport:
    """Compute the frame-drift diagnostic from probe embeddings ([conventions §5](../spec/conventions.md#5-public-api-surface)).

    Input: participant_id -> f_c(P), each (k, d), computed on the pinned public probe.
    Each pairwise entry uses procrustes_align; the rotation angle is derived from Q* as
    arccos((tr(Q*) - (d - dim_of_rotation_plane)) / ...) reduced to a scalar mean angle.
    Determinism: pure function of its inputs; identical embeddings + probe yield an
      identical report (reproducible headline figure, INV-AGG-DETERMINISM spirit).
    Cost: O(C^2) Procrustes solves per round; sampling pairs at large C is an Open Question
      (RFC-0015).
    """
```

### 10. Failure modes and system response

| Failure mode | Trigger | Detection | Error ([conventions §6](../spec/conventions.md#6-error-taxonomy)) | System response |
|---|---|---|---|---|
| Frame drift beyond threshold | Anchor too weak / large $H$ / strong non-IID | `gauge/drift_angle_deg` exceeds configured threshold at aggregation | `FrameDriftExceeded` (`GaugeError`) | Fire the Layer-3 Procrustes backstop for the offending participant(s); if it still exceeds after alignment, abort the round and log at ERROR for a $\lambda_{\text{anc}}$ / $H$ review |
| Degenerate Procrustes SVD | Near-equal/zero singular values in $M = T^\top S$ | Smallest singular value below condition tolerance in `procrustes_align` | `DegenerateProcrustes` (`GaugeError`) | Clamp/condition the SVD and retry once; if still degenerate, skip the backstop for that participant, keep its un-aligned delta, and log `gauge/procrustes_residual` at WARN |
| Under-determined landmark frame | $k < d$ generic landmarks (Variant A) | Shape check at `FrameAnchor.loss` | `FrameDriftExceeded` (`GaugeError`) | Reject the configuration at construction with remediation "increase probe landmark count to k >= d (RFC-0004 §3)"; fail-closed before training |
| Sketch seed mismatch | A participant uses an $A$ from the wrong seed | Seed comparison at participant ingress | `GaugeError` | Reject the participant's round contribution; remediation "reconstruct A from the RoundOpen seed (INV-SKETCH-CONSISTENCY)" |
| Warm-start mismatch at $t{=}0$ | Participant's encoder hash $\neq$ pinned warm-start | Hash comparison at round-0 admission | `GaugeError` | Refuse admission to round 0; remediation "reload the pinned warm-start (INV-WARMSTART-T0)" |
| Probe hash mismatch | Probe content $\neq$ `RoundOpen` probe hash | Hash recompute in `anchor.py` loader | `ProbeError` | Fail-closed; refuse to build targets; remediation "re-pin the probe (RFC-0004 §3, INV-PROBE-PIN)" |
| Nondeterministic alignment reduction | Atomics / unordered reduction on the alignment path | Determinism self-check on the outer step | `NonDeterministicAggregation` (`AggregationError`) | Never swallowed ([conventions §6](../spec/conventions.md#6-error-taxonomy)); abort and recompute with fixed reduction order ([RFC-0003 §7](RFC-0003-federated-protocol.md#7-determinism-concurrency-error-propagation)) |

Error-handling rules ([conventions §6](../spec/conventions.md#6-error-taxonomy)): never a bare `except`; `NonDeterministicAggregation` is never swallowed;
all gauge errors carry `.code` (a `LensembleErrorCode`) and a `.remediation` string. Frame drift is
*recoverable* via the backstop; warm-start, probe-hash, and determinism failures are *fail-closed*.

### 11. Concurrency, determinism, dtype

- The anchor loss and SIGReg statistics are computed on each participant's inner-parallel workers and
  reduced *within* the participant's trust domain (RFC-0008); only the resulting $\Delta_c$ crosses a
  boundary.
- The Layer-3 backstop, the outer step, and `frame_drift` run on the aggregation path and MUST be
  bitwise-deterministic given their inputs (`INV-AGG-DETERMINISM`): fixed reduction order, fp32 (or
  fp64 when configured) accumulation, no atomics. A determinism self-check runs each outer step and
  raises `NonDeterministicAggregation` on divergence ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)).
- bf16 forward, fp32 accumulation per [conventions §9](../spec/conventions.md#9-determinism-dtype-device); the Procrustes SVD and the diagnostic upcast to fp32/fp64
  before the decomposition so the public recomputation matches the coordinator bit-for-bit.
- Device: CUDA primary; the small CI configs run the differentiable SVD and Procrustes on CPU
  (`torch>=2.4` provides a differentiable SVD and deterministic algorithms; [conventions §11](../spec/conventions.md#11-external-dependencies)).

## Alternatives Considered

- **Alignment-only (Procrustes at aggregation, no anchor loss).** *What:* keep Layer 3, drop Layer 2 —
  align frames at each outer step and average. *Why considered:* cheapest; no per-step loss term.
  *Why rejected:* insufficient — the drift re-opens *between* rounds because nothing during local
  fine-tuning holds the frame, so each round starts from re-rotated frames and the backstop fights a
  losing battle, especially as $H$ grows. Layer 2 is what keeps Layer 3 from binding.
- **Permutation-only alignment (Git-Re-Basin-style).** *What:* align by matching/permuting neurons
  before averaging. *Why considered:* standard model-merging removes the permutation symmetry every
  network has. *Why rejected:* it handles the neuron-permutation symmetry but NOT the $O(d)$ latent
  rotation, which is a continuous symmetry of the SIGReg-JEPA objective that anchored models never see.
  Permutation is a measure-zero subgroup of $O(d)$; the frame can rotate arbitrarily within the
  permutation-invariant set.
- **Variant A (landmark anchoring) vs Variant B (rotational-drift penalty).** *What:* A pins absolute
  targets; B penalizes only the Procrustes rotation, leaving content drift free. *Why considered:* B is
  the cleaner statement of "pin frame, not content" and gives fuller frame control. *Why A is the
  default:* A needs no differentiable SVD on the training backward path, so it has no degenerate-SVD
  instability and is trivially deterministic; B's SVD can be ill-conditioned near degenerate singular
  values (`DegenerateProcrustes`). B is offered as an opt-in once A's behavior is characterized in
  Stage B.
- **Function-space distillation as the primary aggregator (Layer 4 promoted).** *What:* always
  aggregate behaviors on $\mathcal{P}$, never weights. *Why considered:* gauge-invariant by
  construction and admits heterogeneous encoder sizes. *Why rejected as primary:* higher per-round cost
  (an extra probe-distillation pass) and it discards the weight-space DiLoCo machinery that the rest of
  the protocol and the verifiability story rely on. Retained as the heterogeneity/instability fallback
  (Layer 4).
- **Fork A (freeze the encoder, federate only the predictor).** *What:* keep the warm-started encoder
  frozen; the frozen shared encoder *is* a shared frame, so the gauge dissolves and Layers 2–4 are
  unnecessary. *Why considered:* it is the clean, safe degrade ([RFC-0001 §5](RFC-0001-architecture.md#5-training-topology-two-level),
  this RFC's fallback below). *Why not the lead:* it sacrifices the end-to-end (Fork B) novelty that is
  the project's core claim ([conventions §0](../spec/conventions.md#0-project-identity)). Documented and tested as the fallback, not the default.

## Drawbacks

- **Anchor too strong clamps quality.** A large $\lambda_{\text{anc}}$ ties the encoder to the
  reference frame *and* to the reference encoder's quality, capping the gain federation can deliver.
  Mitigation: the Stage-B $\lambda_{\text{anc}}$ sweep and the hypothesis that a *small*
  $\lambda_{\text{anc}}$ suffices given the warm-start.
- **Variant B SVD instability.** The differentiable Procrustes SVD is ill-conditioned near degenerate
  singular values, producing large or undefined gradients (`DegenerateProcrustes`). Mitigation: clamp/
  condition the SVD; default to Variant A.
- **Compounding failure regime.** Non-IID marginals, collapse re-entry, and DiLoCo drift compound: the
  hardest setting is strong non-IID with a long inner horizon. The design must be validated across the
  non-IID severity and $C$/$H$ sweeps of [RFC-0005 §7](RFC-0005-evaluation.md), not just the easy
  near-IID case.
- **Dependence on a released warm-start.** The $t{=}0$ frame anchor and Variant-A targets derive from
  $f_{\text{ref}}$, a pinned external release (V-JEPA 2, [conventions §11](../spec/conventions.md#11-external-dependencies)). A change to the warm-start is a
  re-anchoring event (it changes $E_{\text{ref}}$ and every $t_i$). Mitigation: pin the warm-start hash
  (`INV-WARMSTART-T0`) and version it via the artifact format ([RFC-0010](RFC-0010-artifact-checkpoint-format.md)).
- **$O(C^2)$ diagnostic cost.** The pairwise frame-drift diagnostic is quadratic in participant count;
  at large $C$ this is a real cost (Open Question, [RFC-0015](RFC-0015-observability-diagnostics.md)).

## Migration / Rollout

The four layers are **additive** and are introduced in order; they constitute the ablation ladder of
[RFC-0005 §6](RFC-0005-evaluation.md), so the rollout and the experiment are the same sequence:

1. Layer 1 (shared sketch) ships with the objective in `v0.1` (Stage A, centralized) — it is needed for
   objective consistency even single-site and is cheap.
2. Layer 2 Variant A (landmark anchor) is the default for `v0.2` (Stage B) simulated federation; the
   warm-start guarantee (`INV-WARMSTART-T0`) is live from round 0.
3. Layer 3 (Procrustes backstop) is wired in `v0.2` but configured off-by-default (fires only above the
   drift threshold), so its contribution is measurable as a ladder rung.
4. Layer 4 (function-space distillation) and Layer 2 Variant B are opt-in rungs, enabled only if the
   Stage-B drift-vs-quality curves show A+Layer 3 insufficient at scale.

Start with Variant A and a small $\lambda_{\text{anc}}$. The drift threshold that fires Layer 3 and the
production $\lambda_{\text{anc}}$ are Stage-B sweep outputs (Open Questions). Over the real-network
transition (`v0.3`, Stage C) the gauge layers are unchanged — only the transport and secure-aggregation
backend change ([RFC-0011](RFC-0011-secure-aggregation.md)).

### Fork A fallback

If Fork-B gauge control proves unstable at foundation scale, freeze the warm-started encoder and
federate only the predictor (Fork A, the V-JEPA-2-AC structure;
[RFC-0001 §5](RFC-0001-architecture.md#5-training-topology-two-level)). The frozen shared encoder *is* a shared frame, so the gauge
problem dissolves and Layers 2–4 become unnecessary (Layer 1 still applies to any SIGReg on the
predictor side, if used). This sacrifices the end-to-end novelty but preserves the sovereignty story
and is the documented safe degrade. Fork A must be supported and tested by `v1.0` ([conventions §12](../spec/conventions.md#12-milestones-and-stages)).

## Testing Strategy

ML-specific tests (the differentiating layer of [07-testing-strategy](../spec/07-testing-strategy.md)),
each runnable on CPU with tiny synthetic fixtures:

- **Gauge invariance** — sample a random $Q\in O(d)$ (via QR of a Gaussian matrix), apply
  $f_\theta\mapsto Qf_\theta,\ g_\phi\mapsto Qg_\phi Q^\top$, and assert the bare objective
  ($\lambda_{\text{anc}}=0$: prediction + SIGReg) is unchanged within fp32 tolerance, while the full
  objective ($\lambda_{\text{anc}}>0$) *changes* (the anchor breaks the symmetry, by design).
- **Anchor pins the frame** — with $k\ge d$ generic landmark constraints, optimize a free $Q$ against
  $\mathcal{L}_{\text{anchor}}$ and assert the recovered rotation $Q\approx I$; with $k<d$, assert
  `FrameDriftExceeded` is raised (under-determined).
- **Procrustes closed-form correctness** — assert `procrustes_align` agrees with a brute-force search
  over a discretized $O(d)$ on small $d$, and that $Q^\star=VU^\top$ minimizes the Frobenius residual.
- **SIGReg statistic correctness** — assert the Epps–Pulley statistic is near zero on a known
  standard-normal sample and large on a known non-normal sample (owned by
  [RFC-0008](RFC-0008-model-objective-numerics.md); referenced here because Layer 1 depends on it).
- **Frame-drift diagnostic on synthetically rotated silos** — construct $C$ encoders that are the
  warm-start composed with known rotations $Q_c$; assert naive averaging's `frame_drift` diverges over
  rounds while the anchored configuration holds the pairwise angle flat/low. This is the unit-scale
  proxy for the headline figure.
- **Aggregation determinism** — run the Layer-3 align + outer step twice on identical inputs and assert
  bitwise-identical output (`INV-AGG-DETERMINISM`); inject a nondeterministic reduction and assert
  `NonDeterministicAggregation`.
- **`DegenerateProcrustes` handling** — feed `procrustes_align` a near-rank-deficient $M$ and assert it
  raises `DegenerateProcrustes`, and that the backstop's clamp-and-skip path keeps the round alive.
- **Probe-pin / warm-start invariants** — assert `FrameAnchor` raises `ProbeError` on a probe-hash
  mismatch and that round-0 admission raises `GaugeError` on a warm-start-hash mismatch.
- **$\lambda_{\text{anc}}$ sweep harness** — a small integration test that runs the ablation rungs on a
  toy config and emits the drift-vs-quality curve, exercising the same code path Stage B uses at scale.

Numerical tolerance: exact (bitwise) equality is required on the aggregation/outer-step path; the
gauge-invariance and Procrustes tests use `atol/rtol` appropriate to fp32 (the policy is set in
[07-testing-strategy](../spec/07-testing-strategy.md)).

## Open Questions

OPEN QUESTION: The production value of $\lambda_{\text{anc}}$ — the central knob trading frame
stability against representational freedom. Owner @AbdelStark. Resolution: the Stage-B
$\lambda_{\text{anc}}$ sweep against the frame-drift diagnostic and MPC success
([RFC-0005 §6](RFC-0005-evaluation.md)), milestone v0.2.

OPEN QUESTION: Variant A (landmark) vs Variant B (rotational-drift penalty) at video-world-model scale —
whether B's fuller frame control justifies its differentiable-SVD instability once encoder size grows.
Owner @AbdelStark. Resolution: the scale step of [RFC-0005 §7](RFC-0005-evaluation.md), milestone v0.2
toward Stage E.

OPEN QUESTION: The drift threshold (in `gauge/drift_angle_deg`) at which the Layer-3 Procrustes backstop
fires. Owner @AbdelStark. Resolution: Stage-B characterization of the drift distribution under the
anchored configuration; set the threshold above the anchored regime and below the naive-divergence
regime, milestone v0.2.

OPEN QUESTION: The interaction between the inner horizon $H$ and frame drift (longer $H$ ⇒ more drift
before the outer step). Owner @AbdelStark. Resolution: jointly with the $H$ schedule of
[RFC-0003](RFC-0003-federated-protocol.md) in the Stage-B $C$/$H$ sweep, milestone v0.2.

RISK: Variant B's differentiable Procrustes SVD may produce unstable gradients near degenerate singular
values even with conditioning, making B unusable at scale. Resolution plan: default to Variant A; gate B
behind the Stage-B comparison above; if B is needed and remains unstable, fall back to Layer 4
(function-space distillation), which is gauge-invariant without an SVD on the training path.

## References

- Internal: [RFC-0001 — Architecture](RFC-0001-architecture.md) (§1 model, §2 module map, §4
  federation map, §5 training topology, §6 trust boundaries, §7 data-flow lifecycles); [RFC-0003 — Federated Protocol](RFC-0003-federated-protocol.md)
  (DiLoCo outer step, the $H$ horizon, deterministic aggregation); [RFC-0004 — Data & Provenance](RFC-0004-data-provenance.md)
  (§3 the public probe, landmark targets, content-hash pin); [RFC-0005 — Evaluation](RFC-0005-evaluation.md)
  (§2 the frame-drift diagnostic, §6 the ablation ladder, §7 sweeps); [RFC-0006 — Verifiable
  Contribution](RFC-0006-verifiable-contribution.md) (§3 the public-recomputation property and the
  anchored-frame-keeps-aggregation-near-linear synergy); [RFC-0008 — Model, Objective & Numerics](RFC-0008-model-objective-numerics.md)
  (the SIGReg objective and numerical contracts this RFC consumes); [RFC-0010 — Artifact & Checkpoint
  Format](RFC-0010-artifact-checkpoint-format.md) (the committed-weights hash);
  [RFC-0011 — Secure Aggregation](RFC-0011-secure-aggregation.md) and
  [RFC-0012 — Differential Privacy](RFC-0012-differential-privacy.md) (the boundary steps of the
  per-round algorithm); [RFC-0015 — Observability & Diagnostics](RFC-0015-observability-diagnostics.md)
  (the frame-drift emission schema). Spec: [01 — Architecture](../spec/01-architecture.md),
  [03 — Data Model](../spec/03-data-model.md) (`FrameDriftReport`),
  [04 — Error Model](../spec/04-error-model.md) (`GaugeError`),
  [07 — Testing Strategy](../spec/07-testing-strategy.md).
- External ([conventions §11](../spec/conventions.md#11-external-dependencies)): LeJEPA / LeWM — the SIGReg objective (random-projection + characteristic-function
  Gaussianity, Cramér–Wold / Epps–Pulley); V-JEPA 2 — the encoder warm-start and AC predictor recipe
  ($f_{\text{ref}}$ and the $t{=}0$ frame anchor); the orthogonal-Procrustes / Cramér–Wold literature
  (the closed-form $Q^\star=VU^\top$ and the projection-based distribution-matching argument);
  DiLoCo / OpenDiLoCo / INTELLECT — the outer-loop optimizer whose drift interacts with the gauge.
