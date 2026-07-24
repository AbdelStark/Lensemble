# RFC-0012 — Differential Privacy Accounting

| | |
|---|---|
| **RFC** | 0012 |
| **Title** | Differential Privacy Accounting |
| **Slug** | differential-privacy |
| **Status** | Accepted |
| **Track** | Standards |
| **Authors** | @AbdelStark |
| **Created** | 2026-06-02 |
| **Target milestone** | v0.2 (simulated federation, one cluster); real-boundary enforcement in v0.3 |
| **Area** | `area:privacy` |
| **Requires** | [RFC-0003](RFC-0003-federated-protocol.md) (the round that releases `Δ_c`), [RFC-0011](RFC-0011-secure-aggregation.md) (the sum the noise protects) |
| **Defers to** | [RFC-0013](RFC-0013-coordinator-runtime.md) (round state machine / churn semantics) |

> **Implementation status (2026-07-24 — claim-grade release and cumulative enforcement blocked).**
> `Participant.local_round()` does apply the configured clip-then-noise mechanism before constructing
> the released `PseudoGradient`, and the stateful RDP/PRV accountant implementations have direct tests.
> That Gaussian draw is currently deterministic from
> `(config.determinism.root_seed, round_index, participant_id)`. The coordinator receives the same config,
> and the root seed is recorded in the run manifests, so an observer with that seed and the matching
> runtime can reconstruct and subtract the noise. This is deterministic mechanism/test plumbing, not a
> claim-grade DP release against the coordinator or a manifest reader. Claim-grade DP requires
> participant-secret CSPRNG entropy that is never exposed in shared config, messages, reports, or
> manifests.
>
> Separately, the Phase 3 service does not retain either accountant across rounds. After each successful
> coordinator commit, `build_phase3_aggregation_privacy_report()` constructs a fresh accountant, checks
> and consumes exactly one step, and reports `rounds_accounted=1`,
> `status=deterministic_replay_only`, and `effective_dp=false`. It does not perform a cumulative
> pre-release `would_exceed` check and cannot stop a multi-round run at its configured total budget.
> Historical Phase 3 `DP-accounted rounds` and `ε≈5.30` values are therefore one-round reporting
> snapshots, not a composed ten-round epsilon or evidence of run-budget enforcement.

## Summary

This RFC specifies the intended differential-privacy (DP) contract around each participant's
already-aligned pseudo-gradient `Δ'_c`, and the `(ε,δ)` accountant that must bound cumulative privacy loss over the
planned number of federated rounds. The mechanism is the Gaussian mechanism on a clipped update: clip
`Δ_c` to L2 norm `C_clip` (`INV-DP-BOUND`), then add isotropic Gaussian noise `N(0, σ² C_clip² I)`. The
clip-then-noise *operations and their ordering* are pinned at the protocol level in
[RFC-0003 §4](RFC-0003-federated-protocol.md#4-differential-privacy-protocol-level); this RFC owns the
mechanism's privacy semantics, the accountant contract, the calibration of `σ` to a target budget, and
the fail-closed behavior when the budget is spent.

The privacy unit is **the participant's contribution to a round** — per-participant *update-level* DP,
not per-example DP-SGD inside the inner loop. One participant's entire `H`-step local update is one
"record"; neighboring datasets differ by the presence or absence of one participant's round
contribution. This RFC states that scope and its honest limits plainly, because the unit determines what
the released `(ε,δ)` actually protects.

The accountant contract lives behind `lensemble.privacy.accountant` as a swappable interface so a reference
implementation (RDP or PRV; the [conventions document](../spec/conventions.md) names `opacus` or a vendored accountant,
[conventions §11](../spec/conventions.md#11-external-dependencies)) can be replaced without touching the
mechanism or the round. The mechanism lives in `lensemble.privacy.dp` and is invoked from
`Participant.local_round` before the `PseudoGradient`
([03 §6](../spec/03-data-model.md#6-pseudogradient--the-one-private-object-that-does-cross-the-boundary))
leaves the participant boundary. The persistent-accountant invocation required by this RFC is not yet
integrated into that release path.

## Motivation

The intended secure-aggregation protocol ([RFC-0011](RFC-0011-secure-aggregation.md)) hides every
individual `Δ_c` from the coordinator, but it protects only against an honest-but-curious aggregator
observing the *transcript*; it does not bound what the *revealed sum* `Σ_c Δ_c`, or the released global model
`(θ_{t+1}, φ_{t+1})`, leaks about any one participant's sovereign data. An individual `H`-step update,
and to a lesser degree the aggregate, can memorize and re-expose training trajectories. The federation
must therefore add a calibrated-noise guarantee on top of the residency boundary (`INV-RESIDENCY`,
[06 §3](../spec/06-security.md#3-residency-enforcement-inv-residency)) so that a participant's contribution is
provably indistinguishable, up to `(ε,δ)`, from its absence.

Two facts shape the design:

1. **The unit must be the participant-round, not the example.** Per-example DP-SGD would clip and noise
   every micro-batch gradient inside the `H`-step inner loop. At ViT-L scale over `H ∈ [50, 500]` steps
   ([RFC-0003 §2](RFC-0003-federated-protocol.md#2-round-structure-diloco-outer-loop)) that is a large,
   recurring per-step cost, and it privatizes the wrong object: what crosses the boundary is the
   aggregate `H`-step delta, not individual examples. Clipping and noising the released `Δ_c` once per
   round privatizes exactly the object that leaves the boundary, at one clip + one noise draw per round.

2. **DP noise interacts with the rest of the objective.** The predictor `g_φ` is compact
   ([RFC-0008 §Predictor](RFC-0008-model-objective-numerics.md)); its delta is small, so a fixed `σ`
   degrades it more than the larger encoder delta. The noise also perturbs the SIGReg variance statistic
   and the anchor term that closes the gauge ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md)). The
   joint calibration of `(σ, λ_sig, λ_anc, C_clip)` is a Stage-B experiment shared with
   [RFC-0003 §Open Questions](RFC-0003-federated-protocol.md#open-questions) and
   [RFC-0002 §7](RFC-0002-gauge-and-aggregation.md#7-the-central-hyperparameter), not a shipped default.

## Goals

- Define the Gaussian mechanism on the clipped pseudo-gradient and pin `INV-DP-BOUND` to where it is
  enforced (`lensemble.privacy.dp`).
- State the privacy unit (participant-round, update-level DP) and what `(ε,δ)` does and does not protect.
- Specify the accountant interface (`privacy.accountant`): RDP or PRV accounting over the planned rounds,
  budget query before release, `PrivacyBudgetExceeded` when the budget is spent, swappable backend.
- Specify how `σ` is calibrated to a target `(ε,δ)` given the round count, and how the budget is consumed
  per round.
- Separate deterministic, seeded replay for tests from participant-secret CSPRNG entropy for claim-grade
  DP, while preserving `INV-AGG-DETERMINISM` on the downstream sum once a released update is fixed.
- Enumerate the DP failure modes with the error raised and the system response, using the
  [conventions §6](../spec/conventions.md#6-error-taxonomy) taxonomy.
- State the interaction with secure aggregation (noise added per-participant *before* masking) and with
  int8 quantization (quantization applied after noising), while enforcing that participant-specific
  alignment/re-difference occurs before all three (`INV-ALIGN-BEFORE-RELEASE`).
- Record the joint-calibration and privacy-unit-under-churn open questions with owner and resolution
  path.

## Non-Goals

- The clip-then-noise protocol *placement* in the round lifecycle — owned by
  [RFC-0003 §4](RFC-0003-federated-protocol.md#4-differential-privacy-protocol-level); this RFC does not
  re-specify the round.
- The secure-aggregation cryptography (pairwise masking, threshold secret sharing, TEE backend) that
  hides individual `Δ_c` — owned by [RFC-0011](RFC-0011-secure-aggregation.md). DP and secure aggregation
  are complementary, not substitutes (`Drawbacks` and §6 below).
- Per-example DP-SGD inside the inner loop (rejected; `Alternatives Considered`).
- The joint `(σ, λ_sig, λ_anc, C_clip)` *sweet spot* — that is the Stage-B experiment, an Open Question
  here, not a normative default.
- The contribution ledger / provenance binding that records *which* participants contributed
  ([RFC-0014](RFC-0014-provenance-commitments.md)); DP bounds what a contribution leaks, not its origin.
- Stage-D cryptographic proofs and Stage-E own-pretrain (out of v1.0 scope,
  [00 §8](../spec/00-overview.md#8-v10-scope-boundary)).

## Proposed Design

### 1. The mechanism

For participant `c` in round `t`, first form the raw local full weights and apply any required
participant-specific Procrustes transform, then re-difference
`Δ'_c = W'_c − W_t` (`Δ'_c = Δ_c` when the backstop does not fire;
[RFC-0002 §5](RFC-0002-gauge-and-aggregation.md#5-layer-3--procrustes-re-alignment-at-aggregation-backstop)).
Before the `PseudoGradient` leaves the boundary, apply two operations to `Δ'_c` in this order:

1. **Clip** to a fixed L2 bound:

   $$\Delta'_c \leftarrow \Delta'_c \cdot \min\!\left(1,\; \frac{C_{\text{clip}}}{\lVert\Delta'_c\rVert_2}\right).$$

   After clipping, $\lVert\Delta'_c\rVert_2 \le C_{\text{clip}}$ holds exactly. This is **`INV-DP-BOUND`**,
   enforced in `lensemble.privacy.dp` and asserted on the post-clip norm. The bound is the per-record
   sensitivity that makes the noise calibration sound: with the participant-round as the unit, adding or
   removing one participant changes the summed update by at most `C_clip` in L2 norm.

2. **Noise** with the isotropic Gaussian mechanism:

   $$\Delta'_c \leftarrow \Delta'_c + \xi,\qquad \xi \sim N\!\left(0,\; \sigma^2 C_{\text{clip}}^2 I\right),$$

   where `σ` is the noise multiplier calibrated to a target `(ε,δ)` over the planned round count (§3).
   The noised vector becomes `PseudoGradient.delta`. In the live implementation,
   `PseudoGradient.l2_norm` records the released **post-noise** norm and may exceed `C_clip`; the bounded
   post-clip, pre-noise norm is checked internally before release.

This is `INV-ALIGN-BEFORE-RELEASE`. A backstop that consults raw local weights after the Gaussian
mechanism would be an additional data-dependent access not represented by this mechanism contract; it
must not be smuggled in as generic post-processing. A deterministic common function of only an already
DP-released vector retains DP by post-processing, but it is not the required full-weight alignment and
cannot recover participant-specific information after a secure sum. Optional quantization remains
downstream of noise.

The mechanism signature:

```python
# lensemble/privacy/dp.py
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass(frozen=True)
class DPConfig:
    """Per-participant update-DP parameters (a config group; see RFC-0009)."""
    clip_norm: float          # C_clip, L2 sensitivity bound; > 0
    noise_multiplier: float   # sigma; noise std = sigma * C_clip; >= 0 (0 disables noise, NOT DP)
    target_epsilon: float     # epsilon budget the accountant calibrates / checks against
    target_delta: float       # delta budget; convention delta < 1 / (number of participants)
    enabled: bool = True      # if False, no clip/noise applied; the run is recorded as non-private

def clip_delta(delta: Tensor, clip_norm: float) -> tuple[Tensor, float]:
    """Clip `delta` to L2 norm `clip_norm`. Pure, deterministic, device-agnostic.

    Returns (clipped_delta, post_clip_norm). Postcondition (INV-DP-BOUND):
        post_clip_norm <= clip_norm  (within fp32 tolerance asserted by the caller).
    """
    ...

def add_gaussian_noise(
    delta: Tensor, clip_norm: float, noise_multiplier: float, generator: torch.Generator
) -> Tensor:
    """Add N(0, (noise_multiplier * clip_norm)^2 I) using a seeded generator.

    The live replay path derives `generator` from the shared run root seed and
    (round_index, participant_id). A claim-grade path must instead supply participant-secret CSPRNG
    entropy that is not recorded in a cross-boundary artifact.
    """
    ...

def privatize(delta: Tensor, cfg: DPConfig, generator: torch.Generator) -> tuple[Tensor, float]:
    """clip then noise. Returns (private_delta, post_clip_norm). Raises ConfigError on cfg.clip_norm<=0
    or cfg.noise_multiplier<0. If cfg.enabled is False, returns (delta, ||delta||) unchanged."""
    ...
```

`clip_delta` is a pure function and is deterministic. `add_gaussian_noise` is deterministic *given its
`generator`*. In the live path, `Participant.local_round()` derives that generator with the public
`derive` function from the shared `config.determinism.root_seed`, round index, and participant id. The
coordinator is constructed from the same config; generic `RunManifest` artifacts record `root_seed` and
the fully resolved config, and the Phase 3 run manifest records `run_shape.root_seed`
([RFC-0009 §RunManifest](RFC-0009-configuration-reproducibility.md),
[conventions §9](../spec/conventions.md#9-determinism-dtype-device)). The draw is therefore replayable by
the coordinator and later manifest readers. Because they can subtract it from the released delta, the
current deterministic draw provides no DP protection against those observers.

**Required entropy boundary.** A claim-grade run MUST draw Gaussian noise from participant-secret CSPRNG
entropy that the coordinator cannot derive and that is not placed in shared config, a cross-boundary
message, a report, or a manifest. Reproducible seed injection remains useful for mechanism tests and
non-private replay runs, but those runs MUST be labeled non-private. A public artifact may record the
entropy-source policy and mechanism parameters, never the secret entropy. This integration is a blocker
alongside persistent accounting; its runtime owner is
[RFC-0013](RFC-0013-coordinator-runtime.md).

### 2. The privacy unit (scope)

Under the required entropy and accounting integration, the unit of privacy is one **participant's
contribution to one round**. Formally, two federated runs are
neighbors if they differ in the presence/absence of one participant's `Δ_c` in one round. The released
`(ε,δ)` then bounds, for that neighboring relation, how distinguishable the released artifacts (the
revealed sum, the committed global model) are with vs. without that contribution.

This is **update-level DP**, and what it protects is correspondingly coarse and honest. It bounds
leakage of *the participant's round contribution as a whole*, not of any single training example, frame,
or episode within it — example-level DP would require per-example clipping inside the inner loop
(rejected, `Alternatives Considered`). Over `T` rounds a participant present in all rounds contributes
`T` times; the accountant must compose the per-round mechanism over those releases (§3), and both the
per-round `(ε,δ)` and the composed total must be reported. It does **not** by itself protect against a
participant that under-reports its unit (pools many users into one contribution): honest accounting of
what a unit *contains* is a Phase-1 trust assumption
([06 §1](../spec/06-security.md#1-threat-model)) that Phase 2
([RFC-0006](RFC-0006-verifiable-contribution.md)) moves toward attestation but does not fully close.

### 3. The accountant

The accountant contract is stateful: it tracks cumulative privacy loss across rounds and calibrates /
checks `σ` against a target `(ε,δ)`. It is abstracted behind an interface so the reference backend (RDP
via `opacus`, or a vendored PRV accountant;
[conventions §11](../spec/conventions.md#11-external-dependencies)) is swappable. The implementations
support this composition, but the live Phase 3 service currently discards that state after one
post-commit report step.

```python
# lensemble/privacy/accountant.py
from typing import Protocol

class Accountant(Protocol):
    """Composes the per-round Gaussian mechanism and reports the spent (epsilon, delta)."""

    def calibrate_sigma(
        self, *, target_epsilon: float, target_delta: float, num_rounds: int, sample_rate: float = 1.0
    ) -> float:
        """Return the smallest noise_multiplier sigma whose composition over `num_rounds` releases
        stays within (target_epsilon, target_delta). `sample_rate` is the per-round participation
        probability (1.0 when every participant joins every round; < 1.0 enables privacy amplification
        by subsampling when participation is Poisson-sampled). Raises ConfigError on infeasible targets."""
        ...

    def step(self, *, noise_multiplier: float, sample_rate: float = 1.0) -> None:
        """Account for one round's release at the given sigma. Called once per round per participant
        after a successful release. Accumulates RDP / PRV state."""
        ...

    def spent(self, *, target_delta: float) -> float:
        """Return the cumulative epsilon spent so far at the fixed target_delta."""
        ...

    def would_exceed(self, *, target_epsilon: float, target_delta: float,
                     noise_multiplier: float, sample_rate: float = 1.0) -> bool:
        """True iff accounting for one MORE round at this sigma would push spent epsilon past target."""
        ...
```

Reference backends: `RDPAccountant` (Rényi-DP, the `opacus` default — fast, slightly loose composition)
and `PRVAccountant` (privacy-loss-distribution / PRV — tighter, the preferred reporting backend). Both
satisfy the `Accountant` protocol; the active backend is selected by config and recorded in the
`RunManifest`.

**Required budget lifecycle per round.** Before a participant releases its `PseudoGradient`, the runtime MUST query
`would_exceed`. If accounting for this round would push the cumulative `ε` past `target_epsilon` at
`target_delta`, the round refuses to release and raises `PrivacyBudgetExceeded`; training stops
(fail-closed — [04 §5.5 Privacy](../spec/04-error-model.md#55-privacy-lensembleprivacy)). Otherwise the
participant privatizes and releases, and on a successful round the runtime calls `step` to consume the
budget. The check-before-release / consume-after-success ordering ensures the released `(ε,δ)` is never
exceeded even if the round later aborts (a failed round does not consume budget).

### 4. Determinism, dtype, device

- **Clipping** is a pure, deterministic function (`clip_delta`), computed in fp32 over the flat delta
  ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)). It runs on CUDA or the CPU fallback;
  the CPU path is exercised by CI on tiny fixtures.
- **Noise** is a Gaussian draw supplied through an explicit generator (§1). The current
  shared-root-seed draw and its derivation are recorded for reproducibility and are therefore suitable
  only for tests and explicitly
  non-private replay evidence. Claim-grade mode requires participant-secret CSPRNG entropy (§1); outer
  aggregation remains deterministic with respect to the already released, fixed noised deltas.
- **Aggregation interaction (`INV-AGG-DETERMINISM`).** Participant-specific alignment happens before
  the DP transform; the DP transform then happens entirely *before* the delta enters the
  secure-aggregation / outer-step path. The bitwise-determinism contract of the outer
  step ([RFC-0003 §7](RFC-0003-federated-protocol.md#7-determinism-concurrency-error-propagation)) is a
  property of summing the *already-privatized* deltas in a fixed order; DP adds no nondeterminism to that
  path, because the noise is part of the (fixed, recorded) input `Δ_c`, not part of the reduction.
- **dtype.** Clipping and noise are computed in fp32 to keep the sensitivity bound and the noise scale
  exact; the privatized delta is the fp32 flat vector the `PseudoGradient` carries
  ([RFC-0003 §3](RFC-0003-federated-protocol.md#3-the-pseudogradient-contract)). Optional int8
  quantization is applied *after* noising (§6).

### 5. Module responsibilities

- `lensemble.privacy.dp` — the mechanism (`clip_delta`, `add_gaussian_noise`, `privatize`, `DPConfig`);
  enforces `INV-DP-BOUND`. Internal.
- `lensemble.privacy.accountant` — the `Accountant` protocol and the `RDPAccountant` / `PRVAccountant`
  reference backends; the active backend is config-selected. Internal.
- `lensemble.federation.participant` — target owner of the full-weight alignment/re-difference, followed
  by `privatize` on `Δ'_c`. It must also own or receive a persistent accountant, query it before
  release, and source noise from participant-secret CSPRNG entropy. Those responsibilities remain
  integration blockers
  ([RFC-0013](RFC-0013-coordinator-runtime.md)).

No public-API symbol is added by this RFC; DP is configured via the `privacy` config group
([RFC-0009 §Config Groups](RFC-0009-configuration-reproducibility.md)) and observed via the `dp/*`
metrics ([05 §2.4](../spec/05-observability.md#24-privacy-metrics-dp)).

### 6. Interaction with secure aggregation and quantization

- **Gauge-order precondition.** The input to clipping is the already-aligned displacement $\Delta'_c$.
  Alignment is participant-side (or inside a declared trusted/MPC boundary) and precedes clipping,
  noise, quantization, encoding, and masking (`INV-ALIGN-BEFORE-RELEASE`). A coordinator that receives
  only a secure sum cannot apply distinct participant rotations afterward.
- **Secure aggregation.** Under the completed target integration, DP noise is added **per-participant,
  before** the secure-aggregation mask
  ([RFC-0003 §5](RFC-0003-federated-protocol.md#5-secure-aggregation-requirement),
  [RFC-0011 §DP Interaction](RFC-0011-secure-aggregation.md)). The two are complementary: masking hides
  the individual `Δ_c` from the aggregator's *view*; DP bounds what the *revealed sum* and the released
  model leak. With both, the aggregator sees only `Σ_c (clip+noise)(Δ_c)`. Independent noise per
  participant means the summed noise is `N(0, C·σ²C_clip²I)`, which is the basis for the central-DP
  accounting on the aggregate when distributed-DP composition is used. The current runtime has neither
  coordinator-hidden noise entropy nor secure aggregation on its commit path, so it does not realize
  this composition.
- **Quantization order.** Optional int8 pseudo-gradient quantization
  ([RFC-0003 §6](RFC-0003-federated-protocol.md#6-heterogeneity--fault-tolerance)) is applied to the
  *already clipped-and-noised* delta, before masking. Quantization adds a bounded, tested round-trip
  error (RFC-0003 Testing Strategy) on top of the DP noise; it does not change the privacy unit. The
  privacy analysis treats the quantization error as an additional (bounded, data-independent) perturbation
  and does not credit it as privacy.

### 7. Failure modes

Every DP failure mode uses the [conventions §6](../spec/conventions.md#6-error-taxonomy) taxonomy. The
table states required integrated behavior. Config validation and clip-bound checks are live; cumulative
budget failure is not yet wired into the federation runtime.

| Trigger | Detection | Error | System response |
|---|---|---|---|
| Cumulative `(ε,δ)` would be exceeded by this round | `Accountant.would_exceed` before release | `PrivacyBudgetExceeded` | refuse release; stop training (fail-closed); budget NOT consumed for the refused round |
| Post-clip norm exceeds `C_clip` | assertion on clipped norm (`INV-DP-BOUND`) | defect (assertion failure), escalated as `PrivacyBudgetExceeded` only if the breach invalidates accounting | abort; the clip path is a correctness bug, not a privacy event ([RFC-0003 §9](RFC-0003-federated-protocol.md#9-failure-modes)) |
| `clip_norm <= 0` or `noise_multiplier < 0` in config | `DPConfig` validation at config load | `ConfigError` | reject the run before any round (validate-at-boundary) |
| `calibrate_sigma` infeasible (target `ε` too small for the round count) | accountant calibration | `ConfigError` | reject the run; remediation: raise `ε`, raise `δ`, lower round count, or accept more noise |
| `enabled=False` (no noise) on a run claiming privacy | manifest check | none at runtime; the `RunManifest` records `dp.enabled=False` | the run is reported as **non-private**; no `(ε,δ)` is claimed (an honesty guard, not an error) |
| Claim-grade mode uses coordinator-known or manifest-recorded noise entropy | privacy preflight | `ConfigError` | reject the claim-grade run or label an explicit deterministic replay mode non-private; record only the entropy-source policy |
| Participant CSPRNG unavailable in claim-grade mode | privacy preflight | `ConfigError` | reject before release; never fall back to the shared root seed |

Under the required integration, `PrivacyBudgetExceeded` is never caught-and-ignored; its `.remediation`
states the path (increase budget, reduce rounds, or accept the stop). It carries the spent `ε`, the
target `(ε,δ)`, and the round index.

## Alternatives Considered

- **Per-example DP-SGD (inner loop).** What it is: clip and noise every per-example (or per-micro-batch)
  gradient inside the `H` inner steps, giving example-level DP. Why considered: it is the standard, gives
  the finer-grained (and stronger) example-level guarantee, and has mature accounting (RDP for the
  subsampled Gaussian mechanism). Why rejected: it privatizes the wrong object — what crosses the
  boundary is the aggregate `H`-step delta, not individual examples — and it imposes per-step clipping
  cost across `H ∈ [50, 500]` steps at ViT-L scale every round. Update-level DP on the released `Δ_c`
  privatizes exactly the boundary-crossing object at one clip + one draw per round. The example-level
  guarantee is recoverable later as a stronger mode if a use case demands it, without changing the
  protocol.
- **Local DP (per-participant, large noise, no aggregation trust).** What it is: each participant adds
  enough noise that its released update is private *on its own*, requiring no trust in aggregation. Why
  considered: it removes the secure-aggregation trust assumption entirely. Why rejected: the noise needed
  for a meaningful local guarantee on a single update is large enough to swamp the small predictor delta
  and the SIGReg/anchor signal; utility collapses. Lensemble instead pairs *modest* per-participant noise
  with secure aggregation ([RFC-0011](RFC-0011-secure-aggregation.md)) so the aggregator never sees an
  individual update, getting a usable utility/privacy point.
- **Central DP only (trusted aggregator adds noise to the sum).** What it is: participants send raw
  clipped deltas; a trusted aggregator adds one noise draw to the sum. Why considered: minimum total
  noise for a target `(ε,δ)`. Why rejected for Phase 1: it requires the aggregator to see individual
  clipped deltas, contradicting the secure-aggregation requirement
  ([RFC-0003 §5](RFC-0003-federated-protocol.md#5-secure-aggregation-requirement)) and the honest-but-
  curious threat model. The chosen design — per-participant noise *before* masking — composes to the same
  aggregate noise as central DP when every participant noises with `σ` scaled by `1/√C`, while keeping
  every individual update hidden (distributed DP).
- **Gaussian mechanism vs. alternatives (Laplace, discrete Gaussian).** What it is: the choice of noise
  distribution. Why Gaussian: it is the natural mechanism for an L2-bounded sensitivity (the clip is in
  L2), composes cleanly under RDP/PRV, and is the basis of the federated-DP literature. Discrete Gaussian
  is reconsidered if the secure-aggregation modular arithmetic ([RFC-0011](RFC-0011-secure-aggregation.md))
  requires integer noise; that coupling is an Open Question deferred to RFC-0011's wire format. Laplace
  (L1 sensitivity) is rejected because the clip is L2.
- **RDP vs. PRV accountant.** What it is: two accounting methods for composing the subsampled/composed
  Gaussian mechanism. Why both are supported: RDP (the `opacus` default) is fast and has a mature,
  widely-used reference implementation but composition is slightly loose; PRV (privacy-loss-distribution) is tighter and is the preferred
  *reporting* backend. The `Accountant` protocol (§3) lets either back the same mechanism; the chosen
  backend is recorded in the `RunManifest`. Neither is rejected — RDP is the development default for
  speed, PRV the reporting default for tightness.

## Drawbacks

- **Utility loss, concentrated on the predictor.** The compact predictor delta is small relative to
  `C_clip`-scaled noise, so a fixed `σ` degrades `g_φ` more than the encoder `θ`. The noise also perturbs
  the SIGReg variance and the anchor term that pins the frame
  ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md)). This is the central tuning risk — the
  joint-calibration Open Question below, not resolved by a default.
- **DP and secure aggregation are complementary, not substitutes.** DP without secure aggregation lets
  the aggregator see individual noised updates; secure aggregation without DP hides individuals but
  leaves the *sum/model* leakage unbounded. The guarantee requires both; either alone is a weaker,
  clearly-stated point ([06 §Differential Privacy](../spec/06-security.md)).
- **The privacy unit relies on honest reporting of its contents** (the §2 / Phase-1 trust assumption);
  Phase 2 attestation ([RFC-0006](RFC-0006-verifiable-contribution.md)) reduces but does not eliminate it.
- **Accounting is over a *planned* round count.** In the intended lifecycle, `σ` is calibrated to
  `num_rounds`; running longer spends the budget and stops training. The current service lacks that
  cumulative stop, so a target run budget must not be claimed from its per-round reports.
- **Claim-grade entropy conflicts with bitwise replay of the noise draw.** Secret CSPRNG entropy cannot
  be published while remaining secret from the coordinator. Tests may inject deterministic seeds, but
  public claim evidence must reproduce mechanism parameters and accounting rather than the secret draw
  itself.

## Migration / Rollout

DP rolls out along the staged plan ([00 §8](../spec/00-overview.md#8-v10-scope-boundary),
[conventions §12](../spec/conventions.md#12-milestones-and-stages)):

- **v0.2 / Stage B — simulated federation, one cluster.** The clip/noise mechanism and stateful
  accountant primitives run in-process and have direct tests. The runtime noise is deterministically
  replayable from the shared, manifest-recorded root seed, so it is mechanism plumbing rather than a DP
  guarantee. The current Phase 3 reporting seam also constructs a fresh accountant after each committed
  round, so its `epsilon_spent` is a one-round value and its `rounds_accounted` is always one.
  Participant-secret entropy, persistent state, pre-release checking, and cumulative reporting remain
  to be integrated before this stage satisfies the RFC.
- **v0.3 / Stage C — two real sovereign nodes.** The same mechanism runs over the real boundary with the
  noise seed drawn from a participant-local CSPRNG (the §1 required entropy boundary), real secure
  aggregation
  ([RFC-0011](RFC-0011-secure-aggregation.md)), and the budget enforced as a hard stop. The privacy unit
  under participant churn (a participant present in a subset of rounds) is finalized here (Open Questions).

No artifact migration is needed: DP is a transform on `Δ_c` and a counter in the runtime; the
`PseudoGradient` schema ([03 §6](../spec/03-data-model.md#6-pseudogradient--the-one-private-object-that-does-cross-the-boundary))
is unchanged across stages. The DP parameters and the active accountant backend are part of the `privacy`
config group and the `RunManifest`; a pre-1.0 change to those is a manifest `schema_version` bump
([09 §Schema Versioning](../spec/09-release-and-versioning.md)).

## Testing Strategy

CPU-runnable tests on tiny synthetic deltas (no large downloads; cf.
[07 §ML-Specific Tests](../spec/07-testing-strategy.md)). Each test names the property and the invariant
it guards.

The mechanism tests use deterministic seed injection, and the accountant tests validate composition and
simulate check-before-release / consume-after-success in the test harness. They do not demonstrate a
secret entropy boundary, nor do they exercise the accounting lifecycle through
`Participant.local_round()` or `Phase3CoordinatorService`. Claim-grade gates still require an entropy
separation test and a multi-round integration test that exhausts a budget before release.

- **Clip-bound invariant (`INV-DP-BOUND`).** For adversarial input deltas with norms both above and below
  `C_clip`, assert the post-clip norm satisfies `‖Δ_c‖ ≤ C_clip` (within fp32 tolerance). Property test
  (`hypothesis`) over random delta shapes and clip norms. A norm below `C_clip` is unchanged; a norm
  above is scaled to exactly `C_clip`.
- **Clip determinism.** `clip_delta` returns bitwise-identical output on repeated calls with the same
  input; it is a pure function of `(delta, clip_norm)`.
- **Noise calibration (mechanism test only).** For a fixed `σ` and `C_clip`, draw many noise vectors
  with a seeded generator and assert the empirical per-coordinate variance is `σ²C_clip²` within a
  tolerance; assert the draw is reproducible from the injected test seed and *differs* across distinct
  (round, participant) seeds.
- **Claim-grade entropy separation (missing integration gate).** In claim-grade mode, assert the
  coordinator config, messages, reports, and manifests contain no participant noise seed or derivable
  entropy; assert deterministic replay mode is labeled non-private and cannot satisfy a privacy gate.
- **Accountant correctness vs. a reference.** For a known `(σ, num_rounds, sample_rate)`, assert
  `calibrate_sigma` and `spent` match a reference computation (an independent RDP/PRV calculation) within
  tolerance; assert PRV reports `ε` no larger than RDP for the same composition (tightness ordering).
- **Budget exhaustion raises `PrivacyBudgetExceeded` (accountant harness).** Configure a budget that the
  planned round count must exceed; assert `would_exceed` returns `True` at the right round and that the
  simulated lifecycle raises `PrivacyBudgetExceeded` without consuming budget for the refused round.
- **Check-before / consume-after ordering.** Simulate a round that aborts after the privacy check but
  before success; assert `step` was not called (budget not consumed by a failed round).
- **Config validation.** `DPConfig` with `clip_norm ≤ 0` or `noise_multiplier < 0` raises `ConfigError`
  at load; `calibrate_sigma` with an infeasible target raises `ConfigError`.
- **Alignment-order preflight (`INV-ALIGN-BEFORE-RELEASE`).** Assert a coordinator-side
  participant-specific backstop rejects DP-enabled or already-clipped updates, while a participant-local
  aligned displacement is clipped/noised normally.
- **Determinism-path non-interference (`INV-AGG-DETERMINISM`).** Privatize fixed deltas with recorded
  seeds, then run the outer step twice; assert the privatized-then-aggregated result is bitwise-identical
  (DP adds no nondeterminism to the reduction; cf.
  [RFC-0003 Testing](RFC-0003-federated-protocol.md#testing-strategy)).
- **Secure-agg / quantization interaction.** Assert noise is applied before masking and before int8
  quantization (order assertion); assert the summed independent noise has variance `C·σ²C_clip²` within
  tolerance.
- **Non-private honesty guard.** With `enabled=False`, assert the delta is unchanged and the `RunManifest`
  records the run as non-private (no `(ε,δ)` is claimed).

The DP-mechanism rungs of the ablation ladder (clip/noise configured vs. disabled at fixed parameters)
are realized as small-config integration tests per
[07 §Ablation Ladder](../spec/07-testing-strategy.md) and
[RFC-0005 §6](RFC-0005-evaluation.md). Runs that use deterministic replay entropy measure the mechanism's
utility effect; they are not private-vs-non-private claim evidence.

## Open Questions

OPEN QUESTION: The joint calibration of `(σ, λ_sig, λ_anc, C_clip)` — DP noise on the small predictor
delta interacts with the SIGReg variance statistic and the anchor term that closes the gauge, so the
utility/privacy/gauge sweet spot is not a default. Owner @AbdelStark; resolution: a Stage-B (v0.2)
sweep, shared with [RFC-0003 §Open Questions](RFC-0003-federated-protocol.md#open-questions) and
[RFC-0002 §7](RFC-0002-gauge-and-aggregation.md#7-the-central-hyperparameter), surfaced via the `dp/*` and `gauge/*` metrics
([05 §2.2](../spec/05-observability.md#22-gauge-metrics-gauge),
[05 §2.4](../spec/05-observability.md#24-privacy-metrics-dp)).

OPEN QUESTION: The exact definition of the privacy unit under participant churn — a participant present
in only a subset of rounds contributes fewer times, which changes its composed budget and may enable
privacy amplification by subsampling (the `sample_rate < 1.0` path). Owner @AbdelStark; resolution: Stage
C (v0.3), finalized alongside the churn/elasticity semantics in
[RFC-0013](RFC-0013-coordinator-runtime.md).

OPEN QUESTION: Whether the secure-aggregation wire format ([RFC-0011](RFC-0011-secure-aggregation.md))
forces a discrete-Gaussian noise mechanism (integer noise compatible with modular masking) instead of the
continuous Gaussian. Owner @AbdelStark; resolution: deferred to RFC-0011's wire-format decision, Stage C
(v0.3); if so, the accountant gains a discrete-Gaussian backend behind the same `Accountant` protocol.

OPEN QUESTION: The default reporting backend and the development/reporting split — whether to ship PRV as
the single backend once its calibration cost is acceptable, or keep RDP as the development default. Owner
@AbdelStark; resolution: Stage B (v0.2) once the accountant backends are benchmarked.

## References

- [RFC-0003 — Federated Training Protocol](RFC-0003-federated-protocol.md) (§3 the `PseudoGradient`
  contract; §4 the protocol-level clip-then-noise and its ordering; §5 the secure-aggregation requirement;
  §6 int8 quantization; §7 the aggregation-determinism contract; §9 failure modes).
- [RFC-0011 — Secure Aggregation Protocol](RFC-0011-secure-aggregation.md) (the masked sum DP protects;
  noise-before-masking interaction; the discrete-noise wire-format open question).
- [RFC-0002 — The Latent Gauge & Frame-Anchored Aggregation](RFC-0002-gauge-and-aggregation.md) (§4 the
  anchor/SIGReg terms DP noise perturbs; §7 the joint-calibration open question).
- [RFC-0008 — Model, Objective & Numerical Contracts](RFC-0008-model-objective-numerics.md) (the compact
  predictor whose small delta is noise-sensitive; the SIGReg statistic).
- [RFC-0009 — Configuration, Run Manifest & Reproducibility](RFC-0009-configuration-reproducibility.md)
  (the `privacy` config group, the recorded entropy-source policy, the `RunManifest`).
- [RFC-0013 — Coordinator & Participant Runtime](RFC-0013-coordinator-runtime.md) (where `privatize` is
  invoked and the persistent budget check must be integrated; churn semantics; the Stage-C noise-seed source).
- [RFC-0006 — Verifiable Contribution](RFC-0006-verifiable-contribution.md) (the Phase-2 attestation that
  reduces the honest-reporting trust assumption on the privacy unit).
- [03 — Data Model](../spec/03-data-model.md#6-pseudogradient--the-one-private-object-that-does-cross-the-boundary)
  (`PseudoGradient`) · [04 §5.5 Privacy](../spec/04-error-model.md#55-privacy-lensembleprivacy)
  (`PrivacyBudgetExceeded`) · [05 §2.4 Privacy metrics](../spec/05-observability.md#24-privacy-metrics-dp)
  (`dp/epsilon_cumulative`, `dp/clip_fraction`) · [06 — Security](../spec/06-security.md) (the DP
  guarantee and its honest limits).
- External: Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (Gaussian mechanism,
  composition); Abadi et al., *Deep Learning with Differential Privacy* (DP-SGD, moments accountant);
  Mironov, *Rényi Differential Privacy*; Gopi, Lee & Wutschitz, *Numerical Composition of DP via PRV*;
  Bonawitz et al., practical secure aggregation (distributed-DP pairing); `opacus` (the reference RDP
  accountant behind `privacy.accountant`).
