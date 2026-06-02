# RFC-0006 — Verifiable Contribution

| | |
|---|---|
| **RFC** | 0006 |
| **Title** | Verifiable Contribution |
| **Slug** | verifiable-contribution |
| **Status** | Draft · **Phase 2 (Deferred)** |
| **Track** | Standards |
| **Authors** | @AbdelStark |
| **Created** | 2026-06-02 |
| **Target milestone** | Proof-*ready* disciplines: `v0.1`/`v0.2`; the proofs themselves: post-`v1.0` (Stage D) |
| **Area** | verify |
| **Requires** | [RFC-0001](RFC-0001-architecture.md), [RFC-0002](RFC-0002-gauge-and-aggregation.md), [RFC-0003](RFC-0003-federated-protocol.md), [RFC-0004](RFC-0004-data-provenance.md), [RFC-0014](RFC-0014-provenance-commitments.md) |

> Verifiability is **Phase 2**: prove the federated-JEPA science first ([RFC-0002](RFC-0002-gauge-and-aggregation.md), [RFC-0005](RFC-0005-evaluation.md)). This RFC specifies the eventual cryptographic layer and — crucially — the small set of properties Phase 1 must satisfy so that adding proofs later requires **no rework**. The Phase-1 proof-*ready* disciplines (§3 below) are in scope for `v0.1`–`v1.0` and are tested now; the proofs (Stage D) are out of `v1.0` scope and tracked as future work per the [conventions §12](../spec/conventions.md#12-milestones-and-stages) milestone table. Verifiable contribution is the project's long-run differentiator, not Phase-1 scope.

## Summary

Existing decentralized-training networks hold themselves together with redundancy, attestation, and economic incentives — not with succinct cryptographic proof of the training computation. Lensemble's distinguishing claim is to move the trust model from **"trust via redundancy/economics"** to **"trust via cryptographic attestation of contribution."**

This RFC specifies four provable claims and the mechanism that backs each (§ Proposed Design, item 2): aggregation correctness via a Circle-STARK over the outer-step circuit (Stwo); update-from-committed-data via the Merkle commitment $R_c$ ([RFC-0014](RFC-0014-provenance-commitments.md)); frame-alignment correctness via **free public recomputation** of a deterministic function of the public probe and committed weights; and faithful inner-step execution via TEE attestation. The leverage is structural: the frame anchor of [RFC-0002](RFC-0002-gauge-and-aggregation.md) keeps the aggregation step a near-linear operation (a weighted average plus a Nesterov update), whose arithmetic circuit is small enough to prove. The gauge fix is therefore also the verifiability enabler — the same design decision that makes the science well-posed makes the proof cheap.

The deliverable for `v0.1`–`v1.0` is **proof-readiness**: deterministic aggregation, hash-committed model versions, Merkle-rooted episode commitments, a content-pinned public probe, and a reproducible outer step — five inexpensive disciplines that are the entire prerequisite for Phase 2. The proofs land in Stage D once prover capability and a STARK-friendly commitment hash are in place.

## Motivation

A federation of mutually-distrusting sovereign participants needs a trust story stronger than "we ran it twice and the numbers matched." Three trust questions arise once raw data never leaves a boundary (`INV-RESIDENCY`, enforced in `lensemble.data.residency`):

1. **Did the coordinator aggregate honestly?** A malicious or buggy coordinator could weight participants incorrectly, drop a participant's $\Delta_c$, or fabricate the committed global model. Under the Phase-1 honest-but-curious model this is assumed away; under a malicious-coordinator model it must be detectable.
2. **Did a participant compute its update from the data it claims?** Without binding, a participant can submit an update derived from data it never committed to, defeating contribution accounting and any downstream licensing or governance.
3. **Did the inner training step run faithfully?** The data-touching computation is the most expensive thing to prove and the hardest to attest.

Phase 1 answers (1) and (3) by assumption and (2) by tamper-evidence ([RFC-0004 §4](RFC-0004-data-provenance.md)). The purpose of this RFC is to (a) define the Phase-2 cryptographic answers, and (b) constrain Phase-1 engineering so those answers attach with no redesign. The cost of honoring the constraints now is low; the cost of retrofitting determinism, commitments, and a pinned probe after the fact is high. This is the connection between [RFC-0002](RFC-0002-gauge-and-aggregation.md) (the anchored frame keeps aggregation near-linear) and verifiability: closing the gauge is what makes the aggregation circuit cheap to prove.

This RFC makes cryptographic the Tapestry Evaluation-Certification and Data-Governance pillars, and connects to prior work on post-quantum verifiable AI ("The Half-Life of Trust") and composed assurance.

## Goals

- **G1.** Specify the four provable claims and the mechanism backing each, with the honest cost and trust assumption of each stated plainly (no overclaiming).
- **G2.** Specify the Phase-1 proof-*ready* requirements precisely enough that a contributor implementing [RFC-0003](RFC-0003-federated-protocol.md), [RFC-0010](RFC-0010-artifact-checkpoint-format.md), and [RFC-0014](RFC-0014-provenance-commitments.md) satisfies them without further coordination, and so that Phase 2 needs no rework.
- **G3.** Specify the Phase-1 **public recomputation** surface as a concrete, testable API (`recompute_alignment`) that reproduces the coordinator's frame alignment from public inputs alone, deterministically.
- **G4.** State the composed trust statement for each roadmap stage (2a–2d): which parts of a contribution are *proven*, which are *attested*, which are *assumed*.
- **G5.** Define the boundary plainly: what verifiable contribution does **not** prove (not data quality, not gradient honesty), so the guarantee is not misread.
- **G6.** Keep the Phase-2 commitment-hash migration path open (SHA-256 now; a STARK-friendly hash later) without breaking the Phase-1 commitment format.

## Non-Goals

- **Proving a full 1.2 B training step in zero knowledge.** Infeasible today; this RFC does not promise it. The inner step is covered by the weaker TEE assumption (item 2, claim 4), and full proof-of-training is research, deferred to roadmap 2d (Stage D).
- **Incentive / payment mechanisms, slashing, on-chain settlement, tokens.** Lensemble specifies the *attestation* layer only. Economic design layers on top and is deliberately separated. Out of scope for the entire v0.x–v1.0 line.
- **Proving data quality or gradient honesty.** Provenance proves *origin*, not goodness. No mechanism here certifies that a participant's data is useful or that its locally-computed gradient was the result of honest optimization.
- **Shipping any proof system in `v0.1`–`v1.0`.** Only the proof-*ready* disciplines (§ Proposed Design, item 3) ship in that line. The Stwo prover, TEE attestation, and the STARK-friendly hash migration are Stage D, post-`v1.0`, captured as future work per [conventions §12](../spec/conventions.md#12-milestones-and-stages) (not filed as implementable issues against v1.0).
- **Replacing secure aggregation or DP.** Verifiable contribution is orthogonal to confidentiality: secure aggregation ([RFC-0011](RFC-0011-secure-aggregation.md)) hides individual updates; DP ([RFC-0012](RFC-0012-differential-privacy.md)) bounds leakage; this RFC proves the aggregation of those (already-protected) updates was computed correctly.

## Proposed Design

### 1. Trust model

| Phase | Coordinator / aggregator | Participants | What protects raw data | What gives integrity |
|---|---|---|---|---|
| **Phase 1** | honest-but-curious | honest, may drop out | secure aggregation + DP ([RFC-0011](RFC-0011-secure-aggregation.md), [RFC-0012](RFC-0012-differential-privacy.md)) | provenance commitments ([RFC-0014](RFC-0014-provenance-commitments.md)) give tamper-*evidence*; aggregation integrity is *assumed* |
| **Phase 2** | may be malicious; misbehavior **detectable** | may misreport; binding makes it **detectable** | unchanged | aggregation correctness *proven* (STARK); provenance binding *proven* (Merkle); alignment *publicly recomputable*; inner step *TEE-attested* |

In Phase 1, honest contribution is *assumed*. In Phase 2, honest contribution becomes *verifiable*. The transition is additive: Phase-2 mechanisms attach to the Phase-1 commitments and determinism without changing the protocol of [RFC-0003](RFC-0003-federated-protocol.md). The security spec consolidates the residual trust assumptions per phase ([06-security.md](../spec/06-security.md)).

### 2. The provable surface (and what stays cheap)

Each claim is backed by exactly one mechanism, with its honest cost and the residual assumption stated.

| Claim | Mechanism | Cost / residual assumption |
|---|---|---|
| **Aggregation computed correctly** over the submitted deltas | **STARK over the outer-step circuit** (Stwo / Circle-STARK) | Tractable — and cheap *because* the anchored frame ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md)) keeps aggregation a plain weighted average / Nesterov step, i.e. a near-linear operation. This is the synergy: the gauge fix is also the verifiability enabler. Residual assumption: the STARK soundness (post-quantum, transparent setup). |
| **Update derived from committed data** | **Merkle commitment** $R_c$ + binding to the update ([RFC-0014 §3](RFC-0014-provenance-commitments.md), `INV-COMMIT-BINDING`) | Cheap. Proves provenance, **not** quality or honesty of the gradient. Residual assumption: collision resistance of the commitment hash (SHA-256 Phase 1; see item 6). |
| **Frame alignment correct** | **Public recomputation** — a deterministic function of the public probe + committed weights | **Free.** No proof needed; anyone recomputes and checks ([RFC-0002 §5](RFC-0002-gauge-and-aggregation.md)). Residual assumption: none beyond `INV-PROBE-PIN` and `INV-CHECKPOINT-HASH`. |
| **Inner training step executed faithfully** | **TEE attestation** (pragmatic proxy) | Moderate; a weaker trust than a SNARK but far cheaper than proving training. Residual assumption: TEE hardware/firmware trust. |

The four claims compose into a single statement per round (item 5): *the global model at round $t{+}1$ is the deterministic aggregate of $C$ updates, each bound to a committed dataset root, each produced inside an attested enclave, with the frame alignment publicly verifiable.* The strength of that statement is exactly the weakest link, which is why each link's residual assumption is named.

#### 2.1 The synergy with the gauge (why aggregation stays provable)

The outer step ([RFC-0002 §8, step 5](RFC-0002-gauge-and-aggregation.md)) is

$$(\theta_{t+1}, \phi_{t+1}) \leftarrow (\theta_t, \phi_t) - \eta_{\text{out}} \cdot \mathrm{OuterOpt}\big(\mathrm{mean}_c\,\Delta_c\big),$$

a weighted average of committed deltas followed by a fixed Nesterov update. Were participants in mutually-rotated coordinate frames (the $O(d)$ gauge of [RFC-0002 §2](RFC-0002-gauge-and-aggregation.md)), averaging would require a per-participant orthogonal alignment $Q_c$ *inside* the aggregation — turning a linear reduction into a sequence of SVDs and matrix products that is expensive to express as an arithmetic circuit. Because the warm-start + frame anchor keep every participant in the same frame, the Layer-3 Procrustes backstop ([RFC-0002 §5](RFC-0002-gauge-and-aggregation.md)) rarely binds, and when it does its alignment is itself publicly recomputable (free, claim 3) rather than something the STARK must internalize. The aggregation circuit the STARK proves is therefore the near-linear average + Nesterov step only. **Closing the gauge keeps the proof cheap.**

### 3. Phase-1 proof-ready requirements (cheap to honor now)

Phase 1 MUST bake in the following five disciplines. Each maps to an existing RFC and an `INV-*` invariant ([conventions §7](../spec/conventions.md#7-named-invariants)), so the requirement is enforced and tested as part of normal Phase-1 work, not as Phase-2 speculation.

1. **Deterministic aggregation.** The outer step ([RFC-0003 §7](RFC-0003-federated-protocol.md)) is a fixed, bitwise-reproducible function of (committed deltas, round seed, prior global params): fixed reduction order, fp32 with fixed summation order (or fp64), no atomics, no nondeterministic reductions on the aggregation path (`INV-AGG-DETERMINISM`, [conventions §9](../spec/conventions.md#9-determinism-dtype-device)). A determinism self-check runs each outer step in `lensemble.aggregation`; on mismatch it raises `NonDeterministicAggregation` (never swallowed; [conventions §6](../spec/conventions.md#6-error-taxonomy)) and the round aborts to the `ABORTED` path of the runtime state machine ([RFC-0013](RFC-0013-coordinator-runtime.md)), to be recomputed.
2. **Committed model versions.** Every $(\theta_t, \phi_t)$ artifact is hash-committed: its content hash equals the hash in the `Commitment` / `RoundClose` message (`INV-CHECKPOINT-HASH`), with a `parent_hash` chain ([RFC-0010 § Proposed Design](RFC-0010-artifact-checkpoint-format.md)). The canonical Phase-1 hash is SHA-256 over a canonical byte serialization ([conventions §11](../spec/conventions.md#11-external-dependencies)). On mismatch the artifact loader raises `CheckpointIntegrityError`; an unknown/too-new schema raises `SchemaVersionMismatch`.
3. **Episode hashing + Merkle roots from day one.** Each participant content-hashes its episodes and commits a Merkle root $R_c$ ([RFC-0014](RFC-0014-provenance-commitments.md), `DatasetCommitment`) even before proofs exist. Every released $\Delta_c$ is bound to exactly one $R_c$ (`INV-COMMIT-BINDING`); a mismatch raises `CommitmentMismatch` (never swallowed) and the update is rejected ([RFC-0003 §9](RFC-0003-federated-protocol.md)).
4. **Pinned public probe.** The probe content hash and landmark targets are pinned ([RFC-0004 §3](RFC-0004-data-provenance.md), `INV-PROBE-PIN`); landmark targets $t_i$ derive only from the round-0 reference encoder $f_{\text{ref}}$. The probe hash a participant uses must equal the hash committed in `RoundOpen`; a mismatch raises `ProbeError`. Alignment must be recomputable from public inputs alone (item 4 below).
5. **Reproducible outer step.** Fixed seeds and fixed order so a verifier can recompute the claimed aggregation. The round sketch seed is derived deterministically, $s_t = \mathrm{derive}(\text{root\_seed}, t)$ ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)), and recorded in the `RunManifest` ([RFC-0009](RFC-0009-configuration-reproducibility.md)); all participants in round $t$ use the identical projection matrix $A$ from $s_t$ (`INV-SKETCH-CONSISTENCY`). The warm-start is hash-identical across participants at round 0 (`INV-WARMSTART-T0`), closing the gauge at $t{=}0$.

These five are inexpensive engineering disciplines in Phase 1 and are the entire prerequisite for Phase 2. They are tested now (item § Testing Strategy); none is deferred.

### 4. Public recomputation (Phase-1, free, in scope now)

The only Phase-2 *mechanism* that ships in Phase 1 is the free one: public recomputation of the frame alignment. It is a deterministic, side-effect-free function of public data, exposed under `lensemble.verify`:

```python
# lensemble/verify/recompute.py — Phase 1, shipped in v0.1/v0.2.

from pathlib import Path
from lensemble.gauge import FrameDriftReport  # ([conventions §5](../spec/conventions.md#5-public-api-surface))

def recompute_alignment(
    checkpoint: Path,          # committed (θ_t, φ_t) artifact; hash-verified on load (INV-CHECKPOINT-HASH)
    probe: Path,               # pinned public probe P; content-hash checked vs round commitment (INV-PROBE-PIN)
    *,
    reference: Path,           # round-0 reference encoder f_ref (for landmark targets / E_ref)
    expected: "AlignmentClaim | None" = None,  # the coordinator's claimed alignment, if checking
) -> "AlignmentRecomputation": ...
```

```python
# Result schema (pydantic v2 on-disk record; schema_version: int).

class AlignmentClaim(BaseModel):           # what the coordinator published for round t, per participant pair
    schema_version: int
    round_index: int
    participant_a: str
    participant_b: str
    procrustes_q_hash: str                 # SHA-256 of canonical bytes of Q* (the claimed O(d) alignment)
    procrustes_residual: float             # ‖f_θ(P)·Q* − E_ref‖_F, the residual (RFC-0002 §5)
    rotation_angle_deg: float              # mean principal rotation angle of Q* on P

class AlignmentRecomputation(BaseModel):
    schema_version: int
    round_index: int
    probe_hash: str                        # must equal the RoundOpen commitment, else ProbeError
    recomputed: AlignmentClaim             # locally recomputed from public inputs alone
    matches_expected: bool | None          # None if `expected` was not supplied
    max_abs_residual_delta: float | None   # |recomputed.residual − expected.residual|, within tolerance?
```

Semantics and contract:
- **Determinism.** `recompute_alignment` is bitwise-deterministic given (committed weights, pinned probe, reference encoder). It runs the same closed-form Procrustes alignment $Q^\star = VU^\top$ from the SVD $E_{\text{ref}}^\top f_\theta(\mathcal{P}) = U\Sigma V^\top$ ([RFC-0002 §4 Variant B](RFC-0002-gauge-and-aggregation.md) / [§5](RFC-0002-gauge-and-aggregation.md)) on the fp32 / fp64 path ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)). Anyone with the public probe and the committed weights reproduces the coordinator's alignment and checks it; **no ZK proof is required** ([RFC-0002 §5](RFC-0002-gauge-and-aggregation.md)).
- **Preconditions.** The checkpoint hash is verified on load (`INV-CHECKPOINT-HASH`); the probe content hash is checked against the round commitment (`INV-PROBE-PIN`). A near-degenerate SVD raises `DegenerateProcrustes`; the routine clamps/conditions singular values per [RFC-0002 §4](RFC-0002-gauge-and-aggregation.md) before reporting.
- **Errors raised** ([conventions §6](../spec/conventions.md#6-error-taxonomy)): `CheckpointIntegrityError` (hash mismatch on the committed weights), `ProbeError` (probe hash mismatch / under-coverage), `DegenerateProcrustes` (ill-conditioned alignment), `SchemaVersionMismatch` (unknown/too-new claim record).
- **CLI surface.** `lensemble verify recompute --checkpoint <path> --probe <path> --reference <path> [--expected <claim.json>]` emits an `AlignmentRecomputation` JSON and exits non-zero if `matches_expected is False`. The companion `lensemble verify prove` is a **Phase-2 stub** that raises `NotImplementedError` with a remediation string pointing to Stage D ([conventions §5](../spec/conventions.md#5-public-api-surface)).

### 5. The composed trust statement (per roadmap stage)

Each Stage-D substage adds one mechanism and tightens the composed statement. State which parts are *proven* (P), *attested* (A), or *assumed* (Z):

| Substage | Aggregation | Provenance binding | Frame alignment | Inner step | Composed statement |
|---|---|---|---|---|---|
| **Phase 1** (`v0.1`–`v1.0`) | Z (deterministic + recomputable) | Z (tamper-evident) | P (free public recomputation) | Z | "Honest if the coordinator and participants are honest; any deviation in aggregation or alignment is *detectable* by recomputation." |
| **2a** | **P** (STARK) | Z | P | Z | "Aggregation provably correct over submitted deltas; provenance and inner step assumed." |
| **2b** | P | **P** (Merkle binding) | P | Z | "Each update provably bound to a committed dataset root; aggregation provably correct; inner step assumed." |
| **2c** | P | P | P | **A** (TEE) | "Inner step attested by enclave; aggregation and binding proven; alignment recomputable. The full composed statement." |
| **2d** *(research)* | P | P | P | toward **P** | "Push the inner step from attested toward proven as prover capability improves." |

### 6. Commitment-hash migration (open, deferred)

The canonical Phase-1 commitment hash is **SHA-256** (conservative, interoperable; [conventions §11](../spec/conventions.md#11-external-dependencies)). A Circle-STARK over the aggregation circuit (substage 2a) is far cheaper when commitments use a STARK-friendly hash (e.g. Poseidon2). The Phase-1 commitment format ([RFC-0014](RFC-0014-provenance-commitments.md), [RFC-0010](RFC-0010-artifact-checkpoint-format.md)) records the hash function as a versioned choice so the migration is a `schema_version` bump rather than a format break; readers reject an unknown hash id with `SchemaVersionMismatch`. The migration itself is an Open Question (below), owned by @AbdelStark, resolved in Stage D.

### 7. Roadmap

- **2a** — aggregation-correctness STARK (Stwo) over the committed deltas → committed global model.
- **2b** — provenance binding: tie each update to its dataset Merkle root ([RFC-0014](RFC-0014-provenance-commitments.md)).
- **2c** — TEE attestation for the inner step; define the composed trust statement (item 5) over which parts are proven vs attested vs assumed.
- **2d** *(research)* — push toward heavier proof-of-training / proof-of-learning as prover capability improves; benchmark prover cost (Stwo) against circuit size as the model scales from ViT-L toward the 1.2 B target.

### 8. Failure modes (Phase-1 surface)

| Trigger | Detection | Error raised ([conventions §6](../spec/conventions.md#6-error-taxonomy)) | System response |
|---|---|---|---|
| Outer step not bitwise reproducible | per-step determinism self-check ([RFC-0003 §7](RFC-0003-federated-protocol.md)) | `NonDeterministicAggregation` | abort round to `ABORTED`; recompute; never swallowed (security-critical) |
| Recomputed alignment ≠ coordinator's claim | `recompute_alignment(..., expected=...)` returns `matches_expected=False` | none (returns a record) | CLI exits non-zero; the claim is rejected by the verifier; surfaced to observability ([RFC-0015](RFC-0015-observability-diagnostics.md)) |
| Probe hash ≠ `RoundOpen` commitment | `INV-PROBE-PIN` check on load | `ProbeError` | reject; recomputation aborts |
| Committed weights hash mismatch | `INV-CHECKPOINT-HASH` check on load | `CheckpointIntegrityError` | reject artifact; do not aggregate |
| $\Delta_c$ not bound to a single $R_c$ | binding check ([RFC-0014](RFC-0014-provenance-commitments.md)) | `CommitmentMismatch` | reject update; never swallowed (security-critical) |
| Near-degenerate SVD in alignment | conditioning check ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md)) | `DegenerateProcrustes` | clamp/condition; report; if still ill-conditioned, fail closed |
| `lensemble verify prove` invoked in Phase 1 | stub guard | `NotImplementedError` (remediation → Stage D) | exit non-zero with a clear message |

## Alternatives Considered

**Trust via redundancy / economics (existing decentralized-training networks).**
*What it is:* hold the system together with replicated computation, attestation heuristics, and economic incentives (staking, slashing, rewards).
*Why considered:* it is the deployed state of the art for decentralized training and requires no advanced cryptography.
*Why rejected (as the trust root):* redundancy proves nothing succinctly — it raises cost linearly and still trusts the majority; economics deters but does not detect a correctly-incentivized-but-dishonest aggregator. Lensemble layers economics *on top* (Non-Goals) but roots trust in cryptographic attestation of the aggregation and provenance binding, which is detectable rather than merely disincentivized.

**SNARK vs Circle-STARK (Stwo) for the aggregation proof.**
*What it is:* a succinct proof of the outer-step circuit; SNARKs (pairing-based, trusted setup) vs transparent STARKs (FRI/Circle-STARK, no trusted setup, post-quantum).
*Why considered:* SNARKs have smaller proofs and faster verification.
*Why rejected (for now):* SNARKs require a trusted setup (a trust assumption Lensemble is trying to remove) and are not post-quantum; the "Half-Life of Trust" argument favors transparent, post-quantum proofs for a long-lived federation. Circle-STARK (Stwo) is transparent and post-quantum, and the near-linear aggregation circuit (item 2.1) is exactly the regime where a STARK is practical. The choice is revisited in 2d if prover cost dominates.

**Prove everything vs the layered model.**
*What it is:* a single end-to-end proof of the entire round (inner training + aggregation) vs the layered "TEE-attested inner step + commitment-bound provenance + STARK-proven aggregation."
*Why considered:* a single proof would give the strongest statement.
*Why rejected:* proving a full 1.2 B training step in zero knowledge is infeasible today (Non-Goals). The layered model puts the cheapest sufficient mechanism behind each claim — free recomputation for alignment, a small STARK for the near-linear aggregation, a Merkle binding for provenance, and a TEE for the heavy data-touching inner step — and composes them into one honest statement (item 5). It degrades gracefully: each substage tightens one link.

**TEE-only (attest the whole round) vs proof-backed aggregation.**
*What it is:* run the entire round inside an enclave and rely on remote attestation for everything.
*Why considered:* simplest to deploy; one trust primitive.
*Why rejected (as the sole mechanism):* a TEE is a weaker assumption than a STARK (it trusts hardware/firmware and is vulnerable to side channels), and attestation does not give a third party a transferable, post-quantum proof. Lensemble reserves the TEE for the inner step (where proof is infeasible) and uses a STARK for the aggregation (where proof is cheap), so the strongest available guarantee backs each claim.

## Drawbacks

- **Proving full training in ZK is infeasible today.** The inner step rests on the TEE assumption (substage 2c), which is strictly weaker than a SNARK/STARK. We state this rather than imply a proof-of-training that does not exist.
- **Provenance proves origin, not honesty.** The Merkle binding attests *which committed dataset* an update came from — a contribution-accounting and licensing guarantee — **not** that the data is good or the gradient was honestly computed. Stating this boundary is part of the design's integrity ([RFC-0004 §4](RFC-0004-data-provenance.md), [RFC-0014](RFC-0014-provenance-commitments.md)).
- **TEE trust.** TEE attestation depends on hardware and firmware trust and a current attestation root; it is a pragmatic proxy, not a cryptographic proof of computation.
- **Hash migration debt.** SHA-256 is conservative but not STARK-friendly; the eventual migration to a STARK-friendly hash (item 6) is real work, deferred to Stage D, and could force re-commitment of historical roots if not designed as a versioned, additive change.
- **Phase-1 cost of discipline.** Bitwise-deterministic aggregation forecloses some fast nondeterministic reductions on the aggregation path ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)); the performance budget ([08-performance-budget.md](../spec/08-performance-budget.md)) accounts for this as a deliberate constraint, justified by proof-readiness.

## Migration / Rollout

The rollout is the roadmap (item 7), gated stage by stage. The Phase-1 disciplines (item 3) land first and are non-negotiable for `v0.1`–`v1.0`:

- **`v0.1` (Stage A) and `v0.2` (Stage B):** the five proof-ready requirements (deterministic aggregation, committed model versions, Merkle roots, pinned probe, reproducible outer step) ship and are tested. The free public-recomputation mechanism (`recompute_alignment`, `lensemble verify recompute`) ships. `lensemble verify prove` is a stub that raises `NotImplementedError`.
- **`v0.3` (Stage C):** the disciplines operate over a real network boundary with real secure aggregation and DP; the `ContributionLedger` ([RFC-0014](RFC-0014-provenance-commitments.md)) is populated. No proofs yet.
- **`v1.0`:** proof-*ready* guarantees verified end-to-end (this RFC's item 3, the [conventions §12](../spec/conventions.md#12-milestones-and-stages) v1.0 row): the determinism check, the public recomputation, the commitment binding, and the pinned probe are all exercised in the reproducibility package.
- **Post-`v1.0` (Stage D, Phase 2):** substages 2a → 2b → 2c → 2d, each additive and each gating the next. The commitment-hash migration (item 6) is sequenced before 2a so the STARK circuit is cheap. These are tracked as future work ([conventions §12](../spec/conventions.md#12-milestones-and-stages)), not filed as v1.0 issues.

No Phase-2 substage changes the [RFC-0003](RFC-0003-federated-protocol.md) protocol or the [RFC-0014](RFC-0014-provenance-commitments.md) commitment format beyond a versioned hash-id bump; that is the meaning of "no rework."

## Testing Strategy

Phase-1 proof-readiness is in scope **now**; these tests run in CI (CPU, tiny synthetic fixtures — [conventions §9](../spec/conventions.md#9-determinism-dtype-device)):

- **Aggregation determinism (`INV-AGG-DETERMINISM`).** Run the outer step twice on identical inputs (committed deltas, round seed, prior global params); assert bitwise-equal results. Inject a nondeterministic reduction and assert `NonDeterministicAggregation` is raised and the round aborts ([RFC-0003](RFC-0003-federated-protocol.md), [RFC-0011](RFC-0011-secure-aggregation.md) interaction). Exact equality required.
- **Public alignment recomputation.** `recompute_alignment` reproduces the coordinator's alignment from the public probe + committed weights to within the fp32/fp64 tolerance; assert `matches_expected=True` for an honest claim and `False` for a perturbed claim. Determinism: two independent recomputations on different processes/platforms agree bitwise on `procrustes_q_hash` (this is also the [RFC-0015](RFC-0015-observability-diagnostics.md) diagnostic-reproducibility test).
- **Commitment binding (`INV-COMMIT-BINDING`).** An update bound to the wrong root is rejected with `CommitmentMismatch`; the correct binding round-trips ([RFC-0014](RFC-0014-provenance-commitments.md)).
- **Pinned-probe (`INV-PROBE-PIN`).** A probe whose content hash differs from the `RoundOpen` commitment raises `ProbeError`; landmark targets derive only from $f_{\text{ref}}$.
- **Checkpoint integrity (`INV-CHECKPOINT-HASH`).** Tampering with committed weights is detected as `CheckpointIntegrityError` on load before aggregation ([RFC-0010](RFC-0010-artifact-checkpoint-format.md)).
- **Degenerate alignment.** A synthetically rank-deficient probe embedding raises `DegenerateProcrustes` (or is clamped within tolerance), never silently producing a garbage $Q^\star$ ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md)).
- **Stub guard.** `lensemble verify prove` raises `NotImplementedError` with a Stage-D remediation string; the CLI exits non-zero.
- **CLI exit-code contract.** `lensemble verify recompute` exits zero on an honest claim, non-zero on a mismatched one (consumable by CI / a verifier).

Phase-2 proof tests (Stwo circuit correctness, TEE attestation round-trips, Poseidon2 hash equivalence) are Stage D and are not in the `v0.1`–`v1.0` suite.

## Open Questions

OPEN QUESTION: Migrate the Phase-1 commitments from SHA-256 to a STARK-friendly hash (e.g. Poseidon2) to keep the substage-2a proof circuit cheap, without breaking the Phase-1 commitment format. Owner: @AbdelStark. Resolution path: a versioned hash-id field in [RFC-0014](RFC-0014-provenance-commitments.md) / [RFC-0010](RFC-0010-artifact-checkpoint-format.md) commitment headers (shared with [RFC-0014](RFC-0014-provenance-commitments.md) and the [conventions §11](../spec/conventions.md#11-external-dependencies) open question); decided in Stage D before substage 2a.

OPEN QUESTION: Prover cost vs circuit size as the model scales from ViT-L toward the 1.2 B target — does the Stwo aggregation proof stay tractable when $\Delta_c$ grows, or must the aggregation be proven over a compressed/quantized delta (interacting with the int8 pseudo-gradient quantization of [RFC-0003](RFC-0003-federated-protocol.md))? Owner: @AbdelStark. Resolution path: benchmarked in roadmap substage 2d (Stage D, research).

OPEN QUESTION: The TEE attestation root and the precise inner-step measurement to attest (whole inner loop vs a measured training kernel) for substage 2c — what is attested, and how is the attestation bound to the committed $(\theta_t, \phi_t)$ and $R_c$? Owner: @AbdelStark. Resolution path: follow-up RFC superseding this section's item 5, scoped in Stage D.

OPEN QUESTION: When the Layer-3 Procrustes backstop ([RFC-0002 §5](RFC-0002-gauge-and-aggregation.md)) does bind, does its alignment stay outside the STARK circuit (publicly recomputed, claim 3) or must it be folded into the proven aggregation? Owner: @AbdelStark. Resolution path: determined by the Stage-B frame-drift sweep (how often the backstop fires; [RFC-0002 §7](RFC-0002-gauge-and-aggregation.md), [RFC-0005](RFC-0005-evaluation.md)) and finalized in substage 2a.

RISK: The whole Phase-2 value proposition rests on the aggregation circuit staying near-linear (item 2.1), which in turn rests on the frame anchor keeping the Layer-3 backstop quiet. If the gauge proves hard to control at video-WM scale ([RFC-0002 §7](RFC-0002-gauge-and-aggregation.md) and Open Questions), the aggregation circuit grows and the cheap-proof claim weakens. Resolution plan: the Stage-B frame-drift sweep quantifies backstop frequency before any Stage-D proof work begins; if drift is severe, the Fork-A fallback ([RFC-0002 — Fork A fallback](RFC-0002-gauge-and-aggregation.md#fork-a-fallback)) dissolves the gauge entirely and the aggregation is trivially linear (a frozen-encoder predictor average), preserving cheap aggregation proofs at the cost of the end-to-end novelty.

## References

- [RFC-0001 — Architecture & System Overview](RFC-0001-architecture.md): trust boundaries (§6), module map; the committed-model-version requirement.
- [RFC-0002 — The Latent Gauge & Frame-Anchored Aggregation](RFC-0002-gauge-and-aggregation.md): the $O(d)$ gauge (§2), frame anchoring (§4), Procrustes re-alignment and the public-recomputation bonus (§5), the central hyperparameter (§7), the per-round algorithm (§8) — the verifiability synergy (anchored frame ⇒ near-linear aggregation ⇒ cheap to prove).
- [RFC-0003 — Federated Training Protocol](RFC-0003-federated-protocol.md): deterministic outer step, the message table (§8), the commitment hook.
- [RFC-0004 — Data, Sovereignty & Provenance](RFC-0004-data-provenance.md): the public probe (§3), provenance commitments and the origin-not-quality boundary (§4), contribution accounting (§5).
- [RFC-0005 — Evaluation & Benchmark Protocol](RFC-0005-evaluation.md): the frame-drift diagnostic and the $\lambda_{\text{anc}}$ sweep that bounds backstop frequency.
- [RFC-0010 — Checkpoint & Artifact Format](RFC-0010-artifact-checkpoint-format.md): SHA-256 content hashing, `parent_hash` chain, `INV-CHECKPOINT-HASH`.
- [RFC-0011 — Secure Aggregation Protocol](RFC-0011-secure-aggregation.md): the determinism of the revealed sum under masking.
- [RFC-0012 — Differential Privacy Accounting](RFC-0012-differential-privacy.md): per-participant DP, orthogonal to verifiability.
- [RFC-0013 — Coordinator & Participant Runtime](RFC-0013-coordinator-runtime.md): the `ABORTED` round path on a determinism-check failure.
- [RFC-0014 — Provenance Commitments & Merkle Scheme](RFC-0014-provenance-commitments.md): episode hashing, Merkle root $R_c$, binding (`INV-COMMIT-BINDING`), the `ContributionLedger`, the versioned hash-id.
- [RFC-0015 — Observability, Diagnostics & Telemetry](RFC-0015-observability-diagnostics.md): the deterministic frame-drift diagnostic emission, reproducible from committed weights + probe.
- Security spec ([06-security.md](../spec/06-security.md)): the consolidated residual trust assumptions per phase.
- Performance budget ([08-performance-budget.md](../spec/08-performance-budget.md)): the deliberate cost of deterministic aggregation.
- Stwo (Circle-STARK prover), [conventions §11](../spec/conventions.md#11-external-dependencies): the Phase-2 aggregation-correctness prover.
- "The Half-Life of Trust" (post-quantum verifiable AI): motivation for transparent, post-quantum proofs.
- Tapestry Evaluation-Certification and Data-Governance pillars: the trust model made cryptographic.
