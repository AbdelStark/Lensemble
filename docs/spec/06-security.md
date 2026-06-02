# 06 — Security

This section specifies Lensemble's threat model, trust boundaries, and the mechanisms that enforce
data sovereignty. It is the stable security contract; the RFCs it cites hold the rationale and the
protocol detail. Lensemble's security posture is layered and honest about its limits: Phase 1 protects
*confidentiality of raw data* against an honest-but-curious infrastructure and gives *tamper-evidence*
of contributions; Phase 2 ([RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md)) upgrades selected
claims to *cryptographic attestation* against a malicious coordinator and misreporting participants.
Every guarantee below states what it does and does not prove, and which residual trust assumptions
remain at each phase.

Normative scope: this document governs what crosses a trust boundary and what must never cross. The
controlling invariant is `INV-RESIDENCY` ([conventions §7](conventions.md#7-named-invariants)): no raw observation, action, or private-data
embedding is serialized into any outbound message or artifact. Where this document and an RFC appear to
disagree, the [conventions document](conventions.md) and the named invariant win; report the discrepancy as an `OPEN QUESTION:`.

## 1. Threat model

Lensemble is a federation of mutually distrusting participants coordinated by a partially trusted
coordinator. The asset to protect, in priority order, is: (1) the confidentiality of each
participant's raw trajectory data and its embeddings; (2) the integrity and provenance of the
contributions aggregated into the global model; (3) the availability of the training round under
churn. Confidentiality is paramount — it is the precondition for sovereign participation.

### 1.1 Actors and their trust level

| Actor | Phase 1 trust | Phase 2 trust | Holds |
|---|---|---|---|
| Participant $c$ | Honest-but-curious toward peers; trusted with its own data | May misreport its $\Delta_c$ (detectable) | Sovereign dataset, local encoder/predictor weights, per-embodiment head $h_\psi^{(c)}$, DP/mask secrets |
| Coordinator | Honest-but-curious; trusted to run the outer step | May be malicious (aggregation correctness provable) | Canonical global model $(\theta_t,\phi_t)$, round seed $s_t$, probe hash, contribution ledger |
| Secure aggregator | Honest-but-curious; learns only $\sum_c \Delta_c$ | Same, plus aggregation attestable | Masked updates; never an individual $\Delta_c$ |
| Public verifier | Untrusted; given only public inputs | Verifies STARK + recomputes alignment | Public probe $\mathcal{P}$, committed model hashes, dataset roots $R_c$ |

The coordinator is **untrusted with respect to raw data in every phase** — sovereignty does not depend
on coordinator honesty, only on residency enforcement and secure aggregation
([RFC-0003 §2](../rfcs/RFC-0003-federated-protocol.md)). The coordinator's *honesty about the
aggregation arithmetic* is assumed in Phase 1 and made verifiable in Phase 2.

### 1.2 Attacker capabilities (in scope)

Phase 1 (honest-but-curious infrastructure):

- The coordinator/aggregator inspects every message it legitimately receives and attempts to infer a
  participant's private data from the messages it sees (`RoundOpen`, `Update`, `Commitment`,
  `RoundClose`; §3). It follows the protocol but is curious.
- A peer participant attempts to infer another silo's data from the shared global model, the public
  probe, or any leaked individual update.
- A passive network observer reads boundary-crossing messages in transit.
- An accidental code path attempts to serialize a raw observation, action, or private embedding into
  an outbound message, log, or artifact (the residency-breach failure mode; §4).

Phase 2 (additional, malicious):

- A malicious coordinator computes the outer step incorrectly, drops or reweights a participant's
  contribution, or substitutes weights — detectable via the aggregation-correctness STARK
  ([RFC-0006 §2](../rfcs/RFC-0006-verifiable-contribution.md#2-the-provable-surface-and-what-stays-cheap)).
- A participant misreports the dataset its update was computed from — detectable via provenance
  binding ($R_c$ bound to $\Delta_c$, `INV-COMMIT-BINDING`).
- A participant submits an update that does not match its committed weights/round — detectable via the
  public recomputation of alignment and the committed-version hash chain.

### 1.3 Out of scope

- **Gradient honesty and data quality.** Provenance proves *origin* (which committed dataset produced
  an update), not that the data is good or the local gradient honestly computed
  ([RFC-0004 §4](../rfcs/RFC-0004-data-provenance.md#4-provenance-commitments-the-bridge-to-phase-2),
  [RFC-0006 §4](../rfcs/RFC-0006-verifiable-contribution.md)). A participant that
  trains on garbage-but-committed data is not caught by any cryptographic mechanism here. Quality
  gating beyond declared metadata is out of v0.1 scope (RFC-0004 §7).
- **Faithful execution of the full inner training step.** Proving a full 1.2B training step in zero
  knowledge is infeasible today (RFC-0006 §4). Phase 2 attests the inner step with a TEE (a weaker
  assumption than a SNARK), not a proof.
- **Membership-inference and model-inversion attacks on the released global model** beyond the
  protection differential privacy provides; DP bounds per-participant-round leakage but does not make
  the model attack-proof (§6).
- **Byzantine robustness of aggregation** (poisoning that stays within the committed dataset). Secure
  aggregation hides individual updates but does not by itself reject a malicious-but-well-formed update;
  robust-aggregation defenses are future work.
- **Incentives, payments, slashing, and on-chain settlement** (RFC-0006 §7) — the attestation layer is
  specified; economic design layers on top and is deliberately separated.
- **Physical/side-channel attacks on a participant's own hardware**, and compromise of a participant's
  own training process before residency enforcement runs.
- **TEE hardware compromise** in Phase 2 (the TEE root of trust is assumed).

## 2. Trust boundaries

The trust boundary is the participant's sovereignty perimeter. Raw data lives inside it and never
leaves; only privacy-protected model deltas, dataset commitments, and shared coordination state cross.
The following diagram reproduces the architecture trust-boundary model
([RFC-0001 §6](../rfcs/RFC-0001-architecture.md#6-trust-boundaries)).

```
┌── Participant c (sovereign) ─────────────────────────┐
│  raw trajectories  ──►  local train (inner-parallel) │
│        │ (never leaves)         │                     │
│        ▼                        ▼                     │
│  Merkle commitment R_c     pseudo-gradient Δ_c        │
└───────────────│──────────────────│───────────────────┘
                │                  │  (DP-clipped + noised, then masked)
                ▼                  ▼
        ┌──────────── Coordinator / secure aggregator ─────────┐
        │  Σ_c Δ_c  (individual Δ_c never revealed)             │
        │  outer Nesterov step → θ^{global}_{t+1} (hash-committed)│
        │  frame re-alignment on public probe (recomputable)    │
        └───────────────────────────────────────────────────────┘
```

In prose: inside participant $c$'s boundary, raw $(o_t, a_t, o_{t+1})$ trajectories drive local
training under intra-participant parallelism. Two artifacts and nothing else leave the boundary — the
dataset Merkle root $R_c$ (a commitment, not data) and the pseudo-gradient $\Delta_c$, the latter only
after DP clipping and noising and then masking for secure aggregation. The coordinator/aggregator
observes only the masked sum $\sum_c \Delta_c$, applies the deterministic outer Nesterov step, commits
the resulting global weights by hash, and recomputes the frame alignment on the *public* probe (which a
third party can reproduce). What crosses: model deltas (privacy-protected), dataset commitments, and
shared coordination state (sketch seed $s_t$, probe hash, global-model hash). What never crosses: raw
observations, actions, or embeddings of private data, and the per-embodiment action head $h_\psi^{(c)}$
(`INV-ACTIONHEAD-LOCAL`; it is never broadcast or aggregated, so it is never a leakage surface).

### 2.1 Boundary-crossing message table

Every message that crosses the sovereignty boundary, its direction, contents, and the protection
applied. This reproduces and annotates the protocol message summary
([RFC-0003 §8](../rfcs/RFC-0003-federated-protocol.md#8-message-summary)). The wire schemas are typed in
[03 — Data Model](03-data-model.md); the runtime that emits and validates them is
[RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md).

| Message | Direction | Contents | Protection | Validated against | Error on failure |
|---|---|---|---|---|---|
| `RoundOpen` | coord → participant | $(\theta_t,\phi_t)$ ref/hash, sketch seed $s_t$, probe hash, $H$ | Integrity (SHA-256 hash) | `INV-CHECKPOINT-HASH`, `INV-PROBE-PIN`, `INV-SKETCH-CONSISTENCY` | `CheckpointIntegrityError`, `ProbeError` |
| `Update` | participant → aggregator | $\Delta_c$ (flat delta + L2 norm + bound `dataset_root`) | DP (clip+noise, §6) then secure-aggregation mask (§5) | `INV-DP-BOUND`, `INV-RESIDENCY` | `PrivacyBudgetExceeded`, `SecureAggregationError`, `ResidencyViolation` |
| `Commitment` | participant → coord | dataset Merkle root $R_c$, episode count, WMCP metadata | Binding to $\Delta_c$ | `INV-COMMIT-BINDING` | `CommitmentMismatch`, `MerkleVerificationError` |
| `RoundClose` | coord → all | $(\theta_{t+1},\phi_{t+1})$ content hash, `parent_hash` | Integrity (SHA-256 hash chain) | `INV-CHECKPOINT-HASH`, `INV-AGG-DETERMINISM` | `CheckpointIntegrityError`, `NonDeterministicAggregation` |

Raw observations, actions, and embeddings of private data appear in **no** message — this is the
contract `INV-RESIDENCY` enforces and §4 below details. The `Update` message carries a
`dataset_root` field equal to $R_c$ so the aggregator can bind the contribution without learning the
data (`INV-COMMIT-BINDING`).

RISK: messages cross a real network only at Stage C (v0.3). Stage B (v0.2) simulates the federation in
one process on one cluster ([RFC-0001 Migration / Rollout](../rfcs/RFC-0001-architecture.md#migration--rollout)), so transport
confidentiality and authentication are *unexercised* until v0.3. Resolution plan: §7 below mandates
mutual-TLS transport and message authentication as a v0.3 acceptance gate, specified in
[RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md) (transport choice); the in-process Stage-B harness
must not be mistaken for a secure deployment.

## 3. Residency enforcement (`INV-RESIDENCY`)

Data residency is the load-bearing security property: the training process MUST refuse to emit raw
observations, actions, or their embeddings across a boundary; only pseudo-gradients leave
([RFC-0004 §2](../rfcs/RFC-0004-data-provenance.md#2-residency-the-sovereignty-guarantee-inv-residency)).

**Invariant.** `INV-RESIDENCY` ([conventions §7](conventions.md#7-named-invariants)): no raw observation/action/private-embedding tensor is
serialized into any outbound message or artifact that crosses a trust boundary.

**Where enforced.** `lensemble.data.residency`. Every local dataset carries a non-exportable
`residency` flag (RFC-0004 §2). The residency guard sits on the egress path of two subsystems: the
federation message serializer (`lensemble.federation`, before any `Update`/`Commitment` leaves) and the
observability sinks (`lensemble.observability.redaction`, before any record is written; see
[05 — Observability](05-observability.md) and
[RFC-0015](../rfcs/RFC-0015-observability-diagnostics.md)).

**Mechanism.** The guard is a fail-closed predicate over every object on an egress path. It permits a
value to cross only if the value is one of the explicitly allowed kinds: a pseudo-gradient
`PseudoGradient` (a flat parameter delta, not data), a `DatasetCommitment` (a Merkle root and counts,
not data), shared coordination state (sketch seed, probe hash, model hash), or a derived non-invertible
summary — a content hash, an L2 norm, a tensor shape, an integer count, or a scalar metric. It refuses
any tensor that is a raw observation, a raw action, or an encoder output of private data. The guard is
allowlist-based, not denylist-based: an object whose provenance cannot be proven non-private is
rejected.

**Failure mode.** Trigger: a code path attempts to place a residency-tagged tensor (or a tensor whose
non-private provenance is not established) onto an egress path. Detection: the residency guard
predicate at the serializer/sink boundary. Error: `ResidencyViolation` ([conventions §6](conventions.md#6-error-taxonomy)), a security-critical
error that is **never caught-and-ignored** ([conventions §6](conventions.md#6-error-taxonomy) rule). System response: **fail-closed** — the egress
operation aborts, the round/run halts for that participant, the violation is logged (with the offending
object's *type and shape only*, never its contents), and an operator must intervene. `ResidencyViolation`
is `.recoverable = False`; there is no retry path, because a retry would re-attempt the leak.

The pseudo-gradient itself is a function of private data and therefore a (weak) leakage surface; that
residual leakage is bounded by differential privacy (§6) and hidden in the aggregate by secure
aggregation (§5). Residency enforcement guarantees *raw data and embeddings* never leave; DP and secure
aggregation bound *what the permitted delta reveals*.

## 4. Secure-aggregation guarantee (`INV-AGG-DETERMINISM` interaction)

The coordinator must learn only the sum $\sum_c \Delta_c$, never an individual $\Delta_c$ — because an
individual update leaks more about a silo's data than the sum does
([RFC-0003 §5](../rfcs/RFC-0003-federated-protocol.md#5-secure-aggregation-requirement)). The protocol and its
security proof are specified in [RFC-0011](../rfcs/RFC-0011-secure-aggregation.md); this section states
the guarantee, its assumptions, and its interactions.

**Guarantee.** Under the pairwise-mask (Bonawitz-style) protocol or a TEE-based aggregator, the
aggregator's view is computationally indistinguishable from learning only $\sum_c \Delta_c$ and the
participant set. Per-pair masks cancel in the sum; no individual masked update is invertible without
the colluding pair's shared secret.

**Assumptions (residual trust).**

- Phase 1 honest-but-curious aggregator: it follows the protocol and does not collude with participants
  beyond a bounded threshold. Threshold secret sharing of masks tolerates participant dropout down to a
  configured threshold; below it the round cannot reconstruct and raises `SecureAggregationError`
  (RFC-0011).
- A collusion of the aggregator with $C-1$ participants can isolate the remaining participant's update —
  the standard secure-aggregation collusion bound. Lensemble inherits this bound; it is not removed in
  Phase 1.
- The TEE backend substitutes hardware-attestation trust for the masking-protocol trust (RFC-0011) — a
  different, not strictly weaker, assumption.

**Interactions.**

- With DP (§6): per-participant noise is added *before* masking, so the revealed sum carries the summed
  noise; DP and secure aggregation compose (RFC-0011, RFC-0012).
- With the deterministic outer step: masking must not introduce nondeterminism into the *revealed* sum.
  The aggregation/outer-step path is bitwise-deterministic given its inputs (`INV-AGG-DETERMINISM`,
  [conventions §9](conventions.md#9-determinism-dtype-device)), checked each outer step; a check failure raises `NonDeterministicAggregation`
  (security- and proof-critical, never swallowed) and the step aborts and recomputes
  ([04 — Error Model](04-error-model.md)).
- With int8 pseudo-gradient quantization
  ([RFC-0003 §6](../rfcs/RFC-0003-federated-protocol.md#6-heterogeneity--fault-tolerance)): quantization
  is orthogonal to masking; the quantization round-trip error is bounded and tested (RFC-0011).

**Failure modes.** Dropout below threshold → `SecureAggregationError` (the round retries with the
remaining participants reconciled next round, per elasticity; see
[RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md)). Mask non-cancellation → the sum is wrong and the
determinism/commitment checks downstream fail. A single trusted aggregator that sees plaintext
$\Delta_c$ is rejected by design (it leaks individual updates; RFC-0011 Alternatives).

## 5. Differential-privacy guarantee and its limits

Differential privacy bounds how much a single participant's contribution to a round can be inferred
from the released (summed) update. The mechanism and accounting are specified in
[RFC-0012](../rfcs/RFC-0012-differential-privacy.md); the protocol placement is
[RFC-0003 §4](../rfcs/RFC-0003-federated-protocol.md#4-differential-privacy-protocol-level).

**Mechanism.** Per-participant, before release: clip
$\Delta_c \leftarrow \Delta_c \cdot \min(1, C_{\text{clip}}/\lVert\Delta_c\rVert)$ then add
$\mathcal{N}(0,\sigma^2 C_{\text{clip}}^2 I)$, calibrated to a target $(\varepsilon,\delta)$ over the
planned number of rounds. The unit of privacy is **a participant's contribution to a round**
(per-participant update DP), **not** per-example DP-SGD in the inner loop (RFC-0012) — state this scope
when reporting any privacy guarantee.

**Invariant.** `INV-DP-BOUND` ([conventions §7](conventions.md#7-named-invariants)): after clipping and before noising,
$\lVert\Delta_c\rVert \le C_{\text{clip}}$. Enforced in `lensemble.privacy.dp`; a violation (clipping
not applied or applied wrongly) is a correctness defect caught by the property test in
[07 — Testing Strategy](07-testing-strategy.md).

**Accounting.** `lensemble.privacy.accountant` tracks the cumulative $(\varepsilon,\delta)$ over rounds
(Rényi-DP / PRV accounting). When the budget is spent, training **stops** with `PrivacyBudgetExceeded`
([conventions §6](conventions.md#6-error-taxonomy)); the run does not silently continue past its budget (RFC-0012). The accountant is abstracted so
a reference implementation is swappable ([conventions §11](conventions.md#11-external-dependencies)).

**Honest limits.** DP is a *statistical* guarantee, not a confidentiality wall: it bounds the
distinguishing advantage between neighboring participant-round inputs by $(\varepsilon,\delta)$; it does
not prevent inference that is already implied at the chosen $\varepsilon$, and a loose $\varepsilon$ buys
little protection. DP noise on small predictor deltas interacts with SIGReg variance and the anchor
term; joint calibration of $(\sigma, \lambda_{\text{sig}}, \lambda_{\text{anc}}, C_{\text{clip}})$ is a
Stage-B experiment, not a shipped default (RFC-0003 §4, RFC-0012). DP also does not protect against the
residual leakage in *which* participants contributed (membership at the participant level) beyond what
the secure-aggregation participant set already reveals.

## 6. Provenance: tamper-evidence vs cryptographic proof

Provenance binds each contribution to the data it was computed from, creating an audit trail. The
commitment scheme is specified in [RFC-0014](../rfcs/RFC-0014-provenance-commitments.md); its Phase-2
upgrade is [RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md).

**Phase 1 — tamper-evidence.** Each participant content-hashes its episodes and commits a Merkle root
$R_c$ over those hashes before an epoch (`Commitment` message; RFC-0004 §4, RFC-0014). A contributed
pseudo-gradient is bound to the $R_c$ under which it was computed (`INV-COMMIT-BINDING`: every released
$\Delta_c$ is bound to exactly one $R_c$; enforced in `lensemble.provenance`). The
`ContributionLedger` is an append-only log of (round, contributing participants, their roots, resulting
global-model hash) — the audit substrate (RFC-0004 §5). This gives **contribution accounting** (which
committed dataset fed which round) and **tamper-evidence**: a mismatch between a claimed root and the
bound update, or a corrupted Merkle proof, is detectable.

**Phase 2 — cryptographic proof.** The same commitments upgrade to a cryptographic claim — *"this
update was computed from data under committed root $R_c$"* — provable against a malicious participant
(RFC-0006 §3, roadmap 2b). The aggregation arithmetic becomes a STARK-proven claim (roadmap 2a) and the
inner step a TEE-attested claim (roadmap 2c).

**What provenance does NOT prove (every phase).** It proves data *origin/provenance*, **not** data
quality and **not** honest computation of the gradient
([RFC-0004 §4](../rfcs/RFC-0004-data-provenance.md#4-provenance-commitments-the-bridge-to-phase-2),
[RFC-0006 §4](../rfcs/RFC-0006-verifiable-contribution.md)). A participant can
commit a genuine root over low-quality or adversarially curated data and the binding still verifies.
Stating this boundary is part of the design's integrity, not a caveat to bury.

**Invariants.** `INV-COMMIT-BINDING`, `INV-CHECKPOINT-HASH` (every committed $(\theta_t,\phi_t)$
artifact's content hash equals the `Commitment`/`RoundClose` hash; enforced in `lensemble.artifacts`,
[RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md)), `INV-PROBE-PIN` (the probe content hash
equals the hash committed in `RoundOpen`; landmark targets derive only from $f_{\text{ref}}$;
[RFC-0002 §4](../rfcs/RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)).

**Failure modes.** A claimed root that does not match the bound update → `CommitmentMismatch`
(security-critical, never swallowed; the update is **rejected**). A corrupted/invalid Merkle inclusion
proof → `MerkleVerificationError` (the update is rejected). A probe hash that does not match the pinned
hash → `ProbeError` (the round is refused; the anchor frame is undefined without the pinned probe). A
tampered checkpoint whose recomputed content hash differs from its committed hash →
`CheckpointIntegrityError`. See [04 — Error Model](04-error-model.md).

### 6.1 Public recomputation of alignment (proof-ready, free)

Frame alignment is a deterministic function of the public probe $\mathcal{P}$ and the committed
weights, so its correctness needs **no proof** — any third party recomputes it
([RFC-0006 §4](../rfcs/RFC-0006-verifiable-contribution.md#4-public-recomputation-phase-1-free-in-scope-now),
[RFC-0002 §5](../rfcs/RFC-0002-gauge-and-aggregation.md#5-layer-3--procrustes-re-alignment-at-aggregation-backstop)). The public surface exposes
`recompute_alignment` ([conventions §5](conventions.md#5-public-api-surface)): given the public probe and committed weights, it reproduces the
coordinator's claimed alignment, and a mismatch surfaces a `GaugeError`. This is the cheapest verifiable
guarantee and it is *already enforceable in Phase 1*; the test that `recompute_alignment` reproduces the
coordinator's alignment from public inputs is in scope now (RFC-0006 §5,
[07 — Testing Strategy](07-testing-strategy.md)).

## 7. Secrets handling and supply chain

- **No secrets in artifacts or logs.** No credentials, keys, tokens, mask seeds, or DP private state are
  serialized into any `RunManifest`, checkpoint header, contribution-ledger entry, log record, or
  metrics file. The observability redaction guard (`INV-RESIDENCY`; §3,
  [05 — Observability](05-observability.md)) permits only hashes, norms, shapes, counts, and scalar
  metrics into any sink — secrets fall outside that allowlist and are refused.
- **No `pickle`.** Tensors and weights serialize via `safetensors` ([conventions §8](conventions.md#8-core-data-types), §11), which stores tensors
  without executable code; the artifact format ([RFC-0010](../rfcs/RFC-0010-artifact-checkpoint-format.md))
  forbids `pickle`/`torch.save` precisely because they permit arbitrary-code execution on load and are
  non-deterministic. A no-pickle assertion is a CI/test gate (RFC-0010 Testing Strategy).
- **Key management for secure-aggregation masks / TEE.** Pairwise-mask key agreement and threshold
  shares are held only inside a participant process and never leave its boundary (they are not
  residency-tagged data, but they are secrets and are excluded from every sink by the redaction
  allowlist). The TEE backend's attestation keys are rooted in the hardware enclave. Key lifecycle
  (generation, rotation, revocation) is specified in
  [RFC-0011](../rfcs/RFC-0011-secure-aggregation.md).
- **Transport authentication and confidentiality** (v0.3). Over a real network boundary, messages are
  carried over an authenticated, confidential transport (mutual-TLS) with per-participant identity, so a
  passive observer learns nothing and a peer cannot impersonate another participant. Specified in
  [RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md); a v0.3 acceptance gate.
- **Supply chain.** All external dependencies are pinned with explicit version constraints and a stated
  reason ([conventions §11](conventions.md#11-external-dependencies)); pinned releases include the V-JEPA 2 warm-start weights, the SIGReg objective
  reference (LeJEPA/LeWM), the data layer (stable-worldmodel), and the Phase-2 prover (Stwo).
  Reproducible builds and the dependency manifest are part of the release discipline
  ([09 — Release & Versioning](09-release-and-versioning.md)). The canonical commitment hash in Phase 1
  is **SHA-256** (conservative, interoperable; [conventions §11](conventions.md#11-external-dependencies)).

OPEN QUESTION: migrate the Phase-1 commitment hash from SHA-256 to a STARK-friendly hash (e.g.
Poseidon2) so the Phase-2 proof circuit stays cheap. Owner `@AbdelStark`; resolution path: Stage D /
[RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md) and
[RFC-0014](../rfcs/RFC-0014-provenance-commitments.md). Until resolved, all on-disk and on-wire
commitments use SHA-256 and the hash function is recorded as a versioned choice in artifact headers
(RFC-0010, RFC-0014).

## 8. Phase-2 upgrades and residual trust per phase

[RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md) moves the trust model from *trust via
redundancy/economics* to *trust via cryptographic attestation of contribution*. Phase 1 ships
**proof-ready** so Phase 2 needs no rework; the proof-ready disciplines (deterministic aggregation,
committed model versions, episode hashing + Merkle roots, pinned probe, reproducible outer step) are in
scope for v0.1–v1.0 and are enforced by the invariants above (RFC-0006 §5). The proofs themselves are
post-v1.0 (Stage D, [conventions §12](conventions.md#12-milestones-and-stages)), out of the Phase-1 critical path.

**Phase-2 provable surface** (RFC-0006 §3, roadmap 2a–2d):

| Claim | Phase-2 mechanism | Cost |
|---|---|---|
| Aggregation computed correctly over submitted deltas | STARK over the outer-step circuit (Stwo / Circle STARK) | Tractable — cheap *because* the anchored frame keeps aggregation a near-linear weighted average/Nesterov step (the gauge fix is the verifiability enabler; [RFC-0002](../rfcs/RFC-0002-gauge-and-aggregation.md)) |
| Update derived from committed data | Merkle commitment $R_c$ bound to the update | Cheap; proves provenance, not quality or honesty |
| Frame alignment correct | Public recomputation (deterministic; §6.1) | Free; no proof needed |
| Inner training step executed faithfully | TEE attestation (pragmatic proxy) | Moderate; weaker trust than a SNARK |

### 8.1 Residual trust assumptions, by phase

| Surface | Phase 1 (v0.1–v1.0) | Phase 2 (Stage D, post-v1.0) |
|---|---|---|
| Raw-data confidentiality | Trusted: residency enforcement (`INV-RESIDENCY`, fail-closed) | Same |
| Individual update confidentiality | Trusted: secure aggregation hides $\Delta_c$ (honest-but-curious aggregator, collusion bound) | Same; aggregation also attestable |
| Per-participant-round leakage bound | Trusted: differential privacy $(\varepsilon,\delta)$ within stated limits | Same |
| Aggregation arithmetic correctness | **Assumed** (honest coordinator) | **Proven** (STARK) |
| Provenance / data origin | Tamper-evident (Merkle binding) | **Proven** (cryptographic claim) |
| Frame alignment correctness | Publicly recomputable (free) | Same (already free) |
| Inner-step faithfulness | **Assumed** | **Attested** (TEE) — still an assumption on TEE hardware |
| Data quality / gradient honesty | **Not guaranteed** (out of scope) | **Not guaranteed** (out of scope) |

The honest reading: Phase 1 guarantees *confidentiality* of raw data and individual updates against
honest-but-curious infrastructure, and *tamper-evidence* of provenance — but the coordinator's
aggregation honesty and the participant's inner-step honesty are *assumed*. Phase 2 replaces those two
assumptions with a proof (aggregation) and an attestation (inner step), and upgrades provenance to a
cryptographic claim. Data quality and gradient honesty remain unprovable in both phases and are stated
as out-of-scope rather than papered over.

## 9. Cross-references

- Trust boundaries and the federation map: [RFC-0001 §4, §6](../rfcs/RFC-0001-architecture.md#6-trust-boundaries).
- Round mechanics, DP placement, secure-agg pointer, message table:
  [RFC-0003](../rfcs/RFC-0003-federated-protocol.md).
- Residency, public probe, provenance commitments, WMCP precondition:
  [RFC-0004](../rfcs/RFC-0004-data-provenance.md).
- Secure-aggregation protocol and proof: [RFC-0011](../rfcs/RFC-0011-secure-aggregation.md).
- Differential-privacy mechanism and accounting: [RFC-0012](../rfcs/RFC-0012-differential-privacy.md).
- Provenance commitments and Merkle scheme: [RFC-0014](../rfcs/RFC-0014-provenance-commitments.md).
- Verifiable contribution, trust-model transition, proof-ready requirements:
  [RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md).
- Coordinator/participant runtime and transport: [RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md).
- Error taxonomy and recovery: [04 — Error Model](04-error-model.md).
- Redaction and never-log rules: [05 — Observability](05-observability.md),
  [RFC-0015](../rfcs/RFC-0015-observability-diagnostics.md).
- Security tests (residency guard, DP clip bound, commitment binding, public recomputation):
  [07 — Testing Strategy](07-testing-strategy.md).
- Supply-chain pinning and release discipline: [09 — Release & Versioning](09-release-and-versioning.md).
