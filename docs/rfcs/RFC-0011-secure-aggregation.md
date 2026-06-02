# RFC-0011 — Secure Aggregation Protocol

| | |
|---|---|
| **RFC** | 0011 |
| **Title** | Secure Aggregation Protocol |
| **Slug** | secure-aggregation |
| **Status** | Accepted |
| **Track** | Standards |
| **Authors** | @AbdelStark |
| **Created** | 2026-06-02 |
| **Target milestone** | v0.2 (simulated, single-process) → v0.3 (real multi-party over a network boundary) |
| **Area** | `area:aggregation` |
| **Requires** | [RFC-0003](RFC-0003-federated-protocol.md) (the round that invokes this), [RFC-0012](RFC-0012-differential-privacy.md) (the noise added before masking), [RFC-0013](RFC-0013-coordinator-runtime.md) (the round state machine that drives dropout recovery) |
| **Defers to** | [RFC-0013](RFC-0013-coordinator-runtime.md) (transport, key-exchange channel, state-machine handling of `SecureAggregationError`) |

## Summary

This RFC specifies how the coordinator obtains the sum `Σ_c Δ_c` of participant pseudo-gradients while
learning **nothing about any individual** `Δ_c`. It owns the cryptographic *mechanics* of the
secure-aggregate step (step 5 of the round in
[RFC-0003 §2](RFC-0003-federated-protocol.md#2-round-structure-diloco-outer-loop)); the round
*requirement* — "the coordinator MUST learn only `Σ_c Δ_c`" and the two protocol-level constraints
(dropout robustness, determinism preservation) — lives in
[RFC-0003 §5](RFC-0003-federated-protocol.md#5-secure-aggregation-requirement).

Two backends are specified, both supported and config-selectable. The default is a **pairwise-mask
protocol** (Bonawitz-style): each ordered participant pair agrees a shared seed, derives a
pseudo-random mask vector that is added by one party and subtracted by the other, so masks cancel
exactly in the sum while each individual masked update is computationally hiding. Self-masks plus
threshold (Shamir) secret sharing of every mask seed make the round **dropout-robust**: it completes if
at least a configured threshold `t_agg` of participants survive. The alternative backend is a
**TEE-attested aggregator** that receives plaintext `Δ_c` inside an attested enclave and emits only the
sum. The masking backend introduces **no nondeterminism** into the revealed sum (masks cancel exactly
over the integer field, then the deterministic outer step runs unchanged,
`INV-AGG-DETERMINISM`). DP noise ([RFC-0012](RFC-0012-differential-privacy.md)) is added **per
participant before masking**; int8 quantization
([RFC-0003 §6](RFC-0003-federated-protocol.md#6-heterogeneity--fault-tolerance)) is **orthogonal** and
applied before masking.

The guarantee and residual-trust statement this RFC backs are reproduced by the security model
([06 §4 Secure-aggregation guarantee](../spec/06-security.md#4-secure-aggregation-guarantee-inv-agg-determinism-interaction)); the dropout failure mode and its fields
are catalogued in [04 §Secure-aggregation dropout](../spec/04-error-model.md).

## Motivation

The federated round releases one `PseudoGradient` per participant per round
([RFC-0003 §3](RFC-0003-federated-protocol.md#3-the-pseudogradient-contract)). A single participant's
`Δ_c` is far more revealing about that silo's sovereign data than the population sum: it is the
direction the model would move to fit one silo's trajectories, and gradient-inversion attacks can
reconstruct training inputs from individual updates. Even with per-participant differential privacy
([RFC-0012](RFC-0012-differential-privacy.md)), an honest-but-curious coordinator that observes the
*cleartext per-participant* `Δ_c` defeats the project's residency posture (`INV-RESIDENCY`,
[01 §5 Trust boundaries](../spec/01-architecture.md#5-trust-boundaries)): raw data does not cross the boundary, but a
sufficiently informative function of it would.

Secure aggregation closes this: the aggregator obtains only `Σ_c Δ_c`, which is exactly the input the
outer Nesterov step uses ([RFC-0003 §2 step 7](RFC-0003-federated-protocol.md#2-round-structure-diloco-outer-loop)),
so revealing the sum loses no information the protocol needs. It also keeps the Phase-2
aggregation-correctness proof cheap: the anchored frame ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md))
keeps aggregation a near-linear weighted average, and masking does not perturb that linearity — masks
cancel and the revealed sum is bit-for-bit the plaintext sum, so the public-recomputation / STARK target
is unchanged ([RFC-0006 §2](RFC-0006-verifiable-contribution.md#2-the-provable-surface-and-what-stays-cheap)).

The hard requirement is **dropout robustness without breaking determinism**. Participants churn (compute
heterogeneity, network partitions,
[RFC-0003 §6](RFC-0003-federated-protocol.md#6-heterogeneity--fault-tolerance)); a naive pairwise-mask
scheme stalls when any masked-but-not-cancelled party vanishes mid-round. The protocol must recover the
sum over the *surviving* set deterministically, or fail closed with `SecureAggregationError`.

## Goals

- Specify the pairwise-mask backend: key agreement, mask derivation from a shared seed, the integer
  fixed-point encoding over which masks cancel exactly, the masked-update wire format, and the
  reconstruction of `Σ_c Δ_c`.
- Specify dropout robustness: self-masks plus threshold (Shamir) secret sharing of mask seeds and
  self-mask seeds, so a round completes iff at least `t_agg` participants survive; below `t_agg` raises
  `SecureAggregationError`.
- State the security property precisely (the aggregator's view is computationally indistinguishable from
  learning only `Σ_c Δ_c` and the participant set) and its honest assumptions (the collusion bound).
- Pin the interaction with DP (noise before masking, per participant), with the deterministic outer step
  (masks cancel exactly; no nondeterminism in the revealed sum, `INV-AGG-DETERMINISM`), and with int8
  quantization (orthogonal; applied before masking).
- Specify a TEE-attested aggregator as a second supported, config-selectable backend, with its distinct
  trust assumption stated.
- Enumerate the failure modes (dropout below threshold, mask non-cancellation, key-agreement failure,
  attestation failure) and the typed error each raises ([04 — Error Model](../spec/04-error-model.md); taxonomy: [conventions §6](../spec/conventions.md#6-error-taxonomy)).

## Non-Goals

- The federated round lifecycle, the `PseudoGradient` contract, and the outer Nesterov step. Owned by
  [RFC-0003](RFC-0003-federated-protocol.md).
- The differential-privacy mechanism, the `(ε,δ)` accountant, and joint calibration. Owned by
  [RFC-0012](RFC-0012-differential-privacy.md); this RFC only fixes the *ordering* (noise before mask).
- The transport (gRPC vs HTTP/REST), the key-exchange channel, and the state-machine handling of a
  raised `SecureAggregationError` (retry vs abort). Owned by
  [RFC-0013](RFC-0013-coordinator-runtime.md).
- The Phase-2 aggregation-correctness STARK. Owned by
  [RFC-0006](RFC-0006-verifiable-contribution.md); this RFC only preserves the determinism and linearity
  that proof depends on.
- The dataset-commitment binding (`Commitment` message, `R_c`). Owned by
  [RFC-0014](RFC-0014-provenance-commitments.md); secure aggregation operates on `delta`, the commitment
  travels alongside in the clear.
- A defense against a coordinator colluding with `C-1` participants (the standard collusion bound is
  inherited, not removed in Phase 1 — see Drawbacks).

## Proposed Design

The aggregation subsystem lives in `lensemble.aggregation` (`secure_agg.py`, `masking.py`,
[conventions §1](../spec/conventions.md#1-repository-and-package-layout)). It exposes a backend-agnostic aggregator interface; the round
([RFC-0003 §2](RFC-0003-federated-protocol.md#2-round-structure-diloco-outer-loop)) calls it after the
per-participant privatize step and before the outer step.

### 1. The contract: what the aggregator computes

The aggregator takes one masked update per surviving participant and returns the plaintext sum of their
*unmasked* pseudo-gradients. It never returns, stores, or logs an individual `Δ_c`.

```python
from dataclasses import dataclass
from typing import Protocol, Mapping
from torch import Tensor

# Fixed-point integer encoding parameters (pinned per round in RoundOpen; see §3).
@dataclass(frozen=True)
class FieldParams:
    """Modular integer field over which masks cancel exactly."""
    modulus: int          # 2**k, k chosen so C * max|encoded delta| < modulus (no wrap on the sum)
    scale: float          # fixed-point scale: encoded = round(value * scale); decoded = encoded / scale
    dim: int              # length of the flat (θ, φ) delta vector

@dataclass(frozen=True)
class MaskedUpdate:
    """Participant c's masked, encoded pseudo-gradient. Carries NO recoverable Δ_c on its own."""
    participant_id: str   # surviving-set membership; correlation only (redacted from metrics, RFC-0015)
    round_index: int      # the round t this update is for
    masked: Tensor        # int64 tensor, shape (dim,), values in [0, modulus): encode(Δ_c) + Σ masks
    dataset_root: bytes   # the bound Merkle root R_c, in the clear (RFC-0014; INV-COMMIT-BINDING)

class SecureAggregator(Protocol):
    """Backend-agnostic. Pairwise-mask (default) or TEE. lensemble.aggregation.secure_agg."""
    def aggregate(
        self,
        updates: Mapping[str, MaskedUpdate],   # surviving participants only
        *,
        field: FieldParams,
        round_index: int,
        threshold: int,                         # t_agg: minimum survivors to reconstruct
        recovery: "DropoutRecovery",            # secret-shared seeds for dropped/surviving parties (§4)
    ) -> Tensor:                                # fp32 plaintext Σ_c Δ_c, shape (dim,)
        ...
```

`aggregate` returns the fp32 plaintext sum `Σ_c Δ_c` over the surviving set. The result is the exact
input to the outer step ([RFC-0003 §2 step 7](RFC-0003-federated-protocol.md#2-round-structure-diloco-outer-loop));
no individual `Δ_c` is materialized in plaintext anywhere on the aggregator. If fewer than `threshold`
participants survive, `aggregate` raises `SecureAggregationError` and does **not** return a partial sum.

### 2. Backend A — pairwise additive masking (default)

The default backend is a Bonawitz-style single-server secure-aggregation protocol with pairwise masks,
self-masks, and threshold secret sharing for dropout recovery.

**Setup (per round, before masking).** Each participant `c` holds a Diffie–Hellman static keypair whose
public key is distributed through the coordinator (the coordinator routes public keys but is not trusted
to read masks — it never holds a private key). For every ordered pair `(c, c')` with `c < c'` both
present in the round, the two derive a shared secret `s_{c,c'} = KA(sk_c, pk_{c'})` via key agreement,
then a per-round seed `seed_{c,c'} = KDF(s_{c,c'} ‖ round_index)` domain-separated by the round index.
Key agreement and the channel that carries public keys are part of the control plane
([RFC-0013](RFC-0013-coordinator-runtime.md)); this RFC fixes only what the seeds are used for.

**Mask derivation.** From `seed_{c,c'}` each pair deterministically expands a pseudo-random integer mask
vector `m_{c,c'} ∈ [0, modulus)^dim` (a stream cipher / CSPRNG keyed by the seed, same expansion on both
sides). Participant `c` **adds** `m_{c,c'}` for every `c' > c` and **subtracts** `m_{c,c'}` for every
`c' < c`. Each participant also draws a self-mask `b_c ∈ [0, modulus)^dim` from a private seed `r_c` and
adds it.

**Masked update.** Encode the privatized pseudo-gradient to the integer field, add masks, take the sum
modulo `modulus`:

```
encode(Δ_c)[i] = round(Δ_c[i] * scale) mod modulus
masked_c       = ( encode(Δ_c) + b_c + Σ_{c' > c} m_{c,c'} − Σ_{c' < c} m_{c,c'} ) mod modulus
```

`masked_c` is the `MaskedUpdate.masked` field. Because the field is modular and `b_c`, `m_{c,c'}` are
uniform over it, `masked_c` is computationally indistinguishable from a uniform random vector to anyone
without the seeds; it carries no recoverable `Δ_c` alone.

**Reconstruction (the sum).** When every participant in a set `S` is present, the pairwise masks cancel
because each `m_{c,c'}` appears once with `+` (in `masked_c`) and once with `−` (in `masked_{c'}`). The
aggregator must also remove the self-masks `Σ_{c∈S} b_c`. With the secret-shared seeds (§4) it
reconstructs `b_c` for every *surviving* participant and the *pairwise* seeds for every *dropped*
participant:

```
Σ_{c∈S} encode(Δ_c) = ( Σ_{c∈S} masked_c − Σ_{c∈S} b_c − Σ_{dropped d, c∈S} (±m_{c,d}) ) mod modulus
Σ_{c∈S} Δ_c         = lift_signed( Σ_{c∈S} encode(Δ_c) ) / scale     # fp32, recentred from [0,modulus)
```

`lift_signed` maps the modular residue back to a signed integer in `(−modulus/2, modulus/2]` before
dividing by `scale`. The modulus is sized (`§3`) so the true integer sum never wraps, making this lift
exact. The aggregator never learns any individual `b_c` purpose beyond this cancellation and never holds
two reconstructions that would isolate a single `encode(Δ_c)`.

### 3. Integer encoding, modulus, and exactness

Masks cancel over a finite integer field, not over floating point — float addition is non-associative
and would break both cancellation and `INV-AGG-DETERMINISM`. The encoding is pinned per round and
recorded in the `RunManifest` ([RFC-0009](RFC-0009-configuration-reproducibility.md)):

- **Fixed-point scale.** `encode(x) = round(x * scale) mod modulus`. `scale` is a power of two so the
  encode/decode is exact-rounding and reproducible across platforms.
- **Modulus.** `modulus = 2**k` with `k` chosen so that the maximum possible *signed* integer sum
  `C * round(C_clip * scale)` fits strictly inside `(−modulus/2, modulus/2]` (clipping bounds each
  `‖Δ_c‖ ≤ C_clip`, `INV-DP-BOUND`, [RFC-0012](RFC-0012-differential-privacy.md)). No-wrap on the sum is
  the precondition that makes `lift_signed` exact; it is asserted at setup. Storage is int64 per
  coordinate; the modulus stays below `2**62` to leave headroom for the additive masks.
- **Exact cancellation.** Modular integer addition is associative and order-independent, so
  `Σ_c masked_c` is identical regardless of the order in which masked updates arrive. The plaintext sum
  recovered is therefore a deterministic function of the surviving set and the committed updates — it
  feeds the bitwise-deterministic outer step unchanged
  ([RFC-0003 §7](RFC-0003-federated-protocol.md#7-determinism-concurrency-error-propagation),
  `INV-AGG-DETERMINISM`). The per-outer-step determinism self-check ([RFC-0003 §7](RFC-0003-federated-protocol.md#7-determinism-concurrency-error-propagation)) re-derives the sum in
  the fixed coordinate order and compares; a mismatch raises `NonDeterministicAggregation`
  (security- and proof-critical, never swallowed).

`FieldParams` (modulus, scale, dim) is broadcast in `RoundOpen`
([RFC-0003 §8](RFC-0003-federated-protocol.md#8-message-summary)) so every participant encodes against
identical parameters; a participant encoding against different parameters produces a sum that fails the
no-wrap assertion or the downstream commitment/determinism checks.

### 4. Dropout robustness (threshold secret sharing)

Participants may vanish between submitting a `MaskedUpdate` and the reconstruction phase
([RFC-0003 §6](RFC-0003-federated-protocol.md#6-heterogeneity--fault-tolerance)). The recovery data lets
the aggregator cancel masks for both survivors and dropouts without ever learning an individual
plaintext update.

```python
@dataclass(frozen=True)
class DropoutRecovery:
    """Shamir shares collected during the round, used only at reconstruction."""
    threshold: int                              # t_agg: shares needed to reconstruct any seed
    # For each participant, shares of BOTH its self-mask seed r_c and its DH private key sk_c,
    # distributed to peers during setup. The aggregator collects shares from survivors only.
    self_mask_shares: Mapping[str, list[bytes]] # survivor c -> shares of r_c (reconstruct b_c)
    pairwise_shares: Mapping[str, list[bytes]]  # dropped d  -> shares of sk_d (reconstruct m_{c,d})
```

**The double-masking rule (no isolation under late drop).** A participant is either *surviving* or
*dropped* for a given round, never both, and the aggregator reconstructs **exactly one** seed per
participant:

- For a **surviving** participant `c`, the aggregator reconstructs only the **self-mask seed** `r_c`
  (hence `b_c`) and never `c`'s pairwise seeds. This prevents an adversary that observed `masked_c` from
  removing `c`'s pairwise masks and isolating `encode(Δ_c)`.
- For a **dropped** participant `d`, the aggregator reconstructs only `d`'s **DH private key** `sk_d`
  (hence its pairwise seeds `m_{c,d}` for survivors `c`) and never `d`'s self-mask. This removes the
  uncancelled pairwise terms that survivors added against `d`.

Both seed kinds are Shamir-secret-shared with threshold `t_agg` during setup; the aggregator collects
shares from survivors. Reconstruction succeeds iff at least `t_agg` participants survive; below `t_agg`
the aggregator cannot recover the masks and raises `SecureAggregationError` (`SECURE_AGG_FAILED`) with
fields `round`, `present`, `threshold`, `cause` ([04 §error table](../spec/04-error-model.md)). The
state machine ([RFC-0013](RFC-0013-coordinator-runtime.md)) then retries the round with the surviving
set or aborts; dropped participants reconcile from the latest committed checkpoint next round
([RFC-0010](RFC-0010-artifact-checkpoint-format.md)).

`t_agg` is a deployment-policy value, not yet pinned (Open Questions; shared with
[04 §Open Questions](../spec/04-error-model.md) and the `FaultToleranceExceeded` minimum in
[RFC-0003 §6](RFC-0003-federated-protocol.md#6-heterogeneity--fault-tolerance)). The quorum check for
`FaultToleranceExceeded` (too few to run a round at all) is distinct from and at least as strict as the
secure-agg threshold for `SecureAggregationError` (too few to reconstruct the masks).

### 5. Backend B — TEE-attested aggregator

The second supported backend replaces the masking protocol with a hardware enclave. Participants send
**plaintext** (DP-privatized) `Δ_c` over a channel terminated *inside* an attested enclave; the enclave
computes `Σ_c Δ_c`, returns only the sum, and proves via remote attestation that it ran the pinned,
auditable aggregation code and retains nothing.

```python
@dataclass(frozen=True)
class TEEAttestation:
    enclave_measurement: bytes  # MRENCLAVE-equivalent: hash of the loaded aggregation code
    quote: bytes                # signed attestation quote, verified against the vendor root
    code_hash: bytes            # SHA-256 of the aggregation source pinned in the RunManifest
```

The participant verifies `TEEAttestation` (measurement matches the pinned `code_hash`; quote verifies
against the vendor attestation root) **before** opening the channel; a failed verification raises
`SecureAggregationError` (`cause="attestation_failed"`) and the participant refuses to send. The trust
assumption is different from masking: it is hardware-attestation trust rather than the
collusion-bounded honest-but-curious assumption — neither strictly stronger nor weaker
([06 §4 assumptions](../spec/06-security.md#4-secure-aggregation-guarantee-inv-agg-determinism-interaction)). The TEE backend is config-selected; the security model
records that it substitutes one residual-trust assumption for the other.

### 6. Backend selection, data flow, and where it sits in the round

The backend is a config choice under the `federation`/`aggregation` group
([RFC-0009](RFC-0009-configuration-reproducibility.md)); both implement `SecureAggregator`. In the round
([RFC-0003 §2](RFC-0003-federated-protocol.md#2-round-structure-diloco-outer-loop)):

```text
participant c:                                          aggregator:
  Δ_c (post inner loop)
  clip ‖Δ_c‖ ≤ C_clip                  (RFC-0012)
  add N(0, σ²C_clip²I)                 (RFC-0012)   ─── DP noise added BEFORE masking
  [optional] int8 quantize Δ_c         (RFC-0003 §6) ─ orthogonal to masking
  encode to int field (scale, modulus) (§3)
  add self-mask b_c + pairwise masks   (§2)
  ── MaskedUpdate ───────────────────────────────────►  Σ_c masked_c  (mod modulus)
  (Shamir shares distributed at setup) (§4)              reconstruct b_c (survivors), m_{c,d} (dropouts)
                                                          Σ_c Δ_c (fp32, exact)
                                                          determinism self-check (RFC-0003 §7)
                                                          ── plaintext sum ──► outer Nesterov step
```

The ordering is normative: **DP noise → (optional) int8 quantize → integer encode → mask**. Noise before
masking means the revealed sum already carries the summed Gaussian noise, so DP and secure aggregation
compose ([RFC-0012](RFC-0012-differential-privacy.md), [06 §4](../spec/06-security.md#4-secure-aggregation-guarantee-inv-agg-determinism-interaction)). Quantization
before encoding means masks operate on the already-quantized vector, keeping quantization orthogonal to
the gauge ([RFC-0003 §6](RFC-0003-federated-protocol.md#6-heterogeneity--fault-tolerance)); its
round-trip L2 error is bounded and tested (Testing Strategy).

### 7. Invariants enforced

- `INV-AGG-DETERMINISM` ([conventions §7](../spec/conventions.md#7-named-invariants)). The revealed sum is a pure,
  order-independent function of the surviving set and the committed masked updates: masks cancel over an
  associative integer field (§3), and the subsequent outer step is bitwise-deterministic
  ([RFC-0003 §7](RFC-0003-federated-protocol.md#7-determinism-concurrency-error-propagation)). The
  masking layer adds **no** nondeterministic reduction. Enforced by the per-outer-step determinism
  self-check in `lensemble.aggregation`; a failure raises `NonDeterministicAggregation` and the round
  aborts and recomputes. Never swallowed.
- `INV-COMMIT-BINDING` ([conventions §7](../spec/conventions.md#7-named-invariants)). The `dataset_root` travels in the clear on
  each `MaskedUpdate`; secure aggregation does not hide it (the binding must remain auditable). A masked
  update bound to no root or the wrong root is rejected at ingress with `CommitmentMismatch`
  ([RFC-0014](RFC-0014-provenance-commitments.md)) before it enters the sum.
- `INV-DP-BOUND` ([conventions §7](../spec/conventions.md#7-named-invariants)). The clip bound `‖Δ_c‖ ≤ C_clip` is the
  precondition for sizing `modulus` (§3); it is enforced upstream in `lensemble.privacy.dp`
  ([RFC-0012](RFC-0012-differential-privacy.md)). This RFC depends on it but does not enforce it.
- `INV-RESIDENCY` ([conventions §7](../spec/conventions.md#7-named-invariants)). The aggregator never materializes, logs, or
  returns an individual plaintext `Δ_c`; only the sum leaves the aggregation boundary. Redaction
  ([RFC-0015](RFC-0015-observability-diagnostics.md), [05 §5 Redaction](../spec/05-observability.md#5-redaction-inv-residency))
  forbids emitting `MaskedUpdate.masked` content or any reconstructed seed to a log or metric — permitted
  emissions are the participant set, counts, and the sum's norm.

### 8. Failure modes

| Trigger | Detection | Error | System response |
|---|---|---|---|
| Surviving participants fall below `t_agg` | aggregator threshold check before reconstruction (§4) | `SecureAggregationError` (`SECURE_AGG_FAILED`), fields `round`, `present`, `threshold`, `cause="below_threshold"` | retry the round with survivors; dropouts reconcile next round ([RFC-0013](RFC-0013-coordinator-runtime.md)) |
| Mask reconstruction fails (insufficient/invalid Shamir shares) | share-recombination check (§4) | `SecureAggregationError`, `cause="mask_reconstruction_failed"` | retry; if persistent, abort via state machine |
| Key agreement fails / inconsistent public keys | setup-phase KA verification (§2) | `SecureAggregationError`, `cause="key_agreement_failed"` | retry setup; exclude the offending participant |
| Masks do not cancel (sum wrong) | revealed sum fails the downstream determinism / commitment checks | `NonDeterministicAggregation` (`AGG_NONDETERMINISTIC`), fields `round`, `expected_hash`, `got_hash` | fail-closed; abort outer step and recompute; never swallowed (`INV-AGG-DETERMINISM`) |
| Integer sum would wrap the modulus | no-wrap assertion at setup / encode (§3) | `AggregationError` (`AGGREGATION_FAILED`) | abort; re-size `modulus` (config defect, not a runtime path) |
| TEE attestation verification fails | participant verifies quote before sending (§5) | `SecureAggregationError`, `cause="attestation_failed"` | participant refuses to send; round retries or aborts |
| A backend tries to return an individual `Δ_c` | residency guard on aggregator egress (`INV-RESIDENCY`) | `ResidencyViolation` (`RESIDENCY_VIOLATION`) | fail-closed; never caught-and-ignored |

`SecureAggregationError` is **recoverable (retry)** — dropout is an expected operating condition, not a
breach. `NonDeterministicAggregation` and `ResidencyViolation` are **fail-closed and security-critical**:
never caught-and-ignored, never downgraded to a warning ([04 §error-handling rules](../spec/04-error-model.md)).

## Alternatives Considered

- **Pairwise additive masking (chosen default).** What it is: the Bonawitz-style single-server protocol
  with pairwise masks, self-masks, and threshold secret sharing. Why considered: it gives an
  honest-but-curious aggregator a view computationally indistinguishable from learning only the sum, with
  practical cost and well-understood dropout recovery, and it is purely additive — so it preserves the
  near-linear aggregation the Phase-2 STARK and `INV-AGG-DETERMINISM` rely on
  ([RFC-0006 §2](RFC-0006-verifiable-contribution.md#2-the-provable-surface-and-what-stays-cheap)). Why chosen: lowest residual-trust assumption that
  needs no special hardware, and it leaves the outer step bit-identical to the no-security path.
- **TEE-attested aggregator (chosen alternative backend).** What it is: an attested enclave that sees
  plaintext `Δ_c` and returns only the sum. Why considered: simpler protocol, no per-pair key agreement,
  and it can also host the Phase-2c attested inner step
  ([RFC-0006 §7](RFC-0006-verifiable-contribution.md#7-roadmap)). Why offered as a backend, not the only path: it
  substitutes hardware-attestation trust (vendor root, side-channel exposure) for the
  collusion-bounded cryptographic assumption — a different trust model some deployments prefer and others
  reject; making it config-selectable lets the deployment choose.
- **MPC / homomorphic encryption for the sum.** What it is: compute `Σ_c Δ_c` under a general MPC
  protocol or additively-homomorphic encryption. Why considered: stronger malicious-security properties
  than single-server masking. Why rejected for Phase 1: order-of-magnitude higher communication and
  compute for a `dim ~ 10^8`–`10^9` vector, and the malicious-security gap is exactly what Phase 2
  addresses cryptographically through the aggregation STARK over the *committed* deltas
  ([RFC-0006](RFC-0006-verifiable-contribution.md)) rather than at aggregation time. Revisit if a
  malicious-aggregator guarantee is needed before the STARK lands.
- **A single trusted aggregator (no masking).** What it is: send plaintext `Δ_c` to the coordinator and
  trust it to only compute and reveal the sum. Why considered: trivially simple and the lowest overhead.
  Why rejected: it defeats the entire point — an honest-but-curious coordinator that sees individual
  `Δ_c` can run gradient inversion against a silo, violating the residency posture
  ([RFC-0003 §5](RFC-0003-federated-protocol.md#5-secure-aggregation-requirement),
  [06 §4](../spec/06-security.md#4-secure-aggregation-guarantee-inv-agg-determinism-interaction)). Rejected by design.
- **Float-domain masking (mask in fp32, not integer).** What it is: add/subtract real-valued masks
  directly to the fp32 delta. Why considered: skips the integer encode/decode. Why rejected: fp32
  addition is non-associative, so masks would not cancel exactly and the revealed sum would depend on
  arrival order — a direct violation of `INV-AGG-DETERMINISM`
  ([RFC-0003 §7](RFC-0003-federated-protocol.md#7-determinism-concurrency-error-propagation)) and a broken
  Phase-2 recomputation. The integer field (§3) is required precisely to make cancellation exact.

## Drawbacks

- **Key-agreement and secret-sharing overhead.** The pairwise backend needs a setup round of public-key
  distribution and `O(C²)` per-pair seeds (mask *vectors* are derived, not stored, so memory is `O(dim)`
  per participant, but the share-distribution traffic grows with `C`). At small `C` (Stage B/C) this is
  negligible; at large `C` it motivates grouped/quantized variants (Open Questions).
- **Dropout-recovery round-trip.** Reconstructing masks for late dropouts adds a round-trip after the
  masked updates arrive; an adaptive adversary that drops participants strategically can force repeated
  recovery (Open Questions: behavior under adaptive dropout).
- **Inherited collusion bound.** An aggregator colluding with `C-1` participants can isolate the
  remaining participant's update — the standard single-server secure-aggregation bound. Lensemble
  inherits it; Phase 1 does not remove it ([06 §4 assumptions](../spec/06-security.md#4-secure-aggregation-guarantee-inv-agg-determinism-interaction)). DP
  ([RFC-0012](RFC-0012-differential-privacy.md)) is the residual protection in that worst case.
- **TEE backend relies on hardware trust.** Enclave attestation depends on a vendor root of trust and is
  exposed to side-channel and microarchitectural attacks; it is a pragmatic proxy, not a cryptographic
  guarantee.
- **Modulus sizing couples to `C_clip`.** The no-wrap requirement (§3) ties the integer field width to
  `C * C_clip * scale`; a mis-set `C_clip` or an unexpectedly large `C` is caught by the no-wrap
  assertion, but it is a coupling a deployment must respect when changing either knob.

## Migration / Rollout

Staged along [conventions §12](../spec/conventions.md#12-milestones-and-stages) and the protocol rollout
([RFC-0003 §Migration](RFC-0003-federated-protocol.md#migration--rollout)):

- **v0.2 / Stage B — simulated, single-process.** The full masking protocol runs in-process: `C`
  simulated participants derive and exchange masks and Shamir shares within one process; the aggregator
  reconstructs the sum and the determinism self-check runs each round. This validates exact cancellation,
  dropout recovery below/above `t_agg`, and the no-individual-leak property without any network. The
  simulated aggregation is the negative/positive controls' substrate for the ablation ladder
  ([RFC-0005 §6](RFC-0005-evaluation.md#6-ablation-ladder-the-core-experiment)).
- **v0.3 / Stage C — real multi-party over a network boundary.** Real key agreement and share
  distribution over the transport ([RFC-0013](RFC-0013-coordinator-runtime.md)); the masked-update wire
  format becomes the on-the-wire `Update` message
  ([RFC-0003 §8](RFC-0003-federated-protocol.md#8-message-summary)). The TEE backend, if selected,
  attests over the real channel. int8 quantization may be enabled once its round-trip error bound is
  validated.

No contract migration is required between stages: `MaskedUpdate`, `FieldParams`, and `DropoutRecovery`
are stable from v0.2; Stage C swaps the transport and the key-exchange channel, not the protocol
semantics. The on-disk `RunManifest` records `FieldParams` and the selected backend so a run is
reproducible ([RFC-0009](RFC-0009-configuration-reproducibility.md)).

## Testing Strategy

CPU-runnable tests on tiny synthetic fixtures (no large downloads;
[07 §8 CI gates](../spec/07-testing-strategy.md#8-ci-gates)):

- **Mask cancellation correctness.** For random `Δ_c` (small `C`, small `dim`), assert the reconstructed
  `Σ_c Δ_c` equals the plaintext `Σ_c Δ_c` to exact integer equality after decode (the integer field
  makes this exact, not approximate, modulo the fixed-point `scale`). This is the core property
  ([06 §4](../spec/06-security.md#4-secure-aggregation-guarantee-inv-agg-determinism-interaction)).
- **Order independence (`INV-AGG-DETERMINISM`).** Feed the masked updates to `aggregate` in several
  permutations and assert a bit-identical sum each time; assert the downstream outer step's
  determinism self-check passes ([RFC-0003 §Testing](RFC-0003-federated-protocol.md#testing-strategy)).
- **Dropout recovery, above and below threshold.** Simulate participant dropout: with `≥ t_agg`
  survivors, the reconstructed sum equals the plaintext sum over survivors; with `< t_agg`, `aggregate`
  raises `SecureAggregationError` (`present`, `threshold`) and returns no partial sum. Assert the
  double-masking rule: the aggregator reconstructs the self-mask for survivors and the pairwise seeds for
  dropouts, never both for the same participant.
- **No-individual-leak property test.** A property test (hypothesis) that no single `MaskedUpdate.masked`
  nor any combination short of the full surviving set lets an adversary recover an individual
  `encode(Δ_c)`; assert the aggregator never materializes a plaintext individual `Δ_c`
  (`INV-RESIDENCY`).
- **No-wrap / modulus sizing.** Assert the no-wrap assertion fires when `C * round(C_clip * scale)` would
  exceed `modulus/2` (raising `AggregationError`), and that a correctly sized modulus makes `lift_signed`
  exact for adversarial sign patterns.
- **DP-then-mask ordering.** Assert noise is applied to `Δ_c` before encoding/masking and that the
  revealed sum carries the summed noise (compose with [RFC-0012 §Testing](RFC-0012-differential-privacy.md#testing-strategy)).
- **int8 quantization interaction.** Quantize/dequantize `Δ_c` before masking; assert the L2 round-trip
  error is within the stated bound ([RFC-0003 §Testing](RFC-0003-federated-protocol.md#testing-strategy)) and that the
  dequantized, masked, reconstructed sum still passes the determinism self-check.
- **TEE backend (simulated).** A simulated enclave fixture: a participant rejects a `TEEAttestation`
  whose measurement ≠ the pinned `code_hash` with `SecureAggregationError` (`cause="attestation_failed"`);
  a valid attestation yields the correct sum.

## Open Questions

OPEN QUESTION: The secure-aggregation dropout threshold `t_agg` (the minimum survivors to reconstruct the
masked sum) is a deployment-policy value, not yet pinned, and is distinct from the `FaultToleranceExceeded`
minimum ([RFC-0003 §6](RFC-0003-federated-protocol.md#6-heterogeneity--fault-tolerance),
[04 §Open Questions](../spec/04-error-model.md)). Owner @AbdelStark; resolution: Stage C (v0.3) deployment
policy, informed by the Stage-B churn simulation.

OPEN QUESTION: The default backend choice (pairwise masking vs TEE) for the real Stage-C deployment, and
whether both are exercised on the two-node demonstration. Owner @AbdelStark; resolution: Stage C (v0.3),
coupled to the transport/runtime decision in [RFC-0013 §Open Questions](RFC-0013-coordinator-runtime.md#open-questions).

OPEN QUESTION: Behavior under adaptive (adversarial) dropout, where a party drops strategically to force
repeated mask recovery or to influence which seeds the aggregator reconstructs. The double-masking rule
(§4) preserves confidentiality per round; the cost and liveness impact under adaptive churn is
uncharacterized. Owner @AbdelStark; resolution: Stage C (v0.3) stress simulation.

RISK: At the 1.2B target scale ([conventions §12](../spec/conventions.md#12-milestones-and-stages)) the `dim`-length integer mask vectors
and `O(C²)` per-pair seed material may dominate the per-round budget the masking backend can carry
([08 §4 Communication budget](../spec/08-performance-budget.md#4-communication-budget--the-diloco-efficiency-claim)). Resolution plan: measure the masking
overhead in the Stage-C comms accountant; if it dominates, evaluate grouped masking (mask over
participant subgroups), quantized masks, or fall back to the TEE backend (which has no per-pair seed
cost). int8 quantization (§6) already reduces the masked payload by ~4×; the masking overhead itself is
the residual concern.

## References

- [RFC-0003 — Federated Training Protocol](RFC-0003-federated-protocol.md) — §2 the round that invokes
  this step, §5 the secure-aggregation requirement and the two protocol-level constraints, §6 int8
  quantization and fault tolerance, §7 the determinism contract, §8 the message table.
- [RFC-0012 — Differential Privacy Accounting](RFC-0012-differential-privacy.md) — the Gaussian mechanism
  and the noise-before-masking ordering this RFC fixes.
- [RFC-0013 — Coordinator & Participant Runtime](RFC-0013-coordinator-runtime.md) — the transport, the
  key-exchange channel, and the state-machine handling of `SecureAggregationError`.
- [RFC-0014 — Provenance Commitments & Merkle Scheme](RFC-0014-provenance-commitments.md) — the
  `dataset_root` binding that travels in the clear alongside each masked update.
- [RFC-0002 — The Latent Gauge & Frame-Anchored Aggregation](RFC-0002-gauge-and-aggregation.md) — the
  anchored frame that keeps aggregation near-linear, which secure aggregation must not perturb.
- [RFC-0006 — Verifiable Contribution](RFC-0006-verifiable-contribution.md) — §2 the provable surface
  (the aggregation-correctness STARK) and §4 the public-recomputation surface, both depending on
  determinism and linearity; §3 the Phase-1 proof-ready requirements.
- [03 — Data Model](../spec/03-data-model.md) (`PseudoGradient`, `GlobalState`) ·
  [04 — Error Model §Secure-aggregation dropout](../spec/04-error-model.md) ·
  [05 — Observability §5 Redaction](../spec/05-observability.md#5-redaction-inv-residency) ·
  [06 — Security §4 Secure-aggregation guarantee](../spec/06-security.md#4-secure-aggregation-guarantee-inv-agg-determinism-interaction) ·
  [08 — Performance Budget](../spec/08-performance-budget.md).
- External: Bonawitz et al., *Practical Secure Aggregation for Privacy-Preserving Machine Learning*
  (pairwise masking, threshold secret sharing for dropout); Shamir secret sharing; DiLoCo / INTELLECT-1
  (the int8 all-reduce that composes with masking).
