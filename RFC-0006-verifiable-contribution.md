# RFC-0006 — Verifiable Contribution

| | |
|---|---|
| **RFC** | 0006 |
| **Title** | Verifiable Contribution |
| **Status** | Draft · **Phase 2 (Deferred)** |
| **Track** | Standards |
| **Author** | Abdelhamid Bakhta (@AbdelStark) |
| **Requires** | RFC-0001, RFC-0003, RFC-0004 |
| **Date** | June 2026 |

> Verifiability is **Phase 2**: prove the federated-JEPA science first (RFC-0002, 0005). This RFC specifies the eventual layer and — crucially — the small set of properties Phase 1 must satisfy so that adding proofs later requires **no rework**. It is the project's long-run differentiator.

## 1. The thesis

Existing decentralized-training networks hold themselves together with redundancy, attestation, and economic incentives — not with succinct cryptographic proof of the training computation. Lensemble's distinguishing claim is to move the trust model from **"trust via redundancy/economics"** to **"trust via cryptographic attestation of contribution."** This is the Tapestry Evaluation-Certification and Data-Governance pillars made cryptographic, and it connects directly to prior work on post-quantum verifiable AI ("The Half-Life of Trust") and composed assurance.

## 2. Trust model

- **Phase 1**: coordinator and aggregator are honest-but-curious; secure aggregation + DP (RFC-0003) protect raw data; provenance commitments (RFC-0004 §4) give tamper-evidence.
- **Phase 2**: a malicious coordinator/aggregator, and participants who may misreport, are detectable. Honest contribution becomes *verifiable*, not assumed.

## 3. The provable surface (and what stays cheap)

| Claim | Mechanism | Cost / note |
|---|---|---|
| **Aggregation computed correctly** over the submitted deltas | **STARK over the outer-step circuit** (Stwo / Circle STARK) | Tractable — and cheap *because* the anchored frame (RFC-0002) keeps aggregation a plain weighted average / Nesterov step, i.e. a near-linear operation. This is the synergy: the gauge fix is also the verifiability enabler. |
| **Update derived from committed data** | **Merkle commitment** $R_c$ + binding to the update (RFC-0004 §4) | Cheap. Proves provenance, **not** quality or honesty of the gradient. |
| **Frame alignment correct** | **Public recomputation** — deterministic function of public probe + committed weights | **Free.** No proof needed; anyone recomputes and checks (RFC-0002 §4.3). |
| **Inner training step executed faithfully** | **TEE attestation** (pragmatic proxy) | Moderate; weaker trust than a SNARK but far cheaper than proving training. |

## 4. Honest cost gradient

- Proving a **full 1.2 B training step** in zero knowledge is **infeasible today** — this RFC does not promise it.
- The leverage is structural: the anchored-frame design lets aggregation stay plain weighted averaging, whose circuit is small enough to prove in Stwo. Keep the heavy, data-touching computation behind cheaper guarantees (TEE, provenance commitments) and reserve succinct proofs for the linear-ish aggregation and the provenance binding.
- Provenance proofs attest *which committed dataset* an update came from — a contribution-accounting and licensing guarantee — **not** that the data is good or the gradient honest. Stating this boundary is part of the design's integrity.

## 5. Proof-ready requirements on Phase 1 (cheap to honor now)

Phase 1 MUST bake in the following so Phase 2 needs no rework:

1. **Deterministic aggregation** — the outer step (RFC-0003 §3) is a fixed, reproducible function of the inputs (no nondeterministic reductions in the aggregation path).
2. **Committed model versions** — hash-commit every $(\theta_t,\phi_t)$ (already in RFC-0001 §5 / RFC-0003 §7).
3. **Episode hashing + Merkle roots from day one** — even before proofs exist (RFC-0004 §4).
4. **Pinned public probe** — content-hash the probe and landmarks; alignment must be recomputable from public inputs alone (RFC-0004 §3).
5. **Reproducible outer step** — fixed seeds/order so a verifier can recompute the claimed aggregation.

These are inexpensive engineering disciplines in Phase 1 and are the entire prerequisite for Phase 2.

## 6. Roadmap

- **2a** — aggregation-correctness STARK (Stwo) over the committed deltas → committed global model.
- **2b** — provenance binding: tie each update to its dataset Merkle root.
- **2c** — TEE attestation for the inner step; define the composed trust statement (which parts are proven vs attested vs assumed).
- **2d** *(research)* — push toward heavier proof-of-training / proof-of-learning as prover capability improves; benchmark prover cost (Stwo) against circuit size as the model scales.

## 7. Out of scope (v0.1)

Incentive/payment mechanisms, slashing, and on-chain settlement. Lensemble specifies the *attestation* layer; economic design layers on top and is deliberately separated.
