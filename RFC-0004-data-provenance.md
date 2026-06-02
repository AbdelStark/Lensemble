# RFC-0004 — Data, Sovereignty & Provenance

| | |
|---|---|
| **RFC** | 0004 |
| **Title** | Data, Sovereignty & Provenance |
| **Status** | Draft |
| **Track** | Standards |
| **Author** | Abdelhamid Bakhta (@AbdelStark) |
| **Requires** | RFC-0001, RFC-0003 |
| **Date** | June 2026 |

## 1. Scope

How sovereign data participates without leaving its boundary, how the shared public probe is governed, and how contributions are committed so the Phase-2 verifiable layer (RFC-0006) can bind to them.

## 2. Per-participant data layer

Each participant holds a **local** dataset of interaction trajectories and trains against it only. Reuse `stable-worldmodel`'s data layer:

- **Schema** — episodes of $(o_t, a_t, o_{t+1})$ tuples; the loader yields fixed windows (`num_steps`) for next-embedding prediction.
- **Formats** — `lance` (default; append-friendly, fast indexed reads), `hdf5` (portable single-file), and the `lerobot://<repo_id>` adapter for training directly on LeRobot-Hub robot datasets.
- **Residency** — every local dataset carries a non-exportable flag; the training process MUST refuse to emit raw observations/actions or their embeddings across a boundary. Only pseudo-gradients (RFC-0003) leave.

## 3. The public probe set $\mathcal{P}$

The probe is the substrate for the frame anchor (RFC-0002 §4.2) and for publicly-recomputable alignment (RFC-0006 §3). Requirements:

- **Public & licensed for redistribution** — contains no participant's private data.
- **Fixed & versioned** — content-hash-pinned; a probe change is a versioned event (it redefines the reference frame and forces re-anchoring).
- **Representative** — spans the modalities/embodiments in the federation enough to anchor a meaningful frame; under-coverage weakens the anchor, over-size raises per-round alignment cost.
- **Landmarks** — a designated subset of $k \ge d$ landmark points with reference targets $t_i = f_{\text{ref}}(p_i)$ from the round-0 encoder (RFC-0002 §4.2 Variant A).
- **Governance** — curated openly; the curator set and the update procedure are declared. (Maps to Tapestry's Data-Governance work group; this is the one shared, agreed artifact in an otherwise data-sovereign system.)

## 4. Provenance commitments (bridge to Phase 2)

To make contributions attributable and, later, provable:

- **Episode hashing** — each participant content-hashes its episodes.
- **Dataset Merkle root** — before an epoch, the participant commits a Merkle root $R_c$ over its episode hashes (`Commitment` message, RFC-0003 §7).
- **Binding** — a contributed pseudo-gradient is associated with the $R_c$ under which it was computed.

In Phase 1 this enables **contribution accounting** (which committed dataset fed which round) and tamper-evidence. In Phase 2 (RFC-0006) it upgrades to a cryptographic claim: *"this update was computed from data under committed root $R_c$."* It proves provenance, **not** data quality or honest computation — state this limit plainly.

## 5. Contribution accounting

Maintain an append-only log: per round, the set of contributing participants, their committed roots, and the resulting global-model hash. This supports credit/governance (who improved the model, on what data) and is the audit substrate the verifiable layer formalizes.

## 6. The shared latent contract (WMCP)

Heterogeneous embodiments can only federate into one model if they agree on the latent interface. **WMCP (WM-RFC-0001)** defines it: the shape/semantics of the latent state every encoder emits and every predictor consumes, and the action-conditioning interface each per-embodiment head must satisfy. This is the type-safety layer that makes cross-silo federation well-posed — the explicit analogue of the fixed vocabulary LLM federation gets for free. Conformance to the WMCP contract is a precondition for joining a Lensemble federation.

## 7. Data-quality expectations

Each participant declares minimal data-quality metadata (modality, embodiment, action-space spec per WMCP, episode count, collection conditions). The federation MAY weight or gate contributions on declared quality; quality enforcement beyond provenance is out of scope for v0.1.
