# RFC-0001 — Architecture & System Overview

| | |
|---|---|
| **RFC** | 0001 |
| **Title** | Architecture & System Overview |
| **Status** | Draft |
| **Track** | Standards |
| **Author** | Abdelhamid Bakhta (@AbdelStark) |
| **Requires** | — |
| **Informs** | RFC-0002, 0003, 0004, 0005, 0006 |
| **Date** | June 2026 |

## 1. Scope

This RFC specifies *what Lensemble builds*: the model, which parts are federated versus local, the training topology, and the trust boundaries. Mechanisms are deferred to the focused RFCs: the gauge solution to 0002, the wire protocol to 0003, data/provenance to 0004, evaluation to 0005, verifiability to 0006.

## 2. Model

Lensemble trains an **action-conditioned JEPA** used as a latent world model for planning, following the V-JEPA 2-AC shape but **co-training the encoder** (Fork B) rather than freezing it.

- **Encoder** $f_\theta:\text{video clip}\to\mathbb{R}^{N\times d}$ — a video Vision Transformer, **warm-started from released V-JEPA 2 weights**. Co-trained under SIGReg (RFC-0002 §3). The warm-start is also the gauge anchor at $t{=}0$ (RFC-0002 §4).
- **Latent predictor** $g_\phi$ — a compact transformer predicting future latents autoregressively, conditioned on an action embedding (LeWM `ARPredictor` shape).
- **Action encoder / embodiment head** $h_\psi^{(c)}$ — **per-participant**, mapping that embodiment's action space into the shared latent-conditioning space. Never averaged (action spaces differ: a quadruped ≠ a 7-DoF arm).
- **Objective** (per local step): next-embedding prediction loss + **SIGReg** (collapse prevention during co-training) + **frame-anchor loss** (RFC-0002 §4).
- **Planning / evaluation**: latent model-predictive control — CEM / iCEM / MPPI minimizing an $L_1$ goal-energy in latent space, exactly as `stable-worldmodel` provides (RFC-0005).

The shared latent interface — what every encoder emits and every predictor consumes — is the **WMCP** contract (RFC-0004 §6). It is what makes heterogeneous-embodiment federation type-safe; it is the role the fixed token vocabulary plays for free in LLM federation.

## 3. Federation map

| Component | Disposition | Why |
|---|---|---|
| Encoder backbone $f_\theta$ | **Federated** (gauge-controlled) | Shared physics; the point of the project |
| Predictor core $g_\phi$ | **Federated** | Shared dynamics; frame-pinned so averaging is valid |
| Action encoder / heads $h_\psi^{(c)}$ | **Local — personalized** | Embodiment-specific action spaces |
| SIGReg sketch matrix $A$ | **Shared per round** (broadcast seed) | Objective consistency (RFC-0002 §4.1) |
| Public probe $\mathcal{P}$ + landmark targets | **Shared, fixed, hash-pinned** | The manufactured frame anchor (RFC-0002 §4.2) |
| Raw trajectories | **Never leaves the boundary** | Sovereignty |

## 4. Training topology

A two-level nesting (this is where the standard "distributed training" stack is the *inner* loop, not the contribution):

- **Inner — intra-participant, for scale.** Within a participant, standard FSDP / tensor / context parallelism trains the warm-started 1.2B-class model. SIGReg projection statistics are reduced *within* this trust domain freely (RFC-0002 §4.1). This is the only place the large-model-parallelism playbook (INTELLECT-1 / PRIME) applies.
- **Outer — inter-participant, for sovereignty.** DiLoCo: each participant runs $H$ local steps (inner AdamW), then an outer Nesterov step synchronizes pseudo-gradients $\Delta_c=\theta_c^{\text{local}}-\theta^{\text{global}}$ (RFC-0003 §3). Only $\Delta_c$ crosses the boundary, via secure aggregation + DP.

## 5. Trust boundaries

```
┌── Participant c (sovereign) ─────────────────────────┐
│  raw trajectories  ──►  local train (inner-parallel) │
│        │ (never leaves)         │                     │
│        ▼                        ▼                     │
│  Merkle commitment R_c     pseudo-gradient Δ_c        │
└───────────────│──────────────────│───────────────────┘
                │                  │  (DP-clipped + noised)
                ▼                  ▼
        ┌──────────── Coordinator / secure aggregator ─────────┐
        │  Σ_c Δ_c  (individual Δ_c never revealed)             │
        │  outer Nesterov step → θ^{global}_{t+1} (hash-committed)│
        │  frame re-alignment on public probe (recomputable)    │
        └───────────────────────────────────────────────────────┘
```

What crosses a boundary: model deltas (privacy-protected), dataset commitments, and shared coordination state (sketch seed, probe hash, global-model hash). What never crosses: raw observations, actions, or embeddings of private data.

## 6. Staged plan

| Stage | Goal | Compute |
|---|---|---|
| **A** | Single-site, warm-started, modest size (ViT-L/~300 M) end-to-end SIGReg + AC predictor on pooled robot data. Centralized upper bound; validate objective + MPC eval. | Handful of GPUs, days |
| **B** | **Simulated federation** on one cluster: $N$ silos, non-IID partition, DiLoCo + frame anchor, full ablation ladder + the frame-drift diagnostic (RFC-0005). *The scientific core / the paper.* | Same hardware |
| **C** | **Two real sovereign nodes**: data never crosses, one model emerges, secure aggregation + DP. The Tapestry-for-physical-AI demonstration. | Two small clusters |
| **D** | **Verifiable layer** (RFC-0006): aggregation STARK + provenance commitments. | + prover |
| **E** | **Scale** to V-JEPA-2 class (1.2 B); optionally federated encoder pretraining from scratch. *Optional headline.* | INTELLECT-class program |

Warm-starting keeps A–D modest and runnable. Stage E (own large-scale federated video pretrain) is the expensive frontier — out of the Phase-1 / paper critical path.

## 7. Honest risks

- **Anchor strength** is the central knob (RFC-0002 §4.5): too strong clamps quality to the reference encoder; too weak lets the frame drift.
- **SIGReg at video-WM scale** is demonstrated only to ViT-H on images; Stage A de-risks the objective before federation.
- **Personalization tension**: very heterogeneous silos may not share one global encoder; the shared-backbone + per-embodiment-head split is the hedge, boundary is empirical.
- **Fallback**: if Fork-B gauge control is unstable at scale, Fork A (frozen shared encoder, federate the predictor) recovers a clean federation and most of the sovereignty story, minus the end-to-end novelty (RFC-0002 §7).

## 8. Dependencies

V-JEPA 2 (encoder warm-start, AC recipe) · LeJEPA/LeWM (SIGReg objective; `jepa-rs`/`lewm-rs` as a verifiable-by-construction reference path) · stable-worldmodel (data, envs, MPC eval, `lance`/`hdf5`/`lerobot`) · WMCP (latent contract) · DiLoCo/PRIME (inner/outer optimizer, elastic fault tolerance) · Stwo (Phase-2 prover).
