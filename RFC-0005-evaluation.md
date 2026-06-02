# RFC-0005 — Evaluation & Benchmark Protocol

| | |
|---|---|
| **RFC** | 0005 |
| **Title** | Evaluation & Benchmark Protocol |
| **Status** | Draft |
| **Track** | Standards |
| **Author** | Abdelhamid Bakhta (@AbdelStark) |
| **Requires** | RFC-0001, RFC-0002 |
| **Date** | June 2026 |

> This RFC defines how Lensemble proves its claims. It is the backbone of the paper. The central result is not "we trained distributed" (solved prior art) but "we measured and controlled the latent gauge under federation, and federation closes the gap to centralized without moving data."

## 1. Claims to test

1. **Naive end-to-end `FedAvg` of a JEPA degrades / collapses** under non-IID silos (the gauge in action).
2. **Frame anchoring (RFC-0002 §4) holds the latent frame pinned** across participants over training.
3. **Anchored federation closes most of the centralized–local gap** on downstream planning, *without moving data*.
4. Robustness across non-IID severity and participant count.

## 2. Headline diagnostic — latent frame drift

The novel measurement at the center of the paper. On the fixed public probe $\mathcal{P}$, after each round compute, for each pair of participants $(c,c')$, the optimal Procrustes rotation between $f_{\theta_c}(\mathcal{P})$ and $f_{\theta_{c'}}(\mathcal{P})$, and report:

- **mean inter-participant rotation angle** (or Procrustes residual) over training, and
- the same against the global model (drift from consensus).

Expected figure: naive `FedAvg` curves diverge (frames rotate apart); the anchored configuration stays flat/low. To our knowledge this is the first measurement of latent frame-drift under federated self-supervision; it stands on its own as a contribution.

## 3. Downstream metric — planning success

The thing that ultimately matters. Via `stable-worldmodel`:

- Use the trained model as the cost/world model for **latent MPC** (CEM / iCEM / MPPI; $L_1$ goal-energy in latent space).
- Report **success rate** (`world.evaluate`) on **held-out** environments and held-out factors-of-variation, with goal-image specification.
- Report planning cost (samples, time/action) for parity with baselines.

## 4. Supporting metrics

- **Representation quality** — linear / attentive probe accuracy on a held-out downstream task on the frozen learned encoder.
- **Collapse** — effective dimension of the embedding covariance (eigenspectrum / rank); guards against silent partial collapse that success rate alone might mask.
- **Communication** — total bytes and rounds (to contextualize against centralized cost and to support the DiLoCo efficiency claim).

## 5. Baselines

| Baseline | Role |
|---|---|
| **Centralized-pooled** (all silo data in one place, end-to-end) | **Upper bound** — what federation aspires to |
| **Local-only** (each silo trains alone, no sync) | **Lower bound** — value of federating at all |
| **Naive end-to-end `FedAvg`** (no gauge control) | **Negative control** — the collapse/divergence the design fixes |
| **Fork A** (frozen shared encoder, federate predictor only) | Reference point; the safe degrade (RFC-0002 §7) |

The headline claim is quantified as the fraction of the **centralized − local** gap recovered by anchored federation.

## 6. Ablation ladder (the core experiment)

Each rung adds one mechanism; report all three metric families (frame drift, MPC success, collapse) at each:

1. Naive end-to-end `FedAvg`.
2. **+** shared sketch matrix (RFC-0002 §4.1).
3. **+** Procrustes align-then-average (RFC-0002 §4.3).
4. **+** frame anchor loss — landmark (Variant A) or rotational (Variant B) (RFC-0002 §4.2). **← expected recommended configuration.**
5. **+** function-space distillation (RFC-0002 §4.4).

Also sweep the central knob $\lambda_{\text{anc}}$ (RFC-0002 §4.5): frame drift and MPC success vs $\lambda_{\text{anc}}$, to locate the "pin frame, not content" sweet spot.

## 7. Non-IID severity & scale sweeps

- **Non-IID** — partition a multi-environment / multi-FoV corpus across silos by factor-of-variation or by embodiment; `stable-worldmodel`'s factors-of-variation give *controlled, reproducible* heterogeneity. Sweep from near-IID to strongly non-IID; report degradation curves for each ladder rung.
- **Participant count $C$** — vary $C$ and the inner horizon $H$; report effect on convergence and frame drift (validates DiLoCo robustness in the JEPA setting).
- **Scale** — repeat the key rungs at increasing encoder size toward V-JEPA-2 class (Stage E) to show the recipe holds.

## 8. Reproducibility & reporting

- Hydra configs, fixed seeds, pinned probe hash and sketch seeds per round.
- Report: hardware, rounds, $H$, communication bytes, DP $(\varepsilon,\delta)$, $\lambda$ settings.
- Release checkpoints + configs in the reference repo (RFC-0001 §8) so the centralized/local/federated triple is reproducible end to end.

## 9. Success criteria (paper-grade)

- Claim 1 demonstrated (naive `FedAvg` measurably worse on frame drift and MPC success).
- Claim 2 demonstrated (anchored frame drift flat where naive diverges).
- Claim 3 quantified (≥ a stated fraction of the centralized–local gap recovered, no data moved).
- Claims 1–3 hold across at least one non-IID severity sweep and one scale step.
