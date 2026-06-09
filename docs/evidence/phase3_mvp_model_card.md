---
license: apache-2.0
library_name: lensemble
tags:
- federated-learning
- world-model
- jepa
- robotics
- phase3
- mvp
---

# Lensemble MVP — First From-Scratch Distributed JEPA LeWorldModel (converged, gauge-held)

The MVP deliverable of epic [#259](https://github.com/AbdelStark/Lensemble/issues/259): the **first
end-to-end distributed (federated) JEPA-based LeWorldModel trained from scratch (no foundation-model
warm-start)** whose aggregated global **holds its latent gauge** — it does **not** collapse — where the
naive-FedAvg control catastrophically does. Trained on four sovereign SO-100 silos with a held-out fifth
split, on Hugging Face Jobs (`a10g-large`), relaxed-DP (DP-off) probe regime, simulated secure-aggregation.

## Headline result

The M1 anchored federation (strengthened frame anchor pinned to the fixed round-0 reference, live
Procrustes backstop on the encoder terminal frame + predictor, tamed DiLoCo outer step) **prevents the
gauge collapse** that is the #259 root cause. `effective_rank` holds and grows (no collapse to ~1), frame
drift is controlled, and held-out `val_pred` stays **~4 orders of magnitude below** naive-FedAvg.

## Results table (real HF Jobs runs, from-scratch, latent_dim=256, depth=8, 224px, 4 silos)

| control | effective_rank (held-out) | val_pred (held-out) | frame_drift_deg | verdict |
|---|---|---|---|---|
| local-only (per-silo) | **~105** (healthy) | **~0.025** | 180 (inter-silo) | silos learn alone; gauges diverge maximally |
| naive-FedAvg | 1.1 → **~1** (collapse) | 3 → **203 776** (explode) | **180** (every round) | catastrophic gauge collapse |
| **anchored (M1)** | 2.6 → **14.8** (held, grows) | 1.4 → **22.2** (bounded) | 7–124 (controlled) | **gauge held, rank builds, no collapse** |

Pinned immutable revisions: anchored `3c2258ce…` · naive `cd8481c4…` · local-only `9345bc3c…`
(`abdelstark/lensemble-phase3-converged-checkpoint` / `-naive-control` / `-local-only-control`).

## Inference (latent-space, held-out SO-100 silo4 — NO simulator)

The converged model is **used** for multi-step latent prediction + latent-MPC goal-reaching on the held-out
split. It is dramatically more usable than the collapsed naive control:

| control | multistep `val_pred_model` | `skill_vs_identity` | latent-MPC `success_rate` |
|---|---|---|---|
| converged (M1) | **19.2** | 5.3e7 | 0.0 |
| naive-FedAvg | 103 320 | 4.0e11 | 0.0 |

## Honest boundaries

- Convergence is demonstrated in the **gauge sense** (no collapse; `effective_rank` held; drift controlled;
  `val_pred` bounded ≪ naive) — the #259 root cause is **solved**. The aggregated global's prediction
  quality does **not** yet reach the single-silo local-only baseline (`val_pred` ~0.025): under DiLoCo
  separate-averaging of the co-adapted encoder/predictor over heterogeneous silos, representation richness
  (`effective_rank`) and predictability **trade off** — a documented remaining limitation, **not** a
  collapse. The latent-MPC `success_rate` is 0 on this near-static slow-video task for both models (the
  predict-current baseline is very strong on consecutive frames); the falsifying signal is the
  converged-≫-naive contrast on `val_pred` / `effective_rank`.
- Relaxed-DP (DP-off) probe regime for the gauge measurement; DP–utility is a separate thread.
- Latent-space inference only. **Closed-loop physical task-success stays gated** on the unvendored
  `stable-worldmodel` simulator ([#96](https://github.com/AbdelStark/Lensemble/issues/96)).
- Consortium-engineering + from-scratch federated-training evidence — **not** a cryptographic proof of
  honest participant computation; not a paper-scale robotics performance result.

Spec: RFC-0002 (latent gauge), RFC-0003 (federated protocol), RFC-0005 (evaluation), RFC-0010 (artifacts).
