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

# Lensemble MVP — Corrected SO-100 Gauge-Only Evidence

The MVP deliverable of epic [#259](https://github.com/AbdelStark/Lensemble/issues/259): the **first
end-to-end distributed (federated) JEPA-based LeWorldModel trained from scratch (no foundation-model
warm-start)** whose M1 anchored run reduces the naive-FedAvg latent-gauge failure. This card is corrected:
the SO-100 checkpoint is **gauge evidence only**, not a downstream-useful world model. The held-out
representation has magnitude collapse (`~7.5e-6` latent variance; `thoughts/collapse_fix_probe.py`), and
the central ceiling probe (`thoughts/central_ceiling_probe.py`) shows the downstream ceiling is not cleared.
Trained on four sovereign SO-100 silos with a held-out fifth split, on Hugging Face Jobs (`a10g-large`),
relaxed-DP (DP-off) probe regime, simulated secure-aggregation.

## Headline result

The M1 anchored federation (strengthened frame anchor pinned to the fixed round-0 reference, live
Procrustes backstop on the encoder terminal frame + predictor, tamed DiLoCo outer step) **prevents the
naive-FedAvg gauge collapse**. That result is narrower than the original card claimed: `effective_rank` is
scale-invariant and blind to the held-out magnitude collapse, and `skill_vs_identity` is gameable. In plain
text: skill_vs_identity is gameable; effective_rank is scale-invariant. The usable-world-model claim is not
supported on SO-100; RFC-0017's dynamic-env pivot is the ground-truth path.

## Results table (real HF Jobs runs, from-scratch, latent_dim=256, depth=8, 224px, 4 silos)

| control | effective_rank (held-out) | val_pred (held-out) | frame_drift_deg | verdict |
|---|---|---|---|---|
| local-only (per-silo) | **~105** (healthy) | **~0.025** | 180 (inter-silo) | silos learn alone; gauges diverge maximally |
| naive-FedAvg | 1.1 → **~1** (collapse) | 3 → **203 776** (explode) | **180** (every round) | catastrophic gauge collapse |
| **anchored (M1)** | 2.6 → **14.8** (held, grows) | 1.4 → **22.2** (bounded) | 7–124 (controlled) | gauge held, but downstream usefulness not shown |

Pinned immutable revisions: anchored `3c2258ce…` · naive `cd8481c4…` · local-only `9345bc3c…`
(`abdelstark/lensemble-phase3-converged-checkpoint` / `-naive-control` / `-local-only-control`).

## Inference Correction (latent-space, held-out SO-100 silo4 — NO simulator)

The original inference framing overclaimed. The held-out SO-100 report is now a proxy audit, not a
usefulness result. `skill_vs_identity` is gameable, `effective_rank` is scale-invariant, and
latent-MPC `success_rate=0.0` is a negative result rather than a near-static-video success story:

| control | multistep `val_pred_model` | `skill_vs_identity` | latent-MPC `success_rate` |
|---|---|---|---|
| anchored (M1) | **19.2** | 5.3e7 | 0.0 |
| naive-FedAvg | 103 320 | 4.0e11 | 0.0 |

## Honest boundaries

- Convergence is demonstrated only in the **gauge sense**: M1 improves over naive-FedAvg on frame drift and
  proxy `val_pred`. It does **not** demonstrate SO-100 downstream usefulness.
- Held-out magnitude collapse is disclosed explicitly: `~7.5e-6` latent variance in
  `thoughts/collapse_fix_probe.py`. The central ceiling probe in `thoughts/central_ceiling_probe.py`
  shows the checkpoint does not clear a downstream usefulness ceiling.
- `skill_vs_identity` is **gameable** and `effective_rank` is **scale-invariant**; neither can be the binding
  usefulness metric. The latent-MPC `success_rate` is 0 and is reported as a negative result.
- The dynamic-env RFC-0017 pivot replaces the SO-100 proxy story with a ground-truth `state_probe_r2`
  acceptance gate.
- Relaxed-DP (DP-off) probe regime for the gauge measurement; DP–utility is a separate thread.
- Latent-space inference only. **Closed-loop physical task-success stays gated** on the unvendored
  `stable-worldmodel` simulator ([#96](https://github.com/AbdelStark/Lensemble/issues/96)).
- Consortium-engineering + from-scratch federated-training evidence — **not** a cryptographic proof of
  honest participant computation; not a paper-scale robotics performance result.

Spec: RFC-0002 (latent gauge), RFC-0003 (federated protocol), RFC-0005 (evaluation), RFC-0010 (artifacts).
