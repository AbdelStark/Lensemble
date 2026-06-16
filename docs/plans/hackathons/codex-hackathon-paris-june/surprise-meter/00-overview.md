# Surprise-meter — Master Plan (PRIORITY)

> **Project:** Surprise-meter — *watch a JEPA world model become measurably **less surprised** after a room of people trains it together.*
> **Parent issue:** [#338](https://github.com/AbdelStark/Lensemble/issues/338)
> **Event:** Codex Hackathon, Paris (Thursday). Full-day, solo-builder, Codex-sponsored, Demo-Night.
> **Priority:** This is the **primary** hackathon deliverable. [Cartographer / #339](../cartographer/) is stretch.

---

## 1. One-paragraph pitch

First we run a **clean federated training run** — a room of people each train a tiny adapter on the frozen JEPA world model using their own local trajectory data; nothing leaves their device; the clipped deltas aggregate into one shared revision. Then the **Surprise-meter** makes the result visceral: as an agent moves through the TwoRooms world, we show the model's **surprise** — the gap between what it predicted would happen next and what actually happened, measured in latent space — live. Perturb the world and surprise spikes. Then flip **pre- vs post-federation**: the post-federation model is measurably *less surprised* by normal dynamics. Surprise *is* the prediction error the adapter reduced (the certified +12.3% / +16.8%), so the federated story is told by the exact quantity that improved — not a metaphor.

Two milestones:

- **Milestone 0 (must) — a clean federated training run.** The reliable, one-command, end-to-end federated **adapter-continuation** run + committed evidence + rehearsal gate. Demoable on its own.
- **Milestone 1 (ship) — the Surprise-meter.** The live visualization that makes Milestone 0's improvement tangible.

---

## 2. The two corrections this plan bakes in

The earlier #338 issue draft was written before the feasibility spike. The spike changed two things — both are now load-bearing:

1. **Surprise is a scalar per frame, not a per-patch spatial heatmap.** The exported model is a **CLS-latent** world model: one 192-d vector per frame, no per-patch prediction head. So the visual is a **surprise meter / oscilloscope** (a live prediction-error signal that spikes on perturbation), not a heatmap over the image. This is honest, fast, and ties exactly to the federated number. A true spatial heatmap would require re-exporting a per-patch encoder *and* training a per-patch predictor — out of scope.
2. **Claim language is "federated adapter continuation on a frozen checkpoint,"** never "federated world-model training." The #335 spike's verdict is explicit: full-model federated training is a GPU-scale research problem and **NO-GO in-browser**. The backbone stays frozen; only the 12,512-param (0.069%) adapter trains and federates.

---

## 3. Why this is low-risk and fast

- The surprise quantity is **already implemented**: `web/federated-demo/lewm_probe.mjs::probeAdapterOffset` computes `baselineMse = MSE(frozen_predictor_output, true_next_latent)` and the adapted version. That **is** per-pair surprise. Milestone 1 is mostly: run it **per-step over a live trajectory** + visualize + add perturbation + a pre/post toggle.
- The federated run is **already implemented and certified**: `lensemble/demo/system_probe.py` + `web/federated-demo/` (QR join, real adapter gradients, clipped-delta aggregation, system-composed probe), seed-robust evidence in `docs/evidence/`. Milestone 0 is mostly making it a **clean, one-command, rehearsed** demo.
- Inference is **~6 ms/step on CPU** (spike); in-browser via the existing `lewm_runtime.mjs` (WebGPU EP, WASM fallback). Real-time is free.
- The env, runtime, adapter, and probe modules already exist in `web/federated-demo/` and are pixel-exact / hash-checked.

---

## 4. Scope

### In scope
1. **Milestone 0:** a documented, one-command, end-to-end federated adapter-continuation run (headless rehearsal + live browser QR), with committed, audited evidence.
2. **Milestone 1:** an in-browser **per-step surprise engine** (encode frame → predict next latent → compare to actual next latent → scalar MSE), over a live TwoRooms trajectory, using the exported ONNX graphs.
3. A **surprise visualization**: a live meter/oscilloscope + the env canvas, with the agent/scene tinted by current surprise.
4. **Perturbation controls** (teleport the agent, force an off-distribution action, push through the wall) that make surprise spike, with a **frame-diff baseline** panel proving *surprise ≠ motion*.
5. A **pre- vs post-federation toggle** (apply the Milestone-0 adapter offset) showing surprise on in-distribution dynamics drops after the room trains it.
6. An **evidence JSON** (`lewm-surprise/1`) + producer + `tests/ml` test, sourcing the federated number from existing evidence.
7. **Demo-day** rehearsal gate, pre-baked fallback, capture assets.

### Out of scope (non-goals)
- No per-patch spatial heatmap (no per-patch head; §2).
- No full-model / in-browser world-model training (spike #335 NO-GO).
- No webcam general-encoder surprise as a *claim* (OOD for the TwoRooms backbone; allowed only as a clearly-labelled stretch easter egg).
- No secure-aggregation / DP claims on this path.
- No #339 manifold/planning (separate stretch plan).

---

## 5. Locked decisions

| # | Decision | Rationale |
|---|---|---|
| **S1** | **Scalar surprise** = `MSE(predicted_next_latent, actual_next_latent)` per step, reusing the `lewm_probe.mjs` math. | The model is CLS-latent; no spatial head. Already implemented. |
| **S2** | **Federation = adapter continuation on a frozen checkpoint.** | Honest, certified, laptop-demonstrable (spike #335 NO-GO on full-model in-browser). |
| **S3** | **Milestone 0 ships before Milestone 1.** A clean federated run is the priority and a standalone demo. | User priority; de-risks the day. |
| **S4** | **Act-2 runs in-browser** via the existing `lewm_runtime.mjs` (WebGPU EP, WASM fallback) over a **new page** `web/surprise-meter/` that imports the existing `federated-demo` modules (env, runtime, adapter, probe). | Reuse pixel-exact env + hash-checked ONNX + the proven adapter math; cross-page import is an established pattern. |
| **S5** | **Pre/post-federation** uses the real adapter offset via `lewm_adapter.mjs::adapterFromInitAndOffset`; surprise computed both ways live. | The certified improvement made visible per-step. |
| **S6** | **Frame-diff baseline on screen** to prove surprise ≠ motion. | The whole "wow" and the credibility depend on this contrast. |
| **S7** | **Headline number is the certified federated improvement** (+12.3% / +16.8%), read from existing evidence — plus live mean-surprise pre/post. No new federated claim is derived. | Claim discipline; don't recompute a certified number. |
| **S8** | **Perturbation-spike is validated offline first**; if the CLS predictor doesn't spike cleanly on teleports, the demo leans on the (certain) in-distribution-vs-OOD-action contrast and the pre/post toggle. | De-risks an empirical unknown (R1). |

---

## 6. Architecture (high level)

```
 MILESTONE 0 (federated run, mostly exists)            MILESTONE 1 (surprise-meter, new page)
 ┌───────────────────────────────────────┐   ┌──────────────────────────────────────────────┐
 │ browser federated-demo (QR join)       │   │ web/surprise-meter/  (imports federated-demo) │
 │  lewm_local_trainer → adapter delta     │   │  tworooms_env.mjs ── live trajectory          │
 │  → FederatedDemoService.submit_update   │   │  lewm_runtime.mjs ── encode/predict (ONNX)     │
 │  → _close_round_lewm (aggregate)        │──▶│  per-step surprise = MSE(pred_next, actual)    │
 │  → model revision + adapterState offset │   │  lewm_adapter.mjs ── pre/post offset           │
 │  system_probe.py (headless, evidence)   │   │  → meter/oscilloscope + env tint               │
 │  → docs/evidence/..._system_probe.json  │   │  → perturbation controls + frame-diff baseline │
 └───────────────────────────────────────┘   │  → pre/post-federation toggle + HUD            │
                                              └──────────────────────────────────────────────┘
```

Live on stage: the federated round (Milestone 0) in `web/federated-demo/`, then the surprise meter in `web/surprise-meter/` on the resulting revision (or a pre-baked offset fallback). Component detail: `01-architecture.md`. JSON/config: `02-data-contracts.md`.

---

## 7. Workstreams → child issues

| WS | Issue | Milestone | Depends on |
|----|-------|-----------|-----------|
| WS0 Clean federated run | SM-1 | 0 (must) | — |
| WS1 In-browser surprise engine | SM-2 | 1 | SM-1 (for ONNX/runtime parity) |
| WS2 Surprise UI + perturbation + frame-diff | SM-3 | 1 | SM-2 |
| WS3 Pre/post-federation toggle | SM-4 | 1 | SM-1, SM-2 |
| WS4 Evidence JSON + test | SM-5 | 1 | SM-2, SM-4 |
| WS5 Rehearsal + fallback + capture | SM-6 | 1 | SM-3, SM-4 |

Critical path: **SM-1 → SM-2 → SM-3 → SM-6**. Full breakdown: `03-workstreams-and-issues.md`.

---

## 8. Day-of timeline (target)

| Block | Work |
|-------|------|
| Pre-hack (Wed eve) | SM-1: get the clean federated run one-command + rehearsal green; pre-bake a fallback post-federation offset. Scaffold `web/surprise-meter/` page. |
| 09:00–11:00 | SM-2 surprise engine (per-step loop, parity vs `lewm_probe.mjs`). |
| 11:00–13:00 | SM-3 UI: meter/oscilloscope + env tint + perturbation + frame-diff baseline. |
| 13:00–14:30 | SM-4 pre/post-federation toggle on the real revision. |
| 14:30–16:00 | SM-5 evidence + test; polish. |
| 16:00–17:30 | SM-6 rehearsal, capture clip, prep Demo-Night. |
| 19:00–22:00 | Demo Night. |

Hard rule: **Milestone 0 alone is a valid demo at every hour.** Surprise-meter only adds to it.

---

## 9. Definition of done (project-level)

- [ ] **Milestone 0:** one command launches a clean federated adapter-continuation run; `docs/evidence/lewm_tworooms_system_probe.json` regenerates and passes `audit_real_lewm_evidence`; rehearsal gate green.
- [ ] **Milestone 1:** `web/surprise-meter/` renders a live per-step surprise meter over a TwoRooms trajectory, in-browser.
- [ ] Perturbation controls spike surprise; frame-diff baseline shows surprise ≠ motion.
- [ ] Pre/post-federation toggle shows in-distribution surprise drop, with the certified number in the HUD.
- [ ] `docs/evidence/lewm_tworooms_surprise.json` (`lewm-surprise/1`) generated + `tests/ml/test_lewm_surprise.py` green.
- [ ] Claim-discipline checklist (doc `05`) signed off.
- [ ] Rehearsal gate passes; ≤20 s capture + result card exist.
- [ ] `ruff` + `pyright` clean; touched `pytest` gates green; `check_docs_links.py` + `mkdocs build --strict` pass.
