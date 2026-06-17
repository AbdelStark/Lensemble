# Less Surprised — Master Plan (PRIORITY)

> **Project:** **Less Surprised** *(working title; the UI is the "surprise-meter")* — *watch a JEPA world model become measurably **less surprised** after a room of strangers improves it together — without sharing any data.*
> **Parent issue:** [#338](https://github.com/AbdelStark/Lensemble/issues/338)
> **Event:** Codex Hackathon, Paris (Thursday). Full-day, solo-builder, Codex-sponsored, Demo-Night.
> **Priority:** This is the **primary** hackathon deliverable. [Cartographer / #339](../cartographer/) is stretch.
> **Claim discipline (binding):** never "federated training run" / "the room trained the world model." It is a **federated adapter-continuation round on a frozen checkpoint** — only a 12,512-param (0.069%) adapter moves. The seed-robust improvement is **+16.8% mean across 5 seeds, worst +5.4% (seed 2)** — both surfaced, never the mean alone.

---

## 1. One-paragraph pitch

First we run a **clean federated adapter-continuation round** — a room of people each train a tiny adapter on the frozen JEPA world model using their own local trajectory data; nothing leaves their device; the clipped deltas aggregate into one shared revision. Then the **surprise-meter** makes the result visceral: as an agent moves through the TwoRooms world, we show the model's **surprise** — the gap between what it predicted would happen next and what actually happened, measured in latent space — live. Perturb the world and surprise spikes. Then flip **pre- vs post-federation**: the post-federation model is measurably *less surprised* by normal dynamics. Surprise *is* the prediction error the adapter reduced (the certified **+12.3%** on this run; **+16.8% mean / +5.4% worst** across 5 seeds), so the federated story is told by the exact quantity that improved — not a metaphor.

Two milestones:

- **Milestone 0 (must) — a clean federated adapter-continuation round.** The reliable, one-command, end-to-end run + committed audited evidence + rehearsal gate, with the post-round 12,512-float adapter offset exported. Demoable on its own.
- **Milestone 1 (ship) — the surprise-meter.** The live visualization that makes Milestone 0's improvement tangible.

## 1a. Name & the 10-second hook

- **Name:** **Less Surprised** (the meter is the mechanism, not the pitch). Tagline: *"A world model that gets less surprised after a crowd improves it — no data shared, on a laptop, every number audited."*
- **Impact-first hook (replaces the jargon-first open):** *"This model is about to be surprised — watch."* — show the spike **first**, explain JEPA **second**. The full 0:00–0:15 rewrite is in `04-demo-runsheet.md`.

## 1b. Why this wins (Codex / research-sponsor Demo-Night)

Against flashier consumer demos, this entry's edge is **substance the judges can verify**:
1. **Crowd-trained live** — QR-join audience participation is the interactive wow, and it is *real* federation (clipped adapter deltas, nothing leaves the device).
2. **Evidence-audited** — every on-screen number traces to a committed JSON that passes the repo's claim-audit; the anti-hype stance ("here is the worst seed too: +5.4%") is the differentiator at a research-sponsor table.
3. **Runs on a laptop** — ~6 ms/step, WASM-safe; nothing hangs live.
4. **Genuine JEPA world-model substance** — prediction error in latent space, the exact quantity the federation improved.
5. **Built in a Codex loop** — plan → typed contracts → evidence gate, agentically (Decision S9). Foreground it: it is a Codex hackathon.

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
| **S5** | **Pre/post-federation** uses the real adapter offset via `lewm_adapter.mjs::adapterFromInitAndOffset({inputDim:192, hiddenDim:32, initSeed:42, offset})` (object arg, **not** a bare offset); `adapterForward(adapter, x, n)`. | The certified improvement made visible. Verified signatures (`lewm_adapter.mjs:208,231`); a bare-offset call throws. |
| **S6** | **Frame-diff baseline on screen** to prove surprise ≠ motion. | The whole "wow" and the credibility depend on this contrast. |
| **S7** | **Headline number is the certified federated improvement read from evidence at full precision** (`result.relativeImprovement=0.1227556578…` → "+12.3% this run"; `distribution.relativeImprovementMean` → "+16.8% mean"; `distribution.relativeImprovementMin` → "+5.4% worst"). The **live** mean-surprise drop is shown separately, labelled "this run". No new federated claim is derived. | Claim discipline; don't recompute a certified number, don't store a truncated literal (it would fail the contract's own 1e-6 sourcing check). |
| **S8** | **Perturbation-spike is validated in-browser before stage** (onnxruntime is absent offline — Chrome is the only runtime); if the CLS predictor doesn't spike cleanly on teleports, the demo leans on the (certain) OOD-action contrast and the pre/post toggle. | De-risks the empirical unknown (R1), with the correct runtime. |
| **S9** | **Foreground the Codex/agentic-build story.** One closing beat + a DoD item: this was built plan→typed-contracts→evidence-gate in a Codex loop. | It is a Codex hackathon; the agentic-build angle is a scored differentiator and is currently absent from the narrative. |
| **S10** | **The pre/post toggle runs on the held-out probe-pair distribution** (`collectResidentPairs`, seed 991), computed **once at load** — *not* a freshly-rolled live trajectory. So the on-stage drop **is** the certified `baselineMse→adaptedMse` (= +12.3%), guaranteed. SM-3's perturbation spikes stay on the pre-only meter. | The certified +12.3% is only guaranteed in-distribution; a free-running trajectory could show a smaller/negative drop (worst seed +5.4%). Pinning the distribution removes that stage risk (R11). |

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

- [ ] **Milestone 0:** one command (`scripts/surprise/run_clean_round.py`) launches a clean federated adapter-continuation round; `docs/evidence/lewm_tworooms_system_probe.json` regenerates and passes `audit_real_lewm_evidence`; the **12,512-float offset is exported** to a sidecar and a committed copy at `web/surprise-meter/fixtures/adapter_offset.json` (tracked — **not** under gitignored `runs/`); rehearsal gate green.
- [ ] **Milestone 1:** `web/surprise-meter/` renders a live per-step surprise meter over a TwoRooms trajectory, in-browser, on the **WASM** path (no WebGPU dependency; `?ep=wasm` force-WASM hatch confirmed).
- [ ] Perturbation controls spike surprise (validated **in-browser** before stage, R1); frame-diff baseline shows surprise ≠ motion.
- [ ] Pre/post-federation toggle shows the in-distribution surprise drop **on the held-out probe-pair set** (S10), with the certified number (+12.3%) **and the worst-case +5.4%** in the HUD, both sourced at full precision.
- [ ] `docs/evidence/lewm_tworooms_surprise.json` (`lewm-surprise/1`, incl. `federatedSeedWorst`/`federatedSeedStdev`/`warmupSteps`) generated + `tests/ml/test_lewm_surprise.py` green (skip-when-absent); `CHANGELOG.md [Unreleased]` notes the new schema.
- [ ] Claim-discipline checklist (doc `05`) signed off; the Codex/agentic-build beat (S9) is in the script.
- [ ] Rehearsal gate passes; ≤20 s capture + result card exist (card shows +12.3% **and** +5.4% worst).
- [ ] `ruff` + `pyright` clean; touched `pytest` gates green; `check_docs_links.py` + `mkdocs build --strict` pass.
