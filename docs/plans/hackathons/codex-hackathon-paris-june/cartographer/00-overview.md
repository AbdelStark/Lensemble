# Cartographer — Master Plan

> **Project:** Cartographer — *map the latent manifold a room of people trained together, and watch it plan.*
> **Parent issue:** [#339](https://github.com/AbdelStark/Lensemble/issues/339)
> **Event:** Codex Hackathon, Paris (Thursday). Full-day, solo-builder, Codex-sponsored, Demo-Night format.
> **Constraints (from the builder):** viral-first; runs on a laptop (CPU; WebGPU allowed); builds on the existing LeWM / Lensemble federated stack; JEPA at the core.

---

## 1. One-paragraph pitch

We run a **live federated adapter-continuation round** on a frozen JEPA world model — a room of people each train a tiny adapter on their own local trajectory data, nothing leaves their device, and the clipped deltas aggregate into one shared revision. Then **Cartographer** turns that exact model into an interactive, WebGPU-rendered *map of its imagination*: a rotating 3D point cloud of the world model's latent states, with **MPC planning trajectories** igniting across the manifold toward a goal, a **healthy-vs-collapsed** toggle that shows the anti-collapse "representation physics" the project is built on, and a **before-vs-after-federation** toggle that makes the room's contribution visible. Every quantitative claim on screen is backed by a generated evidence JSON that passes the repo's claim-audit discipline.

Two acts:

- **Act 1 — the backbone (substance):** a federated adapter-continuation round on the frozen LeWorldModel TwoRooms checkpoint. *The room trains it together; no data is shared.*
- **Act 2 — the cherry (the wow):** the latent-manifold + planning viewer that makes Act 1's result legible and beautiful.

---

## 2. Why Cartographer (vs. the two sibling ideas)

Cartographer is issue [#339](https://github.com/AbdelStark/Lensemble/issues/339), chosen over [#337 Latent Genie](https://github.com/AbdelStark/Lensemble/issues/337) and [#338 Surprise-meter](https://github.com/AbdelStark/Lensemble/issues/338). The feasibility spike (see `docs/plans/.../05` and the issue comments) established:

- The exported world model is a **global / CLS-latent** model: one **192-dim** vector per frame, **3-frame** predictor window, **no latent→pixel decoder**. This kills the "decoded dream frames" of #337 and the "per-patch spatial heatmap" of #338 without new model training.
- Cartographer has **no architecture gap**: 192-d CLS latents are exactly what we project to 3D; the heavy compute is **pre-baked offline** and the laptop only renders a static WebGPU scene, so **nothing can hang live on stage**. It is the lowest-risk Act 2 and still produces a screensaver-grade, shareable visual.

The one real build item Cartographer adds is an **instrumented MPC planner** that exposes candidate trajectories (the stock `eval/mpc.py` returns only the winning plan). That is the highest-value, highest-uncertainty piece and is scoped as its own child issue.

---

## 3. Scope

### In scope
1. A headless **latent-harvest pipeline** over the real cached TwoRooms dataset (`~/.cache/lensemble-lewm/tworoom.h5`, 12.7 GB, 10k episodes) → 192-d CLS latents + actions + episode metadata.
2. An **instrumented MPC planner** that captures per-iteration candidate latent trajectories, costs, elites, and the chosen plan, over a LeWMTwoRooms 3-frame-history dynamics closure.
3. A **federated before/after producer** that runs one demo-path adapter round (reusing `FederatedDemoService` / `system_probe`) and applies the resulting offset to produce pre- vs post-federation predicted latents.
4. A **projection + gauge-alignment** stage: deterministic PCA to 3D, with `gauge.procrustes.procrustes_align` keeping the pre/post clouds in a common frame, plus structure metrics (`effective_rank`, `effective_dim`, `sigreg_statistic`, `state_probe_r2`).
5. An **honest collapse counterfactual** (synthetic rank-1 / magnitude collapse, clearly labelled) for the healthy-vs-collapsed visual.
6. A **bake orchestrator** that emits a single deterministic, hash-stamped `manifold.json` per the data contract (doc `02`).
7. A new **WebGPU / Three.js viewer** page `web/latent-manifold-viewer/` (templated from `web/dynamic-env-demo/`), served by the existing static server.
8. An **evidence JSON** (`lewm-manifold/1`) + producer script + `tests/ml` test + docs wiring.
9. **Demo-day** rehearsal gate, pre-baked fallback bundle, and capture assets.

### Out of scope (explicit non-goals)
- No new world-model training. The backbone stays **frozen**; only the 12,512-param adapter moves.
- No latent→pixel decoder. The viewer renders **latent geometry**, never reconstructed frames.
- No head-to-head "latent vs pixel-rollout compute" benchmark — there is no pixel-rollout world model in-repo to compare against (see Decision **D7**; this would be a fabricated number).
- No secure-aggregation / DP claims on the demo path (it has neither; AGENTS.md §Claim Discipline).
- No multi-operator decentralized run (issue #331, deferred).

---

## 4. Locked decisions

| # | Decision | Rationale |
|---|---|---|
| **D1** | **Use the demo-path adapter** (`FederatedDemoService`, frozen backbone + 12.5k residual adapter), **not** the `Coordinator` full-model path. | Honest, claim-safe, fast, and matches the project's only system-composed credible result. Claim language is fixed: *"federated adapter continuation on a frozen checkpoint."* |
| **D2** | **Harvest latents from the real cached H5 dataset** via a new function alongside `eval/lewm_tworooms_probe_pairs.build_probe_split`. | The TwoRooms env is JS-only; the dataset gives real episodes (frames+actions) with zero renderer-port cost. |
| **D3** | **Before/after = two adapter states** applied to frozen predictor outputs (pre = identity; post = `z + adapter(z)` with federated offset). The measured delta reuses the **existing system-composed probe** (+12.3% committed / +16.8% seed-mean). | Don't re-derive a number the repo already certifies; just visualize it. |
| **D4** | **Build an instrumented planner** maintaining a 3-frame ring buffer for LeWMTwoRooms dynamics; cost = accumulated **L1** in latent space to goal latent (match `mpc.py`). | Stock planner discards candidates and assumes the harness's spatial-token predictor; both must be addressed. |
| **D5** | **Deterministic PCA to 3D** (no UMAP dependency in the critical path; UMAP is a stretch); cross-model stability via `procrustes_align` on a shared landmark set; report variance-explained. | Reproducible, dependency-light, honest about being a projection. |
| **D6** | **Synthesize the collapse counterfactual** (rank-1 and magnitude-collapse), labelled "synthetic illustration"; compute real `effective_rank`/`effective_dim` on both. Ablation naive-FedAvg rung is a stretch. | No collapse generator exists; synthetic is cheapest and, when labelled, honest. Real healthy eff-rank ≈ **9.86 / 192**; rank-1 ≈ **1.0** — a true, measured contrast. |
| **D7** | **Headline numbers are measured, not comparative-to-pixels:** (a) per-step latent inference latency (~6 ms CPU from the spike) and planner wall-time; (b) TwoRooms goal-reach success rate under latent MPC; (c) federated held-out improvement (+12.3%/+16.8%); (d) eff-rank healthy vs collapsed. "Latent, not pixels" is **architectural narrative**, not a benchmarked ratio. | Avoids a fabricated benchmark; everything on screen is reproducible. |
| **D8** | **New self-contained viewer** `web/latent-manifold-viewer/`, templated from `dynamic-env-demo/`, **Three.js vendored** into `vendor/`, default **fully pre-baked** (no live model dependency). Optional live ONNX mini-inference is a stretch. | Bulletproof on stage; reuses existing static serving with zero server changes. |
| **D9** | **New evidence schema `lewm-manifold/1`**, producer `scripts/lewm_manifold_check.py` → `docs/evidence/lewm_tworooms_manifold.json`, with explicit `nonClaims`, validated by `tests/ml/test_lewm_manifold.py`. New schema → add a `CHANGELOG.md [Unreleased]` entry. | Matches repo evidence discipline (AGENTS.md **"Claim Discipline"** + **"Working Loop"** named sections — AGENTS.md has no numbered §; do not hand-edit generated evidence); not a `demo-evidence/1` real-mode bundle, so no `audit_real_lewm_evidence` requirement, but `nonClaims` is mandatory. The federated numbers are sourced at **full precision** (`relativeImprovement=0.1227556578424805`), not the display rounding. |
| **D10** | **Bake orchestrator emits one hash-stamped `manifold.json`**, referencing the evidence file's checkpoint hash and the export-graph hashes, so the viewer can display provenance. | Provenance = credibility; reuses the `lewm-browser-export/1` hash-binding pattern. |

---

## 5. High-level architecture

```
                       OFFLINE BAKE (Python, CPU, pre-stage)                         LIVE (browser, WebGPU)
 ┌────────────────────────────────────────────────────────────────────┐    ┌──────────────────────────────┐
 │ tworoom.h5 ──harvest──▶ latents(192-d)+actions  ─────────┐          │    │  web/latent-manifold-viewer  │
 │ frozen ckpt ─load─┘                                       │          │    │   index.html + app.mjs       │
 │                                                           ▼          │    │   vendor/three.module.min.js │
 │ FederatedDemoService ──1 round──▶ adapter offset ──apply──▶ pre/post │    │                              │
 │                                                  predicted latents   │    │   fetch(./data/manifold.json)│
 │ instrumented MPC ──plan──▶ candidate+chosen latent trajectories ─────┼──▶ │   → point cloud              │
 │ collapse synth ──▶ collapsed latents + metrics                       │    │   → MPC plan trails          │
 │ projection+procrustes ──▶ 3D coords (pre/post/collapsed, aligned)    │    │   → healthy/collapsed toggle │
 │ metrics: eff_rank/eff_dim/sigreg/probe_r2                            │    │   → pre/post-fed toggle      │
 │                                                                      │    │   → metrics HUD + evidence   │
 │ bake orchestrator ──▶ manifold.json (hash-stamped) ──────────────────┼──▶ │     provenance card          │
 │ evidence producer ──▶ docs/evidence/lewm_tworooms_manifold.json      │    └──────────────────────────────┘
 └──────────────────────────────────────────────────────────────────────┘
```

The **only** thing that runs live on stage is the browser rendering a static JSON + (optionally) the live Act-1 federation round via the existing `web/federated-demo/`. The manifold can be re-baked after the live round, or shipped pre-baked as a fallback (Decision D8, runsheet doc `04`).

Detailed component design: doc `01-architecture.md`. Exact JSON shapes: doc `02-data-contracts.md`.

---

## 6. Workstreams → child issues

| WS | Child issue | Depends on |
|----|-------------|-----------|
| WS1 Latent harvest | CART-1 | — |
| WS2 Instrumented MPC | CART-2 | — |
| WS3 Federated before/after | CART-3 | — |
| WS4 Projection + alignment + metrics | CART-4 | CART-1, CART-3 |
| WS5 Collapse counterfactual | CART-5 | CART-4 |
| WS6 Evidence JSON + test | CART-6 | CART-2, CART-3, CART-4 |
| WS7 Bake orchestrator | CART-7 | CART-1..5 |
| WS8 WebGPU viewer | CART-8 | doc `02` (contract) |
| WS9 Rehearsal + fallback + capture | CART-9 | CART-7, CART-8 |

Critical path: **CART-1 → CART-4 → CART-7 → CART-9**. CART-8 runs in parallel against the frozen data contract. Full breakdown, ordering, and parallelization: doc `03-workstreams-and-issues.md`.

---

## 7. Day-of timeline (target)

| Block | Work |
|-------|------|
| Stretch pre-flight | Bake a **v0 fallback** `manifold.json` from current checkpoint so a viewer exists before any live work. Vendor Three.js. Start only after #359 is rehearsal-green. |
| 09:00–11:00 | CART-1 harvest + CART-2 instrumented MPC (parallel). |
| 11:00–13:00 | CART-3 federation round + CART-4 projection/alignment/metrics. |
| 13:00–14:30 | CART-5 collapse + CART-7 bake orchestrator → real `manifold.json`. |
| 14:30–16:30 | CART-8 viewer polish against real data; CART-6 evidence + test. |
| 16:30–17:30 | CART-9 rehearsal gate, capture the clip, prep Demo-Night. |
| 19:00–22:00 | Demo Night. |

Hard rule: a **shippable fallback exists at every hour** (Decision D8 / runsheet). We always demo *something*; later bakes only improve it.

---

## 8. Definition of done (project-level)

- [ ] `manifold.json` validates against the doc-`02` contract and renders in `web/latent-manifold-viewer/` on the demo laptop via WebGPU.
- [ ] Point cloud + MPC plan trails + healthy/collapsed toggle + pre/post-federation toggle all functional.
- [ ] `docs/evidence/lewm_tworooms_manifold.json` (`lewm-manifold/1`) generated by `scripts/lewm_manifold_check.py`, with explicit `nonClaims`, and `tests/ml/test_lewm_manifold.py` green.
- [ ] All structure metrics on screen trace to the evidence JSON (no bare numbers).
- [ ] Claim-discipline checklist (doc `05`) signed off: every visible claim uses approved language.
- [ ] Rehearsal gate passes; a ≤20 s capture + result card exist for sharing.
- [ ] `ruff check . && ruff format --check .`, `pyright`, and the touched `pytest` gates pass; `check_docs_links.py` and `mkdocs build --strict` pass.
