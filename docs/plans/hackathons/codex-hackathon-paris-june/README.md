# Codex Hackathon — Paris (June)

Plans for the federated-JEPA-world-model demo at the Codex hackathon (Paris, Thursday; full-day, solo-builder, Codex/OpenAI-sponsored, Demo-Night format).

## Priority ordering (decided 2026-06-16)

The hackathon ships in three milestones, in strict priority order. Each milestone is independently demoable, so we always have something to show.

| Priority | Milestone | Issue | Plan |
|---|---|---|---|
| **0 — must** | **A clean federated training run.** A reliable, one-command, end-to-end **federated adapter-continuation** run on the frozen LeWM TwoRooms checkpoint, with committed evidence and a headless rehearsal gate. This is the foundation both demos stand on. | [#349 (SM-1)](https://github.com/AbdelStark/Lensemble/issues/349) | [`surprise-meter/`](surprise-meter/) (WS0) |
| **1 — ship** | **Surprise-meter (#338).** A live "surprise" meter showing the world model is measurably **less surprised** after the room trains it together. The full hackathon deliverable. | [#338](https://github.com/AbdelStark/Lensemble/issues/338) | [`surprise-meter/`](surprise-meter/) |
| **2 — stretch** | **Cartographer (#339).** The WebGPU latent-manifold + planning viewer. Built only if #338 ships with time to spare. | [#339](https://github.com/AbdelStark/Lensemble/issues/339) | [`cartographer/`](cartographer/) |

> Rationale for the order: the **#335 spike** proved full-model federated training is a GPU-scale research problem (NO-GO in-browser) — so the honest, laptop-demonstrable backbone is **adapter continuation on a frozen checkpoint** (system-composed, seed-robust, +12.3%/+16.8%). #338's surprise quantity *is* the prediction error the adapter reduces, so it tells the federated story most directly and with the least build risk. #339 is the higher-ceiling, higher-effort visual; it stays fully planned but stretch.

## Directories

- [`surprise-meter/`](surprise-meter/) — **the priority plan** (#338 + the Milestone-0 clean federated run). Read this first.
- [`cartographer/`](cartographer/) — the stretch plan (#339). Fully scoped; build only if time allows.

## Shared facts (feasibility spike + 4 code-recon passes)

- World model is **CLS-latent, 192-d, 3-frame predictor window, no decoder**; inference **~6 ms/step on CPU** (>80 fps). WebGPU allowed but not required.
- Federation = **adapter continuation on a frozen checkpoint** (12,512-param / 0.069% residual adapter; raw data never leaves the participant). Never call it "federated world-model training" (AGENTS.md §Claim Discipline; spike #335).
- Certified federated improvement: **+12.3% committed / +16.8% seed-mean** held-out prediction-error reduction (`docs/evidence/lewm_tworooms_system_probe.json`, `..._probe_seedsweep.json`).
- The surprise quantity (#338) is already implemented in `web/federated-demo/lewm_probe.mjs` as `MSE(frozen_predictor_output, true_next_latent)` — the adapter reduces exactly this.

## Sibling idea not pursued
- [#337 Latent Genie](https://github.com/AbdelStark/Lensemble/issues/337) — playable latent world; blocked by the no-decoder finding.
