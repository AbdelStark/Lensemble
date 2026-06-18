# Codex Hackathon — Paris (June)

Plans for the federated-JEPA-world-model demo at the Codex hackathon (Paris, Thursday; full-day, solo-builder, Codex/OpenAI-sponsored, Demo-Night format).

## Start here today - 2026-06-18

The hackathon is active. Read [`TODAY.md`](TODAY.md), then execute the
[#359](https://github.com/AbdelStark/Lensemble/issues/359) queue without
re-planning:

1. [#364](https://github.com/AbdelStark/Lensemble/issues/364) - sovereignty /
   economics narrative and deck arc.
2. [#361](https://github.com/AbdelStark/Lensemble/issues/361) - deterministic
   contribution ledger and reward split.
3. [#360](https://github.com/AbdelStark/Lensemble/issues/360) - Mollie test
   checkout/payment links with mock fallback.
4. [#338](https://github.com/AbdelStark/Lensemble/issues/338) - surprise-meter
   technical track, including [#349](https://github.com/AbdelStark/Lensemble/issues/349)-[#354](https://github.com/AbdelStark/Lensemble/issues/354).
5. [#363](https://github.com/AbdelStark/Lensemble/issues/363) - integrated
   economics dashboard.
6. [#362](https://github.com/AbdelStark/Lensemble/issues/362) - rehearsal,
   fallback, validation, capture.

Do not start [#339 Cartographer](https://github.com/AbdelStark/Lensemble/issues/339)
until `#359` is rehearsal-green with fallback assets on disk.

## Priority ordering (updated 2026-06-18)

The hackathon ships in three milestones, in strict priority order. Each milestone is independently demoable, so we always have something to show.

| Priority | Milestone | Issue | Plan |
|---|---|---|---|
| **0 — must** | **Sovereign robotics world-model economy (#359).** Narrative, ledger, Mollie test checkout, surprise proof, reward split, integrated rehearsal. | [#359](https://github.com/AbdelStark/Lensemble/issues/359) | [`sovereign-economy/`](sovereign-economy/) |
| **1 — ship** | **Surprise-meter technical track (#338).** Clean adapter-continuation run and scalar surprise proof. | [#338](https://github.com/AbdelStark/Lensemble/issues/338) | [`surprise-meter/`](surprise-meter/) |
| **2 — stretch** | **Cartographer (#339).** The WebGPU latent-manifold + planning viewer. Built only if #359 ships with time to spare. | [#339](https://github.com/AbdelStark/Lensemble/issues/339) | [`cartographer/`](cartographer/) |

> Rationale for the order: the **#335 spike** proved full-model federated training is a GPU-scale research problem (NO-GO in-browser) — so the honest, laptop-demonstrable backbone is **adapter continuation on a frozen checkpoint** (system-composed, seed-robust, +12.3%/+16.8%). #338's surprise quantity *is* the prediction error the adapter reduces, so it tells the federated story most directly and with the least build risk. #339 is the higher-ceiling, higher-effort visual; it stays fully planned but stretch.

## Directories

- [`sovereign-economy/`](sovereign-economy/) — **the active parent plan** (#359). Read this first.
- [`surprise-meter/`](surprise-meter/) — the model-quality proof track (#338 + the Milestone-0 clean adapter-continuation round).
- [`TODAY.md`](TODAY.md) — the current hackathon start brief and no-fluff priority queue.
- [`cartographer/`](cartographer/) — the stretch plan (#339). Fully scoped; build only if time allows.
- [`presentation/`](presentation/) — the **Demo-Night reveal.js deck** (sovereign-economy draft). Self-contained; grounded in these plans; edited live during the build. Run with `python3 -m http.server --directory presentation`.

## Shared facts (feasibility spike + 4 code-recon passes)

- World model is **CLS-latent, 192-d, 3-frame predictor window, no decoder**; inference **~6 ms/step on CPU** (>80 fps). WebGPU allowed but not required.
- Federation = **adapter continuation on a frozen checkpoint** (12,512-param / 0.069% residual adapter; raw data never leaves the participant). Never call it "federated world-model training" (AGENTS.md §Claim Discipline; spike #335).
- Certified federated improvement: **+12.3% committed / +16.8% seed-mean / +5.4% worst seed** (stdev 0.11; all 5 seeds improved) held-out prediction-error reduction (`docs/evidence/lewm_tworooms_system_probe.json`, `..._probe_seedsweep.json`). Always surface the worst case beside the mean; source at full float precision.
- The surprise quantity (#338) is already implemented in `web/federated-demo/lewm_probe.mjs` as `MSE(frozen_predictor_output, true_next_latent)` — the adapter reduces exactly this.
- Mollie integration is server-side only. The first shippable economics path is a deterministic ledger plus test checkout/payment links; webhooks/payout simulation are bonus.

## Sibling idea not pursued
- [#337 Latent Genie](https://github.com/AbdelStark/Lensemble/issues/337) — playable latent world; blocked by the no-decoder finding.
