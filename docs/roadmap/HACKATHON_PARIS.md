# Roadmap — Codex Hackathon (Paris, June)

Status tracker for the hackathon demo built on the LeWM federated stack. Detailed plans live under
[`docs/plans/hackathons/codex-hackathon-paris-june/`](../plans/hackathons/codex-hackathon-paris-june/).

## Framing (claim discipline)

The hackathon demo is built on **federated adapter continuation on a frozen LeWorldModel TwoRooms
checkpoint** — not federated world-model training. The [#335 spike](../spikes/0001-federated-world-model-training/)
settled this: full-model federated training is a GPU-scale research problem and is **NO-GO in-browser**
(183.5× byte budget, 1100.7× parameter cap, no in-browser ViT autograd). The honest, laptop-demonstrable
backbone is the bounded 12,512-parameter (0.069%) adapter, whose held-out improvement is system-composed
and seed-robust (`docs/evidence/lewm_tworooms_system_probe.json`, `..._probe_seedsweep.json`).

## Priority ladder

| Priority | Milestone | Issue(s) | Status |
|---|---|---|---|
| **0 — must** | A **clean federated training run** (adapter continuation): one-command end-to-end run, committed audited evidence, headless rehearsal gate, live-browser QR runbook. | #338 → SM-1 | Planned |
| **1 — ship** | **Surprise-meter** (#338): live scalar prediction-error meter; perturbation spikes; frame-diff baseline; pre/post-federation toggle. | [#338](https://github.com/AbdelStark/Lensemble/issues/338) (SM-1…SM-6) | Planned |
| **2 — stretch** | **Cartographer** (#339): WebGPU latent-manifold + planning viewer. Build only if #338 ships early. | [#339](https://github.com/AbdelStark/Lensemble/issues/339) (CART-1…CART-9) | Planned (stretch) |

Sibling idea [#337 (Latent Genie)](https://github.com/AbdelStark/Lensemble/issues/337) is **not pursued**
(blocked by the no-decoder finding from the feasibility spike).

## Key technical facts (feasibility spike + code recon)

- Exported world model is **CLS-latent, 192-d, 3-frame predictor window, no latent→pixel decoder**;
  inference **~6 ms/step on CPU** (>80 fps) via the in-browser ONNX runtime (WebGPU EP, WASM fallback).
- **Surprise = scalar** per-frame next-latent prediction error (no per-patch head). It is already
  implemented in `web/federated-demo/lewm_probe.mjs` as `MSE(frozen_predictor_output, true_next_latent)`;
  the adapter reduces exactly this quantity.
- Certified federated improvement: **+12.3% committed / +16.8% seed-mean** held-out prediction-error
  reduction.

## New artifacts these milestones introduce

- Evidence schema `lewm-surprise/1` → `docs/evidence/lewm_tworooms_surprise.json` (producer
  `scripts/lewm_surprise_check.py`, test `tests/ml/test_lewm_surprise.py`).
- Web demo `web/surprise-meter/` (imports the existing `web/federated-demo/` runtime/env/adapter/probe).
- (Stretch / #339) `cartographer-manifold/1` + `lewm-manifold/1` and `web/latent-manifold-viewer/`.

## Non-claims (carried on every surface)

No federated world-model training claim; no secure-aggregation or differential-privacy claim on this
path; no per-patch spatial surprise claim; no calibrated-anomaly-detector claim; no paper-scale or
closed-loop-robot claim.
