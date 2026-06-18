# Roadmap — Codex Hackathon (Paris, June)

Status tracker for the hackathon demo built on the LeWM federated stack. Detailed plans live under
[`docs/plans/hackathons/codex-hackathon-paris-june/`](../plans/hackathons/codex-hackathon-paris-june/).

## Today — 2026-06-18 start mode

The hackathon is active now. The parent tracker is [#359](https://github.com/AbdelStark/Lensemble/issues/359):
sovereign robotics world-model economy. The previous parent [#338](https://github.com/AbdelStark/Lensemble/issues/338)
is now the technical surprise-meter child. Build in parallel: narrative/deck
[#364](https://github.com/AbdelStark/Lensemble/issues/364), ledger
[#361](https://github.com/AbdelStark/Lensemble/issues/361), Mollie test checkout
[#360](https://github.com/AbdelStark/Lensemble/issues/360), surprise-meter
[#338](https://github.com/AbdelStark/Lensemble/issues/338), dashboard
[#363](https://github.com/AbdelStark/Lensemble/issues/363), then rehearsal
[#362](https://github.com/AbdelStark/Lensemble/issues/362). Cartographer
[#339](https://github.com/AbdelStark/Lensemble/issues/339) remains locked until
`#359` has a green integrated rehearsal gate.

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
| **0 — must** | **Sovereign-economy narrative and deck arc**: claim-safe sovereignty problem, decentralized run solution, humanoid robotics buyer placeholder. | [#359](https://github.com/AbdelStark/Lensemble/issues/359) -> [#364](https://github.com/AbdelStark/Lensemble/issues/364) | **Active now** |
| **1 — must** | **Contribution ledger**: deterministic split of simulated sale proceeds by data/compute contribution. | [#361](https://github.com/AbdelStark/Lensemble/issues/361) | Active now |
| **2 — must** | **Mollie test checkout**: server-side SDK/test checkout link or deterministic mock fallback. | [#360](https://github.com/AbdelStark/Lensemble/issues/360) | After ledger contract |
| **3 — ship** | **Surprise-meter proof**: clean adapter-continuation round, scalar surprise, pre/post toggle, evidence. | [#338](https://github.com/AbdelStark/Lensemble/issues/338) -> [#349](https://github.com/AbdelStark/Lensemble/issues/349)-[#354](https://github.com/AbdelStark/Lensemble/issues/354) | Parallel technical track |
| **4 — ship** | **Economics dashboard**: buyer checkout, orchestrator share, community pool, participant rewards beside model proof. | [#363](https://github.com/AbdelStark/Lensemble/issues/363) | After #361/#360 + enough #338 |
| **5 — ship** | **Integrated rehearsal and fallbacks**: live path, mock path, recorded path, capture card. | [#362](https://github.com/AbdelStark/Lensemble/issues/362) | Final gate |
| **6 — stretch** | **Cartographer** (#339): WebGPU latent-manifold + planning viewer. | [#339](https://github.com/AbdelStark/Lensemble/issues/339) | Locked until #359 rehearsal-green |

Sibling idea [#337 (Latent Genie)](https://github.com/AbdelStark/Lensemble/issues/337) is **not pursued**
(blocked by the no-decoder finding from the feasibility spike).

## Key technical facts (feasibility spike + code recon)

- Exported world model is **CLS-latent, 192-d, 3-frame predictor window, no latent→pixel decoder**;
  inference **~6 ms/step on CPU** (>80 fps) via the in-browser ONNX runtime (WebGPU EP, WASM fallback).
- **Surprise = scalar** per-frame next-latent prediction error (no per-patch head). It is already
  implemented in `web/federated-demo/lewm_probe.mjs` as `MSE(frozen_predictor_output, true_next_latent)`;
  the adapter reduces exactly this quantity.
- Certified federated improvement: **+12.3% committed run / +16.8% seed-mean / +5.4% worst seed (2)**,
  stdev 0.11, all 5 seeds improved — held-out prediction-error reduction. The **worst case is always
  surfaced beside the mean** (claim discipline); numbers are sourced at full float precision.

## New artifacts these milestones introduce

- Evidence schema `lewm-surprise/1` → `docs/evidence/lewm_tworooms_surprise.json` (producer
  `scripts/lewm_surprise_check.py`, test `tests/ml/test_lewm_surprise.py`).
- Clean-round and rehearsal gates: `scripts/surprise/run_clean_round.py` exports the real
  12,512-float adapter offset sidecar; `scripts/surprise/rehearsal.py` validates the synthetic
  system path, JS self-test, and served fallback assets.
- Web demo `web/surprise-meter/` with served fallback assets:
  `fixtures/adapter_offset.json`, `data/result_card.json`, and `data/surprise_trajectory.json`.
- Sovereign-economy contracts `sovereign-sale/1` and `sovereign-contribution-ledger/1`.
- Server-side Mollie test checkout path with `.env.example` placeholders and mock fallback.
- (Stretch / #339) `cartographer-manifold/1` + `lewm-manifold/1` and `web/latent-manifold-viewer/`.

## Non-claims (carried on every surface)

No federated world-model training claim; no secure-aggregation or differential-privacy claim on this
path; no per-patch spatial surprise claim; no calibrated-anomaly-detector claim; no paper-scale or
closed-loop-robot claim; no real legal payout/revenue-share claim; no named company or incident claim
without separate verification.
