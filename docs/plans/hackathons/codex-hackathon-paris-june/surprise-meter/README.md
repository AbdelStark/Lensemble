# Less Surprised / Surprise-meter (#338) — technical track

The primary Codex-hackathon deliverable (working title **Less Surprised**): a clean federated **adapter-continuation round** on the frozen LeWM world model, then a live **surprise-meter** showing the model is measurably *less surprised* after a room of strangers each train a tiny adapter on it — without sharing data.

> **Claim discipline:** it is an *adapter-continuation round*, never a "federated training run." Improvement is **+12.3% this run, +16.8% mean / +5.4% worst across 5 seeds** — the worst case is always shown. See `05-risks-and-claim-discipline.md`. The `01-architecture.md` "Corrections" box (C1–C12) lists every source-verified API fix.

> Parent: [#359](https://github.com/AbdelStark/Lensemble/issues/359). Technical tracker: [#338](https://github.com/AbdelStark/Lensemble/issues/338). Stretch sibling: [Cartographer / #339](../cartographer/). Hackathon index: [`../README.md`](../README.md).

## Today - ship queue

It is hackathon day (2026-06-18). This track now supports the
sovereign-economy parent plan. Do not reopen scope. Execute this order:

1. `#349` - make Milestone 0 clean, one-command, audited, and fallback-safe.
2. `#350` - build the per-step scalar surprise engine using the verified runtime
   APIs.
3. `#351` - put the meter, perturbation spike, and frame-diff baseline on
   screen.
4. `#352` - wire the held-out probe-pair pre/post toggle and source the
   certified numbers.
5. `#353` - generate/test `lewm-surprise/1`.
6. `#354` - rehearse, bake fallback assets, capture the clip/card.

Critical path: `#349 -> #350 -> #351 -> #354`. Cartographer is blocked until
this path is stage-ready.

## Two milestones
- **Milestone 0 (must):** a reliable, one-command, end-to-end federated adapter-continuation run + committed audited evidence + rehearsal gate. Standalone-demoable.
- **Milestone 1 (ship):** the live surprise-meter (scalar prediction-error signal, perturbation spikes, frame-diff baseline, pre/post-federation toggle).

## Read in order
| Doc | Purpose |
|---|---|
| [`00-overview.md`](00-overview.md) | Pitch, the two corrections (scalar surprise; adapter-continuation language), scope, decisions S1–S8, timeline, DoD |
| [`01-architecture.md`](01-architecture.md) | Per-component reference with verified APIs (`lewm_runtime`/`lewm_probe`/`lewm_adapter`/`tworooms_env`/`system_probe`); `[BUILD]` items |
| [`02-data-contracts.md`](02-data-contracts.md) | `lewm-surprise/1` evidence, recorded-trajectory fallback, `adapter_offset.json` |
| [`03-workstreams-and-issues.md`](03-workstreams-and-issues.md) | The 6 child issues (SM-1 Milestone-0; SM-2..6 Milestone-1), dependency graph, gate matrix |
| [`04-demo-runsheet.md`](04-demo-runsheet.md) | 90-second script, live Milestone-0 opener, 4-rung fallback ladder, pre-flight |
| [`05-risks-and-claim-discipline.md`](05-risks-and-claim-discipline.md) | Risk register + binding claim checklist |

## Why low-risk
- The surprise quantity already exists (`web/federated-demo/lewm_probe.mjs` computes `MSE(predicted_next, actual_next)`); Milestone 1 makes it per-step + live + visual.
- The federated run already exists and is certified (`lensemble/demo/system_probe.py`, seed-robust evidence); Milestone 0 makes it clean and one-command.
- Inference ~6 ms/step on CPU; in-browser via existing WebGPU/WASM runtime.
