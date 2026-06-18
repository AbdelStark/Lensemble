# Surprise-meter — Risks & Claim Discipline

## 1. Risk register

| # | Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R1 | Perturbations don't cleanly spike the CLS predictor's error (surprise looks flat) | Med | High | **Validate IN-BROWSER before stage** (onnxruntime is absent offline — Chrome is the only runtime; R13). Test teleport / OOD-action / through-wall; the `passes` predicate requires ≥1 channel to spike (else demo leans on pre/post per S8). Tune which perturbation reads best. | SM-2/3 |
| R2 | Surprise just tracks motion (not novel) | Med | High | On-screen frame-diff baseline + `frameDiffCorrelation < 0.6` gate; pick demo moments where motion/surprise diverge | SM-3/5 |
| R3 | In-browser surprise diverges from the certified `lewm_probe.mjs` math | Low-Med | Med | Parity gate ≤1e-4 vs a `lewm_probe` fixture in `surprise_selftest.mjs` | SM-2 |
| R4 | Live federated round flakes on stage | Med | Med | Pre-baked offset (rung B) + recorded trajectory (rung C); Milestone 0 rehearsed headlessly | SM-1/6 |
| R5 | WebGPU absent/unstable on projector machine | Med | High | **Mitigated:** `web/surprise-meter/?engine=live&ep=wasm` forces WASM, and `loadLewmRuntime` retries provider sets before failing. `engine=auto` falls back to the recorded trajectory if live ONNX fails; still verify the presentation browser before stage. | SM-3 |
| R6 | Over-claim slips onto screen/X (esp. "federated training", or mean-without-worst) | Med | High (credibility) | Checklist §2; `nonClaims` rendered from the artifact; evidence test asserts negations; worst-case +5.4% surfaced beside +16.8% | all |
| R7 | Time overrun; Milestone 1 unfinished | Med | Low | **Milestone 0 is a standalone demo**; ship it first. If squeezed, cut frame-diff panel + one perturbation to stretch (the live meter + pre/post toggle is the minimum win) | — |
| R8 | Normalization mismatch in the surprise loop (uint8/ImageNet) → garbage | Med | Med | **Resolved:** use `frameToModelInput(rgb)` (÷255 + HWC→CHW); ImageNet z-score is baked **inside** the encoder graph (do not apply in JS); action z-score baked inside the action graph (pass raw blocks). Parity gate (`surprise_selftest.mjs`) catches drift. | SM-2 |
| R9 | Offset-export path does not exist (`run_system_composed_probe` returns only the count) | High | High | SM-1 adds `offset_out` kwarg writing `service.model_revision(...)["adapterState"]` (len 12512) to a **sidecar** (never the audited evidence — the `"adapterState"` substring is forbidden there). | SM-1 |
| R10 | Committed fallback offset can't live at `runs/…` (gitignored) | High | Med | Commit it at `web/surprise-meter/fixtures/adapter_offset.json` (tracked, served); keep `runs/surprise/` as throwaway. | SM-1/6 |
| R11 | `mean(post)<mean(pre)` not guaranteed on a free-running live trajectory (certified +12.3% is held-out-pairs only; worst seed +5.4%) | Med | High | The toggle runs on the **held-out probe-pair set** via `collectResidentPairs` (seed 991), computed once at load (Decision S10) → the drop *is* the certified number. Perturbation spikes stay on the pre-only meter. | SM-4 |
| R12 | Node-based offline bake can't load local ONNX (global `fetch` has no `file://`) | Med | Med | The node bake injects a `readFileSync`-backed `fetchFn` and forces `preferredProviders:["cpu"]` (`01-architecture.md` §7). The pure-JS `surprise_selftest.mjs` needs no runtime. | SM-2/6 |
| R13 | `onnxruntime` absent from uv venv, system node, **and** npm — no offline runtime for validation/bake | High | High | Start-of-day BLOCKING pre-req (2026-06-18): Python bake via `uv run --with onnxruntime --with hdf5plugin ...`; node trajectory bake via a throwaway `npm install onnxruntime-node`. R1 validation is **in-browser only**. | SM-1/6 |

## 2. Claim-discipline checklist (AGENTS.md §Claim Discipline + spike #335 are binding)

- [ ] Federation is "federated **adapter continuation** on a frozen checkpoint" — **never** "federated world-model training" / "the room trained the world model" / "a clean federated training run." Only a **12,512-param (0.069%) adapter** moves; the backbone is frozen. (Spike #335: full-model in-browser is NO-GO.)
- [ ] The federated improvement is stated as **+12.3% this run, +16.8% mean / +5.4% worst across 5 seeds** — the **worst case is always surfaced beside the mean** (AGENTS.md binds the seed claim *with* its worst case). It is **sourced** to `lewm_tworooms_system_probe.json` / `..._probe_seedsweep.json` at **full float precision**, not re-derived and not truncated to the display rounding.
- [ ] The **live** on-stage surprise drop is labelled "this run" and kept **distinct** from the certified held-out headline — they are different measurements (S10).
- [ ] Surprise is described as a **scalar per-frame next-latent prediction error** (CLS-latent model) — **never** a per-patch spatial heatmap.
- [ ] Perturbation/anomaly behavior is "illustrative on the TwoRooms backbone," not a calibrated anomaly detector.
- [ ] The webcam easter egg (if shown) is labelled clearly out-of-distribution; no claim attached.
- [ ] **No** secure-aggregation, differential-privacy, cryptographic-proof, paper-scale, or closed-loop-robot claims.
- [ ] Every on-screen number traces to `docs/evidence/lewm_tworooms_surprise.json` (which itself cross-references the system probe + seed-sweep).
- [ ] The Codex/agentic-build mention (S9) is framed as *how it was built*, with no claim that the model or result was Codex-generated.

## 3. Approved one-liners (safe on stage / X)
- "A JEPA world model predicts the future in latent space — here it is being surprised, live, on a laptop."
- "Surprise is prediction error, not motion — watch it stay calm through fast predictable movement and spike on the unexpected."
- "A room of people each trained a tiny 12k-param adapter on their own private TwoRooms trajectories; held-out next-step prediction error dropped ~12% on this run — +16.8% mean / +5.4% worst across five seeds. No data left their devices." *(say "TwoRooms / next-step prediction error", not "physics" — it's a held-out probe number on one env, not a general physics claim.)*
- "Everything on screen is backed by a generated evidence file — including the worst seed."
- "Built plan-to-evidence-gate in a Codex loop."

## 4. Forbidden phrasings
- ✗ "We federated-trained a world model." → adapter continuation on a frozen model.
- ✗ "Per-pixel/per-region surprise heatmap." → it's a scalar (CLS-latent model).
- ✗ "Private by differential privacy / cryptographic proof." → not wired in this path.
- ✗ "Real-time anomaly detector / beats baseline X." → illustrative on TwoRooms, not benchmarked.
- ✗ "Beats local-only / paper-scale." → not claimed.
- ✗ "16.8% improvement" stated alone → always pair with the worst seed (+5.4%) and "mean across 5 seeds."
- ✗ "A clean federated training run." (the draft's own phrasing) → "a clean federated adapter-continuation round."

## 5. Definition of done — claims
- [ ] Checklist §2 fully ticked.
- [ ] `tests/ml/test_lewm_surprise.py` asserts the mandatory `nonClaims` negations **and** the full-precision sourcing equalities (`federatedRelativeImprovement`/`federatedSeedMean`/`federatedSeedWorst` match the cited evidence files within 1e-6).
- [ ] The result card and HUD show `+12.3%` (this run) **and** `+5.4%` (worst seed) — never the mean alone.
- [ ] X post re-read against §3/§4 before posting.
