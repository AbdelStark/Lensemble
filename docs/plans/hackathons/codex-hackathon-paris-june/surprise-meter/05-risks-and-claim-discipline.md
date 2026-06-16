# Surprise-meter — Risks & Claim Discipline

## 1. Risk register

| # | Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R1 | Perturbations don't cleanly spike the CLS predictor's error (surprise looks flat) | Med | High | Validate offline in SM-2 across teleport / OOD-action / through-wall; the `passes` predicate requires ≥1 channel to spike (else demo leans on pre/post). Tune which perturbation reads best. | SM-2/3 |
| R2 | Surprise just tracks motion (not novel) | Med | High | On-screen frame-diff baseline + `frameDiffCorrelation < 0.6` gate; pick demo moments where motion/surprise diverge | SM-3/5 |
| R3 | In-browser surprise diverges from the certified `lewm_probe.mjs` math | Low-Med | Med | Parity gate ≤1e-4 vs a `lewm_probe` fixture in `surprise_selftest.mjs` | SM-2 |
| R4 | Live federated round flakes on stage | Med | Med | Pre-baked offset (rung B) + recorded trajectory (rung C); Milestone 0 rehearsed headlessly | SM-1/6 |
| R5 | WebGPU absent on projector machine | Med | High | WASM EP fallback (already in `lewm_runtime.mjs`); Canvas/SVG meter via `charts.mjs` (no WebGPU needed); clip rung D | SM-3 |
| R6 | Over-claim slips onto screen/X (esp. "federated training") | Med | High (credibility) | Checklist §2; nonClaims rendered from the artifact; evidence test asserts negations | all |
| R7 | Time overrun; Milestone 1 unfinished | Med | Low | **Milestone 0 is a standalone demo**; ship it first | — |
| R8 | Normalization mismatch in the surprise loop (uint8/ImageNet) → garbage | Med | Med | Reuse `lewm_runtime.mjs` encode path (normalization baked into ONNX); parity gate catches drift | SM-2 |

## 2. Claim-discipline checklist (AGENTS.md §Claim Discipline + spike #335 are binding)

- [ ] Federation is "federated **adapter continuation** on a frozen checkpoint" — **never** "federated world-model training" / "the room trained the world model." Only a **12,512-param (0.069%) adapter** moves; the backbone is frozen. (Spike #335: full-model in-browser is NO-GO.)
- [ ] The federated improvement (+12.3% committed / +16.8% seed-mean) is **sourced** to `lewm_tworooms_system_probe.json` / `..._probe_seedsweep.json`, not re-derived.
- [ ] Surprise is described as a **scalar per-frame next-latent prediction error** (CLS-latent model) — **never** a per-patch spatial heatmap.
- [ ] Perturbation/anomaly behavior is "illustrative on the TwoRooms backbone," not a calibrated anomaly detector.
- [ ] The webcam easter egg (if shown) is labelled clearly out-of-distribution; no claim attached.
- [ ] **No** secure-aggregation, differential-privacy, cryptographic-proof, paper-scale, or closed-loop-robot claims.
- [ ] Every on-screen number traces to `docs/evidence/lewm_tworooms_surprise.json` (which itself cross-references the system probe).

## 3. Approved one-liners (safe on stage / X)
- "A JEPA world model predicts the future in latent space — here it is being surprised, live, on a laptop."
- "Surprise is prediction error, not motion — watch it stay calm through fast predictable movement and spike on the unexpected."
- "A room of people each trained a tiny adapter on their own private data; the model got ~12% less surprised by normal physics. No data left their devices."
- "Everything on screen is backed by a generated evidence file."

## 4. Forbidden phrasings
- ✗ "We federated-trained a world model." → adapter continuation on a frozen model.
- ✗ "Per-pixel/per-region surprise heatmap." → it's a scalar (CLS-latent model).
- ✗ "Private by differential privacy / cryptographic proof." → not wired in this path.
- ✗ "Real-time anomaly detector / beats baseline X." → illustrative on TwoRooms, not benchmarked.
- ✗ "Beats local-only / paper-scale." → not claimed.

## 5. Definition of done — claims
- [ ] Checklist §2 fully ticked.
- [ ] `tests/ml/test_lewm_surprise.py` asserts the mandatory `nonClaims` negations.
- [ ] X post re-read against §3/§4 before posting.
