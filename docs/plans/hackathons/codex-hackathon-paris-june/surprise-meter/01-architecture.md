# Surprise-meter — Technical Architecture

Implementation reference. APIs below were verified by reading source during planning; `[BUILD]` marks new work. Conventions: see `../cartographer/01-architecture.md` §top (AGENTS.md/CONTRIBUTING.md rules apply identically).

---

## 0. The two existing pillars

### A. The exported world model (in-browser)
`web/federated-demo/model/lewm-tworooms/{encoder,action,predictor}.onnx` + `manifest.json` (`lewm-browser-export/1`). Loaded by `web/federated-demo/lewm_runtime.mjs`:
- `loadLewmRuntime(ortApi=globalThis.ort)` (line ~43): provider preference `["webgpu","wasm"]`; hash-checks each ONNX against `manifest.files[*].sha256` via WebCrypto; throws `LewmUnsupportedError("no-ort")` if `ort` absent.
- `createLewmRuntime()` reads `manifest.architecture.{hiddenDim=192, numFrames=3, imageSize=224, actionDim=10}`.
- IO (verified, feasibility spike): encoder `pixels (B,3,224,224) → latent (B,192)`; action `(B,T,10) → (B,T,192)`; predictor `latents (B,3,192) × action_embeddings (B,3,192) → predicted_latents (B,3,192)`, next = `[:, -1, :]`.

### B. The surprise quantity (already implemented)
`web/federated-demo/lewm_probe.mjs::probeAdapterOffset` (line ~112): for held-out pairs it computes
`baselineMse = MSE(frozen_predictor_output, true_next_latent)` and `adaptedMse = MSE(adapter(frozen_predictor_output), true_next_latent)`.
**`baselineMse` is exactly per-pair surprise.** Milestone 1 evaluates the same MSE **per step over a live trajectory** and renders it.

### C. The env (pixel-exact, JS)
`web/federated-demo/tworooms_env.mjs`: `renderFrameRGB(pos, {renderTarget})`, `step(action)`, `renderGoalFrameRGB(targetPos)`. Geometry: `IMG_SIZE=224`, wall `x=112` thick 10, door `y=49` half 14, `AGENT_SPEED=5`, `SUCCESS_DISTANCE=16`, `ACTION_BLOCK=5` (5 env-steps → `Float32Array(10)`), `ACTION_DIM=2 ∈ [-1,1]²`. `tworooms_panel.mjs` already visualizes it.

### D. The adapter (pre/post offset)
`web/federated-demo/lewm_adapter.mjs`: `adapterForward(z) = z + W2·tanh(W1·z+b1) + b2` (line ~231); `createAdapter({inputDim:192,hiddenDim:32,seed:42})` (W2,b2 zero-init; W1 Xavier via mulberry32); `adapterFromInitAndOffset(offset)` (line ~208) = `init + server_offset`. Offset len 12512.

---

## 1. WS0 — Clean federated run (`SM-1`, Milestone 0, the priority)

**Goal:** a reliable, one-command, end-to-end federated **adapter-continuation** run, with committed audited evidence and a rehearsal gate. Most of this exists (#314/#332); SM-1 makes it *clean and demonstrable*.

**Reuse (verified):**
- Headless system-composed run: `lensemble/demo/system_probe.run_system_composed_probe(participants, validation, checkpoint, manifest, rounds=3, steps_per_round=20, batch_size=32, seed=20260612, dim=192)` → `write_evidence(Path("docs/evidence/lewm_tworooms_system_probe.json"), evidence)`. Driver script `scripts/lewm_system_probe.py`.
- Audit gate: `lensemble/demo/evidence_audit.audit_real_lewm_evidence(evidence)` (line ~61) — schema `demo-evidence/1`, required non-claim phrases, pinned checkpoint hash, adapter spec, per-round health metrics.
- Live browser path: `uv run lensemble demo federated --port 8765` → `http://127.0.0.1:8765/web/federated-demo/` (QR join, `lewm_local_trainer.mjs`, `lewm_adapter.mjs`, `lewm_delta_artifact.mjs`, `lewm_system_round.mjs`).
- Rehearsal style: `scripts/hackathon_demo_rehearsal.py`, `scripts/lewm_demo_rehearsal.py`.

**Tasks:**
- [ ] A single command (Make target or `scripts/cartographer`-style `scripts/surprise/run_clean_round.py`) that: runs the headless system-composed round, regenerates `lewm_tworooms_system_probe.json`, runs `audit_real_lewm_evidence`, and prints the +improvement and round health.
- [ ] A documented **runbook** (extend `docs/roadmap/TAPESTRY_LEWM_RUNBOOK.md` or a new `docs/roadmap/HACKATHON_PARIS.md`) for the live browser QR round: start server, join, train, aggregate, read the probe tick-up.
- [ ] A **rehearsal gate** `scripts/surprise/rehearsal.py` asserting the headless round completes, evidence passes audit, and the offset is produced.
- [ ] Export the post-round **adapter offset** to a small JSON file (`runs/surprise/adapter_offset.json`, len 12512) for Milestone 1's pre/post toggle and the fallback.

**Acceptance:** one command → green audit + committed evidence; rehearsal gate exits 0; offset file produced; runbook steps reproduce a live round.

> Honest framing everywhere (AGENTS.md): "federated **adapter continuation** on a frozen checkpoint." Single local coordinator, mean of clipped deltas, no DP/secure-agg.

---

## 2. WS1 — In-browser surprise engine (`SM-2`)

**Goal:** per-step scalar surprise over a live trajectory, in-browser, using the exported graphs.

**[BUILD]** `web/surprise-meter/surprise_engine.mjs`:
```js
// uses lewm_runtime (encode/predict) + tworooms_env
// maintain a 3-frame ring buffer of latents and action embeddings
async function stepSurprise(runtime, histLatents/*[z_{t-2},z_{t-1},z_t]*/, histActEmb, nextFrameRGB) {
  const predNext = runtime.predict(histLatents, histActEmb);     // (3,192) -> take [-1]
  const zNext    = runtime.encode(nextFrameRGB);                 // (192,)
  const surprise = mse(predNext.at(-1), zNext);                  // scalar
  return { surprise, zNext };
}
```
- Mirror `lewm_probe.mjs` math exactly (parity target ≤1e-4 vs a `lewm_probe`-style fixture) so "surprise" is the same number the certified probe uses.
- Maintain the 3-frame window per the predictor contract (`numFrames=3`).
- Action blocks: 5 env-steps → `Float32Array(10)` via the env's existing packing.

**[BUILD]** `web/surprise-meter/surprise_selftest.mjs`: a deterministic self-test (fixed frames/actions) asserting finite surprise and parity vs a recorded fixture.

**Acceptance:** per-step surprise stream over a scripted trajectory; parity with `lewm_probe.mjs` on shared inputs; ≥10 steps/s in-browser (trivially met).

---

## 3. WS2 — Surprise UI + perturbation + frame-diff (`SM-3`)

**Goal:** the visual that makes surprise legible and proves surprise ≠ motion.

**[BUILD]** `web/surprise-meter/{index.html, app.mjs}` (template from `web/dynamic-env-demo/` — self-contained, inline CSS; load `ort.webgpu.min.js` via the same CDN tag; import env/runtime/adapter/probe from `../federated-demo/`).
- **Env canvas:** render the live TwoRooms trajectory (reuse `tworooms_env.mjs` + `tworooms_panel.mjs` patterns); tint the agent/border by current surprise.
- **Surprise meter / oscilloscope:** a scrolling timeline of `surprise_t` (reuse `charts.mjs` SVG helpers), with a moving readout. This is the hero visual (scalar, not a heatmap — Decision S1).
- **Perturbation controls:** buttons to (a) teleport the agent, (b) inject an off-distribution action, (c) force it through the wall. Each should spike the meter.
- **Frame-diff baseline panel (Decision S6):** a second trace = mean abs pixel difference between consecutive frames. Show cases where motion is high but surprise is low (predictable glide) and motion low but surprise high (teleport) — the contrast *is* the wow and the credibility.

**Acceptance:** live meter + env; perturbations visibly spike surprise; frame-diff trace present; runs in-browser at smooth framerate; degrades to WASM if no WebGPU.

> ⚠️ R1: validate offline (in SM-2) that perturbations actually spike the CLS predictor's error. If weak, foreground the OOD-action and pre/post contrasts (Decision S8).

---

## 4. WS3 — Pre/post-federation toggle (`SM-4`)

**Goal:** show the federated adapter reduces surprise on in-distribution dynamics.

**Reuse:** `lewm_adapter.mjs::adapterFromInitAndOffset(offset)` + `adapterForward`. The offset is the SM-1 `adapter_offset.json` (or a freshly-aggregated revision's `adapterState`).

**[BUILD]** in `surprise_engine.mjs` / `app.mjs`:
- Compute two surprise streams on the **same** trajectory: `surprise_pre` (frozen predictor output) and `surprise_post` (apply adapter to the predicted latent before MSE). The adapter corrects the predicted next latent toward the true next latent → `surprise_post < surprise_pre` on in-distribution pairs.
- A toggle/overlay showing both traces and the running mean drop; HUD shows the certified **+12.3% / +16.8%** (read from `manifold`/evidence; Decision S7) alongside the live mean drop.

**Acceptance:** `mean(surprise_post) < mean(surprise_pre)` on an in-distribution trajectory; HUD number sourced to evidence; toggle smooth.

---

## 5. WS4 — Evidence JSON (`SM-5`)

**Goal:** back the on-screen numbers with a generated, tested file. Schema `lewm-surprise/1` → `docs/evidence/lewm_tworooms_surprise.json`. Producer `scripts/lewm_surprise_check.py`.

Fields (full shape in `02-data-contracts.md`): `schema`, `seed`, `checkpoint{repoId,revision,weightsSha256}`, `result{ meanSurprisePre, meanSurprisePost, surpriseDropRatio, perturbationSpikeRatio, frameDiffCorrelation, federatedRelativeImprovement (sourced), federatedSeedMean (sourced) }`, `passes`, `nonClaims`, `crossCheck`.

`nonClaims` MUST include: adapter-continuation-not-training; surprise-is-scalar-CLS-prediction-error; no-secure-agg/DP; perturbation results are illustrative on the TwoRooms backbone.

**Test [BUILD]** `tests/ml/test_lewm_surprise.py`: schema, field presence/types, `nonClaims` negations, `passes` predicate (see contract). Wire into gate 4.

**Acceptance:** deterministic producer; test green; `check_docs_links.py` + `mkdocs --strict` pass.

---

## 6. WS5 — Rehearsal + fallback + capture (`SM-6`)

- **Rehearsal gate [BUILD]** `scripts/surprise/rehearsal.py`: Milestone-0 round completes + audited; surprise engine self-test passes; pre/post offset present; viewer required assets exist.
- **Fallback:** committed `runs/surprise/adapter_offset.json` (pre-event) so the pre/post toggle works even if the live round fails; and a recorded trajectory so the meter runs without live env quirks.
- **Capture:** ≤20 s clip (move → perturb → spike → pre/post toggle) + result card (the +12% number + "less surprised after the room trained it"). Cite the evidence file. Runsheet: `04-demo-runsheet.md`.

**Acceptance:** rehearsal exits 0; fallback present; clip + card exported.

---

## 7. External APIs touched (index)

| Need | Symbol | Location |
|---|---|---|
| In-browser encode/predict | `loadLewmRuntime`, `createLewmRuntime` (WebGPU EP, hash-check) | `web/federated-demo/lewm_runtime.mjs:43` |
| Surprise math (reuse) | `probeAdapterOffset` (`baselineMse`/`adaptedMse`) | `web/federated-demo/lewm_probe.mjs:112` |
| Env (pixel-exact) | `renderFrameRGB`,`step`,`renderGoalFrameRGB` | `web/federated-demo/tworooms_env.mjs` |
| Adapter pre/post | `adapterForward`,`adapterFromInitAndOffset` | `web/federated-demo/lewm_adapter.mjs:231,208` |
| Headless federated round | `run_system_composed_probe`,`write_evidence` | `lensemble/demo/system_probe.py` |
| Evidence audit | `audit_real_lewm_evidence` | `lensemble/demo/evidence_audit.py:61` |
| SVG charts | `charts.mjs` helpers | `web/federated-demo/charts.mjs` |
| Static serving | `_static`, `WEB_ROOT` (serves any path under `web/`) | `lensemble/demo/server.py:560,27` |
