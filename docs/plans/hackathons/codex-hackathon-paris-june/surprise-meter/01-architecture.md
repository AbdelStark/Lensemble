# Surprise-meter — Technical Architecture

Implementation reference. **Every API, signature, constant, and shape below was read from source on 2026-06-17** (file:line cited) and the load-bearing ones were re-verified by a 16-agent ground-truth + deepening pass. `[BUILD]` marks new work. Conventions: AGENTS.md ("Claim Discipline", "Working Loop") + CONTRIBUTING.md apply identically (see `../cartographer/01-architecture.md` §top).

> **Read the corrections box first — the original draft contained build-blocking API drift.**

---

## 0. Corrections baked into this revision (verified)

These were wrong or underspecified in the first draft and would have cost a solo builder the morning. They are fixed throughout this doc; this box is the index.

| # | Was | Verified truth | Source |
|---|---|---|---|
| C1 | `runtime.encode(rgb)` / `runtime.predict(...)` | The runtime exposes **no** `encode`/`predict`. Real (all `async`, all **flat** `Float32Array`): `encodeFrames(frames, batch)`, `embedActionBlocks(blocks, batch, time)`, `predictLatents(latents, actEmb, batch, time)`, plus `rollout`, `planAction`, and scalars `hidden=192, numFrames=3, imageSize=224, actionDim=10`. | `lewm_runtime.mjs:271-294` |
| C2 | `encode(nextFrameRGB)` (raw RGB) | `encodeFrames` wants a **CHW Float32Array in [0,1]**; the env returns **HWC [0,255]**. Use `frameToModelInput(rgb)` (does `/255` **and** HWC→CHW). **Do not** apply ImageNet mean/std in JS — it is baked into the encoder ONNX graph. | `tworooms_env.mjs:249`; `manifest.json normalization.pixels` |
| C3 | `surprise = mse(predNext.at(-1), zNext)` | Outputs are flat — slice the last timestep `preds.subarray((numFrames-1)*hidden, numFrames*hidden)`. The MSE must **divide by `latentDim=192`** (mean over dims) so the scale matches the certified `baselineMse=0.06038`. | `lewm_probe.mjs:127-134` |
| C4 | `adapterFromInitAndOffset(offset)` | Real signature is a **single options object**: `adapterFromInitAndOffset({ inputDim, hiddenDim, initSeed, offset })`. The federated/probe path uses **`initSeed=42`**. A bare-offset call destructures the array and **throws**. | `lewm_adapter.mjs:208`; `lewm_system_round.mjs:33`; `lewm_probe.mjs:117` |
| C5 | `createAdapter({…seed:42})`; "W1 Xavier" | `createAdapter` default `seed=1` (`DEFAULT_HIDDEN=32`); **42 is the value the real path passes**, not the default. W1 init is **Gaussian × 1/√(inputDim)** (LeCun fan-in), not Xavier. `adapterForward(adapter, x, n) → {y, h}` is **batched** (3 args). | `lewm_adapter.mjs:151,154,231` |
| C6 | env `step(action)` | No `step()`. Real: `stepAgent(pos, action)`, `stepEpisode(episode, action)` (one 2-DOF env step), `stepEpisodeBlock(episode, actionBlock)` (advances 5 env-steps from a `Float32Array(10)`). `renderFrameRGB(agentPos, {renderTarget=false, targetPos=null})`. | `tworooms_env.mjs:99,121,294,220` |
| C7 | "reuse `tworooms_panel.mjs`" | `tworooms_panel.mjs` is a **planner lab** (`createTwoRoomsLab`, `mountTwoRoomsLab`); its `drawFrame` is **private**. SM-3 must **copy** the 6-line draw pattern, not mount the panel. | `tworooms_panel.mjs:27,106,113` |
| C8 | "`charts.mjs` oscilloscope helper" | `charts.mjs` has **no** oscilloscope/scrolling primitive. It exports `lineChart(opts)` (returns a `<div>` with `.__chartUpdate(series)`) + pure helpers. The scrolling meter is **[BUILD]** on top, and `lineChart` needs the `chart-*` CSS from `web/federated-demo/style.css:511-563` copied into our page. | `charts.mjs:270,292`; `style.css:511-563` |
| C9 | `run_system_composed_probe(...)` returns the offset | It is **keyword-only** (`seed` required, no default) and **does not return the 12,512-float offset** — only `serverOffsetParameterCount`. The offset is `service.model_revision(run_id, final_id)["adapterState"]`, computed at `system_probe.py:214` and discarded. SM-1 must add an explicit export path. | `system_probe.py:119,214`; `federated.py:674-683` |
| C10 | committed fallback at `runs/surprise/adapter_offset.json` | **`runs/` is gitignored** (`.gitignore:37`). The committed fallback must live at **`web/surprise-meter/fixtures/adapter_offset.json`** (tracked, served under `WEB_ROOT`). `runs/surprise/` is the throwaway working path only. | `.gitignore:37`; `server.py:560` |
| C11 | pre/post drop on a free-running live trajectory | The certified +12.3% is on **held-out probe pairs** (seed 991). `mean(post)<mean(pre)` is **not guaranteed** on an arbitrary scripted trajectory. The toggle must run on the **same `collectResidentPairs` distribution** so the drop *is* the certified number. | `lewm_probe.mjs:112`; `lewm_local_trainer.mjs:32` |
| C12 | "WASM fallback already there"; offline validation | There is **no** clean WebGPU→WASM degradation (sessions created once; failure throws) and **`onnxruntime` is absent everywhere** (uv venv, node, npm). Add a `?ep=wasm` force-WASM hatch; run offline bakes via `uv run --with onnxruntime …` (Python) or a node project with `onnxruntime-node` + an injected `file://` fetchFn. | `lewm_runtime.mjs:85-92,45` |

---

## 1. The two existing pillars

### A. The exported world model (in-browser)
`web/federated-demo/model/lewm-tworooms/{encoder,action,predictor}.onnx` + `manifest.json` (`lewm-browser-export/1`). Loaded by `web/federated-demo/lewm_runtime.mjs`:
- `loadLewmRuntime({ baseUrl, fetchFn=globalThis.fetch, ortApi=globalThis.ort, preferredProviders=["webgpu","wasm"] })` (line 43): SHA-256 hash-checks each graph against `manifest.files[*].sha256` via WebCrypto; throws `LewmUnsupportedError("no-ort"|"no-fetch"|"manifest-schema"|"hash-mismatch"|"session-create-failed")`.
- `createLewmRuntime(...)` (line 108) reads `manifest.architecture.{hiddenDim=192, numFrames=3, imageSize=224, actionDim=10}` and returns the object whose **real** methods are listed in C1. `actionDim=10` is the full action-block dim (5 env-steps × 2-DOF), **not** `ACTION_DIM=2`.
- **Verified IO:** encoder `pixels (B,3,224,224) [0,1] CHW → latent (B,192)`; action `(B,T,10) raw → (B,T,192)`; predictor `latents (B,T≤3,192) × action_embeddings (B,T≤3,192) → (B,T,192)`, next latent `= subarray((T-1)*192, T*192)`.

### B. The surprise quantity (already implemented)
`web/federated-demo/lewm_probe.mjs::probeAdapterOffset({ validationPairs, adaptedState, inputDim, hiddenDim=32, initSeed=42, improvementThreshold=0.02, seed=DEFAULT_PROBE_SEED })` (line 112). `baselineMse = mseOf(pairs.x)`, `adaptedMse = mseOf(adapter(pairs.x))`, where `mseOf` sums squared error over all samples × dims then **divides by `n*d`** (line 127-134). **For one pair the matching scalar is `(1/192)·Σ_k (pred_k − target_k)²`.** `baselineMse` **is** per-pair surprise; the live meter computes the same quantity per step.

### C. The env (pixel-exact, JS)
`web/federated-demo/tworooms_env.mjs`. Constants: `IMG_SIZE=224`, wall `x=112` thickness 10, door `y=49` half-width 14, `AGENT_SPEED=5`, `SUCCESS_DISTANCE=16`, `ACTION_BLOCK=5`, `ACTION_DIM=2 ∈ [-1,1]²`. Key exports (all verified): `sampleEpisode(rng)`, `createExpertPolicy({actionNoise, actionRepeatProb})` (line 140), `stepEpisode(episode, [dx,dy])` (121, immutable → new episode), `stepEpisodeBlock(episode, block)` (294, 5 env-steps from a `Float32Array(10)`), `packActionBlock(fiveActions) → Float32Array(10)` (281), `renderFrameRGB(agentPos, {renderTarget, targetPos})` (220, HWC [0,255]), `renderGoalFrameRGB(targetPos)` (261), `frameToModelInput(rgb) → Float32Array(3*224*224)` (249, /255 + HWC→CHW), `rgbToRGBA`, `distanceToTarget`, `clamp`. **`stepAgent` enforces hard wall-collision clamping** (forbidden band `x∈[112±12]` except the door `y∈[49±15.75]`): you **cannot** push the agent through a wall via an action — a "through-wall"/teleport perturbation must set `episode.agent` directly.

### D. The adapter (pre/post offset)
`web/federated-demo/lewm_adapter.mjs`: `adapterForward(adapter, x, n) → {y, h}` where `y = x + W2·tanh(W1·x + b1) + b2` (line 231; batched, `x` flat `n*192`). `createAdapter({inputDim=192, hiddenDim=DEFAULT_HIDDEN=32, seed=1})` (151): W2,b2 zero-init (identity residual at start), W1 = `gaussian(mulberry32(seed)) · 1/√inputDim`. `adapterFromInitAndOffset({inputDim, hiddenDim, initSeed, offset})` (208) = `createAdapter({…, seed:initSeed})` then `init + offset` (length-checked, **throws on mismatch**). **Offset length = 12512** (= 32·192 + 32 + 192·32 + 192). The real federated/probe path uses **`initSeed=42`**.

---

## 2. WS0 — Clean federated run (`SM-1`, Milestone 0, the priority)

**Goal:** a reliable, one-command, end-to-end federated **adapter-continuation** round, with committed audited evidence, a rehearsal gate, and the post-round offset exported. **SM-1 wraps the existing driver** `scripts/lewm_system_probe.py` (which already runs the round, regenerates the evidence, audits it inside the core, and `raise SystemExit` on `not passes`) and adds the **one missing step: offset export.** No Makefile exists — the "one command" is `scripts/surprise/run_clean_round.py`.

**Verified facts:**
- `run_system_composed_probe(*, participants, validation, checkpoint, manifest, rounds=3, steps_per_round=20, batch_size=32, seed, dim=192, deployment_target="system-probe") -> dict` (`system_probe.py:119`). **All keyword-only; `seed` is required (no default — `20260612` is the producer/CLI argparse default).** Returns the evidence dict (`schema, result, passes, serverOffsetParameterCount=12512, …`) but **not** the offset.
- `write_evidence(out: Path, evidence: dict)` (`system_probe.py:273`) = `out.write_text(json.dumps(evidence, indent=2)+"\n")`.
- Offset readback (verified end-to-end, 0.2 s, no onnxruntime/h5): `service.model_revision(run_id, final_id)["adapterState"]` → `list[float]` length 12512 (`federated.py:674-683`).
- `node` v25.2.0 is required (each round + the probe subprocess to `lewm_system_round.mjs`); it is pure-JS, **no onnxruntime-node**. The **real** probe path needs `onnxruntime + hdf5plugin + node + the 12.7 GB h5`; the **synthetic** path needs only `node`.
- Gitignored: `runs/` (`.gitignore:37`) and `web/federated-demo/model/` (manifest present locally).

**Offset export — Option B (recommended, non-breaking).** Add `offset_out: Path | None = None` to `run_system_composed_probe`; when set, write the captured `server_offset` to a **sidecar** file (never the evidence dict — `evidence_audit.py:48-54` forbids the substring `"adapterState"` in any `demo-evidence/1`/headline bundle, and `test_system_composed_headline_evidence_is_pinned` pins the file):
```python
server_offset = service.model_revision(run_id, final_id)["adapterState"]   # existing line 214
if len(server_offset) != PARAMS:
    raise RuntimeError("server offset has the wrong parameter count")
if offset_out is not None:                       # NEW sidecar
    offset_out.parent.mkdir(parents=True, exist_ok=True)
    offset_out.write_text(json.dumps([round(float(v), 8) for v in server_offset]) + "\n")
```
All four callers pass kwargs (`tests/ml/test_lewm_system_probe.py:79,206`, `scripts/lewm_probe_seedsweep.py:68`, `scripts/lewm_system_probe.py:68`), so a default-`None` kwarg breaks nothing. *(Option A — re-drive `FederatedDemoService` in the wrapper and call `model_revision(...)["adapterState"]` — avoids the core edit but duplicates the round loop; choose only if you must not touch `system_probe.py`.)*

**File layout (gitignore-correct):**
- Working output (ephemeral, gitignored OK): `runs/surprise/adapter_offset.json` — `list[float]` len 12512.
- **Committed fallback (tracked, served):** `web/surprise-meter/fixtures/adapter_offset.json` — `git add` a copy of the run output. **Never** commit under `runs/`.

**`scripts/surprise/run_clean_round.py` [BUILD]** — mirror `scripts/lewm_system_probe.py:35-105`; add `--offset-out` (default `runs/surprise/adapter_offset.json`) and pass `offset_out=`:
```python
# argparse: --h5 (required) --model-dir --out (default docs/evidence/lewm_tworooms_system_probe.json)
#   --offset-out --participants 2 --episodes-per-participant 8 --validation-episodes 4
#   --rounds 3 --steps-per-round 20 --batch-size 32 --seed 20260612
split = build_probe_split(h5_path=args.h5, model_dir=args.model_dir, seed=args.seed,
    participants=args.participants, episodes_per_participant=..., validation_episodes=...)  # KEYWORD-ONLY, seed REQUIRED
manifest = load_lewm_manifest(str(args.model_dir / "manifest.json"))   # SystemExit if falsy
evidence = run_system_composed_probe(participants=split.participants, validation=split.validation,
    checkpoint=split.checkpoint, manifest=manifest, rounds=args.rounds, steps_per_round=...,
    batch_size=..., seed=args.seed, dim=split.dim, offset_out=args.offset_out)
write_evidence(args.out, evidence)
print(json.dumps({k: evidence["result"][k] for k in ("verdict","baselineMse","adaptedMse","relativeImprovement")}
    | {"passes": evidence["passes"], "offsetFile": str(args.offset_out),
       "offsetParameterCount": evidence["serverOffsetParameterCount"]}, indent=2))
if not evidence["passes"]:
    raise SystemExit("clean round is not a clean pass — blocks public positive claims")
```
Real invocation (onnxruntime is **not** in the venv — use the ephemeral install):
```
uv run --with onnxruntime --with hdf5plugin python scripts/surprise/run_clean_round.py \
    --h5 ~/.cache/lensemble-lewm/tworoom.h5
```

**`scripts/surprise/rehearsal.py` [BUILD]** (CI-safe, **synthetic, node-only, no onnxruntime/h5**) — mirror `scripts/lewm_demo_rehearsal.py`. Use the `_bias_correctable_pairs` synthetic pairs (copy `tests/ml/test_lewm_system_probe.py:52-68`):
```python
evidence = run_system_composed_probe(participants=[pairs(24,192,s) for s in (11,22)],
    validation=pairs(16,192,99), checkpoint=manifest()["checkpoint"], manifest=manifest(),
    rounds=2, steps_per_round=10, batch_size=16, seed=7, dim=192,
    offset_out=Path("runs/surprise/rehearsal_offset.json"))
assert evidence["passes"] is True and evidence["claimAuditViolations"] == 0
assert evidence["serverOffsetParameterCount"] == 12512
assert evidence["result"]["verdict"] == "improved" and evidence["result"]["collapseRisk"] is False
assert len(json.loads(Path("runs/surprise/rehearsal_offset.json").read_text())) == 12512
print(json.dumps({"ok": True, "schema": "surprise-clean-round-rehearsal/1",
    "passes": evidence["passes"], "offsetParameterCount": 12512,
    "relativeImprovement": evidence["result"]["relativeImprovement"]}, indent=2, sort_keys=True))
```
Companion test **`tests/ml/test_surprise_clean_round.py` [BUILD]** — mirror `tests/ml/test_lewm_rehearsal.py`: load the script by path (`importlib.util.spec_from_file_location`), run as subprocess, assert `returncode==0` and `report["schema"]`. Guard with `pytest.mark.skipif(shutil.which("node") is None)`. Lands in `tests/ml/` → CI gate-4 picks it up automatically.

**Runbook [BUILD]:** extend `docs/roadmap/HACKATHON_PARIS.md` (the live browser QR round: `uv run lensemble demo federated --port 8765` → `http://127.0.0.1:8765/web/federated-demo/`, join, train, aggregate, read the probe tick-up).

**Acceptance:** one command → green audit + committed evidence; rehearsal exits 0; offset (len 12512) written to the sidecar and copied to the committed fallback; runbook reproduces a live round.

> Honest framing everywhere (AGENTS.md "Claim Discipline"): "federated **adapter continuation** on a frozen checkpoint." Single local coordinator, mean of clipped deltas, no DP/secure-agg.

---

## 3. WS1 — In-browser surprise engine (`SM-2`)

**Goal:** per-step scalar surprise over a live, **on-distribution** trajectory, in-browser, via the exported graphs — the same quantity as the certified `baselineMse=0.06038`.

`web/surprise-meter/` does not exist; create it. Cross-page import is the proven symmetric pattern (`browser_learner.mjs:10` imports from `../dynamic-env-demo/`), so `surprise_engine.mjs` imports from `../federated-demo/{tworooms_env,lewm_runtime,lewm_adapter,rng}.mjs`.

**Preprocessing (definitive, resolves R8):** pixels → `frameToModelInput(renderFrameRGB(pos))` (`/255` + HWC→CHW); **never** apply ImageNet mean/std in JS (baked into the encoder graph, `mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]`). Actions → `packActionBlock(fiveSubActions) → Float32Array(10)`, passed **raw** to `embedActionBlocks` (the expert z-score is baked into the action graph). On-distribution actions come from `createExpertPolicy({actionNoise:1.0, actionRepeatProb:0})` (matches the probe validation set, `lewm_probe.mjs:42`), seeded by a single `mulberry32(seed)` stream (evidence `seed=20260618`).

**The surprise scalar:** `surprise = (1/192)·Σ_{k} (predNext_k − zNext_k)²` — mean over the 192 latent dims, identical to `mseOf` for one pair.

**[BUILD]** `web/surprise-meter/surprise_engine.mjs` — `createSurpriseEngine` mirrors `collectResidentPairs` (`lewm_local_trainer.mjs:56-101`):
```js
import { renderFrameRGB, frameToModelInput, packActionBlock, stepEpisodeBlock,
         sampleEpisode, createExpertPolicy } from "../federated-demo/tworooms_env.mjs";
import { mulberry32 } from "../federated-demo/rng.mjs";
import { adapterFromInitAndOffset, adapterForward } from "../federated-demo/lewm_adapter.mjs";

const HIDDEN = 192, WINDOW = 3;                 // = runtime.hidden, runtime.numFrames

export function createSurpriseEngine(runtime, { seed = 20260618, actionNoise = 1.0,
                                                actionRepeatProb = 0, adapterOffset = null } = {}) {
  const rng = mulberry32(seed >>> 0);
  const expert = createExpertPolicy({ actionNoise, actionRepeatProb });
  let episode = sampleEpisode(rng);
  const latBuf = [], actEmbBuf = [];            // ring buffers of Float32Array(192)
  const adapter = adapterOffset
    ? adapterFromInitAndOffset({ inputDim: 192, hiddenDim: 32, initSeed: 42, offset: adapterOffset })
    : null;

  async function encodeCurrent() {
    const rgb = renderFrameRGB(episode.agent, { renderTarget: true, targetPos: episode.target });
    const lat = await runtime.encodeFrames(frameToModelInput(rgb), 1);    // flat (1*192)
    return { rgb, latent: Float32Array.from(lat.subarray(0, HIDDEN)) };
  }
  async function reset() { episode = sampleEpisode(rng); latBuf.length = 0; actEmbBuf.length = 0;
                           latBuf.push((await encodeCurrent()).latent); }

  // advance one MODEL step (= 5 env steps). perturb(episode) bypasses wall-collision (teleport/through-wall).
  async function step({ forcedBlock = null, perturb = null } = {}) {
    if (episode.done) return { done: true };
    const subs = []; for (let i = 0; i < 5; i++) subs.push(expert(episode, rng));
    const block = forcedBlock ?? packActionBlock(subs);                  // Float32Array(10)
    const ae = await runtime.embedActionBlocks(block, 1, 1);             // raw block -> (1,1,192)
    actEmbBuf.push(Float32Array.from(ae.subarray(0, HIDDEN))); if (actEmbBuf.length > WINDOW) actEmbBuf.shift();
    const prevRgb = renderFrameRGB(episode.agent, { renderTarget: true, targetPos: episode.target });
    episode = perturb ? perturb(episode) : stepEpisodeBlock(episode, block);
    const { rgb: nextRgb, latent: zNext } = await encodeCurrent();
    const frameDiff = meanAbsPixelDiff(prevRgb, nextRgb);
    // WARM-UP: HOLD until 3 latents + 3 action-embeds exist (see bootstrap rule below)
    if (latBuf.length < WINDOW || actEmbBuf.length < WINDOW) {
      latBuf.push(zNext); if (latBuf.length > WINDOW) latBuf.shift();
      return { surprisePre: null, surprisePost: null, frameDiff, warmup: true, agent: { ...episode.agent } };
    }
    const hl = new Float32Array(WINDOW*HIDDEN), ha = new Float32Array(WINDOW*HIDDEN);
    for (let i = 0; i < WINDOW; i++) { hl.set(latBuf[i], i*HIDDEN); ha.set(actEmbBuf[i], i*HIDDEN); }
    const preds = await runtime.predictLatents(hl, ha, 1, WINDOW);       // flat (1*3*192)
    const predNext = preds.subarray((WINDOW-1)*HIDDEN, WINDOW*HIDDEN);   // last timestep = predicted next latent
    const surprisePre = mse(predNext, zNext);
    let surprisePost = null;
    if (adapter) surprisePost = mse(adapterForward(adapter, Float32Array.from(predNext), 1).y, zNext);
    latBuf.push(zNext); if (latBuf.length > WINDOW) latBuf.shift();
    return { surprisePre, surprisePost, frameDiff, warmup: false, agent: { ...episode.agent }, done: episode.done };
  }
  return { reset, step, get episode() { return episode; } };
}
const mse = (a, b) => { let s = 0; for (let k = 0; k < 192; k++) { const d = a[k]-b[k]; s += d*d; } return s/192; };
const meanAbsPixelDiff = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i]-b[i]); return s/(a.length*255); };
```

**Ring-buffer bootstrap (definitive):** `t=0` → only `z₀` exists → **HOLD** (`surprisePre=null`, meter greyed). `t=1` → `z₀,z₁` → **HOLD** (a `T<3` window is a valid graph input — parity checks pass at `[1,1,192]`/`[1,2,192]` — but it is **not on the certified `baselineMse` scale**, which is measured only on full `T=3` windows by `collectResidentPairs:89`). `t=2` (third model step) → buffers full → **first real surprise** (matches `collectResidentPairs`' first pair at `t = window-1`). We HOLD rather than emit partial-window numbers to protect parity and `frameDiffCorrelation`.

**[BUILD]** `web/surprise-meter/surprise_selftest.mjs` — **ONNX-free** parity test (mirror `lewm_probe_selftest.mjs:40-72`). Build a `fakeRuntime()` (`{hidden, numFrames, imageSize, actionDim, encodeFrames, embedActionBlocks, predictLatents}`) whose predictor applies a known dim-0 bias, then assert: (a) the engine's `surprisePre` equals `lewm_probe.mjs mseOf` on the shared `(pred, target)` to **≤1e-4**, (b) finite surprise, (c) `surprisePre === null` for steps 0,1 (warm-up HOLD). Run: `node web/surprise-meter/surprise_selftest.mjs` (no ort). The **real CLS-predictor-error** behaviour (does surprise spike on perturbation — R1) **cannot be validated offline** here; it is an **in-browser-only** check (see §4 R1 and the runsheet pre-flight).

**Acceptance:** per-step surprise stream over a deterministic expert trajectory (seed 20260618, actionNoise 1.0); warm-up holds 2 steps, first surprise at step index 2; `surprise_selftest.mjs` exits 0 (parity ≤1e-4, finite, warm-up null); ≥10 steps/s in-browser (trivial).

---

## 4. WS2 — Surprise UI + perturbation + frame-diff (`SM-3`)

**Goal:** the hero visual — a live meter that makes surprise legible and proves surprise ≠ motion.

**[BUILD]** `web/surprise-meter/{index.html, app.mjs}`. Template from `web/dynamic-env-demo/`; inline CSS. ORT tag (copy `dynamic-env-demo/index.html:7`) — **pin the version AND vendor it locally** (the projector may be offline; a floating CDN tag is a silent stage-breaker): vendor `ort.webgpu.min.js` under `web/surprise-meter/vendor/` and load that, with the pinned CDN as a documented fallback:
```html
<script src="./vendor/ort.webgpu.min.js"></script>   <!-- vendored; pin = onnxruntime-web@<verified> -->
```

**Env canvas (copy, don't mount):** `tworooms_panel.mjs`'s `drawFrame` is private — copy the 6 lines (`tworooms_panel.mjs:106-111`); canvas must be `224×224`:
```js
function drawFrame(canvas, rgb) {
  const ctx = canvas.getContext("2d", { alpha: false });
  ctx.imageSmoothingEnabled = false;
  ctx.putImageData(new ImageData(rgbToRGBA(rgb), IMG_SIZE, IMG_SIZE), 0, 0);   // IMG_SIZE = 224
}
```
Tint the agent/border by current surprise **on the canvas/DOM** (translucent overlay or `canvas.style.outline`) — never mutate the RGB fed to the encoder.

**Surprise meter / oscilloscope (100% [BUILD] on `lineChart`):** `charts.mjs` exposes `lineChart(opts)` → a `<div.chart>` with `.__chartUpdate(nextSeries)` (line 292) + `seriesExtent`, `niceTicks`, `formatTick`, `CHART_PALETTE`. There is **no** scrolling primitive — build it: keep a sliding window of `{x:t, y:surprise}` (cap ~120 points) and call `wrap.__chartUpdate(series)` each tick. **CSS dependency:** `lineChart` emits `chart`, `chart-title`, `chart-svg`, `chart-grid`, `chart-tick`, `chart-legend`, `chart-legend-item`, `chart-swatch` whose styles + `:root` variables live **only** in `web/federated-demo/style.css:511-563` — copy that block into our inline CSS or the chart renders unstyled.

**Perturbations (exact):** each must visibly raise the **pre** meter.
- **Teleport:** `perturb = (ep) => ({ ...ep, agent: { x: <new x∈[14,209]>, y: <new y> } })` — sets position directly (the env clamps wall crossings, C6).
- **OOD action:** `forcedBlock` = an action block far outside the expert z-score range (e.g. all `±3`).
- **Through-wall:** teleport across `x=112` outside the door band — again via `perturb`, not an action.
A teleport invalidates the 3-frame buffer; **keep the stale history deliberately** so the meter spikes (that *is* the demo). The 2 post-teleport steps are the spike, not warm-up.

**Frame-diff baseline:** a second `lineChart` trace = `meanAbsPixelDiff(frame_{t-1}, frame_t)`. ⚠️ **Scale mismatch — fix it or the credibility beat is invisible:** surprise lives at ~0.05–0.30 while normalized frame-diff is ~0–0.02 (1–2 orders apart). On a shared y-axis the contrast vanishes. **Normalize both traces to [0,1] by their own running max (or z-score) before plotting**, or use two stacked single-series charts; document the normalization on screen. Curate demo moments where motion is high but surprise low (predictable glide) and motion low but surprise high (teleport) — that contrast is the credibility. Render the "illustrative on the TwoRooms backbone — not a calibrated anomaly detector" nonClaim **visibly** on this screen's footer.

**Pre/post overlay:** uses SM-4's math (§5). HUD reads the certified numbers (S7), never recomputes the headline.

> ⚠️ **R1 (in-browser-only):** validate before stage in Chrome that ≥1 perturbation channel raises surprise (`perturbationSpikeRatio` or `oodActionSpikeRatio` > 1.5). If neither does, foreground the OOD-action + pre/post contrasts (Decision S8). It **cannot** be checked offline (no onnxruntime).

**Acceptance:** live meter + 224×224 env at smooth framerate; chart styled (CSS copied); perturbations spike the pre meter; frame-diff trace present; runs on the **WASM** path (no WebGPU dependency for the meter); `?ep=wasm` force-WASM hatch works (§ R5).

---

## 5. WS3 — Pre/post-federation toggle (`SM-4`)

**Goal:** show the federated adapter measurably reduces surprise on **in-distribution** dynamics — and make that drop *be* the certified +12.3%.

**Load-bearing subtlety (C11):** the certified +12.3% is on **held-out probe pairs** (deterministic noisy-expert rollouts, `DEFAULT_PROBE_SEED=991`). On a teleport/OOD trajectory the adapter was never trained for those pairs and the drop can vanish or reverse. **Guarantee the drop by reusing the exact certified pipeline** — `collectResidentPairs` (`lewm_local_trainer.mjs:32`), the same source the probe uses. Then `surprise_pre`/`surprise_post` **are** `baselineMse`/`adaptedMse` per-pair, and `surpriseDropRatio == result.relativeImprovement (0.12275)`. SM-3's perturbation spikes stay on the **pre-only** meter; SM-4's drop runs on a clean in-distribution pair set **once at load** (deterministic — no per-toggle re-rolling, which would inject seed variance, worst case +5.4%).

**[BUILD]** in `surprise_engine.mjs`:
```js
import { collectResidentPairs } from "../federated-demo/lewm_local_trainer.mjs";
import { adapterFromInitAndOffset, adapterForward } from "../federated-demo/lewm_adapter.mjs";
const ADAPTER_INPUT_DIM = 192, ADAPTER_HIDDEN = 32, ADAPTER_INIT_SEED = 42, PROBE_SEED = 991;

const pairSurprise = (pred, target, i, d) => {           // /d => matches baselineMse=0.06038 scale
  let s = 0; for (let k = 0; k < d; k++) { const diff = pred[i*d+k] - target[i*d+k]; s += diff*diff; } return s/d;
};

export async function buildPrePostStreams({ runtime, offset, seed = PROBE_SEED, episodes = 2, maxModelSteps = 10 }) {
  const { pairs } = await collectResidentPairs({ runtime, seed, episodes, maxModelSteps,
    policyOptions: { actionNoise: 1.0, actionRepeatProb: 0 }, minPairs: 4 });   // == buildValidationSet's policy
  const d = runtime.hidden;                                                      // 192
  const adapter = adapterFromInitAndOffset({ inputDim: ADAPTER_INPUT_DIM, hiddenDim: ADAPTER_HIDDEN,
                                             initSeed: ADAPTER_INIT_SEED, offset });
  const prePred = pairs.x;                                                        // identity = frozen predictor output
  const postPred = adapterForward(adapter, pairs.x, pairs.count).y;               // adapter-corrected
  const surprisePre = [], surprisePost = [];
  for (let i = 0; i < pairs.count; i++) {
    surprisePre.push(pairSurprise(prePred, pairs.target, i, d));
    surprisePost.push(pairSurprise(postPred, pairs.target, i, d));
  }
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const meanPre = mean(surprisePre), meanPost = mean(surprisePost);
  return { surprisePre, surprisePost, meanPre, meanPost,
           surpriseDropRatioLive: meanPre > 0 ? (meanPre - meanPost) / meanPre : 0, pairCount: pairs.count };
}
```
**Identity check:** with the certified revision's offset and matching seed/episodes, `meanPre == baselineMse`, `meanPost == adaptedMse`, `surpriseDropRatioLive == result.relativeImprovement`.

**Offset sourcing & guard:** `const offset = await fetch("./fixtures/adapter_offset.json").then(r => r.json());` (committed fallback; the live SM-1 sidecar is `runs/surprise/adapter_offset.json`, not committed). **Assert `offset.length === 12512` and `offset.some(v => v !== 0)`** — an all-zero offset = identity = zero drop = dead demo.

**HUD sourcing (read, never recompute — S7):**

| HUD label | File | JSON path | Value |
|---|---|---|---|
| +12.3% (this run) | `lewm_tworooms_system_probe.json` | `result.relativeImprovement` | 0.12275377883038366 |
| +16.8% (mean, 5 seeds) | `lewm_tworooms_probe_seedsweep.json` | `distribution.relativeImprovementMean` | 0.16787180214169914 |
| +5.4% (worst, seed 2) | `lewm_tworooms_probe_seedsweep.json` | `distribution.relativeImprovementMin` | 0.054144202317108696 |

Show the **worst case** beside the mean (AGENTS.md binds the seed-robustness claim *with* its worst case). Label the live drop "this run", distinct from the certified headline.

**Acceptance:** `mean(surprise_post) < mean(surprise_pre)` on the held-out pair set; `surpriseDropRatioLive ≈ 0.12275` on the certified offset; HUD numbers sourced (full precision); adapter call uses the object signature; toggle smooth.

---

## 6. WS4 — Evidence JSON (`SM-5`)

**Goal:** back the on-screen numbers with a generated, tested file. Schema `lewm-surprise/1` → `docs/evidence/lewm_tworooms_surprise.json`. Producer `scripts/lewm_surprise_check.py` [BUILD] (mirror `scripts/lewm_system_probe.py`).

**Offline surprise generation** — mirror `_pairs_from_episodes` (`lewm_tworooms_probe_pairs.py:38-71`), the canonical verified ONNX replay (needs `onnxruntime` via the ephemeral install). One model step = 5 env steps:
```python
import h5py, hdf5plugin, onnxruntime as ort, numpy as np
enc  = ort.InferenceSession(model_dir/"lewm_tworooms_encoder.onnx",  providers=["CPUExecutionProvider"])
act  = ort.InferenceSession(model_dir/"lewm_tworooms_action.onnx",   providers=["CPUExecutionProvider"])
pred = ort.InferenceSession(model_dir/"lewm_tworooms_predictor.onnx",providers=["CPUExecutionProvider"])
frames  = (pixels[idx].astype(np.float32)/255.0).transpose(0,3,1,2)   # ImageNet norm is INSIDE the encoder graph
latents = enc.run(None, {"pixels": frames})[0]                        # (T,192)
act_emb = act.run(None, {"actions": raw_blocks.reshape(1,n,10)})[0][0] # raw 10-d blocks; z-score inside the graph
for t in range(2, n):                                                  # WARM-UP: undefined for first 2 model steps
    preds    = pred.run(None, {"latents": latents[t-2:t+1][None], "action_embeddings": act_emb[t-2:t+1][None]})[0]
    predNext = preds[0,-1]; zNext = latents[t+1]
    surprise = float(((predNext - zNext)**2).mean())                  # MEAN OVER 192 DIMS — == baselineMse scale
```
`meanSurprisePre` = mean over valid windows (≈ `baselineMse=0.06038`). `meanSurprisePost`/`surpriseDropRatio` are **sourced** from `result.adaptedMse` / `result.relativeImprovement` (assert match ≤1e-6 — read **full precision**, never the display rounding `0.1227`). Perturbation channels (`perturbationSpikeRatio`, `oodActionSpikeRatio`) and `frameDiffCorrelation` are **measured** by re-running the replay with perturbed inputs — the contract's `4.8/3.1/0.12` are **placeholders**, emit observed values.

**`result` fields** (`02-data-contracts.md` §1 is the canonical shape; additions: `federatedSeedWorst`, `federatedSeedStdev`, `warmupSteps:int==2`, and split `surpriseDropRatio` into the **sourced** `federatedRelativeImprovement` vs the **measured** `surpriseDropRatioLive`).

**`passes` predicate** (thresholds verified against the real scale): `meanSurprisePost < meanSurprisePre` (0.0530<0.0604) **and** `surpriseDropRatioLive > 0.02` **and** (`perturbationSpikeRatio>1.5` **or** `oodActionSpikeRatio>1.5`) **and** `abs(frameDiffCorrelation) < 0.6` **and** `nonClaims` contains the four mandatory negations.

**Four mandatory `nonClaims` (verbatim):** adapter-continuation-not-training; surprise-is-scalar-CLS-prediction-error; perturbation-illustrative-not-calibrated-detector; single-coordinator-no-secure-agg/DP.

**Test [BUILD]** `tests/ml/test_lewm_surprise.py` — mirror `test_lewm_system_probe.py:109-178`, **skip-when-absent** (`if not committed.is_file(): pytest.skip(...)`) so CI gate-4 (no onnxruntime) does not hard-fail; assert schema, field types, `nonClaims` negations, `passes` predicate, and the sourcing equalities (`federatedRelativeImprovement == system_probe.result.relativeImprovement` within 1e-6; `federatedSeedMean == seedsweep.distribution.relativeImprovementMean`). New schema → add a `CHANGELOG.md [Unreleased]` "Added" entry.

**Acceptance:** deterministic producer; test green; `uv run python scripts/check_docs_links.py docs SPEC.md README.md` and `uv run python -m mkdocs build --strict` pass (new evidence JSON needs **no** mkdocs nav entry; only new `.md` outside `docs/plans/` would).

---

## 7. WS5 — Rehearsal + fallback + capture (`SM-6`)

**Pre-flight 0 (start-of-day 2026-06-18, BLOCKING): install an offline ONNX runtime.** `onnxruntime` is absent from the uv venv, system node, and npm — so **no fallback can be baked at the venue** without this. Two routes:
- **Python (offset, via h5):** `uv run --with onnxruntime --with hdf5plugin python scripts/surprise/run_clean_round.py --h5 ~/.cache/lensemble-lewm/tworoom.h5` (h5 present, 12.7 GB).
- **Node (trajectory, via rendered env frames):** `cd web/surprise-meter && npm init -y && npm install onnxruntime-node@1.*` (build-time only — do **not** add a root dependency). The node bake **must** pass `preferredProviders:["cpu"]` (onnxruntime-node has no webgpu/wasm EP) and **must inject a `file://` fetchFn** (node global `fetch` cannot load `file://`):

**[BUILD]** `scripts/surprise/bake_trajectory.mjs` (node + onnxruntime-node):
```js
import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-node";
import { loadLewmRuntime } from "../../web/federated-demo/lewm_runtime.mjs";
const BASE = new URL("../../web/federated-demo/model/lewm-tworooms/", import.meta.url);
const fetchFn = async (url) => {
  const buf = readFileSync(new URL(url, BASE));
  return { ok: true, json: async () => JSON.parse(buf.toString("utf8")),
           arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
};
const runtime = await loadLewmRuntime({ baseUrl: BASE.href, fetchFn, ortApi: ort, preferredProviders: ["cpu"] });
// then the §3 per-step surprise loop (encodeFrames -> embedActionBlocks -> predictLatents -> mse/192),
// warm-up HOLD 2 steps, optional adapter for surprisePost; cap steps <= 600.
```
Output schema **pinned `lewm-surprise-traj/1`** (rename from the plan's stray `cartographer-surprise-traj/1`) → `web/surprise-meter/data/surprise_trajectory.json`; include `perturbations[]`, `provenance.{adapterOffsetFile, bakeSeed}`. The viewer validates this schema string before replaying.

**Rehearsal gate [BUILD]** `scripts/surprise/rehearsal.py` (extend SM-1's): Milestone-0 synthetic round passes + audited; `surprise_selftest.mjs` exits 0; offset present (len 12512); `web/surprise-meter/{index.html,app.mjs,surprise_engine.mjs}` + `fixtures/adapter_offset.json` + `data/surprise_trajectory.json` exist. Companion `tests/ml/test_surprise_rehearsal.py` (skip-if-no-node).

**Fallback (committed):** `web/surprise-meter/fixtures/adapter_offset.json` (len 12512) + `web/surprise-meter/data/surprise_trajectory.json`. **Both must be tracked (not under `runs/`).**

**Capture:** ≤20 s clip (glide → perturb → spike → frame-diff contrast → pre/post toggle) + result card (+12.3% **and** worst-case +5.4%, "less surprised after the room trained an adapter — no data shared"), citing the evidence file.

**Acceptance:** rehearsal exits 0; committed fallbacks present and tracked; clip + card exported; pre-flight checklist (runsheet) green on the presentation browser at projector resolution with the `?ep=wasm` path confirmed.

---

## 8. External APIs touched (verified index)

| Need | Symbol (real signature) | Location |
|---|---|---|
| Load runtime | `loadLewmRuntime({baseUrl, fetchFn, ortApi, preferredProviders})`, `createLewmRuntime(...)` | `lewm_runtime.mjs:43,108` |
| Encode/embed/predict | `encodeFrames(frames,batch)`, `embedActionBlocks(blocks,batch,time)`, `predictLatents(latents,actEmb,batch,time)` (flat arrays) | `lewm_runtime.mjs:271-294` |
| Surprise math (reuse) | `probeAdapterOffset({validationPairs,adaptedState,inputDim,hiddenDim,initSeed,…})`; `mseOf` (÷ n·d) | `lewm_probe.mjs:112,127` |
| In-distribution pairs | `collectResidentPairs({runtime,seed,episodes,maxModelSteps,policyOptions,minPairs})` | `lewm_local_trainer.mjs:32` |
| Env | `sampleEpisode`,`createExpertPolicy`,`stepEpisode`,`stepEpisodeBlock`,`packActionBlock`,`renderFrameRGB`,`frameToModelInput`,`rgbToRGBA` | `tworooms_env.mjs:99-294` |
| Adapter | `adapterFromInitAndOffset({inputDim,hiddenDim,initSeed,offset})`, `adapterForward(adapter,x,n)→{y,h}` | `lewm_adapter.mjs:208,231` |
| Charts | `lineChart(opts)→div.__chartUpdate(series)`, `seriesExtent`,`niceTicks`,`formatTick`,`CHART_PALETTE` (+ CSS `style.css:511-563`) | `charts.mjs:270,292` |
| Headless round | `run_system_composed_probe(*, …, seed, offset_out)`, `write_evidence(out, evidence)` | `system_probe.py:119,273` |
| Offset readback | `service.model_revision(run_id, final_id)["adapterState"]` (len 12512) | `federated.py:674-683` |
| Evidence audit | `audit_real_lewm_evidence` (forbids the `"adapterState"` substring) | `evidence_audit.py:48,61` |
| Probe split | `build_probe_split(*, h5_path, model_dir, seed, participants, episodes_per_participant, validation_episodes)` | `lewm_tworooms_probe_pairs.py:74` |
| Static serving | `_static`, `WEB_ROOT` (serves any path under `web/`) | `server.py:560,27` |
