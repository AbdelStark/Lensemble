// Node-runnable selftest for the browser-local LeWM adapter continuation (#319).
//
// Run: node web/federated-demo/lewm_adapter_selftest.mjs [--dump-sigreg-fixture]
// Deterministic and ONNX-free. Pins: identity-at-init, an analytic-vs-numerical gradient check,
// real loss decrease, variance-floor activation on collapsing targets, SIGReg statistic behavior
// (≈0 on standard normal, large on collapse), delta clipping, and the resident-pair collection
// loop over a fake frozen runtime. --dump-sigreg-fixture prints a JSON fixture that the Python
// suite replays through lensemble.model.sigreg to prove the JS port matches torch.

import {
  adapterForward,
  buildSketch,
  computeAdapterDelta,
  createAdapter,
  createOptimizer,
  flattenParams,
  latentDiagnostics,
  lossAndGrads,
  parameterCount,
  sigregStatistic,
  trainAdapterOnPairs,
  trainStep,
} from "./lewm_adapter.mjs";
import { collectResidentPairs, runLocalAdapterContinuation } from "./lewm_local_trainer.mjs";
import { mulberry32 } from "./rng.mjs";

const failures = [];
let total = 0;

function check(name, fn) {
  total += 1;
  try {
    fn();
  } catch (error) {
    failures.push({ name, error: String(error?.message ?? error) });
  }
}

async function checkAsync(name, fn) {
  total += 1;
  try {
    await fn();
  } catch (error) {
    failures.push({ name, error: String(error?.message ?? error) });
  }
}

function assert(cond, message) {
  if (!cond) throw new Error(message);
}

function gaussian(rng) {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// --- fixture dump mode (consumed by tests/ml/test_lewm_adapter.py for torch parity) ---

if (process.argv.includes("--dump-sigreg-fixture")) {
  const rng = mulberry32(123);
  const n = 256;
  const d = 16;
  const normal = new Float32Array(n * d);
  for (let i = 0; i < normal.length; i += 1) normal[i] = gaussian(rng);
  const collapsed = new Float32Array(n * d);
  for (let i = 0; i < n; i += 1) {
    for (let k = 0; k < d; k += 1) collapsed[i * d + k] = 0.05 * gaussian(rng);
  }
  const sketch = buildSketch(42, d, 32);
  const payload = JSON.stringify({
    d,
    n,
    sketchDim: 32,
    sketch: Array.from(sketch.data),
    normal: Array.from(normal),
    collapsed: Array.from(collapsed),
    normalStatistic: sigregStatistic(normal, n, d, sketch),
    collapsedStatistic: sigregStatistic(collapsed, n, d, sketch),
  });
  // drain stdout before exiting — process.exit right after console.log truncates large pipes
  await new Promise((resolve) => process.stdout.write(payload + "\n", () => resolve()));
  process.exit(0);
}

// ---------------------------------------------------------------------------
// adapter math
// ---------------------------------------------------------------------------

check("adapter is the identity at init (zero-init residual head)", () => {
  const adapter = createAdapter({ inputDim: 8, hiddenDim: 4, seed: 3 });
  const rng = mulberry32(1);
  const x = Float32Array.from({ length: 3 * 8 }, () => gaussian(rng));
  const { y } = adapterForward(adapter, x, 3);
  for (let i = 0; i < y.length; i += 1) assert(Math.abs(y[i] - x[i]) < 1e-7, "identity at init");
});

check("analytic gradients match numerical gradients", () => {
  const adapter = createAdapter({ inputDim: 6, hiddenDim: 3, seed: 5 });
  const rng = mulberry32(7);
  // move off the zero-init point so every parameter has gradient flow
  for (const key of ["w1", "b1", "w2", "b2"]) {
    const p = adapter.params[key];
    for (let i = 0; i < p.length; i += 1) p[i] += 0.3 * gaussian(rng);
  }
  const n = 5;
  const x = Float32Array.from({ length: n * 6 }, () => gaussian(rng));
  const target = Float32Array.from({ length: n * 6 }, () => 2.0 * gaussian(rng));
  const opts = { lambda: 0.7, varFloorRatio: 0.9 };
  const { grads } = lossAndGrads(adapter, { x, target, n }, opts);
  const epsStep = 1e-3;
  for (const key of ["w1", "b1", "w2", "b2"]) {
    const p = adapter.params[key];
    for (const idx of [0, Math.floor(p.length / 2), p.length - 1]) {
      const orig = p[idx];
      p[idx] = orig + epsStep;
      const plus = lossAndGrads(adapter, { x, target, n }, opts).totalLoss;
      p[idx] = orig - epsStep;
      const minus = lossAndGrads(adapter, { x, target, n }, opts).totalLoss;
      p[idx] = orig;
      const numeric = (plus - minus) / (2 * epsStep);
      const analytic = grads[key][idx];
      const denom = Math.max(1e-4, Math.abs(numeric), Math.abs(analytic));
      assert(
        Math.abs(numeric - analytic) / denom < 0.05,
        `${key}[${idx}]: numeric=${numeric.toExponential(3)} analytic=${analytic.toExponential(3)}`,
      );
    }
  }
});

check("training reduces prediction loss on a learnable synthetic task", () => {
  const d = 12;
  const rng = mulberry32(11);
  const n = 64;
  const x = new Float32Array(n * d);
  const target = new Float32Array(n * d);
  for (let i = 0; i < n; i += 1) {
    for (let k = 0; k < d; k += 1) {
      const v = gaussian(rng);
      x[i * d + k] = v;
      target[i * d + k] = 0.85 * v + 0.4 + 0.02 * gaussian(rng); // learnable affine gap
    }
  }
  const adapter = createAdapter({ inputDim: d, hiddenDim: 8, seed: 2 });
  const report = trainAdapterOnPairs(adapter, { x, target, count: n }, { steps: 120, batchSize: 32, seed: 4 });
  assert(report.lastLoss < report.firstLoss * 0.5, `loss halved: ${report.firstLoss} -> ${report.lastLoss}`);
  assert(report.history.every((h) => Number.isFinite(h.totalLoss)), "finite losses");
  assert(report.diagnostics.length >= 2, "diagnostics emitted during training");
});

check("variance floor activates when targets would collapse the output", () => {
  const d = 6;
  const rng = mulberry32(13);
  const n = 48;
  const x = new Float32Array(n * d);
  const target = new Float32Array(n * d);
  for (let i = 0; i < n; i += 1) {
    for (let k = 0; k < d; k += 1) {
      x[i * d + k] = gaussian(rng);
      target[i * d + k] = 1.5 * gaussian(rng);
    }
  }
  // force a collapsed adapter output: large negative residual cancels x
  const adapter = createAdapter({ inputDim: d, hiddenDim: 4, seed: 6 });
  const { varLoss } = lossAndGrads(
    adapter,
    { x: target, target, n },
    { lambda: 0.1, varFloorRatio: 0.5 },
  );
  assert(varLoss === 0, "healthy output has zero floor loss");
  const collapsedX = new Float32Array(n * d); // all-zero predictions
  const { varLoss: collapsedLoss } = lossAndGrads(
    adapter,
    { x: collapsedX, target, n },
    { lambda: 0.1, varFloorRatio: 0.5 },
  );
  assert(collapsedLoss > 0.01, `collapsed output penalized, got ${collapsedLoss}`);
});

check("SIGReg statistic ≈ 0 on standard normal, large on collapse", () => {
  const rng = mulberry32(17);
  const n = 512;
  const d = 8;
  const normal = new Float32Array(n * d);
  for (let i = 0; i < normal.length; i += 1) normal[i] = gaussian(rng);
  const sketch = buildSketch(42, d, 32);
  const statNormal = sigregStatistic(normal, n, d, sketch);
  assert(statNormal < 2e-3, `normal sample ≈ 0, got ${statNormal}`);
  const collapsed = new Float32Array(n * d);
  for (let i = 0; i < collapsed.length; i += 1) collapsed[i] = 0.05 * gaussian(rng);
  const statCollapsed = sigregStatistic(collapsed, n, d, sketch);
  assert(statCollapsed > 50 * statNormal, `collapse detected: ${statCollapsed} vs ${statNormal}`);
});

check("latent diagnostics expose effective-rank collapse", () => {
  const rng = mulberry32(19);
  const n = 200;
  const d = 8;
  const iso = new Float32Array(n * d);
  for (let i = 0; i < iso.length; i += 1) iso[i] = gaussian(rng);
  const isoDiag = latentDiagnostics(iso, n, d);
  assert(isoDiag.effectiveRank > d * 0.7, `isotropic rank near d, got ${isoDiag.effectiveRank}`);
  const rank1 = new Float32Array(n * d);
  for (let i = 0; i < n; i += 1) {
    const v = gaussian(rng);
    for (let k = 0; k < d; k += 1) rank1[i * d + k] = v;
  }
  const rank1Diag = latentDiagnostics(rank1, n, d);
  assert(rank1Diag.effectiveRank < 1.5, `rank-1 detected, got ${rank1Diag.effectiveRank}`);
});

check("adapter delta is bounded by the clip norm", () => {
  const adapter = createAdapter({ inputDim: 8, hiddenDim: 4, seed: 21 });
  const initial = flattenParams(adapter);
  const rng = mulberry32(23);
  for (const key of ["w1", "b1", "w2", "b2"]) {
    const p = adapter.params[key];
    for (let i = 0; i < p.length; i += 1) p[i] += gaussian(rng);
  }
  const delta = computeAdapterDelta(initial, adapter, { clipNorm: 1.0 });
  assert(delta.unclippedNorm > 1.0, "perturbation exceeds the clip");
  assert(Math.abs(delta.l2Norm - 1.0) < 1e-5, "clipped to the bound");
  assert(delta.clipSaturation === 1, "saturation flagged");
  assert(delta.parameterCount === parameterCount(adapter), "delta covers all params");
  let norm = 0;
  for (const v of delta.delta) norm += v * v;
  assert(Math.abs(Math.sqrt(norm) - 1.0) < 1e-4, "vector actually clipped");
});

check("optimizer steps are real (params move, loss is recomputed)", () => {
  const adapter = createAdapter({ inputDim: 6, hiddenDim: 3, seed: 25 });
  const opt = createOptimizer(adapter, { lr: 1e-2 });
  const rng = mulberry32(27);
  const n = 16;
  const x = Float32Array.from({ length: n * 6 }, () => gaussian(rng));
  const target = Float32Array.from({ length: n * 6 }, () => gaussian(rng) + 1);
  const before = flattenParams(adapter);
  const m1 = trainStep(adapter, opt, { x, target, n });
  const after = flattenParams(adapter);
  let moved = 0;
  for (let i = 0; i < before.length; i += 1) moved += Math.abs(after[i] - before[i]);
  assert(moved > 0, "parameters moved");
  assert(Number.isFinite(m1.gradNorm) && m1.gradNorm > 0, "real gradient norm");
});

// ---------------------------------------------------------------------------
// resident collection over a fake frozen runtime
// ---------------------------------------------------------------------------

const HIDDEN = 6;

function fakeFrozenRuntime() {
  // encoder: deterministic 6-dim summary of the frame; action embed: passthrough of leading dims;
  // predictor: weighted history mix (so frozen predictions correlate with, but differ from, targets)
  return {
    hidden: HIDDEN,
    numFrames: 3,
    imageSize: 224,
    actionDim: 10,
    encodeFrames: async (frames, batch) => {
      const per = frames.length / batch;
      const out = new Float32Array(batch * HIDDEN);
      for (let b = 0; b < batch; b += 1) {
        for (let i = 0; i < per; i += 7) out[b * HIDDEN + (i % HIDDEN)] += frames[b * per + i] / per;
      }
      return out;
    },
    embedActionBlocks: async (blocks, batch, time) => {
      const out = new Float32Array(batch * time * HIDDEN);
      for (let b = 0; b < batch; b += 1) {
        for (let t = 0; t < time; t += 1) {
          for (let k = 0; k < HIDDEN; k += 1) out[(b * time + t) * HIDDEN + k] = blocks[(b * time + t) * 10 + (k % 10)];
        }
      }
      return out;
    },
    predictLatents: async (latents, actEmb, batch, time) => {
      const out = new Float32Array(batch * time * HIDDEN);
      for (let i = 0; i < out.length; i += 1) out[i] = 0.8 * latents[i] + 0.1 * actEmb[i];
      return out;
    },
  };
}

await checkAsync("resident pair collection produces aligned (prediction, next-latent) pairs", async () => {
  const runtime = fakeFrozenRuntime();
  const collected = await collectResidentPairs({ runtime, seed: 5, episodes: 2, maxModelSteps: 8 });
  assert(collected.pairs.count > 0, "pairs collected");
  assert(collected.pairs.x.length === collected.pairs.count * HIDDEN, "x sized");
  assert(collected.pairs.target.length === collected.pairs.count * HIDDEN, "target sized");
  assert(collected.envSteps % 5 === 0, "frameskip-5 env stepping");
});

await checkAsync("the full local continuation reports real metrics and a bounded delta", async () => {
  const runtime = fakeFrozenRuntime();
  const result = await runLocalAdapterContinuation({
    runtime,
    seed: 9,
    episodes: 3,
    maxModelSteps: 8,
    trainSteps: 60,
    batchSize: 16,
    clipNorm: 2.0,
  });
  const m = result.metrics;
  assert(m.lossDecreased, `loss decreased: ${m.predLossFirst} -> ${m.predLossLast}`);
  assert(m.predLossLast < m.predLossFirst, "first/last consistent");
  assert(m.optimizerSteps === 60, "real step count");
  assert(m.deltaL2Norm <= 2.0 + 1e-6, "delta bounded");
  assert(Number.isFinite(m.sigregStatistic) && m.sigregStatistic >= 0, "real SIGReg diagnostic");
  assert(Number.isFinite(m.effectiveRank) && m.effectiveRank > 0, "real effective rank");
  assert(m.parameterCount === result.delta.parameterCount, "bounded subset size consistent");
  // residency: the summary must not contain raw frames/latents/tensors
  const summary = JSON.stringify(result.metrics);
  assert(!summary.includes("pixels") && !("frames" in result), "no raw data in the summary");
});

await checkAsync("the delta artifact carries the binding and only bounded fields", async () => {
  const { buildAdapterDeltaArtifact, LEWM_UPDATE_SCHEMA } = await import("./lewm_delta_artifact.mjs");
  const runtime = fakeFrozenRuntime();
  const result = await runLocalAdapterContinuation({
    runtime,
    seed: 13,
    episodes: 2,
    maxModelSteps: 8,
    trainSteps: 10,
    batchSize: 8,
  });
  const binding = {
    checkpoint: { repoId: "quentinll/lewm-tworooms", revision: "77adaae0bc31deab21c93740d1f8bb947cd0bdec", weightsSha256: "ab".repeat(32) },
    exportGraphHashes: { "lewm_tworooms_encoder.onnx": "11".repeat(32) },
    adapterSpec: [
      { name: "w1", shape: [4, 6] },
      { name: "b1", shape: [4] },
      { name: "w2", shape: [6, 4] },
      { name: "b2", shape: [6] },
    ],
  };
  const artifact = await buildAdapterDeltaArtifact({
    result,
    runId: "run-test1234",
    participantId: "browser-abc123",
    round: 1,
    modelRevisionId: "initial",
    binding,
  });
  assert(artifact.schema === LEWM_UPDATE_SCHEMA, "schema tag");
  assert(artifact.delta.length === result.delta.parameterCount, "full delta");
  assert(/^[0-9a-f]{64}$/.test(artifact.hash), "64-hex hash");
  assert(artifact.baseCheckpoint.revision.length === 40, "pinned revision");
  assert(artifact.metrics.predLossLast < artifact.metrics.predLossFirst, "honest metrics");
  const keys = JSON.stringify(Object.keys(artifact)) + JSON.stringify(Object.keys(artifact.metrics));
  for (const forbidden of ["frames", "pixels", "latents", "tensors", "tokens", "weights", "rollouts"]) {
    assert(!keys.includes(`"${forbidden}"`), `no ${forbidden} key`);
  }
});

// ---------------------------------------------------------------------------
// the autonomous real-mode participant round (#321)
// ---------------------------------------------------------------------------

function fakeRunSnapshot() {
  return {
    id: "run-test1234",
    round: 1,
    runMode: "real-lewm-tworooms",
    currentModelRevisionId: "initial",
    lewmBinding: {
      checkpoint: { repoId: "quentinll/lewm-tworooms", revision: "77adaae0bc31deab21c93740d1f8bb947cd0bdec", weightsSha256: "ab".repeat(32) },
      exportGraphHashes: { "lewm_tworooms_encoder.onnx": "11".repeat(32) },
      adapterHiddenDim: 4,
      adapterSpec: [
        { name: "w1", shape: [4, 6] },
        { name: "b1", shape: [4] },
        { name: "w2", shape: [6, 4] },
        { name: "b2", shape: [6] },
      ],
    },
  };
}

await checkAsync("the autonomous round trains, builds, and submits the bounded delta", async () => {
  const { runRealLewmRound, resetLewmRuntimeCache } = await import("./lewm_participant.mjs");
  resetLewmRuntimeCache();
  const progressCalls = [];
  const submitted = [];
  const client = {
    progress: async (runId, pid, token, value) => progressCalls.push(value),
    submitUpdate: async (runId, pid, token, artifact) => submitted.push({ runId, pid, token, artifact }),
  };
  const run = fakeRunSnapshot();
  const { artifact, metrics } = await runRealLewmRound({
    run,
    me: { id: "browser-abc123" },
    participantToken: "ptok-secret",
    client,
    loadRuntime: async () => fakeFrozenRuntime(),
    seed: 21,
    budget: { episodes: 2, maxModelSteps: 8, trainSteps: 15, batchSize: 8, clipNorm: 3.0 },
  });
  assert(submitted.length === 1, "one bounded submission");
  assert(submitted[0].artifact.schema === "lewm-adapter-delta/1", "adapter schema");
  assert(submitted[0].artifact.baseCheckpoint.revision === run.lewmBinding.checkpoint.revision, "binding carried");
  assert(metrics.optimizerSteps === 15, "real steps");
  assert(progressCalls.length >= 1 && progressCalls.every((v) => v >= 0 && v <= 1), "real progress");
  // w1 4x6 + b1 4 + w2 6x4 + b2 6 = 58 params, sized from the binding's adapterHiddenDim
  assert(artifact.delta.length === 58, `full delta for the 4x6 adapter, got ${artifact.delta.length}`);
});

await checkAsync("a missing runtime fails the round visibly with no fallback submission", async () => {
  const { runRealLewmRound, resetLewmRuntimeCache } = await import("./lewm_participant.mjs");
  resetLewmRuntimeCache();
  const submitted = [];
  const client = {
    progress: async () => {},
    submitUpdate: async (...args) => submitted.push(args),
  };
  let threw = null;
  try {
    await runRealLewmRound({
      run: fakeRunSnapshot(),
      me: { id: "browser-abc123" },
      participantToken: "ptok-secret",
      client,
      loadRuntime: async () => {
        throw new Error("real-lewm runtime unavailable: manifest-missing");
      },
      seed: 3,
    });
  } catch (error) {
    threw = error;
  }
  assert(threw !== null, "round fails");
  assert(String(threw.message).includes("unavailable"), "explicit unsupported reason");
  assert(submitted.length === 0, "nothing submitted on failure");
  resetLewmRuntimeCache();
});

await checkAsync("surrogate runs are refused by the real round driver", async () => {
  const { runRealLewmRound } = await import("./lewm_participant.mjs");
  let threw = false;
  try {
    await runRealLewmRound({
      run: { id: "run-x", runMode: "surrogate-swipe-dot" },
      me: { id: "b" },
      participantToken: "t",
      client: { progress: async () => {}, submitUpdate: async () => {} },
      loadRuntime: async () => fakeFrozenRuntime(),
      seed: 1,
    });
  } catch {
    threw = true;
  }
  assert(threw, "mode mismatch rejected");
});

const report = { total, passed: total - failures.length, failed: failures.length, failures };
console.log(JSON.stringify(report));
if (failures.length > 0) process.exit(1);
