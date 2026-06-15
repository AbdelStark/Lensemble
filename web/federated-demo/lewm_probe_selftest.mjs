// Node-runnable selftest for the before/after probe and health assessment (#322).
//
// Run: node web/federated-demo/lewm_probe_selftest.mjs
// Deterministic, ONNX-free: the probe runs over a fake frozen runtime with a KNOWN systematic
// predictor bias, so a bias-correcting offset must read "improved", a harmful offset must read
// "worse", and a zero offset must read "flat". Health flags trigger on collapse/flat-loss
// patterns and stay quiet on healthy metrics. Driven by tests/ml/test_lewm_probe.py.

import {
  DEFAULT_PROBE_SEED,
  assessRealRoundHealth,
  buildValidationSet,
  compareRevisions,
  heldOutCollapseDiagnostics,
  scoreAdapterOnPairs,
} from "./lewm_probe.mjs";
import { mulberry32 } from "./rng.mjs";
import {
  adapterFromInitAndOffset,
  flattenParams,
} from "./lewm_adapter.mjs";

const failures = [];
let total = 0;

async function check(name, fn) {
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

const HIDDEN = 6;
const BIAS = 0.8; // the fake frozen predictor systematically under-predicts dim 0 by BIAS

function fakeRuntime() {
  return {
    hidden: HIDDEN,
    numFrames: 3,
    imageSize: 224,
    actionDim: 10,
    encodeFrames: async (frames, batch) => {
      const per = frames.length / batch;
      const out = new Float32Array(batch * HIDDEN);
      for (let b = 0; b < batch; b += 1) {
        for (let i = 0; i < per; i += 11) out[b * HIDDEN + (i % HIDDEN)] += frames[b * per + i] / per;
      }
      return out;
    },
    embedActionBlocks: async (blocks, batch, time) => {
      const out = new Float32Array(batch * time * HIDDEN);
      for (let i = 0; i < out.length; i += 1) out[i] = blocks[(i * 10) % blocks.length] ?? 0;
      return out;
    },
    // prediction = encoder latent shifted by -BIAS on dim 0: the "true next" recorded by
    // collectResidentPairs is the encoder latent of the next frame, so the residual carries a
    // systematic +BIAS on dim 0 that a bias-correcting adapter can fix
    predictLatents: async (latents, actEmb, batch, time) => {
      const out = new Float32Array(batch * time * HIDDEN);
      for (let i = 0; i < out.length; i += 1) {
        out[i] = latents[i] - (i % HIDDEN === 0 ? BIAS : 0);
      }
      return out;
    },
  };
}

function offsetWithBias(value) {
  // build the offset that sets ONLY b2[0] = value relative to the shared init (init b2 is zero)
  const init = adapterFromInitAndOffset({ inputDim: HIDDEN, hiddenDim: 4, initSeed: 42 });
  const flat = flattenParams(init);
  const offset = new Float32Array(flat.length);
  const b2Start = 4 * HIDDEN + 4 + HIDDEN * 4; // w1 + b1 + w2
  offset[b2Start] = value;
  return offset;
}

await check("the validation set is deterministic for a fixed seed", async () => {
  const a = await buildValidationSet({ runtime: fakeRuntime(), seed: DEFAULT_PROBE_SEED });
  const b = await buildValidationSet({ runtime: fakeRuntime(), seed: DEFAULT_PROBE_SEED });
  assert(a.pairs.count === b.pairs.count, "same pair count");
  for (let i = 0; i < a.pairs.x.length; i += 1) {
    assert(a.pairs.x[i] === b.pairs.x[i], "identical fixture pairs");
  }
});

await check("a bias-correcting offset reads 'improved'", async () => {
  const report = await compareRevisions({
    runtime: fakeRuntime(),
    adaptedState: offsetWithBias(BIAS), // exactly cancels the systematic residual
    adapterHiddenDim: 4,
    adapterInitSeed: 42,
  });
  assert(report.verdict === "improved", `improved, got ${report.verdict} (${report.relativeImprovement})`);
  assert(report.adaptedMse < report.baselineMse, "mse actually dropped");
  // a pure bias-correction (constant shift on one dim) is NOT a collapse: held-out std/rank hold
  assert(report.collapseRisk === false, "bias correction must not read as collapse");
  assert(report.displayVerdict === "improved", "honest verdict tracks the clean improvement");
  assert(report.diagnostics?.adapted?.latentStdMean > 0, "held-out diagnostics are populated");
});

await check("a harmful offset reads 'worse' — negative results are reported", async () => {
  const report = await compareRevisions({
    runtime: fakeRuntime(),
    adaptedState: offsetWithBias(-2 * BIAS),
    adapterHiddenDim: 4,
    adapterInitSeed: 42,
  });
  assert(report.verdict === "worse", `worse, got ${report.verdict}`);
});

await check("a zero offset reads 'flat' (identity == frozen predictor)", async () => {
  const report = await compareRevisions({
    runtime: fakeRuntime(),
    adaptedState: new Float32Array(offsetWithBias(0).length),
    adapterHiddenDim: 4,
    adapterInitSeed: 42,
  });
  assert(report.verdict === "flat", `flat, got ${report.verdict}`);
  assert(Math.abs(report.adaptedMse - report.baselineMse) < 1e-9, "identity scores identically");
});

await check("the identity adapter scores exactly like the raw predictions", async () => {
  const validation = await buildValidationSet({ runtime: fakeRuntime() });
  const raw = scoreAdapterOnPairs(null, validation.pairs, HIDDEN);
  const identity = adapterFromInitAndOffset({ inputDim: HIDDEN, hiddenDim: 4, initSeed: 42 });
  const viaInit = scoreAdapterOnPairs(identity, validation.pairs, HIDDEN);
  assert(Math.abs(raw - viaInit) < 1e-9, "shared init is the identity residual");
});

await check("health flags trigger on collapse and flat loss, stay quiet when healthy", async () => {
  const healthy = assessRealRoundHealth({
    effectiveRankMean: 11,
    latentStdMeanMean: 0.8,
    predLossFirstMean: 0.06,
    predLossLastMean: 0.02,
    sigregStatisticMean: 0.05,
  });
  assert(healthy.healthy && healthy.flags.length === 0, "healthy round has no flags");

  const collapsed = assessRealRoundHealth({
    effectiveRankMean: 2,
    latentStdMeanMean: 0.01,
    predLossFirstMean: 0.06,
    predLossLastMean: 0.06,
    sigregStatisticMean: 2.5,
  });
  assert(!collapsed.healthy, "collapsed round is unhealthy");
  assert(collapsed.flags.some((f) => f.includes("effective rank")), "rank flag");
  assert(collapsed.flags.some((f) => f.includes("magnitude collapse")), "std flag");
  assert(collapsed.flags.some((f) => f.includes("flat-loss")), "flat loss flag");
  assert(collapsed.flags.some((f) => f.includes("sigreg")), "sigreg flag");

  const worsened = assessRealRoundHealth({
    predLossFirstMean: 0.02,
    predLossLastMean: 0.05,
  });
  assert(worsened.flags.some((f) => f.includes("loss-worsened")), "worsening flagged");
});

await check("a magnitude-collapsed adapter is flagged even when its held-out MSE improves", async () => {
  // The #259 trap: an adapter that predicts the per-dim mean of the targets collapses all latent
  // variance (std → 0) yet can score a LOWER MSE than a noisy frozen baseline. MSE alone reads
  // "improved"; the held-out collapse diagnostics must override that to "collapse-risk".
  const d = 4;
  const n = 40;
  const rng = mulberry32(7);
  const targets = new Float32Array(n * d);
  const baseline = new Float32Array(n * d); // frozen predictor: target + heavy noise (high MSE, healthy std)
  const mean = new Float64Array(d);
  for (let i = 0; i < n; i += 1) {
    for (let k = 0; k < d; k += 1) {
      const t = (rng() - 0.5) * 2; // moderate target variance
      targets[i * d + k] = t;
      baseline[i * d + k] = t + (rng() - 0.5) * 6; // large noise dominates → big baseline MSE
      mean[k] += t;
    }
  }
  for (let k = 0; k < d; k += 1) mean[k] /= n;
  const adapted = new Float32Array(n * d); // collapsed: constant per-dim mean (zero variance)
  for (let i = 0; i < n; i += 1) {
    for (let k = 0; k < d; k += 1) adapted[i * d + k] = mean[k];
  }
  const mse = (p) => {
    let s = 0;
    for (let i = 0; i < n * d; i += 1) s += (p[i] - targets[i]) ** 2;
    return s / (n * d);
  };
  assert(mse(adapted) < mse(baseline), "collapsed mean-predictor must beat the noisy baseline on MSE");
  const diag = heldOutCollapseDiagnostics({ baseline, adapted, n, d });
  assert(diag.collapseRisk === true, "magnitude collapse must be flagged");
  assert(
    diag.collapseFlags.some((f) => f.includes("magnitude collapse")),
    "the std-collapse flag is raised",
  );
  assert(diag.adapted.latentStdMean < diag.baseline.latentStdMean, "adapted std is below baseline");
});

const report = { total, passed: total - failures.length, failed: failures.length, failures };
console.log(JSON.stringify(report));
if (failures.length > 0) process.exit(1);
