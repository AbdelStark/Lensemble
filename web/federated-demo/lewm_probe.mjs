// Before/after validation probe + health assessment for the real-LeWM mode (#322, epic #314).
//
// The probe is the demo's honest "did federation improve anything that matters" check: a FIXED
// seeded TwoRooms validation task (deterministic expert episodes → teacher-forced windows through
// the frozen exported graphs) on which any adapter state can be scored. Comparing the parent
// revision (identity adapter) against an aggregated global revision on the SAME pairs gives a
// real before/after signal; a negative result is reported as "worse"/"flat", never hidden.
//
// Health assessment encodes the anti-collapse lessons: a low effective-rank ratio or a latent-std
// collapse versus the frozen baseline raises explicit warnings; flat or worsening loss is flagged
// instead of being dressed up. Collapse diagnostics are necessary but never treated as the sole
// success signal — the probe MSE is the binding before/after metric.

import { collectResidentPairs } from "./lewm_local_trainer.mjs";
import {
  adapterForward,
  adapterFromInitAndOffset,
  buildSketch,
  latentDiagnostics,
  sigregStatistic,
} from "./lewm_adapter.mjs";

export const PROBE_VERSION = "lewm-validation-probe/1";
export const DEFAULT_PROBE_SEED = 991;

// Held-out collapse thresholds (the #259 lesson: a "converged" adapter can be magnitude-collapsed
// on held-out while its MSE/identity look fine). We compare the adapted held-out predictions
// against the frozen baseline's: a materially lower latent std or effective rank is collapse, and
// SIGReg drifting markedly worse than baseline flags a distributional shift. These never *create*
// a positive verdict — they can only OVERRIDE a naive "improved" into "collapse-risk".
export const HELDOUT_STD_COLLAPSE_RATIO = 0.7; // adapted std < 70% of baseline std → collapse
export const HELDOUT_RANK_COLLAPSE_RATIO = 0.7; // adapted eff-rank < 70% of baseline → collapse
export const HELDOUT_SIGREG_DRIFT_RATIO = 1.5; // adapted SIGReg > 1.5× baseline (and >0.5) → drift

// Deterministic validation pairs (the same fixture for every revision being compared).
export async function buildValidationSet({ runtime, seed = DEFAULT_PROBE_SEED, episodes = 2, maxModelSteps = 10 }) {
  const collected = await collectResidentPairs({
    runtime,
    seed,
    episodes,
    maxModelSteps,
    policyOptions: { actionNoise: 1.0, actionRepeatProb: 0 },
    minPairs: 4,
  });
  if (collected.pairs.count < 2) {
    throw new Error("validation episodes ended too early to form a probe set");
  }
  return { ...collected, probeVersion: PROBE_VERSION, seed };
}

// Predictions for one adapter state on the validation pairs (null = identity/parent baseline,
// which is exactly the frozen predictor output pairs.x).
export function predictOnPairs(adapter, pairs) {
  if (!adapter) return pairs.x;
  return adapterForward(adapter, pairs.x, pairs.count).y;
}

// Score one adapter state (null = identity/parent baseline) on the validation pairs.
export function scoreAdapterOnPairs(adapter, pairs, hidden) {
  const n = pairs.count;
  const predictions = predictOnPairs(adapter, pairs);
  let mse = 0;
  for (let i = 0; i < n * hidden; i += 1) {
    const diff = predictions[i] - pairs.target[i];
    mse += diff * diff;
  }
  return mse / (n * hidden);
}

// Held-out collapse diagnostics: latent std + effective rank + SIGReg on the VALIDATION
// predictions for the frozen baseline and the adapted revision, with a collapse verdict. This is
// the blind spot #328 closes — the same metrics that #259 needed but only ever ran on training
// tensors. Pure over two prediction tensors so it is directly unit-testable.
export function heldOutCollapseDiagnostics({ baseline, adapted, n, d, sketchSeed = 42 }) {
  const sketch = buildSketch(sketchSeed, d);
  const base = {
    ...latentDiagnostics(baseline, n, d),
    sigregStatistic: sigregStatistic(baseline, n, d, sketch),
  };
  const adapt = {
    ...latentDiagnostics(adapted, n, d),
    sigregStatistic: sigregStatistic(adapted, n, d, sketch),
  };
  const flags = [];
  if (base.latentStdMean > 0 && adapt.latentStdMean < HELDOUT_STD_COLLAPSE_RATIO * base.latentStdMean) {
    flags.push(
      "collapse-risk: adapted held-out latent std materially below the frozen baseline (magnitude collapse)",
    );
  }
  if (base.effectiveRank > 0 && adapt.effectiveRank < HELDOUT_RANK_COLLAPSE_RATIO * base.effectiveRank) {
    flags.push(
      "collapse-risk: adapted held-out effective rank materially below the frozen baseline",
    );
  }
  if (
    base.sigregStatistic > 0 &&
    adapt.sigregStatistic > HELDOUT_SIGREG_DRIFT_RATIO * base.sigregStatistic &&
    adapt.sigregStatistic > 0.5
  ) {
    flags.push(
      "sigreg-drift: adapted held-out marginals drifted markedly further from the isotropic target than baseline",
    );
  }
  return { baseline: base, adapted: adapt, collapseRisk: flags.length > 0, collapseFlags: flags };
}

// Runtime-free before/after probe over an already-built validation set. Given the global adapter
// OFFSET (from the model-revision endpoint) and the validation pairs, score identity vs adapted on
// the SAME pairs and attach held-out collapse diagnostics. Both compareRevisions (browser, builds
// pairs from a frozen runtime) and the system-composed gate (Python builds pairs through ONNX)
// route through this so the verdict logic is single-sourced.
export function probeAdapterOffset({
  validationPairs,
  adaptedState,
  inputDim,
  hiddenDim = 32,
  initSeed = 42,
  improvementThreshold = 0.02, // <2% relative change reads as "flat", not "improved"
  seed = DEFAULT_PROBE_SEED,
}) {
  const d = inputDim;
  const n = validationPairs.count;
  // "before" = the shared init adapter (identity residual: exactly the frozen predictor)
  const baselinePred = predictOnPairs(null, validationPairs);
  const adapter = adapterFromInitAndOffset({ inputDim: d, hiddenDim, initSeed, offset: adaptedState });
  const adaptedPred = predictOnPairs(adapter, validationPairs);
  const mseOf = (pred) => {
    let mse = 0;
    for (let i = 0; i < n * d; i += 1) {
      const diff = pred[i] - validationPairs.target[i];
      mse += diff * diff;
    }
    return mse / (n * d);
  };
  const baselineMse = mseOf(baselinePred);
  const adaptedMse = mseOf(adaptedPred);
  const relativeImprovement = baselineMse > 0 ? (baselineMse - adaptedMse) / baselineMse : 0;
  let verdict = "flat";
  if (relativeImprovement > improvementThreshold) verdict = "improved";
  else if (relativeImprovement < -improvementThreshold) verdict = "worse";
  // Held-out collapse check: an improved MSE under a magnitude/rank collapse is NOT a win.
  const diagnostics = heldOutCollapseDiagnostics({ baseline: baselinePred, adapted: adaptedPred, n, d });
  const displayVerdict = diagnostics.collapseRisk ? "collapse-risk" : verdict;
  return {
    probeVersion: PROBE_VERSION,
    seed,
    pairCount: n,
    baselineMse,
    adaptedMse,
    relativeImprovement,
    verdict,
    displayVerdict,
    collapseRisk: diagnostics.collapseRisk,
    collapseFlags: diagnostics.collapseFlags,
    diagnostics,
    honest: true,
  };
}

// The before/after comparison the dashboard claims rest on.
export async function compareRevisions({
  runtime,
  adaptedState, // global adapter OFFSET from the model-revision endpoint
  adapterHiddenDim = 32,
  adapterInitSeed = 42,
  seed = DEFAULT_PROBE_SEED,
  episodes = 2,
  maxModelSteps = 10,
  improvementThreshold = 0.02,
}) {
  const validation = await buildValidationSet({ runtime, seed, episodes, maxModelSteps });
  return probeAdapterOffset({
    validationPairs: validation.pairs,
    adaptedState,
    inputDim: runtime.hidden,
    hiddenDim: adapterHiddenDim,
    initSeed: adapterInitSeed,
    improvementThreshold,
    seed,
  });
}

// ---------------------------------------------------------------------------
// health assessment over real round metrics (mirrors the server-side flags)
// ---------------------------------------------------------------------------

export function assessRealRoundHealth(metric, { hidden = 192 } = {}) {
  const flags = [];
  const effectiveRank = metric.effectiveRankMean ?? metric.effectiveRank ?? null;
  if (effectiveRank !== null && effectiveRank / hidden < 0.03) {
    flags.push("collapse-warning: effective rank below 3% of the latent width");
  }
  const latentStd = metric.latentStdMeanMean ?? metric.latentStdMean ?? null;
  if (latentStd !== null && latentStd < 0.05) {
    flags.push("collapse-warning: adapted latent std near zero (magnitude collapse)");
  }
  const first = metric.predLossFirstMean ?? metric.predLossFirst ?? null;
  const last = metric.predLossLastMean ?? metric.predLossLast ?? null;
  if (first !== null && last !== null) {
    if (last > first * 1.05) flags.push("loss-worsened: local training increased prediction loss");
    else if (last > first * 0.99) flags.push("flat-loss: no measurable local improvement");
  }
  const sigreg = metric.sigregStatisticMean ?? metric.sigregStatistic ?? null;
  if (sigreg !== null && sigreg > 1.0) {
    flags.push("sigreg-warning: projected marginals far from the isotropic target");
  }
  return { healthy: flags.length === 0, flags };
}
