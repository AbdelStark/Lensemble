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
import { adapterForward, adapterFromInitAndOffset } from "./lewm_adapter.mjs";

export const PROBE_VERSION = "lewm-validation-probe/1";
export const DEFAULT_PROBE_SEED = 991;

// Deterministic validation pairs (the same fixture for every revision being compared).
export async function buildValidationSet({ runtime, seed = DEFAULT_PROBE_SEED, episodes = 2, maxModelSteps = 10 }) {
  const collected = await collectResidentPairs({
    runtime,
    seed,
    episodes,
    maxModelSteps,
    policyOptions: { actionNoise: 1.0, actionRepeatProb: 0 },
  });
  if (collected.pairs.count < 2) {
    throw new Error("validation episodes ended too early to form a probe set");
  }
  return { ...collected, probeVersion: PROBE_VERSION, seed };
}

// Score one adapter state (null = identity/parent baseline) on the validation pairs.
export function scoreAdapterOnPairs(adapter, pairs, hidden) {
  const n = pairs.count;
  let predictions = pairs.x;
  if (adapter) {
    predictions = adapterForward(adapter, pairs.x, n).y;
  }
  let mse = 0;
  for (let i = 0; i < n * hidden; i += 1) {
    const diff = predictions[i] - pairs.target[i];
    mse += diff * diff;
  }
  return mse / (n * hidden);
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
  improvementThreshold = 0.02, // <2% relative change reads as "flat", not "improved"
}) {
  const validation = await buildValidationSet({ runtime, seed, episodes, maxModelSteps });
  // "before" = the shared init adapter (identity residual: exactly the frozen predictor)
  const baselineMse = scoreAdapterOnPairs(null, validation.pairs, runtime.hidden);
  const adapter = adapterFromInitAndOffset({
    inputDim: runtime.hidden,
    hiddenDim: adapterHiddenDim,
    initSeed: adapterInitSeed,
    offset: adaptedState,
  });
  const adaptedMse = scoreAdapterOnPairs(adapter, validation.pairs, runtime.hidden);
  const relativeImprovement = baselineMse > 0 ? (baselineMse - adaptedMse) / baselineMse : 0;
  let verdict = "flat";
  if (relativeImprovement > improvementThreshold) verdict = "improved";
  else if (relativeImprovement < -improvementThreshold) verdict = "worse";
  return {
    probeVersion: PROBE_VERSION,
    seed,
    pairCount: validation.pairs.count,
    baselineMse,
    adaptedMse,
    relativeImprovement,
    verdict,
    honest: true,
  };
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
