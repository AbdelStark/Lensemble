// Browser-local Tapestry-like LeWM continuation loop (#319, epic #314).
//
// Generates TwoRooms rollouts locally (noisy expert policy), keeps frames/actions/latents
// resident, runs the FROZEN exported graphs to produce teacher-forced next-latent predictions,
// trains the bounded residual adapter (lewm_adapter.mjs) on (frozen-prediction → true-next-latent)
// pairs, and returns metrics plus the bounded clipped delta for federation. Real optimizer steps,
// no artificial sleeps, no fabricated dashboard values: every metric is computed from the actual
// tensors of this loop. Raw frames, actions, latents, and adapter tensors never leave the browser.

import {
  createExpertPolicy,
  frameToModelInput,
  packActionBlock,
  renderFrameRGB,
  sampleEpisode,
  stepEpisode,
} from "./tworooms_env.mjs";
import {
  computeAdapterDelta,
  createAdapter,
  flattenParams,
  latentDiagnostics,
  parameterCount,
  trainAdapterOnPairs,
} from "./lewm_adapter.mjs";
import { mulberry32 } from "./rng.mjs";

export const LOCAL_TRAINER_RUNTIME = "lewm-local-continuation-v1";

// Collect resident training pairs from local rollouts through the frozen graphs.
// Returns { pairs: {x, target, count}, episodes, envSteps, encodeMs } — everything stays local.
export async function collectResidentPairs({
  runtime,
  seed = 1,
  episodes = 2,
  maxModelSteps = 12,
  policyOptions = {},
  // TwoRooms episodes terminate when the agent reaches the goal, so a run of short
  // episodes can yield too few resident pairs. Keep sampling extra (deterministic,
  // rng-seeded) episodes until at least `minPairs` are collected, bounded by
  // `maxEpisodes` so a pathological seed can never run unboundedly.
  minPairs = 0,
  maxEpisodes = null,
  now = () => (typeof performance !== "undefined" ? performance.now() : Date.now()),
} = {}) {
  const rng = mulberry32(seed >>> 0);
  const hidden = runtime.hidden;
  const window = runtime.numFrames;
  const xs = [];
  const targets = [];
  let envSteps = 0;
  let episodesRun = 0;
  const started = now();
  const episodeCap = maxEpisodes ?? (minPairs > 0 ? episodes + 6 : episodes);

  while (episodesRun < episodes || (xs.length < minPairs && episodesRun < episodeCap)) {
    const expert = createExpertPolicy(policyOptions);
    let episode = sampleEpisode(rng);
    // roll the episode at frameskip granularity, recording one frame per model step
    const frames = [renderFrameRGB(episode.agent)];
    const blocks = [];
    for (let step = 0; step < maxModelSteps && !episode.done; step += 1) {
      const subActions = [];
      for (let i = 0; i < 5; i += 1) {
        const action = expert(episode, rng);
        subActions.push(action);
        episode = stepEpisode(episode, action);
      }
      envSteps += 5;
      blocks.push(packActionBlock(subActions));
      frames.push(renderFrameRGB(episode.agent));
    }
    episodesRun += 1;
    if (blocks.length < window) continue; // too short to form a full history window

    // encode all frames in one batch through the frozen encoder
    const batch = frames.length;
    const pixelBatch = new Float32Array(batch * 3 * runtime.imageSize * runtime.imageSize);
    for (let i = 0; i < batch; i += 1) pixelBatch.set(frameToModelInput(frames[i]), i * 3 * runtime.imageSize * runtime.imageSize);
    const latents = await runtime.encodeFrames(pixelBatch, batch);

    // embed all action blocks in one call
    const time = blocks.length;
    const blockData = new Float32Array(time * runtime.actionDim);
    for (let i = 0; i < time; i += 1) blockData.set(blocks[i], i * runtime.actionDim);
    const actEmb = await runtime.embedActionBlocks(blockData, 1, time);

    // teacher-forced frozen predictions for every full window: history [t-2, t-1, t] -> target t+1
    for (let t = window - 1; t < time; t += 1) {
      const histLatents = new Float32Array(window * hidden);
      const histActs = new Float32Array(window * hidden);
      for (let i = 0; i < window; i += 1) {
        const frameIdx = t - (window - 1) + i;
        histLatents.set(latents.subarray(frameIdx * hidden, (frameIdx + 1) * hidden), i * hidden);
        histActs.set(actEmb.subarray(frameIdx * hidden, (frameIdx + 1) * hidden), i * hidden);
      }
      const preds = await runtime.predictLatents(histLatents, histActs, 1, window);
      xs.push(Float32Array.from(preds.subarray((window - 1) * hidden, window * hidden)));
      targets.push(Float32Array.from(latents.subarray((t + 1) * hidden, (t + 2) * hidden)));
    }
  }

  const count = xs.length;
  const x = new Float32Array(count * hidden);
  const target = new Float32Array(count * hidden);
  for (let i = 0; i < count; i += 1) {
    x.set(xs[i], i * hidden);
    target.set(targets[i], i * hidden);
  }
  return {
    pairs: { x, target, count },
    episodes: episodesRun,
    envSteps,
    encodeMs: now() - started,
  };
}

// Comfortable margin above the 4-pair training floor so a bad seed never starves a round.
export const MIN_RESIDENT_PAIRS = 8;

// The full local continuation: collect -> train -> bounded delta + honest metric summary.
export async function runLocalAdapterContinuation({
  runtime,
  seed = 1,
  episodes = 2,
  maxModelSteps = 12,
  trainSteps = 40,
  batchSize = 24,
  lambda = 0.1,
  clipNorm = 3.0,
  adapterHidden = 32,
  initialAdapter = null, // continuation start (shared init + global offset); fresh init when null
  onProgress = () => {},
} = {}) {
  onProgress(0.05, { phase: "rollout-collection" });
  const collected = await collectResidentPairs({
    runtime,
    seed,
    episodes,
    maxModelSteps,
    minPairs: MIN_RESIDENT_PAIRS,
  });
  if (collected.pairs.count < 4) {
    // Unreachable in practice now (the collector resamples up to its episode cap);
    // kept as a last-resort guard so training never runs on a starved batch.
    throw new Error(
      `insufficient resident pairs (${collected.pairs.count}) after ${collected.episodes} episodes`,
    );
  }
  onProgress(0.35, { phase: "adapter-training", pairs: collected.pairs.count });

  const adapter =
    initialAdapter ?? createAdapter({ inputDim: runtime.hidden, hiddenDim: adapterHidden, seed });
  const initialFlat = flattenParams(adapter);
  const baseline = latentDiagnostics(collected.pairs.x, collected.pairs.count, runtime.hidden);
  const report = trainAdapterOnPairs(adapter, collected.pairs, {
    steps: trainSteps,
    batchSize,
    seed,
    lambda,
  });
  onProgress(0.9, { phase: "delta-computation" });
  const delta = computeAdapterDelta(initialFlat, adapter, { clipNorm });

  const lastDiag = report.diagnostics[report.diagnostics.length - 1];
  return {
    runtime: LOCAL_TRAINER_RUNTIME,
    adapter,
    delta,
    metrics: {
      pairCount: collected.pairs.count,
      episodes: collected.episodes,
      envSteps: collected.envSteps,
      optimizerSteps: report.steps,
      batchSize: report.batchSize,
      predLossFirst: report.firstLoss,
      predLossLast: report.lastLoss,
      lossDecreased: report.lastLoss < report.firstLoss,
      varLossLast: report.history[report.history.length - 1].varLoss,
      gradClipEvents: report.history.filter((h) => h.clipped).length,
      latentStdMean: lastDiag.latentStdMean,
      effectiveRank: lastDiag.effectiveRank,
      effectiveRankRatio: lastDiag.effectiveRankRatio,
      sigregStatistic: lastDiag.sigregStatistic,
      baselineLatentStdMean: baseline.latentStdMean,
      baselineEffectiveRank: baseline.effectiveRank,
      collectMs: collected.encodeMs,
      trainMs: report.runtimeMs,
      deltaL2Norm: delta.l2Norm,
      deltaUnclippedNorm: delta.unclippedNorm,
      deltaClipSaturation: delta.clipSaturation,
      parameterCount: parameterCount(adapter),
    },
    lossHistory: report.history.map((h) => ({ step: h.step, predLoss: h.predLoss, totalLoss: h.totalLoss })),
    diagnostics: report.diagnostics,
  };
}
