// Scalar LeWM surprise helpers for the Codex-Paris surprise-meter.
//
// The certified quantity is whole-frame CLS-latent next-step prediction error:
//   surprise = (1 / latentDim) * sum((predicted_next_latent - actual_next_latent)^2)
// There is no spatial heatmap in this export surface.

import { adapterForward, adapterFromInitAndOffset } from "../federated-demo/lewm_adapter.mjs";

export const SURPRISE_ENGINE_VERSION = "lewm-surprise-engine/1";
export const DEFAULT_LATENT_DIM = 192;
export const DEFAULT_WINDOW = 3;
export const DEFAULT_ADAPTER_HIDDEN = 32;
export const DEFAULT_ADAPTER_INIT_SEED = 42;

export function mseOf(pred, target, dim = pred.length) {
  if (pred.length !== target.length) {
    throw new Error(`mse input length mismatch: ${pred.length} != ${target.length}`);
  }
  if (dim <= 0 || pred.length % dim !== 0) {
    throw new Error(`invalid latent dim ${dim} for ${pred.length} values`);
  }
  let total = 0;
  for (let i = 0; i < pred.length; i += 1) {
    const diff = pred[i] - target[i];
    total += diff * diff;
  }
  return total / pred.length;
}

export function concatFloat32(chunks, expectedLength = null) {
  const size = expectedLength ?? chunks.reduce((acc, chunk) => acc + chunk.length, 0);
  const out = new Float32Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  if (offset !== size) throw new Error(`concat length mismatch: ${offset} != ${size}`);
  return out;
}

export async function stepSurprise({
  runtime,
  histLatents,
  histActionEmbeddings,
  nextFrameInput,
  adapter = null,
}) {
  const latentDim = runtime.hidden ?? DEFAULT_LATENT_DIM;
  const window = runtime.numFrames ?? DEFAULT_WINDOW;
  if (histLatents.length !== window) {
    throw new Error(`histLatents must contain ${window} frames`);
  }
  if (histActionEmbeddings.length !== window) {
    throw new Error(`histActionEmbeddings must contain ${window} action embeddings`);
  }
  const latents = concatFloat32(histLatents, window * latentDim);
  const actions = concatFloat32(histActionEmbeddings, window * latentDim);
  const preds = await runtime.predictLatents(latents, actions, 1, window);
  const predNext = Float32Array.from(preds.subarray((window - 1) * latentDim, window * latentDim));
  const adaptedPred = adapter ? adapterForward(adapter, predNext, 1).y : predNext;
  const zNextAll = await runtime.encodeFrames(nextFrameInput, 1);
  const zNext = Float32Array.from(zNextAll.subarray(0, latentDim));
  return {
    schema: SURPRISE_ENGINE_VERSION,
    surprise: mseOf(adaptedPred, zNext, latentDim),
    predNext,
    zNext,
  };
}

export function createOnlineSurpriseEngine({
  runtime,
  adapter = null,
  latentDim = runtime.hidden ?? DEFAULT_LATENT_DIM,
  window = runtime.numFrames ?? DEFAULT_WINDOW,
} = {}) {
  if (!runtime) throw new Error("runtime is required");
  const latents = [];
  const actionEmbeddings = [];
  let modelStep = -1;

  return {
    get warmupSteps() {
      return Math.max(0, window - 1);
    },
    async observeTransition({ currentFrameInput, actionBlock, nextFrameInput }) {
      modelStep += 1;
      const encoded = await runtime.encodeFrames(currentFrameInput, 1);
      const latent = Float32Array.from(encoded.subarray(0, latentDim));
      const embedded = await runtime.embedActionBlocks(actionBlock, 1, 1);
      const actionEmbedding = Float32Array.from(embedded.subarray(0, latentDim));
      latents.push(latent);
      actionEmbeddings.push(actionEmbedding);
      if (latents.length < window) {
        return {
          schema: SURPRISE_ENGINE_VERSION,
          modelStep,
          surprise: null,
          warmup: true,
        };
      }
      const histLatents = latents.slice(latents.length - window, latents.length);
      const histActionEmbeddings = actionEmbeddings.slice(actionEmbeddings.length - window, actionEmbeddings.length);
      const scored = await stepSurprise({
        runtime,
        histLatents,
        histActionEmbeddings,
        nextFrameInput,
        adapter,
      });
      return { ...scored, modelStep, warmup: false };
    },
    async observe({ frameInput, actionBlock, nextFrameInput = frameInput }) {
      return this.observeTransition({
        currentFrameInput: frameInput,
        actionBlock,
        nextFrameInput,
      });
    },
  };
}

export function adapterFromOffset({
  offset,
  inputDim = DEFAULT_LATENT_DIM,
  hiddenDim = DEFAULT_ADAPTER_HIDDEN,
  initSeed = DEFAULT_ADAPTER_INIT_SEED,
}) {
  const expected = hiddenDim * inputDim + hiddenDim + inputDim * hiddenDim + inputDim;
  if (!Array.isArray(offset) && !(offset instanceof Float32Array)) {
    throw new Error("adapter offset must be an array");
  }
  if (offset.length !== expected) {
    throw new Error(`adapter offset length ${offset.length} != ${expected}`);
  }
  if (!Array.from(offset).some((value) => Number(value) !== 0)) {
    throw new Error("adapter offset is all-zero");
  }
  return adapterFromInitAndOffset({
    inputDim,
    hiddenDim,
    initSeed,
    offset: Float32Array.from(offset),
  });
}

export function buildPrePostStreams({
  pairs,
  offset,
  inputDim = DEFAULT_LATENT_DIM,
  hiddenDim = DEFAULT_ADAPTER_HIDDEN,
  initSeed = DEFAULT_ADAPTER_INIT_SEED,
}) {
  if (!pairs || pairs.count <= 0) throw new Error("non-empty validation pairs required");
  const n = pairs.count;
  const total = n * inputDim;
  if (pairs.x.length !== total || pairs.target.length !== total) {
    throw new Error("validation pair tensor length mismatch");
  }
  const adapter = adapterFromOffset({ offset, inputDim, hiddenDim, initSeed });
  const adapted = adapterForward(adapter, Float32Array.from(pairs.x), n).y;
  const pre = [];
  const post = [];
  for (let i = 0; i < n; i += 1) {
    const start = i * inputDim;
    const end = start + inputDim;
    const target = Float32Array.from(pairs.target.slice(start, end));
    pre.push(mseOf(Float32Array.from(pairs.x.slice(start, end)), target, inputDim));
    post.push(mseOf(adapted.subarray(start, end), target, inputDim));
  }
  const mean = (values) => values.reduce((acc, value) => acc + value, 0) / values.length;
  const meanPre = mean(pre);
  const meanPost = mean(post);
  return {
    schema: "lewm-surprise-prepost/1",
    count: n,
    pre,
    post,
    meanPre,
    meanPost,
    surpriseDropRatioLive: meanPre > 0 ? (meanPre - meanPost) / meanPre : 0,
  };
}
