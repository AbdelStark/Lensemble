// Browser-local tiny learner contract (#298/#307).
//
// The first implementation path is deliberately a Web Worker JavaScript
// learner over resident synthetic swipe-dot samples. It does not train a
// production model. It produces only a versioned, clipped, derived update
// vector plus metadata that the backend can aggregate. No raw observations,
// actions, state labels, latents, tensors, participant tokens, or model weights
// leave the browser.

import { stepSwipeDot } from "../dynamic-env-demo/swipe_dot_core.mjs";
import { mulberry32 } from "./rng.mjs";

export const UPDATE_SCHEMA = "browser-update/1";
export const LEARNER_RUNTIME = "js-worker-tiny-jepa-v1";
export const DEFAULT_CLIP_NORM = 1.0;

function l2Norm(vector) {
  return Math.sqrt(vector.reduce((total, value) => total + value * value, 0));
}

function clipVector(vector, clipNorm = DEFAULT_CLIP_NORM) {
  const norm = l2Norm(vector);
  if (norm <= clipNorm || norm === 0) return vector;
  const scale = clipNorm / norm;
  return vector.map((value) => value * scale);
}

function hashString(text) {
  let h0 = 0x811c9dc5;
  let h1 = 0x9e3779b9;
  for (let i = 0; i < text.length; i += 1) {
    const code = text.charCodeAt(i);
    h0 = Math.imul(h0 ^ code, 0x01000193) >>> 0;
    h1 = Math.imul(h1 + code + (h0 >>> 3), 0x85ebca6b) >>> 0;
  }
  let out = "";
  let a = h0;
  let b = h1;
  for (let i = 0; i < 8; i += 1) {
    a = Math.imul(a ^ (b >>> 11), 0x27d4eb2d) >>> 0;
    b = Math.imul(b ^ (a >>> 13), 0x165667b1) >>> 0;
    out += (a >>> 0).toString(16).padStart(8, "0");
  }
  return out.slice(0, 64);
}

export function computeSurrogateUpdate({
  runId,
  participantId,
  round,
  roundId = `${runId}:round-${round}`,
  modelRevisionId = "initial",
  seed = 1,
  sampleCount = 16,
  localSteps = 8,
  clipNorm = DEFAULT_CLIP_NORM,
}) {
  const started = typeof performance !== "undefined" ? performance.now() : Date.now();
  const rng = mulberry32((seed + round * 1009 + participantId.length * 9173) >>> 0);
  const accumulator = [0, 0, 0, 0];
  let predictionError = 0;
  let state = { x: 0.2 + 0.6 * rng(), y: 0.2 + 0.6 * rng() };
  for (let i = 0; i < sampleCount; i += 1) {
    const action = [rng() * 2 - 1, rng() * 2 - 1];
    const next = stepSwipeDot(state, action);
    accumulator[0] += action[0];
    accumulator[1] += action[1];
    accumulator[2] += next.x - state.x;
    accumulator[3] += next.y - state.y;
    predictionError += Math.abs(next.x - state.x - action[0] * 0.08);
    predictionError += Math.abs(next.y - state.y - action[1] * 0.08);
    state = next;
  }
  const unclipped = accumulator.map((value) => value / sampleCount);
  const vector = clipVector(unclipped, clipNorm).map((value) => Number(value.toFixed(8)));
  const norm = l2Norm(vector);
  const loss = predictionError / Math.max(1, sampleCount * 2);
  const probe = Math.max(0, 1 - loss);
  const runtimeMs = Math.max(1, (typeof performance !== "undefined" ? performance.now() : Date.now()) - started);
  const hash = hashString(
    JSON.stringify({ runId, participantId, round, roundId, modelRevisionId, vector, sampleCount, localSteps }),
  );
  return {
    schema: UPDATE_SCHEMA,
    source: "browser-local-surrogate",
    runtime: LEARNER_RUNTIME,
    runId,
    participantId,
    round,
    roundId,
    modelRevisionId,
    shape: [vector.length],
    parameterCount: vector.length,
    vector,
    sampleCount,
    localSteps,
    hash,
    l2Norm: Number(norm.toFixed(8)),
    clipNorm,
    loss: Number(loss.toFixed(8)),
    probe: Number(probe.toFixed(8)),
    runtimeMs: Number(runtimeMs.toFixed(1)),
    seed,
    simulated: false,
  };
}

export function simulatorUpdateArtifact({ runId, participantId, round, seed = 1 }) {
  const artifact = computeSurrogateUpdate({ runId, participantId, round, seed, sampleCount: 8 });
  return {
    ...artifact,
    source: "simulator",
    runtime: "backend-simulator",
    simulated: true,
  };
}

export function runBrowserLearner(task, onProgress = () => {}) {
  if (typeof Worker !== "undefined") {
    return new Promise((resolve, reject) => {
      const worker = new Worker(new URL("./learner_worker.mjs", import.meta.url), { type: "module" });
      worker.onmessage = (event) => {
        const msg = event.data;
        if (msg.type === "progress") onProgress(msg.progress);
        if (msg.type === "result") {
          worker.terminate();
          resolve(msg.artifact);
        }
        if (msg.type === "error") {
          worker.terminate();
          reject(new Error(msg.message));
        }
      };
      worker.onerror = (event) => {
        worker.terminate();
        reject(new Error(event.message || "learner worker failed"));
      };
      worker.postMessage(task);
    });
  }
  return new Promise((resolve) => {
    onProgress(0.25);
    setTimeout(() => {
      onProgress(0.75);
      const artifact = computeSurrogateUpdate(task);
      onProgress(1);
      resolve(artifact);
    }, 40);
  });
}
