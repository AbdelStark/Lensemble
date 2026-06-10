// Browser-local surrogate learner contract (#298).
//
// The first implementation path is deliberately a Web Worker JavaScript
// surrogate over resident synthetic swipe-dot samples. It does not train a
// production model. It produces only versioned update metadata that the backend
// can aggregate: shape, sample count, hash, norm, and runtime labels. No raw
// observations/actions/state labels leave the browser.

import { stepSwipeDot } from "../dynamic-env-demo/swipe_dot_core.mjs";
import { mulberry32 } from "./rng.mjs";

export const UPDATE_SCHEMA = "browser-update/1";
export const LEARNER_RUNTIME = "js-worker-surrogate-v1";

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
  seed = 1,
  sampleCount = 16,
}) {
  const rng = mulberry32((seed + round * 1009 + participantId.length * 9173) >>> 0);
  const accumulator = [0, 0, 0, 0];
  let state = { x: 0.2 + 0.6 * rng(), y: 0.2 + 0.6 * rng() };
  for (let i = 0; i < sampleCount; i += 1) {
    const action = [rng() * 2 - 1, rng() * 2 - 1];
    const next = stepSwipeDot(state, action);
    accumulator[0] += action[0];
    accumulator[1] += action[1];
    accumulator[2] += next.x - state.x;
    accumulator[3] += next.y - state.y;
    state = next;
  }
  const vector = accumulator.map((value) => Number((value / sampleCount).toFixed(8)));
  const l2Norm = Math.sqrt(vector.reduce((total, value) => total + value * value, 0));
  const hash = hashString(JSON.stringify({ runId, participantId, round, vector, sampleCount }));
  return {
    schema: UPDATE_SCHEMA,
    source: "browser-local-surrogate",
    runtime: LEARNER_RUNTIME,
    runId,
    participantId,
    round,
    shape: [vector.length],
    sampleCount,
    hash,
    l2Norm: Number(l2Norm.toFixed(8)),
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
