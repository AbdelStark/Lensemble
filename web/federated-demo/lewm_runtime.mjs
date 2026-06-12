// Checkpoint-backed LeWM browser runtime for the Tapestry-like real mode (#318, epic #314).
//
// Loads the hash-bound exported inference graphs (#317: lewm_tworooms_{encoder,action,predictor}
// .onnx + manifest.json), verifies their SHA-256 with WebCrypto, runs them through ONNX Runtime
// Web (WebGPU preferred, WASM fallback), and provides the upstream LeWM inference semantics:
// frame encoding, action-block embedding, windowed autoregressive latent rollout, and CEM-style
// goal-latent planning. There is no silent fallback to the surrogate learner: every failure mode
// surfaces as an explicit `unsupported` state that the UI and evidence record.
//
// Sessions are injectable so node selftests can exercise the rollout/planning wiring with fake
// sessions and no ONNX runtime.

import { ACTION_BLOCK, ACTION_DIM } from "./tworooms_env.mjs";

export const LEWM_MODEL_BASE = "./model/lewm-tworooms/";
export const LEWM_RUNTIME_VERSION = "lewm-ort-web-v1";

const GRAPH_FILES = Object.freeze({
  encoder: "lewm_tworooms_encoder.onnx",
  action: "lewm_tworooms_action.onnx",
  predictor: "lewm_tworooms_predictor.onnx",
});

export class LewmUnsupportedError extends Error {
  constructor(reason, detail) {
    super(`real-lewm runtime unavailable: ${reason}${detail ? ` (${detail})` : ""}`);
    this.reason = reason;
    this.detail = detail ?? null;
  }
}

async function sha256Hex(buffer) {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null; // hash check skipped (recorded in state.integrity)
  const digest = await subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

// ---------------------------------------------------------------------------
// loading
// ---------------------------------------------------------------------------

export async function loadLewmRuntime({
  baseUrl = LEWM_MODEL_BASE,
  fetchFn = globalThis.fetch?.bind(globalThis),
  ortApi = globalThis.ort,
  preferredProviders = ["webgpu", "wasm"],
} = {}) {
  if (!fetchFn) throw new LewmUnsupportedError("no-fetch", "fetch API unavailable");
  if (!ortApi) throw new LewmUnsupportedError("no-ort", "ONNX Runtime Web script not loaded");

  let manifest;
  try {
    const res = await fetchFn(`${baseUrl}manifest.json`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    manifest = await res.json();
  } catch (error) {
    throw new LewmUnsupportedError(
      "manifest-missing",
      `exported graphs not found under ${baseUrl}. Run scripts/lewm_tworooms_export.py (${error.message})`,
    );
  }
  if (manifest.schema !== "lewm-browser-export/1") {
    throw new LewmUnsupportedError("manifest-schema", String(manifest.schema));
  }

  const sessions = {};
  const integrity = {};
  let backend = null;
  for (const [key, file] of Object.entries(GRAPH_FILES)) {
    let buffer;
    try {
      const res = await fetchFn(`${baseUrl}${file}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      buffer = await res.arrayBuffer();
    } catch (error) {
      throw new LewmUnsupportedError("graph-missing", `${file}: ${error.message}`);
    }
    const expected = manifest.files?.[file]?.sha256 ?? null;
    const actual = await sha256Hex(buffer);
    if (expected && actual && expected !== actual) {
      throw new LewmUnsupportedError("hash-mismatch", `${file}: expected ${expected.slice(0, 12)}…, got ${actual.slice(0, 12)}…`);
    }
    integrity[file] = { expected, actual, verified: Boolean(expected && actual && expected === actual) };
    try {
      sessions[key] = await ortApi.InferenceSession.create(buffer, {
        executionProviders: preferredProviders,
      });
      backend = backend ?? (sessions[key].executionProviders?.[0] ?? preferredProviders.join("|"));
    } catch (error) {
      throw new LewmUnsupportedError("session-create-failed", `${file}: ${error.message}`);
    }
  }

  return createLewmRuntime({
    sessions,
    manifest,
    backend,
    integrity,
    tensorFactory: (data, dims) => new ortApi.Tensor("float32", data, dims),
  });
}

// ---------------------------------------------------------------------------
// runtime (sessions injectable for tests)
// ---------------------------------------------------------------------------

export function createLewmRuntime({ sessions, manifest, backend = "test", integrity = {}, tensorFactory }) {
  const hidden = manifest?.architecture?.hiddenDim ?? 192;
  const numFrames = manifest?.architecture?.numFrames ?? 3;
  const imageSize = manifest?.architecture?.imageSize ?? 224;
  const actionDim = manifest?.architecture?.actionDim ?? ACTION_BLOCK * ACTION_DIM;
  const makeTensor = tensorFactory ?? ((data, dims) => ({ data, dims }));

  async function runSession(session, feeds) {
    const outputs = await session.run(feeds);
    const first = Object.values(outputs)[0];
    return first;
  }

  // frames: Float32Array of B*3*H*W in [0,1] -> Float32Array of B*hidden latents
  async function encodeFrames(frames, batch) {
    const out = await runSession(sessions.encoder, {
      pixels: makeTensor(frames, [batch, 3, imageSize, imageSize]),
    });
    return Float32Array.from(out.data);
  }

  // blocks: Float32Array of B*T*actionDim raw env actions -> Float32Array of B*T*hidden
  async function embedActionBlocks(blocks, batch, time) {
    const out = await runSession(sessions.action, {
      actions: makeTensor(blocks, [batch, time, actionDim]),
    });
    return Float32Array.from(out.data);
  }

  // latents/actEmb: Float32Array of B*T*hidden -> predictions Float32Array of B*T*hidden
  async function predictLatents(latents, actEmb, batch, time) {
    const out = await runSession(sessions.predictor, {
      latents: makeTensor(latents, [batch, time, hidden]),
      action_embeddings: makeTensor(actEmb, [batch, time, hidden]),
    });
    return Float32Array.from(out.data);
  }

  // Windowed autoregressive rollout, batched over candidates (upstream LeWM.rollout semantics).
  // history: Array of Float32Array(hidden) shared across the batch (encoded real frames).
  // actEmb: Float32Array of batch*T*hidden embedded action blocks (T >= history length).
  // Returns { frames: Array(batch) of Array(H+steps+1) of Float32Array(hidden) }.
  async function rollout(history, actEmb, batch, time) {
    const h = history.length;
    const steps = time - h;
    const perCandidate = [];
    for (let b = 0; b < batch; b += 1) {
      perCandidate.push(history.map((f) => Float32Array.from(f)));
    }
    for (let step = 0; step <= steps; step += 1) {
      const lo = Math.max(0, h + step - numFrames);
      const t = h + step - lo;
      const latents = new Float32Array(batch * t * hidden);
      const acts = new Float32Array(batch * t * hidden);
      for (let b = 0; b < batch; b += 1) {
        for (let i = 0; i < t; i += 1) {
          latents.set(perCandidate[b][lo + i], (b * t + i) * hidden);
          acts.set(actEmb.subarray((b * time + lo + i) * hidden, (b * time + lo + i + 1) * hidden), (b * t + i) * hidden);
        }
      }
      const preds = await predictLatents(latents, acts, batch, t);
      for (let b = 0; b < batch; b += 1) {
        const last = preds.subarray((b * t + t - 1) * hidden, (b * t + t) * hidden);
        perCandidate[b].push(Float32Array.from(last));
      }
    }
    return perCandidate;
  }

  function latentDistance(a, b) {
    let total = 0;
    for (let i = 0; i < a.length; i += 1) {
      const d = a[i] - b[i];
      total += d * d;
    }
    return total;
  }

  // CEM-style planning over action blocks (upstream: sample candidates, roll out, score by
  // terminal latent distance to the goal, refit on elites, return the best first block).
  async function planAction({
    historyLatents, // Array of Float32Array(hidden), most recent last
    historyActionBlocks = null, // Array of (h-1) Float32Array(actionDim) that produced the history
    goalLatent, // Float32Array(hidden)
    rng,
    horizon = 2,
    samples = 32,
    eliteCount = 6,
    iterations = 2,
    initialStd = 1.0,
  }) {
    const blockDim = actionDim;
    let mean = new Float32Array(horizon * blockDim);
    let std = new Float32Array(horizon * blockDim).fill(initialStd);
    let best = null;

    for (let iter = 0; iter < iterations; iter += 1) {
      const candidates = [];
      for (let s = 0; s < samples; s += 1) {
        const seq = new Float32Array(horizon * blockDim);
        for (let i = 0; i < seq.length; i += 1) {
          // Box-Muller normal sample around the CEM mean
          const u1 = Math.max(rng(), 1e-12);
          const u2 = rng();
          const n = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
          seq[i] = Math.min(1, Math.max(-1, mean[i] + n * std[i]));
        }
        candidates.push(seq);
      }

      const h = historyLatents.length;
      const time = h + horizon - 1; // action blocks consumed by the rollout window
      const batch = candidates.length;
      const blocks = new Float32Array(batch * time * blockDim);
      for (let b = 0; b < batch; b += 1) {
        for (let t = 0; t < time; t += 1) {
          // positions before h-1 carry the real action blocks that produced the history frames
          // (zeros only when the episode just started and no history actions exist yet)
          const fromCandidate = t - (h - 1);
          if (fromCandidate >= 0) {
            blocks.set(
              candidates[b].subarray(fromCandidate * blockDim, (fromCandidate + 1) * blockDim),
              (b * time + t) * blockDim,
            );
          } else if (historyActionBlocks?.[t]) {
            blocks.set(historyActionBlocks[t], (b * time + t) * blockDim);
          }
        }
      }
      const actEmb = await embedActionBlocks(blocks, batch, time);
      const rollouts = await rollout(historyLatents, actEmb, batch, time);
      const scored = rollouts.map((frames, idx) => ({
        idx,
        cost: latentDistance(frames[frames.length - 1], goalLatent),
        terminal: frames[frames.length - 1],
      }));
      scored.sort((a, b) => a.cost - b.cost);
      const elites = scored.slice(0, Math.min(eliteCount, scored.length));
      best = { sequence: candidates[elites[0].idx], cost: elites[0].cost, iteration: iter, costs: scored.map((s) => s.cost) };

      // refit mean/std on elites
      for (let i = 0; i < mean.length; i += 1) {
        let m = 0;
        for (const e of elites) m += candidates[e.idx][i];
        m /= elites.length;
        let v = 0;
        for (const e of elites) v += (candidates[e.idx][i] - m) ** 2;
        mean[i] = m;
        std[i] = Math.sqrt(v / elites.length) + 0.05;
      }
    }

    return {
      actionBlock: best.sequence.subarray(0, blockDim),
      fullSequence: best.sequence,
      cost: best.cost,
      candidateCosts: best.costs,
      horizon,
      samples,
      iterations,
    };
  }

  return {
    runtime: LEWM_RUNTIME_VERSION,
    backend,
    manifest,
    integrity,
    hidden,
    numFrames,
    imageSize,
    actionDim,
    encodeFrames,
    embedActionBlocks,
    predictLatents,
    rollout,
    planAction,
    identity: {
      checkpointRepo: manifest?.checkpoint?.repoId ?? "unknown",
      checkpointRevision: manifest?.checkpoint?.revision ?? "unknown",
      weightsSha256: manifest?.checkpoint?.weightsSha256 ?? "unknown",
      graphVersion: manifest?.graphVersion ?? null,
      opset: manifest?.opset ?? null,
      backend,
    },
  };
}
