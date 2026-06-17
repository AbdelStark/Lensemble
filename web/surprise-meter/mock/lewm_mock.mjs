// ─────────────────────────────────────────────────────────────────────────
// MOCK stand-ins for the real LeWM browser modules, shaped to the verified
// signatures (surprise-meter/01-architecture.md, corrections C1–C6) so the
// engine composes them exactly as it will compose the real ones tomorrow.
//
//   SWAP FOR REAL (one line each in engine.mjs):
//     mockRuntime  →  loadLewmRuntime(...)        from ../../federated-demo/lewm_runtime.mjs
//     mockEnv      →  the tworooms_env exports     from ../../federated-demo/tworooms_env.mjs
//     mockProbe    →  probeAdapterOffset math      from ../../federated-demo/lewm_probe.mjs
//     mockAdapter  →  adapterFromInitAndOffset/... from ../../federated-demo/lewm_adapter.mjs
//
// Only latent *generation* is mocked. The prediction→actual→MSE pipeline and
// every public signature match real, so the real swap touches imports, not flow.
// ─────────────────────────────────────────────────────────────────────────
import { MODEL, ADAPTER } from "./fixtures.mjs";

const D = MODEL.latentDim; // 192

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
// unit-variance noise so MSE/D ≈ amplitude² (sum of 4 uniforms has var 1/3 → ×√3)
function noiseVec(seed) {
  const r = mulberry32(seed), v = new Float32Array(D);
  for (let i = 0; i < D; i++) v[i] = (r() + r() + r() + r() - 2) * 1.732;
  return v;
}

// ── runtime (C1): real is async + flat Float32Array; encodeFrames / predictLatents
export const mockRuntime = {
  hidden: MODEL.latentDim, numFrames: MODEL.numFrames,
  imageSize: MODEL.imageSize, actionDim: MODEL.actionDim,
  // REAL: predictLatents(latents, actEmb, batch, time) → last step = subarray((T-1)*192, T*192)
  predictLatents(stateSeed) {
    // the model's *expected* next latent — smooth function of state
    const r = mulberry32(stateSeed | 0), z = new Float32Array(D);
    for (let i = 0; i < D; i++) z[i] = (r() - 0.5) * 2;
    return z;
  },
};

// ── env (C2/C6): real is stepAgent / renderFrameRGB over a 2-DOF TwoRooms world
const ROOMS = Object.freeze({
  left: { x0: 0.07, x1: 0.45, y0: 0.12, y1: 0.88 },
  right: { x0: 0.55, x1: 0.93, y0: 0.12, y1: 0.88 },
  door: { y0: 0.41, y1: 0.59 }, // gap between rooms
});
export const mockEnv = {
  ROOMS,
  // a smooth lap through both rooms via the doorway (predictable in-distribution motion)
  path(t) {
    const s = t * 0.34;
    const cx = 0.5 + 0.42 * Math.sin(s);
    const cy = 0.5 + 0.30 * Math.sin(s * 1.7 + 0.6) * (1 - 0.7 * Math.exp(-((Math.cos(s)) ** 2) * 6));
    return { x: Math.min(0.95, Math.max(0.05, cx)), y: Math.min(0.9, Math.max(0.1, cy)) };
  },
  randomPos(seed) {
    const r = mulberry32(seed | 0), room = r() < 0.5 ? ROOMS.left : ROOMS.right;
    return { x: room.x0 + r() * (room.x1 - room.x0), y: room.y0 + r() * (room.y1 - room.y0) };
  },
  // REAL: encodeFrames(frameToModelInput(renderFrameRGB(pos)), batch) → latent (B,192).
  // Here the "actual" next latent = prediction + deviation sized so MSE/D ≈ targetSurprise.
  actualNextLatent(predicted, targetSurprise, seed) {
    const amp = Math.sqrt(Math.max(0, targetSurprise));
    const n = noiseVec(seed | 0), z = new Float32Array(D);
    for (let i = 0; i < D; i++) z[i] = predicted[i] + amp * n[i];
    return z;
  },
};

// ── probe (C3): MSE over the last timestep, divided by latentDim (mean over dims)
export const mockProbe = {
  // REAL, unchanged on swap: surprise = (1/192)·Σ_k (pred_k − actual_k)²
  surprise(predNext, actualNext) {
    let acc = 0;
    for (let i = 0; i < D; i++) { const e = predNext[i] - actualNext[i]; acc += e * e; }
    return acc / D;
  },
};

// ── adapter (C4/C5): the 12,512-float offset that federation produces.
// REAL: const a = adapterFromInitAndOffset({inputDim:192, hiddenDim:32, initSeed:42, offset});
//       const { y } = adapterForward(a, z, 1);  post-latent = y (z + adapter residual)
// The pre/post drop is the held-out-pairs result (C11); the engine models its
// effect at the surprise baseline (pre=baselineMse → post=adaptedMse).
export const mockAdapter = {
  params: ADAPTER.params, inputDim: ADAPTER.inputDim,
  hiddenDim: ADAPTER.hiddenDim, initSeed: ADAPTER.initSeed,
  loaded: true, // SWAP: fetch web/surprise-meter/fixtures/adapter_offset.json (len 12512)
};

export const MOCK = true;
