// Browser-local LeWM adapter continuation: the bounded trainable subset (#319, epic #314).
//
// The Tapestry-like real mode keeps the exported checkpoint graphs FROZEN (encoder, action
// encoder, predictor run through ONNX Runtime Web) and trains a small residual adapter on the
// predictor output, in plain JavaScript with manual gradients:
//
//   y = z_hat + W2 · tanh(W1 · z_hat + b1) + b2          (W2, b2 zero-init → identity start)
//
// Objective (LeWM-compatible where feasible, deviations documented):
//   L = mean‖y − z_next‖² / D  +  λ · L_var
// where z_next is the frozen encoder's latent of the actually-observed next frame and L_var is a
// variance-floor anti-collapse surrogate (per-dim std of adapted predictions must stay above a
// fraction of the target batch's per-dim std). The TRUE SIGReg Epps–Pulley statistic — an exact
// port of lensemble.model.sigreg (centered, not rescaled, 17-knot Gaussian-weighted CF grid) — is
// computed as a diagnostic on real training tensors every few steps; training itself uses the
// variance-floor surrogate because its gradient is closed-form in JS. This is checkpoint
// ADAPTATION around a frozen base, not from-scratch LeWM browser pretraining.
//
// Residency: raw frames, actions, latents, and adapter tensors stay in the browser. The only
// artifact that may leave is the bounded clipped delta (computeAdapterDelta) of #320.

import { mulberry32 } from "./rng.mjs";

export const ADAPTER_RUNTIME = "lewm-js-adapter-v1";
export const DEFAULT_HIDDEN = 32;
export const SIGREG_KNOTS = 17;
export const SIGREG_T_MAX = 5.0;

function gaussian(rng) {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ---------------------------------------------------------------------------
// SIGReg diagnostic (exact port of lensemble.model.sigreg.sigreg_statistic)
// ---------------------------------------------------------------------------

export function buildSketch(seed, d, sketchDim = 64) {
  const rng = mulberry32(seed >>> 0);
  const sketch = new Float32Array(d * sketchDim); // (d, S) column-major directions
  for (let i = 0; i < sketch.length; i += 1) sketch[i] = gaussian(rng);
  for (let s = 0; s < sketchDim; s += 1) {
    let norm = 0;
    for (let i = 0; i < d; i += 1) norm += sketch[i * sketchDim + s] ** 2;
    norm = Math.max(Math.sqrt(norm), 1e-12);
    for (let i = 0; i < d; i += 1) sketch[i * sketchDim + s] /= norm;
  }
  return { data: sketch, d, sketchDim };
}

export function sigregStatistic(embeddings, n, d, sketch, knots = SIGREG_KNOTS) {
  const s = sketch.sketchDim;
  // project: (n, d) x (d, s)
  const proj = new Float32Array(n * s);
  for (let i = 0; i < n; i += 1) {
    for (let k = 0; k < d; k += 1) {
      const v = embeddings[i * d + k];
      if (v === 0) continue;
      for (let j = 0; j < s; j += 1) proj[i * s + j] += v * sketch.data[k * sketchDim_(sketch) + j];
    }
  }
  // center (do NOT rescale — the statistic tests unit variance per direction)
  for (let j = 0; j < s; j += 1) {
    let mu = 0;
    for (let i = 0; i < n; i += 1) mu += proj[i * s + j];
    mu /= n;
    for (let i = 0; i < n; i += 1) proj[i * s + j] -= mu;
  }
  const ts = new Float64Array(knots);
  const weight = new Float64Array(knots);
  const target = new Float64Array(knots);
  let weightSum = 0;
  for (let k = 0; k < knots; k += 1) {
    const t = -SIGREG_T_MAX + (2 * SIGREG_T_MAX * k) / (knots - 1);
    ts[k] = t;
    weight[k] = Math.exp(-0.5 * t * t);
    target[k] = weight[k]; // standard-normal CF
    weightSum += weight[k];
  }
  let total = 0;
  for (let j = 0; j < s; j += 1) {
    let dirSum = 0;
    for (let k = 0; k < knots; k += 1) {
      let re = 0;
      let im = 0;
      for (let i = 0; i < n; i += 1) {
        const tu = proj[i * s + j] * ts[k];
        re += Math.cos(tu);
        im += Math.sin(tu);
      }
      re /= n;
      im /= n;
      dirSum += ((re - target[k]) ** 2 + im ** 2) * weight[k];
    }
    total += dirSum / weightSum;
  }
  return total / s;
}

function sketchDim_(sketch) {
  return sketch.sketchDim;
}

// ---------------------------------------------------------------------------
// latent diagnostics (real tensors, no fabrication)
// ---------------------------------------------------------------------------

export function latentDiagnostics(latents, n, d) {
  const mean = new Float64Array(d);
  for (let i = 0; i < n; i += 1) {
    for (let k = 0; k < d; k += 1) mean[k] += latents[i * d + k];
  }
  for (let k = 0; k < d; k += 1) mean[k] /= n;
  const varPerDim = new Float64Array(d);
  for (let i = 0; i < n; i += 1) {
    for (let k = 0; k < d; k += 1) varPerDim[k] += (latents[i * d + k] - mean[k]) ** 2;
  }
  let stdSum = 0;
  for (let k = 0; k < d; k += 1) {
    varPerDim[k] /= n;
    stdSum += Math.sqrt(varPerDim[k]);
  }
  // participation ratio over the covariance: tr(C)^2 / tr(C^2) — no eigendecomposition needed
  let trC = 0;
  for (let k = 0; k < d; k += 1) trC += varPerDim[k];
  // tr(C^2) = ||C||_F^2 with C = X^T X / n (X centered)
  let trC2 = 0;
  for (let a = 0; a < d; a += 1) {
    for (let b = a; b < d; b += 1) {
      let cab = 0;
      for (let i = 0; i < n; i += 1) {
        cab += (latents[i * d + a] - mean[a]) * (latents[i * d + b] - mean[b]);
      }
      cab /= n;
      trC2 += a === b ? cab * cab : 2 * cab * cab;
    }
  }
  const effectiveRank = trC2 > 0 ? (trC * trC) / trC2 : 0;
  return {
    latentStdMean: stdSum / d,
    effectiveRank,
    effectiveRankRatio: effectiveRank / d,
  };
}

// ---------------------------------------------------------------------------
// the residual adapter (bounded trainable subset)
// ---------------------------------------------------------------------------

export function createAdapter({ inputDim = 192, hiddenDim = DEFAULT_HIDDEN, seed = 1 } = {}) {
  const rng = mulberry32(seed >>> 0);
  const w1 = new Float32Array(hiddenDim * inputDim);
  const scale = 1 / Math.sqrt(inputDim);
  for (let i = 0; i < w1.length; i += 1) w1[i] = gaussian(rng) * scale;
  return {
    runtime: ADAPTER_RUNTIME,
    inputDim,
    hiddenDim,
    seed,
    params: {
      w1,
      b1: new Float32Array(hiddenDim),
      w2: new Float32Array(inputDim * hiddenDim), // zero-init → identity residual at start
      b2: new Float32Array(inputDim),
    },
  };
}

export function parameterCount(adapter) {
  const { w1, b1, w2, b2 } = adapter.params;
  return w1.length + b1.length + w2.length + b2.length;
}

export function flattenParams(adapter) {
  const flat = new Float32Array(parameterCount(adapter));
  let off = 0;
  for (const key of ["w1", "b1", "w2", "b2"]) {
    flat.set(adapter.params[key], off);
    off += adapter.params[key].length;
  }
  return flat;
}

export function adapterParameterSpec(adapter) {
  return [
    { name: "w1", shape: [adapter.hiddenDim, adapter.inputDim] },
    { name: "b1", shape: [adapter.hiddenDim] },
    { name: "w2", shape: [adapter.inputDim, adapter.hiddenDim] },
    { name: "b2", shape: [adapter.inputDim] },
  ];
}

// forward for a batch: X (n, D) -> { y (n, D), h (n, H) }
export function adapterForward(adapter, x, n) {
  const { inputDim: D, hiddenDim: H } = adapter;
  const { w1, b1, w2, b2 } = adapter.params;
  const h = new Float32Array(n * H);
  const y = new Float32Array(n * D);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < H; j += 1) {
      let acc = b1[j];
      for (let k = 0; k < D; k += 1) acc += w1[j * D + k] * x[i * D + k];
      h[i * H + j] = Math.tanh(acc);
    }
    for (let k = 0; k < D; k += 1) {
      let acc = x[i * D + k] + b2[k];
      for (let j = 0; j < H; j += 1) acc += w2[k * H + j] * h[i * H + j];
      y[i * D + k] = acc;
    }
  }
  return { y, h };
}

// ---------------------------------------------------------------------------
// training step (manual gradients + Adam + global-norm clipping)
// ---------------------------------------------------------------------------

export function createOptimizer(adapter, { lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 } = {}) {
  const state = {};
  for (const key of ["w1", "b1", "w2", "b2"]) {
    state[key] = {
      m: new Float32Array(adapter.params[key].length),
      v: new Float32Array(adapter.params[key].length),
    };
  }
  return { lr, beta1, beta2, eps, t: 0, state };
}

export function lossAndGrads(adapter, batch, { lambda = 0.1, varFloorRatio = 0.5 } = {}) {
  const { x, target, n } = batch; // x: frozen predictor outputs, target: frozen encoder next latents
  const { inputDim: D, hiddenDim: H } = adapter;
  const { w2 } = adapter.params;
  const { y, h } = adapterForward(adapter, x, n);

  // prediction loss: mean over samples and dims
  let predLoss = 0;
  const gy = new Float32Array(n * D);
  for (let i = 0; i < n * D; i += 1) {
    const diff = y[i] - target[i];
    predLoss += diff * diff;
    gy[i] = (2 * diff) / (n * D);
  }
  predLoss /= n * D;

  // variance-floor surrogate: per-dim std of y must stay above varFloorRatio * std of targets
  const muY = new Float64Array(D);
  const muT = new Float64Array(D);
  for (let i = 0; i < n; i += 1) {
    for (let k = 0; k < D; k += 1) {
      muY[k] += y[i * D + k];
      muT[k] += target[i * D + k];
    }
  }
  for (let k = 0; k < D; k += 1) {
    muY[k] /= n;
    muT[k] /= n;
  }
  const stdY = new Float64Array(D);
  const stdT = new Float64Array(D);
  for (let i = 0; i < n; i += 1) {
    for (let k = 0; k < D; k += 1) {
      stdY[k] += (y[i * D + k] - muY[k]) ** 2;
      stdT[k] += (target[i * D + k] - muT[k]) ** 2;
    }
  }
  let varLoss = 0;
  const floorGap = new Float64Array(D);
  for (let k = 0; k < D; k += 1) {
    stdY[k] = Math.sqrt(stdY[k] / n + 1e-8);
    stdT[k] = Math.sqrt(stdT[k] / n + 1e-8);
    const gap = varFloorRatio * stdT[k] - stdY[k];
    floorGap[k] = gap > 0 ? gap : 0;
    varLoss += floorGap[k] ** 2;
  }
  varLoss /= D;
  if (lambda > 0) {
    for (let i = 0; i < n; i += 1) {
      for (let k = 0; k < D; k += 1) {
        if (floorGap[k] > 0) {
          // d/dy of (gap)^2 = 2*gap * (-(y-mu)/(n*std)) averaged over D
          gy[i * D + k] += lambda * ((-2 * floorGap[k] * (y[i * D + k] - muY[k])) / (n * stdY[k] * D));
        }
      }
    }
  }
  const totalLoss = predLoss + lambda * varLoss;

  // backprop: y = x + W2 h + b2; h = tanh(W1 x + b1)
  const gw2 = new Float32Array(D * H);
  const gb2 = new Float32Array(D);
  const gh = new Float32Array(n * H);
  for (let i = 0; i < n; i += 1) {
    for (let k = 0; k < D; k += 1) {
      const g = gy[i * D + k];
      gb2[k] += g;
      for (let j = 0; j < H; j += 1) {
        gw2[k * H + j] += g * h[i * H + j];
        gh[i * H + j] += g * w2[k * H + j];
      }
    }
  }
  const gw1 = new Float32Array(H * D);
  const gb1 = new Float32Array(H);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < H; j += 1) {
      const gpre = gh[i * H + j] * (1 - h[i * H + j] ** 2);
      gb1[j] += gpre;
      for (let k = 0; k < D; k += 1) gw1[j * D + k] += gpre * x[i * D + k];
    }
  }

  const grads = { w1: gw1, b1: gb1, w2: gw2, b2: gb2 };
  let gradNormSq = 0;
  for (const key of Object.keys(grads)) {
    const g = grads[key];
    for (let i = 0; i < g.length; i += 1) gradNormSq += g[i] * g[i];
  }
  return { predLoss, varLoss, totalLoss, grads, gradNorm: Math.sqrt(gradNormSq) };
}

export function trainStep(adapter, optimizer, batch, { lambda = 0.1, varFloorRatio = 0.5, clipNorm = 5.0 } = {}) {
  const { predLoss, varLoss, totalLoss, grads, gradNorm } = lossAndGrads(adapter, batch, {
    lambda,
    varFloorRatio,
  });
  const clipped = gradNorm > clipNorm;
  const clipScale = clipped ? clipNorm / gradNorm : 1;

  // Adam update
  optimizer.t += 1;
  const { lr, beta1, beta2, eps, t, state } = optimizer;
  const bc1 = 1 - beta1 ** t;
  const bc2 = 1 - beta2 ** t;
  for (const key of Object.keys(grads)) {
    const g = grads[key];
    const p = adapter.params[key];
    const { m, v } = state[key];
    for (let i = 0; i < g.length; i += 1) {
      const gi = g[i] * clipScale;
      m[i] = beta1 * m[i] + (1 - beta1) * gi;
      v[i] = beta2 * v[i] + (1 - beta2) * gi * gi;
      p[i] -= (lr * (m[i] / bc1)) / (Math.sqrt(v[i] / bc2) + eps);
    }
  }

  return {
    predLoss,
    varLoss,
    totalLoss,
    gradNorm,
    clipped,
  };
}

// ---------------------------------------------------------------------------
// the bounded update artifact input (#320 consumes this)
// ---------------------------------------------------------------------------

export function computeAdapterDelta(initialFlat, adapter, { clipNorm = 3.0 } = {}) {
  const current = flattenParams(adapter);
  const delta = new Float32Array(current.length);
  let normSq = 0;
  for (let i = 0; i < delta.length; i += 1) {
    delta[i] = current[i] - initialFlat[i];
    normSq += delta[i] * delta[i];
  }
  const rawNorm = Math.sqrt(normSq);
  const scale = rawNorm > clipNorm && rawNorm > 0 ? clipNorm / rawNorm : 1;
  if (scale !== 1) {
    for (let i = 0; i < delta.length; i += 1) delta[i] *= scale;
  }
  return {
    delta,
    l2Norm: Math.min(rawNorm, clipNorm),
    unclippedNorm: rawNorm,
    clipNorm,
    clipSaturation: rawNorm > clipNorm ? 1 : 0,
    parameterCount: delta.length,
  };
}

// ---------------------------------------------------------------------------
// training loop over resident pairs (UI/worker call this; node tests too)
// ---------------------------------------------------------------------------

export function trainAdapterOnPairs(adapter, pairs, {
  steps = 50,
  batchSize = 32,
  seed = 1,
  lambda = 0.1,
  varFloorRatio = 0.5,
  clipNorm = 5.0,
  optimizer = null,
  diagnosticsEvery = 10,
  sketchSeed = 42,
  now = () => (typeof performance !== "undefined" ? performance.now() : Date.now()),
} = {}) {
  const D = adapter.inputDim;
  const n = pairs.count;
  if (n < 2) throw new Error("need at least 2 resident training pairs");
  const opt = optimizer ?? createOptimizer(adapter);
  const rng = mulberry32(seed >>> 0);
  const sketch = buildSketch(sketchSeed, D);
  const started = now();
  const history = [];
  const diagnostics = [];
  for (let step = 0; step < steps; step += 1) {
    const m = Math.min(batchSize, n);
    const x = new Float32Array(m * D);
    const target = new Float32Array(m * D);
    for (let i = 0; i < m; i += 1) {
      const pick = Math.floor(rng() * n) % n;
      x.set(pairs.x.subarray(pick * D, (pick + 1) * D), i * D);
      target.set(pairs.target.subarray(pick * D, (pick + 1) * D), i * D);
    }
    const metrics = trainStep(adapter, opt, { x, target, n: m }, { lambda, varFloorRatio, clipNorm });
    history.push({ step, ...metrics });
    if (step % diagnosticsEvery === 0 || step === steps - 1) {
      const { y } = adapterForward(adapter, x, m);
      diagnostics.push({
        step,
        ...latentDiagnostics(y, m, D),
        sigregStatistic: sigregStatistic(y, m, D, sketch),
      });
    }
  }
  const runtimeMs = now() - started;
  return {
    runtime: ADAPTER_RUNTIME,
    steps,
    batchSize,
    pairCount: n,
    history,
    diagnostics,
    runtimeMs,
    firstLoss: history[0].predLoss,
    lastLoss: history[history.length - 1].predLoss,
  };
}
