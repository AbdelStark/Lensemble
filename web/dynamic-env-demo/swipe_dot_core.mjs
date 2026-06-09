export const DEFAULT_SIZE = 48;
export const DEFAULT_GAIN = 0.18;

export function clamp(value, lo = 0, hi = 1) {
  return Math.max(lo, Math.min(hi, value));
}

export function stepSwipeDot(state, action, gain = DEFAULT_GAIN) {
  const ax = clamp(Number(action[0] ?? 0), -1, 1);
  const ay = clamp(Number(action[1] ?? 0), -1, 1);
  return {
    x: clamp(Number(state.x) + gain * ax),
    y: clamp(Number(state.y) + gain * ay),
  };
}

export function renderSwipeDotRGBA(state, size = DEFAULT_SIZE) {
  const out = new Uint8ClampedArray(size * size * 4);
  const cx = clamp(Number(state.x)) * (size - 1);
  const cy = clamp(Number(state.y)) * (size - 1);
  const sigma = size * 0.08;
  const denom = 2 * sigma * sigma;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const idx = (y * size + x) * 4;
      const dx = x - cx;
      const dy = y - cy;
      const blob = Math.exp(-(dx * dx + dy * dy) / denom);
      out[idx] = Math.round(16 + 220 * blob);
      out[idx + 1] = Math.round(22 + 112 * blob);
      out[idx + 2] = Math.round(30 + 42 * blob);
      out[idx + 3] = 255;
    }
  }
  return out;
}

export function rgbaToNchwFloat(rgba, size = DEFAULT_SIZE) {
  const clip = new Float32Array(1 * 1 * 3 * size * size);
  const plane = size * size;
  for (let i = 0; i < plane; i += 1) {
    clip[i] = rgba[i * 4] / 255;
    clip[plane + i] = rgba[i * 4 + 1] / 255;
    clip[2 * plane + i] = rgba[i * 4 + 2] / 255;
  }
  return clip;
}
