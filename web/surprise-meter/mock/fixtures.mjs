// Fallback evidence constants for the surprise-meter.
//
// The numbers below mirror, at full precision, the certified system-composed
// probe so the HUD can render even if the generated served bundle is missing:
//   docs/evidence/lewm_tworooms_system_probe.json   (this run)
//   docs/evidence/lewm_tworooms_probe_seedsweep.json (5-seed distribution)

export const CERTIFIED = Object.freeze({
  // lewm_tworooms_system_probe.json
  baselineMse: 0.06037897796856822,
  adaptedMse: 0.05296723026100999,
  relativeImprovement: 0.12275377883038366, // +12.3% — "this run"
  // lewm_tworooms_probe_seedsweep.json / distribution
  mean: 0.16787180214169914, // +16.8%
  worst: 0.054144202317108696, // +5.4% — seed 2 (always shown beside the mean)
  worstSeed: 2,
  best: 0.32634644960379955, // +32.6% — seed 4
  stdev: 0.10969915982049769, // 0.11
  seeds: 5,
  allImproved: true,
  collapse: false,
  source: "lewm_tworooms_surprise.json · …_probe_seedsweep.json",
});

// Adapter dimensions (C4/C5). The real offset is a 12,512-float vector.
export const ADAPTER = Object.freeze({
  params: 12512,
  fractionOfModel: 0.00069, // 0.069%
  inputDim: 192,
  hiddenDim: 32,
  initSeed: 42,
});

// World-model facts (C1) — for captions / readouts.
export const MODEL = Object.freeze({
  latentDim: 192,
  numFrames: 3,
  imageSize: 224,
  actionDim: 10,
  msPerStep: 6,
});

// Claim-discipline footer — mirrors surprise-meter/05 + AGENTS.md, rendered from
// data so it can never silently drift off-screen.
export const NON_CLAIMS = Object.freeze([
  "Federated adapter continuation on a frozen checkpoint — not federated world-model training.",
  "Surprise is a scalar next-latent prediction error (CLS-latent), not a per-pixel heatmap.",
  "Perturbation behaviour is illustrative on the TwoRooms backbone, not a calibrated anomaly detector.",
  "No differential-privacy, secure-aggregation, cryptographic-proof, beats-local-only, or paper-scale claim.",
]);

export const pct = (x) => (x >= 0 ? "+" : "") + (x * 100).toFixed(1) + "%";
