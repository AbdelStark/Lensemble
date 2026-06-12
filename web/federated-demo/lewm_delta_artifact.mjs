// lewm-adapter-delta/1 artifact builder (#320, epic #314).
//
// Turns a browser-local continuation result (lewm_local_trainer.mjs) into the bounded update
// artifact the coordinator validates: the clipped flattened adapter delta, the adapter spec, the
// checkpoint/export binding from the run snapshot, and the honest metric summary. Nothing else —
// no frames, actions, latents, raw tensors, or tokens.

export const LEWM_UPDATE_SCHEMA = "lewm-adapter-delta/1";

async function sha256Hex(text) {
  const subtle = globalThis.crypto?.subtle;
  if (subtle) {
    const digest = await subtle.digest("SHA-256", new TextEncoder().encode(text));
    return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
  }
  // deterministic fallback for environments without WebCrypto (still 64-hex; weaker)
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

export async function buildAdapterDeltaArtifact({
  result, // from runLocalAdapterContinuation
  runId,
  participantId,
  round,
  roundId = `${runId}:round-${round}`,
  modelRevisionId,
  binding, // run snapshot .lewmBinding
  participantMode = "auto",
  seed = 1,
  simulated = false,
}) {
  if (!binding?.checkpoint?.revision) {
    throw new Error("run snapshot carries no lewmBinding — not a real-lewm-tworooms run");
  }
  const delta = Array.from(result.delta.delta, (v) => Number(v.toFixed(8)));
  const metrics = result.metrics;
  const hash = await sha256Hex(
    [runId, participantId, round, modelRevisionId, delta.join(",")].join("|"),
  );
  return {
    schema: LEWM_UPDATE_SCHEMA,
    runId,
    participantId,
    round,
    roundId,
    modelRevisionId,
    baseCheckpoint: { ...binding.checkpoint },
    exportGraphHashes: { ...binding.exportGraphHashes },
    adapterSpec: binding.adapterSpec.map((entry) => ({ name: entry.name, shape: [...entry.shape] })),
    dtype: "float32",
    parameterCount: delta.length,
    delta,
    l2Norm: Number(result.delta.l2Norm.toFixed(8)),
    clipNorm: result.delta.clipNorm,
    unclippedNorm: Number(result.delta.unclippedNorm.toFixed(8)),
    hash,
    metrics: {
      pairCount: metrics.pairCount,
      optimizerSteps: metrics.optimizerSteps,
      predLossFirst: metrics.predLossFirst,
      predLossLast: metrics.predLossLast,
      lossDecreased: metrics.lossDecreased,
      varLossLast: metrics.varLossLast,
      sigregStatistic: metrics.sigregStatistic,
      effectiveRank: metrics.effectiveRank,
      effectiveRankRatio: metrics.effectiveRankRatio,
      latentStdMean: metrics.latentStdMean,
      gradClipEvents: metrics.gradClipEvents,
      envSteps: metrics.envSteps,
      episodes: metrics.episodes,
      trainMs: metrics.trainMs,
      collectMs: metrics.collectMs,
      deltaUnclippedNorm: metrics.deltaUnclippedNorm,
    },
    participantMode,
    runtime: result.runtime,
    seed,
    simulated,
  };
}
