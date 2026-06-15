// Offline federated before/after probe over REAL latent pairs (#322 gate G5).
//
// Run: node web/federated-demo/lewm_probe_check.mjs <fixture.json>
// The fixture (scripts/lewm_probe_check.py) carries disjoint train/validation pairs computed
// through the exported checkpoint graphs on official expert episodes, split across participants.
// This driver mirrors the SHIPPING federation algorithm exactly: every round, each participant
// trains from (shared init + global offset) on its own pairs with the shipping trainer, deltas
// are clipped and averaged (the coordinator's deterministic mean), and the offset accumulates.
// The verdict scores the final global adapter against the identity baseline on the held-out
// validation pairs — a negative result prints "worse"/"flat" and fails the gate.

import { readFileSync } from "node:fs";
import {
  adapterFromFlat,
  adapterFromInitAndOffset,
  flattenParams,
  trainAdapterOnPairs,
  computeAdapterDelta,
} from "./lewm_adapter.mjs";
import { heldOutCollapseDiagnostics, predictOnPairs, scoreAdapterOnPairs } from "./lewm_probe.mjs";

const path = process.argv[2];
if (!path) {
  console.error("usage: node lewm_probe_check.mjs <fixture.json>");
  process.exit(2);
}
const fixture = JSON.parse(readFileSync(path, "utf8"));
const d = fixture.dim;
const hiddenDim = fixture.adapterHidden ?? 32;
const initSeed = fixture.adapterInitSeed ?? 42;
const rounds = fixture.rounds ?? 3;
const clipNorm = fixture.clipNorm ?? 3.0;

const participants = fixture.participants.map((p) => ({
  pairs: { x: Float32Array.from(p.x), target: Float32Array.from(p.target), count: p.count },
}));
const validation = {
  x: Float32Array.from(fixture.validation.x),
  target: Float32Array.from(fixture.validation.target),
  count: fixture.validation.count,
};

let offset = new Float32Array(flattenParams(adapterFromInitAndOffset({ inputDim: d, hiddenDim, initSeed })).length);
const roundTelemetry = [];

for (let round = 1; round <= rounds; round += 1) {
  const deltas = [];
  const losses = [];
  for (const [index, participant] of participants.entries()) {
    const start = adapterFromInitAndOffset({ inputDim: d, hiddenDim, initSeed, offset });
    const startFlat = flattenParams(start);
    const report = trainAdapterOnPairs(start, participant.pairs, {
      steps: fixture.stepsPerRound ?? 60,
      batchSize: fixture.batchSize ?? 24,
      seed: round * 101 + index,
      lambda: 0.1,
    });
    const delta = computeAdapterDelta(startFlat, start, { clipNorm });
    deltas.push(delta.delta);
    losses.push({ first: report.firstLoss, last: report.lastLoss });
  }
  // the coordinator's deterministic mean, folded into the offset
  for (let i = 0; i < offset.length; i += 1) {
    let mean = 0;
    for (const delta of deltas) mean += delta[i];
    offset[i] += mean / deltas.length;
  }
  roundTelemetry.push({ round, losses });
}

const baselineMse = scoreAdapterOnPairs(null, validation, d);
const adapted = adapterFromInitAndOffset({ inputDim: d, hiddenDim, initSeed, offset });
const adaptedMse = scoreAdapterOnPairs(adapted, validation, d);
const relativeImprovement = (baselineMse - adaptedMse) / baselineMse;
let verdict = "flat";
if (relativeImprovement > 0.02) verdict = "improved";
else if (relativeImprovement < -0.02) verdict = "worse";

// Held-out collapse diagnostics (#328): prove the MSE gain is bias-correction, not collapse.
const diagnostics = heldOutCollapseDiagnostics({
  baseline: predictOnPairs(null, validation),
  adapted: predictOnPairs(adapted, validation),
  n: validation.count,
  d,
});
const displayVerdict = diagnostics.collapseRisk ? "collapse-risk" : verdict;

console.log(
  JSON.stringify({
    rounds,
    participants: participants.length,
    validationPairs: validation.count,
    baselineMse,
    adaptedMse,
    relativeImprovement,
    verdict,
    displayVerdict,
    collapseRisk: diagnostics.collapseRisk,
    collapseFlags: diagnostics.collapseFlags,
    diagnostics,
    roundTelemetry,
  }),
);
if (verdict === "worse" || diagnostics.collapseRisk) process.exit(1);
