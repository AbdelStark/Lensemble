// One-browser overfit/regression driver for the adapter continuation (#319 gate G3).
//
// Run: node web/federated-demo/lewm_adapter_overfit.mjs <pairs.json>
// The input JSON carries REAL (frozen-prediction, next-latent) pairs precomputed through the
// exported checkpoint graphs on official expert-dataset trajectories
// (scripts/lewm_adapter_overfit_check.py). This driver runs the actual JS trainer the browser
// ships and prints the training report, proving real loss decrease for the bounded trainable
// subset before federation is attempted.

import { readFileSync } from "node:fs";
import {
  createAdapter,
  computeAdapterDelta,
  flattenParams,
  trainAdapterOnPairs,
} from "./lewm_adapter.mjs";

const path = process.argv[2];
if (!path) {
  console.error("usage: node lewm_adapter_overfit.mjs <pairs.json>");
  process.exit(2);
}
const fixture = JSON.parse(readFileSync(path, "utf8"));
const d = fixture.dim;
const count = fixture.count;
const pairs = {
  x: Float32Array.from(fixture.x),
  target: Float32Array.from(fixture.target),
  count,
};

const adapter = createAdapter({ inputDim: d, hiddenDim: fixture.hiddenDim ?? 32, seed: fixture.seed ?? 1 });
const initial = flattenParams(adapter);
const report = trainAdapterOnPairs(adapter, pairs, {
  steps: fixture.steps ?? 200,
  batchSize: fixture.batchSize ?? 32,
  seed: fixture.seed ?? 1,
  lambda: fixture.lambda ?? 0.1,
});
const delta = computeAdapterDelta(initial, adapter, { clipNorm: fixture.clipNorm ?? 3.0 });

console.log(
  JSON.stringify({
    runtime: report.runtime,
    pairCount: count,
    dim: d,
    steps: report.steps,
    firstPredLoss: report.firstLoss,
    lastPredLoss: report.lastLoss,
    lossDecreased: report.lastLoss < report.firstLoss,
    relativeImprovement: (report.firstLoss - report.lastLoss) / report.firstLoss,
    finalDiagnostics: report.diagnostics[report.diagnostics.length - 1],
    delta: {
      l2Norm: delta.l2Norm,
      unclippedNorm: delta.unclippedNorm,
      clipSaturation: delta.clipSaturation,
      parameterCount: delta.parameterCount,
    },
    lossCurve: report.history
      .filter((h, i) => i % 10 === 0 || i === report.history.length - 1)
      .map((h) => ({ step: h.step, predLoss: Number(h.predLoss.toFixed(6)) })),
  }),
);
