// Node side of the system-composed federation gate (#327, epic #332).
//
// This is the REAL browser-local training step, factored so the Python orchestrator
// (scripts/lewm_system_probe.py) can drive the ACTUAL server aggregation/validation path with
// genuine ONNX-trained adapter deltas — not a reimplemented coordinator mean. It never touches the
// coordinator math: it only does what a browser participant does (train the bounded adapter from
// shared-init + the CURRENT server offset, clip the delta) and, for the final verdict, scores the
// SERVER-PRODUCED global offset on the held-out pairs.
//
// Usage: node web/federated-demo/lewm_system_round.mjs <request.json>
//   request.op === "train-round": train every participant from (init + offset) -> bounded deltas
//   request.op === "probe":       score (init + serverOffset) vs identity on the validation pairs
//
// stdout: a single JSON line with the result (the orchestrator reads the last line).

import { readFileSync } from "node:fs";
import {
  adapterFromInitAndOffset,
  computeAdapterDelta,
  flattenParams,
  trainAdapterOnPairs,
} from "./lewm_adapter.mjs";
import { probeAdapterOffset } from "./lewm_probe.mjs";

const path = process.argv[2];
if (!path) {
  console.error("usage: node lewm_system_round.mjs <request.json>");
  process.exit(2);
}
const req = JSON.parse(readFileSync(path, "utf8"));
const dim = req.dim;
const hiddenDim = req.hiddenDim ?? 32;
const initSeed = req.initSeed ?? 42;

function toPairs(p) {
  return { x: Float32Array.from(p.x), target: Float32Array.from(p.target), count: p.count };
}

if (req.op === "train-round") {
  const offset = req.offset ? Float32Array.from(req.offset) : null;
  const clipNorm = req.clipNorm ?? 3.0;
  const deltas = req.participants.map((participant, index) => {
    const pairs = toPairs(participant);
    // exactly the browser participant's job: continue the bounded adapter from (shared init +
    // the current GLOBAL offset the server handed back), in plain-JS manual-gradient training
    const start = adapterFromInitAndOffset({ inputDim: dim, hiddenDim, initSeed, offset });
    const startFlat = flattenParams(start);
    const report = trainAdapterOnPairs(start, pairs, {
      steps: req.stepsPerRound ?? 20,
      batchSize: req.batchSize ?? 32,
      seed: req.round * 101 + index,
      lambda: 0.1,
    });
    const info = computeAdapterDelta(startFlat, start, { clipNorm });
    const lastDiag = report.diagnostics[report.diagnostics.length - 1];
    return {
      delta: Array.from(info.delta),
      l2Norm: info.l2Norm,
      unclippedNorm: info.unclippedNorm,
      clipNorm: info.clipNorm,
      clipSaturation: info.clipSaturation,
      parameterCount: info.parameterCount,
      metrics: {
        pairCount: report.pairCount,
        optimizerSteps: report.steps,
        predLossFirst: report.firstLoss,
        predLossLast: report.lastLoss,
        sigregStatistic: lastDiag.sigregStatistic,
        effectiveRank: lastDiag.effectiveRank,
        latentStdMean: lastDiag.latentStdMean,
        lossDecreased: report.lastLoss < report.firstLoss,
        trainMs: report.runtimeMs,
      },
    };
  });
  console.log(JSON.stringify({ op: "train-round", round: req.round, deltas }));
} else if (req.op === "probe") {
  const report = probeAdapterOffset({
    validationPairs: toPairs(req.validation),
    adaptedState: req.offset ? Float32Array.from(req.offset) : null,
    inputDim: dim,
    hiddenDim,
    initSeed,
    seed: req.seed ?? 991,
  });
  console.log(JSON.stringify({ op: "probe", report }));
} else {
  console.error(`unknown op ${req.op}`);
  process.exit(2);
}
