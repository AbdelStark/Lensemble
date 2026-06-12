// Node-runnable selftest for the dashboard chart data preparation (pure functions only).
// Run: node web/federated-demo/charts_selftest.mjs

import {
  formatTick,
  niceTicks,
  participantLossSeries,
  roundSeries,
  seriesExtent,
} from "./charts.mjs";

const failures = [];
let total = 0;

function check(name, fn) {
  total += 1;
  try {
    fn();
  } catch (error) {
    failures.push({ name, error: String(error?.message ?? error) });
  }
}

function assert(cond, message) {
  if (!cond) throw new Error(message);
}

check("participant loss series reads real-mode adapter metrics in round order", () => {
  const run = {
    participants: [
      {
        id: "browser-a",
        displayName: "Alice",
        updateMetadata: {
          2: { metrics: { predLossLast: 0.02 } },
          1: { metrics: { predLossLast: 0.03 } },
        },
      },
      { id: "browser-b", displayName: null, updateMetadata: { 1: { loss: 0.4 } } }, // surrogate shape
      { id: "browser-c", updateMetadata: {} }, // no submissions -> no series
    ],
  };
  const series = participantLossSeries(run);
  assert(series.length === 2, `two series, got ${series.length}`);
  assert(series[0].label === "Alice", "display name used");
  assert(series[0].points[0].x === 1 && series[0].points[1].x === 2, "rounds sorted");
  assert(series[0].points[0].y === 0.03, "real metric read");
  assert(series[1].label === "browser-b" && series[1].points[0].y === 0.4, "surrogate fallback");
});

check("round series pulls only finite values and keeps labels/dashes", () => {
  const run = {
    roundMetrics: [
      { round: 1, predLossLastMean: 0.03, sigregStatisticMean: 0.02 },
      { round: 2, predLossLastMean: null, sigregStatisticMean: 0.04 },
      { round: 3, predLossLastMean: 0.01, sigregStatisticMean: 0.03 },
    ],
  };
  const series = roundSeries(run, [
    { key: "predLossLastMean", label: "loss", dashed: true },
    { key: "missingKey" },
  ]);
  assert(series.length === 1, "missing key produces no series");
  assert(series[0].label === "loss" && series[0].dashed === true, "spec carried");
  assert(series[0].points.length === 2, "null round skipped");
  assert(series[0].points[1].x === 3, "gap preserved, not interpolated");
});

check("series extent handles flat and single-point data without zero spans", () => {
  const flat = seriesExtent([{ points: [{ x: 1, y: 0.5 }, { x: 2, y: 0.5 }] }]);
  assert(flat.yMax > flat.yMin, "flat y padded");
  const single = seriesExtent([{ points: [{ x: 3, y: 1 }] }]);
  assert(single.xMax > single.xMin, "single x padded");
  assert(seriesExtent([{ points: [] }]) === null, "empty -> null");
});

check("nice ticks are monotonic, bounded, and cover the domain", () => {
  const ticks = niceTicks(0, 0.061, 4);
  assert(ticks.length >= 2 && ticks.length <= 6, `tick count sane, got ${ticks.length}`);
  for (let i = 1; i < ticks.length; i += 1) assert(ticks[i] > ticks[i - 1], "monotonic");
  assert(ticks[0] >= 0 && ticks[ticks.length - 1] <= 0.061 + 1e-9, "in range");
  assert(niceTicks(5, 5).length === 1, "degenerate domain -> single tick");
});

check("tick labels stay compact across magnitudes", () => {
  assert(formatTick(0) === "0", "zero");
  assert(formatTick(212.4) === "212", "large rounds");
  assert(formatTick(3.25) === "3.3", "unit scale");
  assert(formatTick(0.025) === "0.025", "metric scale");
  assert(formatTick(0.00042).includes("e"), "tiny goes exponential");
});

const report = { total, passed: total - failures.length, failed: failures.length, failures };
console.log(JSON.stringify(report));
if (failures.length > 0) process.exit(1);
