// Node-runnable selftest for the TwoRooms real-LeWM modules (#318).
//
// Run: node web/federated-demo/tworooms_selftest.mjs
// Deterministic and ONNX-free: environment geometry/dynamics/policy tests run the JS port
// directly; runtime wiring tests inject fake sessions into createLewmRuntime to pin the rollout
// windowing, history action placement, and CEM planning contract the browser relies on.
// Driven by tests/ml/test_lewm_tworooms_browser.py.

import {
  ACTION_BLOCK,
  AGENT_RADIUS,
  BORDER_SIZE,
  DOOR_CENTER_Y,
  IMG_SIZE,
  SUCCESS_DISTANCE,
  WALL_CENTER,
  WALL_HALF,
  createExpertPolicy,
  distanceToTarget,
  frameFingerprint,
  frameToModelInput,
  packActionBlock,
  renderFrameRGB,
  renderGoalFrameRGB,
  sampleEpisode,
  stepAgent,
  stepEpisode,
  stepEpisodeBlock,
} from "./tworooms_env.mjs";
import { createLewmRuntime } from "./lewm_runtime.mjs";
import { createTwoRoomsLab } from "./tworooms_panel.mjs";
import { parseRoute } from "./join_url.mjs";
import { mulberry32 } from "./rng.mjs";

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

async function checkAsync(name, fn) {
  total += 1;
  try {
    await fn();
  } catch (error) {
    failures.push({ name, error: String(error?.message ?? error) });
  }
}

function assert(cond, message) {
  if (!cond) throw new Error(message);
}

function px(rgb, x, y) {
  const idx = (y * IMG_SIZE + x) * 3;
  return [rgb[idx], rgb[idx + 1], rgb[idx + 2]];
}

// ---------------------------------------------------------------------------
// environment: geometry validated against the released dataset frames
// ---------------------------------------------------------------------------

check("wall stripe spans x in [107,117] outside the door span", () => {
  const rgb = renderFrameRGB({ x: 60, y: 180 });
  assert(px(rgb, 107, 100).every((v) => v === 0), "wall left edge black");
  assert(px(rgb, 117, 100).every((v) => v === 0), "wall right edge black");
  assert(px(rgb, 106, 100).every((v) => v === 255), "left of wall white");
  assert(px(rgb, 118, 100).every((v) => v === 255), "right of wall white");
});

check("door cuts the wall at y in [35,63]", () => {
  const rgb = renderFrameRGB({ x: 60, y: 180 });
  assert(px(rgb, WALL_CENTER, DOOR_CENTER_Y).every((v) => v === 255), "door pixel white");
  assert(px(rgb, WALL_CENTER, 34).every((v) => v === 0), "above door is wall");
  assert(px(rgb, WALL_CENTER, 64).every((v) => v === 0), "below door is wall");
});

check("border lines at offsets [10,13] on all sides", () => {
  const rgb = renderFrameRGB({ x: 60, y: 180 });
  for (const c of [10, 13]) {
    assert(px(rgb, c, 150).every((v) => v === 0), `left border col ${c}`);
    assert(px(rgb, IMG_SIZE - BORDER_SIZE, 150).every((v) => v === 0), "right border");
    assert(px(rgb, 150, c).every((v) => v === 0), `top border row ${c}`);
  }
});

check("agent renders as a red Gaussian dot at its position", () => {
  const rgb = renderFrameRGB({ x: 60.5, y: 180.25 });
  const [r, g, b] = px(rgb, 60, 180);
  assert(r > 250 && g < 30 && b < 30, `agent pixel red, got ${[r, g, b]}`);
  assert(px(rgb, 100, 180)[1] > 254, "away from dot stays white");
});

check("goal frame draws the agent at the target position", () => {
  const rgb = renderGoalFrameRGB({ x: 170, y: 60 });
  const [r, g] = px(rgb, 170, 60);
  assert(r > 250 && g < 30, "goal frame has red dot at target");
});

check("frame fingerprints are deterministic", () => {
  const a = frameFingerprint(renderFrameRGB({ x: 60, y: 180 }));
  const b = frameFingerprint(renderFrameRGB({ x: 60, y: 180 }));
  const c = frameFingerprint(renderFrameRGB({ x: 61, y: 180 }));
  assert(a === b, "same state, same fingerprint");
  assert(a !== c, "different state, different fingerprint");
});

check("model input is CHW in [0,1]", () => {
  const rgb = renderFrameRGB({ x: 60, y: 180 });
  const chw = frameToModelInput(rgb);
  assert(chw.length === 3 * IMG_SIZE * IMG_SIZE, "CHW length");
  const plane = IMG_SIZE * IMG_SIZE;
  const idx = 180 * IMG_SIZE + 60;
  assert(chw[idx] > 0.98, "R channel high at agent");
  assert(chw[plane + idx] < 0.12, "G channel low at agent");
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of chw) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  assert(hi <= 1 && lo >= 0, "range [0,1]");
});

// ---------------------------------------------------------------------------
// dynamics
// ---------------------------------------------------------------------------

check("border clamp accounts for the agent radius", () => {
  const next = stepAgent({ x: 22, y: 22 }, [-1, -1]);
  assert(next.x === BORDER_SIZE + AGENT_RADIUS, `x clamped, got ${next.x}`);
  assert(next.y === BORDER_SIZE + AGENT_RADIUS, `y clamped, got ${next.y}`);
});

check("wall blocks crossing outside the door", () => {
  const next = stepAgent({ x: 98, y: 150 }, [1, 0]);
  assert(next.x === WALL_CENTER - WALL_HALF - AGENT_RADIUS - 0.5, `clamped at wall, got ${next.x}`);
  const fromRight = stepAgent({ x: 126, y: 150 }, [-1, 0]);
  assert(fromRight.x === WALL_CENTER + WALL_HALF + AGENT_RADIUS + 0.5, "clamped from right");
});

check("the door lets the agent through", () => {
  let pos = { x: 98, y: DOOR_CENTER_Y };
  for (let i = 0; i < 8; i += 1) pos = stepAgent(pos, [1, 0]);
  assert(pos.x > WALL_CENTER + WALL_HALF, `crossed through door, got x=${pos.x}`);
});

check("episode terminates within the success distance", () => {
  const episode = { agent: { x: 60, y: 60 }, target: { x: 70, y: 60 }, steps: 0, done: false };
  const next = stepEpisode(episode, [1, 0]);
  assert(next.done === true, "within 16px is done");
  assert(distanceToTarget(next) < SUCCESS_DISTANCE, "distance check");
});

check("action blocks pack 5 env actions and step 5 times", () => {
  const block = packActionBlock([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]);
  assert(block.length === 10, "flattened block");
  const episode = { agent: { x: 30, y: 150 }, target: { x: 200, y: 150 }, steps: 0, done: false };
  const next = stepEpisodeBlock(episode, block);
  assert(next.steps === ACTION_BLOCK, "5 env steps");
  assert(Math.abs(next.agent.x - 55) < 1e-9, `moved 5*speed, got ${next.agent.x}`);
  let threw = false;
  try {
    packActionBlock([[1, 0]]);
  } catch {
    threw = true;
  }
  assert(threw, "short blocks rejected");
});

// ---------------------------------------------------------------------------
// expert policy
// ---------------------------------------------------------------------------

check("noise-free expert reaches a cross-room target through the door", () => {
  const rng = mulberry32(7);
  const expert = createExpertPolicy({ actionNoise: 0, actionRepeatProb: 0 });
  let episode = { agent: { x: 40, y: 180 }, target: { x: 190, y: 170 }, steps: 0, done: false };
  for (let i = 0; i < 200 && !episode.done; i += 1) {
    episode = stepEpisode(episode, expert(episode, rng));
  }
  assert(episode.done, `expert reached target, steps=${episode.steps}`);
});

check("noisy expert (dataset settings) still reaches the target", () => {
  const rng = mulberry32(11);
  const expert = createExpertPolicy();
  let episode = sampleEpisode(rng);
  for (let i = 0; i < 600 && !episode.done; i += 1) {
    episode = stepEpisode(episode, expert(episode, rng));
  }
  assert(episode.done, `noisy expert reached target, steps=${episode.steps}`);
});

check("episode sampling avoids the wall zone for the agent", () => {
  const rng = mulberry32(13);
  for (let i = 0; i < 50; i += 1) {
    const { agent } = sampleEpisode(rng);
    const inWall =
      agent.x >= WALL_CENTER - WALL_HALF - AGENT_RADIUS &&
      agent.x <= WALL_CENTER + WALL_HALF + AGENT_RADIUS;
    assert(!inWall, `agent sampled outside wall zone, got x=${agent.x}`);
  }
});

// ---------------------------------------------------------------------------
// runtime wiring with fake sessions (no ONNX in node)
// ---------------------------------------------------------------------------

const HIDDEN = 4;

function fakeManifest() {
  return {
    schema: "lewm-browser-export/1",
    graphVersion: 1,
    opset: 18,
    architecture: { hiddenDim: HIDDEN, numFrames: 3, imageSize: 8, actionDim: 10 },
    checkpoint: { repoId: "quentinll/lewm-tworooms", revision: "77adaae0", weightsSha256: "ab".repeat(32) },
    files: {},
  };
}

// encoder: latent = mean of each image quadrant (any deterministic 4-dim summary works)
const fakeEncoder = {
  run: async (feeds) => {
    const t = feeds.pixels;
    const batch = t.dims[0];
    const out = new Float32Array(batch * HIDDEN);
    const per = t.data.length / batch;
    for (let b = 0; b < batch; b += 1) {
      for (let i = 0; i < per; i += 1) out[b * HIDDEN + (i % HIDDEN)] += t.data[b * per + i] / per;
    }
    return { latent: { data: out, dims: [batch, HIDDEN] } };
  },
};

// action embed: first HIDDEN entries of each raw block (records calls for placement checks)
const actionCalls = [];
const fakeAction = {
  run: async (feeds) => {
    const t = feeds.actions;
    const [batch, time, dim] = t.dims;
    actionCalls.push({ dims: t.dims.slice(), data: Float32Array.from(t.data) });
    const out = new Float32Array(batch * time * HIDDEN);
    for (let b = 0; b < batch; b += 1) {
      for (let i = 0; i < time; i += 1) {
        for (let d = 0; d < HIDDEN; d += 1) out[(b * time + i) * HIDDEN + d] = t.data[(b * time + i) * dim + d];
      }
    }
    return { action_embedding: { data: out, dims: [batch, time, HIDDEN] } };
  },
};

// predictor: next latent = action embedding at the last position (records window sizes)
const predictorCalls = [];
const fakePredictor = {
  run: async (feeds) => {
    const lat = feeds.latents;
    const act = feeds.action_embeddings;
    predictorCalls.push({ time: lat.dims[1] });
    const [batch, time] = lat.dims;
    const out = new Float32Array(batch * time * HIDDEN);
    for (let b = 0; b < batch; b += 1) {
      for (let i = 0; i < time; i += 1) {
        for (let d = 0; d < HIDDEN; d += 1) {
          out[(b * time + i) * HIDDEN + d] = act.data[(b * time + i) * HIDDEN + d];
        }
      }
    }
    return { predicted_latents: { data: out, dims: [batch, time, HIDDEN] } };
  },
};

function fakeRuntime() {
  return createLewmRuntime({
    sessions: { encoder: fakeEncoder, action: fakeAction, predictor: fakePredictor },
    manifest: fakeManifest(),
  });
}

await checkAsync("rollout truncates the history window to numFrames", async () => {
  const rt = fakeRuntime();
  predictorCalls.length = 0;
  const history = [new Float32Array(HIDDEN), new Float32Array(HIDDEN), new Float32Array(HIDDEN)];
  const time = 6;
  const actEmb = new Float32Array(1 * time * HIDDEN);
  const frames = await rt.rollout(history, actEmb, 1, time);
  assert(frames[0].length === time + 1, `rollout length h+steps+1, got ${frames[0].length}`);
  assert(Math.max(...predictorCalls.map((c) => c.time)) === 3, "window capped at numFrames");
});

await checkAsync("planAction places history blocks and returns the elite candidate", async () => {
  const rt = fakeRuntime();
  actionCalls.length = 0;
  const rng = mulberry32(5);
  const history = [
    Float32Array.from([0.1, 0.1, 0.1, 0.1]),
    Float32Array.from([0.2, 0.2, 0.2, 0.2]),
    Float32Array.from([0.3, 0.3, 0.3, 0.3]),
  ];
  const historyBlocks = [
    Float32Array.from({ length: 10 }, () => 0.5),
    Float32Array.from({ length: 10 }, () => -0.5),
  ];
  // with the fake stack, terminal latent == first 4 dims of the final candidate block
  const goalLatent = Float32Array.from([1, 1, -1, -1]);
  const plan = await rt.planAction({
    historyLatents: history,
    historyActionBlocks: historyBlocks,
    goalLatent,
    rng,
    horizon: 2,
    samples: 16,
    eliteCount: 4,
    iterations: 2,
  });
  assert(plan.actionBlock.length === 10, "one frameskip block returned");
  assert(plan.candidateCosts.length === 16, "all candidates scored");
  assert(plan.cost === Math.min(...plan.candidateCosts), "chosen = elite");
  // history block placement: first call's first two time slots carry the provided history blocks
  const call = actionCalls[0];
  assert(call.dims[1] === history.length + 2 - 1, "time = h + horizon - 1");
  assert(call.data[0] === 0.5 && call.data[10] === -0.5, "history blocks at positions 0 and 1");
  // under the fake stack the cost only depends on the FINAL block (terminal latent = its first
  // 4 dims), so the CEM should drive that block's leading dims toward the goal pattern
  const finalBlock = plan.fullSequence.subarray(10, 20);
  assert(
    finalBlock[0] > 0 && finalBlock[1] > 0 && finalBlock[2] < 0 && finalBlock[3] < 0,
    `goal-directed final block, got ${Array.from(finalBlock.slice(0, 4))}`,
  );
});

await checkAsync("the lab loop encodes, plans, steps, and reports diagnostics", async () => {
  const rt = fakeRuntime();
  const lab = await createTwoRoomsLab({ runtime: rt, seed: 9, planOptions: { samples: 4, iterations: 1, horizon: 2 } });
  const first = await lab.reset();
  assert(first.historyLength === 1, "fresh episode has one encoded frame");
  assert(first.distance > 0, "distance reported");
  const snap = await lab.planStep();
  assert(snap.stepsPlanned === 1, "one planned step");
  assert(snap.envSteps === ACTION_BLOCK, "5 env steps per planned block");
  assert(snap.lastPlan.candidates === 4, "candidate costs surfaced");
  assert(typeof snap.lastPlan.cost === "number" && Number.isFinite(snap.lastPlan.cost), "finite cost");
  const snap2 = await lab.planStep();
  // planning with the fake stack can finish the episode early; otherwise history reaches the window
  assert(snap2.done || snap2.historyLength === 3, "history grows to the model window");
});

check("the tworooms route parses", () => {
  assert(parseRoute("#/tworooms").view === "tworooms", "route view");
});

const report = { total, passed: total - failures.length, failed: failures.length, failures };
console.log(JSON.stringify(report));
if (failures.length > 0) process.exit(1);
