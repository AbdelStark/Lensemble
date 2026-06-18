#!/usr/bin/env node
import assert from "node:assert/strict";

import {
  buildLiveSurpriseTrajectory,
  createOnlineSurpriseEngine,
  mseOf,
  stepSurprise,
} from "./surprise_engine.mjs";

const D = 192;
const WINDOW = 3;

function vec(seed, scale = 1) {
  let state = seed >>> 0;
  const out = new Float32Array(D);
  for (let i = 0; i < D; i += 1) {
    state = (Math.imul(state ^ (state >>> 15), 1 | state) + 0x6d2b79f5) >>> 0;
    out[i] = (((state >>> 8) / 0x01000000) - 0.5) * scale;
  }
  return out;
}

const bias = new Float32Array(D);
for (let i = 0; i < D; i += 1) bias[i] = i === 0 ? 0.25 : 0;

const fakeRuntime = {
  hidden: D,
  numFrames: WINDOW,
  async encodeFrames(frameInput, batch) {
    assert.equal(batch, 1);
    return Float32Array.from(frameInput);
  },
  async embedActionBlocks(actionBlock, batch, time) {
    assert.equal(batch, 1);
    assert.equal(time, 1);
    return Float32Array.from(actionBlock);
  },
  async predictLatents(latents, actions, batch, time) {
    assert.equal(batch, 1);
    assert.equal(time, WINDOW);
    assert.equal(actions.length, D * WINDOW);
    const last = latents.subarray((WINDOW - 1) * D, WINDOW * D);
    const out = new Float32Array(D * WINDOW);
    for (let t = 0; t < WINDOW; t += 1) {
      for (let i = 0; i < D; i += 1) {
        out[t * D + i] = last[i] + bias[i];
      }
    }
    return out;
  },
};

const histLatents = [vec(1, 0.5), vec(2, 0.5), vec(3, 0.5)];
const histActionEmbeddings = [vec(11, 0.1), vec(12, 0.1), vec(13, 0.1)];
const target = Float32Array.from(histLatents[2]);
target[0] += 0.5;

const direct = await stepSurprise({
  runtime: fakeRuntime,
  histLatents,
  histActionEmbeddings,
  nextFrameInput: target,
});
const pred = Float32Array.from(histLatents[2]);
pred[0] += 0.25;
assert.equal(direct.schema, "lewm-surprise-engine/1");
assert.ok(Number.isFinite(direct.surprise));
assert.ok(Math.abs(direct.surprise - mseOf(pred, target, D)) <= 1e-8);

const online = createOnlineSurpriseEngine({ runtime: fakeRuntime });
const step0 = await online.observeTransition({
  currentFrameInput: histLatents[0],
  actionBlock: histActionEmbeddings[0],
  nextFrameInput: histLatents[1],
});
const step1 = await online.observeTransition({
  currentFrameInput: histLatents[1],
  actionBlock: histActionEmbeddings[1],
  nextFrameInput: histLatents[2],
});
const step2 = await online.observeTransition({
  currentFrameInput: histLatents[2],
  actionBlock: histActionEmbeddings[2],
  nextFrameInput: target,
});
assert.equal(step0.surprise, null);
assert.equal(step1.surprise, null);
assert.equal(step2.warmup, false);
assert.ok(Math.abs(step2.surprise - direct.surprise) <= 1e-8);

const liveRuntime = {
  hidden: D,
  numFrames: WINDOW,
  imageSize: 224,
  actionDim: 10,
  backend: "fake-wasm",
  async encodeFrames(frameInput, batch) {
    assert.equal(batch, 1);
    const out = new Float32Array(D);
    for (let i = 0; i < D; i += 1) out[i] = frameInput[(i * 31) % frameInput.length] * 0.5;
    return out;
  },
  async embedActionBlocks(actionBlock, batch, time) {
    assert.equal(batch, 1);
    assert.equal(time, 1);
    const out = new Float32Array(D);
    for (let i = 0; i < D; i += 1) out[i] = actionBlock[i % actionBlock.length] * 0.02;
    return out;
  },
  async predictLatents(latents, actions, batch, time) {
    assert.equal(batch, 1);
    assert.equal(time, WINDOW);
    const last = latents.subarray((WINDOW - 1) * D, WINDOW * D);
    const out = new Float32Array(D * WINDOW);
    for (let t = 0; t < WINDOW; t += 1) {
      for (let i = 0; i < D; i += 1) out[t * D + i] = last[i] + actions[(WINDOW - 1) * D + i] + 0.01;
    }
    return out;
  },
};

const live = await buildLiveSurpriseTrajectory({
  runtime: liveRuntime,
  steps: 8,
  perturbations: [{ step: 4, kind: "ood" }],
});
assert.equal(live.schema, "lewm-surprise-live-traj/1");
assert.equal(live.steps.length, 8);
assert.equal(live.steps[0].surprisePre, null);
assert.equal(live.steps[1].surprisePre, null);
assert.ok(Number.isFinite(live.steps[2].surprisePre));
assert.equal(live.steps[4].event, "ood");

console.log(
  JSON.stringify({
    ok: true,
    schema: "lewm-surprise-selftest/1",
    warmupSteps: 2,
    surprise: direct.surprise,
    parityTolerance: 1e-8,
  }),
);
