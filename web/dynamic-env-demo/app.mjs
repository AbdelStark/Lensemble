import {
  DEFAULT_SIZE,
  renderSwipeDotRGBA,
  rgbaToNchwFloat,
  stepSwipeDot,
} from "./swipe_dot_core.mjs";

const canvas = document.querySelector("#env");
const ctx = canvas.getContext("2d", { alpha: false });
const statusEl = document.querySelector("#status");
const modelInput = document.querySelector("#modelFile");
const actionX = document.querySelector("#actionX");
const actionY = document.querySelector("#actionY");
const runButton = document.querySelector("#run");
const resetButton = document.querySelector("#reset");
const metrics = document.querySelector("#metrics");

let state = { x: 0.25, y: 0.55 };
let session = null;

function draw() {
  const rgba = renderSwipeDotRGBA(state, DEFAULT_SIZE);
  const image = new ImageData(rgba, DEFAULT_SIZE, DEFAULT_SIZE);
  ctx.imageSmoothingEnabled = false;
  ctx.putImageData(image, 0, 0);
}

function setStatus(text) {
  statusEl.textContent = text;
}

function action() {
  return [Number(actionX.value), Number(actionY.value)];
}

async function loadModel(file) {
  if (!globalThis.ort) {
    setStatus("ONNX Runtime Web failed to load.");
    return;
  }
  const buffer = await file.arrayBuffer();
  session = await ort.InferenceSession.create(buffer, {
    executionProviders: ["webgpu", "wasm"],
  });
  setStatus(`Loaded ${file.name}`);
}

async function runStep() {
  const act = action();
  state = stepSwipeDot(state, act);
  draw();
  if (!session) {
    metrics.textContent = `state=(${state.x.toFixed(3)}, ${state.y.toFixed(3)})`;
    return;
  }

  const rgba = renderSwipeDotRGBA(state, DEFAULT_SIZE);
  const clip = rgbaToNchwFloat(rgba, DEFAULT_SIZE);
  const feeds = {
    clip: new ort.Tensor("float32", clip, [1, 1, 3, DEFAULT_SIZE, DEFAULT_SIZE]),
    action: new ort.Tensor("float32", Float32Array.from(act), [1, 2]),
  };
  const started = performance.now();
  const outputs = await session.run(feeds);
  const elapsed = performance.now() - started;
  const predicted = outputs.predicted_tokens;
  metrics.textContent = [
    `state=(${state.x.toFixed(3)}, ${state.y.toFixed(3)})`,
    `predicted_tokens=${predicted.dims.join("x")}`,
    `inference=${elapsed.toFixed(1)}ms`,
  ].join(" · ");
}

modelInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) void loadModel(file);
});

runButton.addEventListener("click", () => {
  void runStep();
});

resetButton.addEventListener("click", () => {
  state = { x: 0.25, y: 0.55 };
  draw();
  metrics.textContent = "state=(0.250, 0.550)";
});

draw();
setStatus("Load dynamic_env_world_model.onnx to run client-side inference.");
