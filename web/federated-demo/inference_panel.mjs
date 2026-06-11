// Inference-panel helpers for the federated browser demo (#300).

import {
  DEFAULT_SIZE,
  renderSwipeDotRGBA,
  rgbaToNchwFloat,
  stepSwipeDot,
} from "../dynamic-env-demo/swipe_dot_core.mjs";

export function initialInferenceState() {
  return { x: 0.25, y: 0.55 };
}

export function selectRunInferenceArtifact(run) {
  return (run?.artifacts ?? []).find((artifact) => artifact.kind === "inference-model") ?? null;
}

export function modelIdentity(artifact) {
  if (!artifact) {
    return {
      modelId: "none",
      revision: "none",
      schema: "none",
      runtime: "none",
      source: "no run artifact selected",
    };
  }
  return {
    modelId: artifact.modelId ?? artifact.label ?? "unknown",
    revision: artifact.revision ?? String(artifact.sha256 ?? "").slice(0, 12),
    schema: artifact.schema ?? "unknown",
    runtime: artifact.runtime ?? "metadata",
    source: artifact.sourceCheckpoint ? `checkpoint ${artifact.sourceCheckpoint.slice(0, 12)}` : artifact.source,
  };
}

export function stepEnvironment(state, action) {
  return stepSwipeDot(state, action);
}

export function noModelMetrics(state) {
  return {
    stateText: `state=(${state.x.toFixed(3)}, ${state.y.toFixed(3)})`,
    status: "Environment step only; no tiny JS or ONNX model is loaded.",
    predicted: null,
    latencyMs: null,
  };
}

export function canRunTinyRevision(artifact) {
  return (
    artifact?.kind === "inference-model" &&
    artifact.runtime === "tiny-js-vector-v1" &&
    Array.isArray(artifact.vector) &&
    artifact.vector.length >= 4
  );
}

export function runTinyRevisionStep(artifact, state, action, now = () => performance.now()) {
  const started = now();
  const vector = artifact.vector;
  const adjusted = [
    Math.max(-1, Math.min(1, action[0] + vector[0] * 0.5 + vector[2])),
    Math.max(-1, Math.min(1, action[1] + vector[1] * 0.5 + vector[3])),
  ];
  const nextState = stepSwipeDot(state, adjusted);
  return {
    state: nextState,
    metrics: {
      stateText: `state=(${nextState.x.toFixed(3)}, ${nextState.y.toFixed(3)})`,
      status: "Tiny JS model revision inference completed.",
      predicted: `adjusted_action=(${adjusted[0].toFixed(3)}, ${adjusted[1].toFixed(3)})`,
      latencyMs: Number((now() - started).toFixed(1)),
    },
  };
}

export function summarizeInference(state, outputs, latencyMs) {
  const predicted = outputs?.predicted_tokens;
  return {
    stateText: `state=(${state.x.toFixed(3)}, ${state.y.toFixed(3)})`,
    status: "Inference completed.",
    predicted: predicted?.dims ? `predicted_tokens=${predicted.dims.join("x")}` : "predicted_tokens=unknown",
    latencyMs: Number(latencyMs.toFixed(1)),
  };
}

export function modelLoadFailureMessage(error) {
  const message = error instanceof Error ? error.message : String(error);
  return `Model load failed: ${message}`;
}

export async function loadOnnxSession(file) {
  if (!globalThis.ort) {
    throw new Error("ONNX Runtime Web is unavailable");
  }
  const buffer = await file.arrayBuffer();
  return globalThis.ort.InferenceSession.create(buffer, {
    executionProviders: ["webgpu", "wasm"],
  });
}

export async function runOnnxStep(session, state, action, now = () => performance.now()) {
  const nextState = stepSwipeDot(state, action);
  const rgba = renderSwipeDotRGBA(nextState, DEFAULT_SIZE);
  const clip = rgbaToNchwFloat(rgba, DEFAULT_SIZE);
  const feeds = {
    clip: new globalThis.ort.Tensor("float32", clip, [1, 1, 3, DEFAULT_SIZE, DEFAULT_SIZE]),
    action: new globalThis.ort.Tensor("float32", Float32Array.from(action), [1, 2]),
  };
  const started = now();
  const outputs = await session.run(feeds);
  return {
    state: nextState,
    metrics: summarizeInference(nextState, outputs, now() - started),
  };
}

export function drawSwipeDot(canvas, state) {
  const ctx = canvas.getContext("2d", { alpha: false });
  if (!ctx) return;
  const rgba = renderSwipeDotRGBA(state, DEFAULT_SIZE);
  const image = new ImageData(rgba, DEFAULT_SIZE, DEFAULT_SIZE);
  ctx.imageSmoothingEnabled = false;
  ctx.putImageData(image, 0, 0);
}
