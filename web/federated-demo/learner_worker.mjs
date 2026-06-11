import { computeSurrogateUpdate } from "./browser_learner.mjs";

self.onmessage = (event) => {
  try {
    self.postMessage({ type: "progress", progress: 0.15, telemetry: { phase: "sampling" } });
    const artifact = computeSurrogateUpdate(event.data);
    self.postMessage({ type: "progress", progress: 0.85, telemetry: { phase: "anti-collapse check" } });
    self.postMessage({ type: "result", artifact });
  } catch (error) {
    self.postMessage({ type: "error", message: error instanceof Error ? error.message : String(error) });
  }
};
