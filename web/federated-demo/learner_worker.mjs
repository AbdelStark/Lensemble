import { computeSurrogateUpdate } from "./browser_learner.mjs";

self.onmessage = (event) => {
  try {
    self.postMessage({ type: "progress", progress: 0.2 });
    const artifact = computeSurrogateUpdate(event.data);
    self.postMessage({ type: "progress", progress: 1 });
    self.postMessage({ type: "result", artifact });
  } catch (error) {
    self.postMessage({ type: "error", message: error instanceof Error ? error.message : String(error) });
  }
};
