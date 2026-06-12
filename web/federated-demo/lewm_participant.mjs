// Real-lewm-tworooms participant round driver (#321, epic #314).
//
// One autonomous round of browser-local Tapestry-like work: load the checkpoint-backed runtime
// (cached across rounds), collect resident TwoRooms rollouts, train the bounded adapter, build
// the lewm-adapter-delta/1 artifact, and submit it. Injectable client/runtime so node selftests
// drive the exact flow the browser runs. There is NO fallback to the surrogate learner: when the
// runtime is unavailable the error surfaces to the participant UI and the coordinator sees no
// update from this browser.

import { runLocalAdapterContinuation } from "./lewm_local_trainer.mjs";
import { buildAdapterDeltaArtifact } from "./lewm_delta_artifact.mjs";

let cachedRuntimePromise = null;

export function cacheLewmRuntime(loadRuntime) {
  if (!cachedRuntimePromise) cachedRuntimePromise = loadRuntime();
  return cachedRuntimePromise;
}

export function resetLewmRuntimeCache() {
  cachedRuntimePromise = null;
}

// Auto-mode work budget per round: small enough for a phone browser round, real enough to move
// the loss. All steps are real optimizer steps — progress reflects actual work, never sleeps.
export const REAL_ROUND_DEFAULTS = Object.freeze({
  episodes: 1,
  maxModelSteps: 8,
  trainSteps: 40,
  batchSize: 16,
  clipNorm: 3.0,
});

export async function runRealLewmRound({
  run,
  me,
  participantToken,
  client,
  loadRuntime,
  seed,
  participantMode = "auto",
  onProgress = () => {},
  budget = REAL_ROUND_DEFAULTS,
}) {
  if (run.runMode !== "real-lewm-tworooms") {
    throw new Error(`not a real-lewm run (runMode=${run.runMode})`);
  }
  if (!run.lewmBinding) {
    throw new Error("run snapshot carries no lewmBinding");
  }
  await client.progress(run.id, me.id, participantToken, 0.05);
  onProgress(0.05, { phase: "loading-runtime" });
  const runtime = await cacheLewmRuntime(loadRuntime);

  const result = await runLocalAdapterContinuation({
    runtime,
    seed,
    episodes: budget.episodes,
    maxModelSteps: budget.maxModelSteps,
    trainSteps: budget.trainSteps,
    batchSize: budget.batchSize,
    clipNorm: budget.clipNorm,
    // the trainable subset is fixed by the run binding — a mismatched local adapter would be
    // rejected server-side as an adapterSpec violation
    adapterHidden: run.lewmBinding.adapterHiddenDim ?? 32,
    onProgress: (progress, telemetry) => {
      onProgress(0.05 + progress * 0.85, telemetry);
      void client
        .progress(run.id, me.id, participantToken, 0.05 + progress * 0.85)
        .catch(() => {});
    },
  });

  const artifact = await buildAdapterDeltaArtifact({
    result,
    runId: run.id,
    participantId: me.id,
    round: run.round,
    modelRevisionId: run.currentModelRevisionId ?? "initial",
    binding: run.lewmBinding,
    participantMode,
    seed,
  });
  onProgress(0.95, { phase: "submitting", metrics: result.metrics });
  await client.submitUpdate(run.id, me.id, participantToken, artifact);
  onProgress(1, { phase: "submitted", metrics: result.metrics });
  return { artifact, metrics: result.metrics };
}
