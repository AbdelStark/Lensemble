// Node-runnable selftest for the federated demo's pure modules.
//
// Run: node web/federated-demo/selftest.mjs
// Prints a JSON result line and exits non-zero on any failure. Driven by
// tests/ml/test_federated_demo_app.py so the JS lifecycle contract is part of
// the normal pytest gate (skipped when node is unavailable).

import {
  EVENT_KINDS,
  PARTICIPANT_STATES,
  RUN_STATES,
  canTransitionParticipant,
  canTransitionRun,
} from "./lifecycle.mjs";
import {
  activeElementBlocksAutoRefresh,
  shouldDeferAutoRefresh,
} from "./auto_refresh.mjs";
import { buildJoinUrl, isValidRunId, parseRoute } from "./join_url.mjs";
import { BackendClient } from "./api_client.mjs";
import {
  LEARNER_RUNTIME,
  UPDATE_SCHEMA,
  computeSurrogateUpdate,
  simulatorUpdateArtifact,
} from "./browser_learner.mjs";
import {
  initialInferenceState,
  modelIdentity,
  modelLoadFailureMessage,
  noModelMetrics,
  selectRunInferenceArtifact,
  stepEnvironment,
  summarizeInference,
} from "./inference_panel.mjs";
import { mulberry32, newRunId, RUN_ID_PATTERN } from "./rng.mjs";
import {
  SIMULATOR_MODE,
  SimError,
  abortSimRun,
  activeParticipants,
  createSimRun,
  dropSimParticipant,
  failSimRun,
  joinSimRun,
  simTick,
  startSimRun,
} from "./sim_engine.mjs";
import { HostBus, ParticipantBus, memoryAdapters } from "./local_bus.mjs";
import qrcode from "./vendor/qrcode.mjs";

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
  if (!cond) throw new Error(message ?? "assertion failed");
}

function assertEqual(actual, expected, message) {
  if (actual !== expected) {
    throw new Error(`${message ?? "assertEqual"}: expected ${expected}, got ${actual}`);
  }
}

function assertThrowsCode(fn, code) {
  try {
    fn();
  } catch (error) {
    if (error instanceof SimError && error.code === code) return;
    throw new Error(`expected SimError(${code}), got ${error}`);
  }
  throw new Error(`expected SimError(${code}), nothing thrown`);
}

// --- run id + join URL ---

check("run id format is deterministic and valid", () => {
  const rng = mulberry32(42);
  const id = newRunId(rng);
  assert(RUN_ID_PATTERN.test(id), `bad run id: ${id}`);
  const rng2 = mulberry32(42);
  assertEqual(newRunId(rng2), id, "run id not deterministic for fixed seed");
});

check("join URL round-trips through the route parser", () => {
  const url = buildJoinUrl("https://demo.local/app/index.html", "run-abc123de", "tok42");
  const hash = url.split("#")[1];
  const route = parseRoute(`#${hash}`);
  assertEqual(route.view, "join");
  assertEqual(route.runId, "run-abc123de");
  assertEqual(route.token, "tok42");
  const noToken = parseRoute(buildJoinUrl("http://x/", "run-zzz").split("#")[1]);
  assertEqual(noToken.view, "join");
  assertEqual(noToken.token, null);
});

check("backend client keeps the documented local API base path", () => {
  const client = new BackendClient("/demo-api/");
  assertEqual(client.basePath, "/demo-api");
});

check("route parser handles host/admin/home/unknown", () => {
  assertEqual(parseRoute("").view, "home");
  assertEqual(parseRoute("#/host/run-abc").view, "host");
  assertEqual(parseRoute("#/admin/run-abc").view, "admin");
  assertEqual(parseRoute("#/nope").view, "unknown");
});

check("isValidRunId rejects junk", () => {
  assert(!isValidRunId("run-UPPER!!"), "accepted invalid id");
  assert(!isValidRunId(""), "accepted empty id");
});

check("auto refresh defers while hidden, unfocused, or editing form fields", () => {
  assert(shouldDeferAutoRefresh({ documentHidden: true }), "hidden document must defer");
  assert(
    shouldDeferAutoRefresh({ documentHasFocus: false }),
    "unfocused document must defer",
  );
  assert(
    activeElementBlocksAutoRefresh({ tagName: "INPUT" }),
    "input focus must block DOM replacement",
  );
  assert(
    activeElementBlocksAutoRefresh({ tagName: "textarea" }),
    "textarea focus must block DOM replacement",
  );
  assert(
    activeElementBlocksAutoRefresh({ tagName: "select" }),
    "select focus must block DOM replacement",
  );
  assert(
    activeElementBlocksAutoRefresh({ tagName: "div", isContentEditable: true }),
    "contenteditable focus must block DOM replacement",
  );
  assert(!activeElementBlocksAutoRefresh({ tagName: "BUTTON" }), "buttons should not freeze refreshes");
  assert(
    !shouldDeferAutoRefresh({
      documentHidden: false,
      documentHasFocus: true,
      activeElement: { tagName: "BODY" },
    }),
    "focused non-editing document should refresh",
  );
});

// --- lifecycle vocabulary ---

check("lifecycle states cover the #295 acceptance set", () => {
  for (const state of [
    "created",
    "joining",
    "ready",
    "running_round",
    "aggregating",
    "checkpoint_ready",
    "inference_ready",
    "aborted",
    "failed",
  ]) {
    assert(RUN_STATES.includes(state), `missing run state ${state}`);
  }
  for (const state of ["joined", "ready", "training", "submitted", "completed", "dropped", "error"]) {
    assert(PARTICIPANT_STATES.includes(state), `missing participant state ${state}`);
  }
  assert(EVENT_KINDS.includes("round.closed"), "missing round.closed event kind");
});

check("invalid transitions are rejected", () => {
  assert(!canTransitionRun("created", "running_round"), "created -> running_round must be invalid");
  assert(!canTransitionRun("completed", "running_round"), "terminal state must not transition");
  assert(!canTransitionParticipant("joined", "submitted"), "joined -> submitted must be invalid");
});

// --- simulator engine ---

function makeRun(seed = 7, config = {}) {
  return createSimRun(
    { maxParticipants: 3, quorum: 2, rounds: 2, preset: "swipe-dot-tiny", ...config },
    seed,
  );
}

check("run creation emits run.created and starts in created", () => {
  const run = makeRun();
  assertEqual(run.state, "created");
  assertEqual(run.mode, SIMULATOR_MODE);
  assertEqual(run.events[0].kind, "run.created");
  assert(run.events[0].simulated === true, "events must be marked simulated");
});

check("config validation rejects bad quorum", () => {
  assertThrowsCode(
    () => createSimRun({ maxParticipants: 2, quorum: 5, rounds: 1 }, 1),
    "invalid_config",
  );
});

check("participant limit is enforced", () => {
  const run = makeRun();
  joinSimRun(run);
  joinSimRun(run);
  joinSimRun(run);
  assertThrowsCode(() => joinSimRun(run), "run_full");
  assert(
    run.events.some((e) => e.kind === "participant.rejected"),
    "rejection must be on the timeline",
  );
});

check("quorum moves the run to ready", () => {
  const run = makeRun();
  joinSimRun(run);
  assertEqual(run.state, "joining");
  joinSimRun(run);
  assertEqual(run.state, "ready");
});

check("start before quorum is rejected", () => {
  const run = makeRun();
  joinSimRun(run);
  assertThrowsCode(() => startSimRun(run), "not_ready");
});

check("join after start is rejected", () => {
  const run = makeRun();
  joinSimRun(run);
  joinSimRun(run);
  startSimRun(run);
  assertThrowsCode(() => joinSimRun(run), "already_started");
});

check("full mock round path reaches inference_ready then completed", () => {
  const run = makeRun();
  joinSimRun(run);
  joinSimRun(run);
  startSimRun(run);
  assertEqual(run.state, "running_round");
  assertEqual(run.round, 1);
  let guard = 0;
  while (simTick(run) && guard < 100) guard += 1;
  assert(guard < 100, "simulator did not terminate");
  assertEqual(run.state, "completed");
  const visited = new Set(run.events.map((e) => e.runState));
  for (const state of ["running_round", "aggregating", "checkpoint_ready", "inference_ready"]) {
    assert(visited.has(state), `timeline never visited ${state}`);
  }
  assertEqual(run.artifacts.filter((a) => a.kind === "checkpoint").length, 2);
  assertEqual(run.artifacts.filter((a) => a.kind === "inference-model").length, 1);
  assert(run.artifacts.every((a) => a.simulated === true), "artifacts must be marked simulated");
  const completedParticipants = run.participants.filter((p) => p.state === "completed");
  assertEqual(completedParticipants.length, 2, "participants must finish completed");
});

check("simulated timeline is deterministic for a fixed seed", () => {
  const a = makeRun(99);
  const b = makeRun(99);
  for (const run of [a, b]) {
    joinSimRun(run);
    joinSimRun(run);
    startSimRun(run);
    while (simTick(run)) { /* drain */ }
  }
  assertEqual(a.id, b.id, "run ids diverged");
  assertEqual(
    JSON.stringify(a.artifacts.map((x) => x.sha256)),
    JSON.stringify(b.artifacts.map((x) => x.sha256)),
    "artifact hashes diverged",
  );
  assertEqual(a.events.length, b.events.length, "event counts diverged");
});

check("abort works from a live round and is terminal", () => {
  const run = makeRun();
  joinSimRun(run);
  joinSimRun(run);
  startSimRun(run);
  simTick(run);
  abortSimRun(run, "host clicked abort");
  assertEqual(run.state, "aborted");
  assertThrowsCode(() => abortSimRun(run), "run_closed");
  assert(!simTick(run), "tick after abort must be a no-op");
});

check("failure marks live participants as error", () => {
  const run = makeRun();
  joinSimRun(run);
  joinSimRun(run);
  startSimRun(run);
  failSimRun(run, "simulated crash");
  assertEqual(run.state, "failed");
  assert(
    run.participants.every((p) => p.state === "error"),
    "live participants must be marked error",
  );
});

check("dropped participants stop counting toward active set", () => {
  const run = makeRun();
  const p1 = joinSimRun(run);
  joinSimRun(run);
  dropSimParticipant(run, p1.id, "tab closed");
  assertEqual(activeParticipants(run).length, 1);
});

// --- browser learner contract ---

check("browser surrogate update contains only metadata", () => {
  const artifact = computeSurrogateUpdate({
    runId: "run-abc123de",
    participantId: "browser-abc123",
    round: 1,
    seed: 123,
  });
  assertEqual(artifact.schema, UPDATE_SCHEMA);
  assertEqual(artifact.runtime, LEARNER_RUNTIME);
  assertEqual(artifact.source, "browser-local-surrogate");
  assertEqual(artifact.simulated, false);
  assertEqual(artifact.shape.length, 1);
  assertEqual(artifact.hash.length, 64);
  const encoded = JSON.stringify(artifact);
  for (const forbidden of ["observations", "actions", "latents", "weights"]) {
    assert(!encoded.includes(forbidden), `artifact leaked ${forbidden}`);
  }
});

check("browser surrogate update is deterministic for a fixed task", () => {
  const task = { runId: "run-abc123de", participantId: "browser-abc123", round: 2, seed: 77 };
  assertEqual(computeSurrogateUpdate(task).hash, computeSurrogateUpdate(task).hash);
});

check("simulator update artifact is visibly distinguished", () => {
  const artifact = simulatorUpdateArtifact({
    runId: "run-abc123de",
    participantId: "browser-abc123",
    round: 1,
    seed: 5,
  });
  assertEqual(artifact.source, "simulator");
  assertEqual(artifact.simulated, true);
});

// --- inference helpers ---

check("inference panel selects run-produced inference artifacts", () => {
  const artifact = {
    kind: "inference-model",
    schema: "demo-inference-artifact/1",
    modelId: "lensemble-demo/run-abc123de",
    revision: "abc123",
    sourceCheckpoint: "f".repeat(64),
  };
  const selected = selectRunInferenceArtifact({ artifacts: [{ kind: "checkpoint" }, artifact] });
  assertEqual(selected, artifact);
  const identity = modelIdentity(selected);
  assertEqual(identity.schema, "demo-inference-artifact/1");
  assert(identity.source.includes("checkpoint"), "identity should name checkpoint source");
});

check("inference panel supports no-model environment stepping", () => {
  const state = initialInferenceState();
  const next = stepEnvironment(state, [1, -1]);
  assert(next.x > state.x, "x did not move");
  assert(next.y < state.y, "y did not move");
  const metrics = noModelMetrics(next);
  assert(metrics.status.includes("no ONNX model"), "no-model status missing");
});

check("inference summaries expose prediction dimensions and load errors", () => {
  const summary = summarizeInference(
    { x: 0.1, y: 0.2 },
    { predicted_tokens: { dims: [1, 4, 16] } },
    12.34,
  );
  assertEqual(summary.predicted, "predicted_tokens=1x4x16");
  assertEqual(summary.latencyMs, 12.3);
  assert(modelLoadFailureMessage(new Error("bad file")).includes("bad file"));
});

// --- multi-tab bus ---

check("participant tab can join through the host bus", async () => {
  const adapters = memoryAdapters();
  const run = makeRun(5);
  const host = new HostBus(
    run,
    (intent) => {
      if (intent.type === "join") {
        try {
          const participant = joinSimRun(run, intent.displayName ?? null);
          return { ok: true, participantId: participant.id };
        } catch (error) {
          return { ok: false, code: error.code ?? "error", message: error.message };
        }
      }
      return { ok: false, code: "unknown_intent" };
    },
    adapters,
  );
  let snapshots = 0;
  const participantTab = new ParticipantBus(run.id, () => (snapshots += 1), adapters);
  const reply = await participantTab.send({ type: "join", displayName: "tab-2" });
  assert(reply.ok, `join over bus failed: ${JSON.stringify(reply)}`);
  assertEqual(run.participants.length, 1);
  assert(snapshots > 0, "participant tab never received a snapshot");
  host.close();
  participantTab.close();
});

// --- QR vendor ---

check("vendored QR encoder produces modules for a join URL", () => {
  const qr = qrcode(0, "M");
  qr.addData(buildJoinUrl("https://localhost:8000/web/federated-demo/", "run-abc123de"));
  qr.make();
  assert(qr.getModuleCount() > 20, "QR module count too small");
  assert(typeof qr.isDark(0, 0) === "boolean", "QR isDark not functional");
});

// Async checks above push failures synchronously before this resolves because
// memory adapters deliver messages synchronously; flush microtasks anyway.
await new Promise((resolve) => setTimeout(resolve, 50));

const result = { total, passed: total - failures.length, failed: failures.length, failures };
console.log(JSON.stringify(result));
if (failures.length > 0) {
  process.exit(1);
}
