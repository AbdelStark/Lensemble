// UI glue for the Lensemble federated run demo (#294).
//
// Modes:
//   frontend-simulator  browser-only mock, useful as the #295 tracer bullet
//   backend-api         local API served by `lensemble demo federated`
//
// Both modes render the same lifecycle vocabulary and keep claim boundaries
// visible. Backend mode accepts only bounded derived browser update artifacts.

import { backendClient } from "./api_client.mjs";
import { shouldDeferAutoRefreshForDocument } from "./auto_refresh.mjs";
import { runBrowserLearner } from "./browser_learner.mjs";
import { buildJoinUrl, parseRoute } from "./join_url.mjs";
import {
  drawSwipeDot,
  initialInferenceState,
  loadOnnxSession,
  modelIdentity,
  modelLoadFailureMessage,
  noModelMetrics,
  canRunTinyRevision,
  runTinyRevisionStep,
  runOnnxStep,
  selectRunInferenceArtifact,
  stepEnvironment,
} from "./inference_panel.mjs";
import {
  HostBus,
  ParticipantBus,
  defaultAdapters,
  loadRunSnapshot,
} from "./local_bus.mjs";
import { loadLewmRuntime } from "./lewm_runtime.mjs";
import { runRealLewmRound } from "./lewm_participant.mjs";
import { mountTwoRoomsLab } from "./tworooms_panel.mjs";
import { randomSeed } from "./rng.mjs";
import {
  abortSimRun,
  activeParticipants,
  createSimRun,
  dropSimParticipant,
  failSimRun,
  joinSimRun,
  restoreSimRun,
  simTick,
  startSimRun,
} from "./sim_engine.mjs";
import qrcode from "./vendor/qrcode.mjs";

const app = document.querySelector("#app");
const adapters = defaultAdapters();

let hostSession = null; // { run, bus, timer }
let participantSession = null; // frontend-simulator participant session
let backendPoll = null; // { view, runId, timer, socket, transport }
const inferenceByRun = new Map();
const backendLearnerJobs = new Map();
const backendLearnerTelemetry = new Map();

// ---------------------------------------------------------------- utilities

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function stateBadge(state) {
  return el("span", { class: `badge state-${state}`, text: state });
}

function note(text) {
  return el("p", { class: "note", text });
}

function errorBox(text) {
  return el("p", { class: "error-box", text });
}

function formatMetric(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  const fixed = Number(value).toFixed(digits);
  if (!fixed.includes(".")) return fixed;
  return fixed.replace(/\.?0+$/, "");
}

function latestUpdateMetadata(participant, preferredRound = null) {
  const updates = participant?.updateMetadata ?? {};
  if (preferredRound !== null && updates[String(preferredRound)]) return updates[String(preferredRound)];
  const latestRound = Object.keys(updates)
    .map((round) => Number(round))
    .filter((round) => Number.isFinite(round))
    .sort((a, b) => b - a)[0];
  return latestRound === undefined ? null : updates[String(latestRound)];
}

function learnerJobKey(run, participant) {
  return [
    run.id,
    participant.id,
    participant.round ?? run.round,
    run.currentModelRevisionId ?? "initial",
  ].join(":");
}

function learnerTelemetryPayload(progress, telemetry) {
  const payload = {
    progress,
    phase: telemetry?.phase ?? null,
    loss: telemetry?.loss ?? null,
    probe: telemetry?.probe ?? null,
    l2Norm: telemetry?.l2Norm ?? null,
    clipNorm: telemetry?.clipNorm ?? null,
    clipSaturation: telemetry?.clipSaturation ?? null,
    effectiveDim: telemetry?.effectiveDim ?? null,
    effectiveDimRatio: telemetry?.effectiveDimRatio ?? null,
    collapseRisk: telemetry?.collapseRisk ?? null,
    runtimeMs: telemetry?.runtimeMs ?? null,
    error: telemetry?.error ?? null,
  };
  return payload;
}

function metricTile(label, value, detail = null) {
  return el("div", { class: "metric-tile" }, [
    el("span", { class: "metric-label", text: label }),
    el("strong", { text: value }),
    detail ? el("span", { class: "muted", text: detail }) : null,
  ]);
}

function drawQr(canvas, text) {
  const qr = qrcode(0, "M");
  qr.addData(text);
  qr.make();
  const count = qr.getModuleCount();
  const quiet = 2;
  const total = count + quiet * 2;
  canvas.width = total;
  canvas.height = total;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#fffdf7";
  ctx.fillRect(0, 0, total, total);
  ctx.fillStyle = "#18221c";
  for (let r = 0; r < count; r += 1) {
    for (let c = 0; c < count; c += 1) {
      if (qr.isDark(r, c)) ctx.fillRect(c + quiet, r + quiet, 1, 1);
    }
  }
}

function timelineList(events) {
  const items = events
    .slice(-80)
    .reverse()
    .map((event) =>
      el("li", { class: event.severity }, [
        el("span", { text: `#${event.seq ?? event.at}` }),
        el("span", { class: "kind", text: event.kind }),
        el("span", { text: `${event.actor ?? "system"} r${event.round ?? 0}: ${event.message}` }),
      ]),
    );
  return el("ul", { class: "timeline" }, items);
}

function participantSlots(run) {
  const participants = run.participants ?? [];
  const slots = [];
  for (let i = 0; i < run.config.maxParticipants; i += 1) {
    const participant = participants[i];
    if (participant) {
      slots.push(
        el("div", { class: "slot filled" }, [
          el("span", { class: "mono", text: participant.displayName || participant.id }),
          stateBadge(participant.state),
          el("span", {
            class: "muted",
            text: [
              participant.connectionState ?? "connected",
              participant.automationMode ? `${participant.automationMode} mode` : null,
            ]
              .filter(Boolean)
              .join(" · "),
          }),
          participant.state === "training"
            ? el("progress", { max: "1", value: String(participant.progress ?? 0) })
            : null,
        ]),
      );
    } else {
      slots.push(el("div", { class: "slot" }, [el("span", { class: "muted", text: `slot ${i + 1} open` })]));
    }
  }
  return el("div", { class: "slots" }, slots);
}

function artifactList(run) {
  const artifacts = run.artifacts ?? [];
  if (artifacts.length === 0) {
    return note("No artifacts yet. Checkpoint-like artifacts appear after aggregation closes a round.");
  }
  return el(
    "ul",
    { class: "artifact-list" },
    artifacts.map((artifact) =>
      el("li", {}, [
        `${artifact.label ?? artifact.kind} - `,
        el("span", { class: "mono", text: `${String(artifact.sha256).slice(0, 16)}...` }),
        artifact.simulated
          ? " (simulated)"
          : artifact.containsTinyDerivedVector
            ? " (tiny derived vector)"
            : " (metadata)",
      ]),
    ),
  );
}

function metricsList(run) {
  const metrics = run.roundMetrics ?? [];
  if (metrics.length === 0) {
    return note("Round metrics appear after the first aggregation closes.");
  }
  const real = run.runMode === "real-lewm-tworooms";
  return el(
    "div",
    { class: "metrics-grid" },
    metrics
      .slice(-4)
      .reverse()
      .map((metric) =>
        el("div", { class: "metric-card" }, [
          el("div", { class: "metric-card-head" }, [
            el("strong", { text: `round ${metric.round}` }),
            stateBadge(
              real
                ? `loss↓ ${metric.lossDecreasedCount ?? 0}/${metric.submitted ?? 0}`
                : metric.collapseRisk ?? "watch",
            ),
          ]),
          el(
            "div",
            { class: "metric-tiles" },
            real
              ? [
                  metricTile("pred loss", formatMetric(metric.predLossLastMean, 5), `from ${formatMetric(metric.predLossFirstMean, 5)}`),
                  metricTile("sigreg", formatMetric(metric.sigregStatisticMean, 5)),
                  metricTile("eff rank", formatMetric(metric.effectiveRankMean, 2)),
                  metricTile("latent std", formatMetric(metric.latentStdMeanMean, 4)),
                  metricTile("Δ norm", formatMetric(metric.aggregateDeltaNorm, 5), `state ${formatMetric(metric.adapterStateNorm, 4)}`),
                  metricTile("clip sat", `${formatMetric((metric.clipSaturationRate ?? 0) * 100, 0)}%`),
                ]
              : [
                  metricTile("loss", formatMetric(metric.localLossMean, 4)),
                  metricTile("probe", formatMetric(metric.probeMean, 4)),
                  metricTile("eff dim", formatMetric(metric.aggregateEffectiveDim, 2), `${formatMetric((metric.aggregateEffectiveDimRatio ?? 0) * 100, 0)}%`),
                  metricTile("clip sat", `${formatMetric((metric.clipSaturationRate ?? 0) * 100, 0)}%`),
                  metricTile("runtime", metric.runtimeMsMean === null || metric.runtimeMsMean === undefined ? "n/a" : `${formatMetric(metric.runtimeMsMean, 1)} ms`),
                  metricTile("agg norm", formatMetric(metric.aggregateNorm, 4)),
                ],
          ),
          el("span", { class: "mono muted", text: metric.modelRevisionId }),
        ]),
      ),
  );
}

function trainingDiagnostics(run) {
  const rows = (run.participants ?? [])
    .map((participant) => ({ participant, metadata: latestUpdateMetadata(participant, run.round) }))
    .filter(({ metadata }) => metadata);
  if (rows.length === 0) {
    return note("Training diagnostics appear after bounded updates are submitted.");
  }
  const real = run.runMode === "real-lewm-tworooms";
  return el(
    "div",
    { class: "diagnostic-table" },
    rows.map(({ participant, metadata }) =>
      el(
        "div",
        { class: "diagnostic-row" },
        real
          ? [
              el("span", { class: "mono", text: participant.displayName || participant.id }),
              el("span", { text: `pred ${formatMetric(metadata.metrics?.predLossLast, 5)} (from ${formatMetric(metadata.metrics?.predLossFirst, 5)})` }),
              el("span", { text: `sigreg ${formatMetric(metadata.metrics?.sigregStatistic, 5)}` }),
              el("span", { text: `rank ${formatMetric(metadata.metrics?.effectiveRank, 1)}` }),
              el("span", { text: `steps ${formatMetric(metadata.metrics?.optimizerSteps, 0)}` }),
              stateBadge(metadata.metrics?.lossDecreased ? "loss↓" : "flat"),
            ]
          : [
              el("span", { class: "mono", text: participant.displayName || participant.id }),
              el("span", { text: `loss ${formatMetric(metadata.loss, 4)}` }),
              el("span", { text: `probe ${formatMetric(metadata.probe, 4)}` }),
              el("span", { text: `eff-dim ${formatMetric(metadata.effectiveDim, 2)}` }),
              el("span", { text: `clip ${formatMetric((metadata.clipSaturation ?? 0) * 100, 0)}%` }),
              stateBadge(metadata.collapseRisk ?? "watch"),
            ],
      ),
    ),
  );
}

function currentRouteStill(view, runId) {
  const route = parseRoute(window.location.hash);
  return route.view === view && route.runId === runId;
}

async function refreshBackendRoute(view, runId) {
  if (!currentRouteStill(view, runId)) {
    clearBackendPoll();
    return;
  }
  const poll = backendPoll;
  if (poll?.transport === "websocket") return;
  if (poll?.refreshing) return;
  if (shouldDeferAutoRefreshForDocument()) return;
  if (poll) poll.refreshing = true;
  try {
    const run = await backendClient.getRun(runId);
    if (!currentRouteStill(view, runId) || shouldDeferAutoRefreshForDocument()) return;
    app.replaceChildren();
    if (view === "host") renderBackendHostSnapshot(run);
    else if (view === "join") renderBackendJoinSnapshot(parseRoute(window.location.hash), run);
  } catch {
    if (currentRouteStill(view, runId) && !shouldDeferAutoRefreshForDocument()) render();
  } finally {
    if (backendPoll === poll && poll) poll.refreshing = false;
  }
}

function backendSocketClosed(socket) {
  return !socket || socket.readyState >= 2;
}

function attachBackendSocket(poll) {
  if (!backendSocketClosed(poll.socket)) return;
  const { view, runId, streamOptions } = poll;
  const socket = backendClient.connectRun(runId, {
    role: streamOptions.role,
    participantId: streamOptions.participantId,
    participantToken: streamOptions.participantToken,
    after: poll.lastSeq,
    onOpen: () => {
      if (backendPoll === poll) poll.transport = "websocket";
    },
    onClose: () => {
      if (backendPoll === poll) {
        poll.transport = "polling";
        if (poll.socket === socket) poll.socket = null;
      }
    },
    onMessage: (message) => {
      if (backendPoll !== poll || !currentRouteStill(view, runId)) return;
      const events = message.events ?? [];
      if (events.length > 0) {
        poll.lastSeq = Math.max(poll.lastSeq, ...events.map((event) => event.seq ?? -1));
      }
      const run = message.run;
      if (run && !shouldDeferAutoRefreshForDocument()) {
        app.replaceChildren();
        if (view === "host") renderBackendHostSnapshot(run);
        else if (view === "join") renderBackendJoinSnapshot(parseRoute(window.location.hash), run);
      } else if (events.length > 0) {
        void refreshBackendRoute(view, runId);
      }
    },
    onError: () => {
      if (backendPoll === poll) poll.transport = "polling";
    },
  });
  poll.socket = socket;
}

function clearBackendPoll() {
  if (backendPoll) {
    clearInterval(backendPoll.timer);
    backendPoll.socket?.close();
    backendPoll = null;
  }
}

function ensureBackendPoll(view, runId, streamOptions = {}) {
  const normalizedOptions = {
    role: streamOptions.role ?? "host",
    participantId: streamOptions.participantId ?? null,
    participantToken: streamOptions.participantToken ?? null,
  };
  const streamKey = JSON.stringify({
    view,
    runId,
    role: normalizedOptions.role,
    participantId: normalizedOptions.participantId,
  });
  if (backendPoll?.streamKey === streamKey) {
    attachBackendSocket(backendPoll);
    return;
  }
  clearBackendPoll();
  backendPoll = {
    view,
    runId,
    streamKey,
    streamOptions: normalizedOptions,
    transport: "polling",
    lastSeq: -1,
    refreshing: false,
    socket: null,
    timer: setInterval(() => {
      void refreshBackendRoute(view, runId);
    }, 1000),
  };
  attachBackendSocket(backendPoll);
}

function downloadJson(filename, value) {
  const blob = new Blob([JSON.stringify(value, null, 2)], { type: "application/json" });
  const link = el("a", {
    href: URL.createObjectURL(blob),
    download: filename,
  });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 2000);
}

// ------------------------------------------------------------------- router

function render() {
  const route = parseRoute(window.location.hash);
  app.replaceChildren();
  if (route.view === "home") renderHome();
  else if (route.view === "host") renderHost(route.runId);
  else if (route.view === "join") renderJoin(route);
  else if (route.view === "tworooms") renderTwoRoomsLab();
  else {
    clearBackendPoll();
    app.append(
      el("section", { class: "panel" }, [
        el("h2", { text: "Unknown route" }),
        note(`No view for #${route.path ?? ""}.`),
        el("p", {}, el("a", { href: "#/", text: "Back to home" })),
      ]),
    );
  }
}

// ------------------------------------------------------- tworooms LeWM lab

function renderTwoRoomsLab() {
  clearBackendPoll();
  teardownHost();
  teardownParticipant();
  app.append(
    el("section", { class: "panel" }, [
      el("p", {}, el("a", { href: "#/", text: "← Back to home" })),
    ]),
  );
  mountTwoRoomsLab(app, { loadRuntime: () => loadLewmRuntime(), el });
}

// --------------------------------------------------------------------- home

function renderHome() {
  clearBackendPoll();
  teardownHost();
  teardownParticipant();

  const modeSelect = el("select", {}, [
    el("option", { value: "backend-api", text: "backend API mode (local server)" }),
    el("option", { value: "frontend-simulator", text: "frontend-only simulator" }),
  ]);
  const maxInput = el("input", { type: "number", min: "1", max: "64", value: "4" });
  const quorumInput = el("input", { type: "number", min: "1", max: "64", value: "2" });
  const roundsInput = el("input", { type: "number", min: "1", max: "1000", value: "1000" });
  const presetSelect = el("select", {}, [
    el("option", { value: "swipe-dot-tiny", text: "swipe-dot-tiny (synthetic dynamic env)" }),
  ]);
  const runModeSelect = el("select", {}, [
    el("option", {
      value: "surrogate-swipe-dot",
      text: "surrogate-swipe-dot (educational tiny-vector path)",
    }),
    el("option", {
      value: "real-lewm-tworooms",
      text: "real-lewm-tworooms (Tapestry-like checkpoint adaptation)",
    }),
  ]);
  const errorNote = el("p", { class: "note" });

  const createButton = el("button", {
    text: "Create run",
    onclick: async () => {
      const config = {
        maxParticipants: Number(maxInput.value),
        quorum: Number(quorumInput.value),
        rounds: Number(roundsInput.value),
        preset: presetSelect.value,
        mode: runModeSelect.value,
      };
      try {
        if (modeSelect.value === "frontend-simulator") {
          if (config.mode === "real-lewm-tworooms") {
            errorNote.textContent =
              "real-lewm-tworooms needs the backend API mode (the simulator has no checkpoint runtime)";
            return;
          }
          delete config.mode;
          const run = createSimRun(config, randomSeed());
          adoptHostRun(run);
          window.location.hash = `#/host/${run.id}`;
          return;
        }
        const run = await backendClient.createRun(config);
        window.location.hash = `#/host/${run.id}`;
      } catch (error) {
        errorNote.textContent = error.message;
      }
    },
  });

  app.append(
    el("section", { class: "panel" }, [
      el("h2", { text: "Host a federated browser demo run" }),
      note(
        "Choose the backend for real browser sessions, WebSocket orchestration, bounded tiny update submission, aggregation, inference attachment, and evidence export. The simulator keeps the first slice available without a server.",
      ),
      el("div", { class: "columns" }, [
        el("div", { class: "panel" }, [
          el("label", {}, ["Mode", modeSelect]),
          el("label", {}, ["Learner path", runModeSelect]),
          el("label", {}, ["Max participants", maxInput]),
          el("label", {}, ["Quorum (min trainers to start)", quorumInput]),
          el("label", {}, ["Rounds", roundsInput]),
          el("label", {}, ["Demo preset", presetSelect]),
          createButton,
          errorNote,
        ]),
        el("div", { class: "panel" }, [
          el("h2", { text: "Claim boundary" }),
          note(
            "This is an educational systems demo of Tapestry-like federated JEPA orchestration. It is not a benchmark win over local-only, not production browser training, not a cryptographic honest-computation proof, and not physical SO-100 success.",
          ),
          note(
            "One-command backend path: uv run lensemble demo federated --port 8765, then open the printed URL.",
          ),
          el("p", {}, [
            el("a", {
              href: "#/tworooms",
              text: "TwoRooms LeWM lab → checkpoint-backed rollout & planning (real-lewm-tworooms component)",
            }),
          ]),
        ]),
      ]),
    ]),
  );
}

// ------------------------------------------------------------ local host mode

function adoptHostRun(run) {
  teardownHost();
  const bus = new HostBus(run, (intent) => applyHostIntent(run, intent), adapters);
  hostSession = { run, bus, timer: null };
}

function applyHostIntent(run, intent) {
  try {
    if (intent.type === "join") {
      const participant = joinSimRun(run, intent.displayName ?? null);
      scheduleHostRefresh();
      return { ok: true, participantId: participant.id };
    }
    if (intent.type === "leave") {
      dropSimParticipant(run, intent.participantId, "participant left");
      scheduleHostRefresh();
      return { ok: true };
    }
    return { ok: false, code: "unknown_intent", message: `unknown intent ${intent.type}` };
  } catch (error) {
    scheduleHostRefresh();
    return { ok: false, code: error.code ?? "error", message: error.message };
  }
}

function scheduleHostRefresh() {
  setTimeout(() => {
    if (
      parseRoute(window.location.hash).view === "host"
      && !shouldDeferAutoRefreshForDocument()
    ) {
      render();
    }
  }, 0);
}

function teardownHost() {
  if (hostSession) {
    if (hostSession.timer) clearInterval(hostSession.timer);
    hostSession.bus.close();
    hostSession = null;
  }
}

function ensureHostTicker() {
  if (!hostSession || hostSession.timer) return;
  hostSession.timer = setInterval(() => {
    const { run, bus } = hostSession;
    if (["running_round", "aggregating", "checkpoint_ready", "inference_ready"].includes(run.state)) {
      simTick(run);
      bus.publish();
      if (
        parseRoute(window.location.hash).view === "host"
        && !shouldDeferAutoRefreshForDocument()
      ) {
        render();
      }
    }
  }, 700);
}

function renderHost(runId) {
  teardownParticipant();
  if (hostSession?.run.id === runId) {
    clearBackendPoll();
    renderLocalHost(runId);
    return;
  }
  const snapshot = loadRunSnapshot(adapters.storage, runId);
  const restored = restoreSimRun(snapshot);
  if (restored) {
    adoptHostRun(restored);
    clearBackendPoll();
    renderLocalHost(runId);
    return;
  }
  renderBackendHost(runId);
}

function renderLocalHost(runId) {
  const { run, bus } = hostSession;
  ensureHostTicker();

  const joinUrl = buildJoinUrl(window.location.href, run.id);
  const qrCanvas = el("canvas", { id: "qr", "aria-label": "Join URL QR code" });
  const urlInput = el("input", { type: "text", readonly: "readonly", value: joinUrl });
  const copyButton = el("button", {
    class: "secondary",
    text: "Copy",
    onclick: () => navigator.clipboard?.writeText(joinUrl),
  });

  const active = activeParticipants(run);
  const startButton = el("button", {
    text: run.state === "ready" ? "Start run" : `Start run (${active.length}/${run.config.quorum})`,
    onclick: () => {
      try {
        startSimRun(run);
        bus.publish();
        render();
      } catch (error) {
        window.alert(error.message);
      }
    },
  });
  if (run.state !== "ready") startButton.disabled = true;

  const abortButton = el("button", {
    class: "danger",
    text: "Abort run",
    onclick: () => {
      try {
        abortSimRun(run, "host abort");
        bus.publish();
        render();
      } catch (error) {
        window.alert(error.message);
      }
    },
  });
  if (["completed", "aborted", "failed"].includes(run.state)) abortButton.disabled = true;

  const failButton = el("button", {
    class: "secondary",
    text: "Simulate failure",
    onclick: () => {
      try {
        failSimRun(run, "simulated infrastructure failure");
        bus.publish();
        render();
      } catch (error) {
        window.alert(error.message);
      }
    },
  });
  if (["completed", "aborted", "failed"].includes(run.state)) failButton.disabled = true;

  app.append(
    el("section", { class: "panel" }, [
      el("h2", {}, [`Host dashboard - ${runId} `, stateBadge(run.state)]),
      note(`Mode: ${run.mode}. Quorum ${run.config.quorum}, max ${run.config.maxParticipants}, rounds ${run.config.rounds}.`),
      el("div", { class: "columns" }, [
        el("div", { class: "panel" }, [
          el("h2", { text: "Invite participants" }),
          qrCanvas,
          el("div", { class: "join-url" }, [urlInput, copyButton]),
          note("Frontend-only mode shares state between tabs of the same browser."),
          startButton,
          abortButton,
          failButton,
          el("p", {}, el("a", { href: "#/", text: "New run" })),
        ]),
        el("div", { class: "panel" }, [
          el("h2", { text: run.round > 0 ? `Round ${run.round} of ${run.config.rounds}` : "Waiting for participants" }),
          participantSlots(run),
          el("h2", { text: "Artifacts" }),
          artifactList(run),
          el("h2", { text: "Event timeline (simulated)" }),
          timelineList(run.events),
        ]),
      ]),
    ]),
  );
  drawQr(qrCanvas, joinUrl);
}

// ---------------------------------------------------------- backend host mode

async function renderBackendHost(runId) {
  if (backendPoll?.view !== "host" || backendPoll?.runId !== runId) clearBackendPoll();
  app.append(el("section", { class: "panel" }, [el("h2", { text: "Loading backend run..." })]));
  try {
    const run = await backendClient.getRun(runId);
    if (!currentRouteStill("host", runId)) return;
    app.replaceChildren();
    ensureBackendPoll("host", runId, { role: "host" });
    renderBackendHostSnapshot(run);
  } catch (error) {
    if (!currentRouteStill("host", runId)) return;
    app.replaceChildren(
      el("section", { class: "panel" }, [
        el("h2", { text: "Run not found" }),
        errorBox(error.message),
        note("Create a run from this browser, or start the local backend with uv run lensemble demo federated."),
        el("p", {}, el("a", { href: "#/", text: "Create a run" })),
      ]),
    );
  }
}

function renderBackendHostSnapshot(run) {
  ensureBackendPoll("host", run.id, { role: "host" });
  const joinUrl = run.joinUrl || buildJoinUrl(window.location.href, run.id, run.joinToken);
  const qrCanvas = el("canvas", { id: "qr", "aria-label": "Join URL QR code" });
  const urlInput = el("input", { type: "text", readonly: "readonly", value: joinUrl });
  const statusNote = el("p", { class: "note" });
  const startButton = el("button", {
    text: run.controls.canStart ? "Start run" : `Start run (${run.participants.length}/${run.config.quorum})`,
    onclick: async () => {
      try {
        await backendClient.control(run.id, "start");
        render();
      } catch (error) {
        statusNote.textContent = error.message;
      }
    },
  });
  startButton.disabled = !run.controls.canStart;

  const abortButton = el("button", {
    class: "danger",
    text: "Abort run",
    onclick: async () => {
      try {
        await backendClient.control(run.id, "abort", { reason: "host abort" });
        render();
      } catch (error) {
        statusNote.textContent = error.message;
      }
    },
  });
  abortButton.disabled = !run.controls.canAbort;

  const timeoutButton = el("button", {
    class: "secondary",
    text: "Drop timed-out participants",
    onclick: async () => {
      try {
        await backendClient.control(run.id, "timeout-missing", { reason: "host timeout" });
        render();
      } catch (error) {
        statusNote.textContent = error.message;
      }
    },
  });
  timeoutButton.disabled = run.state !== "running_round";

  const exportButton = el("button", {
    class: "secondary",
    text: "Export evidence JSON",
    onclick: async () => {
      try {
        const evidence = await backendClient.exportEvidence(run.id);
        downloadJson(`${run.id}-evidence.json`, evidence);
      } catch (error) {
        statusNote.textContent = error.message;
      }
    },
  });

  app.append(
    el("section", { class: "panel" }, [
      el("h2", {}, [
        `Host dashboard - ${run.id} `,
        stateBadge(run.state),
        " ",
        stateBadge(run.runMode ?? "surrogate-swipe-dot"),
      ]),
      note(
        `Mode: ${run.mode}; learner path: ${run.runMode ?? "surrogate-swipe-dot"}; transport: ${backendPoll?.transport ?? run.deployment?.transportMode ?? "polling"}; aggregation: ${run.aggregationMode}; learner: ${run.learnerRuntime}. Quorum ${run.config.quorum}, max ${run.config.maxParticipants}, rounds ${run.config.rounds}.`,
      ),
      run.runMode === "real-lewm-tworooms"
        ? note(
            `Tapestry-like real-LeWM run: checkpoint ${run.lewmBinding?.checkpoint?.repoId}@${String(run.lewmBinding?.checkpoint?.revision ?? "").slice(0, 12)}, ` +
              `bounded adapter subset of ${run.lewmBinding?.adapterParameterCount} params. ${run.claimBoundary ?? ""}`,
          )
        : null,
      el("div", { class: "columns" }, [
        el("div", { class: "panel" }, [
          el("h2", { text: "Invite participants" }),
          qrCanvas,
          el("div", { class: "join-url" }, [
            urlInput,
            el("button", { class: "secondary", text: "Copy", onclick: () => navigator.clipboard?.writeText(joinUrl) }),
          ]),
          note("Backend mode coordinates browser sessions through the coordinator API and WebSocket stream, with REST polling retained as fallback."),
          startButton,
          abortButton,
          timeoutButton,
          exportButton,
          statusNote,
          el("p", {}, el("a", { href: "#/", text: "New run" })),
        ]),
        el("div", { class: "panel" }, [
          el("h2", { text: run.round > 0 ? `Round ${run.round} of ${run.config.rounds}` : "Waiting for participants" }),
          participantSlots(run),
          el("h2", { text: "Training diagnostics" }),
          trainingDiagnostics(run),
          el("h2", { text: "Round metrics" }),
          metricsList(run),
          el("h2", { text: "Artifacts" }),
          artifactList(run),
          el("h2", { text: "Event timeline" }),
          timelineList(run.events),
        ]),
      ]),
      renderInferencePanel(run),
    ]),
  );
  drawQr(qrCanvas, joinUrl);
}

// -------------------------------------------------------------- participants

function teardownParticipant() {
  if (participantSession) {
    participantSession.bus.close();
    participantSession = null;
  }
}

function participantStorageKey(runId) {
  return `lensemble-demo-participant:${runId}`;
}

function backendParticipantStorageKey(runId) {
  return `lensemble-demo-backend-participant:${runId}`;
}

function sessionIdForRun(runId) {
  const key = `lensemble-demo-session:${runId}`;
  let value = window.sessionStorage.getItem(key);
  if (!value) {
    value = `session-${randomSeed().toString(16)}`;
    window.sessionStorage.setItem(key, value);
  }
  return value;
}

function readBackendParticipant(runId) {
  const raw = window.sessionStorage.getItem(backendParticipantStorageKey(runId));
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw);
    return {
      ...parsed,
      automationMode: parsed.automationMode === "manual" ? "manual" : "auto",
    };
  } catch {
    return null;
  }
}

function writeBackendParticipant(runId, participant) {
  window.sessionStorage.setItem(backendParticipantStorageKey(runId), JSON.stringify(participant));
}

function renderModeControl(runId) {
  const name = `automation-${runId}`;
  const auto = el("input", { type: "radio", name, value: "auto", checked: "checked" });
  const manual = el("input", { type: "radio", name, value: "manual" });
  return el("div", { class: "segmented", role: "radiogroup", "aria-label": "Run mode" }, [
    el("label", {}, [auto, el("span", { text: "Auto" })]),
    el("label", {}, [manual, el("span", { text: "Manual" })]),
  ]);
}

function selectedMode(control) {
  return control.querySelector("input:checked")?.value === "manual" ? "manual" : "auto";
}

function startBackendLearner(run, me, participantToken, { force = false } = {}) {
  const key = learnerJobKey(run, me);
  const existing = backendLearnerJobs.get(key);
  if (!force && existing) return;
  backendLearnerJobs.set(key, { status: "running" });
  backendLearnerTelemetry.set(key, learnerTelemetryPayload(me.progress ?? 0, { phase: "queued" }));

  void (async () => {
    try {
      if (run.runMode === "real-lewm-tworooms") {
        // checkpoint-backed local continuation; no surrogate fallback on failure
        const { metrics } = await runRealLewmRound({
          run,
          me,
          participantToken,
          client: backendClient,
          loadRuntime: () => loadLewmRuntime(),
          seed: randomSeed(),
          participantMode: me.automationMode ?? "auto",
          onProgress: (progress, telemetry) => {
            backendLearnerTelemetry.set(key, learnerTelemetryPayload(progress, telemetry));
          },
        });
        backendLearnerTelemetry.set(key, learnerTelemetryPayload(1, metrics));
      } else {
        await backendClient.progress(run.id, me.id, participantToken, 0.1);
        const artifact = await runBrowserLearner(
          {
            runId: run.id,
            participantId: me.id,
            round: run.round,
            roundId: `${run.id}:round-${run.round}`,
            modelRevisionId: run.currentModelRevisionId ?? "initial",
            seed: randomSeed(),
            sampleCount: 24,
            localSteps: 8,
          },
          (progress, telemetry) => {
            backendLearnerTelemetry.set(key, learnerTelemetryPayload(progress, telemetry));
            void backendClient.progress(run.id, me.id, participantToken, progress).catch(() => {});
          },
        );
        backendLearnerTelemetry.set(key, learnerTelemetryPayload(1, artifact));
        await backendClient.submitUpdate(run.id, me.id, participantToken, artifact);
      }
      backendLearnerJobs.set(key, { status: "submitted" });
      if (currentRouteStill("join", run.id) && !shouldDeferAutoRefreshForDocument()) render();
    } catch (error) {
      backendLearnerJobs.set(key, { status: "error", error: error.message });
      backendLearnerTelemetry.set(
        key,
        learnerTelemetryPayload(me.progress ?? 0, { error: error.message, phase: "error" }),
      );
      if (currentRouteStill("join", run.id) && !shouldDeferAutoRefreshForDocument()) render();
    }
  })();
}

function renderJoin(route) {
  teardownHost();
  if (route.token) {
    renderBackendJoin(route);
  } else {
    clearBackendPoll();
    renderLocalJoin(route);
  }
}

function renderLocalJoin(route) {
  const runId = route.runId;
  if (!participantSession || participantSession.runId !== runId) {
    teardownParticipant();
    participantSession = {
      runId,
      participantId: window.sessionStorage.getItem(participantStorageKey(runId)),
      snapshot: null,
      error: null,
      bus: null,
    };
    participantSession.bus = new ParticipantBus(
      runId,
      (snapshot) => {
        participantSession.snapshot = snapshot;
        if (
          parseRoute(window.location.hash).view === "join"
          && !shouldDeferAutoRefreshForDocument()
        ) {
          render();
        }
      },
      adapters,
    );
  }
  const session = participantSession;
  const snapshot = session.snapshot;
  const panel = el("section", { class: "panel participant-stage" }, [el("h2", { text: `Participant room - ${runId}` })]);

  if (!snapshot) {
    panel.append(note("Waiting for the host tab. Frontend-only simulator joins require the host dashboard to stay open in this browser."));
    app.append(panel);
    return;
  }

  panel.append(
    el("p", { class: "note" }, [
      "Run state: ",
      stateBadge(snapshot.state),
      ` · round ${snapshot.round}/${snapshot.config.rounds} · mode ${snapshot.mode}`,
    ]),
  );

  const me = snapshot.participants.find((p) => p.id === session.participantId) ?? null;
  if (!me) {
    if (["aborted", "failed", "completed"].includes(snapshot.state)) {
      panel.append(errorBox(`This run is ${snapshot.state}; joining is closed.`));
    } else if (!["created", "joining", "ready"].includes(snapshot.state)) {
      panel.append(errorBox("This run already started; joining is closed."));
    } else {
      const nameInput = el("input", { type: "text", placeholder: "display name (optional)" });
      const joinError = el("p", { class: "note" });
      panel.append(
        el("label", {}, ["Display name", nameInput]),
        el("button", {
          text: "Join run",
          onclick: async () => {
            try {
              const reply = await session.bus.send({
                type: "join",
                displayName: nameInput.value.trim() || null,
              });
              if (reply?.ok) {
                session.participantId = reply.participantId;
                window.sessionStorage.setItem(participantStorageKey(runId), reply.participantId);
              } else {
                joinError.textContent = reply?.message ?? "join failed";
              }
            } catch (error) {
              joinError.textContent = error.message;
            }
            render();
          },
        }),
        joinError,
      );
    }
    app.append(panel);
    return;
  }

  panel.append(renderParticipantState(snapshot, me, true));
  app.append(panel);
}

async function renderBackendJoin(route) {
  teardownParticipant();
  app.append(el("section", { class: "panel" }, [el("h2", { text: "Loading participant room..." })]));
  try {
    const run = await backendClient.getRun(route.runId);
    if (!currentRouteStill("join", route.runId)) return;
    app.replaceChildren();
    renderBackendJoinSnapshot(route, run);
  } catch (error) {
    if (!currentRouteStill("join", route.runId)) return;
    app.replaceChildren(el("section", { class: "panel" }, [el("h2", { text: "Join failed" }), errorBox(error.message)]));
  }
}

function renderBackendJoinSnapshot(route, run) {
  const stored = readBackendParticipant(run.id);
  const me = stored ? run.participants.find((p) => p.id === stored.participantId) : null;
  if (stored && me) {
    ensureBackendPoll("join", run.id, {
      role: "participant",
      participantId: stored.participantId,
      participantToken: stored.participantToken,
    });
  } else {
    ensureBackendPoll("join", run.id, { role: "host" });
  }
  const panel = el("section", { class: "panel participant-stage" }, [
    el("h2", { text: `Participant room - ${run.id}` }),
    el("p", { class: "note" }, [
      "Run state: ",
      stateBadge(run.state),
      ` · round ${run.round}/${run.config.rounds} · ${backendPoll?.transport ?? "polling"}`,
    ]),
  ]);

  if (!me) {
    if (["aborted", "failed", "completed"].includes(run.state)) {
      panel.append(errorBox(`This run is ${run.state}; joining is closed.`));
    } else if (!["created", "joining", "ready"].includes(run.state)) {
      panel.append(errorBox("This run already started; joining is closed."));
    } else {
      const nameInput = el("input", { type: "text", placeholder: "display name (optional)" });
      const modeControl = renderModeControl(run.id);
      const joinError = el("p", { class: "note" });
      panel.append(
        el("label", {}, ["Display name", nameInput]),
        el("label", {}, ["Run mode", modeControl]),
        el("button", {
          text: "Join run",
          onclick: async () => {
            try {
              const automationMode = selectedMode(modeControl);
              const reply = await backendClient.joinRun(run.id, {
                joinToken: route.token,
                displayName: nameInput.value.trim() || null,
                sessionId: sessionIdForRun(run.id),
                automationMode,
              });
              writeBackendParticipant(run.id, {
                participantId: reply.participantId,
                participantToken: reply.participantToken,
                automationMode,
              });
              render();
            } catch (error) {
              joinError.textContent = error.message;
            }
          },
        }),
        joinError,
      );
    }
    app.append(panel);
    return;
  }

  panel.append(
    renderParticipantState(run, me, false, stored.participantToken, {
      automationMode: me.automationMode ?? stored.automationMode ?? "auto",
    }),
  );
  app.append(panel);
}

function renderLearnerTelemetry(run, me, telemetry) {
  const value = telemetry ?? latestUpdateMetadata(me, me.round ?? run.round) ?? {};
  const progress = value.progress ?? me.progress ?? 0;
  return el("div", { class: "learner-telemetry" }, [
    el("div", { class: "telemetry-head" }, [
      el("strong", { text: `round ${me.round ?? run.round} learner` }),
      value.phase ? el("span", { class: "muted", text: value.phase }) : stateBadge(value.collapseRisk ?? "watch"),
    ]),
    el("progress", { max: "1", value: String(progress) }),
    el("div", { class: "metric-tiles" }, [
      metricTile("progress", `${formatMetric(progress * 100, 0)}%`),
      metricTile("loss", formatMetric(value.loss, 4)),
      metricTile("probe", formatMetric(value.probe, 4)),
      metricTile("eff dim", formatMetric(value.effectiveDim, 2), value.effectiveDimRatio !== null && value.effectiveDimRatio !== undefined ? `${formatMetric(value.effectiveDimRatio * 100, 0)}%` : null),
      metricTile("clip", `${formatMetric((value.clipSaturation ?? 0) * 100, 0)}%`),
      metricTile("runtime", value.runtimeMs === null || value.runtimeMs === undefined ? "n/a" : `${formatMetric(value.runtimeMs, 1)} ms`),
    ]),
    value.error ? errorBox(value.error) : null,
  ]);
}

function renderParticipantState(run, me, simulated, participantToken = null, options = {}) {
  const automationMode = options.automationMode ?? me.automationMode ?? "manual";
  const stageText = {
    joined: "Joined. Waiting for the host to start the run.",
    ready: "Ready. Waiting for the host to start the run.",
    assigned: simulated
      ? `Assigned round ${me.round}. Preparing simulated local work.`
      : automationMode === "auto"
        ? `Assigned round ${me.round}. Auto learner queued.`
        : run.runMode === "real-lewm-tworooms"
          ? `Assigned round ${me.round}. Ready to run browser-local LeWM adapter continuation.`
          : `Assigned round ${me.round}. Ready to run the browser-local tiny learner.`,
    training: simulated
      ? `Simulating local work for round ${me.round}.`
      : run.runMode === "real-lewm-tworooms"
        ? `Browser-local LeWM adapter continuation in progress for round ${me.round}.`
        : `Browser-local tiny learner work in progress for round ${me.round}.`,
    submitted: "Bounded update artifact submitted. Waiting for aggregation.",
    completed: "Run complete. Thanks for participating.",
    dropped: "You were dropped from this run.",
    error: `Run failed: ${run.failureReason ?? me.error ?? "demo error"}`,
  }[me.state];
  const children = [
    el("p", {}, [el("span", { class: "stage", text: stageText }), " ", stateBadge(me.state)]),
  ];
  if (!simulated) {
    children.push(el("p", { class: "note" }, ["Run mode: ", stateBadge(automationMode)]));
  }
  if (me.state === "training") {
    children.push(el("progress", { max: "1", value: String(me.progress ?? 0) }));
  }
  if (!simulated && ["assigned", "training"].includes(me.state)) {
    const key = learnerJobKey(run, me);
    const job = backendLearnerJobs.get(key) ?? null;
    const telemetry = backendLearnerTelemetry.get(key) ?? null;
    if (automationMode === "auto" && participantToken) {
      startBackendLearner(run, me, participantToken);
    }
    const button = el("button", {
      class: automationMode === "auto" ? "secondary" : null,
      text: automationMode === "auto" ? "Retry local learner" : "Run local learner and submit update",
      onclick: () => {
        startBackendLearner(run, me, participantToken, { force: true });
      },
    });
    if (job?.status === "running" || job?.status === "submitted") button.disabled = true;
    if (automationMode === "auto" && job?.status !== "error") button.disabled = true;
    children.push(renderLearnerTelemetry(run, me, telemetry));
    children.push(button);
  }
  const submittedMetadata = latestUpdateMetadata(me, me.round ?? run.round);
  if (!simulated && submittedMetadata) {
    children.push(renderLearnerTelemetry(run, me, submittedMetadata));
  }
  if (run.state === "aborted") {
    children.push(errorBox("The host aborted this run."));
  }
  children.push(
    note(
      simulated
        ? "Local work in this slice is simulated; no data leaves your browser and no real training happens."
        : run.runMode === "real-lewm-tworooms"
          ? "Tapestry-like real-LeWM round: this browser generates TwoRooms rollouts locally, runs checkpoint-backed inference through the exported graphs, trains a bounded adapter, and submits only the clipped adapter delta plus metric summaries. Raw frames, actions, latents, tensors, tokens, and base checkpoint weights never leave this browser. If the runtime is unavailable the round fails visibly — there is no surrogate fallback."
          : "This browser computes a tiny clipped update vector from resident synthetic samples in a worker. Only the derived vector and shape/hash/norm metadata are submitted; raw observations, actions, labels, latents, tensors, participant tokens, and model weights are not uploaded.",
    ),
  );
  return el("div", {}, children);
}

// --------------------------------------------------------------- inference UI

function renderInferencePanel(run) {
  const state = inferenceByRun.get(run.id) ?? {
    env: initialInferenceState(),
    session: null,
    status: "Ready. Step the env, load the tiny run revision, or load an ONNX file explicitly.",
    metrics: "state=(0.250, 0.550)",
  };
  inferenceByRun.set(run.id, state);
  const artifact = selectRunInferenceArtifact(run);
  const identity = modelIdentity(artifact);
  const canvas = el("canvas", { class: "env-canvas", width: "48", height: "48", "aria-label": "Swipe-dot environment" });
  const actionX = el("input", { type: "range", min: "-1", max: "1", step: "0.05", value: "0.2" });
  const actionY = el("input", { type: "range", min: "-1", max: "1", step: "0.05", value: "-0.1" });
  const statusEl = el("p", { class: "note", text: state.status });
  const metricsEl = el("p", { class: "note mono", text: state.metrics });
  const fileInput = el("input", {
    type: "file",
    accept: ".onnx",
    onchange: async (event) => {
      const file = event.target.files?.[0];
      if (!file) return;
      try {
        state.session = await loadOnnxSession(file);
        state.status = `Loaded ${file.name}`;
      } catch (error) {
        state.session = null;
        state.status = modelLoadFailureMessage(error);
      }
      statusEl.textContent = state.status;
    },
  });
  const stepButton = el("button", {
    text: "Step env / run inference",
    onclick: async () => {
      const action = [Number(actionX.value), Number(actionY.value)];
      if (canRunTinyRevision(artifact) && !state.session) {
        const result = runTinyRevisionStep(artifact, state.env, action);
        state.env = result.state;
        state.status = result.metrics.status;
        state.metrics = [
          result.metrics.stateText,
          result.metrics.predicted,
          `latency=${result.metrics.latencyMs}ms`,
        ].join(" · ");
      } else if (!state.session) {
        state.env = stepEnvironment(state.env, action);
        const metrics = noModelMetrics(state.env);
        state.status = metrics.status;
        state.metrics = metrics.stateText;
      } else {
        const result = await runOnnxStep(state.session, state.env, action);
        state.env = result.state;
        state.status = result.metrics.status;
        state.metrics = [
          result.metrics.stateText,
          result.metrics.predicted,
          `inference=${result.metrics.latencyMs}ms`,
        ].join(" · ");
      }
      drawSwipeDot(canvas, state.env);
      statusEl.textContent = state.status;
      metricsEl.textContent = state.metrics;
    },
  });
  const resetButton = el("button", {
    class: "secondary",
    text: "Reset env",
    onclick: () => {
      state.env = initialInferenceState();
      state.status = "Environment reset.";
      state.metrics = "state=(0.250, 0.550)";
      drawSwipeDot(canvas, state.env);
      statusEl.textContent = state.status;
      metricsEl.textContent = state.metrics;
    },
  });
  const panel = el("section", { class: "panel inference-panel" }, [
    el("h2", { text: "Inference panel" }),
    note("Browser inference/env-sim is supported. A completed run loads the tiny JS model revision directly; loading a separate ONNX export remains explicit."),
    el("div", { class: "inference-grid" }, [
      canvas,
      el("div", { class: "panel compact" }, [
        el("dl", { class: "kv" }, [
          el("dt", { text: "Model" }),
          el("dd", { text: identity.modelId }),
          el("dt", { text: "Revision" }),
          el("dd", { text: identity.revision }),
          el("dt", { text: "Schema" }),
          el("dd", { text: identity.schema }),
          el("dt", { text: "Runtime" }),
          el("dd", { text: identity.runtime }),
          el("dt", { text: "Source" }),
          el("dd", { text: identity.source ?? "none" }),
        ]),
        el("label", {}, ["ONNX model", fileInput]),
        el("label", {}, ["Action X", actionX]),
        el("label", {}, ["Action Y", actionY]),
        stepButton,
        resetButton,
        statusEl,
        metricsEl,
      ]),
    ]),
  ]);
  setTimeout(() => drawSwipeDot(canvas, state.env), 0);
  return panel;
}

window.addEventListener("hashchange", render);
window.addEventListener("focus", () => {
  if (!shouldDeferAutoRefreshForDocument()) render();
});
window.addEventListener("visibilitychange", () => {
  if (!shouldDeferAutoRefreshForDocument()) render();
});
window.addEventListener("pagehide", () => {
  if (participantSession?.participantId) {
    try {
      participantSession.bus.channel.postMessage({
        type: "leave",
        participantId: participantSession.participantId,
        requestId: "leave-fire-and-forget",
      });
    } catch {
      // Page is closing.
    }
  }
  clearBackendPoll();
  teardownHost();
  teardownParticipant();
});

render();
