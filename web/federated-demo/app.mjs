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
import { lineChart, participantLossSeries, roundSeries } from "./charts.mjs";
import { morph } from "./morph.mjs";
import { loadLewmRuntime } from "./lewm_runtime.mjs";
import { compareRevisions } from "./lewm_probe.mjs";
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

const TIMELINE_MAX_EVENTS = 80;
const METRICS_SHOWN = 4;

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

// --------------------------------------------------- reactive update plumbing
//
// Data updates (rounds, polls, socket messages) reconcile a freshly-built tree
// into the live DOM via morph() instead of tearing #app down — graphs and
// metrics update in place, the QR/inference panels and scroll/focus survive.
//
// runRef is the single source of truth every interactive handler reads at click
// time, so no event listener kept alive by morph ever acts on a stale snapshot.
const runRef = { current: null };
let currentRoute = null; // routeKey of the currently mounted view

function routeKey(route) {
  return [route.view, route.runId ?? "", Boolean(route.token)].join("|");
}

// keyed(): stable identity for the reconciler. preserve(): never reconcile while
// it exists (an input mid-typing). markStatic(): opaque subtree (QR, inference).
function keyed(name, node) {
  if (node && node.dataset) node.dataset.key = name;
  return node;
}
function preserve(name, node) {
  if (node && node.dataset) {
    node.dataset.key = name;
    node.dataset.preserve = "";
  }
  return node;
}
function markStatic(name, node) {
  if (node && node.dataset) {
    node.dataset.key = name;
    node.dataset.static = "";
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
  const t = telemetry ?? {};
  return {
    progress,
    phase: t.phase ?? null,
    loss: t.loss ?? null,
    probe: t.probe ?? null,
    l2Norm: t.l2Norm ?? null,
    clipNorm: t.clipNorm ?? null,
    clipSaturation: t.clipSaturation ?? null,
    effectiveDim: t.effectiveDim ?? null,
    effectiveDimRatio: t.effectiveDimRatio ?? null,
    collapseRisk: t.collapseRisk ?? null,
    runtimeMs: t.runtimeMs ?? null,
    error: t.error ?? null,
  };
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
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, total, total);
  ctx.fillStyle = "#191c22";
  for (let r = 0; r < count; r += 1) {
    for (let c = 0; c < count; c += 1) {
      if (qr.isDark(r, c)) ctx.fillRect(c + quiet, r + quiet, 1, 1);
    }
  }
}

function timelineList(events) {
  const items = events
    .slice(-TIMELINE_MAX_EVENTS)
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
    return note("No artifacts yet. They appear as rounds aggregate.");
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

function runStatusStrip(run) {
  const latest = (run.roundMetrics ?? [])[run.roundMetrics?.length - 1] ?? null;
  const real = run.runMode === "real-lewm-tworooms";
  const active = (run.participants ?? []).filter((p) => !["dropped", "error"].includes(p.state)).length;
  const progressValue = run.config.rounds > 0 ? (run.roundMetrics?.length ?? 0) / run.config.rounds : 0;
  const tiles = [
    el("div", { class: "metric-tile" }, [
      el("span", { class: "metric-label", text: "progress" }),
      el("strong", { text: `${run.roundMetrics?.length ?? 0}/${run.config.rounds}` }),
      el("progress", { max: "1", value: String(progressValue) }),
    ]),
    metricTile("participants", `${active}/${run.config.maxParticipants}`, `quorum ${run.config.quorum}`),
  ];
  if (latest && real) {
    tiles.push(
      metricTile("pred loss", formatMetric(latest.predLossLastMean, 5), `from ${formatMetric(latest.predLossFirstMean, 5)}`),
      metricTile("sigreg", formatMetric(latest.sigregStatisticMean, 5)),
      metricTile("eff rank", formatMetric(latest.effectiveRankMean, 1)),
      metricTile("adapter ‖θ‖", formatMetric(latest.adapterStateNorm, 3), `Δ ${formatMetric(latest.aggregateDeltaNorm, 3)}`),
    );
  } else if (latest) {
    tiles.push(
      metricTile("loss", formatMetric(latest.localLossMean, 4)),
      metricTile("agg norm", formatMetric(latest.aggregateNorm, 4)),
    );
  }
  const flags = latest?.healthFlags ?? [];
  return el("div", { class: "status-strip" }, [
    el("div", { class: "metric-tiles" }, tiles),
    flags.length > 0 ? errorBox(`health: ${flags.join("; ")}`) : null,
  ]);
}

function runAnalytics(run) {
  const rounds = run.roundMetrics ?? [];
  if (rounds.length === 0) return [];
  const real = run.runMode === "real-lewm-tworooms";
  const charts = [];

  const perParticipant = participantLossSeries(run);
  const meanSeries = roundSeries(run, [
    { key: real ? "predLossLastMean" : "localLossMean", label: "mean", dashed: true },
  ]).map((s) => ({ ...s, color: "#8b909a" }));
  if (perParticipant.length > 0) {
    charts.push(
      lineChart({
        series: [...perParticipant, ...meanSeries],
        title: real ? "Prediction loss by participant" : "Local loss by participant",
        yZero: true,
      }),
    );
  }
  if (real) {
    charts.push(
      lineChart({
        series: roundSeries(run, [
          { key: "predLossFirstMean", label: "round start" },
          { key: "predLossLastMean", label: "after local training" },
        ]),
        title: "Round-mean loss: start vs trained",
        yZero: true,
      }),
      lineChart({
        series: roundSeries(run, [{ key: "sigregStatisticMean", label: "SIGReg" }]),
        title: "SIGReg statistic (anti-collapse)",
        yZero: true,
      }),
      lineChart({
        series: roundSeries(run, [
          { key: "effectiveRankMean", label: "effective rank" },
          { key: "latentStdMeanMean", label: "latent std", dashed: true },
        ]),
        title: "Latent geometry",
        yZero: true,
      }),
      lineChart({
        series: roundSeries(run, [
          { key: "aggregateDeltaNorm", label: "round Δ norm" },
          { key: "adapterStateNorm", label: "adapter ‖θ‖", dashed: true },
        ]),
        title: "Adapter norms",
        yZero: true,
      }),
    );
  }
  if (charts.length === 0) return [];
  return [
    keyed("analytics-head", el("h2", { text: "Run analytics" })),
    keyed("analytics-grid", el("div", { class: "charts-grid" }, charts)),
  ];
}

function metricsList(run) {
  const metrics = run.roundMetrics ?? [];
  if (metrics.length === 0) {
    return note("Round metrics appear once the first round aggregates.");
  }
  const real = run.runMode === "real-lewm-tworooms";
  return el(
    "div",
    { class: "metrics-grid" },
    metrics
      .slice(-METRICS_SHOWN)
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
          real && (metric.healthFlags ?? []).length > 0
            ? errorBox(`health: ${metric.healthFlags.join("; ")}`)
            : null,
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
    return note("Per participant diagnostics appear once updates start arriving.");
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
    if (view === "host") morphHostSnapshot(run);
    else if (view === "join") morphJoinSnapshot(parseRoute(window.location.hash), run);
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
        if (view === "host") morphHostSnapshot(run);
        else if (view === "join") morphJoinSnapshot(parseRoute(window.location.hash), run);
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
  const url = URL.createObjectURL(blob);
  const link = el("a", { href: url, download: filename });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

// ------------------------------------------------------------------- router

// render() is route-only: a genuine route change (or empty #app) MOUNTS a view
// (the one place #app is torn down); same-route ticks/polls go through morph.
function render() {
  const route = parseRoute(window.location.hash);
  const key = routeKey(route);
  if (key !== currentRoute || !app.firstElementChild) {
    currentRoute = key;
    mountRoute(route);
  } else {
    updateRoute(route);
  }
}

function mountRoute(route) {
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

// Same-route data update: reconcile in place, never replaceChildren.
function updateRoute(route) {
  if (shouldDeferAutoRefreshForDocument()) return;
  if (!app.firstElementChild) {
    mountRoute(route);
    return;
  }
  if (route.view === "host") {
    if (hostSession?.run.id === route.runId) updateLocalHost(hostSession.run);
    else if (runRef.current) morphHostSnapshot(runRef.current);
  } else if (route.view === "join") {
    if (!route.token) morphLocalJoin();
    else if (runRef.current) morphJoinSnapshot(route, runRef.current);
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

  const maxInput = el("input", { type: "number", min: "1", max: "64", value: "4" });
  const quorumInput = el("input", { type: "number", min: "1", max: "64", value: "2" });
  const roundsInput = el("input", { type: "number", min: "1", max: "1000", value: "10" });
  const errorNote = el("p", { class: "note" });

  const createButton = el("button", {
    text: "Create run",
    onclick: async () => {
      const config = {
        maxParticipants: Number(maxInput.value),
        quorum: Number(quorumInput.value),
        rounds: Number(roundsInput.value),
        mode: "real-lewm-tworooms",
      };
      try {
        const run = await backendClient.createRun(config);
        window.location.hash = `#/host/${run.id}`;
      } catch (error) {
        errorNote.textContent = error.message;
      }
    },
  });

  app.append(
    el("section", { class: "panel hero" }, [
      el("h1", { text: "Federate adapter updates on a real world-model checkpoint" }),
      note(
        "Each participant adapts a real LeWorldModel checkpoint right in their browser: the world model stays frozen and only a small, bounded residual adapter on its predictor output trains and federates. Rollouts stay on the device; only the bounded adapter delta is shared. Rounds aggregate into hash-bound global revisions you can probe, inspect, and export.",
      ),
      el("p", {}, [
        el("span", { class: "chip", text: "quentinll/lewm-tworooms · 77adaae0bc31" }),
      ]),
    ]),
    el("section", { class: "panel" }, [
      el("div", { class: "columns" }, [
        el("div", { class: "panel" }, [
          el("h2", { text: "New run" }),
          el("label", {}, ["Max participants", maxInput]),
          el("label", {}, ["Quorum (min trainers to start)", quorumInput]),
          el("label", {}, ["Rounds", roundsInput]),
          createButton,
          errorNote,
        ]),
        el("div", { class: "panel" }, [
          el("h2", { text: "How it works" }),
          el("ol", { class: "steps" }, [
            el("li", { text: "Create a run and share the QR code. Each participant joins from their own browser and trains on its own." }),
            el("li", { text: "Watch rounds aggregate live. Prediction loss, SIGReg, effective rank, and adapter revisions stream in as they happen." }),
            el("li", { text: "When you are ready, run the before and after probe against the latest revision, then export the evidence bundle." }),
          ]),
          el("p", {}, [
            el("a", {
              href: "#/tworooms",
              text: "Open the TwoRooms lab and watch the model plan in real time",
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
      hostSession
      && parseRoute(window.location.hash).view === "host"
      && !shouldDeferAutoRefreshForDocument()
    ) {
      updateLocalHost(hostSession.run);
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
        updateLocalHost(run);
      }
    }
  }, 700);
}

function renderHost(runId) {
  teardownParticipant();
  if (hostSession?.run.id === runId) {
    clearBackendPoll();
    mountLocalHost();
    return;
  }
  const snapshot = loadRunSnapshot(adapters.storage, runId);
  const restored = restoreSimRun(snapshot);
  if (restored) {
    adoptHostRun(restored);
    clearBackendPoll();
    mountLocalHost();
    return;
  }
  renderBackendHost(runId);
}

function localHostQrUrl(run) {
  return buildJoinUrl(window.location.href, run.id);
}

function drawLocalHostQr(run) {
  const canvas = app.querySelector("#qr");
  if (!canvas) return;
  const url = localHostQrUrl(run);
  if (canvas.dataset.qrUrl !== url || !canvas.width) {
    canvas.dataset.qrUrl = url;
    drawQr(canvas, url);
  }
}

function mountLocalHost() {
  const { run } = hostSession;
  ensureHostTicker();
  runRef.current = run;
  app.append(buildLocalHostTree(run));
  drawLocalHostQr(run);
}

function updateLocalHost(run) {
  if (!hostSession) return;
  ensureHostTicker();
  runRef.current = run;
  if (!app.firstElementChild) {
    app.append(buildLocalHostTree(run));
    drawLocalHostQr(run);
    return;
  }
  morph(app.firstElementChild, buildLocalHostTree(run));
  drawLocalHostQr(run);
}

function buildLocalHostTree(run) {
  const runId = run.id;
  const joinUrl = localHostQrUrl(run);
  const qrCanvas = markStatic("qr", el("canvas", { id: "qr", "aria-label": "Join URL QR code" }));
  const urlInput = el("input", { type: "text", readonly: "readonly", value: joinUrl });
  const copyButton = el("button", {
    class: "secondary",
    text: "Copy",
    onclick: () => navigator.clipboard?.writeText(localHostQrUrl(runRef.current ?? run)),
  });

  const active = activeParticipants(run);
  const startButton = keyed("lhost-start", el("button", {
    text: run.state === "ready" ? "Start run" : `Start run (${active.length}/${run.config.quorum})`,
    onclick: () => {
      try {
        startSimRun(hostSession.run);
        hostSession.bus.publish();
        updateLocalHost(hostSession.run);
      } catch (error) {
        window.alert(error.message);
      }
    },
  }));
  if (run.state !== "ready") startButton.disabled = true;

  const abortButton = keyed("lhost-abort", el("button", {
    class: "danger",
    text: "Abort run",
    onclick: () => {
      try {
        abortSimRun(hostSession.run, "host abort");
        hostSession.bus.publish();
        updateLocalHost(hostSession.run);
      } catch (error) {
        window.alert(error.message);
      }
    },
  }));
  if (["completed", "aborted", "failed"].includes(run.state)) abortButton.disabled = true;

  const failButton = keyed("lhost-fail", el("button", {
    class: "secondary",
    text: "Simulate failure",
    onclick: () => {
      try {
        failSimRun(hostSession.run, "simulated infrastructure failure");
        hostSession.bus.publish();
        updateLocalHost(hostSession.run);
      } catch (error) {
        window.alert(error.message);
      }
    },
  }));
  if (["completed", "aborted", "failed"].includes(run.state)) failButton.disabled = true;

  return el("section", { class: "panel" }, [
    keyed("lhost-head", el("h2", {}, [`Host dashboard - ${runId} `, stateBadge(run.state)])),
    keyed("lhost-mode", note(`Mode: ${run.mode}. Quorum ${run.config.quorum}, max ${run.config.maxParticipants}, rounds ${run.config.rounds}.`)),
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
        keyed("lhost-round", el("h2", { text: run.round > 0 ? `Round ${run.round} of ${run.config.rounds}` : "Waiting for participants" })),
        keyed("lhost-slots", participantSlots(run)),
        keyed("lhost-art-head", el("h2", { text: "Artifacts" })),
        keyed("lhost-artifacts", artifactList(run)),
        keyed("lhost-tl-head", el("h2", { text: "Event timeline (simulated)" })),
        keyed("lhost-timeline", timelineList(run.events)),
      ]),
    ]),
  ]);
}

// ---------------------------------------------------------- backend host mode

async function renderBackendHost(runId) {
  if (backendPoll?.view !== "host" || backendPoll?.runId !== runId) clearBackendPoll();
  app.append(el("section", { class: "panel" }, [el("h2", { text: "Loading backend run..." })]));
  try {
    const run = await backendClient.getRun(runId);
    if (!currentRouteStill("host", runId)) return;
    app.replaceChildren();
    mountBackendHostSnapshot(run);
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

// Refetch + morph after an explicit host action (transport-agnostic, unlike the
// websocket-gated refreshBackendRoute) so the dashboard reflects the action at once.
async function pullBackendHost(runId) {
  if (!currentRouteStill("host", runId)) return;
  try {
    const run = await backendClient.getRun(runId);
    if (currentRouteStill("host", runId)) morphHostSnapshot(run);
  } catch {
    // The poll / socket stream will reconcile shortly.
  }
}

function hostQrUrl(run) {
  return run.joinUrl || buildJoinUrl(window.location.href, run.id, run.joinToken);
}

// Pure builder: returns the detached host dashboard <section>. Shared by mount
// and morph so the live tree and the freshly-built tree always have the same
// shape. Handlers read runRef.current (never the captured run) so a node kept by
// the reconciler never acts on a stale snapshot.
function buildBackendHostTree(run) {
  const joinUrl = hostQrUrl(run);
  const qrCanvas = markStatic("qr", el("canvas", { id: "qr", "aria-label": "Join URL QR code" }));
  const urlInput = el("input", { type: "text", readonly: "readonly", value: joinUrl });
  const statusNote = keyed("status-note", el("p", { class: "note" }));

  const startButton = keyed("btn-start", el("button", {
    text: run.controls.canStart ? "Start run" : `Start run (${run.participants.length}/${run.config.quorum})`,
    onclick: async () => {
      const live = runRef.current;
      try {
        await backendClient.control(live.id, "start");
        void pullBackendHost(live.id);
      } catch (error) {
        statusNote.textContent = error.message;
      }
    },
  }));
  startButton.disabled = !run.controls.canStart;

  const abortButton = keyed("btn-abort", el("button", {
    class: "danger",
    text: "Abort run",
    onclick: async () => {
      const live = runRef.current;
      try {
        await backendClient.control(live.id, "abort", { reason: "host abort" });
        void pullBackendHost(live.id);
      } catch (error) {
        statusNote.textContent = error.message;
      }
    },
  }));
  abortButton.disabled = !run.controls.canAbort;

  const timeoutButton = keyed("btn-timeout", el("button", {
    class: "secondary",
    text: "Drop timed-out participants",
    onclick: async () => {
      const live = runRef.current;
      try {
        await backendClient.control(live.id, "timeout-missing", { reason: "host timeout" });
        void pullBackendHost(live.id);
      } catch (error) {
        statusNote.textContent = error.message;
      }
    },
  }));
  timeoutButton.disabled = run.state !== "running_round";

  const exportButton = keyed("btn-export", el("button", {
    class: "secondary",
    text: "Export evidence JSON",
    onclick: async () => {
      const live = runRef.current;
      try {
        const evidence = await backendClient.exportEvidence(live.id);
        downloadJson(`${live.id}-evidence.json`, evidence);
      } catch (error) {
        statusNote.textContent = error.message;
      }
    },
  }));

  return el("section", { class: "panel" }, [
    keyed("run-head", el("h2", {}, [`Run ${run.id} `, stateBadge(run.state)])),
    keyed("run-meta", el("p", { class: "muted" }, [
      run.runMode === "real-lewm-tworooms"
        ? el("span", { class: "chip", text: `${run.lewmBinding?.checkpoint?.repoId}@${String(run.lewmBinding?.checkpoint?.revision ?? "").slice(0, 12)} · ${run.lewmBinding?.adapterParameterCount}-param adapter` })
        : el("span", { class: "chip", text: run.learnerRuntime }),
      ` quorum ${run.config.quorum} · up to ${run.config.maxParticipants} participants · ${run.config.rounds} rounds · ${backendPoll?.transport ?? run.deployment?.transportMode ?? "polling"}`,
    ])),
    run.state === "paused"
      ? keyed("paused-banner", el("p", { class: "error-box" }, [
          `Paused — quorum dropped${run.pausedReason ? ` (${run.pausedReason})` : ""}. The run resumes automatically when enough participants reconnect; their progress is kept.`,
        ]))
      : null,
    keyed("status-strip", runStatusStrip(run)),
    el("div", { class: "columns" }, [
      el("div", { class: "panel" }, [
        el("h2", { text: "Invite participants" }),
        qrCanvas,
        el("div", { class: "join-url" }, [
          urlInput,
          el("button", { class: "secondary", text: "Copy", onclick: () => navigator.clipboard?.writeText(runRef.current ? hostQrUrl(runRef.current) : joinUrl) }),
        ]),
        startButton,
        abortButton,
        timeoutButton,
        exportButton,
        statusNote,
        el("p", {}, el("a", { href: "#/", text: "New run" })),
      ]),
      el("div", { class: "panel" }, [
        keyed("round-head", el("h2", { text: run.round > 0 ? `Round ${run.round} of ${run.config.rounds}` : "Waiting for participants" })),
        keyed("participant-slots", participantSlots(run)),
        ...runAnalytics(run),
        run.runMode === "real-lewm-tworooms" ? keyed("claim-boundary", renderClaimBoundary(run)) : null,
        run.runMode === "real-lewm-tworooms" ? keyed("probe-head", el("h2", { text: "Before/after validation probe" })) : null,
        keyed("probe", renderRealModeProbe(run)),
        keyed("diag-head", el("h2", { text: "Training diagnostics" })),
        keyed("diagnostics", trainingDiagnostics(run)),
        keyed("metrics-head", el("h2", { text: "Round metrics" })),
        keyed("metrics", metricsList(run)),
        keyed("artifacts-head", el("h2", { text: "Artifacts" })),
        keyed("artifacts", artifactList(run)),
        keyed("timeline-head", el("h2", { text: "Event timeline" })),
        keyed("timeline", timelineList(run.events)),
      ]),
    ]),
    run.runMode === "real-lewm-tworooms"
      ? keyed("checkpoint-note", el("section", { class: "panel compact" }, [
          el("h2", { text: "Checkpoint-backed inference" }),
          el("p", { class: "note" }, [
            "Watch the real model roll out and plan in the ",
            el("a", { href: "#/tworooms", text: "TwoRooms lab" }),
            ". The probe above scores the latest aggregated revision.",
          ]),
        ]))
      : markStatic("inference", renderInferencePanel(run)),
  ]);
}

function drawHostQr(run) {
  const canvas = app.querySelector("#qr");
  if (!canvas) return;
  const url = hostQrUrl(run);
  if (canvas.dataset.qrUrl !== url || !canvas.width) {
    canvas.dataset.qrUrl = url;
    drawQr(canvas, url);
  }
}

function mountBackendHostSnapshot(run) {
  runRef.current = run;
  ensureBackendPoll("host", run.id, { role: "host" });
  app.append(buildBackendHostTree(run));
  drawHostQr(run);
}

function morphHostSnapshot(run) {
  runRef.current = run;
  ensureBackendPoll("host", run.id, { role: "host" });
  if (!app.firstElementChild) {
    mountBackendHostSnapshot(run);
    return;
  }
  morph(app.firstElementChild, buildBackendHostTree(run));
  drawHostQr(run);
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
      if (currentRouteStill("join", run.id) && !shouldDeferAutoRefreshForDocument()) {
        updateRoute(parseRoute(window.location.hash));
      }
    } catch (error) {
      backendLearnerJobs.set(key, { status: "error", error: error.message });
      backendLearnerTelemetry.set(
        key,
        learnerTelemetryPayload(me.progress ?? 0, { error: error.message, phase: "error" }),
      );
      if (currentRouteStill("join", run.id) && !shouldDeferAutoRefreshForDocument()) {
        updateRoute(parseRoute(window.location.hash));
      }
    }
  })();
}

function joinBlockedMessage(state) {
  if (["aborted", "failed", "completed"].includes(state)) return `This run is ${state}; joining is closed.`;
  if (!["created", "joining", "ready"].includes(state)) return "This run already started; joining is closed.";
  return null;
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
          morphLocalJoin();
        }
      },
      adapters,
    );
  }
  app.append(buildLocalJoinTree(route, participantSession.snapshot));
}

function morphLocalJoin() {
  if (!participantSession) return;
  const route = parseRoute(window.location.hash);
  runRef.current = participantSession.snapshot;
  if (!app.firstElementChild) {
    app.append(buildLocalJoinTree(route, participantSession.snapshot));
    return;
  }
  morph(app.firstElementChild, buildLocalJoinTree(route, participantSession.snapshot));
}

function buildLocalJoinTree(route, snapshot) {
  const runId = route.runId;
  const session = participantSession;
  const children = [keyed("ljoin-head", el("h2", { text: `Participant room - ${runId}` }))];

  if (!snapshot) {
    children.push(keyed("ljoin-wait", note("Waiting for the host tab. Frontend-only simulator joins require the host dashboard to stay open in this browser.")));
    return el("section", { class: "panel participant-stage" }, children);
  }

  children.push(
    keyed("ljoin-state", el("p", { class: "note" }, [
      "Run state: ",
      stateBadge(snapshot.state),
      ` · round ${snapshot.round}/${snapshot.config.rounds} · mode ${snapshot.mode}`,
    ])),
  );

  const me = snapshot.participants.find((p) => p.id === session.participantId) ?? null;
  if (!me) {
    const blocked = joinBlockedMessage(snapshot.state);
    if (blocked) {
      children.push(keyed("ljoin-blocked", errorBox(blocked)));
    } else {
      const nameInput = el("input", { type: "text", placeholder: "display name (optional)" });
      const joinError = el("p", { class: "note" });
      children.push(preserve("ljoin-form", el("div", {}, [
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
            morphLocalJoin();
          },
        }),
        joinError,
      ])));
    }
    return el("section", { class: "panel participant-stage" }, children);
  }

  children.push(keyed("ljoin-body", renderParticipantState(snapshot, me, true)));
  return el("section", { class: "panel participant-stage" }, children);
}

async function renderBackendJoin(route) {
  teardownParticipant();
  app.append(el("section", { class: "panel" }, [el("h2", { text: "Loading participant room..." })]));
  try {
    const run = await backendClient.getRun(route.runId);
    if (!currentRouteStill("join", route.runId)) return;
    app.replaceChildren();
    mountBackendJoinSnapshot(route, run);
  } catch (error) {
    if (!currentRouteStill("join", route.runId)) return;
    app.replaceChildren(el("section", { class: "panel" }, [el("h2", { text: "Join failed" }), errorBox(error.message)]));
  }
}

// Refetch + morph after the participant joins (membership change) for immediate feedback.
async function pullBackendJoin(route) {
  if (!currentRouteStill("join", route.runId)) return;
  try {
    const run = await backendClient.getRun(route.runId);
    if (currentRouteStill("join", route.runId)) morphJoinSnapshot(route, run);
  } catch {
    // The poll / socket stream will reconcile shortly.
  }
}

function ensureJoinPoll(run) {
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
}

function mountBackendJoinSnapshot(route, run) {
  runRef.current = run;
  ensureJoinPoll(run);
  app.append(buildBackendJoinTree(route, run));
}

function morphJoinSnapshot(route, run) {
  runRef.current = run;
  ensureJoinPoll(run);
  if (!app.firstElementChild) {
    mountBackendJoinSnapshot(route, run);
    return;
  }
  morph(app.firstElementChild, buildBackendJoinTree(route, run));
}

function buildBackendJoinTree(route, run) {
  const stored = readBackendParticipant(run.id);
  const me = stored ? run.participants.find((p) => p.id === stored.participantId) : null;
  const children = [
    keyed("bjoin-head", el("h2", { text: `Participant room - ${run.id}` })),
    keyed("bjoin-state", el("p", { class: "note" }, [
      "Run state: ",
      stateBadge(run.state),
      ` · round ${run.round}/${run.config.rounds} · ${backendPoll?.transport ?? "polling"}`,
    ])),
  ];
  if (run.state === "paused") {
    children.push(
      keyed("bjoin-paused", el("p", { class: "error-box" }, [
        "Run paused — waiting for quorum to return. Keep this tab open; it rejoins and resumes automatically.",
      ])),
    );
  }

  if (!me) {
    const blocked = joinBlockedMessage(run.state);
    if (blocked) {
      children.push(keyed("bjoin-blocked", errorBox(blocked)));
    } else {
      const nameInput = el("input", { type: "text", placeholder: "display name (optional)" });
      const modeControl = renderModeControl(run.id);
      const joinError = el("p", { class: "note" });
      children.push(preserve("bjoin-form", el("div", {}, [
        el("label", {}, ["Display name", nameInput]),
        el("label", {}, ["Run mode", modeControl]),
        el("button", {
          text: "Join run",
          onclick: async () => {
            const liveRoute = parseRoute(window.location.hash);
            const liveRun = runRef.current ?? run;
            try {
              const automationMode = selectedMode(modeControl);
              const reply = await backendClient.joinRun(liveRun.id, {
                joinToken: liveRoute.token,
                displayName: nameInput.value.trim() || null,
                sessionId: sessionIdForRun(liveRun.id),
                automationMode,
              });
              writeBackendParticipant(liveRun.id, {
                participantId: reply.participantId,
                participantToken: reply.participantToken,
                automationMode,
              });
              void pullBackendJoin(liveRoute);
            } catch (error) {
              joinError.textContent = error.message;
            }
          },
        }),
        joinError,
      ])));
    }
    return el("section", { class: "panel participant-stage" }, children);
  }

  children.push(keyed("bjoin-body", renderParticipantState(run, me, false, stored.participantToken, {
    automationMode: me.automationMode ?? stored.automationMode ?? "auto",
  })));
  return el("section", { class: "panel participant-stage" }, children);
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
    joined: "You are in. Waiting for the host to start the run.",
    ready: "Ready to go. The run begins when the host starts it.",
    assigned: simulated
      ? `Assigned round ${me.round}. Preparing simulated local work.`
      : automationMode === "auto"
        ? `Round ${me.round} assigned. Your browser is getting to work.`
        : run.runMode === "real-lewm-tworooms"
          ? `Round ${me.round} assigned. Press the button below to train locally.`
          : `Round ${me.round} assigned. Press the button below to run the local learner.`,
    training: simulated
      ? `Simulating local work for round ${me.round}.`
      : run.runMode === "real-lewm-tworooms"
        ? `Training the adapter locally for round ${me.round}.`
        : `Local learner running for round ${me.round}.`,
    submitted: "Your update is in. Waiting for the round to aggregate.",
    completed: "Run complete. Thanks for taking part.",
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
        // Resolve the freshest run + participant at click time, never the captured snapshot.
        const liveRun = runRef.current ?? run;
        const liveMe = (liveRun.participants ?? []).find((p) => p.id === me.id) ?? me;
        startBackendLearner(liveRun, liveMe, participantToken, { force: true });
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
    el("p", { class: "muted" }, [
      simulated
        ? "Local work in this slice is simulated."
        : "Rollouts and training never leave this browser. Only a bounded adapter delta is shared. If the runtime is unavailable the round fails visibly, with no fallback.",
    ]),
  );
  return el("div", {}, children);
}

// --------------------------------------------------- real-mode validation probe

const probeResults = new Map();
// Probe runs are tracked here (not just on the DOM node) so a poll landing
// mid-probe rebuilds the button already-disabled and the reconciler preserves it.
const probeInFlight = new Set();

// Render the strict claim boundary IN the running UI (#329) — not just a footer link. The scoped
// chips and the single-coordinator note make a screenshot of the H1 + probe badge read as what the
// mechanics actually are: a bounded adapter on a frozen base, scored by a fixed held-out probe,
// aggregated by one local coordinator. The full boundary text is sourced from the run snapshot
// (run.claimBoundary === LEWM_CLAIM_BOUNDARY), never hardcoded divergently here.
function renderClaimBoundary(run) {
  if (run.runMode !== "real-lewm-tworooms") return null;
  const chips = [
    "bounded adapter on a frozen base",
    "fixed held-out probe, not a benchmark",
    "single local coordinator",
  ];
  return el("div", { class: "claim-boundary" }, [
    el("div", { class: "chips" }, chips.map((text) => el("span", { class: "chip", text }))),
    el("p", {
      class: "note",
      text: "Single local coordinator · mean-of-clipped-deltas (no robust aggregation / DP in this path). Only a 12,512-param (0.069%) zero-init residual adapter on the frozen predictor output trains and federates.",
    }),
    run.claimBoundary
      ? el("p", { class: "note", text: run.claimBoundary })
      : null,
  ]);
}

function renderRealModeProbe(run) {
  if (run.runMode !== "real-lewm-tworooms") return null;
  const revisions = run.modelRevisions ?? [];
  if (revisions.length === 0) {
    return note("The probe unlocks once the first adapter revision aggregates.");
  }
  const last = revisions[revisions.length - 1];
  const probeKey = `${run.id}:${last.modelRevisionId}`;
  const cached = probeResults.get(probeKey);
  const statusNote = el("p", { class: "note", text: cached ? "" : "Scores the latest global adapter against the plain checkpoint on a fixed validation set, right here in your browser." });
  const resultBox = el("div", {});
  if (cached) renderProbeResult(resultBox, cached);

  const button = el("button", {
    class: "secondary",
    text: `Run before/after probe against ${last.modelRevisionId}`,
    onclick: async () => {
      // Read the latest run + revision at click time, never the captured snapshot.
      const liveRun = runRef.current ?? run;
      const liveRevisions = liveRun.modelRevisions ?? [];
      const liveLast = liveRevisions[liveRevisions.length - 1];
      if (!liveLast) return;
      const key = `${liveRun.id}:${liveLast.modelRevisionId}`;
      probeInFlight.add(key);
      button.disabled = true;
      statusNote.textContent = "Loading the runtime and scoring both revisions on the fixed validation set…";
      try {
        const runtime = await loadLewmRuntime();
        const revision = await backendClient.modelRevision(liveRun.id, liveLast.modelRevisionId);
        if (!Array.isArray(revision.adapterState)) {
          throw new Error("revision carries no adapter state");
        }
        const report = await compareRevisions({
          runtime,
          adaptedState: revision.adapterState,
          adapterHiddenDim: liveRun.lewmBinding?.adapterHiddenDim ?? 32,
          adapterInitSeed: liveRun.lewmBinding?.adapterInitSeed ?? 42,
        });
        report.modelRevisionId = liveLast.modelRevisionId;
        probeResults.set(key, report);
        statusNote.textContent = "";
        renderProbeResult(resultBox, report);
      } catch (error) {
        statusNote.textContent = "";
        resultBox.replaceChildren(errorBox(`Probe failed: ${error.message}`));
      } finally {
        probeInFlight.delete(key);
        button.disabled = false;
      }
    },
  });
  button.disabled = probeInFlight.has(probeKey);
  return el("div", {}, [button, statusNote, resultBox]);
}

function renderProbeResult(container, report) {
  const diag = report.diagnostics ?? null;
  const headlineVerdict = report.displayVerdict ?? report.verdict;
  const children = [
    el("div", { class: "metric-tiles" }, [
      metricTile("baseline mse", formatMetric(report.baselineMse, 6)),
      metricTile("adapted mse", formatMetric(report.adaptedMse, 6)),
      metricTile("Δ relative", `${formatMetric(report.relativeImprovement * 100, 2)}%`),
    ]),
    el("p", {}, [
      "Verdict: ",
      stateBadge(headlineVerdict),
      ` on ${report.pairCount} fixed validation pairs (seed ${report.seed}, ${report.modelRevisionId ?? "latest"}). Held-out probe, not a benchmark.`,
    ]),
  ];
  // Held-out collapse diagnostics (#328): the same latent-std / effective-rank / SIGReg checks that
  // #259 needed, now on the VALIDATION set for baseline vs adapted — proving the gain is
  // bias-correction, not a magnitude/rank collapse that MSE alone cannot see.
  if (diag) {
    children.push(
      el("p", { class: "note", text: "Held-out collapse diagnostics (baseline → adapted):" }),
      el("div", { class: "metric-tiles" }, [
        metricTile(
          "latent std",
          `${formatMetric(diag.baseline.latentStdMean, 3)} → ${formatMetric(diag.adapted.latentStdMean, 3)}`,
        ),
        metricTile(
          "effective rank",
          `${formatMetric(diag.baseline.effectiveRank, 2)} → ${formatMetric(diag.adapted.effectiveRank, 2)}`,
        ),
        metricTile(
          "SIGReg",
          `${formatMetric(diag.baseline.sigregStatistic, 3)} → ${formatMetric(diag.adapted.sigregStatistic, 3)}`,
        ),
      ]),
    );
  }
  if (report.collapseRisk) {
    children.push(
      errorBox(
        `Collapse risk overrides the MSE verdict: ${(report.collapseFlags ?? []).join("; ")}. An improved MSE under latent collapse is not a real win (the #259 failure mode).`,
      ),
    );
  } else if (report.verdict !== "improved") {
    children.push(
      note(
        "The adapter revision did not beat the parent checkpoint on this probe. The result stands as reported, not hidden.",
      ),
    );
  }
  container.replaceChildren(...children);
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

// Coming back to the tab: repaint from in-memory state at once (no teardown),
// then pull a fresh backend snapshot if a run stream is active.
function refreshActiveView() {
  if (shouldDeferAutoRefreshForDocument()) return;
  updateRoute(parseRoute(window.location.hash));
  if (backendPoll) {
    attachBackendSocket(backendPoll);
    void refreshBackendRoute(backendPoll.view, backendPoll.runId);
  }
}

window.addEventListener("hashchange", render);
window.addEventListener("focus", refreshActiveView);
window.addEventListener("visibilitychange", () => {
  if (!document.hidden) refreshActiveView();
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
  inferenceByRun.clear();
  backendLearnerJobs.clear();
  backendLearnerTelemetry.clear();
  probeResults.clear();
});

render();
