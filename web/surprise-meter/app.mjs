// surprise-meter — UI wiring over the live/fallback stage engine.
import { loadLewmRuntime } from "../federated-demo/lewm_runtime.mjs";
import { collectResidentPairs } from "../federated-demo/lewm_local_trainer.mjs";
import { SurpriseEngine } from "./mock/engine.mjs";
import { Recorder } from "./seismograph.mjs";
import { CERTIFIED as FALLBACK_CERTIFIED, NON_CLAIMS as FALLBACK_NON_CLAIMS, MODEL, pct } from "./mock/fixtures.mjs";
import { mockEnv } from "./mock/lewm_mock.mjs"; // ROOMS layout for the world view
import { buildLiveSurpriseTrajectory, buildPrePostStreams } from "./surprise_engine.mjs";

const $ = (id) => document.getElementById(id);
const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const TRACE_INK = "#222f3c", GRAPHITE = "#84786a", OXBLOOD = "#96282b",
      INK2 = "#6a6253", RULE = "rgba(37,28,22,0.5)", PAPER_EDGE = "#efe9dd";

// dev / kiosk hooks:
// ?engine=auto|live|fallback, ?ep=wasm|webgpu, ?steps=96, ?mode=post,
// ?auto=ood|teleport|wall, ?hold=ood.
const params = new URLSearchParams(location.search);

let certified = { ...FALLBACK_CERTIFIED };
let nonClaims = [...FALLBACK_NON_CLAIMS];
let servedTrajectory = null;
let servedOffset = null;
let livePrePost = null;
let runtimeState = {
  kind: "fallback",
  label: "Fallback",
  summary: "recorded surprise trajectory and certified held-out evidence.",
};

async function loadServedBundle() {
  try {
    const [cardRes, trajectoryRes, offsetRes] = await Promise.all([
      fetch("./data/result_card.json", { cache: "no-store" }),
      fetch("./data/surprise_trajectory.json", { cache: "no-store" }),
      fetch("./fixtures/adapter_offset.json", { cache: "no-store" }),
    ]);
    if (cardRes.ok) {
      const card = await cardRes.json();
      if (card.schema === "lewm-surprise-result-card/1") {
        certified = {
          ...certified,
          baselineMse: Number(card.meanSurprisePre),
          adaptedMse: Number(card.meanSurprisePost),
          relativeImprovement: Number(card.thisRun),
          mean: Number(card.seedMean),
          worst: Number(card.seedWorst),
          worstSeed: Number(card.worstSeed),
          source: "lewm_tworooms_surprise.json · served result_card.json",
        };
        if (Array.isArray(card.nonClaims) && card.nonClaims.length) nonClaims = card.nonClaims;
      }
    }
    if (trajectoryRes.ok) {
      const trajectory = await trajectoryRes.json();
      if (trajectory.schema === "lewm-surprise-traj/1" && Array.isArray(trajectory.steps)) {
        servedTrajectory = trajectory.steps;
      }
    }
    if (offsetRes.ok) {
      const offset = await offsetRes.json();
      if (Array.isArray(offset) && offset.length === 12512 && offset.some((value) => Number(value) !== 0)) {
        servedOffset = offset;
      }
    }
  } catch {
    servedTrajectory = null;
  }
}

function preferredProviders() {
  const ep = params.get("ep");
  if (ep === "wasm" || ep === "webgpu") return [ep];
  return ["webgpu", "wasm"];
}

function engineMode() {
  const mode = params.get("engine");
  return mode === "live" || mode === "fallback" ? mode : "auto";
}

async function ensureOrt() {
  if (globalThis.ort) return globalThis.ort;
  const url =
    params.get("ort") ||
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.js";
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = url;
    script.async = true;
    script.onload = () => {
      if (globalThis.ort) resolve(globalThis.ort);
      else reject(new Error("ONNX Runtime script loaded without exposing global ort"));
    };
    script.onerror = () => reject(new Error(`failed to load ONNX Runtime from ${url}`));
    document.head.appendChild(script);
  });
}

async function withTimeout(promise, timeoutMs, label) {
  let timer = null;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs} ms`)), timeoutMs);
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    if (timer !== null) clearTimeout(timer);
  }
}

async function tryBuildLiveTrajectory() {
  if (engineMode() === "fallback") return null;
  if (!servedOffset) throw new Error("adapter offset fixture is missing or invalid");
  const ortApi = await ensureOrt();
  const runtime = await loadLewmRuntime({
    baseUrl: "../federated-demo/model/lewm-tworooms/",
    ortApi,
    preferredProviders: preferredProviders(),
  });
  const steps = Number.parseInt(params.get("steps") || "96", 10);
  const live = await buildLiveSurpriseTrajectory({
    runtime,
    offset: servedOffset,
    steps: Number.isFinite(steps) ? Math.max(24, Math.min(180, steps)) : 96,
  });
  const heldOut = await collectResidentPairs({
    runtime,
    seed: 991,
    episodes: 2,
    maxModelSteps: 10,
    policyOptions: { actionNoise: 1.0, actionRepeatProb: 0 },
    minPairs: 4,
  });
  livePrePost = buildPrePostStreams({
    pairs: heldOut.pairs,
    offset: servedOffset,
    inputDim: runtime.hidden,
  });
  runtimeState = {
    kind: "live",
    label: "Live",
    summary: `live ONNX replay (${runtime.backend ?? live.backend}, ${preferredProviders().join("+")}); browser held-out drop ${pct(livePrePost.surpriseDropRatioLive)} on ${livePrePost.count} pairs.`,
  };
  return live.steps;
}

await loadServedBundle();
try {
  const timeoutMs = Number.parseInt(params.get("liveTimeoutMs") || "20000", 10);
  const liveSteps = await withTimeout(
    tryBuildLiveTrajectory(),
    Number.isFinite(timeoutMs) ? timeoutMs : 20000,
    "live surprise-meter startup",
  );
  if (Array.isArray(liveSteps) && liveSteps.length) {
    servedTrajectory = liveSteps;
  } else if (engineMode() === "live") {
    throw new Error("live mode did not produce a trajectory");
  }
} catch (error) {
  if (engineMode() === "live") {
    throw error;
  }
  runtimeState = {
    kind: "fallback",
    label: "Fallback",
    summary: `recorded trajectory fallback; live ONNX unavailable (${error.message}).`,
  };
}

const engine = new SurpriseEngine({ certified, trajectory: servedTrajectory });
const surpRec = new Recorder($("surpriseCanvas"), {
  color: TRACE_INK, spikes: true, max: 0.30, weight: 2.0,
  baseline: certified.baselineMse, preBaseline: certified.baselineMse,
});
const fdRec = new Recorder($("frameDiffCanvas"), { color: GRAPHITE, spikes: false, max: 1.0, weight: 1.6 });

// ── the TwoRooms world view ───────────────────────────────────────────────
const world = (() => {
  const canvas = $("worldCanvas"), ctx = canvas.getContext("2d");
  let w = 0, h = 0;
  function ensure() {
    const dpr = window.devicePixelRatio || 1, r = canvas.getBoundingClientRect();
    if (!r.width || !r.height) return false;
    if (canvas.width !== Math.round(r.width * dpr) || canvas.height !== Math.round(r.height * dpr)) {
      canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); w = r.width; h = r.height;
    }
    return true;
  }
  const trail = [];
  function draw(st) {
    if (!ensure()) return;
    const pad = 16, X = (x) => pad + x * (w - 2 * pad), Y = (y) => pad + y * (h - 2 * pad);
    ctx.clearRect(0, 0, w, h);
    const rooms = mockEnv.ROOMS, L = rooms.left, Rr = rooms.right, d = rooms.door;
    // faint engraved floor grid — reads as a drafted plate even when the agent is still
    ctx.strokeStyle = "rgba(37,28,22,0.06)"; ctx.lineWidth = 1; ctx.beginPath();
    for (const m of [L, Rr]) {
      for (let gx = m.x0 + 0.07; gx < m.x1 - 0.01; gx += 0.085) { ctx.moveTo(X(gx), Y(m.y0)); ctx.lineTo(X(gx), Y(m.y1)); }
      for (let gy = m.y0 + 0.09; gy < m.y1 - 0.01; gy += 0.13) { ctx.moveTo(X(m.x0), Y(gy)); ctx.lineTo(X(m.x1), Y(gy)); }
    }
    ctx.stroke();
    // room walls — the inner facing walls are BROKEN at the doorway (a real opening)
    ctx.strokeStyle = INK2; ctx.lineWidth = 1.3; ctx.beginPath();
    ctx.moveTo(X(L.x0), Y(L.y0)); ctx.lineTo(X(L.x1), Y(L.y0)); ctx.lineTo(X(L.x1), Y(d.y0)); // top + inner-upper
    ctx.moveTo(X(L.x1), Y(d.y1)); ctx.lineTo(X(L.x1), Y(L.y1)); ctx.lineTo(X(L.x0), Y(L.y1)); // inner-lower + bottom
    ctx.lineTo(X(L.x0), Y(L.y0)); // outer
    ctx.moveTo(X(Rr.x1), Y(Rr.y0)); ctx.lineTo(X(Rr.x0), Y(Rr.y0)); ctx.lineTo(X(Rr.x0), Y(d.y0));
    ctx.moveTo(X(Rr.x0), Y(d.y1)); ctx.lineTo(X(Rr.x0), Y(Rr.y1)); ctx.lineTo(X(Rr.x1), Y(Rr.y1));
    ctx.lineTo(X(Rr.x1), Y(Rr.y0));
    ctx.stroke();
    // doorway threshold rails connecting the openings
    ctx.strokeStyle = "rgba(37,28,22,0.2)"; ctx.lineWidth = 1; ctx.beginPath();
    ctx.moveTo(X(L.x1), Y(d.y0)); ctx.lineTo(X(Rr.x0), Y(d.y0));
    ctx.moveTo(X(L.x1), Y(d.y1)); ctx.lineTo(X(Rr.x0), Y(d.y1));
    ctx.stroke();
    // trail
    if (st.running) { trail.push(st.agent); if (trail.length > 46) trail.shift(); }
    ctx.strokeStyle = "rgba(34,47,60,0.35)"; ctx.lineWidth = 1.4; ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.beginPath();
    trail.forEach((p, i) => { const x = X(p.x), y = Y(p.y); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
    // agent
    const ax = X(st.agent.x), ay = Y(st.agent.y);
    if (st.hot) { // registration crosshair marks the surprised step (instrument, not radar ping)
      ctx.strokeStyle = "rgba(150,40,43,0.55)"; ctx.lineWidth = 1; ctx.beginPath();
      ctx.moveTo(ax - 12, ay); ctx.lineTo(ax - 5, ay); ctx.moveTo(ax + 5, ay); ctx.lineTo(ax + 12, ay);
      ctx.moveTo(ax, ay - 12); ctx.lineTo(ax, ay - 5); ctx.moveTo(ax, ay + 5); ctx.lineTo(ax, ay + 12);
      ctx.stroke(); }
    ctx.fillStyle = OXBLOOD; ctx.beginPath(); ctx.arc(ax, ay, 5, 0, Math.PI * 2); ctx.fill();
  }
  return { draw };
})();

// ── HUD: certified ledger + non-claims, rendered from data ────────────────
function fillHUD() {
  $("certLedger").innerHTML =
    `<tr class="is-run"><td>this run</td><td class="num">${pct(certified.relativeImprovement)}</td></tr>` +
    `<tr><td>mean · ${certified.seeds} seeds</td><td class="num">${pct(certified.mean)}</td></tr>` +
    `<tr class="is-worst"><td>worst · seed ${certified.worstSeed}</td><td class="num">${pct(certified.worst)}</td></tr>`;
  $("certSrc").textContent = "Source — " + certified.source + " · all seeds improved, no collapse";
  if (livePrePost) {
    $("certSrc").textContent += ` · browser held-out check ${pct(livePrePost.surpriseDropRatioLive)} on ${livePrePost.count} pairs`;
  }
  $("nonClaims").innerHTML = nonClaims.map((c) => `<li>${c}</li>`).join("");
  $("runtimeTag").textContent = runtimeState.label;
  $("runtimeSummary").textContent = runtimeState.summary;
  $("srcLine").textContent = ` Adapter ${MODEL.latentDim}-d latent · ~${MODEL.msPerStep} ms/step.`;
}

// ── controls ──────────────────────────────────────────────────────────────
const readout = document.querySelector(".readout");
let shownSurprise = certified.baselineMse, surprised = false;

function setMode(mode) {
  engine.setMode(mode);
  surpRec.setBaseline(engine.baselineLevel());
  $("modeLabel").textContent = mode === "post" ? "post-federation" : "pre-federation";
  if (mode === "post") {
    const certifiedDrop = `certified held-out error ↓ ${pct(certified.relativeImprovement)} vs pre-federation`;
    $("fedDelta").textContent = livePrePost
      ? `${certifiedDrop}; browser spot-check ${pct(livePrePost.surpriseDropRatioLive)} (${livePrePost.count} pairs)`
      : certifiedDrop;
  } else {
    $("fedDelta").textContent = "";
  }
  document.querySelectorAll(".seg-btn").forEach((b) => b.classList.toggle("is-on", b.dataset.mode === mode));
}
document.querySelectorAll(".seg-btn").forEach((b) =>
  b.addEventListener("click", () => setMode(b.dataset.mode)));

document.querySelectorAll("[data-perturb]").forEach((b) =>
  b.addEventListener("click", () => {
    engine.perturb(b.dataset.perturb);
    if (!engine.running) setRunning(true);
    b.classList.add("is-armed");
    setTimeout(() => b.classList.remove("is-armed"), 650);
  }));

function setRunning(run) {
  engine.running = run;
  const btn = $("playToggle");
  btn.textContent = run ? "PAUSE" : "PLAY";
  btn.setAttribute("aria-label", run ? "pause" : "resume");
}
$("playToggle").addEventListener("click", () => setRunning(!engine.running));

window.addEventListener("resize", () => { surpRec.resize(); fdRec.resize(); });

// ── loop ───────────────────────────────────────────────────────────────────
let last = null;
function frame(ts) {
  if (last === null) last = ts;
  const dt = Math.min(0.05, (ts - last) / 1000); last = ts;
  if (!surpRec.sized) surpRec.resize();
  if (!fdRec.sized) fdRec.resize();

  if (holdType) { engine.event.type = holdType; engine.event.energy = 0.2; } // ?hold= pins a spike
  const st = engine.tick(dt);
  surpRec.push(st.surprise, st.hot);
  fdRec.push(st.frameDiff, false);
  surpRec.draw(); fdRec.draw();
  world.draw(st);

  // smoothed readout with hysteresis on the calm/surprised state
  shownSurprise += (st.surprise - shownSurprise) * 0.18;
  $("surpriseNow").textContent = shownSurprise.toFixed(3);
  const hi = st.baseline * 2.1, lo = st.baseline * 1.5;
  if (!surprised && shownSurprise > hi) surprised = true;
  else if (surprised && shownSurprise < lo) surprised = false;
  readout.classList.toggle("is-surprised", surprised);
  $("surpriseState").textContent = surprised ? "surprised" : "calm";

  requestAnimationFrame(frame);
}

const autoType = params.get("auto");
const holdType = params.get("hold");

fillHUD();
globalThis.__surpriseMeter = {
  runtimeState,
  livePrePost,
  certified,
  trajectoryKind: runtimeState.kind,
  trajectorySteps: servedTrajectory?.length ?? 0,
};
// warm the recorders so the paper is already inked, and leave the engine mid-stride
// (no reset — resetting would zero frameDiff and leave a one-frame dip at the seam).
for (let i = 0; i < 260; i++) { const s = engine.tick(1 / 60); surpRec.push(s.surprise, false); fdRec.push(s.frameDiff, false); }
setMode(params.get("mode") === "post" ? "post" : "pre");
if (reduceMotion && !autoType && !holdType) setRunning(false); // honour the preference; waits for ▶
if (autoType) setInterval(() => document.querySelector(`[data-perturb="${autoType}"]`)?.click(), 1800);
requestAnimationFrame(frame);
