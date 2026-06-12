// TwoRooms real-LeWM lab: checkpoint-backed rollout + planning view (#318, epic #314).
//
// The lab object is UI-free and node-testable with an injected runtime (see lewm_runtime.mjs);
// mountTwoRoomsLab wires it to DOM canvases and readouts. Everything shown is produced by the
// exported checkpoint graphs — current frame, goal frame, candidate rollout costs, the chosen
// action block, latent goal distance, model identity, runtime backend, and explicit errors.

import {
  ACTION_BLOCK,
  ACTION_DIM,
  IMG_SIZE,
  TWOROOMS_DEVIATIONS,
  distanceToTarget,
  frameToModelInput,
  renderFrameRGB,
  renderGoalFrameRGB,
  rgbToRGBA,
  sampleEpisode,
  stepEpisodeBlock,
} from "./tworooms_env.mjs";
import { mulberry32 } from "./rng.mjs";

// ---------------------------------------------------------------------------
// UI-free planning lab (node-testable)
// ---------------------------------------------------------------------------

export async function createTwoRoomsLab({ runtime, seed = 1, planOptions = {} }) {
  const rng = mulberry32(seed >>> 0);
  const state = {
    episode: null,
    historyLatents: [],
    historyActionBlocks: [],
    goalLatent: null,
    stepsPlanned: 0,
    lastPlan: null,
  };

  async function encodeEpisodeFrame() {
    const rgb = renderFrameRGB(state.episode.agent);
    const latent = await runtime.encodeFrames(frameToModelInput(rgb), 1);
    return { rgb, latent: Float32Array.from(latent.subarray(0, runtime.hidden)) };
  }

  async function reset() {
    state.episode = sampleEpisode(rng);
    state.stepsPlanned = 0;
    state.lastPlan = null;
    state.historyActionBlocks = [];
    const { latent } = await encodeEpisodeFrame();
    state.historyLatents = [latent];
    const goalRgb = renderGoalFrameRGB(state.episode.target);
    const goal = await runtime.encodeFrames(frameToModelInput(goalRgb), 1);
    state.goalLatent = Float32Array.from(goal.subarray(0, runtime.hidden));
    return snapshot();
  }

  async function planStep() {
    if (!state.episode) await reset();
    if (state.episode.done) return snapshot();
    const plan = await runtime.planAction({
      historyLatents: state.historyLatents,
      historyActionBlocks: state.historyActionBlocks,
      goalLatent: state.goalLatent,
      rng,
      ...planOptions,
    });
    state.episode = stepEpisodeBlock(state.episode, plan.actionBlock);
    const { latent } = await encodeEpisodeFrame();
    state.historyLatents.push(latent);
    state.historyActionBlocks.push(Float32Array.from(plan.actionBlock));
    while (state.historyLatents.length > runtime.numFrames) state.historyLatents.shift();
    while (state.historyActionBlocks.length > runtime.numFrames - 1) state.historyActionBlocks.shift();
    state.stepsPlanned += 1;
    state.lastPlan = plan;
    return snapshot();
  }

  function snapshot() {
    return {
      agent: state.episode ? { ...state.episode.agent } : null,
      target: state.episode ? { ...state.episode.target } : null,
      distance: state.episode ? distanceToTarget(state.episode) : null,
      done: state.episode?.done ?? false,
      stepsPlanned: state.stepsPlanned,
      envSteps: state.episode?.steps ?? 0,
      historyLength: state.historyLatents.length,
      lastPlan: state.lastPlan
        ? {
            cost: state.lastPlan.cost,
            bestCost: Math.min(...state.lastPlan.candidateCosts),
            worstCost: Math.max(...state.lastPlan.candidateCosts),
            candidates: state.lastPlan.candidateCosts.length,
            actionBlock: Array.from(state.lastPlan.actionBlock),
          }
        : null,
    };
  }

  return { reset, planStep, snapshot, state };
}

// ---------------------------------------------------------------------------
// DOM panel
// ---------------------------------------------------------------------------

function drawFrame(canvas, rgb) {
  const ctx = canvas.getContext("2d", { alpha: false });
  if (!ctx) return;
  ctx.imageSmoothingEnabled = false;
  ctx.putImageData(new ImageData(rgbToRGBA(rgb), IMG_SIZE, IMG_SIZE), 0, 0);
}

export function mountTwoRoomsLab(container, { loadRuntime, el }) {
  const statusNote = el("p", { class: "note", text: "Loading checkpoint-backed LeWM graphs…" });
  const identityNote = el("p", { class: "note", text: "" });
  const planNote = el("p", { class: "note", text: "No plan yet." });
  const errorNote = el("p", { class: "note error", text: "" });
  const currentCanvas = el("canvas", { width: String(IMG_SIZE), height: String(IMG_SIZE) });
  const goalCanvas = el("canvas", { width: String(IMG_SIZE), height: String(IMG_SIZE) });

  let lab = null;
  let runtime = null;
  let autoTimer = null;

  function redraw() {
    if (!lab?.state.episode) return;
    drawFrame(currentCanvas, renderFrameRGB(lab.state.episode.agent, { renderTarget: true, targetPos: lab.state.episode.target }));
    drawFrame(goalCanvas, renderGoalFrameRGB(lab.state.episode.target));
  }

  function describe(snap) {
    if (!snap?.lastPlan) {
      planNote.textContent = `distance=${snap?.distance?.toFixed(1) ?? "?"} px; plan a step to see candidate rollouts.`;
      return;
    }
    const p = snap.lastPlan;
    planNote.textContent =
      `plan ${snap.stepsPlanned}: ${p.candidates} candidate rollouts, terminal latent cost ` +
      `best=${p.bestCost.toFixed(2)} chosen=${p.cost.toFixed(2)} worst=${p.worstCost.toFixed(2)}; ` +
      `chosen block=[${p.actionBlock.slice(0, 4).map((v) => v.toFixed(2)).join(", ")}…]; ` +
      `distance=${snap.distance.toFixed(1)} px${snap.done ? " — target reached" : ""}`;
  }

  async function step() {
    try {
      const snap = await lab.planStep();
      redraw();
      describe(snap);
      if (snap.done && autoTimer) toggleAuto();
    } catch (error) {
      errorNote.textContent = String(error.message ?? error);
      if (autoTimer) toggleAuto();
    }
  }

  function toggleAuto() {
    if (autoTimer) {
      clearInterval(autoTimer);
      autoTimer = null;
      autoButton.textContent = "Auto-plan";
      return;
    }
    autoTimer = setInterval(step, 250);
    autoButton.textContent = "Stop auto-plan";
  }

  const resetButton = el("button", {
    text: "New episode",
    onclick: async () => {
      try {
        const snap = await lab.reset();
        redraw();
        describe(snap);
        errorNote.textContent = "";
      } catch (error) {
        errorNote.textContent = String(error.message ?? error);
      }
    },
  });
  const stepButton = el("button", { text: "Plan + step", onclick: step });
  const autoButton = el("button", { text: "Auto-plan", onclick: toggleAuto });

  container.append(
    el("section", { class: "panel" }, [
      el("h2", { text: "TwoRooms — checkpoint-backed LeWM rollout & planning" }),
      note_(el, "Tapestry-like real-LeWM mode component: the frames below are encoded by the exported "
        + "TwoRooms checkpoint graphs; planning samples action candidates, rolls latents forward with "
        + "the real predictor, and picks the lowest terminal goal-latent distance. Inference only — "
        + "browser-local adaptation is the next stage."),
      note_(el, TWOROOMS_DEVIATIONS),
      statusNote,
      identityNote,
      el("div", { class: "columns" }, [
        el("div", { class: "panel" }, [el("h2", { text: "Current (agent + target)" }), currentCanvas]),
        el("div", { class: "panel" }, [el("h2", { text: "Goal frame (model input)" }), goalCanvas]),
      ]),
      el("div", {}, [resetButton, stepButton, autoButton]),
      planNote,
      errorNote,
    ]),
  );

  (async () => {
    try {
      runtime = await loadRuntime();
      lab = await createTwoRoomsLab({ runtime, seed: 20260612 });
      statusNote.textContent =
        `Runtime ready: ${runtime.runtime} on ${runtime.backend}; window=${runtime.numFrames} frames, ` +
        `action block=${ACTION_BLOCK}×${ACTION_DIM}.`;
      const id = runtime.identity;
      identityNote.textContent =
        `Model: ${id.checkpointRepo}@${String(id.checkpointRevision).slice(0, 12)} ` +
        `(weights ${String(id.weightsSha256).slice(0, 12)}…, graph v${id.graphVersion}, opset ${id.opset}).`;
      const snap = await lab.reset();
      redraw();
      describe(snap);
    } catch (error) {
      statusNote.textContent = "Real-LeWM runtime unavailable.";
      errorNote.textContent = String(error.message ?? error);
    }
  })();

  return { stepOnce: step };
}

function note_(el, text) {
  return el("p", { class: "note", text });
}
