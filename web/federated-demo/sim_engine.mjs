// Frontend-only federated run simulator (#295).
//
// Everything here is mocked in-browser: there is no backend, no network, and
// no real training. Run state, participants, rounds, and artifacts are local
// JavaScript objects driven by a deterministic seeded RNG so the lifecycle is
// reproducible and node-testable. The backend API mode (#296) emits the same
// run states and event kinds; the UI cannot tell the difference by design.

import {
  assertParticipantTransition,
  assertRunTransition,
  isTerminalRunState,
} from "./lifecycle.mjs";
import { mulberry32, newParticipantId, newRunId, pseudoHash, randomToken } from "./rng.mjs";

export const SIMULATOR_MODE = "frontend-simulator";
export const DEMO_PRESETS = Object.freeze(["swipe-dot-tiny"]);
export const RUN_SCHEMA = "demo-run/1";

export class SimError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

function validateConfig(config) {
  const maxParticipants = Number(config?.maxParticipants);
  const quorum = Number(config?.quorum);
  const rounds = Number(config?.rounds);
  const preset = config?.preset ?? DEMO_PRESETS[0];
  if (!Number.isInteger(maxParticipants) || maxParticipants < 1 || maxParticipants > 64) {
    throw new SimError("invalid_config", "maxParticipants must be an integer in [1, 64]");
  }
  if (!Number.isInteger(quorum) || quorum < 1 || quorum > maxParticipants) {
    throw new SimError("invalid_config", "quorum must be an integer in [1, maxParticipants]");
  }
  if (!Number.isInteger(rounds) || rounds < 1 || rounds > 1000) {
    throw new SimError("invalid_config", "rounds must be an integer in [1, 1000]");
  }
  if (!DEMO_PRESETS.includes(preset)) {
    throw new SimError("invalid_config", `unknown preset: ${preset}`);
  }
  return { maxParticipants, quorum, rounds, preset };
}

function emit(run, kind, message, extra = {}) {
  const event = {
    seq: run.events.length,
    at: run.clock,
    kind,
    severity: extra.severity ?? "info",
    message,
    runState: run.state,
    round: extra.round ?? run.round,
    participantId: extra.participantId ?? null,
    payload: extra.payload ?? null,
    simulated: true,
  };
  run.events.push(event);
  return event;
}

function setRunState(run, to, message, extra = {}) {
  assertRunTransition(run.state, to);
  run.state = to;
  emit(run, extra.kind ?? "run.state", message, extra);
}

function setParticipantState(run, participant, to) {
  assertParticipantTransition(participant.state, to);
  participant.state = to;
}

function attachRng(run, burnedCalls = 0) {
  const base = mulberry32(run.seed >>> 0);
  for (let i = 0; i < burnedCalls; i += 1) base();
  run.rngCalls = burnedCalls;
  run._nextRandom = () => {
    run.rngCalls += 1;
    return base();
  };
}

export function createSimRun(config, seed) {
  const cfg = validateConfig(config);
  const run = {
    schema: RUN_SCHEMA,
    mode: SIMULATOR_MODE,
    id: null,
    seed: seed >>> 0,
    config: cfg,
    state: "created",
    round: 0,
    clock: 0,
    participants: [],
    events: [],
    artifacts: [],
    abortReason: null,
    failureReason: null,
  };
  attachRng(run, 0);
  run.id = newRunId(run._nextRandom);
  emit(run, "run.created", `simulated run ${run.id} created`, {
    payload: { maxParticipants: cfg.maxParticipants, quorum: cfg.quorum, rounds: cfg.rounds, preset: cfg.preset },
  });
  return run;
}

// Rebuild a live engine object from a stored snapshot (host tab reload).
export function restoreSimRun(snapshot) {
  if (!snapshot || snapshot.schema !== RUN_SCHEMA || snapshot.mode !== SIMULATOR_MODE) {
    return null;
  }
  const run = {
    schema: snapshot.schema,
    mode: snapshot.mode,
    id: snapshot.id,
    seed: snapshot.seed >>> 0,
    config: snapshot.config,
    state: snapshot.state,
    round: snapshot.round,
    clock: snapshot.clock ?? 0,
    participants: snapshot.participants ?? [],
    events: snapshot.events ?? [],
    artifacts: snapshot.artifacts ?? [],
    abortReason: snapshot.abortReason ?? null,
    failureReason: snapshot.failureReason ?? null,
  };
  attachRng(run, snapshot.rngCalls ?? 0);
  return run;
}

export function joinSimRun(run, displayName = null) {
  if (isTerminalRunState(run.state)) {
    throw new SimError("run_closed", `run is ${run.state}; joining is closed`);
  }
  if (!["created", "joining", "ready"].includes(run.state)) {
    throw new SimError("already_started", "run already started; joining is closed");
  }
  if (run.participants.length >= run.config.maxParticipants) {
    emit(run, "participant.rejected", "join rejected: run is full", {
      severity: "warn",
      payload: { reason: "run_full" },
    });
    throw new SimError("run_full", "run is full");
  }
  const participant = {
    id: newParticipantId(run._nextRandom),
    displayName: displayName || null,
    state: "joined",
    joinedAt: run.clock,
    round: null,
    progress: 0,
    submittedRounds: [],
    error: null,
    simulated: true,
  };
  run.participants.push(participant);
  emit(run, "participant.joined", `participant ${participant.id} joined`, {
    participantId: participant.id,
    payload: { slot: run.participants.length, of: run.config.maxParticipants },
  });
  if (run.state === "created") {
    setRunState(run, "joining", "first participant joined");
  }
  setParticipantState(run, participant, "ready");
  emit(run, "participant.ready", `participant ${participant.id} is ready`, {
    participantId: participant.id,
  });
  if (run.state === "joining" && activeParticipants(run).length >= run.config.quorum) {
    setRunState(run, "ready", `quorum of ${run.config.quorum} reached`);
  }
  return participant;
}

export function activeParticipants(run) {
  return run.participants.filter((p) => !["dropped", "error"].includes(p.state));
}

export function startSimRun(run) {
  if (run.state !== "ready") {
    throw new SimError(
      "not_ready",
      run.state === "joining" || run.state === "created"
        ? "quorum not reached yet"
        : `cannot start from state ${run.state}`,
    );
  }
  beginRound(run, 1);
}

function beginRound(run, roundIndex) {
  run.round = roundIndex;
  setRunState(run, "running_round", `round ${roundIndex} of ${run.config.rounds} started`, {
    round: roundIndex,
  });
  for (const participant of activeParticipants(run)) {
    setParticipantState(run, participant, "assigned");
    participant.round = roundIndex;
    participant.progress = 0;
    emit(run, "round.assigned", `round ${roundIndex} assigned to ${participant.id}`, {
      participantId: participant.id,
      round: roundIndex,
    });
    setParticipantState(run, participant, "training");
    emit(run, "participant.training", `${participant.id} simulating local work`, {
      participantId: participant.id,
      round: roundIndex,
    });
  }
}

// Advance the mocked timeline one deterministic step. Returns true while the
// run still has pending simulated work.
export function simTick(run) {
  if (isTerminalRunState(run.state)) {
    return false;
  }
  run.clock += 1;
  if (run.state === "running_round") {
    const training = activeParticipants(run).filter((p) => p.state === "training");
    if (training.length > 0) {
      for (const participant of training) {
        participant.progress = Math.min(1, participant.progress + 0.34);
      }
      const finishing = training.filter((p) => p.progress >= 1);
      for (const participant of finishing) {
        setParticipantState(run, participant, "submitted");
        participant.submittedRounds.push(run.round);
        emit(run, "update.submitted", `${participant.id} submitted a simulated update`, {
          participantId: participant.id,
          payload: { l2Norm: Number((run._nextRandom() * 2).toFixed(6)), simulated: true },
        });
      }
      if (activeParticipants(run).some((p) => p.state === "training")) {
        return true;
      }
    }
    setRunState(run, "aggregating", `aggregating round ${run.round} (simulated)`, {
      kind: "round.aggregating",
    });
    return true;
  }
  if (run.state === "aggregating") {
    const checkpointHash = pseudoHash(run._nextRandom);
    run.artifacts.push({
      kind: "checkpoint",
      label: `round-${run.round} simulated checkpoint`,
      round: run.round,
      sha256: checkpointHash,
      simulated: true,
    });
    emit(run, "round.closed", `round ${run.round} closed (simulated aggregation)`, {
      payload: { checkpoint: checkpointHash.slice(0, 12), contributing: activeParticipants(run).length },
    });
    if (run.round < run.config.rounds) {
      beginRound(run, run.round + 1);
      return true;
    }
    for (const participant of activeParticipants(run)) {
      if (participant.state === "submitted") {
        setParticipantState(run, participant, "completed");
      }
    }
    setRunState(run, "checkpoint_ready", "final simulated checkpoint ready", {
      kind: "checkpoint.ready",
    });
    return true;
  }
  if (run.state === "checkpoint_ready") {
    run.artifacts.push({
      kind: "inference-model",
      label: "simulated inference artifact",
      round: run.round,
      sha256: pseudoHash(run._nextRandom),
      simulated: true,
    });
    setRunState(run, "inference_ready", "simulated inference artifact ready", {
      kind: "inference.ready",
    });
    return true;
  }
  if (run.state === "inference_ready") {
    setRunState(run, "completed", "simulated run completed", { kind: "run.completed" });
    return false;
  }
  return !isTerminalRunState(run.state);
}

export function abortSimRun(run, reason = "host abort") {
  if (isTerminalRunState(run.state)) {
    throw new SimError("run_closed", `run is already ${run.state}`);
  }
  run.abortReason = reason;
  assertRunTransition(run.state, "aborted");
  run.state = "aborted";
  emit(run, "run.aborted", `run aborted: ${reason}`, { severity: "warn" });
}

export function failSimRun(run, reason = "simulated failure") {
  if (isTerminalRunState(run.state)) {
    throw new SimError("run_closed", `run is already ${run.state}`);
  }
  run.failureReason = reason;
  assertRunTransition(run.state, "failed");
  run.state = "failed";
  for (const participant of run.participants) {
    if (!["completed", "dropped", "error"].includes(participant.state)) {
      participant.state = "error";
      participant.error = reason;
    }
  }
  emit(run, "run.failed", `run failed: ${reason}`, { severity: "error" });
}

export function dropSimParticipant(run, participantId, reason = "left") {
  const participant = run.participants.find((p) => p.id === participantId);
  if (!participant) {
    throw new SimError("unknown_participant", `no participant ${participantId}`);
  }
  if (["completed", "dropped", "error"].includes(participant.state)) {
    return participant;
  }
  participant.state = "dropped";
  emit(run, "participant.dropped", `${participant.id} dropped: ${reason}`, {
    participantId: participant.id,
    severity: "warn",
  });
  return participant;
}

export function runSummary(run) {
  return {
    id: run.id,
    mode: run.mode,
    state: run.state,
    round: run.round,
    rounds: run.config.rounds,
    participants: run.participants.map((p) => ({
      id: p.id,
      state: p.state,
      round: p.round,
      progress: p.progress,
    })),
    artifacts: run.artifacts.length,
    events: run.events.length,
  };
}

export function newSessionToken() {
  // Local-only pseudo token so the simulator join URL shape matches backend mode.
  const rng = mulberry32(randomSeedFallback());
  return `sim-${randomToken(rng, 12)}`;
}

function randomSeedFallback() {
  if (globalThis.crypto?.getRandomValues) {
    const buf = new Uint32Array(1);
    globalThis.crypto.getRandomValues(buf);
    return buf[0] >>> 0;
  }
  return Math.floor(Math.random() * 4294967296) >>> 0;
}
