// Shared run/participant lifecycle vocabulary for the Lensemble federated demo.
//
// The frontend-only simulator (sim_engine.mjs) and the backend run-orchestration
// API (lensemble/demo/lifecycle.py) emit the same states and event kinds so the
// UI renders identically in both modes. Keep the two lists in sync; the
// cross-language contract is pinned by tests/ml/test_federated_demo_app.py and
// tests/unit/test_demo_lifecycle.py.

export const RUN_STATES = Object.freeze([
  "created",
  "joining",
  "ready",
  "running_round",
  "aggregating",
  "checkpoint_ready",
  "inference_ready",
  "completed",
  "aborted",
  "failed",
]);

export const RUN_TRANSITIONS = Object.freeze({
  created: ["joining", "aborted", "failed"],
  joining: ["ready", "aborted", "failed"],
  ready: ["running_round", "aborted", "failed"],
  running_round: ["aggregating", "aborted", "failed"],
  aggregating: ["running_round", "checkpoint_ready", "aborted", "failed"],
  checkpoint_ready: ["inference_ready", "aborted", "failed"],
  inference_ready: ["completed", "aborted", "failed"],
  completed: [],
  aborted: [],
  failed: [],
});

export const TERMINAL_RUN_STATES = Object.freeze(["completed", "aborted", "failed"]);

export const PARTICIPANT_STATES = Object.freeze([
  "joined",
  "ready",
  "assigned",
  "training",
  "submitted",
  "completed",
  "dropped",
  "error",
]);

export const PARTICIPANT_TRANSITIONS = Object.freeze({
  joined: ["ready", "dropped", "error"],
  ready: ["assigned", "dropped", "error"],
  assigned: ["training", "dropped", "error"],
  training: ["submitted", "dropped", "error"],
  submitted: ["assigned", "completed", "dropped", "error"],
  completed: [],
  dropped: [],
  error: [],
});

// Dotted event vocabulary, mirroring CoordinatorServiceEvent naming
// (lensemble/federation/service.py) so the admin timeline reads the same in
// simulator and coordinator-backed modes.
export const EVENT_KINDS = Object.freeze([
  "run.created",
  "run.state",
  "run.aborted",
  "run.failed",
  "connection.opened",
  "connection.closed",
  "participant.joined",
  "participant.resumed",
  "participant.rejected",
  "participant.ready",
  "participant.heartbeat",
  "participant.stale",
  "participant.dropped",
  "participant.error",
  "round.assigned",
  "participant.training",
  "update.submitted",
  "update.rejected",
  "round.aggregating",
  "round.closed",
  "round.retry",
  "checkpoint.ready",
  "inference.ready",
  "run.completed",
]);

export function canTransitionRun(from, to) {
  return (RUN_TRANSITIONS[from] ?? []).includes(to);
}

export function canTransitionParticipant(from, to) {
  return (PARTICIPANT_TRANSITIONS[from] ?? []).includes(to);
}

export function assertRunTransition(from, to) {
  if (!canTransitionRun(from, to)) {
    throw new Error(`invalid run transition: ${from} -> ${to}`);
  }
  return to;
}

export function assertParticipantTransition(from, to) {
  if (!canTransitionParticipant(from, to)) {
    throw new Error(`invalid participant transition: ${from} -> ${to}`);
  }
  return to;
}

export function isTerminalRunState(state) {
  return TERMINAL_RUN_STATES.includes(state);
}
