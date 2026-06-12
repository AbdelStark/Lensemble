// TwoRooms browser environment for the Tapestry-like real-LeWM mode (#318, epic #314).
//
// A deterministic JavaScript port of the upstream stable-worldmodel TwoRoom environment
// (stable_worldmodel/envs/two_room/env.py) at the *released default variation values* — the
// configuration the quentinll/lewm-tworooms checkpoint's expert dataset was collected with:
// 224x224 frames, white background, black vertical wall at x=112 (thickness 10), one white door
// centered at y=49 (half-extent 14), black border lines, red Gaussian agent dot (radius 7),
// green target dot, agent speed 5 px/step, success when within 16 px of the target.
//
// Known deviations from upstream (documented per the #318 claim boundary and carried into
// evidence): only the default variation values are implemented (no color/radius/wall-axis/door
// resampling), rendering uses JS floats instead of torch tensors (sub-pixel float noise), and the
// dataset's expert policy is reimplemented from the upstream source rather than imported. This is
// a TwoRooms-compatible visualization and rollout probe, not the upstream torch environment.

export const TWOROOMS_DEVIATIONS =
  "TwoRooms-compatible JS probe: released default variations only (vertical wall, single door, " +
  "default colors/radii/speed); float canvas rendering, reimplemented expert policy. Not the " +
  "upstream torch environment.";

export const IMG_SIZE = 224;
export const BORDER_SIZE = 14;
export const BORDER_LINE = 4; // border stroke thickness (upstream t=4)
export const WALL_CENTER = 112;
export const WALL_THICKNESS = 10; // => half-extent 5 (upstream integer division)
export const WALL_HALF = Math.floor(WALL_THICKNESS / 2);
export const DOOR_CENTER_Y = 49;
export const DOOR_HALF = 14;
export const DOOR_MARGIN = 1.75;
export const AGENT_RADIUS = 7;
export const TARGET_RADIUS = 7;
export const AGENT_SPEED = 5.0;
export const SUCCESS_DISTANCE = 16.0;
export const ACTION_BLOCK = 5; // checkpoint frameskip: one model step = 5 env steps
export const ACTION_DIM = 2;

const POS_MIN = BORDER_SIZE; // variation-space position bounds
const POS_MAX = IMG_SIZE - BORDER_SIZE - 1;

const COLORS = Object.freeze({
  background: [255, 255, 255],
  wall: [0, 0, 0],
  border: [0, 0, 0],
  door: [255, 255, 255],
  agent: [255, 0, 0],
  target: [0, 255, 0],
});

export function clamp(value, lo, hi) {
  return Math.min(Math.max(value, lo), hi);
}

function inDoor(y, margin = DOOR_MARGIN) {
  return y >= DOOR_CENTER_Y - DOOR_HALF - margin && y <= DOOR_CENTER_Y + DOOR_HALF + margin;
}

// ---------------------------------------------------------------------------
// episode sampling (mirrors the dataset's DEFAULT_VARIATIONS: agent + target positions)
// ---------------------------------------------------------------------------

export function sampleAgentPosition(rng) {
  // uniform in the variation box, rejecting the wall zone inflated by the agent radius
  for (;;) {
    const x = POS_MIN + rng() * (POS_MAX - POS_MIN);
    const y = POS_MIN + rng() * (POS_MAX - POS_MIN);
    const wallMin = WALL_CENTER - WALL_HALF - AGENT_RADIUS;
    const wallMax = WALL_CENTER + WALL_HALF + AGENT_RADIUS;
    if (x < wallMin || x > wallMax) return { x, y };
  }
}

export function sampleTargetPosition(rng) {
  // upstream leaves the target unconstrained inside the variation box
  return {
    x: POS_MIN + rng() * (POS_MAX - POS_MIN),
    y: POS_MIN + rng() * (POS_MAX - POS_MIN),
  };
}

export function sampleEpisode(rng) {
  return {
    agent: sampleAgentPosition(rng),
    target: sampleTargetPosition(rng),
    steps: 0,
    done: false,
  };
}

export function distanceToTarget(episode) {
  const dx = episode.agent.x - episode.target.x;
  const dy = episode.agent.y - episode.target.y;
  return Math.hypot(dx, dy);
}

// ---------------------------------------------------------------------------
// dynamics (faithful to upstream _apply_collisions for the default vertical wall)
// ---------------------------------------------------------------------------

export function stepAgent(pos, action) {
  const ax = clamp(action[0], -1, 1);
  const ay = clamp(action[1], -1, 1);
  let x = pos.x + ax * AGENT_SPEED;
  let y = pos.y + ay * AGENT_SPEED;

  // border clamp accounting for agent radius
  x = clamp(x, BORDER_SIZE + AGENT_RADIUS, IMG_SIZE - BORDER_SIZE - AGENT_RADIUS);
  y = clamp(y, BORDER_SIZE + AGENT_RADIUS, IMG_SIZE - BORDER_SIZE - AGENT_RADIUS);

  // central vertical wall with the agent-radius-inflated zone and door cutout
  const effectiveLeft = WALL_CENTER - WALL_HALF - AGENT_RADIUS;
  const effectiveRight = WALL_CENTER + WALL_HALF + AGENT_RADIUS;
  const startedLeft = pos.x < WALL_CENTER;
  if (startedLeft) {
    if (x > effectiveLeft && !inDoor(y)) x = effectiveLeft - 0.5;
  } else if (x < effectiveRight && !inDoor(y)) {
    x = effectiveRight + 0.5;
  }
  return { x, y };
}

export function stepEpisode(episode, action) {
  const agent = stepAgent(episode.agent, action);
  const next = { ...episode, agent, steps: episode.steps + 1 };
  next.done = distanceToTarget(next) < SUCCESS_DISTANCE;
  return next;
}

// ---------------------------------------------------------------------------
// expert policy (upstream ExpertPolicy: door waypoint then target; noise + repeat)
// ---------------------------------------------------------------------------

function gaussianPair(rng) {
  // Box-Muller from the uniform rng (deterministic given the rng stream)
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  const r = Math.sqrt(-2 * Math.log(u1));
  return [r * Math.cos(2 * Math.PI * u2), r * Math.sin(2 * Math.PI * u2)];
}

export function createExpertPolicy({ actionNoise = 2.0, actionRepeatProb = 0.05, doorReachTol = 10.5 } = {}) {
  let lastAction = null;
  return function expertAction(episode, rng) {
    const { agent, target } = episode;
    const otherRoom = agent.x < WALL_CENTER !== target.x < WALL_CENTER;
    let waypoint = target;
    if (otherRoom) {
      const door = { x: WALL_CENTER, y: DOOR_CENTER_Y };
      const doorDist = Math.hypot(door.x - agent.x, door.y - agent.y);
      waypoint = doorDist > doorReachTol ? door : target;
    }
    let dx = waypoint.x - agent.x;
    let dy = waypoint.y - agent.y;
    const norm = Math.hypot(dx, dy);
    if (norm > 1e-8) {
      dx /= norm;
      dy /= norm;
    } else {
      dx = 0;
      dy = 0;
    }
    let action = [dx, dy];
    if (actionNoise > 0) {
      const [n1, n2] = gaussianPair(rng);
      action = [dx + n1 * actionNoise, dy + n2 * actionNoise];
    }
    if (lastAction !== null && actionRepeatProb > 0 && rng() < actionRepeatProb) {
      action = lastAction;
    }
    action = [clamp(action[0], -1, 1), clamp(action[1], -1, 1)];
    lastAction = action;
    return action;
  };
}

// ---------------------------------------------------------------------------
// rendering (upstream torch renderer: masks + Gaussian alpha-blended dots)
// ---------------------------------------------------------------------------

function blendDot(rgb, cx, cy, radius, color) {
  // alpha = exp(-d^2 / (2 r^2)) normalized by its grid max (≈1 when the center is in-bounds)
  const cut = Math.ceil(radius * 5);
  const x0 = Math.max(0, Math.floor(cx - cut));
  const x1 = Math.min(IMG_SIZE - 1, Math.ceil(cx + cut));
  const y0 = Math.max(0, Math.floor(cy - cut));
  const y1 = Math.min(IMG_SIZE - 1, Math.ceil(cy + cut));
  let maxAlpha = 0;
  for (let y = y0; y <= y1; y += 1) {
    for (let x = x0; x <= x1; x += 1) {
      const d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      const a = Math.exp(-d2 / (2 * radius * radius));
      if (a > maxAlpha) maxAlpha = a;
    }
  }
  if (maxAlpha <= 0) return;
  for (let y = y0; y <= y1; y += 1) {
    for (let x = x0; x <= x1; x += 1) {
      const d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      const a = Math.exp(-d2 / (2 * radius * radius)) / maxAlpha;
      const idx = (y * IMG_SIZE + x) * 3;
      rgb[idx] = rgb[idx] * (1 - a) + color[0] * a;
      rgb[idx + 1] = rgb[idx + 1] * (1 - a) + color[1] * a;
      rgb[idx + 2] = rgb[idx + 2] * (1 - a) + color[2] * a;
    }
  }
}

function paintRect(rgb, x0, x1, y0, y1, color) {
  for (let y = y0; y < y1; y += 1) {
    for (let x = x0; x < x1; x += 1) {
      const idx = (y * IMG_SIZE + x) * 3;
      rgb[idx] = color[0];
      rgb[idx + 1] = color[1];
      rgb[idx + 2] = color[2];
    }
  }
}

// Renders the frame the encoder sees: agent at `agentPos`, optional target dot.
// Returns Float32Array (IMG_SIZE*IMG_SIZE*3) interleaved RGB in [0, 255].
export function renderFrameRGB(agentPos, { renderTarget = false, targetPos = null } = {}) {
  const rgb = new Float32Array(IMG_SIZE * IMG_SIZE * 3).fill(255);

  // central wall stripe (inclusive bounds upstream: grid >= c-half && grid <= c+half)
  const wallX0 = WALL_CENTER - WALL_HALF;
  const wallX1 = WALL_CENTER + WALL_HALF + 1;
  // door pixels stay background-colored (white door on white background by default), but the
  // door span is cut out of the wall stripe exactly as upstream
  const doorY0 = DOOR_CENTER_Y - DOOR_HALF;
  const doorY1 = DOOR_CENTER_Y + DOOR_HALF + 1;
  paintRect(rgb, wallX0, wallX1, 0, doorY0, COLORS.wall);
  paintRect(rgb, wallX0, wallX1, doorY0, doorY1, COLORS.door);
  paintRect(rgb, wallX0, wallX1, doorY1, IMG_SIZE, COLORS.wall);

  // border lines: mask[:, bs-t : bs] etc.
  paintRect(rgb, BORDER_SIZE - BORDER_LINE, BORDER_SIZE, 0, IMG_SIZE, COLORS.border);
  paintRect(rgb, IMG_SIZE - BORDER_SIZE, IMG_SIZE - BORDER_SIZE + BORDER_LINE, 0, IMG_SIZE, COLORS.border);
  paintRect(rgb, 0, IMG_SIZE, BORDER_SIZE - BORDER_LINE, BORDER_SIZE, COLORS.border);
  paintRect(rgb, 0, IMG_SIZE, IMG_SIZE - BORDER_SIZE, IMG_SIZE - BORDER_SIZE + BORDER_LINE, COLORS.border);

  if (renderTarget && targetPos) {
    blendDot(rgb, targetPos.x, targetPos.y, TARGET_RADIUS, COLORS.target);
  }
  blendDot(rgb, agentPos.x, agentPos.y, AGENT_RADIUS, COLORS.agent);
  return rgb;
}

// The model input: CHW Float32Array in [0, 1] (ImageNet normalization happens inside the
// exported encoder graph, so JS never duplicates the constants).
export function frameToModelInput(rgb) {
  const plane = IMG_SIZE * IMG_SIZE;
  const chw = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i += 1) {
    chw[i] = rgb[i * 3] / 255;
    chw[plane + i] = rgb[i * 3 + 1] / 255;
    chw[2 * plane + i] = rgb[i * 3 + 2] / 255;
  }
  return chw;
}

// The goal frame upstream is "the agent rendered at the target position".
export function renderGoalFrameRGB(targetPos) {
  return renderFrameRGB(targetPos, { renderTarget: false });
}

export function rgbToRGBA(rgb) {
  const plane = IMG_SIZE * IMG_SIZE;
  const rgba = new Uint8ClampedArray(plane * 4);
  for (let i = 0; i < plane; i += 1) {
    rgba[i * 4] = rgb[i * 3];
    rgba[i * 4 + 1] = rgb[i * 3 + 1];
    rgba[i * 4 + 2] = rgb[i * 3 + 2];
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

// ---------------------------------------------------------------------------
// action blocks (frameskip 5: the model consumes 5 consecutive env actions, flattened to 10)
// ---------------------------------------------------------------------------

export function packActionBlock(actions) {
  if (actions.length !== ACTION_BLOCK) {
    throw new Error(`action block needs ${ACTION_BLOCK} env actions, got ${actions.length}`);
  }
  const block = new Float32Array(ACTION_BLOCK * ACTION_DIM);
  for (let i = 0; i < ACTION_BLOCK; i += 1) {
    block[i * 2] = actions[i][0];
    block[i * 2 + 1] = actions[i][1];
  }
  return block;
}

// Applies one model-step action block (5 env steps with the same or distinct sub-actions).
export function stepEpisodeBlock(episode, actionBlock) {
  let current = episode;
  for (let i = 0; i < ACTION_BLOCK; i += 1) {
    current = stepEpisode(current, [actionBlock[i * 2], actionBlock[i * 2 + 1]]);
  }
  return current;
}

// Deterministic fingerprint used by selftests to pin rendering regressions.
export function frameFingerprint(rgb) {
  let h = 0x811c9dc5;
  for (let i = 0; i < rgb.length; i += 7) {
    h = Math.imul(h ^ Math.round(rgb[i]), 0x01000193) >>> 0;
  }
  return h >>> 0;
}
