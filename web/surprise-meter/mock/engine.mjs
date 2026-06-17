// The live surprise loop — composes the mock LeWM modules exactly as the real
// app will compose the real ones: predict next latent → encode the actual next
// latent → surprise = MSE/192. Only the latent generation is simulated.
//
// SWAP FOR REAL: change the three imports below to ../../federated-demo/* and
// feed real frames (renderFrameRGB → frameToModelInput → encodeFrames). The
// per-tick contract returned here stays identical, so app.mjs is unaffected.
import { mockRuntime, mockEnv, mockProbe } from "./lewm_mock.mjs";
import { CERTIFIED } from "./fixtures.mjs";

const PEAK = { teleport: 0.205, ood: 0.20, wall: 0.165 };

export class SurpriseEngine {
  constructor() { this.reset(); }

  reset() {
    this.t = 0; this.frame = 0;
    this.mode = "pre";        // "pre" | "post" — held-out in-distribution level (C11)
    this.running = true;
    this.event = { type: null, energy: 0 };
    this.override = null;     // teleport / wall position hold
    this.prev = mockEnv.path(0);
    this.frameDiff = 0;
  }

  setMode(m) { this.mode = m; }
  toggleRun() { this.running = !this.running; return this.running; }

  // pre = baselineMse, post = adaptedMse → their ratio IS the certified +12.3% (C11)
  baselineLevel() { return this.mode === "post" ? CERTIFIED.adaptedMse : CERTIFIED.baselineMse; }

  perturb(type) {
    this.event = { type, energy: PEAK[type] ?? 0.18 };
    if (type === "teleport") this.override = { pos: mockEnv.randomPos(this.frame * 2654435761), until: this.t + 0.55 };
    if (type === "wall") {
      const a = this._agentRaw();
      this.override = { pos: { x: a.x < 0.5 ? 0.055 : 0.945, y: a.y }, until: this.t + 0.5 };
    }
    // OOD action: surprise only, no position change — the surprise≠motion contrast
    return type;
  }

  _agentRaw() {
    return this.override && this.t < this.override.until ? this.override.pos : mockEnv.path(this.t);
  }

  tick(dt) {
    if (this.running) this.t += dt;
    this.frame++;
    this.event.energy *= Math.pow(0.06, dt); // ink-recorder decay
    if (this.override && this.t >= this.override.until) this.override = null;

    const agent = this._agentRaw();
    // frame-diff = motion: a busy, always-on band (pixels are constantly changing as the
    // agent moves) that runs LOUD while surprise stays flat. A teleport / wall jump kicks
    // it higher; it never collapses to zero (that would read as a glitch, not "calm").
    const dx = agent.x - this.prev.x, dy = agent.y - this.prev.y;
    const inst = Math.sqrt(dx * dx + dy * dy) / Math.max(dt, 1 / 120);
    const kick = Math.min(0.42, inst * 2.4); // teleport / wall shows as a motion burst
    const busy = 0.17 * Math.sin(this.t * 6.1) + 0.10 * Math.sin(this.t * 15.3 + 2) + 0.07 * Math.sin(this.t * 28.7 + 5);
    const fdTarget = Math.max(0.3, Math.min(1, 0.52 + busy + kick));
    this.frameDiff += (fdTarget - this.frameDiff) * Math.min(1, dt * 26);
    this.prev = agent;

    const wander = (Math.sin(this.t * 1.7) * 0.5 + Math.sin(this.t * 4.1 + 1) * 0.3) * 0.006;
    const target = Math.max(0.003, this.baselineLevel() + wander + this.event.energy);

    // real pipeline shape — unchanged on swap
    const pred = mockRuntime.predictLatents(this.frame);
    const actual = mockEnv.actualNextLatent(pred, target, this.frame * 9973);
    const surprise = mockProbe.surprise(pred, actual);

    return {
      t: this.t,
      agent,
      room: agent.x < 0.5 ? "left" : "right",
      surprise,
      frameDiff: this.frameDiff,
      baseline: this.baselineLevel(),
      mode: this.mode,
      hot: this.event.energy > 0.045,
      event: this.event.energy > 0.045 ? this.event.type : null,
      running: this.running,
    };
  }
}
