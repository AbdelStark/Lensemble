// A registered-ink chart recorder — the surprise signal as a scrolling pen
// trace on paper. Externally driven (push the live value each tick), so the
// same component records both surprise (ink-blue, oxblood spikes) and the
// frame-diff baseline (faded graphite, no spikes). No glow, no neon.

const INK = "#222f3c";       // --trace-ink
const ACCENT = "#96282b";    // --accent (oxblood)
const LEDGER = "rgba(37,28,22,0.07)";
const RULE = "rgba(37,28,22,0.20)";

export class Recorder {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.color = opts.color || INK;
    this.spikes = opts.spikes !== false; // surprise spikes oxblood; frame-diff doesn't
    this.max = opts.max || 0.3;          // value mapped to the top of the plate
    this.weight = opts.weight || 1.9;
    this.N = opts.samples || 260;
    this.vals = new Array(this.N).fill(0);
    this.hots = new Array(this.N).fill(0);
    this.baseline = opts.baseline ?? null;       // current (pre/post) reference level
    this.preBaseline = opts.preBaseline ?? null; // constant pre-federation level (for the drop band)
    this.sized = false;
    this.resize();
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const r = this.canvas.getBoundingClientRect();
    if (!r.width || !r.height) { this.w = 0; this.h = 0; return; }
    this.canvas.width = Math.round(r.width * dpr);
    this.canvas.height = Math.round(r.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = r.width; this.h = r.height; this.sized = true;
  }

  push(v, hot) {
    this.vals.push(v); this.vals.shift();
    this.hots.push(hot ? 1 : 0); this.hots.shift();
  }
  setBaseline(v) { this.baseline = v; }

  _y(v) {
    const pad = this.h * 0.1;
    const top = pad, bot = this.h - pad;
    const f = Math.max(0, Math.min(1, v / this.max));
    return bot - f * (bot - top);
  }
  _x(i) { return (i / (this.N - 1)) * this.w; }

  draw() {
    if (!this.w) return;
    const ctx = this.ctx, w = this.w, h = this.h, s = this.vals, hot = this.hots, self = this;
    ctx.clearRect(0, 0, w, h);

    // engraved reticule — vertical ticks (plotting paper)
    ctx.strokeStyle = LEDGER; ctx.lineWidth = 1; ctx.beginPath();
    for (let c = 1; c < 14; c++) { const gx = (c / 14) * w; ctx.moveTo(gx, h * 0.12); ctx.lineTo(gx, h * 0.88); }
    ctx.stroke();
    // zero rule
    ctx.strokeStyle = RULE; ctx.beginPath();
    ctx.moveTo(0, this._y(0)); ctx.lineTo(w, this._y(0)); ctx.stroke();
    // pre-federation reference (constant) + the current baseline. After federation the
    // current level drops below the pre line; the gap between them IS the certified
    // in-distribution improvement, shaded oxblood so the payoff is visible on toggle.
    const dropped = this.preBaseline != null && this.baseline != null && this.baseline < this.preBaseline - 1e-6;
    if (this.preBaseline != null) {
      const py = this._y(this.preBaseline);
      if (dropped) { ctx.fillStyle = "rgba(150,40,43,0.10)"; ctx.fillRect(0, py, w, this._y(this.baseline) - py); }
      ctx.save(); ctx.setLineDash([1, 4]); ctx.strokeStyle = "rgba(37,28,22,0.3)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(w, py); ctx.stroke(); ctx.restore();
    }
    if (this.baseline != null) {
      ctx.save(); ctx.setLineDash([4, 4]);
      ctx.strokeStyle = dropped ? "rgba(150,40,43,0.6)" : "rgba(37,28,22,0.26)";
      ctx.beginPath(); const by = this._y(this.baseline); ctx.moveTo(0, by); ctx.lineTo(w, by); ctx.stroke();
      ctx.restore();
    }

    function poly(from, to) {
      if (to - from < 2) return;
      ctx.beginPath();
      for (let i = from; i < to; i++) { const x = self._x(i), y = self._y(s[i]); i === from ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
      ctx.stroke();
    }

    ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.strokeStyle = this.color; ctx.lineWidth = this.weight;
    poly(0, this.N);

    // overdraw spiking runs in oxblood + an event rule at each run's peak
    if (this.spikes) {
      ctx.strokeStyle = ACCENT; ctx.lineWidth = this.weight + 0.4;
      let i = 0;
      while (i < this.N) {
        if (hot[i]) {
          const st = i; let peak = i, pv = s[i];
          while (i < this.N && hot[i]) { if (s[i] > pv) { pv = s[i]; peak = i; } i++; }
          poly(Math.max(0, st - 1), Math.min(this.N, i + 1));
          const ex = this._x(peak), ey = this._y(s[peak]);
          ctx.save(); ctx.globalAlpha = 0.45; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(ex, this._y(0)); ctx.stroke(); ctx.restore();
          ctx.lineWidth = this.weight + 0.4;
        } else i++;
      }
    }

    // recording cursor + pen nib at the live (right) edge — the trailing edge is the
    // one masked off-page in CSS, so the "now" pen stays crisp.
    ctx.strokeStyle = ACCENT; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(w - 0.5, this._y(0)); ctx.lineTo(w - 0.5, this._y(0) - h * 0.5); ctx.stroke();
    ctx.fillStyle = ACCENT; ctx.beginPath(); ctx.arc(w - 1.6, this._y(s[this.N - 1]), 2.2, 0, Math.PI * 2); ctx.fill();
  }
}
