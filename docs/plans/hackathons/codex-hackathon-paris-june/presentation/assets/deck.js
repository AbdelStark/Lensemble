/* ============================================================================
   Less Surprised — "The Offprint"
   The surprise signal as a registered ink seismograph: a sweep-recorder that
   plots a single ink-blue hairline, overdraws the spiking segment in oxblood,
   drops an event rule, and leaves a faint ghost of the prior pass. No glow,
   no neon, no leading-dot orb. Plus the always-alive header heartbeat and the
   reveal.js config (flush-left, uncentered). Respects prefers-reduced-motion.
   ========================================================================== */
(function () {
  "use strict";

  var reduceMotion =
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // sRGB hex of the OKLCH print tokens — set explicitly so the canvas renders
  // identically regardless of a browser's oklch()-in-canvas support.
  var INK = "#222f3c";       // --trace-ink: calm recorded signal
  var ACCENT = "#96282b";    // --accent: oxblood
  var RESOLVED = "#3f6264";  // --resolved: "after federation", quiet & cool
  var LEDGER = "rgba(37,28,22,0.07)"; // --ink @ 7% — graph reticule
  var RULE = "rgba(37,28,22,0.22)";   // --ink @ 22% — baseline
  function loadColors() {}

  // Organic-but-deterministic baseline wander (no Math.random storms).
  function wander(t, seed) {
    return (
      Math.sin(t * 1.3 + seed) * 0.5 +
      Math.sin(t * 3.1 + seed * 2.3) * 0.3 +
      Math.sin(t * 7.7 + seed * 0.7) * 0.2
    );
  }

  // --- one sweep-recorder on a <canvas data-meter> -------------------------
  function Seismo(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.state = canvas.dataset.state || "live"; // live | busy | calm | resolved | heartbeat
    this.heartbeat = this.state === "heartbeat";

    // Per-state character.
    var lv = {
      live: 0.17, busy: 0.24, calm: 0.07, resolved: 0.05, heartbeat: 0.05,
    }[this.state];
    this.level = parseFloat(canvas.dataset.level || lv);
    // the header heartbeat is a calm idling baseline (no decorative spikes) —
    // the real signal only spikes on the captioned plates, as data.
    this.spikes = this.state === "live" || this.state === "busy";
    this.spikeEvery = this.state === "busy" ? 2.6 : this.heartbeat ? 5.5 : 3.8;
    this.color = this.state === "resolved" ? RESOLVED : INK;

    this.seed = Math.random() * 100;
    this.t = 0;
    this.spikeTimer = this.heartbeat ? 2.0 : 1.2;
    this.spikeEnergy = 0;

    this.N = this.heartbeat ? 300 : 240;
    this.samples = new Array(this.N).fill(0);
    this.hots = new Array(this.N).fill(0);
    this.warmed = false;
    this.sized = false;
    this.resize();
  }

  // pre-roll the recorder so the paper is already full of trace on first paint
  Seismo.prototype.warm = function () {
    for (var i = 0; i < this.N; i++) this.step(0.016);
    this.warmed = true;
  };

  Seismo.prototype.resize = function () {
    var dpr = window.devicePixelRatio || 1;
    var rect = this.canvas.getBoundingClientRect();
    // measure the rendered box (plate gives it a definite CSS height); if the
    // slide isn't laid out yet (display:none), bail so we re-measure when shown.
    if (!rect.width || !rect.height) { this.w = 0; this.h = 0; return; }
    this.canvas.width = Math.round(rect.width * dpr);
    this.canvas.height = Math.round(rect.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = rect.width;
    this.h = rect.height;
  };

  Seismo.prototype.step = function (dt) {
    this.t += dt;
    if (this.spikes) {
      this.spikeTimer -= dt;
      if (this.spikeTimer <= 0) {
        this.spikeEnergy = this.heartbeat ? 0.55 : this.state === "busy" ? 1.1 : 0.95;
        this.spikeTimer = this.spikeEvery * (0.8 + 0.4 * Math.abs(Math.sin(this.t)));
      }
    }
    this.spikeEnergy *= Math.pow(0.10, dt); // ink-recorder decay — the event lingers a beat
    var base = wander(this.t, this.seed) * this.level;
    var spiking = this.spikeEnergy > 0.14;
    var transient = this.spikeEnergy * Math.sin(this.t * 30) * 0.95;
    this.samples.push(base + transient); this.samples.shift();
    this.hots.push(spiking ? 1 : 0); this.hots.shift();
  };

  Seismo.prototype._x = function (i) { return (i / (this.N - 1)) * this.w; };

  Seismo.prototype.draw = function () {
    var ctx = this.ctx, w = this.w, h = this.h, mid = h * 0.5;
    var amp = h * (this.heartbeat ? 0.30 : 0.34);
    var s = this.samples, hot = this.hots, self = this;
    ctx.clearRect(0, 0, w, h);

    if (!this.heartbeat) {
      // engraved graph reticule + baseline — reads as plotting paper, not UI
      ctx.strokeStyle = LEDGER; ctx.lineWidth = 1; ctx.beginPath();
      var cols = 12;
      for (var c = 1; c < cols; c++) { var gx = (c / cols) * w; ctx.moveTo(gx, h * 0.12); ctx.lineTo(gx, h * 0.88); }
      ctx.stroke();
      ctx.strokeStyle = RULE; ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke();
    }

    function poly(from, to) {
      if (to - from < 2) return;
      ctx.beginPath();
      for (var i = from; i < to; i++) {
        var x = self._x(i), y = mid - s[i] * amp;
        if (i === from) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    ctx.lineJoin = "round"; ctx.lineCap = "round";

    // the full recorded signal — one continuous ink hairline
    ctx.strokeStyle = this.color;
    ctx.lineWidth = this.heartbeat ? 1.3 : 2.0;
    poly(0, this.N);

    // overdraw each spiking run in oxblood + drop an event rule at its peak
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = this.heartbeat ? 1.5 : 2.4;
    var i = 0;
    while (i < this.N) {
      if (hot[i]) {
        var st = i, peakI = i, peakV = Math.abs(s[i]);
        while (i < this.N && hot[i]) { if (Math.abs(s[i]) > peakV) { peakV = Math.abs(s[i]); peakI = i; } i++; }
        poly(Math.max(0, st - 1), Math.min(this.N, i + 1));
        if (!this.heartbeat) {
          var ex = this._x(peakI), ey = mid - s[peakI] * amp;
          ctx.save(); ctx.globalAlpha = 0.45; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(ex, mid); ctx.stroke(); ctx.restore();
          ctx.lineWidth = 2.4;
        }
      } else i++;
    }

    // recording cursor: a thin oxblood tick where fresh ink lands (right edge)
    ctx.strokeStyle = ACCENT; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(w - 0.5, mid - amp * 0.55); ctx.lineTo(w - 0.5, mid + amp * 0.55); ctx.stroke();
  };

  // --- driver --------------------------------------------------------------
  function startMeters() {
    loadColors();
    var canvases = [].slice.call(document.querySelectorAll("canvas[data-meter]"));
    if (!canvases.length) return;
    var meters = canvases.map(function (c) { return new Seismo(c); });

    function remeasure() { meters.forEach(function (m) { m.sized = false; }); }
    window.addEventListener("resize", remeasure);
    if (window.Reveal) { window.Reveal.on("slidechanged", remeasure); }

    function ensure(m) {
      var visible = m.heartbeat || m.canvas.offsetParent !== null;
      if (!visible) return false;
      if (!m.sized) { m.resize(); if (m.w) { m.sized = true; if (!m.warmed) m.warm(); } }
      return !!m.w;
    }

    if (reduceMotion) {
      // settle on next frame (so plates have laid out), then draw one static pass
      requestAnimationFrame(function () {
        meters.forEach(function (m) {
          if (!ensure(m)) return;
          for (var i = 0; i < m.N * 1.2; i++) m.step(0.016);
          m.draw();
        });
      });
      return;
    }

    var last = null;
    function frame(ts) {
      if (last === null) last = ts;
      var dt = Math.min(0.05, (ts - last) / 1000);
      last = ts;
      for (var i = 0; i < meters.length; i++) {
        var m = meters[i];
        if (ensure(m)) { m.step(dt); m.draw(); }
      }
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  // --- reveal init + journal folio ----------------------------------------
  function updateFolio() {
    var el = document.getElementById("folio");
    if (!el || !window.Reveal) return;
    var idx = window.Reveal.getIndices();
    var total = document.querySelectorAll(".reveal .slides > section").length;
    var n = (idx.h + 1) + (idx.v ? "." + idx.v : "");
    el.textContent = "Fig. " + n + " / " + total;
  }

  window.__initDeck__ = function () {
    if (window.Reveal) {
      window.Reveal.initialize({
        hash: true,
        controls: false,    // generic chevrons are off-voice; folio is the wayfinding
        progress: false,    // no progress hairline competing with the journal rule
        center: false,      // we vertically centre via CSS (full-height flex) so the
        margin: 0,          // deck stays flush-LEFT and full-width without reveal's offset
        width: 1280,
        height: 720,
        transition: "fade",
        transitionSpeed: "fast",
        plugins: [window.RevealNotes, window.RevealHighlight].filter(Boolean),
      });
      window.Reveal.on("ready", updateFolio);
      window.Reveal.on("slidechanged", updateFolio);
      // re-layout once the variable fonts land, so reveal's fit-scale is computed
      // on the final metrics (avoids a stale down-scale when deep-linking a slide).
      if (document.fonts && document.fonts.ready) {
        document.fonts.ready.then(function () { window.Reveal.layout(); });
      }
    }
    startMeters();
  };

  document.addEventListener("DOMContentLoaded", function () {
    setTimeout(function () { if (!window.Reveal) startMeters(); }, 1500);
  });
})();
