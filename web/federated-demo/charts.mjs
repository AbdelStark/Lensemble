// Lightweight SVG charts for the host dashboard — no dependencies.
//
// Data preparation is pure and node-testable; only lineChart() touches the DOM. Every series is
// real run data (participant update metadata and round metrics) — nothing is smoothed or
// fabricated, gaps stay gaps.

export const CHART_PALETTE = ["#4f46e5", "#0d9488", "#d97706", "#e11d48", "#0284c7", "#9333ea", "#65a30d", "#c026d3"];

// ---------------------------------------------------------------------------
// pure data preparation
// ---------------------------------------------------------------------------

function metadataLoss(metadata) {
  // real mode carries metrics.predLossLast; the surrogate path carries a flat loss
  const value = metadata?.metrics?.predLossLast ?? metadata?.loss;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function participantLossSeries(run) {
  const series = [];
  for (const participant of run.participants ?? []) {
    const points = [];
    for (const [round, metadata] of Object.entries(participant.updateMetadata ?? {})) {
      const y = metadataLoss(metadata);
      const x = Number(round);
      if (y !== null && Number.isFinite(x)) points.push({ x, y });
    }
    points.sort((a, b) => a.x - b.x);
    if (points.length > 0) {
      series.push({ label: participant.displayName || participant.id, points });
    }
  }
  return series;
}

export function roundSeries(run, specs) {
  // specs: [{ key, label, dashed? }] over run.roundMetrics
  const metrics = run.roundMetrics ?? [];
  const series = [];
  for (const spec of specs) {
    const points = [];
    for (const metric of metrics) {
      const y = metric[spec.key];
      if (typeof y === "number" && Number.isFinite(y)) points.push({ x: metric.round, y });
    }
    if (points.length > 0) {
      series.push({ label: spec.label ?? spec.key, points, dashed: Boolean(spec.dashed) });
    }
  }
  return series;
}

export function seriesExtent(series) {
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  for (const s of series) {
    for (const p of s.points) {
      if (p.x < xMin) xMin = p.x;
      if (p.x > xMax) xMax = p.x;
      if (p.y < yMin) yMin = p.y;
      if (p.y > yMax) yMax = p.y;
    }
  }
  if (!Number.isFinite(xMin)) return null;
  if (xMin === xMax) {
    xMin -= 0.5;
    xMax += 0.5;
  }
  if (yMin === yMax) {
    const pad = Math.abs(yMin) * 0.2 || 0.5;
    yMin -= pad;
    yMax += pad;
  }
  return { xMin, xMax, yMin, yMax };
}

export function niceTicks(min, max, count = 4) {
  const span = max - min;
  if (!(span > 0)) return [min];
  const step = 10 ** Math.floor(Math.log10(span / count));
  const candidates = [step, step * 2, step * 2.5, step * 5, step * 10];
  const chosen = candidates.find((c) => span / c <= count) ?? step * 10;
  const ticks = [];
  for (let v = Math.ceil(min / chosen) * chosen; v <= max + 1e-12; v += chosen) {
    ticks.push(Number(v.toPrecision(10)));
  }
  return ticks;
}

export function formatTick(value) {
  const abs = Math.abs(value);
  if (abs === 0) return "0";
  if (abs >= 100) return String(Math.round(value));
  if (abs >= 1) return String(Number(value.toFixed(1)));
  if (abs >= 0.01) return String(Number(value.toFixed(3)));
  return value.toExponential(0);
}

// ---------------------------------------------------------------------------
// SVG rendering
// ---------------------------------------------------------------------------

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

// Shared geometry so the initial render and every in-place update compute the
// exact same scales (no one-frame jump when a chart updates).
function buildScales({ series, width = 520, height = 170, yZero = false }) {
  const extent = seriesExtent(series);
  if (!extent) return { extent: null };
  const yMin = yZero ? Math.min(0, extent.yMin) : extent.yMin;
  const yMax = extent.yMax;
  const pad = { top: 10, right: 12, bottom: 22, left: 44 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const sx = (x) => pad.left + ((x - extent.xMin) / (extent.xMax - extent.xMin)) * innerW;
  const sy = (y) => pad.top + (1 - (y - yMin) / (yMax - yMin || 1)) * innerH;
  return { extent, yMin, yMax, pad, innerW, innerH, sx, sy, width, height };
}

function seriesColor(s, index) {
  return s.color ?? CHART_PALETTE[index % CHART_PALETTE.length];
}

function pathD(points, sc) {
  return points
    .map((p, i) => `${i === 0 ? "M" : "L"}${sc.sx(p.x).toFixed(1)},${sc.sy(p.y).toFixed(1)}`)
    .join(" ");
}

// Renders grid + axis ticks into their layer. Cheap (a handful of nodes); cleared
// and rebuilt each update so tick counts can change without leftovers.
function renderAxes(layer, sc) {
  while (layer.firstChild) layer.removeChild(layer.firstChild);
  for (const tick of niceTicks(sc.yMin, sc.yMax)) {
    const y = sc.sy(tick);
    layer.append(
      svgEl("line", { x1: sc.pad.left, x2: sc.width - sc.pad.right, y1: y, y2: y, class: "chart-grid" }),
    );
    const label = svgEl("text", { x: sc.pad.left - 6, y: y + 3.5, class: "chart-tick", "text-anchor": "end" });
    label.textContent = formatTick(tick);
    layer.append(label);
  }
  for (const tick of niceTicks(sc.extent.xMin, sc.extent.xMax, 6).filter(Number.isInteger)) {
    const label = svgEl("text", { x: sc.sx(tick), y: sc.height - 6, class: "chart-tick", "text-anchor": "middle" });
    label.textContent = String(tick);
    layer.append(label);
  }
}

// Reconciles one series' <path> + point <circle>s inside its <g>, reusing
// existing nodes so the line grows in place (no teardown / flash).
function renderSeriesGroup(group, s, color, sc) {
  let path = group.querySelector("path");
  if (!path) {
    path = svgEl("path", { fill: "none", "stroke-width": 2, "stroke-linecap": "round", "stroke-linejoin": "round" });
    group.append(path);
  }
  path.setAttribute("d", pathD(s.points, sc));
  path.setAttribute("stroke", color);
  if (s.dashed) path.setAttribute("stroke-dasharray", "5 4");
  else path.removeAttribute("stroke-dasharray");

  const circles = group.querySelectorAll("circle");
  s.points.forEach((p, i) => {
    let dot = circles[i];
    if (!dot) {
      dot = svgEl("circle", { r: 2.6 });
      group.append(dot);
    }
    dot.setAttribute("cx", sc.sx(p.x));
    dot.setAttribute("cy", sc.sy(p.y));
    dot.setAttribute("fill", color);
  });
  for (let i = circles.length - 1; i >= s.points.length; i -= 1) circles[i].remove();
}

function renderLegend(legend, series) {
  const items = legend.querySelectorAll(".chart-legend-item");
  series.forEach((s, index) => {
    let item = items[index];
    if (!item) {
      item = document.createElement("span");
      item.className = "chart-legend-item";
      const swatch = document.createElement("span");
      swatch.className = "chart-swatch";
      item.append(swatch, document.createTextNode(s.label));
      legend.append(item);
    }
    item.querySelector(".chart-swatch").style.background = seriesColor(s, index);
    item.lastChild.nodeValue = s.label;
  });
  const stale = legend.querySelectorAll(".chart-legend-item");
  for (let i = stale.length - 1; i >= series.length; i -= 1) stale[i].remove();
}

// Patches a previously-built chart node to match new series data in place. The
// generic morph() reconciler routes here via node.__chartUpdate, so the <svg> is
// never torn down — only changed geometry repaints.
function updateChart(wrap, opts) {
  const sc = buildScales(opts);
  const svg = wrap.querySelector("svg");
  const empty = wrap.querySelector(".chart-empty");
  if (!sc.extent) {
    if (svg) svg.remove();
    if (wrap.querySelector(".chart-legend")) wrap.querySelector(".chart-legend").remove();
    if (!empty) {
      const note = document.createElement("p");
      note.className = "muted chart-empty";
      note.textContent = "No data yet.";
      wrap.append(note);
    }
    wrap.__series = opts.series;
    return;
  }
  if (!svg) {
    // empty → has-data transition: build the body fresh once.
    if (empty) empty.remove();
    buildChartBody(wrap, opts, sc);
    wrap.__series = opts.series;
    return;
  }
  renderAxes(svg.querySelector(".chart-axes"), sc);
  const layer = svg.querySelector(".chart-series");
  const groups = layer.querySelectorAll("g[data-series]");
  opts.series.forEach((s, index) => {
    let group = groups[index];
    if (!group) {
      group = svgEl("g", { "data-series": String(index) });
      layer.append(group);
    }
    renderSeriesGroup(group, s, seriesColor(s, index), sc);
  });
  for (let i = groups.length - 1; i >= opts.series.length; i -= 1) groups[i].remove();
  renderLegend(wrap.querySelector(".chart-legend"), opts.series);
  wrap.__series = opts.series;
}

function buildChartBody(wrap, opts, sc) {
  const svg = svgEl("svg", {
    viewBox: `0 0 ${sc.width} ${sc.height}`,
    class: "chart-svg",
    role: "img",
    "aria-label": opts.title ?? "chart",
  });
  const axes = svgEl("g", { class: "chart-axes" });
  const seriesLayer = svgEl("g", { class: "chart-series" });
  svg.append(axes, seriesLayer);
  renderAxes(axes, sc);
  opts.series.forEach((s, index) => {
    const group = svgEl("g", { "data-series": String(index) });
    seriesLayer.append(group);
    renderSeriesGroup(group, s, seriesColor(s, index), sc);
  });
  wrap.append(svg);

  const legend = document.createElement("div");
  legend.className = "chart-legend";
  wrap.append(legend);
  renderLegend(legend, opts.series);
}

export function lineChart(opts) {
  const { series, title } = opts;
  const wrap = document.createElement("div");
  wrap.className = "chart";
  if (title) {
    wrap.dataset.key = title; // stable identity for the keyed reconciler
    const heading = document.createElement("div");
    heading.className = "chart-title";
    heading.textContent = title;
    wrap.append(heading);
  }
  const sc = buildScales(opts);
  if (!sc.extent) {
    const empty = document.createElement("p");
    empty.className = "muted chart-empty";
    empty.textContent = "No data yet.";
    wrap.append(empty);
  } else {
    buildChartBody(wrap, opts, sc);
  }
  // morph() routes here via __chartUpdate instead of descending into the <svg>.
  wrap.__series = series;
  wrap.__chartUpdate = (nextSeries) => updateChart(wrap, { ...opts, series: nextSeries });
  return wrap;
}
