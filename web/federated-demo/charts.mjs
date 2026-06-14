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

export function lineChart({ series, title, width = 520, height = 170, yZero = false }) {
  const wrap = document.createElement("div");
  wrap.className = "chart";
  if (title) {
    const heading = document.createElement("div");
    heading.className = "chart-title";
    heading.textContent = title;
    wrap.append(heading);
  }
  const extent = seriesExtent(series);
  if (!extent) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No data yet.";
    wrap.append(empty);
    return wrap;
  }
  const yMin = yZero ? Math.min(0, extent.yMin) : extent.yMin;
  const yMax = extent.yMax;
  const pad = { top: 10, right: 12, bottom: 22, left: 44 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const sx = (x) => pad.left + ((x - extent.xMin) / (extent.xMax - extent.xMin)) * innerW;
  const sy = (y) => pad.top + (1 - (y - yMin) / (yMax - yMin || 1)) * innerH;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "chart-svg",
    role: "img",
    "aria-label": title ?? "chart",
  });

  for (const tick of niceTicks(yMin, yMax)) {
    const y = sy(tick);
    svg.append(
      svgEl("line", { x1: pad.left, x2: width - pad.right, y1: y, y2: y, class: "chart-grid" }),
    );
    const label = svgEl("text", { x: pad.left - 6, y: y + 3.5, class: "chart-tick", "text-anchor": "end" });
    label.textContent = formatTick(tick);
    svg.append(label);
  }
  const xTicks = niceTicks(extent.xMin, extent.xMax, 6).filter(Number.isInteger);
  for (const tick of xTicks) {
    const label = svgEl("text", {
      x: sx(tick),
      y: height - 6,
      class: "chart-tick",
      "text-anchor": "middle",
    });
    label.textContent = String(tick);
    svg.append(label);
  }

  const colors = series.map((s, index) => s.color ?? CHART_PALETTE[index % CHART_PALETTE.length]);

  series.forEach((s, index) => {
    const color = colors[index];
    const path = s.points.map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ");
    svg.append(
      svgEl("path", {
        d: path,
        fill: "none",
        stroke: color,
        "stroke-width": 2,
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
        ...(s.dashed ? { "stroke-dasharray": "5 4" } : {}),
      }),
    );
    for (const p of s.points) {
      svg.append(svgEl("circle", { cx: sx(p.x), cy: sy(p.y), r: 2.6, fill: color }));
    }
  });
  wrap.append(svg);

  const legend = document.createElement("div");
  legend.className = "chart-legend";
  series.forEach((s, index) => {
    const item = document.createElement("span");
    item.className = "chart-legend-item";
    const swatch = document.createElement("span");
    swatch.className = "chart-swatch";
    swatch.style.background = colors[index];
    item.append(swatch, document.createTextNode(s.label));
    legend.append(item);
  });
  wrap.append(legend);
  return wrap;
}
