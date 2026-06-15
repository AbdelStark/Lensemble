// Node-runnable selftest for the keyed DOM reconciler (morph.mjs).
// Run: node web/federated-demo/morph_selftest.mjs
//
// morph.mjs is browser-only, so this provides a small faithful DOM shim
// (childNodes-as-source-of-truth, data-* dataset view, attribute mirroring)
// that is enough to exercise every branch: text/attr patching, keyed reorder,
// positional reuse, data-static / data-preserve opt-outs, the __chartUpdate
// fast-path, focus-guarded value preservation, and disabled mirroring.

import { keyOf, morph } from "./morph.mjs";

// ---------------------------------------------------------------- DOM shim
globalThis.Node = { TEXT_NODE: 3, COMMENT_NODE: 8, ELEMENT_NODE: 1 };
globalThis.document = { activeElement: null };

class ShimNode {
  constructor() {
    this.parentNode = null;
    this.childNodes = [];
  }
  get firstChild() {
    return this.childNodes[0] ?? null;
  }
  get nextSibling() {
    if (!this.parentNode) return null;
    const i = this.parentNode.childNodes.indexOf(this);
    return this.parentNode.childNodes[i + 1] ?? null;
  }
  remove() {
    if (!this.parentNode) return;
    const i = this.parentNode.childNodes.indexOf(this);
    if (i >= 0) this.parentNode.childNodes.splice(i, 1);
    this.parentNode = null;
  }
  replaceWith(node) {
    if (!this.parentNode) return;
    const p = this.parentNode;
    const i = p.childNodes.indexOf(this);
    if (node.parentNode) node.remove();
    p.childNodes[i] = node;
    node.parentNode = p;
    this.parentNode = null;
  }
  insertBefore(node, ref) {
    if (node.parentNode) node.remove();
    node.parentNode = this;
    if (ref == null) {
      this.childNodes.push(node);
    } else {
      const i = this.childNodes.indexOf(ref);
      this.childNodes.splice(i < 0 ? this.childNodes.length : i, 0, node);
    }
    return node;
  }
  appendChild(node) {
    return this.insertBefore(node, null);
  }
}

class ShimText extends ShimNode {
  constructor(value) {
    super();
    this.nodeType = 3;
    this.nodeName = "#text";
    this.nodeValue = value;
  }
}

class ShimElement extends ShimNode {
  constructor(tag) {
    super();
    this.nodeType = 1;
    this.nodeName = tag.toUpperCase();
    this._attrs = new Map();
  }
  get attributes() {
    return [...this._attrs].map(([name, value]) => ({ name, value }));
  }
  hasAttribute(name) {
    return this._attrs.has(name);
  }
  getAttribute(name) {
    return this._attrs.has(name) ? this._attrs.get(name) : null;
  }
  setAttribute(name, value) {
    this._attrs.set(name, String(value));
  }
  removeAttribute(name) {
    this._attrs.delete(name);
  }
  get dataset() {
    const ds = {};
    for (const [k, v] of this._attrs) {
      if (k.startsWith("data-")) {
        const camel = k.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
        ds[camel] = v;
      }
    }
    return ds;
  }
  get textContent() {
    return this.childNodes.map((c) => (c.nodeType === 3 ? c.nodeValue : c.textContent)).join("");
  }
}

// el-like builder. attrs: {text, value, checked, disabled, "data-*", other attrs}
function h(tag, attrs = {}, children = []) {
  const node = new ShimElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") node.appendChild(new ShimText(String(v)));
    else if (k === "value" || k === "checked" || k === "disabled") node[k] = v;
    else node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    node.appendChild(typeof c === "string" ? new ShimText(c) : c);
  }
  return node;
}

// ---------------------------------------------------------------- harness
const failures = [];
let total = 0;
function check(name, fn) {
  total += 1;
  try {
    fn();
  } catch (error) {
    failures.push({ name, error: String(error?.message ?? error) });
  }
}
function assert(cond, message) {
  if (!cond) throw new Error(message);
}

// ---------------------------------------------------------------- tests
check("text node value is patched in place", () => {
  const live = h("h2", {}, ["Round 1 of 10"]);
  const next = h("h2", {}, ["Round 2 of 10"]);
  const liveText = live.firstChild;
  morph(live, next);
  assert(live.firstChild === liveText, "text node identity preserved");
  assert(live.firstChild.nodeValue === "Round 2 of 10", "text patched");
});

check("attributes are added, changed, and removed", () => {
  const live = h("span", { class: "badge state-running", title: "x" }, ["running"]);
  const next = h("span", { class: "badge state-completed", role: "status" }, ["completed"]);
  morph(live, next);
  assert(live.getAttribute("class") === "badge state-completed", "class changed");
  assert(live.getAttribute("role") === "status", "role added");
  assert(!live.hasAttribute("title"), "title removed");
  assert(live.firstChild.nodeValue === "completed", "text patched");
});

check("keyed children keep identity across reorder", () => {
  const a = h("li", { "data-key": "a" }, ["A"]);
  const b = h("li", { "data-key": "b" }, ["B"]);
  const c = h("li", { "data-key": "c" }, ["C"]);
  const live = h("ul", {}, [a, b, c]);
  // next reorders to c, a, b
  const next = h("ul", {}, [
    h("li", { "data-key": "c" }, ["C"]),
    h("li", { "data-key": "a" }, ["A2"]),
    h("li", { "data-key": "b" }, ["B"]),
  ]);
  morph(live, next);
  assert(live.childNodes[0] === c, "c moved to front, identity kept");
  assert(live.childNodes[1] === a, "a kept");
  assert(live.childNodes[2] === b, "b kept");
  assert(a.firstChild.nodeValue === "A2", "a's text patched");
});

check("stale keyed children are removed and new ones inserted", () => {
  const a = h("li", { "data-key": "a" }, ["A"]);
  const b = h("li", { "data-key": "b" }, ["B"]);
  const live = h("ul", {}, [a, b]);
  const next = h("ul", {}, [
    h("li", { "data-key": "a" }, ["A"]),
    h("li", { "data-key": "z" }, ["Z"]),
  ]);
  morph(live, next);
  assert(live.childNodes.length === 2, "two children");
  assert(live.childNodes[0] === a, "a kept");
  assert(keyOf(live.childNodes[1]) === "z", "z inserted");
  assert(b.parentNode === null, "b removed");
});

check("unkeyed children match positionally by nodeName", () => {
  const p = h("p", {}, ["one"]);
  const live = h("div", {}, [p]);
  const next = h("div", {}, [h("p", {}, ["two"])]);
  morph(live, next);
  assert(live.firstChild === p, "p reused positionally");
  assert(p.firstChild.nodeValue === "two", "patched");
});

check("data-static subtree is never descended into or attribute-patched", () => {
  const canvas = h("canvas", { "data-key": "qr", "data-static": "", width: "120", "data-qr-url": "u1" });
  canvas.appendChild(new ShimText("drawn")); // pretend pixels
  const live = h("div", {}, [canvas]);
  const next = h("div", {}, [h("canvas", { "data-key": "qr", "data-static": "", "data-qr-url": "u1" })]);
  morph(live, next);
  assert(live.firstChild === canvas, "canvas identity kept");
  assert(canvas.getAttribute("width") === "120", "width NOT stripped by patchAttrs");
  assert(canvas.firstChild.nodeValue === "drawn", "children untouched");
});

check("data-preserve subtree is left alone while it exists", () => {
  const input = h("input", { value: "half-typed" });
  const form = h("div", { "data-key": "join-form", "data-preserve": "" }, [input]);
  const live = h("section", {}, [form]);
  const next = h("section", {}, [h("div", { "data-key": "join-form", "data-preserve": "" }, [h("input", { value: "" })])]);
  morph(live, next);
  assert(live.firstChild === form, "form kept");
  assert(form.firstChild === input, "input not replaced");
  assert(input.value === "half-typed", "typed value preserved");
});

check("__chartUpdate fast-path is invoked instead of descending", () => {
  let got = null;
  const chart = h("div", { "data-key": "Latent geometry" });
  chart.appendChild(h("svg", {}, ["OLD"]));
  chart.__chartUpdate = (series) => { got = series; };
  const live = h("div", {}, [chart]);
  const nextChart = h("div", { "data-key": "Latent geometry" });
  nextChart.appendChild(h("svg", {}, ["NEW"]));
  nextChart.__series = [{ label: "x", points: [{ x: 1, y: 2 }] }];
  const next = h("div", {}, [nextChart]);
  morph(live, next);
  assert(live.firstChild === chart, "chart node identity kept");
  assert(got !== null && got[0].label === "x", "__chartUpdate received next.__series");
  assert(chart.firstChild.firstChild.nodeValue === "OLD", "svg NOT descended/replaced");
});

check("disabled is mirrored from the freshly-built node", () => {
  const live = h("button", { disabled: false }, ["Run"]);
  const next = h("button", { disabled: true }, ["Run"]);
  morph(live, next);
  assert(live.disabled === true, "disabled mirrored true (in-flight preserved)");
  const next2 = h("button", { disabled: false }, ["Run"]);
  morph(live, next2);
  assert(live.disabled === false, "disabled mirrored false again");
});

check("focused field value is not clobbered, unfocused is", () => {
  const live = h("input", { value: "user typing" });
  document.activeElement = live;
  morph(live, h("input", { value: "server value" }));
  assert(live.value === "user typing", "focused value preserved");
  document.activeElement = null;
  morph(live, h("input", { value: "server value" }));
  assert(live.value === "server value", "unfocused value updated");
});

check("nodeName mismatch replaces the node", () => {
  const oldNode = h("p", {}, ["x"]);
  const live = h("div", {}, [oldNode]);
  const next = h("div", {}, [h("section", {}, ["y"])]);
  morph(live, next);
  assert(oldNode.parentNode === null, "old node detached");
  assert(live.firstChild.nodeName === "SECTION", "replaced with section");
});

// ---------------------------------------------------------------- report
console.log(JSON.stringify({ total, passed: total - failures.length, failed: failures.length, failures }));
if (failures.length > 0) process.exit(1);
