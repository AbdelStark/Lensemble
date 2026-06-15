// Dependency-free keyed DOM reconciler for the federated demo.
//
// The dashboard rebuilds a fresh, detached view tree on every round/poll/socket
// message using the existing builder functions. Instead of replacing the whole
// #app subtree (which re-creates charts, blinks the QR canvas, and drops scroll
// + focus), morph() patches the live tree to match the freshly-built one in
// place: identical nodes keep their identity, only changed text/attributes are
// touched, and children are reconciled by an explicit `data-key`.
//
// Three opt-out hatches keep stateful/expensive subtrees untouched:
//   - data-static   : opaque subtree — never descended into (QR canvas, inference panel)
//   - data-preserve : never reconciled while it exists (an input mid-typing)
//   - node.__chartUpdate(series) : a live chart patches itself from next.__series
//                                  instead of having its <svg> torn down.
//
// Pure structure helpers (keyOf) are exported for the node selftest.

export function keyOf(node) {
  return node.nodeType === 1 && node.dataset ? node.dataset.key ?? null : null;
}

function patchAttrs(live, next) {
  for (const attr of [...live.attributes]) {
    if (!next.hasAttribute(attr.name)) live.removeAttribute(attr.name);
  }
  for (const attr of [...next.attributes]) {
    if (live.getAttribute(attr.name) !== attr.value) live.setAttribute(attr.name, attr.value);
  }
  // Form fields carry live state as properties, not attributes. Don't clobber a
  // field the user is interacting with; otherwise mirror the freshly-built value.
  if ("value" in next && document.activeElement !== live && live.value !== next.value) {
    live.value = next.value;
  }
  if ("checked" in next && document.activeElement !== live && live.checked !== next.checked) {
    live.checked = next.checked;
  }
  // disabled is derived in the builders from module-level state (job/probe maps),
  // so a mid-flight rebuild is already-disabled and this preserves it.
  if ("disabled" in next && live.disabled !== next.disabled) live.disabled = next.disabled;
}

export function morph(live, next) {
  if (!live) return next;
  if (live.nodeType !== next.nodeType || live.nodeName !== next.nodeName) {
    live.replaceWith(next);
    return next;
  }
  if (live.nodeType === Node.TEXT_NODE || live.nodeType === Node.COMMENT_NODE) {
    if (live.nodeValue !== next.nodeValue) live.nodeValue = next.nodeValue;
    return live;
  }
  if (live.dataset && live.dataset.preserve !== undefined) return live; // mid-typing
  if (live.dataset && live.dataset.static !== undefined) return live; // QR, inference panel — fully opaque
  patchAttrs(live, next);
  if (typeof live.__chartUpdate === "function" && next.__series) {
    live.__chartUpdate(next.__series); // in-place chart patch, no <svg> teardown
    return live;
  }
  morphChildren(live, next);
  return live;
}

function morphChildren(live, next) {
  const liveKeyed = new Map();
  for (const child of live.childNodes) {
    const k = keyOf(child);
    if (k != null) liveKeyed.set(k, child);
  }
  const unkeyedLive = [];
  for (const child of live.childNodes) if (keyOf(child) == null) unkeyedLive.push(child);

  // Pass 1: pick the live node to reuse (or the new node) for each desired child,
  // and recurse so reused subtrees are patched before we reorder.
  const desired = [];
  let unkeyedIdx = 0;
  for (let nc = next.firstChild; nc; nc = nc.nextSibling) {
    const k = keyOf(nc);
    let chosen;
    if (k != null && liveKeyed.has(k)) {
      chosen = morph(liveKeyed.get(k), nc);
    } else if (k == null) {
      let cand = null;
      while (unkeyedIdx < unkeyedLive.length) {
        const candidate = unkeyedLive[unkeyedIdx++];
        if (candidate.nodeName === nc.nodeName && keyOf(candidate) == null) {
          cand = candidate;
          break;
        }
      }
      chosen = cand ? morph(cand, nc) : nc;
    } else {
      chosen = nc; // keyed but no live match → fresh node
    }
    desired.push(chosen);
  }

  // Pass 2: remove live children we no longer want, then splice `desired` into order.
  const keep = new Set(desired);
  for (const child of [...live.childNodes]) {
    if (!keep.has(child)) child.remove();
  }
  let ref = live.firstChild;
  for (const node of desired) {
    if (node === ref) {
      ref = ref.nextSibling;
    } else {
      live.insertBefore(node, ref); // inserts a new node, or moves an out-of-order one
    }
  }
}
