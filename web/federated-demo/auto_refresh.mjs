const EDITING_TAGS = new Set(["input", "select", "textarea"]);

export function activeElementBlocksAutoRefresh(activeElement) {
  if (!activeElement) return false;
  const tagName = String(activeElement.tagName ?? "").toLowerCase();
  if (EDITING_TAGS.has(tagName)) return true;
  return Boolean(activeElement.isContentEditable);
}

export function shouldDeferAutoRefresh({
  documentHasFocus = true,
  activeElement = null,
} = {}) {
  return Boolean(documentHasFocus) && activeElementBlocksAutoRefresh(activeElement);
}

export function shouldDeferAutoRefreshForDocument(doc = globalThis.document) {
  if (!doc) return false;
  const documentHasFocus = typeof doc.hasFocus === "function" ? doc.hasFocus() : true;
  return shouldDeferAutoRefresh({
    documentHasFocus,
    activeElement: doc.activeElement,
  });
}
