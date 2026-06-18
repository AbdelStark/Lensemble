// Join-URL and hash-route helpers.
//
// Join URLs are shareable links of the form
//   <base>#/join/<runId>            (frontend-only simulator mode)
//   <base>#/join/<runId>?t=<token>  (backend mode; token minted by the API)
// The QR code on the host view encodes exactly this URL.

import { RUN_ID_PATTERN } from "./rng.mjs";

export function buildJoinUrl(baseUrl, runId, token = null) {
  const base = String(baseUrl).split("#")[0];
  const suffix = token ? `?t=${encodeURIComponent(token)}` : "";
  return `${base}#/join/${encodeURIComponent(runId)}${suffix}`;
}

export function parseRoute(hash) {
  const raw = String(hash ?? "").replace(/^#/, "");
  if (raw === "" || raw === "/") {
    return { view: "home" };
  }
  const [path, query = ""] = raw.split("?");
  const parts = path.split("/").filter((p) => p.length > 0);
  const params = new URLSearchParams(query);
  if (parts[0] === "join" && parts.length === 2) {
    return {
      view: "join",
      runId: decodeURIComponent(parts[1]),
      token: params.get("t") ?? null,
    };
  }
  if (parts[0] === "host" && parts.length === 2) {
    return { view: "host", runId: decodeURIComponent(parts[1]), saleId: params.get("sale") ?? null };
  }
  if (parts[0] === "economy" && parts.length === 2) {
    return {
      view: "economy",
      runId: decodeURIComponent(parts[1]),
      saleId: params.get("sale") ?? null,
    };
  }
  if (parts[0] === "admin" && parts.length === 2) {
    return { view: "admin", runId: decodeURIComponent(parts[1]) };
  }
  if (parts[0] === "tworooms" && parts.length === 1) {
    return { view: "tworooms" };
  }
  return { view: "unknown", path: raw };
}

export function isValidRunId(runId) {
  return RUN_ID_PATTERN.test(String(runId ?? ""));
}
