// Backend API adapter for the federated browser demo (#296/#297/#299/#301).
//
// The static simulator remains the offline fallback. When served by
// `lensemble demo federated`, this adapter calls the local metadata-only API:
// create/join/status/events, participant progress, update submission, controls,
// and evidence export.

async function requestJson(path, { method = "GET", body = null } = {}) {
  if (typeof fetch !== "function") {
    throw new Error("fetch is unavailable; use simulator mode or serve the demo over HTTP");
  }
  const response = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload?.message ?? `HTTP ${response.status}`;
    const error = new Error(message);
    error.code = payload?.code ?? "api_error";
    error.status = response.status;
    throw error;
  }
  return payload;
}

export class BackendClient {
  constructor(basePath = "/api") {
    this.basePath = basePath.replace(/\/$/, "");
  }

  async health() {
    return requestJson(`${this.basePath}/health`);
  }

  async available() {
    try {
      const reply = await this.health();
      return reply?.ok === true;
    } catch {
      return false;
    }
  }

  createRun(config) {
    return requestJson(`${this.basePath}/runs`, { method: "POST", body: config });
  }

  getRun(runId) {
    return requestJson(`${this.basePath}/runs/${encodeURIComponent(runId)}`);
  }

  joinRun(runId, { joinToken, displayName, sessionId }) {
    return requestJson(`${this.basePath}/runs/${encodeURIComponent(runId)}/join`, {
      method: "POST",
      body: { joinToken, displayName, sessionId },
    });
  }

  control(runId, action, extra = {}) {
    return requestJson(`${this.basePath}/runs/${encodeURIComponent(runId)}/control`, {
      method: "POST",
      body: { action, ...extra },
    });
  }

  heartbeat(runId, participantId, participantToken) {
    return requestJson(
      `${this.basePath}/runs/${encodeURIComponent(runId)}/participants/${encodeURIComponent(
        participantId,
      )}/heartbeat`,
      { method: "POST", body: { participantToken } },
    );
  }

  progress(runId, participantId, participantToken, progress) {
    return requestJson(
      `${this.basePath}/runs/${encodeURIComponent(runId)}/participants/${encodeURIComponent(
        participantId,
      )}/progress`,
      { method: "POST", body: { participantToken, progress } },
    );
  }

  submitUpdate(runId, participantId, participantToken, artifact) {
    return requestJson(
      `${this.basePath}/runs/${encodeURIComponent(runId)}/participants/${encodeURIComponent(
        participantId,
      )}/updates`,
      { method: "POST", body: { participantToken, artifact } },
    );
  }

  async events(runId, after = -1) {
    if (typeof fetch !== "function") {
      throw new Error("fetch is unavailable");
    }
    const response = await fetch(
      `${this.basePath}/runs/${encodeURIComponent(runId)}/events?after=${encodeURIComponent(after)}`,
    );
    if (!response.ok) {
      throw new Error(`event stream failed: HTTP ${response.status}`);
    }
    const text = await response.text();
    return text
      .split(/\n+/)
      .filter(Boolean)
      .map((line) => JSON.parse(line));
  }

  exportEvidence(runId) {
    return requestJson(`${this.basePath}/runs/${encodeURIComponent(runId)}/export`);
  }
}

export const backendClient = new BackendClient();
