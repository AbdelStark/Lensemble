// Backend API adapter for the federated browser demo (#296/#297/#299/#301/#305).
//
// The static simulator remains the offline fallback. When served by
// `lensemble demo federated`, this adapter calls the local demo API:
// create/join/status/events, participant progress, update submission, controls,
// WebSocket event fanout, and evidence export.

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

  modelRevision(runId, revisionId) {
    return requestJson(
      `${this.basePath}/runs/${encodeURIComponent(runId)}/model-revisions/${encodeURIComponent(revisionId)}`,
    );
  }

  webSocketUrl(runId, { role = "host", participantId = null, participantToken = null, after = -1 } = {}) {
    const origin =
      typeof window !== "undefined" && window.location
        ? window.location.origin
        : "http://127.0.0.1";
    const wsOrigin = origin.startsWith("https://")
      ? `wss://${origin.slice("https://".length)}`
      : `ws://${origin.replace(/^http:\/\//, "")}`;
    const params = new URLSearchParams({ role, after: String(after) });
    if (participantId) params.set("participantId", participantId);
    return `${wsOrigin}${this.basePath}/runs/${encodeURIComponent(runId)}/ws?${params.toString()}`;
  }

  webSocketProtocols({ participantToken = null } = {}) {
    return participantToken ? [`ptok.${participantToken}`] : [];
  }

  connectRun(runId, options = {}) {
    if (typeof WebSocket !== "function") {
      return null;
    }
    const protocols = this.webSocketProtocols(options);
    const url = this.webSocketUrl(runId, options);
    const socket = protocols.length > 0 ? new WebSocket(url, protocols) : new WebSocket(url);
    let heartbeatTimer = null;
    const clearHeartbeat = () => {
      if (heartbeatTimer !== null) {
        clearInterval(heartbeatTimer);
        heartbeatTimer = null;
      }
    };
    const sendKeepalive = () => {
      if (socket.readyState !== WebSocket.OPEN) return;
      socket.send(JSON.stringify({ type: options.role === "participant" ? "heartbeat" : "ping" }));
    };
    socket.addEventListener("message", (event) => {
      try {
        options.onMessage?.(JSON.parse(event.data));
      } catch (error) {
        options.onError?.(error);
      }
    });
    socket.addEventListener("open", () => {
      options.onOpen?.();
      const intervalMs = options.role === "participant" ? 5000 : 15000;
      sendKeepalive();
      heartbeatTimer = setInterval(sendKeepalive, intervalMs);
    });
    socket.addEventListener("close", () => {
      clearHeartbeat();
      options.onClose?.();
    });
    socket.addEventListener("error", () => {
      clearHeartbeat();
      options.onError?.(new Error("WebSocket connection failed"));
    });
    return socket;
  }

  exportEvidence(runId) {
    return requestJson(`${this.basePath}/runs/${encodeURIComponent(runId)}/export`);
  }
}

export const backendClient = new BackendClient();
