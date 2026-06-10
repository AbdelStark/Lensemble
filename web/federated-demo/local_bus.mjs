// Multi-tab state sharing for the frontend-only simulator.
//
// The host tab owns the simulated run object and is the only writer. Other
// tabs (participant rooms opened from the join URL) send intents over a
// BroadcastChannel; the host applies them to the simulator and republishes a
// snapshot to storage + channel. Adapters are injectable so the pure logic is
// node-testable without browser globals.

const PREFIX = "lensemble-demo:";

export function defaultAdapters() {
  return {
    channelFactory: (name) => new BroadcastChannel(name),
    storage: globalThis.localStorage,
    now: () => Date.now(),
  };
}

export function memoryAdapters() {
  const channels = new Map();
  const store = new Map();
  const makeChannel = (name) => {
    const subscribers = channels.get(name) ?? [];
    channels.set(name, subscribers);
    const channel = {
      postMessage(data) {
        for (const other of subscribers) {
          if (other !== channel && typeof other.onmessage === "function") {
            other.onmessage({ data });
          }
        }
      },
      onmessage: null,
      close() {
        const idx = subscribers.indexOf(channel);
        if (idx >= 0) subscribers.splice(idx, 1);
      },
    };
    subscribers.push(channel);
    return channel;
  };
  return {
    channelFactory: makeChannel,
    storage: {
      getItem: (k) => (store.has(k) ? store.get(k) : null),
      setItem: (k, v) => store.set(k, String(v)),
      removeItem: (k) => store.delete(k),
    },
    now: () => 0,
  };
}

export function saveRunSnapshot(storage, run) {
  storage.setItem(PREFIX + run.id, JSON.stringify(snapshotOf(run)));
}

export function loadRunSnapshot(storage, runId) {
  const raw = storage.getItem(PREFIX + runId);
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

export function snapshotOf(run) {
  return {
    schema: run.schema,
    mode: run.mode,
    id: run.id,
    seed: run.seed,
    clock: run.clock,
    rngCalls: run.rngCalls,
    config: run.config,
    state: run.state,
    round: run.round,
    participants: run.participants,
    events: run.events.slice(-200),
    artifacts: run.artifacts,
    abortReason: run.abortReason,
    failureReason: run.failureReason,
  };
}

export class HostBus {
  constructor(run, applyIntent, adapters = defaultAdapters()) {
    this.run = run;
    this.adapters = adapters;
    this.applyIntent = applyIntent;
    this.channel = adapters.channelFactory(PREFIX + run.id);
    this.channel.onmessage = (event) => {
      const intent = event.data;
      if (!intent || intent.type === "snapshot") return;
      const reply = this.applyIntent(intent);
      this.publish(intent.requestId, reply);
    };
    this.publish();
  }

  publish(requestId = null, reply = null) {
    saveRunSnapshot(this.adapters.storage, this.run);
    this.channel.postMessage({
      type: "snapshot",
      requestId,
      reply,
      snapshot: snapshotOf(this.run),
    });
  }

  close() {
    this.channel.close();
  }
}

export class ParticipantBus {
  constructor(runId, onSnapshot, adapters = defaultAdapters()) {
    this.adapters = adapters;
    this.onSnapshot = onSnapshot;
    this.pending = new Map();
    this.requestCounter = 0;
    this.channel = adapters.channelFactory(PREFIX + runId);
    this.channel.onmessage = (event) => {
      const msg = event.data;
      if (!msg || msg.type !== "snapshot") return;
      if (msg.requestId && this.pending.has(msg.requestId)) {
        const resolve = this.pending.get(msg.requestId);
        this.pending.delete(msg.requestId);
        resolve(msg.reply);
      }
      this.onSnapshot(msg.snapshot);
    };
    const cached = loadRunSnapshot(adapters.storage, runId);
    if (cached) this.onSnapshot(cached);
  }

  send(intent, timeoutMs = 2000) {
    this.requestCounter += 1;
    const requestId = `req-${this.requestCounter}-${this.adapters.now()}`;
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, resolve);
      this.channel.postMessage({ ...intent, requestId });
      setTimeout(() => {
        if (this.pending.has(requestId)) {
          this.pending.delete(requestId);
          reject(new Error("no host tab responded; open the host dashboard tab first"));
        }
      }, timeoutMs);
    });
  }

  close() {
    this.channel.close();
  }
}
