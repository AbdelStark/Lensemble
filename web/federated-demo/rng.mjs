// Deterministic RNG + identifier helpers for the federated demo simulator.
// mulberry32 keeps the frontend-only mock reproducible from a single seed so
// node-based selftests can pin exact run timelines.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randomSeed() {
  if (globalThis.crypto?.getRandomValues) {
    const buf = new Uint32Array(1);
    globalThis.crypto.getRandomValues(buf);
    return buf[0] >>> 0;
  }
  return Math.floor(Math.random() * 4294967296) >>> 0;
}

// Crockford base32: unambiguous, lowercase, URL-safe.
const ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz";

export function randomToken(rng, length) {
  let out = "";
  for (let i = 0; i < length; i += 1) {
    out += ALPHABET[Math.floor(rng() * ALPHABET.length) % ALPHABET.length];
  }
  return out;
}

export function newRunId(rng) {
  return `run-${randomToken(rng, 8)}`;
}

export function newParticipantId(rng) {
  return `browser-${randomToken(rng, 6)}`;
}

export function pseudoHash(rng) {
  let out = "";
  for (let i = 0; i < 64; i += 1) {
    out += "0123456789abcdef"[Math.floor(rng() * 16) % 16];
  }
  return out;
}

export const RUN_ID_PATTERN = /^run-[0-9abcdefghjkmnpqrstvwxyz]{8}$/;
export const PARTICIPANT_ID_PATTERN = /^browser-[0-9abcdefghjkmnpqrstvwxyz]{6}$/;
