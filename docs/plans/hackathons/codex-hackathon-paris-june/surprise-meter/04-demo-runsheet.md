# Surprise-meter — Demo-Night Runsheet

Two audiences (Luma page): the **breakout group** (mid-day, picks who advances) and the **Demo-Night room** (19:00–22:00, top ~5 live).

Golden rule: **Milestone 0 alone is a valid demo.** The surprise-meter is the upgrade. Never walk on stage without at least the clean federated run + a recorded-trajectory fallback.

---

## The 90-second Demo-Night script

1. **The hook (0:00–0:15) — impact-first, jargon-second.**
   [Agent already gliding, meter low.] "This little AI is predicting the future — and it's about to be *surprised*. Watch." [Teleport → meter spikes.] "That spike is a world model realizing it was wrong. In the next 80 seconds: a room of strangers is going to make it *less* surprised — without sharing any data." *(Name on the title card: **Less Surprised**. Save "JEPA / latent space" for beat 2, once they care.)*

2. **Surprise is real (0:15–0:35).**
   [Agent glides through TwoRooms; meter low.] "The model predicted all of this — low surprise. Watch what happens when the world does something it didn't expect." [Teleport / push through wall → meter spikes.] "That spike is prediction error in latent space — literally the signal this model is trained on."

3. **Surprise ≠ motion (0:35–0:50).**
   [Point to the frame-diff trace.] "This isn't just motion detection. Here the agent moves fast but surprise stays low — predictable. Here it barely moves but surprise spikes — unexpected. Surprise is about *prediction*, not pixels changing."

4. **The room's contribution (0:50–1:15).**
   "Earlier, people in this room each trained a tiny adapter — 12,512 numbers, 0.07% of the model — on their own private trajectories. Nothing left their device. We averaged the clipped updates into one shared revision." [pre→post toggle on the held-out set.] "After that, the model is measurably less surprised by normal physics — held-out prediction error dropped 12% on this run, 17% on average across five seeds, and 5% in the *worst* seed." [HUD shows +12.3% **and** +5.4% worst, both sourced to the evidence file.] *(Showing the worst seed is the point — this is the honest version.)*

5. **Land it (1:15–1:30).**
   "A world model you can watch think, that a crowd improved while keeping their data private — running on a laptop, every number on screen backed by a generated, audited evidence file. And the whole thing — plan, typed contracts, the evidence gate — was built in a Codex loop. Thanks."

---

## Live Milestone 0 (the federated run) — optional theatre, not the proof

The live QR round is **atmosphere**, not the load-bearing number. The certified +12.3% comes from the **headless** `run_system_composed_probe` path; whether a 2–4-phone live round produces a clean positive tick-up in a ~2-minute stage window is **unverified** and must not be the thing the demo's credibility rests on. So:
1. `uv run lensemble demo federated --port 8765` → show the QR at `http://127.0.0.1:8765/web/federated-demo/`.
2. 2–4 people join from phones; each runs a local adapter round; aggregate; narrate the probe tick-up as it happens.
3. **The pre/post toggle still reads the certified evidence + the pre-baked offset** (`web/surprise-meter/fixtures/adapter_offset.json`) — do not re-point the HUD at the live round's number on stage.

Default if fragile or short on time: narrate the live round for the crowd-participation wow, but drive the meter's pre/post from the **pre-baked** offset (rung B below). Per the golden rule, Milestone 0's headless evidence alone is a valid demo.

---

## Fallback ladder (use the highest rung that works)

| Rung | What | When |
|---|---|---|
| A | Live federated round → surprise-meter (pre/post still from certified evidence + pre-baked offset) | only if rehearsed + fast |
| B | Live surprise-meter computing from ONNX, **pre-baked** offset for pre/post | default |
| C | Recorded `surprise_trajectory.json` (`lewm-surprise-traj/1`) replayed in the meter (no live ONNX) | if WebGPU/ONNX won't run |
| D | The ≤20 s capture clip | if the projector machine won't run the page |

> **Start-of-day BLOCKING pre-req (2026-06-18):** verify rungs B/C/D before relying on live runtime. `onnxruntime` is absent from the uv venv, system node, and npm, so bake `web/surprise-meter/fixtures/adapter_offset.json` (len 12512, via `uv run --with onnxruntime --with hdf5plugin python scripts/surprise/run_clean_round.py ...`) **and** regenerate `docs/evidence/lewm_tworooms_surprise.json`, `web/surprise-meter/data/result_card.json`, and `web/surprise-meter/data/surprise_trajectory.json` via `uv run python scripts/lewm_surprise_check.py`. Commit the served fallback assets outside the gitignored `runs/` path. Have rungs C and D on disk before walking on.

---

## Pre-flight checklist (16:00–17:30)
- [ ] Start-of-day pre-bake DONE: `web/surprise-meter/fixtures/adapter_offset.json` (len 12512, nonzero), `web/surprise-meter/data/surprise_trajectory.json` (`lewm-surprise-traj/1`), and the <=20 s clip all committed (R13).
- [ ] `scripts/surprise/rehearsal.py` green; `docs/evidence/lewm_tworooms_surprise.json` passes its test.
- [ ] Milestone-0 one-command run reproduces audited evidence; the offset sidecar (len 12512) is produced.
- [ ] **ONNX integrity (fail-closed gate):** the committed `model/lewm-tworooms/*.onnx` are fully present and their SHA-256 match `manifest.json` (sizes: encoder 25,923,986 · predictor 46,910,027 · action 641,851). A stale/partial graph throws `hash-mismatch` before the meter starts — verify with a 3-line node/python check at pre-flight.
- [ ] `web/surprise-meter/` renders in the **presentation browser** at projector resolution from the **vendored** `ort.webgpu.min.js`. **Confirm the `?ep=wasm` force-WASM path** — do NOT assume WebGPU on an unknown machine; there is no automatic WebGPU→WASM fallback (R5).
- [ ] **In-browser R1 check (R1/R13):** ≥1 perturbation channel spikes the meter (ratio >1.5); else foreground OOD-action + pre/post (S8). Frame-diff trace visible.
- [ ] Pre/post toggle shows the drop on the held-out set; HUD shows **+12.3% and +5.4% worst**, both equal to the evidence files at full precision.
- [ ] nonClaims footer visible/correct; committed fallback offset + recorded trajectory present and tracked; clip + result card exported.
- [ ] Laptop: power, sleep off, notifications off, network for QR (if live Milestone 0).
- [ ] X post drafted (less-surprised-after-private-crowd-training framing, **with the worst seed**), within approved language (doc `05`).

---

## Capture assets
- **Clip (≤20 s):** glide (low) → perturb (spike) → frame-diff contrast → pre/post toggle (drop). 1080p.
- **Result card:** the federated improvement as **"+12.3% this run · +16.8% mean · +5.4% worst (5 seeds)"** + "less surprised after a room trained an adapter — no data shared," citing the evidence file. Optional footer: "built in a Codex loop."
