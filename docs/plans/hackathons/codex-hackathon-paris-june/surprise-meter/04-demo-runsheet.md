# Surprise-meter — Demo-Night Runsheet

Two audiences (Luma page): the **breakout group** (mid-day, picks who advances) and the **Demo-Night room** (19:00–22:00, top ~5 live).

Golden rule: **Milestone 0 alone is a valid demo.** The surprise-meter is the upgrade. Never walk on stage without at least the clean federated run + a recorded-trajectory fallback.

---

## The 90-second Demo-Night script

1. **The hook (0:00–0:15).**
   "This is a JEPA world model — it predicts the future in a latent space, not in pixels. I'm going to show you it being *surprised*, and then show you a room of people making it *less* surprised, without sharing any of their data."

2. **Surprise is real (0:15–0:35).**
   [Agent glides through TwoRooms; meter low.] "The model predicted all of this — low surprise. Watch what happens when the world does something it didn't expect." [Teleport / push through wall → meter spikes.] "That spike is prediction error in latent space — literally the signal this model is trained on."

3. **Surprise ≠ motion (0:35–0:50).**
   [Point to the frame-diff trace.] "This isn't just motion detection. Here the agent moves fast but surprise stays low — predictable. Here it barely moves but surprise spikes — unexpected. Surprise is about *prediction*, not pixels changing."

4. **The room's contribution (0:50–1:15).**
   "Earlier, people in this room each trained a tiny adapter on their own private trajectories — nothing left their device — and we aggregated it. Watch the same run, before and after." [pre→post toggle.] "After federation, the model is measurably less surprised by normal physics — held-out prediction error dropped 12%." [HUD shows the certified number, sourced to the evidence file.]

5. **Land it (1:15–1:30).**
   "A world model you can watch think, that a crowd improved while keeping their data private — running on a laptop. Everything on screen is backed by a generated evidence file. Thanks."

---

## Live Milestone 0 (the federated run) — recommended opener

If the room/network allows, run it live before the meter:
1. `uv run lensemble demo federated --port 8765` → show the QR at `http://127.0.0.1:8765/web/federated-demo/`.
2. 2–4 people join from phones; each runs a local adapter round; aggregate; the probe ticks up on screen.
3. Export the resulting offset → load it into the surprise-meter's pre/post toggle (or use the pre-baked offset if re-loading is slow).

Default if fragile: narrate the live round, but drive the meter's pre/post from the **pre-baked** offset (rung B below).

---

## Fallback ladder (use the highest rung that works)

| Rung | What | When |
|---|---|---|
| A | Live federated round → live surprise-meter on the fresh offset | only if rehearsed + fast |
| B | Live surprise-meter computing from ONNX, **pre-baked** offset for pre/post | default |
| C | Recorded `surprise_trajectory.json` replayed in the meter (no live ONNX) | if WebGPU/ONNX won't run |
| D | The ≤20 s capture clip | if the projector machine won't run the page |

Have rungs C and D ready before walking on.

---

## Pre-flight checklist (16:00–17:30)
- [ ] `scripts/surprise/rehearsal.py` green; `docs/evidence/lewm_tworooms_surprise.json` passes its test.
- [ ] Milestone-0 one-command run reproduces audited evidence.
- [ ] `web/surprise-meter/` renders in the **presentation browser** at projector resolution; WebGL/WASM path confirmed (don't assume WebGPU on an unknown machine).
- [ ] Perturbation buttons spike the meter; frame-diff trace visible.
- [ ] Pre/post toggle shows the drop; HUD number equals the evidence file.
- [ ] nonClaims footer visible/correct; committed fallback offset + recorded trajectory present; clip + result card exported.
- [ ] Laptop: power, sleep off, notifications off, network for QR (if live Milestone 0).
- [ ] X post drafted (less-surprised-after-private-crowd-training framing), within approved language (doc `05`).

---

## Capture assets
- **Clip (≤20 s):** glide (low) → perturb (spike) → frame-diff contrast → pre/post toggle (drop). 1080p.
- **Result card:** the +12% federated number + "less surprised after the room trained it, no data shared," citing the evidence file.
