# Cartographer — Demo-Night Runsheet

Format (from the Luma page): full-day solo-builder hack, Codex-sponsored; breakout groups pick standout projects; top ~5 do live demos at Demo Night (19:00–22:00). So there are **two** audiences: the **breakout group** (mid-day, picks who advances) and the **Demo-Night room** (the big stage).

Golden rule: **a working fallback exists at every hour** (Decision D8). We always demo *something*; the live federation round and the latest bake only upgrade it.

---

## The 90-second Demo-Night script

> Spoken beats in quotes; on-screen action in brackets.

1. **The hook (0:00–0:15).**
   "Genie and Sora generate pixels and need a datacenter. This is a JEPA world model — it thinks in a 192-dimensional latent space, so it plans on a laptop. And a room of people just trained it together without sharing any data. Let me show you its imagination."
   [Viewer open, point cloud slowly rotating.]

2. **The manifold (0:15–0:35).**
   "Every point is a state this world model believes the world can be in — harvested from real trajectories. This isn't a render; it's the geometry of its thoughts."
   [Rotate; hover shows episode/step.]

3. **Watch it plan (0:35–0:55).**
   "Give it a goal." [Click a plan.] "Now it searches — in latent space — for a way there. These faint branches are candidate futures it considered; the bright one is the plan it chose."
   [Plan trail ignites toward the goal node; candidates flicker, elites brighten.]

4. **Why it works — anti-collapse (0:55–1:10).**
   "World models love to cheat by collapsing everything to a point. Toggle that off—" [healthy→collapsed] "—and the structure dies; planning is impossible. Keeping this structure is the hard part, and it's what the project is built on." [Show eff-rank 9.86 → 1.0 in the HUD.]

5. **The room's contribution (1:10–1:25).**
   "And this—" [pre→post-federation toggle] "—is before and after the room trained it together. Held-out prediction error dropped 12%. Nobody shared a single frame of their data." [HUD shows the federated improvement, sourced to the evidence file.]

6. **Land it (1:25–1:30).**
   "A world model you can see, that plans on a laptop, improved by a crowd that kept its data private. Everything on screen is backed by a generated evidence file. Thanks."
   [nonClaims footer visible; provenance card with checkpoint hash.]

---

## Live Act-1 (optional, audience-participation upgrade)

If the room/network allows, run the **live federation round** before opening the viewer:
1. Open `web/federated-demo/` (`uv run lensemble demo federated --port 8765`), show the QR.
2. 2–4 people join from phones; each runs a local adapter round; aggregate.
3. Either re-bake the manifold on the post-round revision (if time/laptop allows) **or** just show the federated demo's own probe tick up, then cut to the pre-baked Cartographer viewer.

Risk: re-baking live is slow/fragile. **Default: do NOT re-bake live.** Use the pre-baked post-federation manifold and narrate the live round as "this is the same kind of round that produced what you're about to see."

---

## Fallback ladder (use the highest rung that works)

| Rung | What | When |
|---|---|---|
| A | Live federation round + freshly-baked manifold | only if rehearsed and fast on the day |
| B | Pre-baked **real** manifold (today's bake) in the viewer | default |
| C | Committed **fallback** manifold (baked pre-event) in the viewer | if today's bake failed |
| D | The ≤20 s screen-capture clip | if WebGPU/WebGL won't run on the projector |

Always have rung C and D ready before walking on stage.

---

## Pre-flight checklist (run during 16:30–17:30)

- [ ] `scripts/cartographer/rehearsal.py` green.
- [ ] Viewer renders `manifold.json` in the **presentation browser** at full-screen on the **projector resolution** (test the actual cable/adapter).
- [ ] WebGL path confirmed (don't assume WebGPU on an unknown projector machine).
- [ ] All HUD numbers equal `docs/evidence/lewm_tworooms_manifold.json`.
- [ ] nonClaims footer visible and correct.
- [ ] Committed fallback `manifold.json` present; clip exported; result-card image exported.
- [ ] Laptop: power, sleep disabled, notifications off, network for QR (if doing live Act-1).
- [ ] One-paragraph X post drafted (latent-not-pixels + crowd-trained-private framing), clip attached.

---

## Capture assets (for sharing)

- **Clip (≤20 s):** rotate → plan ignites → collapse toggle → pre/post toggle. 1080p screen recording, no audio or with a short VO.
- **Result card:** a single image with the eff-rank healthy/collapsed contrast, the +12% federated number, and "plans on a laptop, in latent space." Generate via the repo's `academic-plotting` skill or a simple matplotlib card; cite the evidence file.
- Keep claims in the post within the approved language (doc `05`).
