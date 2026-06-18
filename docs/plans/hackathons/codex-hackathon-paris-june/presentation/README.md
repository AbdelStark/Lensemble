# Sovereign Robotics World-Model Economy — Demo-Night deck (DRAFT)

A self-contained [reveal.js](https://revealjs.com/) presentation for the Codex Hackathon (Paris).
It is the slide backdrop for the 90-second Demo-Night pitch, now grounded in the
parent [sovereign-economy](../sovereign-economy/) plan with the
[surprise-meter](../surprise-meter/) as the model-quality proof.

> **Status: stage v0.4.** The horizontal spine now mirrors
> [`sovereign-economy/04-demo-runsheet.md`](../sovereign-economy/04-demo-runsheet.md)
> beat-for-beat: sovereignty risk -> federated adapter-continuation proof ->
> buyer checkout -> contribution-weighted reward split. The runsheet remains the
> source of truth for the spoken script.

## Design — "The Offprint"

The deck is styled as **an article torn from a scientific journal** — chosen because the
project's whole edge is evidence and restraint, so the layout itself should read as audited,
not hyped. Deliberately *not* the dark-neon-gradient "AI deck" look.

- **Paper, not screen.** Warm bone ground (never pure white), warm near-black ink, and a
  single **oxblood** pigment for the one thing that matters on each slide. No gradients, no
  glow, no glassmorphism, no cards.
- **Real typography.** [Fraunces](https://fonts.google.com/specimen/Fraunces) (display, with
  optical sizing + true italics for the one emphasized phrase per slide),
  [Newsreader](https://fonts.google.com/specimen/Newsreader) (body prose), and
  [Spline Sans Mono](https://fonts.google.com/specimen/Spline+Sans+Mono) **quarantined to
  data only** (figures, file paths, the folio, the formula).
- **The signature figure.** The surprise signal is a **registered ink seismograph**
  (`assets/deck.js`) — a continuous recorded trace that inks its spike in oxblood with an
  event rule, exactly the way a journal once printed an instrument recording. A faint
  "heartbeat" of the same signal runs in the running head on every slide.
- **Journal chrome.** Every slide carries a running head, a live figure folio (`Fig. N / 9`),
  and a footnote rail — so the audited-numbers ethos is encoded in the layout, not asserted
  on top of it. Flush-left asymmetric grid; one gesture per slide.

## Run it

It's a static page — any static server works (reveal.js needs `http://`, not `file://`):

```bash
# from the repo root
python3 -m http.server 8000 --directory docs/plans/hackathons/codex-hackathon-paris-june/presentation
# → open http://localhost:8000
```

Speaker view (notes + timer + next slide): press **`S`**. Fullscreen: **`F`**. Overview: **`Esc`**.
The spoken beats from the runsheet live in each slide's speaker notes.

## Present offline (do this before the venue)

The deck loads reveal.js from a CDN by default. Venue Wi-Fi is exactly the kind of thing
the rest of this project refuses to depend on, so vendor it locally first:

```bash
cd docs/plans/hackathons/codex-hackathon-paris-june/presentation
./vendor.sh                      # downloads reveal.js@5.1.0 AND the 3 variable fonts into ./vendor/
```

Then set the flag near the top of `index.html`:

```js
window.DECK_USE_VENDOR = true;   // load reveal.js + fonts from ./vendor instead of the CDN
```

Once vendored the deck has **zero network dependencies** — reveal.js and all three fonts are
served locally (the deck.js seismograph is pure 2-D canvas, no GPU/CDN). The fonts fall back
to a serif/mono system stack if a face ever fails to load, so a cold venue machine never blanks.

## What's in here

| File | What |
|---|---|
| `index.html`      | The slides (8-slide live spine + an appendix stack), the journal chrome, and the font/reveal loader. |
| `assets/theme.css`| The "Offprint" design system (oklch pigments, the type scale, the seismograph plate, ledger tables). |
| `assets/deck.js`  | The registered-ink **seismograph** recorder (canvas) + reveal.js config (flush-left, uncentered). |
| `vendor.sh`       | One-shot offline vendoring of reveal.js **and** the variable fonts. |

## Slide map

The **horizontal spine = the live 90s arrow path** (matches `../sovereign-economy/04-demo-runsheet.md`). Press
**→** to stay on the spoken script; press **↓ / Space** for the judge/breakout depth under
each slide.

**Live spine (→):**

1. Title — *Sovereign robotics world-model economy*
2. Problem — frontier model access restrictions and sovereignty risk
3. What surprise means — scalar next-latent prediction error
4. Surprise is not motion — frame-diff contrast
5. Federated adapter-continuation setup — raw data stays local
6. Mechanism — 12,512-param adapter, clipped deltas, frozen backbone
7. Proof — held-out surprise drops +12.3% this run / +16.8% mean / +5.4% worst
   · ↓ seed table · ↓ evidence flow · ↓ claim do/don't columns
8. Buyer — a humanoid robotics buyer wants the improved revision
9. Rewards — Mollie test checkout boundary + orchestrator/community split ledger
10. Close — sovereign model improvement with shared upside + the Codex loop beat

**Appendix (↓ under slide 11 — off the live path, for Q&A / the breakout):**

11. Milestones (M0 must / M1 ship / M2 stretch) · Cartographer (#339) · Codex loop ·
   Architecture · fallback ladder · Sources

> Why the split: the runsheet's 90s never visits milestones/architecture/fallback — those
> are build-plan content. Keeping them as press-down depth means a presenter advancing with
> **→** never desyncs from the spoken script, but everything is one keypress away for judges.

## Every number is sourced (verified against the evidence files)

| On screen | Value | Source |
|---|---|---|
| this run | **+12.3%** (`relativeImprovement = 0.1227556578…`) | `docs/evidence/lewm_tworooms_system_probe.json` |
| mean (5 seeds) | **+16.8%** (`0.16787…`) | `docs/evidence/lewm_tworooms_probe_seedsweep.json` |
| worst seed (seed 2) | **+5.4%** (`0.054144…`) | same |
| best seed (seed 4) | **+32.6%** (`0.32634…`) | same |
| stdev / collapse | 0.11 · all 5 improved · no collapse | same |
| adapter | 12,512 params = 0.069% | AGENTS.md § Claim Discipline |
| model | 192-d CLS latent · 3-frame window · no decoder · ~6 ms/step CPU | spike #335 / README |
| eff-rank | healthy 9.86 / 192 → collapsed 1.0 | cartographer plan (D6) |

## Claim discipline (binding — do not drift on stage)

Mirrors [`sovereign-economy/05-risks-and-claim-discipline.md`](../sovereign-economy/05-risks-and-claim-discipline.md),
[`surprise-meter/05-risks-and-claim-discipline.md`](../surprise-meter/05-risks-and-claim-discipline.md),
and `AGENTS.md § Claim Discipline`:

- It is **federated adapter continuation on a frozen checkpoint** — never "federated
  world-model training" / "the room trained the model" / "a clean federated training run".
- The federated improvement is **always** stated with the worst seed beside the mean:
  *+12.3% this run, +16.8% mean / +5.4% worst across 5 seeds.* Never the mean alone.
- Surprise is a **scalar per-frame** next-latent prediction error — never a per-pixel heatmap.
- Use placeholders: "frontier model access restrictions" and "a humanoid robotics buyer."
  Do not name specific companies or incidents without separate verification and approval.
- Payments and rewards are **test/simulation only**. No real payout, legal revenue-share,
  securities, or employment claim.
- **No** DP, secure-aggregation, cryptographic-proof, beats-local-only, anomaly-detector,
  or paper-scale claims on this path.

## Editing during the hackathon

- Swap the animated meters for the real screen-capture clip on the hook / result slides once
  it exists (drop a `<video>` in place of `<canvas data-meter>`).
- Update slide 7 if the live round's "this run" number changes — but keep the **certified
  evidence** numbers for the headline (the live tick-up is labelled "this run" and stays
  distinct, per decision S10).
- Keep slides 8 and the appendix honest: if a claim can't be sourced, cut it.
