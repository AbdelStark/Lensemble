# Sovereign world models · Demo-Night deck

A self-contained [reveal.js](https://revealjs.com/) presentation for the Codex Hackathon
(Paris). It is the slide backdrop for the Demo-Night pitch of Lensemble, a sovereign
robotics world model that a community improves with federated learning, proves a measured
gain, and then monetizes through a contribution-weighted reward ledger.

## Design: "The Offprint"

The deck is styled as **an article torn from a scientific journal**, chosen because the
project's edge is evidence and restraint, so the layout itself should read as audited rather
than hyped. Deliberately not the dark-neon-gradient "AI deck" look.

- **Paper, not screen.** Warm bone ground (never pure white), warm near-black ink, and a
  single **oxblood** pigment for the one thing that matters on each slide. No gradients, no
  glow, no glassmorphism, no cards.
- **Real typography.** [Fraunces](https://fonts.google.com/specimen/Fraunces) for display
  (optical sizing and true italics for the one emphasized phrase per slide),
  [Newsreader](https://fonts.google.com/specimen/Newsreader) for body prose, and
  [Spline Sans Mono](https://fonts.google.com/specimen/Spline+Sans+Mono) kept to data only
  (figures, file paths, the folio, the formula).
- **The signature figure.** The surprise signal is a **registered ink seismograph**
  (`assets/deck.js`): a continuous recorded trace that inks its spike in oxblood, the way a
  journal once printed an instrument recording. A calm "heartbeat" of the same signal runs
  under the running head on every slide. The proof figure uses a dedicated `settle` state so
  the trace reads loud-before settling to quiet-after, never the reverse.
- **Journal chrome.** Every slide carries a running head, a live figure folio (`Fig. N / 13`),
  and a footnote rail, so the audited-numbers ethos is encoded in the layout, not asserted on
  top of it. Flush-left asymmetric grid, one gesture per slide.

## Run it

It is a static page, so any static server works (reveal.js needs `http://`, not `file://`):

```bash
# from the repo root
python3 -m http.server 8000 --directory docs/plans/hackathons/codex-hackathon-paris-june/presentation
# then open http://localhost:8000
```

Speaker view (notes, timer, next slide): press **`S`**. Fullscreen: **`F`**. Overview: **`Esc`**.
The spoken beats live in each slide's speaker notes.

## Standalone export (the fallback)

If the live presentation machine, the venue Wi-Fi, or the CDN ever misbehaves, run
the fallback instead. It is built by one command:

```bash
python3 scripts/build_presentation_export.py          # standalone HTML
python3 scripts/build_presentation_export.py --pdf     # HTML + a static PDF (needs Chrome)
```

This writes into `dist/` (git-ignored, regenerable):

- **`dist/lensemble-deck.html`** is a single self-contained file with reveal.js, both
  plugins, the theme, the seismograph script, and all five variable fonts inlined as
  base64. It has zero external references, so you can **double-click it** (it runs from
  `file://`, no server and no network) and the deck looks identical to the live one,
  animated trace and all.
- **`dist/lensemble-deck.pdf`** is a static, no-JS, one-slide-per-page PDF in talk order.
  It opens on any device and is the most failure-proof backup. It is captured from the
  standalone file so it matches the bespoke layout exactly (reveal's generic `?print-pdf`
  does not handle this deck's persistent fixed chrome).

The launcher reuses an export too: `scripts/presentation.sh` serves the live deck for the
real talk; keep the standalone HTML or the PDF on the same machine as the safety net.

## Present offline (do this before the venue)

The deck loads reveal.js from a CDN by default. Venue Wi-Fi is exactly the kind of thing the
rest of this project refuses to depend on, so vendor it locally first:

```bash
cd docs/plans/hackathons/codex-hackathon-paris-june/presentation
./vendor.sh                      # downloads reveal.js@5.1.0 AND the 3 variable fonts into ./vendor/
```

Then set the flag near the top of `index.html`:

```js
window.DECK_USE_VENDOR = true;   // load reveal.js + fonts from ./vendor instead of the CDN
```

Once vendored, the deck has **zero network dependencies**. The fonts fall back to a
serif/mono system stack if a face ever fails to load, so a cold venue machine never blanks.

## What's in here

| File | What |
|---|---|
| `index.html`      | The slides (12-slide spine + an appendix stack), the journal chrome, the font/reveal loader. |
| `assets/theme.css`| The "Offprint" design system (oklch pigments, type scale, the seismograph plate, ledger tables). |
| `assets/deck.js`  | The registered-ink **seismograph** recorder (canvas) + reveal.js config (flush-left, uncentered). |
| `vendor.sh`       | One-shot offline vendoring of reveal.js **and** the variable fonts. |

## Slide map

The **horizontal spine is the spoken arc**. Press **→** to follow the script, **↓ / Space**
for the depth under a slide.

**Spine (→):**

1. Cover, *Sovereign world models*
2. The question, *Who gets to own intelligence?*
3. The bargain, the data deal and the turn into the alternative
4. The alternative, *a world model the world improves together*
5. Watch it think, what surprise is (↓ the formula)
6. Surprise is not motion, the frame-difference contrast
7. The room, a federated revision that made the model less surprised
8. How it works, frozen backbone, a 12,512-param adapter, clipped deltas
9. The proof, held-out error dropped 12.3% (↓ the five-seed table)
10. The economy, a robotics buyer pays EUR 1,000,000 for the better model
11. The reward ledger, the upside split by contribution weight
12. The close, *intelligence the world improves, and the world owns*

**Appendix (↓ under slide 13, for the breakout and Q&A):**

13. In depth, Cartographer (see the model's mind), how the system is wired, reproducibility.

## Every number is sourced (verified against the evidence files)

| On screen | Value | Source |
|---|---|---|
| this run | **12.3%** (`relativeImprovement = 0.1227556578…`) | `docs/evidence/lewm_tworooms_system_probe.json` |
| mean (5 seeds) | **16.8%** (`0.16787…`) | `docs/evidence/lewm_tworooms_probe_seedsweep.json` |
| worst seed (seed 2) | **5.4%** (`0.054144…`) | same |
| best seed (seed 4) | **32.6%** (`0.32634…`) | same |
| stdev | 0.11 (11 points) · all 5 improved | same |
| adapter | 12,512 params = 0.07% | AGENTS.md § Claim Discipline |
| model | 192-d latent · 3-frame window · no decoder · ~6 ms/step CPU | README / runtime |
| eff-rank | healthy 9.86 / 192 → collapsed 1.0 | cartographer plan |

## A note for editors

The slides present the economy as the real product: a buyer pays, contributors are paid, the
ledger balances. The deck does not narrate methodology, caveats, or the simulated nature of
the payment rails on the slides; those precisions are delivered verbally during the pitch.
Keep that split. If a number changes, update both the slide and its evidence file, and keep
the worst seed beside the mean. There are no em-dashes anywhere in the deck; keep it that way.
