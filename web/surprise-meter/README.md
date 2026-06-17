# surprise-meter — Milestone-1 scaffold (mocked)

The live **surprise-meter** for the Codex-Paris demo (issue
[#338](https://github.com/AbdelStark/Lensemble/issues/338), Milestone 1). This is a
**scaffold**: the full UI, layout, flow, and the recorder are real; the data behind them is
**mocked** so the page runs with zero ONNX/network. On the day, the mocks are swapped for the
real `web/federated-demo/` modules and nothing else moves.

Design language is **"The Offprint"** — the same warm-bone-paper / oxblood / Fraunces+Newsreader+
Spline-Sans-Mono identity as the Demo-Night deck, so the live tool and the slides read as one
piece. The surprise signal is a registered **ink recorder** (a lab-bench chart on paper), not a
glowing widget.

## Run

```bash
# from the repo root — needs http:// (ES modules), not file://
python3 -m http.server 8000 --directory web/surprise-meter
# → http://localhost:8000
```

Buttons: **Teleport / Off-distribution action / Push through a wall** inject a surprise event;
the **Pre / Post-federation** toggle drops the in-distribution baseline to the certified level;
**❚❚** pauses. Watch the frame-diff (Fig. B) run high on motion while surprise (Fig. A) stays
flat — and "Off-distribution action" spike surprise *without* motion. That is the whole point.

## What's mocked, and the swap

| File | Mocks | Swap for real (the day) |
|---|---|---|
| `mock/lewm_mock.mjs` | `mockRuntime` / `mockEnv` / `mockProbe` / `mockAdapter` | `../federated-demo/lewm_runtime.mjs` (`encodeFrames`/`predictLatents`), `tworooms_env.mjs` (`renderFrameRGB`/`stepAgent`), `lewm_probe.mjs` (MSE/192), `lewm_adapter.mjs` (`adapterFromInitAndOffset({…initSeed:42})`) |
| `mock/engine.mjs` | composes the pipeline: predict → actual → surprise | change its 3 imports to `../../federated-demo/*`; feed real frames via `frameToModelInput(renderFrameRGB(pos))` |
| `mock/fixtures.mjs` | certified numbers + adapter dims, inline | `fetch()` the evidence JSONs + the committed `fixtures/adapter_offset.json` (len 12,512) |

The mock keeps the **real pipeline shape** end-to-end (predict the next latent → encode the
actual next latent → `surprise = MSE/192`); only latent *generation* is simulated, so the swap
touches imports, not flow. Signatures follow the verified corrections C1–C6 in
[`docs/plans/.../surprise-meter/01-architecture.md`](../../docs/plans/hackathons/codex-hackathon-paris-june/surprise-meter/01-architecture.md).

`seismograph.mjs` (the recorder) and `app.mjs` (UI + world view + controls) are **final-shape** —
they do not change on the swap.

## Numbers & claim discipline

Every on-screen figure mirrors the certified evidence at full precision:
**+12.3% this run · +16.8% mean · +5.4% worst (seed 2)** of held-out next-step prediction error,
from `lewm_tworooms_system_probe.json` / `…_probe_seedsweep.json`. The pre/post baseline is the
certified `baselineMse 0.0604 → adaptedMse 0.0530` (= +12.3%); on the day the toggle runs on the
**held-out probe-pair set** (seed 991, C11), not a free-running trajectory.

The footer renders the binding non-claims from data (it can't silently drop off-screen): it is
*federated adapter continuation on a frozen checkpoint*, surprise is a *scalar* (not a heatmap),
and there is no DP / secure-aggregation / paper-scale / beats-local-only claim.

## Offline (venue)

Fonts load from the Fontsource CDN. Before the venue, vendor them the same way as the deck
(`docs/plans/.../presentation/vendor.sh`) and point the `@font-face` `src` in `styles.css` at the
local copies — then the page has no network dependency.
