# Hackathon Start Brief - 2026-06-18

This is the start page for agents and humans during the Codex Hackathon in
Paris. The repo is in `#338` ship mode.

## Objective

Ship **Less Surprised**: a clean federated adapter-continuation round on a
frozen LeWorldModel checkpoint, then a live scalar surprise meter showing the
held-out prediction-error drop. If time collapses, Milestone 0 plus fallbacks is
the demo.

## Priority Queue

| Order | Issue | Build target | Done means |
|---|---|---|---|
| 0 | [#349](https://github.com/AbdelStark/Lensemble/issues/349) | Clean federated adapter-continuation round | One command regenerates audited evidence and exports the 12,512-float adapter offset. |
| 1 | [#350](https://github.com/AbdelStark/Lensemble/issues/350) | Per-step scalar surprise engine | Browser engine matches the existing probe math. |
| 2 | [#351](https://github.com/AbdelStark/Lensemble/issues/351) | Surprise UI | Meter/oscilloscope, perturbation spike, and frame-diff baseline render live. |
| 3 | [#352](https://github.com/AbdelStark/Lensemble/issues/352) | Pre/post toggle | Held-out probe-pair toggle shows the sourced +12.3% drop and the +5.4% worst seed. |
| 4 | [#353](https://github.com/AbdelStark/Lensemble/issues/353) | `lewm-surprise/1` evidence | Producer and test pass; no unsourced headline numbers. |
| 5 | [#354](https://github.com/AbdelStark/Lensemble/issues/354) | Rehearsal, fallback, capture | Green rehearsal, fallback trajectory/offset, <=20 s clip, and result card. |

Critical path: **#349 -> #350 -> #351 -> #354**. Run `#352` beside `#351`
when the engine is stable. Run `#353` as soon as the displayed numbers settle.

## Hard Constraints

- No Cartographer `#339` until `#338` is rehearsal-green.
- No Latent Genie `#337`; the no-decoder spike blocks it.
- No "federated world-model training" language. Say **federated adapter
  continuation on a frozen checkpoint**.
- No per-patch heatmap claim. Surprise is a scalar CLS-latent prediction error.
- No secure-aggregation, DP, paper-scale, or closed-loop robot claim on this
  path.
- Always show the worst seed beside the mean: +12.3% this run, +16.8% mean,
  +5.4% worst seed.

## Start Commands

```bash
gh issue view 338 --comments
gh issue view 349 --comments
uv run lensemble demo federated --port 8765
```

Use the narrowest gate for the slice being built, then broaden before a claim
surface changes:

```bash
uv run pytest tests/ml/test_lewm_probe.py tests/ml/test_lewm_system_probe.py tests/ml/test_lewm_evidence_audit.py
node web/federated-demo/lewm_probe_selftest.mjs
uv run python scripts/check_docs_links.py docs SPEC.md README.md
uv run python -m mkdocs build --strict
git diff --check
```

## Fallback Ladder

1. Live federated round plus surprise-meter, using the certified evidence and
   pre-baked offset for the headline number.
2. Live surprise-meter with pre-baked offset.
3. Recorded `surprise_trajectory.json` replay.
4. <=20 s capture clip and result card.

Do not walk into Demo Night without rungs 2-4 verified on disk.
