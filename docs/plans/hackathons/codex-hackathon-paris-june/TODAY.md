# Hackathon Start Brief - 2026-06-18

This is the start page for agents and humans during the Codex Hackathon in
Paris. The repo is in `#359` ship mode.

## Objective

Ship the **sovereign robotics world-model economy**: a federated
adapter-continuation round and surprise meter, plus a simulated buyer purchase
through Mollie test checkout and a contribution-weighted participant reward
ledger. If time collapses, deterministic ledger + mock checkout + surprise
fallbacks is still a coherent demo.

## Priority Queue

| Order | Issue | Build target | Done means |
|---|---|---|---|
| 0 | [#364](https://github.com/AbdelStark/Lensemble/issues/364) | Narrative/deck arc | Sovereignty -> federated run -> surprise proof -> buyer checkout -> reward split. |
| 1 | [#361](https://github.com/AbdelStark/Lensemble/issues/361) | Contribution ledger | Deterministic reward split balances to the simulated sale amount. |
| 2 | [#360](https://github.com/AbdelStark/Lensemble/issues/360) | Mollie test checkout | Server-side test checkout/payment link, mock fallback, no browser secrets. |
| 3 | [#338](https://github.com/AbdelStark/Lensemble/issues/338) | Surprise-meter proof | Existing [#349](https://github.com/AbdelStark/Lensemble/issues/349)-[#354](https://github.com/AbdelStark/Lensemble/issues/354) track shows model-quality improvement. |
| 4 | [#363](https://github.com/AbdelStark/Lensemble/issues/363) | Economics dashboard | Buyer, orchestrator share, community pool, participant rewards, and surprise proof fit together. |
| 5 | [#362](https://github.com/AbdelStark/Lensemble/issues/362) | Integrated rehearsal | Live/mock/recorded/capture rungs are validated. |

Critical path: **#364 -> #361 -> #360 -> #363 -> #362**. Run `#338` /
`#349`-`#354` in parallel as the model-quality proof.

## Hard Constraints

- No Cartographer `#339` until `#359` is rehearsal-green.
- No Latent Genie `#337`; the no-decoder spike blocks it.
- No "federated world-model training" language. Say **federated adapter
  continuation on a frozen checkpoint**.
- No per-patch heatmap claim. Surprise is a scalar CLS-latent prediction error.
- No secure-aggregation, DP, paper-scale, or closed-loop robot claim on this
  path.
- Always show the worst seed beside the mean: +12.3% this run, +16.8% mean,
  +5.4% worst seed.
- Use placeholders: "frontier model access restrictions" and "a humanoid
  robotics buyer." Do not name specific companies or incidents without separate
  verification and approval.
- Mollie keys stay server-side in ignored `.env` or process env; never in
  browser JS or committed docs.
- Rewards are simulation-only, not legal payouts or investment claims.

## Start Commands

```bash
gh issue view 359 --comments
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

1. Live federated round + surprise-meter + Mollie test checkout + reward split.
2. Live surprise-meter + deterministic ledger + mock checkout.
3. Recorded `surprise_trajectory.json` + deterministic ledger + mock checkout.
4. <=20 s capture clip and result/economics card.

Do not walk into Demo Night without rungs 2-4 verified on disk.
