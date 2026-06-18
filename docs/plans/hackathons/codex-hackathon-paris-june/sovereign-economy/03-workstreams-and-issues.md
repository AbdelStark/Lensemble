# Sovereign Economy - Workstreams And Issues

## Dependency Graph

```text
#364 narrative/deck arc
   |
   v
#361 ledger contract -----> #360 Mollie test checkout
   |                            |
   +------------+---------------+
                v
#338 surprise-meter track ---> #363 economics dashboard
                |               |
                +---------------+
                                v
                         #362 rehearsal gate
```

Surprise-meter implementation remains decomposed under [#338](https://github.com/AbdelStark/Lensemble/issues/338):
[#349](https://github.com/AbdelStark/Lensemble/issues/349) -> [#350](https://github.com/AbdelStark/Lensemble/issues/350)
-> [#351](https://github.com/AbdelStark/Lensemble/issues/351) -> [#354](https://github.com/AbdelStark/Lensemble/issues/354),
with [#352](https://github.com/AbdelStark/Lensemble/issues/352) and [#353](https://github.com/AbdelStark/Lensemble/issues/353)
parallel once the engine is stable.

## Issue Table

| Issue | Workstream | Depends on | Gate |
|---|---|---|---|
| [#364](https://github.com/AbdelStark/Lensemble/issues/364) | Narrative, deck arc, claim-safe copy | none | docs copy review + link/docs build |
| [#361](https://github.com/AbdelStark/Lensemble/issues/361) | Contribution ledger contract | #364 optional | unit tests for deterministic split |
| [#360](https://github.com/AbdelStark/Lensemble/issues/360) | Mollie test checkout/payment links | #361 | mock test + optional credentialed smoke |
| [#338](https://github.com/AbdelStark/Lensemble/issues/338) | Surprise-meter technical track | existing #349-#354 | JS/Python surprise gates |
| [#363](https://github.com/AbdelStark/Lensemble/issues/363) | Economics dashboard | #361, #360, #338 | browser smoke + layout checks |
| [#362](https://github.com/AbdelStark/Lensemble/issues/362) | Rehearsal, fallbacks, validation | all above | one-command rehearsal ladder |

## Autonomous Validation Matrix

| Surface | Required gate |
|---|---|
| Docs and issue links | `uv run python scripts/check_docs_links.py docs SPEC.md README.md` |
| Docs build | `uv run python -m mkdocs build --strict` |
| Diff hygiene | `git diff --check` |
| Surprise core | `node web/federated-demo/lewm_probe_selftest.mjs` |
| LeWM evidence | `uv run pytest tests/ml/test_lewm_probe.py tests/ml/test_lewm_system_probe.py tests/ml/test_lewm_evidence_audit.py` |
| Ledger | `uv run pytest tests/ml/test_demo_economy.py` |
| Mollie mock/env | `uv run pytest tests/ml/test_demo_economy.py` |
| Credentialed Mollie smoke | opt-in, skipped unless `MOLLIE_API_KEY` is present |
| Browser dashboard | Playwright/in-browser smoke if available; otherwise static HTML/module inspection plus JS selftest |
| Secret hygiene | tracked-file scan for Mollie key assignments and committed live/test key values outside `.env.example` placeholders |

## Build Order

1. Update narrative and deck outline (#364).
2. Implement pure ledger (#361). This can ship without Mollie.
3. Add Mollie adapter with mock default (#360).
4. Continue surprise-meter work (#338 / #349-#354).
5. Join proof and economics in the UI (#363).
6. Run integrated rehearsal and capture (#362).

If time collapses, ship #364 + #361 + #338 with a mock checkout card. That still
shows the coordination and sustainable-economics premise without relying on
external payment availability.
