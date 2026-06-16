# Surprise-meter — Workstreams & Child Issues

Six child issues under parent [#338](https://github.com/AbdelStark/Lensemble/issues/338). SM-1 is **Milestone 0** (the priority, standalone-demoable); SM-2..SM-6 are **Milestone 1** (the surprise-meter).

## Dependency graph

```
 SM-1 clean federated run (Milestone 0) ─────────────┐
        │ (ONNX/runtime parity, adapter offset)        │
        ▼                                              ▼
 SM-2 surprise engine ──▶ SM-3 surprise UI       SM-4 pre/post toggle
        │                      │                       │
        └──────────────┬───────┴───────────┬──────────┘
                       ▼                    ▼
                 SM-5 evidence + test   SM-6 rehearsal + fallback + capture
```

Critical path: **SM-1 → SM-2 → SM-3 → SM-6**. SM-4 parallels SM-3; SM-5 needs SM-2+SM-4 numbers.

## Per-issue summary

| Issue | Milestone | Title | Primary new files | Key reuse | Accept |
|---|---|---|---|---|---|
| **SM-1** | **0** | Clean federated adapter-continuation run | `scripts/surprise/run_clean_round.py`, `scripts/surprise/rehearsal.py`, runbook update | `system_probe`, `audit_real_lewm_evidence`, `web/federated-demo/` | one-command run → audited evidence; rehearsal green; offset exported |
| **SM-2** | 1 | In-browser per-step surprise engine | `web/surprise-meter/surprise_engine.mjs`, `surprise_selftest.mjs` | `lewm_runtime.mjs`, `lewm_probe.mjs`, `tworooms_env.mjs` | per-step surprise; parity ≤1e-4 vs probe; self-test green |
| **SM-3** | 1 | Surprise UI + perturbation + frame-diff | `web/surprise-meter/{index.html,app.mjs}` | `dynamic-env-demo` template, `tworooms_panel.mjs`, `charts.mjs` | live meter+env; perturbations spike; frame-diff baseline |
| **SM-4** | 1 | Pre/post-federation toggle | (extend `surprise_engine.mjs`/`app.mjs`) | `lewm_adapter.mjs` | `mean(post) < mean(pre)`; certified number in HUD |
| **SM-5** | 1 | Evidence JSON `lewm-surprise/1` + test | `scripts/lewm_surprise_check.py`, `tests/ml/test_lewm_surprise.py` | `write_evidence`, system-probe sourcing | passes predicate; nonClaims; test green |
| **SM-6** | 1 | Rehearsal + fallback + capture | `scripts/surprise/rehearsal.py`, committed `adapter_offset.json` + `surprise_trajectory.json` | `hackathon_demo_rehearsal` style | gate green; fallback present; clip + card |

## Test & gate matrix

| Issue | Lint/Type | Python test | JS/selftest | Docs gate |
|---|---|---|---|---|
| SM-1 | ruff+pyright | `tests/ml` round + audit | reuse `lewm_probe_selftest.mjs` | runbook link check |
| SM-2 | — | — | `web/surprise-meter/surprise_selftest.mjs` | — |
| SM-3 | — | — | scene smoke in selftest | — |
| SM-4 | — | — | parity assertion in selftest | — |
| SM-5 | ruff+pyright | `tests/ml/test_lewm_surprise.py` | — | `check_docs_links`, `mkdocs --strict` |
| SM-6 | ruff+pyright | `tests/ml` rehearsal gate | — | — |

## Ordering for a solo builder
1. **Wed eve:** SM-1 to green (one-command + rehearsal); pre-bake fallback offset; scaffold `web/surprise-meter/` page from `dynamic-env-demo`.
2. **Thu AM:** SM-2 (engine + parity), then SM-3 (UI + perturbation + frame-diff).
3. **Thu midday:** SM-4 (pre/post toggle) on the real revision.
4. **Thu PM:** SM-5 evidence + test; SM-6 rehearsal + capture.

## GitHub issue numbers
Filled after creation:

| Workstream | Issue |
|---|---|
| SM-1 | [#349](https://github.com/AbdelStark/Lensemble/issues/349) (Milestone 0) |
| SM-2 | [#350](https://github.com/AbdelStark/Lensemble/issues/350) |
| SM-3 | [#351](https://github.com/AbdelStark/Lensemble/issues/351) |
| SM-4 | [#352](https://github.com/AbdelStark/Lensemble/issues/352) |
| SM-5 | [#353](https://github.com/AbdelStark/Lensemble/issues/353) |
| SM-6 | [#354](https://github.com/AbdelStark/Lensemble/issues/354) |
