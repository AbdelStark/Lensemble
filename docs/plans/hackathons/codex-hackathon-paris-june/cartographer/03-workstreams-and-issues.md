# Cartographer — Workstreams & Child Issues

Nine child issues under parent [#339](https://github.com/AbdelStark/Lensemble/issues/339). Each is independently shippable and has explicit acceptance criteria. The GitHub issue numbers are filled in once created (see the table at the bottom and `#339`'s task list).

## Dependency graph

```
 CART-1 harvest ───────────────┐
 CART-2 instrumented MPC ───────┼──────────────┐
 CART-3 federated before/after ─┤              │
                                ▼              ▼
                       CART-4 projection ── CART-6 evidence
                                │              │
                                ▼              │
                       CART-5 collapse         │
                                │              │
                                ▼              │
                       CART-7 bake ◀───────────┘
                                │
                CART-8 viewer ──┤  (parallel vs contract doc 02)
                                ▼
                       CART-9 rehearsal + capture
```

**Critical path:** CART-1 → CART-4 → CART-7 → CART-9.
**Parallelizable from minute 0:** CART-1, CART-2, CART-3 (independent); CART-8 (against the frozen doc-`02` contract, using a mock `manifold.json`).

## Ordering for a solo builder (and the pre-bake)

1. **Wed evening:** vendor Three.js; build CART-8 skeleton against a hand-written mock `manifold.json`; bake a **v0 fallback** from the current checkpoint (even latents-only, no plans) so a viewer demo exists before Thursday.
2. **Thu AM:** CART-1, then CART-2 (the risky one) and CART-3 in parallel slots.
3. **Thu midday:** CART-4 → CART-5 → CART-7 (first real `manifold.json`).
4. **Thu PM:** wire real data into CART-8; CART-6 evidence + test; CART-9 rehearsal + capture.

## Per-issue summary

| Issue | Title | Primary new files | Key reuse | Accept |
|---|---|---|---|---|
| **CART-1** | Manifold latent harvest pipeline | `lensemble/eval/manifold_harvest.py`, `scripts/cartographer/harvest.py` | `build_probe_split`, `encode_frames` | deterministic `(N,192)` latents+actions+state; eff_rank ≈ 9–10 |
| **CART-2** | Instrumented MPC planner + LeWM dynamics | `lensemble/eval/mpc_instrumented.py` | `Planner`, `_rollout_costs`, `predict` | captures candidates/iter; reaches goal; unit test on toy dynamics |
| **CART-3** | Federated before/after producer | `lensemble/eval/manifold_federation.py`, `scripts/cartographer/run_round.py` | `run_system_composed_probe`, `lewm_adapter.mjs` | offset len 12512; apply parity ≤1e-3; post MSE < pre |
| **CART-4** | Projection + gauge alignment + metrics | `lensemble/eval/manifold_projection.py` | `procrustes_align`, `effective_rank/dim`, `sigreg_statistic` | shared-basis PCA; residual reported; metrics in certified range |
| **CART-5** | Collapse counterfactual (synthetic) | `manifold_projection.py` (or `manifold_collapse.py`) | spike/test collapse patterns | eff_dim(rank1)≈1; labelled synthetic |
| **CART-6** | Evidence JSON + producer + test | `scripts/lewm_manifold_check.py`, `tests/ml/test_lewm_manifold.py` | `write_evidence`, system-probe sourcing | `lewm-manifold/1` passes predicate; nonClaims present; test green |
| **CART-7** | Bake orchestrator → `manifold.json` | `scripts/cartographer/bake.py`, `lensemble/eval/manifold_bake.py` | all of CART-1..5 | byte-stable per seed; ≤5 MB; validates doc-02 |
| **CART-8** | WebGPU/Three.js viewer page | `web/latent-manifold-viewer/{index.html,app.mjs,vendor/three.module.min.js}` | `dynamic-env-demo` template, `_static` serving | renders contract; toggles + HUD + provenance + nonClaims |
| **CART-9** | Rehearsal + fallback + capture | `scripts/cartographer/rehearsal.py`, committed fallback `manifold.json` | `hackathon_demo_rehearsal` style | gate green; ≤20s clip + result card |

## Test & gate matrix

| Issue | Lint/Type | Python test | JS/selftest | Docs gate |
|---|---|---|---|---|
| CART-1 | ruff+pyright | `tests/unit` shape/determinism | — | — |
| CART-2 | ruff+pyright | `tests/unit` toy-dynamics branching | — | — |
| CART-3 | ruff+pyright | `tests/ml` parity vs JS probe op | reuse `lewm_probe_selftest.mjs` | — |
| CART-4 | ruff+pyright | `tests/unit` PCA/procrustes/metrics | — | — |
| CART-5 | ruff+pyright | `tests/unit` collapse metrics | — | — |
| CART-6 | ruff+pyright | `tests/ml/test_lewm_manifold.py` | — | `check_docs_links`, `mkdocs --strict` |
| CART-7 | ruff+pyright | `tests/ml` contract validation | — | — |
| CART-8 | — | — | a small `selftest.mjs` (schema + scene smoke) | — |
| CART-9 | ruff+pyright | `tests/ml` rehearsal gate | — | — |

## GitHub issue numbers

Filled after creation:

| Workstream | Issue |
|---|---|
| CART-1 | [#340](https://github.com/AbdelStark/Lensemble/issues/340) |
| CART-2 | [#341](https://github.com/AbdelStark/Lensemble/issues/341) |
| CART-3 | [#342](https://github.com/AbdelStark/Lensemble/issues/342) |
| CART-4 | [#343](https://github.com/AbdelStark/Lensemble/issues/343) |
| CART-5 | [#344](https://github.com/AbdelStark/Lensemble/issues/344) |
| CART-6 | [#345](https://github.com/AbdelStark/Lensemble/issues/345) |
| CART-7 | [#346](https://github.com/AbdelStark/Lensemble/issues/346) |
| CART-8 | [#347](https://github.com/AbdelStark/Lensemble/issues/347) |
| CART-9 | [#348](https://github.com/AbdelStark/Lensemble/issues/348) |
