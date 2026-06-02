# Tests

The test pyramid for `lensemble` (see `docs/spec/07-testing-strategy.md` §1). Shared fixtures, the
hypothesis profile, and the named numerical tolerances live in `tests/conftest.py`.

| Layer | Directory | Owns | Runtime budget |
|---|---|---|---|
| Unit | `tests/unit/` | pure-Python logic: errors, config, data-model shapes, packaging, seeding | milliseconds; no torch heavy paths |
| Property | `tests/property/` | `hypothesis` invariants (residency, config) | seconds |
| ML-specific | `tests/ml/` | numerical contracts on the CPU path: WMCP, warm-start, SIGReg, probe, gauge, checkpoint hashing | seconds; tiny tensors only |
| Integration | `tests/integration/` | multi-module wiring (coordinator/participant round, DP accountant, eval harness) | seconds–minutes; in-process |
| Regression | `tests/regression/` | pinned-behavior guards against known regressions | seconds |
| E2E | `tests/e2e/` | the ablation-ladder / train→eval slice on a toy env | minutes; opt-in |

Discipline (`07` §1, §7):

- Fixtures take a **seeded generator** (`rng`) and never call a global RNG, so the property layer
  replays deterministically.
- Numerical tolerances are **named constants** (`conftest.Tolerances` / the `tol` fixture) cited by id —
  never inlined magic numbers (`07` §6).
- Fixtures are tiny, CPU-only, and **download nothing**: `tiny_warmstart` (a 2-layer linear encoder,
  `d=8`), `synthetic_probe` (deterministic points with `k ≥ d` landmarks), `toy_env` (a closed-form
  in-memory env). No real V-JEPA 2 warm-start, dataset, or probe is fetched.
- Unit tests touch the filesystem only under `tmp_path`.
