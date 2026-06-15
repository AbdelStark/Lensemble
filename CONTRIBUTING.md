# Contributing to Lensemble

This is a reference implementation held to research-grade rigor: typed contracts, named invariants,
and a test suite that download nothing and runs on CPU. Before opening a pull request, reproduce the
CI gates locally — they are the same commands the pipeline runs.

## Development setup

```bash
uv venv .venv --python 3.11        # or python -m venv .venv
uv pip install "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cpu   # CPU wheel
uv pip install -e ".[dev]"
```

The dev extra pulls `ruff`, `pyright`, `pytest`, `hypothesis`, `pytest-benchmark`, and `coverage`
([conventions §11](docs/spec/conventions.md#11-external-dependencies)). `pyright` resolves imports
from `.venv` (configured in `pyproject.toml`), so create the venv at the repo root.

## CI status and the merge rule

[![ci](https://github.com/AbdelStark/Lensemble/actions/workflows/ci.yml/badge.svg)](https://github.com/AbdelStark/Lensemble/actions/workflows/ci.yml)

**Every gate below must be green before a pull request can merge.** The pipeline
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on every push and pull request, on a
CPU runner, ordered and fail-fast — it downloads no model, dataset, or probe and never requires a GPU
([07 — Testing Strategy §8](docs/spec/07-testing-strategy.md#8-ci-gates)).

| # | Gate | Command | Pass condition |
|---|---|---|---|
| 1 | Lint | `ruff check .` and `ruff format --check .` | zero findings |
| 2 | Type-check | `pyright` | zero errors on `lensemble/` and `tests/` |
| 3 | Unit + property | `pytest tests/unit tests/property` | all pass (fixed `hypothesis` profile) |
| 4 | Integration + ML + e2e + regression | `pytest tests/integration tests/ml tests/e2e tests/regression` | all pass |
| 5 | Coverage — overall | `coverage report --fail-under=85` | line coverage ≥ 85% |
| 6 | Coverage — security-critical | `coverage report --include=… --fail-under=100` | 100% on `data.residency`, `aggregation.secure_agg` (reveal path), `provenance.commit`, `privacy.dp` |

Reproduce the whole gate locally:

```bash
ruff check . && ruff format --check . && pyright
coverage run -m pytest tests/unit tests/property
coverage run --append -m pytest tests/integration tests/ml tests/e2e tests/regression
coverage report --fail-under=85
coverage report \
  --include='*/lensemble/data/residency.py,*/lensemble/aggregation/secure_agg.py,*/lensemble/provenance/commit.py,*/lensemble/privacy/dp.py' \
  --fail-under=100
```

The four security-critical modules are held to 100% because a single uncovered branch there is an
unverified residency, aggregation, provenance, or privacy path. When a stub in that set is not yet
implemented (e.g. `provenance.commit` before its issue lands), a contract test pins its explicit
failure so the gate stays load-bearing rather than silently satisfied.

Gates wired on top of this pipeline but tracked separately: bitwise-determinism (`INV-AGG-DETERMINISM`),
docs link-check, and the performance smoke. The bitwise-determinism gate ([07 §8](docs/spec/07-testing-strategy.md#8-ci-gates)
gate 5) runs in its own workflow ([`.github/workflows/determinism.yml`](.github/workflows/determinism.yml)):
it executes `tests/ml/test_aggregation_determinism.py` and is **build-blocking and fail-closed** — a
non-reproducible aggregation step is a Phase-1 proof-readiness failure ([RFC-0006 §3](docs/rfcs/RFC-0006-verifiable-contribution.md)),
not a flake, so it aborts the round (`NonDeterministicAggregation`) rather than averaging silently. The
docs link-check gate (gate 7, [`.github/workflows/docs.yml`](.github/workflows/docs.yml)) runs
`python scripts/check_docs_links.py docs/ SPEC.md`: every relative cross-reference `[label](relpath#anchor)`
must resolve to an existing file and heading. Anchors follow GitHub's heading-slug rule — lowercase, drop
every character that is not a letter, number, underscore, hyphen, or whitespace, then replace whitespace
with hyphens (duplicate headings get a `-1`/`-2` suffix). External `http(s)` links are out of scope. The
non-blocking nightly CUDA suite and the x86-64/arm64 cross-platform hash check do not block merges
([07 §8](docs/spec/07-testing-strategy.md#8-ci-gates)).

## Pull request expectations

- One concern per PR; do not bundle unrelated changes.
- A user-visible change (public surface, CLI, on-disk schema, or documented behavior) adds an entry to
  the `## [Unreleased]` block of [`CHANGELOG.md`](CHANGELOG.md) under one of the six permitted categories
  (`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`). Entries are imperative and name the
  affected symbol or `area:*` label and, where applicable, the RFC and `INV-*` id ([09 §3](docs/spec/09-release-and-versioning.md#3-changelog-discipline)).
- Tensors serialize via `safetensors`, never `pickle`/`torch.load` ([06 — Security §7](docs/spec/06-security.md)).
- Named numerical tolerances live in `tests/conftest.py` and are cited by id — never inline a magic
  number ([07 §6](docs/spec/07-testing-strategy.md)).
- New code keeps coverage above the thresholds above; security-critical paths are covered, not excluded.

## Releases

A release is automated by [`.github/workflows/release.yml`](.github/workflows/release.yml), triggered by an
annotated tag `vX.Y.Z`. It runs the eight release-blocking gates of
[09 §5.2](docs/spec/09-release-and-versioning.md#52-release-checklist-release-blocking-gates) **in order**
(`python scripts/release_gates.py plan` prints them) and **fails closed with no waiver** on any
security-critical gate (`INV-RESIDENCY`, `INV-COMMIT-BINDING`, `INV-AGG-DETERMINISM`). The
**version-agreement** gate (`python scripts/release_gates.py version-agreement`) asserts
`pyproject.toml` `[project].version` = `lensemble.__version__` = the newest `CHANGELOG.md` release version
(a non-empty `## [X.Y.Z] - DATE` block). The flow builds the sdist+wheel, runs a clean-venv smoke install
asserting the installed `__version__` equals the tag **before** any upload (PyPI publication is
irreversible per version), then publishes and cuts a GitHub release. The large-binary research-artifact
bundle ([09 §5.4](docs/spec/09-release-and-versioning.md#54-research-artifact-release)) is never committed
and CI never downloads it (`release-bundle/` is git-ignored).
