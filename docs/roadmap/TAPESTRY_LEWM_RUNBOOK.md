# Tapestry-like LeWM TwoRooms — Reproduction and Operations Runbook

Issue [#324](https://github.com/AbdelStark/Lensemble/issues/324) and parent epic
[#314](https://github.com/AbdelStark/Lensemble/issues/314) are closed. This
runbook reproduces and checks the implemented frozen-checkpoint adapter path
described in [TAPESTRY_LEWM.md](TAPESTRY_LEWM.md). It is not a presentation
script or delivery plan.

## Prerequisites

Build the hash-bound browser artifacts from the pinned checkpoint:

```bash
uv run python scripts/lewm_tworooms_ingest.py
uv run --with onnx --with onnxscript --with onnxruntime \
  python scripts/lewm_tworooms_export.py
```

The real-data gates require the expert `tworoom.h5` dataset. Pass its local path
as `<tworoom.h5>` below. Node.js is required for the same JavaScript
adapter-training implementation used by the browser path.

## Binding evidence gates

First verify checkpoint-backed pairs and the bounded adapter's local training
behavior:

```bash
uv run --with onnxruntime --with hdf5plugin \
  python scripts/lewm_tworooms_realdata_check.py --h5 <tworoom.h5>
uv run --with onnxruntime --with hdf5plugin \
  python scripts/lewm_adapter_overfit_check.py --h5 <tworoom.h5>
```

Then produce the headline through the system that ships:

```bash
uv run --with onnxruntime --with hdf5plugin \
  python scripts/lewm_system_probe.py --h5 <tworoom.h5>
uv run --with onnxruntime --with hdf5plugin \
  python scripts/lewm_probe_seedsweep.py \
  --h5 <tworoom.h5> \
  --seeds 20260612 1 2 3 4
```

`scripts/lewm_system_probe.py` composes the real path:

1. the Node/browser implementation trains real adapter deltas;
2. `FederatedDemoService.submit_update` validates them;
3. `_close_round_lewm` aggregates them into hash-chained revisions;
4. the held-out probe scores the server-produced final revision.

`scripts/lewm_probe_seedsweep.py` repeats that path across five independent
seeds and episode splits. A non-improving or collapse-risk draw fails the
robustness gate. The checked-in distribution records five improving draws, mean
relative improvement `16.8%`, and worst-case improvement `+5.4%`.

The binding outputs are:

- [lewm_tworooms_system_probe.json](../evidence/lewm_tworooms_system_probe.json);
- [lewm_tworooms_probe_seedsweep.json](../evidence/lewm_tworooms_probe_seedsweep.json).

The offline command below is optional and non-binding:

```bash
uv run --with onnxruntime --with hdf5plugin \
  python scripts/lewm_probe_check.py --h5 <tworoom.h5>
```

It reimplements the coordinator mean in JavaScript and writes
`lewm_tworooms_probe_check.json`. It checks the update math but does not execute
the shipped Python server validation and aggregation path. Do not use it as the
headline or as a substitute for the system-composed probe and five-seed sweep.

## Orchestration smoke

```bash
uv run python scripts/lewm_demo_rehearsal.py
uv run python scripts/lewm_demo_rehearsal.py --long-rounds 50
```

This rehearsal covers run lifecycle, dropout/reconnect, stale-revision
rejection, revision retrieval, evidence export, and claim-audit plumbing. Its
adapter deltas are deterministic simulated fixtures. It is an orchestration
test, not evidence that the browser adapter learns.

## Standing tests

```bash
uv run pytest \
  tests/ml/test_lewm_tworooms.py \
  tests/ml/test_lewm_export.py \
  tests/ml/test_lewm_tworooms_browser.py \
  tests/ml/test_lewm_adapter.py \
  tests/ml/test_lewm_federation.py \
  tests/ml/test_lewm_demo_integration.py \
  tests/ml/test_lewm_probe.py \
  tests/ml/test_lewm_system_probe.py \
  tests/ml/test_lewm_evidence_audit.py \
  tests/ml/test_federated_demo_app.py
node web/federated-demo/lewm_probe_selftest.mjs
uv run python scripts/check_docs_links.py docs SPEC.md README.md
uv run python -m mkdocs build --strict
git diff --check
```

## Local runtime check

```bash
uv run lensemble demo federated --port 8765
```

Startup must report the real-LeWM mode as available and identify the pinned
checkpoint revision. Create a `real-lewm-tworooms` run, join at least two
participants, and verify:

- each participant loads the hash-checked inference graphs;
- browser-side progress reflects real residual-adapter optimizer steps;
- submitted artifacts are `lewm-adapter-delta/1` and bind the active parent
  revision;
- the coordinator publishes `lewmrev-*` revisions;
- exported evidence carries the run mode, checkpoint/export binding, source
  delta hashes, health flags, and non-claims.

Failure is explicit. Missing runtime support or artifacts do not trigger a
surrogate fallback. Stale, malformed, oversized, wrong-parent, or raw-like
updates are rejected.

## Interpretation boundary

The result is federated continuation of a 12,512-parameter (0.069%) residual
adapter on a frozen checkpoint. The browser does not train the LeWorldModel
encoder, action encoder, predictor, or projection heads. One trusted local
coordinator averages clipped deltas; secure aggregation and differential
privacy are not wired into this path.

The system-composed seed-`20260612` probe records held-out MSE improvement from
`0.06037897796856822` to `0.05296723026100999` (`+12.275%`) without a collapse
flag. The five-seed result is the robustness boundary, not a general benchmark:
all five draws improved, with a `+5.4%` worst case and no collapse-risk draw.

Do not infer full-model federated-training quality, paper-scale TwoRooms/PushT
performance, physical robotics utility, production readiness, decentralized
multi-operator execution, cryptographic honest computation, secure
aggregation, or DP from these artifacts.

## Failure triage

| Symptom | Action |
|---|---|
| `real_lewm_mode=unavailable` | Re-run checkpoint ingestion and export. Real-mode run creation fails closed until the artifacts validate. |
| Participant reports the real-LeWM runtime is unavailable | Check graph fetches and WASM/WebGPU support. Use a supported runtime; do not switch the run to the surrogate mode. |
| Participant submits the wrong schema or stale revision | Hard-reload stale client code, rejoin the same run, and preserve the server rejection in logs/evidence. |
| System probe reports `flat`, `worse`, or collapse risk | Retain the negative artifact and block positive adaptation claims. |
| Any seed in the sweep is non-improving or collapse-risk | Treat the robustness gate as failed; do not headline a favorable draw. |
| Quorum is lost | Start a fresh run after diagnosing participant loss; the local service has no persistent recovery state. |
