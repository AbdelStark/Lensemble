# Browser Federation Reference

Trackers [#294](https://github.com/AbdelStark/Lensemble/issues/294) and
[#314](https://github.com/AbdelStark/Lensemble/issues/314) are closed. All
delivery issues for the real-checkpoint path
([#315](https://github.com/AbdelStark/Lensemble/issues/315) through
[#324](https://github.com/AbdelStark/Lensemble/issues/324)) are also closed.
This page is the maintained reference for the implemented browser boundary.

## Scope

The browser application exposes two separate research-demo modes:

| Mode | Implemented behavior |
|---|---|
| `surrogate-swipe-dot` | A tiny learner trains on synthetic swipe-dot trajectories and submits a clipped `browser-update/1` vector. This is an orchestration fixture, not a world-model result. |
| `real-lewm-tworooms` | Hash-checked ONNX graphs run a pinned LeWorldModel TwoRooms checkpoint. The checkpoint stays frozen while a 12,512-parameter (0.069%) residual adapter on the predictor output trains with real browser-side gradients and submits a clipped `lewm-adapter-delta/1`. |

The second mode is **federated adapter continuation on a frozen checkpoint**.
It is not federated training of the world model. There is no browser training
of the encoder, action encoder, predictor, or projection heads, and the result
must not be shortened to "browser world-model training" or "full-model
training."

Both modes run through one trusted local coordinator. The real-LeWM path takes
the deterministic mean of clipped adapter deltas. It does **not** wire secure
aggregation or differential privacy, and it is not a multi-operator
decentralized run. Cross-operator execution remains deferred in
[#331](https://github.com/AbdelStark/Lensemble/issues/331).

## Local entry point

```bash
uv run lensemble demo federated --port 8765
```

Open the host URL printed at startup, normally:

```text
http://127.0.0.1:8765/web/federated-demo/
```

Run mode is a host-selected, run-level property carried by snapshots, events,
model revisions, and exported evidence. A real-LeWM run fails closed if its
pinned checkpoint export is unavailable; it never falls back silently to the
surrogate learner.

## Runtime contract

Browser clients use coordinator-owned HTTP and WebSocket endpoints. Participant
WebSocket authentication uses a protocol header rather than a URL parameter.
REST and WebSocket commands share the same token, round, revision, shape,
size, norm, duplicate, and stale-update validation.

The runtime retains:

- run creation, admission, quorum, heartbeat, reconnect, timeout, and explicit
  dropout handling;
- monotonic event replay over WebSocket with REST/NDJSON fallback;
- mode-specific update validation and deterministic revision publication;
- model-revision retrieval and residency-safe evidence export.

The coordinator does not expose NATS or Kafka directly to browsers. The shipped
demo is an in-process research service, not a production multi-tenant control
plane.

## Update contracts

`browser-update/1` is the surrogate-only vector contract. It carries a bounded
one-dimensional derived vector, sample/work metadata, metrics, and a content
hash.

`lewm-adapter-delta/1` is the real-LeWM contract. It binds the delta to:

- run, round, participant, and parent model revision;
- pinned checkpoint and browser-export hashes;
- an exact adapter parameter inventory and shape;
- byte, parameter-count, and L2 clip bounds;
- derived training and anti-collapse metrics.

Raw observations, frames, actions, labels, latent batches, participant tokens,
and base checkpoint weights are not accepted as update fields. This validation
limits the artifact contract; it is not a cryptographic proof of honest local
computation or of data residency.

## Binding browser evidence

The claim gate is the system-composed path:

1. real adapter deltas are trained by the Node/browser implementation;
2. `FederatedDemoService.submit_update` performs the shipped validation;
3. `_close_round_lewm` creates the shipped deterministic-mean, hash-chained
   global revision;
4. the held-out probe scores that **server-produced** revision.

The committed
[system-composed probe](../evidence/lewm_tworooms_system_probe.json) records
held-out MSE `0.06037897796856822 -> 0.05296723026100999`
(`+12.275%`) for the seed-`20260612` run, with no collapse flag. The committed
[five-seed sweep](../evidence/lewm_tworooms_probe_seedsweep.json) is the
robustness gate: all five draws improve, the mean relative improvement is
`16.8%`, and the worst draw is `+5.4%` (seed 2), with no collapse-risk draw.

These two artifacts are the binding browser evidence. The
[offline probe](../evidence/lewm_tworooms_probe_check.json), produced by
`scripts/lewm_probe_check.py`, reimplements the mean in JavaScript and is only a
math cross-check. It does not exercise the shipped Python validation and
aggregation path and must not be used as the headline result.

The [demo card](../evidence/lewm_tworooms_demo_card.md) carries the full
artifact inventory and non-claims. The evidence supports only a narrow,
demo-scale result for federated adapter continuation on fixed TwoRooms
validation pairs. It does not establish paper-scale performance, robotics
utility, or an advantage for federated full-model training.

## Validation

```bash
uv run pytest tests/ml/test_federated_demo_app.py
uv run pytest \
  tests/ml/test_lewm_probe.py \
  tests/ml/test_lewm_system_probe.py \
  tests/ml/test_lewm_evidence_audit.py
node web/federated-demo/lewm_probe_selftest.mjs
uv run python scripts/check_docs_links.py docs SPEC.md README.md
uv run python -m mkdocs build --strict
git diff --check
```

## Known unsupported paths

- No secure aggregation or differential privacy in the real-LeWM demo path.
- No multi-process or cross-operator federation; one local coordinator is
  trusted.
- No full-model or from-scratch LeWorldModel training in a browser.
- No production browser-training or multi-tenant deployment claim.
- No cryptographic proof of honest participant computation.
- No paper-scale TwoRooms or PushT performance claim.
- No dynamic-env claim that federation materially beats local-only.
- No closed-loop physical SO-100 success claim.
