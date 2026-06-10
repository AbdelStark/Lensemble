# Browser Federated Demo Roadmap

Tracker: [#294](https://github.com/AbdelStark/Lensemble/issues/294)

The browser federated demo is an educational systems showcase for Lensemble's
Tapestry-like JEPA federation story: sovereign participants join a run, do
resident browser-local work, submit metadata-only update artifacts, and the host
can inspect the lifecycle, artifacts, inference panel, and evidence export.

It is not a benchmark win over local-only, not a production browser-training
claim, not a cryptographic honest-computation proof, and not a closed-loop
physical SO-100 success claim.

## One-Command Local Demo

```bash
uv run lensemble demo federated --port 8765
```

Open the printed URL, usually:

```text
http://127.0.0.1:8765/web/federated-demo/
```

The page can also run the first slice without the backend by selecting
`frontend-only simulator`; because it uses ES modules, serve the repository over
HTTP rather than opening `index.html` with `file://`.

## Host Flow

1. Select `backend API mode (local server)`.
2. Configure max participants, quorum, round count, and the `swipe-dot-tiny`
   preset.
3. Create the run.
4. Share the QR code or join URL with participant browser sessions.
5. Start after quorum is met.
6. Monitor participant state, round status, timeline, checkpoint-like artifacts,
   inference attachment, and common error states.
7. Export the evidence JSON bundle.

The host can abort a live run and can drop timed-out participants. Pause/hold is
not implemented in this local deterministic backend; the UI reports that through
the control surface instead of pretending a paused coordinator exists.

## Participant Flow

1. Open the QR/join URL.
2. Join the run with an optional display name.
3. Wait for assignment.
4. Run the browser-local surrogate learner.
5. Submit the update artifact.
6. Watch the submitted/completed/dropped/error state from backend events.

Participant update artifacts are `browser-update/1` metadata. They contain the
run id, participant id, round, runtime label, vector shape, sample count, L2
norm, hash, and simulator-vs-browser source. They do not contain observations,
actions, state labels, latents, tensors, model weights, or raw participant data.

## Runtime Decision Note (#298)

The first runtime path is a JavaScript Web Worker surrogate over resident
synthetic swipe-dot samples.

| Runtime option | Decision | Reason |
|---|---|---|
| ONNX Runtime Web | Use for inference only | It is already the browser inference path; it does not provide the local training loop needed here. |
| TensorFlow.js training | Defer | It would add a new dependency and a larger parity surface before the product flow is proven. |
| WebGPU custom training kernels | Defer | Too much infrastructure for the first educational slice. |
| JavaScript worker surrogate | Selected | Keeps local work off the UI thread, deterministic, dependency-light, and honest as a research/runtime slice. |

The selected surrogate is intentionally not described as production-grade
browser training. It validates orchestration, cancellation/progress/error
surfaces, metadata shape/hash contracts, and the coordinator submission path.
Exact training parity with the Python world-model objective remains a future
runtime task.

## Architecture

```text
host browser
  -> static app (web/federated-demo)
  -> local API (lensemble.demo.server)
  -> in-memory FederatedDemoService
  -> metadata-only aggregation
  -> checkpoint-like + inference artifact metadata
  -> residency-safe evidence export

participant browser
  -> QR/join URL
  -> backend participant lifecycle
  -> learner_worker.mjs
  -> browser-update/1 metadata
  -> submit update through API

fallback
  -> frontend-only simulator
  -> BroadcastChannel + localStorage
```

The backend API exposes:

- `POST /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/events?after={seq}` as NDJSON
- `POST /api/runs/{run_id}/join`
- `POST /api/runs/{run_id}/control`
- `POST /api/runs/{run_id}/participants/{participant_id}/heartbeat`
- `POST /api/runs/{run_id}/participants/{participant_id}/progress`
- `POST /api/runs/{run_id}/participants/{participant_id}/updates`
- `GET /api/runs/{run_id}/export`

## Inference Panel

The app reuses the swipe-dot JS/Canvas environment. It can step the environment
without a model, load an exported ONNX artifact in the browser when ONNX Runtime
Web is available, and attach a completed run's inference artifact metadata. The
panel reports model id, revision/hash, schema/version, source, state, action,
prediction metadata, latency, and load errors.

## Evidence Export

The evidence bundle is `demo-evidence/1` and includes run config, lifecycle
events, participant summaries, artifact references, aggregation mode, learner
runtime, redaction flags, and claim-boundary text. It deliberately excludes raw
participant data and model weights.

## Known Unsupported Paths

- No production multi-tenant auth.
- No raw data upload or browser data egress.
- No cryptographic honest-computation proof.
- No production browser-training claim.
- No paper-scale LeWorldModel performance claim.
- No dynamic-env claim that federation materially beats local-only.
- No physical SO-100 closed-loop success claim.

## Validation

```bash
uv run pytest tests/ml/test_federated_demo_app.py
uv run python scripts/check_docs_links.py docs SPEC.md README.md
uv run python -m mkdocs build --strict
git diff --check
```
