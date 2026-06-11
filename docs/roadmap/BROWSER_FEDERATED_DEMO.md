# Browser Federated Demo Roadmap

Trackers: [#294](https://github.com/AbdelStark/Lensemble/issues/294) closed the
local educational baseline. [#303](https://github.com/AbdelStark/Lensemble/issues/303)
is the hackathon-readiness layer above it.

The browser federated demo is an educational systems showcase for Lensemble's
Tapestry-like JEPA federation story: sovereign participants join a run, do
resident browser-local work, submit bounded derived update artifacts, and the
host can inspect lifecycle, liveness, aggregation, inference, and evidence
export.

It is not a benchmark win over local-only, not production browser training, not
a cryptographic honest-computation proof, and not a closed-loop physical SO-100
success claim.

## Current Shape

```bash
uv run lensemble demo federated --port 8765
```

Open the printed host URL, usually:

```text
http://127.0.0.1:8765/web/federated-demo/
```

For public or tunnel rehearsal, bind the coordinator and set the external base
URL used by QR joins and WebSocket URLs:

```bash
uv run lensemble demo federated \
  --host 0.0.0.0 \
  --port 8765 \
  --public-base-url https://YOUR-TUNNEL.trycloudflare.com/web/federated-demo \
  --public-demo \
  --deployment-target cloudflare-tunnel
```

Startup output prints the host URL, public base URL, participant join root,
transport mode, fallback mode, deployment target, and safety settings.

## Hackathon Live Flow

1. The host opens the public HTTPS URL and creates a run.
2. The dashboard shows a QR code and join URL for that run.
3. Four participants scan the QR code from mobile browsers and join with short
   display names.
4. The host sees participants arrive, starts the run after quorum, and watches
   live progress.
5. Each phone runs a tiny browser-local learner over resident synthetic
   swipe-dot trajectories in a Web Worker.
6. Each phone submits a clipped `browser-update/1` artifact containing a tiny
   derived vector and metadata; raw samples do not leave the browser.
7. The coordinator validates artifacts, aggregates submitted vectors, publishes
   a deterministic tiny model revision, and advances rounds.
8. The host dashboard shows liveness, training progress, update submission,
   aggregate metrics, model revision metadata, errors, and fallback status.
9. The browser inference panel loads the final tiny JS revision and runs the
   swipe-dot environment live. ONNX loading remains an explicit optional path.
10. The host exports a residency-safe evidence bundle and can replay the run
    story.

## Protocol Map

Architecture decision: browsers talk to a coordinator-owned HTTP/WebSocket
endpoint. Do not introduce Kafka for the hackathon path. Do not expose NATS
directly to browsers. NATS may be considered later as an internal backend bus if
the coordinator splits into services.

```text
host dashboard + phone participants
  -> HTTPS/WSS public endpoint
  -> coordinator backend
      -> run state machine
      -> participant tokens and admission
      -> WebSocket event fanout
      -> REST/NDJSON polling fallback
      -> heartbeat, reconnect, timeout, and drop handling
      -> tiny browser learner task assignment
      -> update validation and residency guard
      -> aggregation and global revision publication
      -> evidence export
  -> browser inference panel
```

REST endpoints retained from #294:

- `POST /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/events?after={seq}` as NDJSON fallback
- `GET /api/runs/{run_id}/export`
- `GET /api/runs/{run_id}/model-revisions/{revision_id}`
- `POST /api/runs/{run_id}/join`
- `POST /api/runs/{run_id}/control`
- `POST /api/runs/{run_id}/participants/{participant_id}/heartbeat`
- `POST /api/runs/{run_id}/participants/{participant_id}/progress`
- `POST /api/runs/{run_id}/participants/{participant_id}/updates`

WebSocket endpoint:

- `GET /api/runs/{run_id}/ws?role=host&after={seq}`
- `GET /api/runs/{run_id}/ws?role=participant&participantId={participant_id}&after={seq}`

Participant WebSocket authentication uses the same participant token validation
as REST. Browser clients pass the participant token in a WebSocket protocol
header, not in the URL. Tokens are not emitted in events or exported evidence.

Event behavior:

- Events carry monotonic `seq`, `kind`, `severity`, `actor`, `participantId`,
  `round`, `runState`, and residency-safe `payload`.
- WebSocket subscribers receive a snapshot plus replay after `after`.
- REST polling keeps the same monotonic replay semantics as fallback.
- Important event kinds include `connection.opened`, `connection.closed`,
  `participant.joined`, `participant.resumed`, `participant.stale`,
  `participant.dropped`, `participant.training`, `update.submitted`,
  `round.aggregating`, `round.closed`, `inference.ready`, and
  `run.completed`.

Host commands:

- `start`
- `abort`
- `fail`
- `timeout-missing`
- `drop`

Participant commands:

- `heartbeat`
- `progress`
- `submitUpdate`

The WebSocket command path routes through the same service-level validation as
REST. No browser command bypasses token checks, raw-data rejection, norm bounds,
or duplicate/stale update guards.

## Tiny Update Artifact

`browser-update/1` is the allowed participant update. It is intentionally tiny
and explicit:

| Field | Meaning |
|---|---|
| `schema` | Must be `browser-update/1`. |
| `runId`, `round`, `roundId` | Must match the active run and round. |
| `participantId` | Must match the authenticated participant. |
| `modelRevisionId` | Must match the active model revision. |
| `source`, `runtime` | Browser-local tiny learner or simulator fixture label. |
| `shape`, `parameterCount`, `vector` | One-dimensional clipped derived update vector. |
| `sampleCount`, `localSteps`, `seed` | Tiny learner work metadata. |
| `hash` | 64-character lowercase hex update hash. |
| `l2Norm`, `clipNorm` | Server-validated norm and bound. |
| `loss`, `probe`, `runtimeMs` | Derived progress/quality telemetry. |
| `simulated` | Marks simulator fixtures separately from browser-local work. |

Current public-demo limits are exposed in run snapshots and evidence export:
max artifact bytes, max message bytes, max vector length, max participants, max
rounds, token TTL, heartbeat stale timeout, participant timeout, rate limit, and
clip norm.

Forbidden fields include raw observations, actions, labels, latents, tensors,
participant tokens, and model weights. Payloads with raw-data-like keys are
rejected before aggregation.

## Aggregation and Inference

When quorum is met, the coordinator computes a deterministic mean over submitted
tiny update vectors and publishes:

- a checkpoint-style integrity artifact;
- a `demo-model-revision/1` artifact with revision id, parent revision, vector,
  source update hashes, contributing participants, aggregate norm, and hash;
- a `demo-inference-artifact/1` attachment that the browser inference panel can
  load without hidden server-local state.

The final inference path is the tiny JS vector runtime. If a separate ONNX export
from #289 is available, the panel can load it explicitly; the hackathon path does
not depend on #289.

## Evidence Bundle

`demo-evidence/1` includes:

- run config;
- public-vs-local mode;
- transport mode;
- deployment target and public base URL;
- participant summaries and liveness state;
- full event trace;
- round metrics;
- model revision refs;
- source update hashes;
- artifact refs;
- fallback mode;
- redaction flags;
- non-claim text.

It excludes raw participant data, participant tokens, and model weights.

## Rehearsal Gate

Automated deterministic rehearsal:

```bash
uv run python scripts/hackathon_demo_rehearsal.py
```

The script exercises:

- host creates a run;
- four synthetic phone participants join;
- quorum start;
- tiny update submission;
- aggregation;
- final model revision publication;
- inference artifact availability;
- evidence export;
- participant dropout with quorum preserved.

Manual day-of checklist:

1. Start with the public HTTPS/WSS path.
2. Confirm `/api/health` reports the expected deployment target and safety
   settings.
3. Create a fresh run from the host dashboard.
4. Scan the QR code from four phones using mobile Safari/Chrome.
5. Confirm stable participant slots and WebSocket transport on the host.
6. Start after quorum, keep phone screens awake, and watch local learner
   progress.
7. Confirm update submission, aggregation, model revision id, and inference
   readiness.
8. Run one inference step from the host panel.
9. Export evidence JSON.
10. Reset and rehearse one failure path: refresh/reconnect, dropout with quorum
    preserved, or quorum-loss abort.

Fallback checklist:

- Cloudflare Tunnel: run the local coordinator with `--host 0.0.0.0`, start the
  tunnel to port `8765`, pass the tunnel HTTPS URL as `--public-base-url`, and
  keep REST polling fallback visible.
- LAN/hotspot: connect host and phones to the same network, bind
  `--host 0.0.0.0`, use the host LAN IP in `--public-base-url`, and rehearse QR
  joins before going on stage.
- Reset: create a new run; no persistent account state is required.

Host narration script:

```text
Each phone keeps its synthetic trajectory samples local, runs a tiny browser
learner, and submits only a clipped derived update vector. The coordinator
validates and averages those bounded updates, publishes a tiny model revision,
and the browser inference panel loads that revision. This is an educational
systems demo, not proof of production browser training or a benchmark win.
```

## Known Unsupported Paths

- No production multi-tenant auth.
- No raw data upload or browser data egress.
- No cryptographic honest-computation proof.
- No production browser-training claim.
- No paper-scale LeWorldModel performance claim.
- No dynamic-env claim that federation materially beats local-only.
- No physical SO-100 closed-loop success claim.
- No Kafka cluster and no participant-facing NATS broker.

## Validation

```bash
uv run pytest tests/ml/test_federated_demo_app.py
uv run python scripts/hackathon_demo_rehearsal.py
uv run python scripts/check_docs_links.py docs SPEC.md README.md
uv run python -m mkdocs build --strict
git diff --check
```
