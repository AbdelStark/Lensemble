# Tapestry-like LeWM TwoRooms Pivot — Contract and Implementation RFC

Tracker: [#314](https://github.com/AbdelStark/Lensemble/issues/314) is the active
epic. This document is the pivot contract required by
[#315](https://github.com/AbdelStark/Lensemble/issues/315): it locks the claim
boundary, the product modes, the external artifacts, the implementation
sequence, and the go/no-go gates before the implementation widens.

The pivot: the next Lensemble web demo is a **Tapestry-like browser-federated
LeWorldModel showcase** built around a **real LeWM TwoRooms checkpoint**, not
another surrogate-vector run. Sovereign browser participants keep TwoRooms
rollouts local, run checkpoint-backed LeWM inference, train a bounded
checkpoint-adaptation subset, submit bounded adapter deltas, and the coordinator
aggregates them into hash-bound global revisions with claim-bounded evidence.

"Tapestry-like" means what it means elsewhere in this corpus
([spec/00-overview.md](../spec/00-overview.md#11-references),
[RFC-0016 §2](../rfcs/RFC-0016-deployment-vendoring-topology.md)): sovereign
participants, a shared protocol, bounded update artifacts, coordinator
aggregation, observability, and claim-bounded evidence — concepts borrowed from
Project Tapestry (AI Alliance), not its primitives and not its scale.

## Accepted claim

> Lensemble demonstrates a **Tapestry-like browser-local federated adaptation
> run** for a real LeWorldModel TwoRooms checkpoint. Browser participants keep
> local rollouts resident, train a bounded checkpoint-adaptation subset, submit
> bounded deltas, aggregate into global revisions, and inspect real inference,
> loss, anti-collapse, probe, planning, and evidence outputs.

The first acceptable milestone, if the browser training runtime cannot support
the full LeWM objective, is checkpoint-backed inference plus browser-local
federated predictor/adapter continuation with clear labels.

## Non-claims

The demo must not claim, and the UI/docs/model card must not imply:

- full from-scratch LeWorldModel base training inside every browser;
- production browser training;
- paper-scale TwoRooms or PushT benchmark parity;
- cryptographic honest-computation proof;
- closed-loop physical robotics success;
- raw participant data, raw frames, actions, labels, latents, tensors, or base
  checkpoint weights uploaded from participant browsers.

These extend, and do not replace, the repository-wide
[README non-claims](../../README.md#non-claims) and the demo non-claims in
[BROWSER_FEDERATED_DEMO.md](BROWSER_FEDERATED_DEMO.md#known-unsupported-paths).

## Product modes

The web demo exposes two clearly separated run modes. Mode is a run-level,
host-selected property, recorded in run snapshots and evidence.

| Mode id | Status | What it is |
|---|---|---|
| `surrogate-swipe-dot` | existing, default until the real mode passes its gates | The educational surrogate path from [#294](https://github.com/AbdelStark/Lensemble/issues/294)/[#303](https://github.com/AbdelStark/Lensemble/issues/303): tiny browser learner over synthetic swipe-dot trajectories, `browser-update/1` tiny-vector artifacts, `tiny-vector-mean` aggregation. |
| `real-lewm-tworooms` | this epic | Checkpoint-backed LeWM TwoRooms inference in the browser, browser-local bounded adapter continuation, `lewm-adapter-delta/1` artifacts, deterministic adapter aggregation, real loss/anti-collapse/probe/planning metrics, hash-bound evidence. |

Rules:

- The surrogate mode keeps its existing labels and cannot be mistaken for the
  real LeWM mode: run snapshots, dashboard headers, inference panels, and
  evidence bundles all carry the mode id and mode-specific non-claim text.
- The real mode never falls back silently to the surrogate learner. If the
  browser cannot run the real path (no WebGPU/WASM capability, artifact fetch
  failure), the participant reports an explicit unsupported/fallback state that
  is visible on the dashboard and recorded in evidence.
- Participant `automationMode` (auto vs manual/debug) is orthogonal to the run
  mode and remains available in both; auto is the default for demos.

## External artifacts (pinned)

| Artifact | Reference | Pin |
|---|---|---|
| LeWorldModel upstream | <https://github.com/lucas-maes/le-wm> | concepts/paper reference |
| LeWM model/source basis | <https://github.com/galilai-group/stable-worldmodel> (`stable_worldmodel.wm.lewm`) | architecture reference for in-tree reconstruction |
| TwoRooms checkpoint | <https://huggingface.co/quentinll/lewm-tworooms> | revision `77adaae0bc31deab21c93740d1f8bb947cd0bdec` (`config.json` + `weights.pt`, MIT) |
| PushT checkpoint (optional, later) | <https://huggingface.co/quentinll/lewm-pusht> | not in scope for the first run |

The TwoRooms checkpoint architecture, from its pinned `config.json`: ViT-Tiny
image encoder (`patch_size=14`, `image_size=224`, `D=192`, 12 layers, 3 heads,
CLS token), `Embedder` action encoder (`input_dim=10` → 192), 6-layer AdaLN
transformer predictor (`num_frames=3`, heads 16, head dim 64, MLP 2048), and
two MLP projection heads (192→2048→192) with BatchNorm (`projector`,
`pred_proj`). Checkpoint handling is server-side/build-time; participant
browsers receive exported inference artifacts, never `weights.pt`.

Vendoring note: `third_party/stable_worldmodel` and
`third_party/stable_pretraining` remain **not vendored**
([#96](https://github.com/AbdelStark/Lensemble/issues/96),
[RFC-0016 §2](../rfcs/RFC-0016-deployment-vendoring-topology.md)). The pivot
reconstructs the checkpoint's module shape in-tree (state-dict-compatible) and
records the upstream references in the checkpoint manifest; it does not import
upstream code at runtime.

## Implementation sequence

The single active queue, in dependency order
([#314](https://github.com/AbdelStark/Lensemble/issues/314) carries the full
graph):

1. [#315](https://github.com/AbdelStark/Lensemble/issues/315) — this contract.
2. [#316](https://github.com/AbdelStark/Lensemble/issues/316) — ingest the
   pinned TwoRooms checkpoint, reconstruct the LeWM modules in-tree, prove
   PyTorch reference parity, emit a hash-bound checkpoint manifest.
3. [#317](https://github.com/AbdelStark/Lensemble/issues/317) — export
   checkpoint-backed inference graphs (encoder+projection, action encoder,
   predictor+pred_proj) for browser execution (ONNX Runtime Web preferred),
   with PyTorch-vs-export parity tests and hash-bound export metadata.
4. [#318](https://github.com/AbdelStark/Lensemble/issues/318) — TwoRooms
   browser environment/viewer running real LeWM rollout/planning inference from
   the exported graphs; deviations from upstream TwoRooms are documented and
   the env is labeled a TwoRooms-compatible visualization probe if exact
   dynamics/assets are unavailable.
5. [#319](https://github.com/AbdelStark/Lensemble/issues/319) — browser-local
   adapter continuation: frozen exported base, bounded trainable subset
   (predictor adapter/LoRA-style residuals), next-latent prediction loss plus
   SIGReg or a documented SIGReg-compatible surrogate, real optimizer steps,
   real metrics, and a deterministic overfit fixture proving loss decrease.
6. [#320](https://github.com/AbdelStark/Lensemble/issues/320) —
   `lewm-adapter-delta/1` update schema, server-side validation, deterministic
   aggregation into global adapter revisions bound to parent checkpoint/export
   hashes, honest privacy/secure-aggregation status.
7. [#321](https://github.com/AbdelStark/Lensemble/issues/321) — host-selectable
   `real-lewm-tworooms` run mode integrated into the existing federated flow
   with autonomous participants by default.
8. [#322](https://github.com/AbdelStark/Lensemble/issues/322) — loss,
   anti-collapse, probe, and planning diagnostics; before/after revision
   comparison on a fixed browser-side validation task.
9. [#323](https://github.com/AbdelStark/Lensemble/issues/323) — evidence
   bundle schema for real-mode runs, model/demo card, claim-audit tests.
10. [#324](https://github.com/AbdelStark/Lensemble/issues/324) — end-to-end
    rehearsal gate and researcher-facing runbook
    ([TAPESTRY_LEWM_RUNBOOK.md](TAPESTRY_LEWM_RUNBOOK.md)).

The previously open dynamic-env backlog
([#273](https://github.com/AbdelStark/Lensemble/issues/273),
[#285](https://github.com/AbdelStark/Lensemble/issues/285),
[#286](https://github.com/AbdelStark/Lensemble/issues/286),
[#287](https://github.com/AbdelStark/Lensemble/issues/287),
[#289](https://github.com/AbdelStark/Lensemble/issues/289),
[#290](https://github.com/AbdelStark/Lensemble/issues/290)) is superseded
historical evidence, not the next tasks to pick up. Its honesty lessons
(ungameable metrics, held-out collapse detection, claim-bounded reporting)
carry forward into the gates below.

## Evidence gates (go/no-go)

Each gate is binding: the next stage does not start, and no public claim is
made, until the gate's artifact exists and its tests pass. A failed gate blocks
with a documented negative result instead of softening the claim.

| Gate | Binding artifact | Pass condition |
|---|---|---|
| G1 checkpoint parity | checkpoint manifest (revision, config hash, weights hash, tensor inventory) | reconstruction loads `weights.pt` strictly; deterministic reference forwards reproduce fixed fixture hashes; tests fail on mismatched config, missing weights, unknown tensor names, or unpinned revision in claim-grade mode |
| G2 browser export parity | export manifest (graph hashes, opset, runtime backend) | PyTorch-vs-export outputs match on fixed fixtures within stated tolerances; artifacts regenerate deterministically from the pinned checkpoint |
| G3 local adaptation | one-browser overfit fixture | prediction loss decreases on fixed resident rollouts with the bounded trainable subset; SIGReg/collapse diagnostics computed from real tensors |
| G4 collapse diagnostics | metric payload schema | latent std / effective rank / collapse statistics are computed from real training outputs and surface low-variance failure modes; fabricated or incomplete payloads are rejected |
| G5 probe/planning | before/after diagnostic report | at least one fixed validation probe (state/coordinate probe, goal-distance, or planning success) compares parent revision vs adapted revision; a negative result blocks positive claims |
| G6 federated aggregation | per-round delta hashes + global revision hashes | bounded adapter deltas validate, aggregate deterministically, and bind to parent checkpoint/export hashes; invalid/stale/oversized/raw-like updates are rejected by tests |
| G7 final rehearsal | rehearsal report + evidence bundle | a two-participant (and documented four-participant) real-mode run completes end to end and exports a residency-safe evidence bundle with the non-claim text |

Gate artifacts shipped so far (all under [docs/evidence](../evidence/lewm_tworooms_demo_card.md)):
G1 → [checkpoint manifest](../evidence/lewm_tworooms_checkpoint_manifest.json) +
[reference report](../evidence/lewm_tworooms_reference_report.json); G2 →
[browser export manifest](../evidence/lewm_tworooms_browser_export_manifest.json) +
[action stats](../evidence/lewm_tworooms_action_stats.json) +
[real-data check](../evidence/lewm_tworooms_realdata_check.json); G3 →
[adapter overfit](../evidence/lewm_tworooms_adapter_overfit.json); G5 →
[federated probe](../evidence/lewm_tworooms_probe_check.json); the summary card is the
[Tapestry-like LeWM demo card](../evidence/lewm_tworooms_demo_card.md).

Cross-cutting gates, inherited from the repo:

```bash
uv run pytest tests/ml/test_federated_demo_app.py
uv run python scripts/check_docs_links.py docs SPEC.md README.md
uv run python -m mkdocs build --strict
git diff --check
```

## Privacy and residency contract

Unchanged from the existing demo and binding for the real mode:

- Raw observations, actions, labels, latents, tensors, participant tokens, and
  base checkpoint weights never leave the participant browser and are rejected
  server-side if they appear in any payload.
- Adapter deltas are bounded (byte budget, declared shapes, clip norm) and
  carry only derived metric summaries.
- Secure-aggregation and DP accounting hooks from the main stack
  ([RFC-0011](../rfcs/RFC-0011-secure-aggregation.md),
  [RFC-0012](../rfcs/RFC-0012-differential-privacy.md)) are preserved where the
  demo backend supports them; where the demo path does not implement them, the
  evidence bundle states exactly what is simulated or absent rather than
  implying protection.

## Relationship to the existing demo

The orchestration substrate from
[BROWSER_FEDERATED_DEMO.md](BROWSER_FEDERATED_DEMO.md) is reused, not rebuilt:
QR joins, participant tokens and admission, WebSocket fanout with REST/NDJSON
fallback, heartbeat/reconnect/drop handling, update validation and residency
guard, aggregation and revision publication, dashboard, and evidence export.
The real mode swaps the learner, the update artifact, the aggregation payload,
the inference panel, and the metrics — the protocol and lifecycle stay.
