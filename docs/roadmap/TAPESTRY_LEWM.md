# Tapestry-like LeWM TwoRooms — Implemented Contract

Epic [#314](https://github.com/AbdelStark/Lensemble/issues/314) and all of its
delivery issues
([#315](https://github.com/AbdelStark/Lensemble/issues/315) through
[#324](https://github.com/AbdelStark/Lensemble/issues/324)) are closed. The
system-composed evidence hardening in
[#332](https://github.com/AbdelStark/Lensemble/issues/332) is also closed. This
document records the implemented contract and evidence boundary.

"Tapestry-like" names the architectural pattern used here: sovereign
participants, a shared update protocol, bounded artifacts, coordinator
aggregation, observability, and claim-bounded evidence. It borrows those
concepts from Project Tapestry (AI Alliance), not its implementation, security
properties, deployment scale, or organizational model.

## Accepted claim

> Lensemble implements federated adapter continuation on a frozen
> LeWorldModel TwoRooms checkpoint. Browser participants run hash-checked
> checkpoint inference, train a bounded 12,512-parameter (0.069%) residual
> adapter with real gradients, and submit clipped adapter deltas to one local
> coordinator. The coordinator validates those deltas and publishes
> deterministic, hash-bound adapter revisions.

The checkpoint's encoder, action encoder, predictor, and projection heads stay
frozen. This is not federated training of the world model and there is no
full-model browser training.

## Non-claims

The UI, evidence, and documentation must not imply:

- full-model or from-scratch LeWorldModel training in participant browsers;
- production browser training or production multi-tenant operation;
- secure aggregation or differential privacy in this adapter path;
- a decentralized multi-operator run;
- paper-scale TwoRooms or PushT benchmark parity;
- cryptographic proof of honest participant computation or data residency;
- closed-loop physical robotics success.

The real-LeWM path uses a single trusted local coordinator and a deterministic
mean of clipped adapter deltas. Secure aggregation and DP from the main
federation stack are not wired into this path.

## Run modes

The browser application records a host-selected mode in run snapshots,
revisions, and evidence:

| Mode id | Implemented behavior |
|---|---|
| `surrogate-swipe-dot` | Synthetic swipe-dot orchestration fixture with a tiny `browser-update/1` vector. It carries no LeWorldModel claim. |
| `real-lewm-tworooms` | Pinned TwoRooms inference plus federated continuation of the bounded residual adapter through `lewm-adapter-delta/1`. |

The modes have separate artifact schemas and labels. A real-LeWM participant
fails visibly when it cannot load the hash-checked runtime; it never falls back
silently to the surrogate learner.

## Pinned external artifacts

| Artifact | Reference | Pin |
|---|---|---|
| LeWorldModel upstream | <https://github.com/lucas-maes/le-wm> | concepts and paper reference |
| LeWM model/source basis | <https://github.com/galilai-group/stable-worldmodel> (`stable_worldmodel.wm.lewm`) | architecture reference for the in-tree reconstruction |
| TwoRooms checkpoint | <https://huggingface.co/quentinll/lewm-tworooms> | revision `77adaae0bc31deab21c93740d1f8bb947cd0bdec`; `config.json` and `weights.pt` |

The checkpoint is a ViT-Tiny image encoder with an action encoder, AdaLN
transformer predictor, and projection heads. Checkpoint ingestion is a
server/build-time operation. Browsers receive hash-bound inference exports,
never the source `weights.pt`.

`third_party/stable_worldmodel` and `third_party/stable_pretraining` remain
unvendored. Lensemble reconstructs the required checkpoint-compatible module
shape in tree and records upstream references in the checkpoint manifest.

## Closed delivery record

| Issue | Delivered contract |
|---|---|
| [#315](https://github.com/AbdelStark/Lensemble/issues/315) | Claim boundary and implementation contract. |
| [#316](https://github.com/AbdelStark/Lensemble/issues/316) | Strict checkpoint ingestion, reference parity, and hash-bound manifest. |
| [#317](https://github.com/AbdelStark/Lensemble/issues/317) | Hash-bound browser inference exports and PyTorch/export parity. |
| [#318](https://github.com/AbdelStark/Lensemble/issues/318) | TwoRooms-compatible browser inference and planning view. |
| [#319](https://github.com/AbdelStark/Lensemble/issues/319) | Frozen-base residual-adapter continuation with real browser gradients and anti-collapse diagnostics. |
| [#320](https://github.com/AbdelStark/Lensemble/issues/320) | Bounded `lewm-adapter-delta/1` validation and deterministic adapter revision aggregation. |
| [#321](https://github.com/AbdelStark/Lensemble/issues/321) | Explicit `real-lewm-tworooms` mode integrated into the existing coordinator lifecycle. |
| [#322](https://github.com/AbdelStark/Lensemble/issues/322) | Loss, probe, and collapse diagnostics. |
| [#323](https://github.com/AbdelStark/Lensemble/issues/323) | Evidence schema, demo card, and claim-audit tests. |
| [#324](https://github.com/AbdelStark/Lensemble/issues/324) | End-to-end rehearsal and researcher runbook. |

## Evidence contract

Checkpoint and local-math evidence includes:

- [checkpoint manifest](../evidence/lewm_tworooms_checkpoint_manifest.json) and
  [reference report](../evidence/lewm_tworooms_reference_report.json);
- [browser export manifest](../evidence/lewm_tworooms_browser_export_manifest.json),
  [action statistics](../evidence/lewm_tworooms_action_stats.json), and
  [real-data check](../evidence/lewm_tworooms_realdata_check.json);
- [adapter overfit report](../evidence/lewm_tworooms_adapter_overfit.json).

The positive adapter claim is bound to two system-composed artifacts:

1. The [system-composed probe](../evidence/lewm_tworooms_system_probe.json)
   sends real adapter deltas through `FederatedDemoService.submit_update` and
   `_close_round_lewm`, then scores the server-produced final revision. Its
   seed-`20260612` held-out MSE improves from
   `0.06037897796856822` to `0.05296723026100999` (`+12.275%`) without a
   collapse flag.
2. The [five-seed sweep](../evidence/lewm_tworooms_probe_seedsweep.json)
   repeats that system path across independent episode splits. All five draws
   improve; mean relative improvement is `16.8%`, worst-case improvement is
   `+5.4%`, and no draw carries a collapse-risk verdict.

Those artifacts are the binding browser evidence. The
[offline probe](../evidence/lewm_tworooms_probe_check.json) produced by
`scripts/lewm_probe_check.py` reimplements the coordinator mean and is only an
independent math cross-check. It cannot substitute for the shipped validation
and aggregation path.

The result is a narrow validation-pair measurement for the frozen-checkpoint
adapter. It does not show that federation materially improves full-model
training, and it does not establish general TwoRooms or robotics usefulness.

## Privacy and deployment boundary

- Participant update payloads reject raw observations, frames, actions,
  labels, latent batches, participant tokens, and base checkpoint weights.
- Adapter deltas are bounded by exact inventory, shape, byte, parameter-count,
  revision, and clip-norm checks.
- These server-side checks are an artifact boundary, not a cryptographic
  residency or honest-computation proof.
- One trusted local coordinator validates and averages clipped deltas.
- Secure aggregation and DP are absent from this demo path.

A true multi-process or cross-operator run remains deferred in
[#331](https://github.com/AbdelStark/Lensemble/issues/331). Full-model
federated training is a separate research question tracked in
[#335](https://github.com/AbdelStark/Lensemble/issues/335); the adapter result
must not be used as evidence for it.

## Reproduction

See [TAPESTRY_LEWM_RUNBOOK.md](TAPESTRY_LEWM_RUNBOOK.md) for the evidence
commands and [BROWSER_FEDERATED_DEMO.md](BROWSER_FEDERATED_DEMO.md) for the
runtime contract.
