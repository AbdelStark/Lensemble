# Phase 3 Consortium Training Roadmap

Phase 3 is the final operational consortium-training phase for Lensemble. It
turns the Phase 2 evidence path into a Tapestry-style multi-party run: separate
participant trust domains, governed membership, networked participant agents,
secure aggregation and DP as runtime controls, longer federated training,
downstream evaluation, and a final public evidence bundle.

Tracker: [#220](https://github.com/AbdelStark/Lensemble/issues/220)

## Boundary

Phase 3 deliberately excludes the provenance ledger and cryptographic proof
layer.

Out of scope:

- provenance-ledger implementation;
- ZK/STARK proof implementation;
- TEE honest-computation proof implementation;
- any claim that participant computation is cryptographically proven;
- paper-scale robotics performance claims without matching evaluation evidence.

In scope:

- ordinary consortium governance metadata;
- participant dataset declarations and dataset cards;
- public-probe versioning and hash pinning;
- run manifests, config hashes, checkpoint hashes, and signatures where already
  supported;
- secure aggregation and DP accounting as operational privacy controls;
- residency checks and redaction gates;
- generated reports and conservative model-card language.

This keeps the operational consortium release independent from RFC-0006. Future
cryptographic contribution work should consume Phase 3 artifacts, not block this
phase.

## Baseline From Phase 2

Phase 2 closed the empirical evidence track in
[#200](https://github.com/AbdelStark/Lensemble/issues/200). The merged repo and
published artifacts show that Lensemble can:

- split and publish two SO-100 participant silos;
- run GPU-backed multi-round federated LeWorldModel-style training;
- publish checkpoint and report artifacts;
- run a compact downstream eval;
- publish a matched naive-FedAvg control;
- generate a final evidence bundle and model card.

Phase 3 raises the bar from a controlled HF Jobs path to a consortium run where
participants are independent operational actors.

## Workstreams

| Issue | Workstream | Exit gate |
|---|---|---|
| [#221](https://github.com/AbdelStark/Lensemble/issues/221) | Roadmap/spec and acceptance matrix | Public docs define the Phase 3 contract, issue graph, and non-claims. |
| [#222](https://github.com/AbdelStark/Lensemble/issues/222) | Consortium membership and run agreement | Coordinator and participants validate the same machine-readable run contract. |
| [#223](https://github.com/AbdelStark/Lensemble/issues/223) | Sovereign participant agent | A participant process joins, preflights, trains locally, and releases only allowed updates. |
| [#224](https://github.com/AbdelStark/Lensemble/issues/224) | Networked coordinator and dropout lifecycle | A coordinator service admits participants, assigns rounds, handles dropout, and closes/aborts deterministically. |
| [#225](https://github.com/AbdelStark/Lensemble/issues/225) | Multi-participant dataset and public-probe registry | At least four participant declarations validate without raw-data leakage or probe mismatch. |
| [#226](https://github.com/AbdelStark/Lensemble/issues/226) | Secure aggregation and DP runtime controls | A consortium smoke run exercises secure aggregation and DP accounting, or records an explicit blocker. |
| [#227](https://github.com/AbdelStark/Lensemble/issues/227) | Long-run consortium orchestration | A reproducible run launches coordinator plus at least four participants and publishes training artifacts. |
| [#228](https://github.com/AbdelStark/Lensemble/issues/228) | Downstream eval and matched controls | Evaluation moves beyond the Phase 2 toy boundary where feasible and records blocked controls honestly. |
| [#229](https://github.com/AbdelStark/Lensemble/issues/229) | Metrics, failure, and dropout report | Residency-safe observability explains participant lifecycle, timing, communication, and failure outcomes. |
| [#230](https://github.com/AbdelStark/Lensemble/issues/230) | Final evidence bundle and model card | A generated bundle verifies all referenced artifacts and preserves the non-cryptographic claim boundary. |

## Consortium Contract

Phase 3 introduces a consortium manifest consumed by both the coordinator and
participant agents. The manifest should capture:

- consortium id and run id;
- coordinator endpoint and accepted protocol version;
- participant ids, roles, owners, and contact metadata;
- participant runtime capabilities;
- accepted WMCP/action/observation contracts;
- public-probe id, version, and content hash;
- model and objective configuration;
- secure-aggregation mode, DP policy, dropout threshold, and retry budget;
- artifact publication targets;
- evaluation budget and model-card claim boundary.

The manifest is a governance and admission contract. It is not a cryptographic
identity proof and does not prove honest computation.

## Participant-Agent Contract

Each participant agent must run outside the coordinator trust boundary. Before
joining a run it must validate:

- its participant id is unique in the manifest;
- local data refs pass the selected adapter smoke checks;
- action and observation shapes match the manifest;
- the public-probe hash matches the consortium version;
- the model config is compatible with the accepted run config;
- residency rules prevent raw observations, raw actions, latents, embeddings,
  and private action-head weights from crossing the boundary.

During training the agent executes assigned local rounds, emits residency-safe
metrics, applies configured privacy controls, and releases only allowed update
artifacts.

## Coordinator Contract

The coordinator service owns round orchestration, not participant data. It must:

- validate the consortium manifest and participant join messages;
- assign rounds and collect heartbeats;
- reject duplicate or late updates;
- apply the configured dropout policy;
- close or abort rounds deterministically;
- publish checkpoint/report artifacts with config and model hashes;
- emit a residency-safe lifecycle trace.

The coordinator must not require access to raw participant trajectories or
participant-local action heads.

## Data And Public-Probe Registry

Phase 3 replaces ad hoc CLI data refs with a registry that declares:

- participant id;
- dataset ref or documented private/mounted location class;
- adapter format;
- windowing and held-out split policy;
- action spec and observation shape;
- dataset smoke report URI/hash;
- public-probe version and hash.

The registry must support at least four participant declarations. Public HF refs
are preferred. Private or unpublished participants are acceptable only if the
registry records the publication blocker and the final model card states the
evidence boundary.

This registry is not a provenance ledger.

## Privacy And Aggregation

Phase 3 should exercise secure aggregation and DP accounting as operational
privacy controls:

- secure-aggregation mode and fallback policy are explicit in config;
- per-participant clipping/noise policy is applied before release;
- the report records effective DP parameters;
- dropout threshold behavior is tested;
- fallback to visible individual updates is allowed only when reported as a
  blocker or limitation.

No Phase 3 artifact should imply that aggregation correctness or participant
computation is cryptographically proven.

## Training Run Shape

The target run should be large enough to demonstrate consortium operation while
remaining affordable and debuggable:

- at least four participants or simulated trust domains;
- at least ten closed federated rounds, unless a blocker records the maximum
  completed evidence;
- claim-mode LeWorldModel objective with `objective.target_stop_gradient=false`;
- public-probe frame anchoring enabled;
- secure aggregation and DP enabled where supported;
- published checkpoint, report, run manifest, and per-round metric artifacts.

Scaling model size should follow the dry-run evidence. A smaller model that
finishes with strong evidence is preferable to a larger run that cannot publish
complete artifacts.

## Evaluation And Controls

Phase 3 must not stop at training scalars. The eval plan should declare:

- task/environment ids;
- held-out policy;
- planner budget;
- seeds;
- metrics;
- expected and falsifying outcomes;
- matched controls.

Controls to attempt:

- local-only;
- naive FedAvg;
- anchored federation;
- Fork A or frozen-encoder fallback;
- centralized/pooled only where licensing and governance allow it.

Missing controls must be explicit blocked rows in the final report.

## Evidence Bundle

The final generated bundle should aggregate:

- consortium manifest;
- dataset/probe registry;
- training report;
- privacy/aggregation report;
- observability/dropout report;
- evaluation and control report;
- checkpoint refs and hashes;
- model-card text.

Bundle generation must verify referenced artifacts exist before emitting a
success bundle. The checked-in and published bundle must be residency-safe.

## Acceptance Matrix

| Gate | Minimum passing evidence |
|---|---|
| Membership | Four participant declarations validate against one consortium manifest. |
| Runtime | Independent participant-agent processes complete a multi-round run through a networked coordinator. |
| Residency | Redaction tests and reports show no raw observations, raw actions, latents, embeddings, private action-head weights, tokens, or secrets cross boundaries. |
| Aggregation/privacy | Secure aggregation and DP accounting are exercised, or fallback/blockers are recorded in report and model card. |
| Training | At least ten closed rounds, or an explicit blocker with the maximum completed closed-round evidence. |
| Evaluation | A downstream eval report goes beyond the Phase 2 toy boundary where feasible, with blocked rows for unavailable controls. |
| Observability | Participant lifecycle, dropout/retry, timing, communication, and artifact-publication status are reported. |
| Release | Checkpoint repo publishes final checkpoint, reports, evidence bundle, and model card at immutable revisions. |
| Claims | Public text says this is consortium-engineering evidence, not paper-scale performance or cryptographic honest-computation proof. |

## Final Claim Boundary

Closing Phase 3 should support this statement:

> Lensemble completed a governed, multi-party consortium training run for a
> federated JEPA / LeWorldModel-flavour world model across participant-local
> data, with independent participant agents, published checkpoints, evaluation,
> privacy controls, lifecycle reporting, and a conservative evidence bundle.

It should not claim:

- cryptographic proof of participant computation;
- provenance-ledger-backed contribution accounting;
- broad robotics generalization;
- paper-scale LeWorldModel performance.
