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

The schema and shared validators live in `lensemble.config.consortium`. Generate
the checked-in four-participant example with:

```bash
uv run --extra dev python scripts/phase3_consortium_manifest.py \
  --output docs/evidence/phase3_consortium_manifest.example.json
```

Validate any candidate manifest with:

```bash
uv run --extra dev python scripts/phase3_consortium_manifest.py \
  --validate docs/evidence/phase3_consortium_manifest.example.json
```

The example uses simulated trust-domain data refs so it is a contract fixture,
not Phase 3 training evidence.

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

The #223 participant-agent runtime lives in `lensemble.federation.agent` and is
exported as `lensemble.federation.Phase3ParticipantAgent`. It wraps the existing
claim-mode `Participant.local_round` implementation rather than changing model
internals. The agent writes local resume state under the participant's private
state directory:

- `delta.safetensors`: the released pseudo-gradient delta only;
- `round_state.json`: hashes, counts, DP/release flags, and round metadata only;
- `lensemble.log.jsonl` and `metrics.jsonl`: residency-safe observability.

The CLI preflight surface is:

```bash
uv run lensemble federate participant-agent \
  --manifest path/to/phase3_consortium_manifest.json \
  --registry path/to/phase3_dataset_registry.json \
  --participant-id phase3-so100-a \
  --coordinator https://coordinator.example.invalid \
  --data-source lerobot-h5://path/to/private-silo.h5 \
  --state-dir runs/phase3/phase3-so100-a \
  data.format=lerobot-h5 \
  data.probe_path=path/to/public-probe.safetensors \
  objective.target_stop_gradient=false \
  objective.lambda_anc=0.01 \
  federation.transport=network \
  federation.aggregation_backend=masking
```

This command validates the local participant boundary before any coordinator
message. Assigned-round execution from the CLI is intentionally left to the
network coordinator service in #224; integration tests exercise the runtime over
the in-process test transport. The checked-in example manifest remains a
contract fixture; runtime preflight requires a manifest whose model agreement
matches the participant `LensembleConfig`.

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

The #224 coordinator-service runtime lives in
`lensemble.federation.service` and is exported as
`lensemble.federation.Phase3CoordinatorService`. It wraps the existing
deterministic `Coordinator.try_round()` engine with the Phase 3 control plane:

- governed admission against the consortium manifest;
- heartbeat, assignment, update submission, explicit abort, and close-round
  flows;
- late-join and duplicate-update rejection;
- explicit dropout policy derived from manifest/config quorum, collect timeout,
  and retry budget;
- residency-safe JSONL trace events for participants and rounds.

The CLI startup surface is:

```bash
uv run lensemble federate coordinator-service \
  --manifest path/to/phase3_consortium_manifest.json \
  --registry path/to/phase3_dataset_registry.json \
  --listen https://coordinator.example.invalid \
  --run-dir runs/phase3/coordinator \
  objective.target_stop_gradient=false \
  federation.transport=network \
  federation.aggregation_backend=masking
```

The command validates and starts the service control plane, writes the startup
trace, and emits a machine-readable service report. The transport layer owns
long-running socket serving; integration tests exercise the service lifecycle
over the in-process transport, including a three-participant smoke with one
induced dropout.

## Data And Public-Probe Registry

Phase 3 replaces ad hoc CLI data refs with a registry that declares:

- participant id;
- dataset ref or documented private/mounted location class;
- adapter format;
- windowing and held-out split policy;
- action spec and observation shape;
- dataset smoke report URI/hash;
- public-probe version and hash.

The #225 registry schema lives in `lensemble.data.phase3`. It validates the
same participant ids, accepted action/observation contracts, public-probe hash,
data refs, adapter formats, windowing, held-out policies, smoke-report hashes,
and license metadata as the consortium manifest. The participant agent and
coordinator service both accept the same optional registry artifact and fail
preflight if it disagrees with the manifest.

Generate the checked-in four-participant example with:

```bash
uv run --extra dev python scripts/phase3_dataset_registry.py \
  --output docs/evidence/phase3_dataset_registry.example.json
```

Validate a candidate registry, including the manifest agreement, with:

```bash
uv run --extra dev python scripts/phase3_dataset_registry.py \
  --validate docs/evidence/phase3_dataset_registry.example.json \
  --against-manifest docs/evidence/phase3_consortium_manifest.example.json
```

The registry must support at least four participant declarations for the public
Phase 3 example. Public HF refs are preferred. Private or unpublished
participants are acceptable only if the registry records the exact publication
blocker and the final model card states the evidence boundary. Public-example
mode rejects raw/private dataset paths unless they are explicit placeholders
with blockers; private-consortium mode requires an explicit raw-path allowance.

Public-probe governance rules:

- the probe hash is immutable for one `run_id`;
- any probe change requires a new version and content hash;
- registry, manifest, participant preflight, and coordinator preflight must all
  be regenerated/validated together;
- model cards must cite the exact probe hash and state whether participant data
  refs are public, private, or blocked placeholders.

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

The #226 runtime report lives in `lensemble.federation.phase3_privacy` and is
attached to `Phase3CoordinatorService` after a successful round close. It
records:

- selected secure-aggregation backend and threshold;
- whether an in-process secure sum was consumed;
- explicit fallback reason when the selected backend cannot run in the local
  smoke transport;
- aggregate-only hashes/counts, not individual participant ids or update
  values;
- DP accountant backend, clip/noise policy, sample rate, and epsilon spent for
  the successful round.

The local Phase 3 smoke uses the simulated secure-sum backend to exercise the
secure-aggregation output path. The masking backend remains the preferred
production backend, but local/HF runtimes without pairwise key routing and
dropout-recovery shares must report an explicit fallback rather than claiming a
masked secure-sum reveal.

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

The #227 reproducible local release-candidate smoke runs the coordinator service
and four `Phase3ParticipantAgent` simulated trust domains in one deterministic
no-GPU command:

```bash
uv run --extra dev python scripts/phase3_consortium_smoke.py \
  --rounds 10 \
  --run-dir runs/phase3-long-run-smoke \
  --output docs/evidence/phase3_long_run_smoke_report.json
```

Validate the checked-in report with:

```bash
uv run --extra dev python scripts/phase3_consortium_smoke.py \
  --validate docs/evidence/phase3_long_run_smoke_report.json
```

The report records the declared run shape before launch: four participants,
ten target rounds, inner horizon, tiny model size, root seed, DP policy,
secure-aggregation backend and threshold, eval budget reservation, and artifact
repo targets. Its dry-run section validates the manifest, dataset/probe
registry agreement, pinned public-probe hash, participant-agent preflight,
participant update release, local mount boundary, secure-aggregation threshold,
DP policy, and report publication path before the run closes rounds.

The checked-in smoke evidence is intentionally local and synthetic. It proves
the Phase 3 orchestration path, checkpoint/ledger/trace/report publication, and
aggregation/privacy accounting for a tiny LeWorldModel-flavour run. It is not a
published HF Jobs robotics result; #228, #229, and #230 own downstream
evaluation, observability, and release-bundle publication.

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

The #228 eval report is generated from the #227 long-run evidence:

```bash
uv run --extra dev python scripts/phase3_eval_report.py \
  --long-run-report docs/evidence/phase3_long_run_smoke_report.json \
  --output docs/evidence/phase3_eval_report.json
```

Validate the checked-in report with:

```bash
uv run --extra dev python scripts/phase3_eval_report.py \
  --validate docs/evidence/phase3_eval_report.json
```

The report declares the local consortium lifecycle eval target and the blocked
public SO-100 task-scale eval target before launch. Completed rows are limited
to the anchored-federation local smoke and are bound to checkpoint hash,
config hash, reconstructed run-manifest hash, task/env id, seed, planner
budget, and source-report hash. Missing local-only, naive-FedAvg, and Fork-A
controls are represented as explicit blocked rows with required-match criteria.

The model-card text generated by #228 must say that Phase 3 currently has
consortium-runtime evidence, not completed public robotics task performance.

## Observability And Dropout

Phase 3 consortium runs must be inspectable without widening the participant
data boundary. The #229 observability report is generated from the #227
long-run report, the #228 eval report, and a deterministic induced-dropout
smoke:

```bash
uv run --extra dev python scripts/phase3_observability_report.py \
  --long-run-report docs/evidence/phase3_long_run_smoke_report.json \
  --eval-report docs/evidence/phase3_eval_report.json \
  --run-dir runs/phase3-observability-smoke \
  --output docs/evidence/phase3_observability_report.json
```

Validate the checked-in report with:

```bash
uv run --extra dev python scripts/phase3_observability_report.py \
  --validate docs/evidence/phase3_observability_report.json
```

The report records participant lifecycle events, closed-round state summaries,
dropout decisions, retry budget and retry consumption, event-index timing,
released pseudo-gradient communication volume, aggregation/privacy mode,
artifact publication status, and metric cross-references back to run ids,
participant ids, config hashes, checkpoint hashes, and source report hashes.

The induced-dropout smoke drops one of four participant agents after assignment.
The remaining three participants satisfy the effective quorum, so the round
closes with `secure_sum` aggregation and DP accounting while the report records
that no retry was consumed. This is operational failure-mode evidence, not a
performance comparison.

The redaction contract is fail-closed: reports may contain hashes, counts,
participant ids, artifact URIs, and finite scalar metrics, but not raw data,
raw observations, raw actions, latents, embeddings, private action-head
weights, model tokens, secrets, or sensitive host-local paths. The #230 final
bundle must consume `docs/evidence/phase3_observability_report.json`; because
the report includes an induced dropout, no no-failure exception is needed.

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

The #230 final bundle generator materializes the run-specific consortium
manifest and dataset/probe registry from the #227 long-run evidence, then
aggregates all Phase 3 reports and the local final checkpoint header/weights:

```bash
uv run --extra dev python scripts/phase3_bundle.py \
  --output docs/evidence/phase3_evidence_bundle.json \
  --model-card-output docs/evidence/phase3_model_card.md \
  --manifest-output docs/evidence/phase3_long_run_manifest.json \
  --registry-output docs/evidence/phase3_long_run_dataset_registry.json
```

Validate the checked-in bundle/model-card pair with:

```bash
uv run --extra dev python scripts/phase3_bundle.py \
  --validate docs/evidence/phase3_evidence_bundle.json \
  --model-card docs/evidence/phase3_model_card.md
```

The generated bundle verifies the local manifest, dataset/probe registry,
long-run report, eval/control report, observability/dropout report,
privacy/aggregation rows embedded in the training report, run manifest, and
final checkpoint header/weights before writing a success bundle. The generated
model card distinguishes consortium-runtime evidence, tiny local
training/eval scale, completed and blocked controls, privacy controls, and
non-claims.

The checked-in Phase 3 bundle is a local-smoke release candidate. It declares
the target Hub model and dataset repositories, but does not claim paper-scale
SO-100 task performance or cryptographic honest-computation proof.

The generated model card, evidence bundle, Phase 3 reports, run manifest, and
final tiny checkpoint header/weights were also uploaded to the target model
repository at immutable revision
`f48176620987a763d2d38dfe09b70a71d88f6db0`. This publication is an artifact
mirror for the local-smoke release candidate; it does not change the
model-card claim boundary or unblock the public SO-100 task-scale eval rows.

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
