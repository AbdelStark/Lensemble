# Phase 3 Consortium Training Roadmap

Phase 3 is the historical operational consortium-training program for
Lensemble. It extended the Phase 2 path into a Tapestry-style multi-party run:
separate participant trust domains, governed membership, networked participant
agents, aggregation/DP mechanism reporting with explicit effectiveness status,
longer federated training, downstream evaluation, and a public audit bundle.
Its checked-in runs predate key correctness fixes; current acceptance status is
recorded below rather than inferred from the closed historical tracker.

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
- aggregation and DP mechanism plumbing with honest reports of whether the optimizer consumed a secure
  sum and whether the noise/accounting path is effective;
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
| [#226](https://github.com/AbdelStark/Lensemble/issues/226) | Secure aggregation and DP runtime controls | A consortium smoke run proves optimizer-path secure aggregation and effective cumulative DP, or records explicit blockers. The current report records both blockers. |
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

Phase 3 was intended to exercise secure aggregation and DP accounting as operational privacy controls.
The current runtime does not meet that privacy boundary:

- secure-aggregation mode and fallback policy are explicit in config;
- the coordinator receives individually visible released pseudo-gradients and commits from them;
- simulated/TEE sums are post-commit fixed-point equivalence checks
  (`secure_sum_consumed=false`), while masking records an explicit fallback;
- per-participant clipping and deterministic replay noise are applied before release, but the shared
  seed is reconstructible;
- each report creates a fresh one-round accountant after commit
  (`effective_dp=false`, `status=deterministic_replay_only`), so the reported epsilon is neither
  cumulative spend nor a budget-enforcement result;
- dropout threshold behavior is tested;
- visible individual updates and replayable noise are reported as privacy blockers.

The #226 runtime report lives in `lensemble.federation.phase3_privacy` and is
attached to `Phase3CoordinatorService` after a successful round close. It
records:

- selected secure-aggregation backend and threshold;
- whether an in-process secure sum was consumed (currently false);
- explicit fallback reason when the selected backend cannot run in the local
  smoke transport;
- aggregate-only hashes/counts, not individual participant ids or update
  values;
- DP accountant backend, clip/noise policy, sample rate, effective-DP status, and the independent
  one-round epsilon snapshot for the successful round.

The local Phase 3 smoke uses the simulated backend to cross-check, after commit, that its fixed-point
sum matches the coordinator's plaintext reduction. That value is not optimizer input and does not hide
individual updates. The masking backend remains the preferred production backend, but local/HF
runtimes without pairwise key routing and dropout-recovery shares report an explicit fallback rather
than claiming a masked secure-sum reveal.

No Phase 3 artifact should imply that aggregation correctness or participant
computation is cryptographically proven.

## Training Run Shape

> **Historical evidence notice.** Every checked-in Phase 3 training and gauge
> result in this section predates the correction of the outer-update direction
> and the `public-probe-v2` target-binding contract. The artifacts remain useful
> for protocol archaeology and failure analysis, but they do not validate the
> corrected runtime. Issue
> [#335](https://github.com/AbdelStark/Lensemble/issues/335) owns the clean rerun.

The target run should be large enough to demonstrate consortium operation while
remaining affordable and debuggable:

- at least four participants or simulated trust domains;
- at least ten closed federated rounds, unless a blocker records the maximum
  completed evidence;
- claim-mode LeWorldModel objective with `objective.target_stop_gradient=false`;
- public-probe frame anchoring enabled;
- aggregation and DP mechanisms configured where supported, with effectiveness status reported;
- published checkpoint, report, run manifest, and per-round metric artifacts.

Scaling model size should follow the dry-run evidence. A smaller model that
finishes with strong evidence is preferable to a larger run that cannot publish
complete artifacts.

The #227 reproducible local lifecycle smoke runs the coordinator service
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

The archived local no-GPU smoke records the Phase 3 orchestration path,
checkpoint/ledger/trace/report publication, and aggregation/privacy mechanism
reporting for a tiny synthetic LeWorldModel-flavour run. A separate historical
run executed a related path on real HF Jobs GPU compute. Neither run is current
correctness evidence after the outer-update and probe-contract fixes. The
[#243 headline run](https://huggingface.co/jobs/abdelstark/6a26885bece949d7b3dcb715)
ran on an `h200` HF Job from pinned commit `056f7407` and closed ten federated
rounds with four participants, all `0`-dropped, at `latent_dim=256` and
`num_tokens=196`. Each historical schema-v1 round recorded a `secure_sum` label and a one-round RDP
epsilon near `5.30` at `(δ=1e-5, noise_multiplier=1.0, clip_norm=0.5)`. Current readers conservatively
classify those fields as a post-commit equivalence cross-check and deterministic-replay-only accounting,
not optimizer-path confidentiality, composed ten-round epsilon, or budget enforcement. The
run produced final global hash
`bb31c0922de639cb9220c4cc5fc35d79aec719eb6fcedb09159bdff8cfb8fd43`, config hash
`27f2c77c9d47a7d053c01ab65f8d43aad79463b27d882f2d85ec28bc062cb2b2`, and
run-manifest SHA-256
`21819c9b936468ffc38f943b4ce13ec2ac150d328410f503fa73d9014e040c9d`.
Per-round `effective_rank` was ≈36–47 of 256, but that scale-invariant
diagnostic cannot validate a trajectory produced by the pre-fix optimizer. It
is neither a current training result nor a DP guarantee. The
checkpoint, manifests, ledger,
report, and pinned public probe (`f1053ffd…`) were published to
[`abdelstark/lensemble-phase3-consortium-checkpoint`](https://huggingface.co/abdelstark/lensemble-phase3-consortium-checkpoint)
at immutable revision `828e210cba4870b2be4ab573a5f0dd4ee30bae29`
(`publication.status: hf_jobs_release` in the historical artifact).

The training silos are published as the four participant trust domains plus a
held-out split in
[`abdelstark/lensemble-phase3-so100-silos`](https://huggingface.co/datasets/abdelstark/lensemble-phase3-so100-silos)
at revision `15f71911432b300dfdf41c998e27492e8c986be4`: `phase3-so100-silo{0..3}.h5`
(1284/1339/1261/1259 windows at `window_steps=4`, distinct Merkle roots) and the
held-out `phase3-so100-silo4.h5` (1216 windows). The long-run dataset registry at
[`docs/evidence/phase3_long_run_dataset_registry.json`](../evidence/phase3_long_run_dataset_registry.json)
is placeholder-free, with all four silos `published`.

## Evaluation And Controls

The controls below are hash-bound historical measurements. The eval producer
now records the lifecycle smoke under `consortium-runtime-smoke`—never
`anchored-federation`—and emits the anchored row only when the immutable
anchored report and run manifest are supplied. These rows do not validate the
corrected optimizer.

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

The eval report at
[`docs/evidence/phase3_eval_report.json`](../evidence/phase3_eval_report.json)
now records four completed matched controls and zero blocked controls. The
controls were run as separate `a10g-large` HF Jobs with configured noise off, six rounds, and
`latent_dim=256`, each published as its own probe checkpoint:

- anchored-probe
  [`abdelstark/lensemble-phase3-consortium-anchored-probe`](https://huggingface.co/abdelstark/lensemble-phase3-consortium-anchored-probe)
  @ `567755d2` — round-0 inter-participant `frame_drift` **48.97°**;
- naive-fedavg `…-naive-fedavg` @ `1aace225` — `frame_drift` **180°**;
- fork-a-frozen-encoder `…-fork-a` @ `148e4217` — `frame_drift` **0°** with
  `effective_rank` constant at 2.39 (the safe-degrade path);
- local-only `…-local-only` @ `a696da17` — per-participant `effective_rank`
  ≈120 with inter-participant `frame_drift` 180°.

The historical gauge observation: on real SO-100 data the old frame-anchor run reduced
inter-participant latent frame-drift at aggregation (anchored round-0 **48.97°**
versus naive **180°**), which is the RFC-0002 signal. Fork-A frozen-encoder is
the 0° safe-degrade, and local-only silos train healthily (`effective_rank`
≈120) but diverge maximally (180°) without a shared frame. This is the gauge
contrast from superseded runs, not a current benchmark or robotics
task-performance result.

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

The observability report at
[`docs/evidence/phase3_observability_report.json`](../evidence/phase3_observability_report.json)
is regenerated from the real headline run: 11 round summaries and one induced
dropout. The induced dropout drops one of four participant agents after
assignment; the remaining three satisfy the effective quorum
(`effective_quorum=3` of 4), so the round closes with historical `secure_sum` and DP-accounting labels
while the report records that no retry was consumed and the
redaction contract stays enforced. Under the current schema these labels mean a post-commit sum
cross-check and an independent one-round replay snapshot, not live secure aggregation or effective DP.
This is operational failure-mode evidence,
not a performance comparison.

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

Bundle generation must verify referenced artifacts exist before emitting an
artifact-integrity-valid bundle. The checked-in and published bundle must be
residency-safe; artifact integrity does not imply optimizer correctness.

The #230 final bundle generator materializes the run-specific consortium
manifest and dataset/probe registry from the #227 long-run evidence, then
aggregates all Phase 3 reports and the local final checkpoint header/weights:

```bash
uv run --extra dev python scripts/phase3_bundle.py \
  --output docs/evidence/phase3_evidence_bundle.json \
  --model-card-output docs/evidence/phase3_model_card.md \
  --manifest-output docs/evidence/phase3_long_run_smoke_manifest.json \
  --registry-output docs/evidence/phase3_long_run_smoke_dataset_registry.json
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
final checkpoint header/weights before writing an audit bundle. Validation
also binds the run-manifest and final checkpoint artifact-check SHA-256 values
to the hashes recorded in the training summary, so the checked-in report cannot
drift from the referenced release artifacts. Absolute local artifact paths are
rendered as residency-safe artifact URIs rather than host-local paths. The
generated model card distinguishes the synthetic lifecycle smoke, separately
hash-bound SO-100 controls, privacy mechanism plumbing, and explicit
non-claims.

The checked-in Phase 3 bundle at
[`docs/evidence/phase3_evidence_bundle.json`](../evidence/phase3_evidence_bundle.json)
and the model card at
[`docs/evidence/phase3_model_card.md`](../evidence/phase3_model_card.md) bind one
coherent historical smoke identity: run `phase3-long-run-smoke-v1`, probe
`ddc8349f…`, its exact checkpoint, and four separately sourced control runs.
The bundle records `publication.status: historical`, nine local artifact
checks, zero optimizer-consumed secure-sum rounds, and zero effective-DP
rounds. The associated SO-100 dataset revision is explicitly not bound to the
synthetic smoke.

The historical downstream eval report at
[`docs/evidence/phase3_downstream_eval_report.json`](../evidence/phase3_downstream_eval_report.json)
(also published to the checkpoint repo) records pre-fix held-out SO-100 latent
metrics. It is a non-usefulness audit: `effective_rank` ≈35.8/256 is
scale-invariant and blind to the approximately `7.5e-6` latent variance
recorded in the generated report. The associated central-ceiling audit did not
establish downstream usefulness. Neither observation validates the corrected
optimizer; [#335](https://github.com/AbdelStark/Lensemble/issues/335) owns the
rerun. Closed-loop physical task-success stays `blocked` with two specific
blockers: stable-worldmodel is unvendored
([#96](https://github.com/AbdelStark/Lensemble/issues/96), maintainer-gated), and
the checkpoint is not downstream-useful despite its proxy `val_pred`/`effective_rank`
scalars ([#244](https://github.com/AbdelStark/Lensemble/issues/244)).

The immutable revision `828e210cba4870b2be4ab573a5f0dd4ee30bae29` remains an
audit snapshot. It contains pre-correction artifacts and must not be presented
as the corrected runtime release. A replacement publication is blocked on
#335.

## Acceptance Matrix

The matrix distinguishes live software contracts from superseded empirical
results. “Historical” means the artifact is retained and validated as data, not
accepted as evidence for the corrected training loop.

| Gate | Current status |
|---|---|
| Membership and data declarations | Implemented and tested. The published SO-100 silo revision exists, but it is not bound to the bundled synthetic smoke. |
| Runtime lifecycle | Implemented and tested for participant admission, round closure/abort, dropout, and artifact emission. A new multi-operator run is still pending. |
| Residency and redaction | Implemented and tested on generated reports. |
| Optimizer-path secure aggregation | Blocked. The coordinator consumes plaintext released updates; simulated/TEE sums are post-commit cross-checks. |
| Effective cumulative DP | Blocked. Historical noise is deterministically replayable and accounting is recreated per round. |
| Corrected multi-round training | Blocked on [#335](https://github.com/AbdelStark/Lensemble/issues/335). All checked-in Phase 3 metrics predate the outer-update and probe-contract fixes. |
| Matched controls | Historical and hash-bound. They are representation-gauge rows, not current benchmark or robotics results. |
| Closed-loop SO-100 usefulness | Blocked on [#96](https://github.com/AbdelStark/Lensemble/issues/96) and a non-collapsing corrected checkpoint. |
| Replacement release | Blocked on the corrected rerun. Revision `828e210c…` remains an immutable audit snapshot. |

## Final Claim Boundary

The repository currently supports this statement:

> Lensemble implements and tests a federated JEPA research runtime with
> participant-local training, typed update contracts, deterministic round
> lifecycle, gauge diagnostics/alignment, artifact binding, and explicit
> privacy-effectiveness reporting. Its preserved Phase 3 runs are historical
> failure-analysis artifacts pending a clean rerun of the corrected loop.

It does not claim:

- that the historical Phase 3 metrics validate the corrected optimizer;
- optimizer-path secure aggregation or effective cumulative DP;
- cryptographic proof of participant computation;
- provenance-ledger-backed contribution accounting;
- closed-loop physical SO-100 success;
- broad robotics generalization;
- paper-scale LeWorldModel performance.
