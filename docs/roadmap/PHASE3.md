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

The local no-GPU smoke proves the Phase 3 orchestration path,
checkpoint/ledger/trace/report publication, and aggregation/privacy accounting
for a tiny LeWorldModel-flavour run. The headline Phase 3 run then executed this
same path at training scale on real HF Jobs GPU compute. The
[#243 headline run](https://huggingface.co/jobs/abdelstark/6a26885bece949d7b3dcb715)
ran on an `h200` HF Job from pinned commit `056f7407` and closed ten federated
rounds with four participants, all `0`-dropped, at `latent_dim=256` and
`num_tokens=196`. Each round recorded `secure_sum` secure aggregation and DP
accounting at `(ε≈5.30, δ=1e-5, rdp, noise_multiplier=1.0, clip_norm=0.5)`. The
run produced final global hash
`bb31c0922de639cb9220c4cc5fc35d79aec719eb6fcedb09159bdff8cfb8fd43`, config hash
`27f2c77c9d47a7d053c01ab65f8d43aad79463b27d882f2d85ec28bc062cb2b2`, and
run-manifest SHA-256
`21819c9b936468ffc38f943b4ce13ec2ac150d328410f503fa73d9014e040c9d`. Per-round
`effective_rank` stays high (≈36–47 of 256): the public-probe frame anchor holds
representational rank under DP federation. The checkpoint, manifests, ledger,
report, and pinned public probe (`f1053ffd…`) were published to
[`abdelstark/lensemble-phase3-consortium-checkpoint`](https://huggingface.co/abdelstark/lensemble-phase3-consortium-checkpoint)
at immutable revision `828e210cba4870b2be4ab573a5f0dd4ee30bae29`
(`publication.status: hf_jobs_release`).

The training silos are published as the four participant trust domains plus a
held-out split in
[`abdelstark/lensemble-phase3-so100-silos`](https://huggingface.co/datasets/abdelstark/lensemble-phase3-so100-silos)
at revision `15f71911432b300dfdf41c998e27492e8c986be4`: `phase3-so100-silo{0..3}.h5`
(1284/1339/1261/1259 windows at `window_steps=4`, distinct Merkle roots) and the
held-out `phase3-so100-silo4.h5` (1216 windows). The long-run dataset registry at
[`docs/evidence/phase3_long_run_dataset_registry.json`](../evidence/phase3_long_run_dataset_registry.json)
is placeholder-free, with all four silos `published`.

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

The eval report at
[`docs/evidence/phase3_eval_report.json`](../evidence/phase3_eval_report.json)
now records four completed matched controls and zero blocked controls. The
controls were run as separate `a10g-large` HF Jobs with DP off, six rounds, and
`latent_dim=256`, each published as its own probe checkpoint:

- anchored-probe
  [`abdelstark/lensemble-phase3-consortium-anchored-probe`](https://huggingface.co/abdelstark/lensemble-phase3-consortium-anchored-probe)
  @ `567755d2` — round-0 inter-participant `frame_drift` **48.97°**;
- naive-fedavg `…-naive-fedavg` @ `1aace225` — `frame_drift` **180°**;
- fork-a-frozen-encoder `…-fork-a` @ `148e4217` — `frame_drift` **0°** with
  `effective_rank` constant at 2.39 (the safe-degrade path);
- local-only `…-local-only` @ `a696da17` — per-participant `effective_rank`
  ≈120 with inter-participant `frame_drift` 180°.

The honest gauge finding: on real SO-100 data the frame anchor reduces
inter-participant latent frame-drift at aggregation (anchored round-0 **48.97°**
versus naive **180°**), which is the RFC-0002 signal. Fork-A frozen-encoder is
the 0° safe-degrade, and local-only silos train healthily (`effective_rank`
≈120) but diverge maximally (180°) without a shared frame. This is the gauge
contrast on real data, not a public robotics task-performance result; the
DP–utility limits are stated in the Final Claim Boundary.

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
(`effective_quorum=3` of 4), so the round closes with `secure_sum` aggregation
and DP accounting while the report records that no retry was consumed and the
redaction contract stays enforced. This is operational failure-mode evidence,
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
final checkpoint header/weights before writing a success bundle. Validation
also binds the run-manifest and final checkpoint artifact-check SHA-256 values
to the hashes recorded in the training summary, so the checked-in report cannot
drift from the referenced release artifacts. Absolute local artifact paths are
rendered as residency-safe artifact URIs rather than host-local paths. The
generated model card distinguishes consortium-runtime evidence, the real HF Jobs
training/eval scale, completed and blocked controls, privacy controls, and
non-claims.

The checked-in Phase 3 bundle at
[`docs/evidence/phase3_evidence_bundle.json`](../evidence/phase3_evidence_bundle.json)
and the model card at
[`docs/evidence/phase3_model_card.md`](../evidence/phase3_model_card.md) now
reflect the real HF Jobs run, not a local smoke. The bundle records
`publication.status: published` at revision `828e210c`, four completed controls
with zero blocked controls, and nine artifact checks all with `exists:true`. It
declares the target Hub model and dataset repositories, but does not claim
paper-scale SO-100 task performance or cryptographic honest-computation proof.

The downstream eval report at
[`docs/evidence/phase3_downstream_eval_report.json`](../evidence/phase3_downstream_eval_report.json)
(also published to the checkpoint repo) records real held-out SO-100 latent
metrics, but it is corrected as a non-usefulness surface: `effective_rank`
≈35.8/256 is scale-invariant and blind to held-out magnitude collapse
(`~7.5e-6` latent variance; `thoughts/collapse_fix_probe.py`). The central
ceiling probe (`thoughts/central_ceiling_probe.py`) shows the SO-100 checkpoint
does not clear a downstream usefulness ceiling. Closed-loop physical task-success
stays `blocked` with two specific blockers: stable-worldmodel is unvendored
([#96](https://github.com/AbdelStark/Lensemble/issues/96), maintainer-gated), and
the checkpoint is not downstream-useful despite its proxy `val_pred`/`effective_rank`
scalars ([#244](https://github.com/AbdelStark/Lensemble/issues/244)).

The generated model card, evidence bundle, Phase 3 reports, run manifest, and
final checkpoint header/weights are published to the target model repository at
immutable revision `828e210cba4870b2be4ab573a5f0dd4ee30bae29`
(`publication.status: hf_jobs_release`). This is the real-run release, not an
artifact mirror; it does not change the model-card claim boundary or unblock the
closed-loop SO-100 task-success rows, which remain documented blockers.

## Acceptance Matrix

Every gate is met by the real HF Jobs run, with two honest residual blockers
recorded under Evaluation (closed-loop SO-100 task-success and sustained
non-collapsing federated training; see the Final Claim Boundary).

| Gate | Minimum passing evidence | Real-run status |
|---|---|---|
| Membership | Four participant declarations validate against one consortium manifest. | Met. Four `published` silos in [`abdelstark/lensemble-phase3-so100-silos`](https://huggingface.co/datasets/abdelstark/lensemble-phase3-so100-silos) @ `15f71911` with distinct Merkle roots; registry placeholder-free. |
| Runtime | Independent participant-agent processes complete a multi-round run through a networked coordinator. | Met. [#243 `h200` job](https://huggingface.co/jobs/abdelstark/6a26885bece949d7b3dcb715) closed ten federated rounds with four participants, all `0`-dropped, from pinned commit `056f7407`. |
| Residency | Redaction tests and reports show no raw observations, raw actions, latents, embeddings, private action-head weights, tokens, or secrets cross boundaries. | Met. Redaction contract enforced across the regenerated observability report (11 round summaries, one induced dropout). |
| Aggregation/privacy | Secure aggregation and DP accounting are exercised, or fallback/blockers are recorded in report and model card. | Met. Per-round `secure_sum` + DP `(ε≈5.30, δ=1e-5, rdp, noise_multiplier=1.0, clip_norm=0.5)`. |
| Training | At least ten closed rounds, or an explicit blocker with the maximum completed closed-round evidence. | Met. Ten closed rounds at `latent_dim=256`, `num_tokens=196`; final global hash `bb31c092…`, run-manifest SHA-256 `21819c9b…`; per-round `effective_rank` ≈36–47/256. |
| Evaluation | A downstream eval report goes beyond the Phase 2 toy boundary where feasible, with blocked rows for unavailable controls. | Corrected. The SO-100 report records real held-out proxy metrics, but `effective_rank` is scale-invariant, `skill_vs_identity` is gameable, and held-out magnitude collapse (`~7.5e-6`; `thoughts/collapse_fix_probe.py`) plus the central ceiling probe (`thoughts/central_ceiling_probe.py`) prevent a usefulness claim; closed-loop task-success stays an explicit blocker ([#96](https://github.com/AbdelStark/Lensemble/issues/96), [#244](https://github.com/AbdelStark/Lensemble/issues/244)). |
| Observability | Participant lifecycle, dropout/retry, timing, communication, and artifact-publication status are reported. | Met. One induced dropout closes with `effective_quorum=3` of 4 and no retry consumed. |
| Release | Checkpoint repo publishes final checkpoint, reports, evidence bundle, and model card at immutable revisions. | Met. [`abdelstark/lensemble-phase3-consortium-checkpoint`](https://huggingface.co/abdelstark/lensemble-phase3-consortium-checkpoint) @ `828e210c` (`publication.status: hf_jobs_release`); bundle `published` with nine artifact checks `exists:true`. |
| Claims | Public text says this is consortium-engineering evidence, not paper-scale performance or cryptographic honest-computation proof. | Met. See the Final Claim Boundary below. |

## Final Claim Boundary

Phase 3 is completed on real HF Jobs GPU compute, supporting this statement:

> Lensemble completed a governed, multi-party consortium training run for a
> federated JEPA / LeWorldModel-flavour world model across participant-local
> SO-100 data, on real HF Jobs GPU compute (an `h200` headline run of ten
> closed federated rounds at `latent_dim=256`), with independent participant
> agents, secure aggregation and DP controls, matched DP-off control probes,
> published checkpoints and datasets at immutable revisions, downstream latent
> evaluation beyond the toy boundary, lifecycle reporting, and a conservative
> evidence bundle.

This is consortium-engineering plus real federated-training evidence. It is
**not** a cryptographic proof of honest participant computation — RFC-0006
remains explicitly out of scope — and not a paper-scale robotics performance
claim. The following limitations are stated as-is and must not be softened:

- **DP–utility frontier**: at four participants × ~8.4M parameters the
  meaningful-DP regime (`noise_multiplier=1.0`, ε≈5.3) is
  gradient-noise-dominated — `val_pred` grows and `frame_drift` saturates over
  rounds — so the gauge contrast rests on the round-0 measurement and the DP-off
  control probes rather than the full DP trajectory.
- **Federated gauge failure — narrowed by the #259 MVP, not a usefulness closeout**: at the default outer step
  (`outer_lr=0.7`) with the #249 weak anchor (`lambda_anc=0.01` re-snapshotting
  the drifting global) the federated global representation collapsed over rounds
  (`effective_rank`→1, `val_pred`→2×10⁵, drift 180°). The MVP M1 fixes
  ([#259](https://github.com/AbdelStark/Lensemble/issues/259): anchor pinned to
  the **fixed** round-0 reference, live Procrustes backstop on the encoder
  terminal frame + predictor, tamed outer step) reduce the naive-FedAvg gauge
  failure — the anchored run holds and grows `effective_rank` (2.6→14.8 over 12
  rounds) with controlled drift and 4 orders of magnitude better proxy `val_pred`
  than naive-FedAvg. This does not prove downstream usefulness: the held-out
  representation has magnitude collapse (`~7.5e-6` latent variance;
  `thoughts/collapse_fix_probe.py`), the central ceiling probe
  (`thoughts/central_ceiling_probe.py`) does not clear, `skill_vs_identity` is
  gameable, and `effective_rank` is scale-invariant. In plain text:
  skill_vs_identity is gameable; effective_rank is scale-invariant. See the corrected MVP
  Benchmarks / Results section in the README and
  `docs/evidence/phase3_mvp_benchmark_report.json`
  (`abdelstark/lensemble-phase3-converged-checkpoint` @ `a6f5a961…`). The
  dynamic-env RFC-0017 pivot is the binding ground-truth usefulness path.
- **Downstream**: closed-loop physical SO-100 task-success remains blocked
  pending stable-worldmodel
  ([#96](https://github.com/AbdelStark/Lensemble/issues/96)).

It does not claim:

- cryptographic proof of participant computation;
- provenance-ledger-backed contribution accounting;
- broad robotics generalization;
- paper-scale LeWorldModel performance.
