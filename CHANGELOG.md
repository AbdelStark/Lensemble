# Changelog

All notable changes to Lensemble are recorded here. The format is
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (pre-1.0 the minor tracks the milestone;
[conventions §10](docs/spec/conventions.md#10-versioning-and-schema-policy)).

Only these six category headings are permitted under a version: **Added**, **Changed**, **Deprecated**,
**Removed**, **Fixed**, **Security**. Entries are imperative and reference the affected symbol or
`area:*` label and, where applicable, the RFC and the invariant `INV-*` id. A change to a versioned
invariant's enforcement point or a `schema_version` names both the invariant and the version bump. The
`Security` category is reserved for fixes to residency, secrets handling, or supply-chain issues
([06 — Security](docs/spec/06-security.md)).

At release the maintainer retitles `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` and opens a fresh
`## [Unreleased]` block ([09 §3](docs/spec/09-release-and-versioning.md#3-changelog-discipline)).

## [Unreleased]

### Added

- `area:eval`/`area:federation`: closed the LeWM federated-demo credibility gap (epic #332). The
  headline probe number is now produced by the **system the demo ships**: `scripts/lewm_system_probe.py`
  (core in `lensemble.eval.lewm_system_probe`) trains real adapter deltas in node and drives them
  through `FederatedDemoService.submit_update` and `_close_round_lewm` (real validation + real
  deterministic-mean aggregation + hash-chained revisions), then probes the **server-produced** final
  `modelRevisionId` (#327, `docs/evidence/lewm_tworooms_system_probe.json`). The offline
  `scripts/lewm_probe_check.py` is relabelled the math cross-check.
- `area:eval`: the before/after probe now computes held-out collapse diagnostics — latent std,
  effective rank, and SIGReg on the **validation** pairs for baseline AND adapted — and overrides a
  naive "improved" MSE to `collapse-risk` when the adapted held-out std/rank materially drops below
  the frozen baseline (the #259 blind spot). Surfaced in the dashboard probe and the evidence export,
  with a node selftest asserting a magnitude-collapsed adapter is flagged even when its MSE improves
  (#328).
- `area:eval`/`area:ci`: `scripts/lewm_probe_seedsweep.py` runs the system-composed probe across
  multiple seeds and reports the distribution and worst case (`docs/evidence/lewm_tworooms_probe_seedsweep.json`);
  the demo card cites the worst draw, not the best. A dataset-free unit test exercises the composed
  path on every CI run and a dataset-gated test regenerates the headline within tolerance to prevent
  silent drift (#330).

### Changed

- `area:docs`/`area:observability`: the real-mode demo UI now renders the strict claim boundary
  in-app (scoped chips + single-coordinator / mean-of-clipped-deltas note sourced from the run
  snapshot), reworded the H1 to "Federate adapter updates on a real world-model checkpoint", and
  labelled the probe badge as a held-out probe (not a benchmark) so a screenshot can no longer read
  as full-model federated training (#329).

### Removed

- `area:federation`: removed the off-spec `federation/outer.py`; `OuterOptimizer`,
  `assert_bitwise_reproducible`, and the `INV-AGG-DETERMINISM` content-hash helper now live in the
  spec-mandated `federation/outer_optimizer.py` ([01-architecture](docs/spec/01-architecture.md),
  [RFC-0013](docs/rfcs/RFC-0013-coordinator-runtime.md)). The `lensemble.federation` public
  re-exports are unchanged.
- `area:data`: removed the empty `data/loaders.py` scaffold stub; the fixed-`num_steps` `Window`
  loader (RFC-0004) is provided by `data.dataset.EpisodeDataset`. Dropped from the `data/` file list
  in [01-architecture](docs/spec/01-architecture.md) and [conventions §1](docs/spec/conventions.md).

### Changed

- `area:artifacts`: consolidated six duplicate file-SHA-256 helpers into one chunked
  `artifacts.hashing.sha256_file`; the Phase 3 evidence/bundle paths now stream large checkpoint
  weight files instead of reading them whole into memory (`INV-CHECKPOINT-HASH` output unchanged).
- `area:federation`: Require Phase 3 evidence-bundle validation to bind the
  run-manifest, final checkpoint header, and final checkpoint weights artifact
  checks to the same SHA-256 hashes recorded in the training summary, while
  redacting absolute local artifact paths to residency-safe artifact URIs.

### Security

- Dependencies: replaced the `lance>=0.10` runtime dependency with **`pylance>=0.10`** (the real Lance
  columnar library; import name `lance`). The bare `lance` distribution on PyPI is an unrelated typosquat
  (`lance==1.2.1`, an empty package) that the prior pin resolved to, so the default-format data backend
  could not have worked from a clean install. Updated `pyproject.toml`, `tests/unit/test_packaging.py`,
  and `conventions §11` in lockstep (#22).

### Changed

- `area:docs`: **Frontend polish pass: humanized copy, no em/en dashes, motion and interaction
  refinements** — every user-facing string in the demo app rewritten in plain product voice
  (participant stage texts, empty states, probe panel, TwoRooms lab status/identity lines, home
  hero and steps, runtime error messages) with em/en dashes removed from all visible text. Visual
  layer gains a gradient hero heading, panel rise-in animation (disabled under
  prefers-reduced-motion), hover elevation on cards/charts/slots, timeline row hover, input hover
  states, accent text selection, and a tidier TwoRooms lab (simpler headings, humanized runtime
  and model identity lines). Pinned test needles preserved ("never leave this browser",
  "fails visibly", "did not beat the parent checkpoint", "reported, not hidden").

- `area:observability`: **Host dashboard analytics: charts, status strip, probe placement** —
  `web/federated-demo/charts.mjs` adds dependency-free SVG line charts (pure node-tested data
  preparation: per-participant loss series with surrogate fallback, round-metric series with
  gaps preserved, nice ticks, compact tick labels). The dashboard gains a status strip (round
  progress bar, active participants, latest pred loss/SIGReg/effective rank/adapter norms,
  latest health flags) and a "Run analytics" grid: prediction loss per participant (+dashed
  mean), round-mean loss start-vs-trained, SIGReg statistic, latent geometry (effective rank +
  std), and adapter Δ/state norms — all real run data over WebSocket. The right column is
  reordered so the before/after probe sits directly under the analytics, above diagnostics,
  round metrics, artifacts, and the event timeline (order pinned by
  `test_dashboard_mounts_analytics_above_probe_and_timeline`).

- `area:docs`: **Demo UI redesign: real-LeWM only, premium theme, product-voice copy** — the web
  app now exposes only the Tapestry-like `real-lewm-tworooms` path (the surrogate learner and the
  frontend simulator remain in the codebase/API for tests, but the host form hardcodes the real
  mode and the simulator entry point is gone). The disclaimer banners and long claim-boundary
  notes moved out of the UI into the footer link to the demo card; the binding claim boundary
  stays in the server constants, every evidence export, and the claim audit (unchanged and still
  gated). New visual system in `web/federated-demo/style.css` (Inter/JetBrains Mono, soft
  elevation, pill badges, accent gradient buttons, refined timeline/metric tiles), a new home
  hero with a three-step explainer, compact run-header chips, and a real-mode inference pointer
  replacing the swipe-dot inference panel on real runs. Tests updated to pin the new contract
  (`test_lewm_demo_integration.py`, `test_federated_demo_app.py`); all node selftests unchanged
  and green.

### Added

- `area:docs`: **End-to-end Tapestry-like rehearsal gate + researcher runbook (gate G7)** (#324,
  epic #314, closes the epic's active queue) — `scripts/lewm_demo_rehearsal.py` exercises the
  real-mode service contract end to end: a two-participant smoke run (auto + manual mix), a
  four-participant run with mid-round dropout (quorum preserved), a WebSocket-style
  reconnect (`connection.closed`→`connection.opened`, reconnectCount), stale-round rejection,
  adapter-state retrieval after aggregation, configurable long runs (validated at 12 rounds),
  evidence export, and a clean fail-closed claim audit on every run; it auto-binds the real
  export manifest when present. `docs/roadmap/TAPESTRY_LEWM_RUNBOOK.md` is the researcher-facing
  runbook: automated gates, the two-browser and four-phone paths, what the metrics prove, the
  narration script, the never-claim list, and the rule that a non-improving probe verdict blocks
  public positive claims. `tests/ml/test_lewm_rehearsal.py` runs the gate in-process and as a
  command and pins the runbook contract.

- `area:docs`: **Tapestry-like LeWM evidence card + claim audit (gate G7 surface)** (#323, epic
  #314) — `docs/evidence/lewm_tworooms_demo_card.md` is the public evidence surface: the accepted
  claim, the what-was-real table linking every gate artifact (G1 manifest/reference, G2
  export/action-stats/real-data check, G3 overfit, G5 federated probe), the privacy/residency
  statement, and the non-claims — including that the first probe configuration was honestly flat.
  `lensemble.demo.evidence_audit` validates real-mode `demo-evidence/1` exports fail-closed:
  required Tapestry-like claim-boundary and non-claim phrases, pinned 40-hex checkpoint revision,
  64-hex weights/graph/revision/update hashes, parent-revision chaining, per-round health flags
  and metric summaries, honest privacy status, no raw participant data/adapter tensors/tokens,
  and no unnegated overclaims (`tests/ml/test_lewm_evidence_audit.py`: clean run passes, 14
  targeted mutations rejected). Linked from README, the pivot contract gate table, and the
  mkdocs Evidence nav.

- `area:eval`: **Tapestry-like LeWM before/after probe, anti-collapse diagnostics, and honest
  failure states (gate G5)** (#322, epic #314) — `web/federated-demo/lewm_probe.mjs` scores any
  global adapter revision against the identity baseline on a FIXED seeded TwoRooms validation
  task; verdicts are `improved`/`flat`/`worse` and the host dashboard probe panel reports
  negative results explicitly ("reported, not hidden"). The adapter protocol gained the
  shared-init + offset semantics this requires (`adapterInitSeed` in the run binding; every
  participant trains from the SAME deterministic init plus the current global offset, fetched
  from the model-revision endpoint — without this, cross-participant delta means were
  incoherent). Round metrics and evidence now carry server-side `healthFlags` mirroring the JS
  assessment (effective-rank collapse, latent-std magnitude collapse, flat/worsened loss,
  SIGReg outliers). Real-data result (`scripts/lewm_probe_check.py` →
  `docs/evidence/lewm_tworooms_probe_check.json`): the frozen predictor's residual is
  systematically biased (bias cosine 0.97 across disjoint episodes), and 3 rounds × 2
  participants of the shipping federation math improve held-out validation MSE 0.0604 → 0.0530
  (**+12.3%**, verdict `improved`); the earlier flat result at 60 steps/4 episodes is what set
  the validated auto-round defaults (more resident pairs, fewer optimizer steps).

- `area:federation`: **real-lewm mode integrated into the federated flow with autonomous
  participants** (#321, epic #314) — the host form gains a learner-path selector
  (`surrogate-swipe-dot` default vs `real-lewm-tworooms`; the frontend simulator refuses the real
  path). Real-mode participants run `web/federated-demo/lewm_participant.mjs` autonomously by
  default (manual/debug mode preserved): load the hash-verified runtime once (cached across
  rounds, adapter sized from the run binding's `adapterHiddenDim`), collect resident rollouts,
  train, and submit the bounded delta — failures surface visibly with no surrogate fallback and
  no artificial sleeps. The host dashboard shows the run mode badge, checkpoint binding, and
  real per-round metrics (pred loss first→last, SIGReg statistic, effective rank, latent std,
  delta/state norms, loss-decreased count); participant rows show per-browser adapter metrics.
  WebSocket reactivity, reconnect, and rate-limit defaults are unchanged (real mode reuses the
  existing event plumbing). Tests: `tests/ml/test_lewm_demo_integration.py` + 3 new node
  selftests (autonomous round, visible-failure path, mode-mismatch rejection).

- `area:federation`: **lewm-adapter-delta/1 schema, validation, aggregation, and privacy honesty
  (gate G6)** (#320, epic #314) — the demo coordinator gains a run-level mode
  (`surrogate-swipe-dot` default vs `real-lewm-tworooms`); real runs require the
  `lewm-browser-export/1` manifest binding (fail-closed `real_mode_unavailable`), accept ONLY
  `lewm-adapter-delta/1` artifacts (clipped flattened adapter delta + adapter spec +
  checkpoint/export-hash binding + metric summary + participant mode), and reject invalid, stale,
  oversized, raw-data-like (forbidden keys now include frame/pixel/rollout and any `raw*`),
  shape-mismatched, checkpoint-mismatched, non-finite, fabricated-metric, over-norm, and replayed
  updates. Aggregation is a deterministic per-round mean folded into a cumulative global adapter
  state (`lewmrev-*` revisions bound to parent checkpoint revision, weights hash, graph hashes,
  spec, and delta hashes; full parameters served by the model-revision endpoint, never in
  snapshots/evidence). Evidence and revisions record secure-aggregation/DP status as
  `absent-in-demo-path` instead of implying protection. Client side:
  `web/federated-demo/lewm_delta_artifact.mjs` builds the bounded artifact (WebCrypto sha256).
  Server wiring: `lensemble demo federated --lewm-manifest` (auto-detects the export dir).
  Tests: `tests/ml/test_lewm_federation.py` (27 cases incl. 17 rejection paths).

- `area:model`: **Browser-local Tapestry-like LeWM adapter continuation + SIGReg diagnostics
  (gate G3)** (#319, epic #314) — `web/federated-demo/lewm_adapter.mjs` trains a bounded
  zero-init residual adapter (12,512 params) on the FROZEN exported predictor's outputs with
  manual-gradient Adam, global-norm clipping, a variance-floor anti-collapse surrogate (gradient
  in closed form), and real diagnostics: per-dim latent std, participation-ratio effective rank,
  and an exact JS port of the `lensemble.model.sigreg` Epps–Pulley statistic (torch parity
  ≤ 2e-7 rel, `tests/ml/test_lewm_adapter.py`). `lewm_local_trainer.mjs` collects resident
  TwoRooms rollouts, builds (frozen-prediction → next-latent) pairs through the exported graphs,
  and emits the bounded clipped delta + honest metric summary; raw frames/actions/latents never
  leave the browser and no sleeps fake progress. One-browser overfit gate on REAL expert-episode
  latents through the shipping JS trainer (`scripts/lewm_adapter_overfit_check.py` →
  `docs/evidence/lewm_tworooms_adapter_overfit.json`): pred loss 0.0559 → 0.00061
  (98.9% improvement, 300 steps, 30 pairs), delta clipped at 3.0. JS selftests include an
  analytic-vs-numerical gradient check (`web/federated-demo/lewm_adapter_selftest.mjs`).

- `area:model`: **TwoRooms browser environment + real LeWM rollout/planning** (#318, epic #314) —
  `web/federated-demo/tworooms_env.mjs` is a deterministic JS port of the upstream TwoRoom env at
  the released default variations, validated pixel-level against the official expert dataset
  (max abs diff 1.0/255 vs real frames; geometry/door/border/collision/expert-policy selftests).
  `web/federated-demo/lewm_runtime.mjs` loads the hash-verified exported graphs through ONNX
  Runtime Web (WebGPU→WASM; explicit `LewmUnsupportedError`, no silent surrogate fallback) and
  implements windowed autoregressive rollout + CEM goal-latent planning with real history action
  conditioning. `web/federated-demo/tworooms_panel.mjs` (+ `#/tworooms` route) shows current/goal
  frames, candidate rollout costs, the chosen block, latent distance, model identity, and
  deviations. Export graphs now bake input normalization in (ImageNet pixels; expert-dataset
  action z-score from `docs/evidence/lewm_tworooms_action_stats.json` @ dataset revision
  `6903a2d`). End-to-end real-data probe (`scripts/lewm_tworooms_realdata_check.py` →
  `docs/evidence/lewm_tworooms_realdata_check.json`): exported-pipeline next-latent MSE 0.070 vs
  copy-last baseline 1.306 (ratio 0.054) over 24 held-out expert episodes. Node selftests:
  `web/federated-demo/tworooms_selftest.mjs` via `tests/ml/test_lewm_tworooms_browser.py`.

- `area:model`: **Checkpoint-backed LeWM browser inference graphs (gate G2)** (#317, epic #314) —
  `lensemble.model.lewm_export` exports the TwoRooms encoder (+BatchNorm projector), action
  encoder, and AdaLN predictor (+pred_proj) as single-file ONNX graphs for ONNX Runtime Web
  (dynamic batch/time axes; short-history predictor calls T∈{1,2,3} parity-checked).
  `scripts/lewm_tworooms_export.py` regenerates them deterministically from the pinned checkpoint
  into gitignored `web/federated-demo/model/lewm-tworooms/` and emits the hash-bound
  `lewm-browser-export/1` manifest
  (`docs/evidence/lewm_tworooms_browser_export_manifest.json`; 73.5 MB total, opset 18,
  PyTorch-vs-onnxruntime max abs diff 4.9e-5 ≤ 1e-4). onnx/onnxscript/onnxruntime stay
  export-time-only deps; `tests/ml/test_lewm_export.py` skips them on the blocking CPU gates and
  rejects tampered graphs and skipped-parity manifests in claim-grade evidence.

- `area:model`: **TwoRooms LeWM checkpoint ingestion + PyTorch parity (gate G1)** (#316, epic #314) —
  `lensemble.model.lewm_tworooms` reconstructs the released `quentinll/lewm-tworooms` module tree
  in plain torch, state-dict-key compatible with `weights.pt` (HF-ViT-tiny encoder schema +
  upstream `stable_worldmodel.wm.lewm` predictor/action-encoder/BatchNorm projection heads;
  303/303 tensors strict-load, 18,034,478 params). `lensemble.model.lewm_checkpoint` resolves the
  pinned revision (`77adaae…`; claim-grade refuses unpinned revisions), fails closed on mismatched
  config/missing weights/unknown tensor names, and emits `lewm-checkpoint-manifest/1` +
  deterministic reference-forward report (`scripts/lewm_tworooms_ingest.py` →
  `docs/evidence/lewm_tworooms_checkpoint_manifest.json`,
  `docs/evidence/lewm_tworooms_reference_report.json`). Parity vs the actual upstream
  implementation (transformers 4.49 `ViTModel` + upstream `lewm` modules) measured at ingest:
  max abs diff ≤ 2e-5 across encode/action/predict/rollout paths
  (`tests/ml/test_lewm_tworooms.py`).

- `area:docs`: **Tapestry-like LeWM TwoRooms pivot contract** (#315, epic #314) —
  `docs/roadmap/TAPESTRY_LEWM.md` locks the real-checkpoint browser-federation pivot before
  implementation widens: the accepted claim and non-claims, the `surrogate-swipe-dot` vs
  `real-lewm-tworooms` product modes, the pinned external artifacts
  (`quentinll/lewm-tworooms` @ `77adaae`, `lucas-maes/le-wm`, `stable_worldmodel.wm.lewm`),
  the issue-by-issue implementation sequence (#316–#324), and the binding evidence gates
  G1–G7. Linked from README, SPEC scope, BROWSER_FEDERATED_DEMO.md, and the mkdocs nav.

- `area:docs`: **Phase 3 completion roadmap + acceptance-matrix closeout** (#248, RFC-0009) — closes
  epic #249. `docs/roadmap/PHASE3.md` (Training Run Shape, Evaluation And Controls, Observability And
  Dropout, Evidence Bundle, Acceptance Matrix, Final Claim Boundary) and the README Implementation Status
  now record Phase 3 as completed on **real HF Jobs `h200` compute**: the 10-round anchored-federation
  run (job `6a26885b…`, checkpoint `abdelstark/lensemble-phase3-consortium-checkpoint` @ `828e210c`,
  `publication.status: hf_jobs_release`), the published SO-100 silos @ `15f71911`, the four completed
  matched controls (0 blocked) with the round-0 anchored-vs-naive frame-drift contrast (48.97° vs 180°),
  the real induced dropout, and the published evidence bundle/model card — with the honest residual
  limitations stated (DP–utility frontier, federated collapse at the default outer-step + unvendored
  V-JEPA-2 warm-start, and the closed-loop downstream task-success blocker on stable-worldmodel #96). The
  evidence is consortium-engineering + real federated-training evidence, **not** a cryptographic proof of
  honest participant computation (RFC-0006 out of scope).
- `area:artifacts`: **Phase 3 final evidence bundle + model card regenerated from the REAL #242 consortium
  run** (#247) — `scripts/phase3_bundle.py` gains `--consortium-manifest`/`--dataset-registry` input options
  that load the checked-in real `docs/evidence/phase3_consortium_manifest.json` +
  `phase3_long_run_dataset_registry.json` (validated via `validate_phase3_registry_against_manifest`),
  copy them to the manifest/registry outputs, and bundle them directly — bypassing the local-smoke
  `materialize_phase3_run_contracts` path (which stays the default for the fresh-smoke generator). The
  checked-in `docs/evidence/phase3_evidence_bundle.json` + `phase3_model_card.md` now carry the REAL run
  identity `lensemble-phase3-consortium` / `phase3-consortium-v1` (was the local smoke), `publication.status
  published` @ model revision `828e210c…` / dataset revision `15f71911…`, 10/10 closed rounds with final
  global hash `bb31c092…`, all four completed controls (`anchored-federation`, `naive-fedavg`,
  `fork-a-frozen-encoder`, `local-only`) and zero blocked, one induced-dropout decision, and honest
  `known_limitations` covering the DP-utility / federated-collapse limitation (#244) and the deferred
  downstream task-success blocker (stable-worldmodel #96) while keeping RFC-0006 honest-computation proofs
  explicitly out of scope. Published to `abdelstark/lensemble-phase3-consortium-checkpoint`
  (`reports/phase3_evidence_bundle.json`, `reports/phase3_model_card.md`) at revision
  `90a71318339802aa2121f71cd82fb05d927b8b81`.
- `area:eval`: **Phase 3 downstream latent-MPC eval report — honest bounded version** (#245, RFC-0005)
  — `lensemble/eval/phase3_downstream.py` adds `Phase3DownstreamEvalReport` (pydantic, frozen,
  `schema_version`-gated) with `parse_*`/`load_*`/`write_*`/`build_phase3_downstream_eval_report` and a
  `scripts/phase3_downstream_eval_report.py --validate` generator, exported from
  `lensemble/eval/__init__.py`. The checked-in `docs/evidence/phase3_downstream_eval_report.json` goes
  **beyond the prior `synthetic://toy`, 1-sample, `success_rate=0.5` placeholder** by binding the REAL
  held-out SO-100 latent metrics — the headline run's final-round `effective_rank ≈ 35.80` and
  `val_pred ≈ 32476.09` measured on the disjoint held-out split `phase3-so100-silo4.h5` (1216 windows,
  #242) — to the published checkpoint `abdelstark/lensemble-phase3-consortium-checkpoint` @ revision
  `828e210c…` (checkpoint hash `bb31c092…`, config hash `27f2c77c…`). It records a NON-TOY latent-MPC
  planner budget a closed-loop run WOULD use (`icem`, horizon 16, 512 planning samples, 8 iterations, 20
  episodes, action_dim 6, `executed=false`) and an explicit `task_success.status="blocked"`,
  `success_rate=null` with two structured blockers — **(#96)** closed-loop physical task-success is
  blocked by the unvendored `stable-worldmodel` suite (`resolve_env` raises `EvaluationError`; a recorded
  held-out dataset is open-loop) and **(#244)** latent-MPC planning success would be uninformative on the
  collapsing federated checkpoints. The report does NOT fabricate a task-success pass, is residency-safe
  (no raw observation/action arrays), and the claim boundary distinguishes real held-out latent evidence
  from a cryptographic proof and from paper-scale performance
  (`tests/ml/test_phase3_downstream_eval.py`).
- `area:observability` / `area:gauge`: **Phase 3 observability/dropout report regenerated from the real
  run** (#246, RFC-0002/RFC-0015) — `docs/evidence/phase3_observability_report.json` is now bound to the
  real headline run (`phase3-consortium-v1`, checkpoint `bb31c092…`): 11 round summaries (the 10 real
  rounds + one induced-dropout round), a real induced dropout with a **quorum-preserving close**
  (`effective_quorum=3` of 4), and 10 eval links carrying the anchored-vs-naive latent frame-drift
  contrast. The `phase3-observability-redaction-v1` contract is enforced — no raw trajectories or secrets
  in the emitted report (`tests/integration/test_phase3_observability_real_run.py`).
- `area:eval`: **Phase 3 matched controls flipped from blocked to completed**
  (#244, RFC-0002) — `build_phase3_eval_report(..., completed_controls=...)` now binds the three
  previously-blocked controls (`local-only`, `naive-fedavg`, `fork-a-frozen-encoder`) to four real
  matched control runs published on HF Jobs (DP-off, `latent_dim=256`, 6 rounds, simulated secure-agg,
  participants `phase3-so100-a..d`, held-out silo4). Each completed control row is bound to its immutable
  checkpoint revision, final global hash, config hash, and run-manifest sha256 (the no-aggregation
  local-only control binds its report sha256 instead), plus the residency-safe round-0 frame-drift and
  effective-rank gauges. `docs/evidence/phase3_eval_report.json` regenerates to **10 metric rows, 0
  blocked controls**. The honest gauge finding: the frame anchor reduces inter-participant latent
  frame-drift at aggregation (anchored round-0 48.97° vs naive-FedAvg 180°), Fork-A's frozen encoder is
  the 0° safe-degrade baseline, and local-only silos train healthily (`effective_rank ≈120`) but diverge
  maximally (180°). The model card states the limitation plainly — at the default outer-step
  (`outer_lr=0.7`) with a random-init warm-start (real V-JEPA weights unvendored, #96) the federated
  global representation collapses over rounds (`effective_rank → 1`), so the clean anchored-vs-naive
  contrast is the round-0 measurement and sustained non-collapsing training is a documented follow-up.
  This is consortium-engineering and training evidence, not a cryptographic honest-computation proof
  (`scripts/phase3_eval_report.py`, `tests/ml/test_phase3_eval_report.py`).
- `area:federation` / `area:deploy`: **Phase 3 headline GPU consortium run published**
  (#243, RFC-0002/RFC-0003) — the anchored-federation run completed on HF Jobs `h200` (job
  `6a26885bece949d7b3dcb715`, pinned commit `056f7407`): **10 closed federated rounds** at non-toy size
  (`latent_dim=256`), all four participants 0-dropped, with per-round secure-aggregation (`secure_sum`)
  and DP `(ε≈5.30, δ=1e-5)` accounting and the four real learning metrics. Checkpoint + manifests +
  ledger + report + pinned probe pushed to
  `abdelstark/lensemble-phase3-consortium-checkpoint` at immutable revision
  `828e210cba4870b2be4ab573a5f0dd4ee30bae29` (`publication.status: hf_jobs_release`); per-round report
  checked in at `docs/evidence/phase3_consortium_run_report.json`. `effective_rank` stays high
  (≈36–47/256 — the frame anchor holds rank under DP federation); the honest DP–utility limitation at 4
  participants × ~8.4M params (gradient-noise-dominated `val_pred`/`frame_drift`) is documented, with the
  gauge contrast moved to a relaxed-DP probe (#244/#246). Also fixes two real GPU bugs surfaced by the
  run: per-silo action-bound disagreement (`union_action_specs`) and DP Gaussian noise device placement.
- `area:data`: **Phase 3 training-scale SO-100 silos + held-out eval split**
  (#242, RFC-0004/RFC-0007) — published four sovereign participant silos
  (`phase3-so100-a..d`, 1259–1339 windows each at `window_steps=4`) plus one
  disjoint held-out eval split to
  [`abdelstark/lensemble-phase3-so100-silos`](https://huggingface.co/datasets/abdelstark/lensemble-phase3-so100-silos),
  a deterministic episode-modulo (`k % 5`) split of the public SO-100 set. The
  generator `scripts/phase3_consortium_silos.py` derives the agreed action/
  observation contracts and the deterministic public-probe pin, builds the
  consortium manifest bound to the immutable `hf://` refs, and regenerates
  `docs/evidence/phase3_long_run_dataset_registry.json` so every participant is
  `published` (zero placeholders) with its real window/episode counts and
  `raw_data_crosses_boundary: false` (`INV-RESIDENCY`). The registry validates
  against the checked-in `docs/evidence/phase3_consortium_manifest.json`
  (`tests/unit/test_phase3_consortium_silos.py`).
- `area:federation` / `area:deploy`: **Phase 3 consortium HF Jobs launcher**
  (#241) — `deploy/hfjobs/train_phase3_consortium.py` drives the full Phase 3
  consortium runtime (networked `Phase3CoordinatorService` plus one sovereign
  `Phase3ParticipantAgent` per mounted participant-local data ref) for
  `--num-rounds` closed federated rounds via the frozen
  `lensemble.federation.run_phase3_consortium` library entry point. It derives
  the agreed consortium manifest and dataset/probe registry from the actual
  loaded data (action/observation contracts from each silo's `ActionSpec` and
  first-window shape), pins the public-probe hash, and emits REAL residency-safe
  per-round JEPA metrics (`val_pred`/`val_sigreg`/`effective_rank`/
  `frame_drift_deg`) measured off the committed global checkpoints and a
  disjoint `--heldout-source` eval split (`INV-RESIDENCY`). `--dry-run`
  validates the manifest + registry and pins the probe hash without running any
  federated round or training compute. `tests/ml/test_phase3_consortium_launcher.py`
  is a deterministic CPU smoke covering both the dry-run preflight and a real
  two-round run with reproducible per-round metric tuples and a residency-safe
  report.
- `area:federation` / `area:docs`: **Phase 3 final evidence bundle and
  model card** (#230) — `lensemble.federation.Phase3EvidenceBundle` aggregates
  the run-specific consortium manifest, dataset/probe registry, long-run
  training report, aggregation/privacy rows, observability/dropout report,
  eval/control report, final checkpoint header/weights hashes, publication
  targets, and generated model-card text. `scripts/phase3_bundle.py`
  materializes the run-specific manifest/registry, verifies every referenced
  local artifact exists before writing a success bundle, and emits
  `docs/evidence/phase3_evidence_bundle.json` plus
  `docs/evidence/phase3_model_card.md` with explicit non-claims for provenance
  ledger implementation, cryptographic honest-computation proof, paper-scale
  LeWorldModel performance, and public SO-100 task success.
- `area:federation` / `area:observability`: **Phase 3 consortium
  observability and dropout report** (#229) —
  `lensemble.federation.Phase3ObservabilityReport` links participant lifecycle
  events, round states, event-index timing, released-update communication
  volume, dropout/retry decisions, aggregation/privacy mode, artifact
  publication status, and training/eval metric cross-references to run ids,
  participant ids, config hashes, checkpoint hashes, and source-report hashes.
  `scripts/phase3_observability_report.py` generates or validates
  `docs/evidence/phase3_observability_report.json` and runs a deterministic
  induced-dropout smoke where one of four participant agents drops while the
  remaining quorum closes the round. The report enforces a fail-closed
  redaction contract for raw data, raw actions, latents, embeddings, private
  action-head weights, model tokens, secrets, and sensitive host-local paths.
- `area:eval` / `area:docs`: **Phase 3 eval and matched-control report**
  (#228) — `lensemble.eval.Phase3EvalReport` binds completed Phase 3
  consortium-runtime metrics to checkpoint, config, run-manifest, task/env,
  seed, planner-budget, and source-report hashes, while representing public
  SO-100 task-scale eval and missing local-only, naive-FedAvg, and Fork-A
  controls as explicit blocked rows. `scripts/phase3_eval_report.py` generates
  or validates `docs/evidence/phase3_eval_report.json`, and the report emits
  conservative model-card text that separates runtime evidence from robotics
  performance claims.
- `area:federation` / `area:deploy`: **Phase 3 long-run consortium
  orchestration smoke** (#227) —
  `lensemble.federation.run_phase3_long_run_smoke` drives a deterministic
  four-participant-agent coordinator-service run for the declared Phase 3
  minimum of ten closed federated rounds, with registry/manifest dry-run
  checks, a real pinned public-probe hash, secure-aggregation and DP report
  consumption, per-round hashes, participant lifecycle counts,
  checkpoint/ledger/trace paths, and conservative local-smoke claim
  boundaries. `scripts/phase3_consortium_smoke.py` generates or validates
  `docs/evidence/phase3_long_run_smoke_report.json`, while CI covers a smaller
  no-GPU version.
- `area:aggregation` / `area:privacy`: **Phase 3 aggregation/privacy runtime
  report** (#226) — `lensemble.federation.Phase3AggregationPrivacyReport`
  records secure-aggregation backend status, threshold, aggregate-only hashes,
  explicit fallback reasons, DP accountant backend, clip/noise policy, sample
  rate, and epsilon spent for successful coordinator-service rounds. The
  service consumes the in-process secure-sum path when the selected backend is
  runnable locally, records masking-backend fallback explicitly when recovery
  shares are unavailable, and keeps participant ids/update values out of the
  report.
- `area:data` / `area:federation`: **Phase 3 dataset/public-probe registry** (#225) —
  `lensemble.data.Phase3DatasetProbeRegistry` records four-participant dataset
  declarations, held-out policy, adapter format, windowing, action/observation
  contracts, dataset smoke report URI/hash, public-probe hash, and publication
  blockers without moving raw data. `scripts/phase3_dataset_registry.py`
  generates and validates the checked-in example registry and can cross-check
  it against the consortium manifest. `Phase3ParticipantAgent` and
  `Phase3CoordinatorService` now accept the same optional registry preflight
  artifact, and the public CLI exposes matching `--registry` options.
- `area:federation` / `area:runtime`: **Phase 3 coordinator-service control
  plane** (#224) — `lensemble.federation.Phase3CoordinatorService` starts from
  the consortium manifest and coordinator config, validates the model/runtime/DP
  run agreement, exposes join, heartbeat, round assignment, update submission,
  abort, and close-round flows, rejects late joins and duplicate updates, and
  records a residency-safe participant/round trace. The service wraps the
  existing deterministic `Coordinator.try_round()` engine, so round close keeps
  the proven outer-step semantics while adding explicit dropout policy
  (`min_participants`, secure-aggregation threshold, collect timeout, retry
  budget). The new `lensemble federate coordinator-service` command emits a
  startup report and trace path for Phase 3 runs.
- `area:federation` / `area:runtime`: **Phase 3 sovereign participant agent** (#223) —
  `lensemble.federation.Phase3ParticipantAgent` validates the consortium
  manifest, participant-local data ref/window count, action and observation
  contracts, public-probe pin, model/runtime/DP agreement, and residency flags
  before contacting a coordinator. It executes assigned rounds through the
  existing claim-mode `Participant.local_round`, releases only the permitted
  `PseudoGradient`, writes deterministic local resume state
  (`delta.safetensors` plus redacted JSON metadata), emits residency-safe logs
  and metrics, and can replay the same committed update hash on rejoin. The new
  `lensemble federate participant-agent` command exposes the manifest-aware
  preflight surface while #224 owns live network coordinator assignment.
- `area:config` / `area:docs`: **Phase 3 consortium manifest and run agreement** (#222) —
  `lensemble.config.Phase3ConsortiumManifest` and `scripts/phase3_consortium_manifest.py`
  define the operational, non-cryptographic membership contract for Phase 3
  consortium runs. The shared coordinator/participant validators reject
  duplicate participant ids, WMCP/action/probe mismatches, missing data
  declarations, and unsupported network, secure-aggregation, or DP capability
  combinations. A generated four-participant example lives at
  `docs/evidence/phase3_consortium_manifest.example.json`, with docs that
  preserve the explicit non-scope: no provenance ledger and no cryptographic
  honest-computation proof.
- `area:data` / `area:deploy`: **Phase 2 participant-silo dataset smoke report** (#201, #204) —
  `lensemble.data.build_phase2_dataset_smoke_report` and `scripts/phase2_dataset_smoke.py` load each
  candidate silo through the public data adapter, count fixed-horizon windows, compute dataset Merkle
  roots, and emit a residency-safe JSON report with participant ids, action specs, and first-window tensor
  shapes. The gate fails closed on duplicate participant ids, mismatched participant/source counts, or
  underfilled silos so Phase 2 GPU HF Jobs start from validated dataset refs rather than private paths or
  zero-window data.
- `area:data`: **Phase 2 LeRobot-H5 silo splitter** (#201) — `scripts/phase2_split_lerobot_h5.py`
  deterministically partitions one LeRobot-H5 source into participant silo files by episode-level modulo
  assignment, remaps each output `episode_index` to local ids, and writes a split manifest with source and
  output hashes, selected source episode ids, frame counts, and paths. The splitter copies row-aligned HDF5
  datasets in chunks so larger camera stacks can be prepared for Phase 2 without materializing the full
  source in memory.
- `area:data` / `area:docs`: **Published Phase 2 SO-100 data refs** (#201) — README, the Phase 2 roadmap,
  and the HF Jobs runbook now record `abdelstark/lensemble-phase2-so100-silos` revision
  `97336927606fea6fbfda308bb7cee6e7b48999fa`, the two participant HDF5 file refs, dataset roots, window
  counts, source/license provenance, adapter normalization, and the declared held-out episode policy for
  downstream evaluation.
- `area:deploy` / `area:federation`: **Phase 2 HF Jobs stream large-window preflights** (#202) — the
  federated HF launcher now counts validation windows by streaming instead of materializing every window,
  and the default participant inner loop materializes only the bounded prefix it can consume for
  `inner_horizon`. This keeps published SO-100 Phase 2 silos from copying thousands of large image windows
  into memory during mount validation or short inner-loop runs.
- `area:eval` / `area:deploy`: **Claim report v2 round metric series** (#202) —
  `ClaimMVPReport.round_metrics` records curve-ready per-round global hashes, participant ids, dataset
  roots, and update L2 norms for HF Jobs, while the report builder derives a minimal series from the
  contribution ledger for non-launcher callers.
- `area:deploy` / `area:docs`: **Published Phase 2 GPU HF Job evidence** (#202, #204) — README, the
  Phase 2 roadmap, and the HF Jobs runbook now record job
  `6a22ba68e6aa50b87b9ebef7`, pinned code SHA
  `4b446a558882f25e47ee6410a4c32982bbf33477`, checkpoint repo
  `abdelstark/lensemble-phase2-so100-checkpoint` revision
  `da52ef380ac87317c89e87f048d65bae65c16b9e`, final global hash
  `8f1494fd9e57b7496daf96e379a3de1457a435080b81b9e0ea1d20a52f4827c4`, scalar
  metrics, and the schema v2 per-round update-norm series while preserving the
  compact engineering-evidence claim boundary.
- `area:eval` / `area:deploy`: **Phase 2 downstream eval report runner and evidence** (#206, #204) —
  `lensemble.eval.Phase2DownstreamEvalReport` wraps the stable `EvalReport`
  with checkpoint provenance, planner budget, held-out task policy, action
  clipping metadata, and claim boundaries. `scripts/phase2_eval_checkpoint.py`
  reconstructs eval config from a self-describing checkpoint header, can
  download public Hub checkpoint refs, and can upload the report back to the
  model repo. The first published report was generated by HF Job
  `6a22c9e3ece949d7b3dca25a` for checkpoint hash
  `8f1494fd9e57b7496daf96e379a3de1457a435080b81b9e0ea1d20a52f4827c4` and is
  checked in at `docs/evidence/phase2_downstream_eval_report.json`.
- `area:eval` / `area:docs`: **Phase 2 baseline/curve evidence report** (#205, #204) —
  `lensemble.eval.Phase2BaselinesCurvesReport` and
  `scripts/phase2_curves_report.py` aggregate completed Phase 2 training and
  downstream reports into a generated, residency-safe curve table. Every point
  records its source-report URI/hash, config hash, checkpoint/global-model hash,
  and run-manifest or eval-config hash. The first checked-in report includes the
  matched `lambda_anc=0` naive-FedAvg control from HF Job
  `6a22cd9eece949d7b3dca260`, while unmatched local-only,
  centralized/pooled, and Fork-A comparisons are explicitly marked blocked so
  model-card text cannot overstate baseline coverage. The checked-in artifact
  lives at `docs/evidence/phase2_baselines_curves_report.json`.
- `area:eval` / `area:docs`: **Phase 2 evidence bundle and model card** (#204) —
  `lensemble.eval.phase2_bundle.Phase2EvidenceBundle` and `scripts/phase2_bundle.py`
  aggregate the Phase 2 dataset smoke/manifest, training claim report,
  downstream eval report, and baseline/curve report into one generated,
  residency-safe bundle with Hub existence checks for every referenced data,
  report, and checkpoint artifact. The generated model card preserves the
  engineering-evidence claim boundary and blocked-comparison language. The
  checked-in outputs are `docs/evidence/phase2_evidence_bundle.json` and
  `docs/evidence/phase2_model_card.md`; the checkpoint repo now publishes
  `README.md`, `reports/phase2_evidence_bundle.json`, and
  `reports/phase2_model_card.md` at revision
  `eaf13136b42cde324758a191c98e377636ded7f8`.
- `area:eval` / `area:docs`: **Phase 2 empirical evidence matrix and roadmap** (#200, #203, #204) —
  `lensemble.eval.Phase2MatrixRow`, `default_phase2_matrix`, and `render_phase2_matrix_markdown` define
  the reviewer-facing Phase 2 experiment matrix: dataset refs, GPU HF Jobs, downstream eval,
  baselines/ablations, evidence bundle, and docs gates, each with expected and falsifying results. New
  `scripts/phase2_matrix.py` renders the matrix as Markdown or JSON for tracker comments, docs, and future
  model-card automation. `docs/roadmap/PHASE2.md`, README, and HF Jobs docs now distinguish the empirical
  scale/evaluation stream (#200) from RFC-0006 cryptographic proof work and point at the final verified
  claim-MVP HF evidence.
- `area:model`: **claim-mode LeWorldModel base prediction target switch** (RFC-0008; #191) —
  `ObjectiveConfig.target_stop_gradient` is now a semantic config field. The default remains `true` to
  preserve the existing proof-ready JEPA-family path, but claim-grade LeWorldModel base runs set
  `objective.target_stop_gradient=false`, causing `Objective` to compare `g_phi(f_theta(o_t), a_t)`
  against the live `f_theta(o_{t+1})` target branch with no EMA/teacher/target stop-gradient. The switch is
  plumbed through `Participant.local_round`, `train_local`, and the federated simulation harness; focused
  tests assert the default detached target branch still behaves as before and that claim mode matches the
  live-target gradient. The default `config_hash` golden is re-pinned because the resolved config gained a
  semantic field.
- `area:federation` / `area:data`: **real-data two-silo LeRobot-H5 federated claim smoke** (RFC-0004,
  RFC-0013; #193) — `DataConfig.format` now accepts `lerobot-h5`, the local LeRobot-layout HDF5 adapter is
  documented as the `lerobot-h5://<path>` / `fmt="lerobot-h5"` read-only source, and the default
  `Participant` data hooks now resolve both training windows and dataset Merkle roots from
  `cfg.data.data_source`. A new e2e smoke writes two deterministic LeRobot-H5 silos, runs the un-subclassed
  `Participant` + `Coordinator` in claim mode (`objective.target_stop_gradient=false`), commits the round,
  and asserts both participants' distinct dataset roots enter the contribution ledger.
- `area:eval` / `area:deploy`: **claim-MVP evidence hardening** (#192, #194, #196) — claim reports now
  include scalar metric slots (`val_pred`, `val_sigreg`, `effective_rank`, `frame_drift_deg`, and a
  launcher-input hash), and the HF Jobs federated launcher populates the available validation metrics from
  the final committed checkpoint over a bounded window sample. New LeWorldModel contract tests assert
  action `a_t -> z_{t+1}` alignment, no future transition-row leakage in the predictor batch path, and a
  SIGReg low-rank/zero-latent collapse guard. The README now links the published claim-MVP HF job,
  datasets, checkpoint/report repo, final checkpoint hash, and the remaining limitations.
- `area:eval`: **Non-IID severity, C/H, and scale sweeps — Claim 4 (robustness)** (RFC-0005 §7; #56) — the
  three §7 robustness sweeps run **over** the §6 ladder rungs, reusing #55's runner (`run_ablation_ladder`)
  and harness (`run_federated_simulation`). Split across the RFC-0001 §3 band (eval may not import
  federation): the **compose** side is new `lensemble/eval/sweeps.py`
  (`lensemble.eval.partition_synthetic_noniid` / `sample_drift_pairs` / `SiloPartition`); the **drivers**
  are new `lensemble/federation/sweeps.py` (`lensemble.federation.non_iid_severity_sweep` /
  `participant_horizon_sweep` / `scale_sweep`). The non-IID severity axis is **synthetic** — the partition
  shifts each silo's synthetic toy distribution by a per-silo mean offset scaled by the severity `s ∈ [0, 1]`
  (`s=0` near-IID/shared draw → `s=1` strongly non-IID/per-silo shift) — because the real
  `stable-worldmodel` factors-of-variation are **deferred** (maintainer-gated vendoring, #96): the real
  factors-of-variation path is a **documented, fail-closed seam** (`factor != "synthetic"` raises
  `EvaluationError`, never silently falling back, mirroring `resolve_env`'s `stable-worldmodel://` branch).
  The `(C, H)` sweep varies `federation.participant_count` / `inner_horizon` (a longer `H` rotates frames
  further apart before the outer step, RFC-0002 §2.1); the scale sweep repeats the key rungs at increasing
  `model.latent_dim` (each a coherent ViT shape via the #166 bridge). `sample_drift_pairs` deterministically
  samples a bounded set of participant pairs for the `O(C²)` drift figure at large `C` (seeded, capped at
  `C-choose-2`), recorded in the `RunManifest` so the figure stays reproducible (RFC-0005 §8). Each sweep
  runs only the load-bearing `naive-fedavg` + `frame-anchor` rungs per point (a new `rung_names` filter on
  `run_ablation_ladder`) to stay CPU-fast, and reuses #55's per-`Coordinator` `tempfile.mkdtemp` cleanup so
  the many-point sweeps leak no temp dir. CPU regression guard `tests/ml/test_noniid_sweeps.py` asserts the
  load-bearing trends — stronger non-IID and longer `H` raise the naive drift; the anchored rung stays low
  at every severity and scale — plus the #96 fail-closed seam, pair-sampling determinism, and zero temp-dir
  leak. Docs: extended the [Ablation Ladder page](docs/ablation-ladder.md) with the three sweep axes, the
  partition-by-factor protocol, the deferred real-factors seam, and the `O(C²)` enumerate-vs-sample policy.
- `area:eval`: **Ablation-ladder runner — the paper's core experiment** (RFC-0005 §6; #55) —
  `lensemble.eval.run_ablation_ladder` / `lambda_anc_sweep` / `RungReport` / `LADDER_RUNGS` (new
  `lensemble/eval/ablation.py`), on top of a new **live multi-round federated-simulation harness**
  `lensemble.federation.run_federated_simulation` / `SiloData` / `SimulationResult` / `RoundMetrics` (new
  `lensemble/federation/simulation.py`). The ladder realizes the RFC-0002 §4 gauge fix additively — one
  mechanism per rung: `naive-fedavg` (negative control) → `shared-sketch` (Layer 1, `lambda_sig>0`,
  `INV-SKETCH-CONSISTENCY`) → `procrustes-backstop` (Layer 3, the #18 coordinator seam ON) → `frame-anchor`
  (Layer 2, `lambda_anc>0`, the recommended config) → `distillation` (Layer 4, the gauge-invariant
  function-space consensus, #20). Each rung is driven through the harness — one `InProcessTransport`, one
  `Coordinator` (subclassed to wire the #18 backstop hooks when the rung enables it), and one `Participant`
  per silo over genuinely DIFFERENT per-silo data so the naive frames actually drift — and reports all three
  metric families at each rung: frame drift (`frame_drift`, §2), MPC `success_rate` (§3), and `effective_dim`
  (§4). `lambda_anc_sweep` resolves each swept value to a distinct, validated `LensembleConfig`
  (RFC-0002 §7, the central hyperparameter). Residency-safe (only pseudo-gradients cross the transport,
  `INV-RESIDENCY`); the harness cleans up each `Coordinator`'s `tempfile.mkdtemp` artifacts dir so the
  multi-rung × sweep runs do not leak temp dirs. CPU regression guard `tests/ml/test_ablation_ladder.py`
  asserts the load-bearing qualitative ordering (naive worst on drift; anchored flat, RFC-0005 §6). Docs:
  new [Ablation Ladder page](docs/ablation-ladder.md), referenced from RFC-0005 §6.
- `area:gauge`: **Layer-3 Procrustes re-alignment backstop at aggregation** (RFC-0002 §5; #18) —
  `lensemble.gauge.procrustes_backstop` / `realign_predictor_delta` (new `lensemble/gauge/backstop.py`).
  Immediately before the outer step, each participant whose latent frame drift exceeds
  `gauge.frame_drift_threshold_deg` has `Q_c* = procrustes_align(f_c(P), E_ref)` applied to its *released*
  delta as a PURE LINEAR operation, so the result stays bitwise-deterministic and publicly recomputable
  (`INV-AGG-DETERMINISM`; RFC-0006 §3). **Activation-space realization (the recorded #18 decision):** only
  the weight-expressible predictor conjugation `g_phi -> Q g_phi Q^T` is folded into the committed delta —
  rotating exactly `predictor.in_proj.weight` (`Δ <- Δ @ Q^T`), `predictor.out_proj.weight` (`Δ <- Q @ Δ`),
  and `predictor.out_proj.bias` (`Δ <- Q @ Δ`); the **encoder delta is left byte-identical**. The
  LayerNorm-terminated encoder has no terminal `(d, d)` linear to fold `Q` into and the maintainer chose
  not to add one, so the encoder-frame component is bounded by the Layer-2 anchor (RFC-0002 §4) and lives
  in activation space, NOT in the committed weights — which is exactly why `recompute_alignment` (#62)
  measures residual encoder drift rather than verifying a weight-fold (the #18/#62 verifiability tradeoff).
  A degenerate Procrustes (`DegenerateProcrustes`) clamp-and-retries once, then skips the backstop for that
  participant (keeping its UNALIGNED delta) and logs `gauge/procrustes_residual` at WARN (RFC-0015) — the
  round is handled in-round, never aborted. Wired into the `Coordinator` ALIGNING phase behind the
  `_probe_embeddings` / `_reference_embeddings` #18/#22 hooks (both `None` by default, so ALIGNING stays the
  byte-identical measured pass-through and the golden config hash is untouched); the backstop is a
  Stage-B / simulated-backend operation, since the masking secure-agg backend reveals only the sum.
- `area:verify`: **proof-readiness audit** — a single integration-test entry point
  (`tests/integration/test_proofready_audit.py`) that exercises all five RFC-0006 §3 Phase-1 proof-ready
  disciplines together over a tiny synthetic federated round, the v1.0 "proof-ready guarantees verified
  end-to-end" gate of the reproducibility package (RFC-0006 § Migration / Rollout; #63). Each discipline
  is asserted POSITIVE (the discipline holds) and NEGATIVE (the documented typed error / report fires and
  fails closed): bitwise-reproducible aggregation (`INV-AGG-DETERMINISM`, a `Coordinator` round commits an
  identical `(θ_{t+1}, φ_{t+1})` hash vs an injected nondeterministic reduction → `NonDeterministicAggregation`
  with the global hash unchanged); committed `(θ_t, φ_t)` content hash with a valid `parent_hash` chain
  (`INV-CHECKPOINT-HASH` vs a tampered `weights.safetensors` byte → `CheckpointIntegrityError`); each
  `Δ_c` bound to exactly one 32-byte `R_c` (`INV-COMMIT-BINDING` vs a wrong/foreign root →
  `CommitmentMismatch`); the pinned-probe hash equal to the `RoundOpen`/`GlobalState.probe_hash`
  commitment with `landmark_targets` derived only from `f_ref` (`INV-PROBE-PIN` vs a mismatched hash →
  `ProbeError`); and public recomputation reproducing the alignment from public inputs alone
  (`recompute_alignment_claim(..., expected=)` → `matches_expected=True` vs a perturbed claim →
  `matches_expected=False`). The audit composes the per-discipline primitives (no public Python symbol is
  added) and is referenced from the v1.0 release checklist (09 §5.2). It reuses the toy fixtures from the
  matching `tests/ml` tests (the 32-token recompute config so the honest LayerNorm-terminated encoder
  recovers `Q* = I`), duplicating the small helpers since the `tests/` tree has no package `__init__`.
- `area:verify`: **public recomputation of frame alignment** — the one Phase-2 verification mechanism that
  ships in Phase 1 because it is free (RFC-0006 §4; #62). `lensemble.verify.recompute_alignment(committed_weights:
  Path, probe: Path) -> FrameDriftReport` (the frozen 02 §1.8 signature) hash-verifies the committed checkpoint
  (`INV-CHECKPOINT-HASH`), reconstructs `f_θ` from its self-describing header (`Encoder.from_header`, #171),
  recomputes the pinned probe's content hash (`INV-PROBE-PIN`), and re-runs the closed-form orthogonal Procrustes
  alignment `Q* = V Uᵀ` from the SVD `E_refᵀ f_θ(P) = U Σ Vᵀ` deterministically — returning a reproducible
  `FrameDriftReport` whose `drift_from_global["committed"]` is the committed model's recovered rotation angle to the
  round-0 reference frame (built by reusing `gauge.frame_drift`, so the probe-pin / `DegenerateProcrustes` paths
  are the same primitive). New frozen pydantic-v2 records `AlignmentClaim` / `AlignmentRecomputation` (RFC-0006 §4)
  carry `procrustes_q_hash` (a platform-stable SHA-256 over the canonical little-endian fp32/fp64 bytes of `Q*` —
  the cross-process verifiability key), `procrustes_residual`, `rotation_angle_deg`, `probe_hash`, and on the
  recomputation `matches_expected` + `max_abs_residual_delta`; `recompute_alignment_claim(..., expected=)` checks a
  published claim (exact `procrustes_q_hash` match plus residual/angle within the fp32/fp64 tolerance), and
  `parse_alignment_claim` / `parse_alignment_recomputation` gate `schema_version` first (`SchemaVersionMismatch`).
  The `lensemble verify recompute --checkpoint PATH --probe PATH [--expected claim.json]` CLI echoes the report /
  record JSON to stdout and **exits non-zero** when a supplied expected claim does not match. Fail-closed and never
  swallowed: `CheckpointIntegrityError` (tamper), `SchemaVersionMismatch` (too-new), `ProbeError` (probe-pin
  mismatch), `ArtifactError` (a non-self-describing checkpoint), and `DegenerateProcrustes` (a rank-deficient probe
  embedding — never a silent garbage `Q*`) all propagate. **#18 caveat (documented honestly):** this MEASURES the
  committed model's alignment to the reference frame; it does NOT verify the Layer-3 re-alignment backstop was
  applied, because that backstop rotates in **activation space** at aggregation (the recorded #18 decision), not as
  a fold into the committed weights. New `tests/ml/test_recompute_alignment.py` (placed in `tests/ml`, the CI-scanned
  dir, not the `tests/verify` the issue named which CI does not scan).
- `area:artifacts`: **self-describing checkpoints** via a `ModelArchDescriptor` on `CheckpointHeader`
  (RFC-0010 §2; #171, unblocks #62). The new frozen pydantic-v2 `ModelArchDescriptor` records the encoder
  architecture `build_encoder` needs — `d` / `depth` / `num_heads` / `num_tokens` / `in_channels` /
  `num_frames` / `image_size` / `patch_size` / `tubelet` / `mlp_ratio` / `wmcp_version` — so
  `recompute_alignment` (#62) can reconstruct `f_θ` to recompute `f_θ(P)` (`num_heads` is unrecoverable
  from weight shapes: `in_proj_weight` is `(3d, d)` for any head count). `save_checkpoint` gains an optional
  `model_arch=` keyword and `model_arch_from_config(cfg)` builds the descriptor reading `cfg.model` exactly
  as `build_encoder` does; the `Coordinator` / `train_local` commit paths now pass it, so committed
  checkpoints ARE self-describing. New `Encoder.from_header(header)` / `build_encoder_from_arch(arch)`
  reconstruct an encoder with the SAME dims as `build_encoder(cfg)` (a legacy `model_arch=None` header
  fails closed with a re-commit-with-a-descriptor remediation). The descriptor is HEADER metadata only —
  it is NEVER fed into `StructuralFields` / `content_hash`, so the canonical hash is byte-identical with or
  without it (`INV-CHECKPOINT-HASH` stays metadata-independent; pinned by a test). No `ModelConfig` change,
  so the default-config golden hash is unchanged.
- `area:foundation`: the autonomous **`third_party/` vendoring scaffold + `deploy/` IaC stubs** for the
  RFC-0016 topology backbone (#96). New `third_party/{stable_worldmodel,stable_pretraining}/UPSTREAM.md`
  VENDORING manifests record every RFC-0016 §2 field — source URL, candidate vendored commit SHA
  (recorded but marked `STATUS: UNCONFIRMED`, a maintainer-gated research lead), vendored date
  (`TBD — not yet vendored`), license `SPDX: MIT` + the in-tree `./LICENSE` path (`pending real vendor`)
  with the maintainer-confirmation note (stable-worldmodel's MIT was maintainer-confirmed despite a
  missing upstream LICENSE — a packaging bug; stable-pretraining ships a real MIT LICENSE), an empty
  local-modification log, and the upstream-sync procedure (bump SHA → re-clone pristine → re-apply
  `patches/*.patch` → update manifest). Each project gets a git-tracked empty `patches/.gitkeep`; a
  `third_party/README.md` points to RFC-0016 and states the "no `third_party` symbol is re-exported from
  `lensemble.__init__`" rule (these subtrees are outside the import DAG). New `deploy/` carries minimal
  valid stubs — `compose.yaml` (coordinator + participant skeleton, `cpu` profile,
  `healthcheck`/`depends_on`), `helm/Chart.yaml` (apiVersion v2, name `lensemble`, version 0.0.0), and
  `kustomize/base/kustomization.yaml` (empty `resources: []`) — plus a `deploy/README.md` pointer, so the
  RFC-0016 acceptance (`helm template` / `kustomize build` / N-node Compose) is structurally unblocked.
  New `tests/unit/test_vendoring.py` asserts the manifests are present and complete, statically checks no
  `stable_worldmodel`/`stable_pretraining` symbol leaks into the `lensemble` public surface, validates
  the deploy stubs parse as YAML, and wires a skip-guarded SHA-drift check. No `lensemble/` code changes.
  The real vendoring (confirmed SHAs + cloned upstream source) is the **maintainer-gated** step deferred
  per the project decision.
- `area:e2e`: the first green end-to-end toy run (RFC-0001 Stage A; #167). `lensemble.federation.train_local`
  is now implemented (was a `NotImplementedError` stub): the single-site Stage-A path builds
  `encoder`/`predictor`/`action_head` from a `LensembleConfig` (the #166/#168 bridge), resolves local
  windows through the #22 data layer, runs `cfg.federation.inner_horizon` inner AdamW steps over the
  composite SIGReg-JEPA objective via a new shared `_inner_loop` helper (used by BOTH
  `Participant._run_inner_loop` and `train_local`, so the two never drift), hash-commits the trained
  `(θ, φ)` (`INV-CHECKPOINT-HASH`; only the shared groups enter the artifact, `INV-ACTIONHEAD-LOCAL`), and
  returns a frozen `RunResult` (checkpoint dir + `content_hash` + manifest hash + final loss). The objective's
  sketch seed is `round_sketch_seed(root_seed, 0)` (single-site == round 0, `INV-SKETCH-CONSISTENCY`).
  `DataConfig` gains `data_source: str | None` (the local-episode source `Participant._local_windows` /
  `_action_spec` now resolve via `load_episodes`, the #22-backed default) and `window_steps: int = 1` (the
  training-window horizon; `validate_config` enforces `> 0`); the golden `config_hash` is re-pinned. A built-in
  deterministic `synthetic://toy` `EvalWorld` self-registers in `lensemble.eval.world` (reads its clip shape
  from `cfg.model`; a KNOWN, non-trivial seed-pinned success rate of 0.5), so `evaluate` runs a real
  latent-MPC eval without the unvendored `stable-worldmodel` suite (#96). The CLI `train` / `eval` /
  `federate coordinator|participant` commands are wired from `_stub_command` to the real entry points
  (`eval` gains `--checkpoint`/`--env-id`; `federate` instantiates the real `Coordinator`/`Participant` and
  reports readiness — a full multi-process round needs the networked transport #45). New
  `tests/e2e/test_toy_pipeline.py` exercises the headline train→commit→eval green run, a federated
  `try_round()` commit, and a CLI smoke.
- `lensemble.federation`: the networked control-plane transport (RFC-0013 §5, Stage C) — the `Stage-C`
  realization of the runtime control plane, layered beneath the unchanged operation-oriented `Transport`
  seam (#42/#43) (#45). New `lensemble.federation.messages` defines the four boundary-crossing
  `ControlMessage`s as frozen pydantic v2 models (`extra="forbid"`, integer `CONTROL_MESSAGE_SCHEMA_VERSION`
  gated FIRST by `parse_control_message`, mirroring `parse_dataset_commitment`): `RoundOpen` (coord →
  participant, integrity hashes + `s_t` + probe/landmark hashes + `H`), `Commitment` (participant → coord,
  the dataset Merkle root `R_c`, `INV-COMMIT-BINDING`), `Update` (participant → aggregator, the released
  masked `Δ_c` as a JSON-native finite `tuple[float, ...]` + `l2_norm`, **never** a raw
  observation/action/embedding, `INV-RESIDENCY`), and `RoundClose` (coord → all, the `(θ_{t+1}, φ_{t+1})`
  content hash, `INV-CHECKPOINT-HASH`); `from_pseudogradient` / `to_delta_tensor` route the carrier through
  `guard_egress` so a non-`PseudoGradient` raw payload fails closed (`ResidencyViolation`, never swallowed).
  New `lensemble.federation.network` adds the low-level RFC-0013 §5 wire `MessageChannel` Protocol
  (`send` / `recv` with the **None-on-timeout** contract / `broadcast` / `peers`), its in-process
  `LoopbackChannel` realization (per-peer FIFO inboxes; `connected_pair` / `connected_mesh`; the testable
  stand-in for the real socket transport — the gRPC-vs-HTTP wire choice is the Stage-C Open Question), and
  `NetworkedTransport`, which implements the SAME `lensemble.federation.transport.Transport` Protocol the
  `Coordinator`/`Participant` consume — so it is **interchangeable with `InProcessTransport` in
  `Coordinator(config, *, transport=...)`** — realizing each operation over the channel: `broadcast_round_open`
  emits a `RoundOpen` and seeds the hash→weights store so `fetch_params` resolves θ/φ out-of-band by locator
  (hash-verified, `INV-CHECKPOINT-HASH`); `submit_update` `send`s an `Update`; `collect_updates` drains the
  coordinator inbox via `recv` until `None`, ingress-validates EVERY message (a malformed/too-new payload
  raises the typed `ValidationError`/`SchemaVersionMismatch` and the update is not counted, so the round
  state does not advance), and binds each `Update`'s `Δ_c` to its committed `R_c` via `verify_binding`
  (`CommitmentMismatch` on mismatch, `INV-COMMIT-BINDING`, never swallowed). A `NetworkedTransport` round
  driven over a `LoopbackChannel` commits a `global_state_hash()` bit-identical to the in-process run on the
  same seed/updates (the deterministic outer step makes the equality exact). The four message symbols,
  `parse_control_message`, `from_pseudogradient`, `to_delta_tensor`, `MessageChannel`, `LoopbackChannel`, and
  `NetworkedTransport` are exported from `lensemble.federation`. The op-oriented `Transport` and
  `InProcessTransport` are unchanged.
- `lensemble.data`: the `lance` / `hdf5` / `lerobot` on-disk storage backends behind the
  `EpisodeDataset.fmt` selector, plus the `fmt`/URI-scheme dispatcher and registry (RFC-0004 §1;
  the extension point of [02 §5.2](docs/spec/02-public-api.md#52-registering-a-new-data-adapter)).
  `save_episodes(dataset, path, *, fmt)` / `load_episodes(source, *, fmt=None)` /
  `register_adapter(fmt, *, loader, saver=None)` are exported from `lensemble.data`. The **lance**
  backend (the default reference store) writes one columnar row per `Transition` — each tensor as raw
  little-endian bytes plus a recorded dtype label + shape so the read reshapes byte-identically
  (`torch.equal` holds) — with per-episode metadata and every `ActionSpec` field as denormalized string
  columns; it is append-friendly with indexed random window reads. The **hdf5** backend writes one
  portable `.h5` file, one group per episode, with stacked `obs_t`/`action_t`/`obs_tp1` datasets and the
  episode metadata + `ActionSpec` fields as group/dataset attrs (`bfloat16` is bit-cast through `uint16`
  and restored). The read-only **`lerobot://<repo_id>`** adapter resolves a LeRobot-Hub dataset to a
  read-only `EpisodeDataset` view, importing `lerobot` lazily and raising a clear `ContractViolation`
  with remediation when the optional library is absent; its on-load conformance check
  (`_validate_episode_conformance`) validates each episode against the `Episode` schema and the WMCP
  `ActionSpec` — an action trailing dim != `ActionSpec.dim`, an embodiment-id disagreement, an invalid
  discrete `num_classes`, or a latent-incompatible modality raises `ContractViolation`
  (`WMCP_CONTRACT_VIOLATION`, RFC-0007 §4). All adapters materialize RAW, local episodes inside the
  trust boundary and expose no egress path (`INV-RESIDENCY`); `save_episodes(..., fmt="lerobot")` raises
  (the `lerobot://` view is read-only by construction, so it never participates in commitment or egress).
  `lance` and `h5py` are required (pinned) runtime deps for the two on-disk backends; `lerobot` is an
  optional extra (lazily imported). The "Format round-trip" test asserts the materialized windows are
  tensor-identical across `lance`, `hdf5`, and the original in-memory dataset (RFC-0004 §Testing,
  RFC-0009 reproducibility). (#22)
- `lensemble.federation`: the `Coordinator` outer-round orchestrator (RFC-0013 §1/§4/§6) —
  `Coordinator(config, *, transport)` with `run(num_rounds) -> None`, `round_state() -> RoundState`,
  `global_state() -> GlobalState`. It holds the canonical global model `(θ_t, φ_t)`, builds the initial
  encoder/predictor, captures the flat `θ⊕φ` in `build_pseudogradient`'s canonical order (encoder group
  sorted, then predictor group sorted) plus a `(group, name, shape)` param manifest to un-flatten the
  post-step vector, and drives a single sequential round loop (round `t+1` does not open until round `t`
  is `CLOSED`/`ABORTED`, §6) through the `RoundState` machine:
  `OPEN → COLLECTING → AGGREGATING → ALIGNING → COMMITTING → CLOSED`. **OPEN** pins `(θ_t, φ_t)`, derives
  `s_t = round_sketch_seed(root_seed, t)`, builds + broadcasts the round `GlobalState`, and seeds the
  transport fetch store so a participant fetching `θ_t/φ_t` round-trips (each `ParamRef.content_hash` is
  minted as `weights_content_hash(group_weights)`, the exact hash `fetch_params` recomputes,
  `INV-CHECKPOINT-HASH`). **COLLECTING** fixes the contributing set; below
  `fault_tolerance_min_participants` aborts with `FaultToleranceExceeded` (the global hash unchanged).
  **AGGREGATING** runs the determinism self-check — `assert_outer_step_deterministic` re-runs the
  reduction `(1/C)·Σ_c Δ_c` under the canonical participant-id-sorted order with a FRESH optimizer per
  call (pure), raising `NonDeterministicAggregation` on a mismatch (security-critical, never swallowed,
  round → `ABORTED`, `INV-AGG-DETERMINISM`; arrival order does not matter). **ALIGNING** is a
  MEASURED PASS-THROUGH — frame drift is measured on the probe when per-participant embeddings are wired,
  but the Layer-3 Procrustes backstop fold-in is **#18 (out of scope here)**, so the gauge is not
  corrected and `θ/φ` are not mutated. **COMMITTING** runs the PERSISTENT Nesterov `OuterOptimizer.step`
  over ONLY `θ/φ` (the deltas carry no action head, `INV-ACTIONHEAD-LOCAL`), un-flattens via the manifest,
  hash-commits `save_checkpoint` (round `t+1`, `parent_hash` = the current hash, `INV-CHECKPOINT-HASH`),
  and appends a `ContributionRecord` (participants sorted, their dataset roots, the new `global_model_hash`,
  `C_t` the averaging denominator) to the `ContributionLedger`. The `probe_hash` is loaded from
  `cfg.data.probe_path` when set, else a fixed 32-byte placeholder — the **#22/#04 boundary**. (#42)
- `lensemble.federation`: the `Participant` local round (RFC-0013 §1/§3) — `Participant(config, *,
  participant_id, transport)` with `local_round(global_state, round_seed) -> PseudoGradient` and
  `join(coordinator_endpoint) -> GlobalState`. One round fetches the global `(θ_t, φ_t)` through the
  transport (hash-verified, `INV-CHECKPOINT-HASH`), runs `H` inner AdamW steps on local windows over the
  encoder/predictor/local-action-head params, forms `Δ_c = (θ_c,φ_c) − (θ_t,φ_t)` over ONLY the
  `(θ, φ)` groups (`INV-ACTIONHEAD-LOCAL`; the per-embodiment head is never federated), DP-privatizes it
  (clip-then-noise, the LOCKED RFC-0003 §4 ordering; noise generator seeded by
  `(root_seed, round_index, participant_id)`), binds it to the dataset Merkle root `R_c`
  (`INV-COMMIT-BINDING`), and returns a `PseudoGradient`. The RELEASED delta is clipped-AND-noised, so its
  `l2_norm` is the HONEST released norm and MAY exceed `C_clip` after noising — `INV-DP-BOUND` is asserted
  on the POST-CLIP norm inside `lensemble.privacy.dp.privatize`, not on the released norm. Preconditions:
  `INV-PROBE-PIN` (a `probe_hash` mismatch raises `ProbeError`), `INV-WARMSTART-T0` (a round-0 encoder-hash
  drift raises **`GaugeError`** — the #43 acceptance criterion pins `GaugeError` here, deviating from SPEC
  03 §7's `CheckpointIntegrityError`), and `INV-SKETCH-CONSISTENCY` (`round_seed != sketch_seed` raises
  `GaugeError`). Only the `delta` crosses (`INV-RESIDENCY`, gated by the `EgressRole.PSEUDO_GRADIENT`
  carrier). The participant's pinned probe, local windows, dataset root, and `ActionSpec` are resolved
  through protected hooks (`_pinned_probe` / `_local_windows` / `_dataset_root` / `_action_spec` /
  `_warmstart_hash`) — the #22 data-layer boundary (a toy seam tests override; the real loader/commitment
  lands with #22). (#43)
- `lensemble.federation`: runtime fault tolerance & elasticity (RFC-0013 §3/§7). `FederationConfig` gains
  two knobs — `secure_agg_threshold: int = 2` (the secure-aggregation reveal threshold `t_agg`, validated
  `0 < t_agg <= participant_count`) and `collect_timeout_s: float = 30.0` (the per-round COLLECTING
  wall-time budget, validated `> 0`). The `Coordinator` round quorum is now
  `K = max(fault_tolerance_min_participants, secure_agg_threshold)` (below `t_agg` survivors the masking
  sum cannot be unblinded, so the higher threshold gates the COLLECTING check). `Coordinator.try_round() ->
  RoundState` attempts the CURRENT round once over the PRESENT contributing set and returns the resulting
  state: a below-`K` round goes to `ABORTED` WITHOUT raising and WITHOUT advancing the round index or
  global hash, so the SAME round `t` can be re-attempted once enough updates are staged; on success the
  hash advances and round `t+1` opens. `run(num_rounds)` loops over `try_round` as the FAIL-FAST driver,
  SURFACING a below-`K` round as a raised `FaultToleranceExceeded` (code `FAULT_TOLERANCE_EXCEEDED`,
  carrying `contributing`/`quorum`). Elastic completion averages over the present `C_t` (absent
  participants are simply not in the `ContributionRecord`); the in-process present set models the
  `collect_timeout_s` drop (the wall-clock timeout is the #45 seam) and a delta for a PAST round is never
  back-applied — a dropped participant reconciles at the NEXT round. `Participant.join` now revalidates
  `INV-WARMSTART-T0` on the recovery path when the recovered `GlobalState.round_index == 0` (fetch θ_0,
  compare the encoder content hash to the pinned warm-start; a drift → `GaugeError`), and a long-absent
  rejoiner adopts the recovered `GlobalState` as the sole source of truth (discards stale local state).
  The default-config `config_hash` golden is re-pinned (#37) to reflect the two new defaults. (#44)
- `lensemble.federation`: the `GlobalState` + `ParamRef` broadcast round-state types (03 §7) — frozen,
  validated dataclasses (`round_index >= 0`, 32-byte `probe_hash`, non-empty `wmcp_version`, 64-hex
  `content_hash`); `GlobalState` is re-exported from `lensemble.federation.round` so the RFC-0013 §1 import
  resolves. (#43)
- `lensemble.federation`: the `Transport` seam (RFC-0013 §1/§5) shared with #42/#45 — a `Protocol`
  (`register` / `recover_global_state` / `fetch_params` / `submit_update` participant-side;
  `broadcast_round_open` / `collect_updates` coordinator-side, exercised by #42) plus the concrete
  in-memory `InProcessTransport`. `fetch_params` resolves a `ParamRef` to weights and re-verifies the
  recomputed canonical-safetensors content hash against `ref.content_hash`, raising
  `CheckpointIntegrityError` on a tamper (`INV-CHECKPOINT-HASH`); `InProcessTransport.commit` seeds the
  store for tests/single-process runs. (#43)
- `area:ci`: the CPU performance smoke (08 §7) — `tests/integration/test_perf_smoke.py` guards the four
  08 §7 regression checks on a tiny synthetic config (no download): a generous wall-time ceiling on one
  toy inner+outer cycle (an overrun is a regression SIGNAL, a plain `assert`, not a raised error), the
  comms-accountant equality `comm_bytes(n) == 4*n` / `comm_bytes(n, quantized=True) == n+4` (08 §4),
  outer-step bitwise determinism via `assert_outer_step_deterministic` (`INV-AGG-DETERMINISM`, raising
  `NonDeterministicAggregation` on a violation), and the int8 `quantize_int8` round-trip within the
  module-documented per-element bound (RFC-0003 §6). It runs under the existing gate-4
  `pytest tests/integration` step; the wall-time ceiling and its "loose by design (catches 10x, not
  micro-opt)" note are recorded on the `_WALLTIME_CEILING_S` module constant, which is the config-of-record
  the 08 §7 OPEN QUESTION asks CI to keep (revisited at v0.1). (#69)
- `lensemble.eval`: the evaluation harness (RFC-0005 §3) — `evaluate(checkpoint, env_id, *, cfg) ->
  EvalReport` runs seed-pinned latent-MPC episodes on a held-out env and the `EvalReport` reporting type
  (03 §13.1; frozen, `extra="forbid"`, `parse_eval_report` with a `schema_version`-first
  `SchemaVersionMismatch` gate; out-of-range fields raise `EvaluationError`). The report carries only
  scalars / hashes / counts — no tensor reaches the sink (`INV-RESIDENCY`); it loads a hash-verified
  checkpoint read-only (`INV-CHECKPOINT-HASH`) and binds itself to an eval-mode `RunManifest` hash. The
  eval-world seam — the `EvalWorld` protocol plus `register_env` / `resolve_env` — resolves `env_id` from
  config rather than a hard-coded list; the real `stable-worldmodel` suite is deferred to #96 (a
  `stable-worldmodel://` id fails closed with remediation until then). `build_action_head` is implemented
  (continuous MLP / discrete per-dim embedding head, `INV-ACTIONHEAD-LOCAL`), filling the orphaned
  substrate of the closed issue #8 that the harness requires. (#52)
- `lensemble.aggregation`: the default secure-aggregation backend — `PairwiseMaskAggregator` (Bonawitz-style
  pairwise additive masking with self-masks, RFC-0011 §2), `DropoutRecovery` + Shamir threshold secret
  sharing for dropout robustness (RFC-0011 §4), and `build_masked_update`. Masks cancel over the integer
  field (`INV-AGG-DETERMINISM`); each masked update is hiding and the aggregator returns only the sum
  (`INV-RESIDENCY`); below the threshold it fails closed with no partial sum. The DH key-agreement /
  transport is the control plane's (#45); a toy prime-field DH stands in for the production X25519.
- `lensemble.aggregation`: the TEE-attested secure-aggregation backend — `TEEAggregator` (Backend B,
  RFC-0011 §5/§6), the `TEEAttestation` report (`enclave_measurement`/`quote`/`code_hash`), and the
  participant-side `verify_attestation`, behind the same structural `SecureAggregator` interface as the
  masking (#47) and simulated (#46) backends, so they are interchangeable in the round. A
  **software-simulated enclave at v0.2**: the enclave measurement is a domain-separated hash of the pinned
  `code_hash` and the attestation quote is an HMAC over `(enclave_measurement, code_hash)` keyed by the
  vendor root; the participant verifies the report against the pinned `code_hash` and the vendor root
  **before** opening the channel and refuses to send on failure (`SecureAggregationError`,
  `cause="attestation_failed"`). The enclave computes the fp32 plaintext `Σ_c Δ_c` inside the boundary,
  reusing the simulated backend's fixed-order integer-field summation + determinism self-check
  (`INV-AGG-DETERMINISM`), and returns only the sum — below the threshold it fails closed with no partial
  sum, and the single `TEEAggregator.egress` checkpoint routes any individual `Δ_c` through the residency
  guard, which is fail-closed with `ResidencyViolation` (`INV-RESIDENCY`, never swallowed). The trust
  assumption is hardware-attestation trust (a vendor root, with side-channel exposure), distinct from — and
  neither stronger nor weaker than — the masking backend's collusion-bounded honest-but-curious assumption.
  The real enclave provisioning, the production attestation channel, and the transport are RFC-0013's
  (#45). `FederationConfig` gains `aggregation_backend: Literal["simulated", "masking", "tee"] = "masking"`
  (the masking backend stays the default); the default-config `config_hash` golden is re-pinned. (#48)
- `lensemble.eval`: the evaluation metric bodies (RFC-0005 §3-4) — `success_rate`, `planning_cost`,
  `effective_dim` (the participation-ratio collapse guard), `linear_probe_accuracy`, `comm_bytes`, and
  `quant_ratio`. Each carries a documented unit and an in-range contract (an out-of-range value raises
  `EvaluationError`); consumed by the eval harness (#52).
- `lensemble.eval`: the four bracketing eval baselines and the gap-recovery reducer (RFC-0005 §5) — named
  Hydra config groups under `configs/baselines/` (`centralized` upper bound, `local-only` lower bound,
  `naive-fedavg` negative control with `lambda_anc=0`, `fork-a` reference / safe-degrade with the encoder
  frozen), `load_baseline`, and `gap_recovery_fraction` (`rho` in `[0, 1]`, fail-closed on a degenerate
  bracket). `ModelConfig` gains `encoder_frozen` (RFC-0002 Fork A); the four baselines share one pinned
  public probe. The default-config `config_hash` golden is re-pinned (#37) to reflect the new default. (#54)
- `lensemble.observability`: the frame-drift diagnostic emission contract (RFC-0015 §3) —
  `FrameDriftRecord` / `PairAngle` / `PairResidual`, `emit_diagnostic`, and `parse_frame_drift_record`.
  One `record_kind="frame_drift"` JSONL record per round, byte-stable (repr-float canonicalization),
  canonical `c < c'` pairing, fail-closed probe/checkpoint pin bindings, routed through the redaction guard.
- `lensemble.data.probe.reanchor_probe`: the probe re-anchoring procedure (RFC-0004 §3.1) — bumps
  `probe_version`, recomputes `content_hash`, and recomputes landmark targets against the current `f_ref`
  in one operation. `RunManifest` gains `probe_version` so a run is reproducible against the exact probe.
- `lensemble.data`: declared data-quality metadata — `DataQualityMetadata` (modality, embodiment id,
  `ActionSpec`, episode count, collection conditions; non-resident) and `validate_join_precondition`, the
  federation-join precondition that validates the `ActionSpec` and gates `wmcp_version` on exact equality
  (`INV-WMCP`, RFC-0004 §6 / RFC-0007 §6).
- `lensemble.artifacts.migrate_header`: the forward-compatible `CheckpointHeader` schema-migration chain
  and dispatcher (RFC-0010 §7). A reader accepts `schema_version <= SCHEMA_VERSION` via ordered
  `migrate_vN_to_vN1` steps and fails closed with `SchemaVersionMismatch`
  (`file_schema_version`/`reader_max_version`) on an unknown/too-new version; wired into the checkpoint
  load boundary.
- `lensemble.privacy`: the `(eps, delta)` accountant — the `Accountant` protocol with `RDPAccountant`
  (sampled-Gaussian Rényi-DP) and `PRVAccountant` (exact analytic-Gaussian) backends and `build_accountant`
  (RFC-0012 §3). Self-contained (no opacus dependency); fail-closed `would_exceed`/`step` lifecycle so a
  refused round never spends budget; PRV reports `eps` no looser than RDP.
- `lensemble.federation`: optional int8 pseudo-gradient wire quantization — `quantize_int8`,
  `dequantize_int8`, `int8_roundtrip_l2_bound`, `wire_roundtrip` (RFC-0003 §6), config-gated by
  `federation.quantize_pseudo_gradient` (default off); orthogonal to the gauge and not credited as
  privacy, with a documented `sqrt(d)·max|Δ|/254` L2 round-trip bound that preserves `INV-AGG-DETERMINISM`.
- `lensemble.provenance`: canonical episode hashing, the dataset Merkle tree and inclusion proofs,
  `DatasetCommitment` + `commit_dataset` + `verify_binding`, and the append-only hash-chained
  `ContributionLedger` (RFC-0014; `INV-COMMIT-BINDING`).
- `lensemble.aggregation.assert_outer_step_deterministic`: the per-outer-step aggregation determinism
  self-check (`INV-AGG-DETERMINISM`, RFC-0006 §3), with the dedicated build-blocking determinism CI gate.
- Reference-implementation scaffolding toward the v0.1 (Stage A) milestone across `core`, `contracts`,
  `model`, `gauge`, `federation`, `privacy`, `data`, `eval`, `config`, `artifacts`, and `observability`
  ([conventions §12](docs/spec/conventions.md#12-milestones-and-stages)).
- `lensemble.gauge`: the Layer-4 function-space distillation fallback — `distill_consensus(probe_predictions,
  *, align=True)` and `distill_to_consensus(consensus_target, *, steps, lr)` (RFC-0002 §6; the top rung of
  the ablation ladder, RFC-0005 §6). Instead of averaging weights it aggregates participant *behaviors* on
  the pinned public probe: with `align=True` each participant's probe embeddings `f_c(P)` are
  Procrustes-aligned (Layer 3) onto a deterministic reference frame (the participant first in sorted id
  order) before the mean, so the consensus is **gauge-invariant by construction** — it depends only on the
  common reference `E_ref` (up to the reference's own frame), never on which per-participant rotation `Q_c`
  each silo drew; `align=False` is the degraded plain-mean baseline. A global student then distills against
  that consensus on the probe via an L2 (squared-Frobenius) function-space loss. A pure function of
  public-probe outputs only — no private data crosses (`INV-RESIDENCY` not at stake; the probe is public) —
  upcasting to fp32 (fp64 kept) like `procrustes_align` for determinism (`INV-AGG-DETERMINISM`) and
  surfacing `DegenerateProcrustes` (`PROCRUSTES_DEGENERATE`) on an under-determined frame rather than a
  silent garbage consensus. (#20)

### Fixed

- `area:model` / `area:federation`: **Phase 2 HF Jobs model input device/dtype fix** (#202) — public probe
  tensors are now cast to the target encoder parameter dtype/device only for reference and anchor forwards,
  while the pinned probe tensor and content hash remain unchanged. `Encoder.forward` normalizes floating
  observation inputs to fp32-master weights before autocast, and `ActionHead.encode` moves actions to the
  head parameter device, preventing GPU jobs from feeding stored bf16 probes/metric windows or CPU actions
  into CUDA model modules.
- `lensemble.federation.Coordinator`: stopped leaking its `tempfile.mkdtemp` artifacts dir (#178). The
  coordinator now accepts an explicit `artifacts_dir` (a persistent, caller-owned run-dir where the
  committed checkpoints live); when omitted it creates a throwaway temp dir it OWNS and cleans up — a
  `weakref.finalize` removes it on GC, and `Coordinator.close()` / using it as a context manager remove it
  eagerly. A constructed-and-dropped coordinator now leaves no `lensemble-coordinator-*` dir behind
  (previously every construction leaked one, which accumulated to fill the disk).

### Changed

- `area:artifacts`: `CheckpointHeader` `schema_version` 1 → 2 (#171); readers migrate v1 on load via the
  no-op `migrate_v1_to_v2` in `_HEADER_MIGRATIONS` (a v1 header reads back with `model_arch=None`, a
  non-self-describing checkpoint). The added `model_arch` field is optional and additive and is NEVER
  hashed, so `INV-CHECKPOINT-HASH` is unaffected and existing v1 artifacts still verify byte-for-byte.
- `ModelConfig` → encoder/predictor bridge (#166): `ModelConfig` gained the ViT-shape fields
  `num_frames` / `tubelet` / `image_size` / `patch_size` / `depth` / `num_heads` / `in_channels` /
  `mlp_ratio` (coherent V-JEPA-class defaults: `(8//2)*(128//16)**2 == 256 == num_tokens`), and
  `build_encoder` / `build_predictor` / `build_action_head` now read `model.latent_dim` (and the new
  fields) instead of the nonexistent `model.d`. `evaluate()` / `Coordinator` / `Participant` are now
  callable from a `load_config()` config (they previously raised `AttributeError`). `validate_config`
  enforces `num_tokens == (num_frames//tubelet)*(image_size//patch_size)**2` and `num_heads | latent_dim`.
  The default-config golden hash is re-pinned. The real ViT shapes remain owned by RFC-0008/#71.
- `FederationConfig`: gained `secure_agg_threshold` and `collect_timeout_s` (RFC-0013 §3); the
  default-config `config_hash` golden vector is re-pinned to
  `aaa0a3f7b98f89bead1c2e63c49fb66e0afdb081f88d85d44d8d03e03886f4fb` to reflect the two new defaults
  (an intentional, reviewed schema addition that shifts the canonical encoding). (#44)
- `FederationConfig`: gained `aggregation_backend: Literal["simulated", "masking", "tee"] = "masking"`,
  the secure-aggregation backend selector (RFC-0011 §6); the default-config `config_hash` golden vector is
  re-pinned to `ccd59866aa2bc174b50f25025733147471c97c9e9f75011fa7ce7f950b45fa7b` to reflect the new
  default (an intentional, reviewed schema addition that shifts the canonical encoding). (#48)
- `lensemble.federation.build_pseudogradient`: the `quantized` keyword is now `quantize` — the action
  flag that applies the int8 wire round-trip on the assembled flat delta and sets
  `PseudoGradient.quantized`. Pre-1.0 minor; no released callers passed the old keyword.
