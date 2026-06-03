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

### Security

- Dependencies: replaced the `lance>=0.10` runtime dependency with **`pylance>=0.10`** (the real Lance
  columnar library; import name `lance`). The bare `lance` distribution on PyPI is an unrelated typosquat
  (`lance==1.2.1`, an empty package) that the prior pin resolved to, so the default-format data backend
  could not have worked from a clean install. Updated `pyproject.toml`, `tests/unit/test_packaging.py`,
  and `conventions §11` in lockstep (#22).

### Added

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

### Changed

- `FederationConfig`: gained `secure_agg_threshold` and `collect_timeout_s` (RFC-0013 §3); the
  default-config `config_hash` golden vector is re-pinned to
  `aaa0a3f7b98f89bead1c2e63c49fb66e0afdb081f88d85d44d8d03e03886f4fb` to reflect the two new defaults
  (an intentional, reviewed schema addition that shifts the canonical encoding). (#44)
- `lensemble.federation.build_pseudogradient`: the `quantized` keyword is now `quantize` — the action
  flag that applies the int8 wire round-trip on the assembled flat delta and sets
  `PseudoGradient.quantized`. Pre-1.0 minor; no released callers passed the old keyword.
