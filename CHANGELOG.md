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

- `lensemble.federation.build_pseudogradient`: the `quantized` keyword is now `quantize` — the action
  flag that applies the int8 wire round-trip on the assembled flat delta and sets
  `PseudoGradient.quantized`. Pre-1.0 minor; no released callers passed the old keyword.
