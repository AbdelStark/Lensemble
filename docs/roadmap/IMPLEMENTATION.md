# Implementation Tracker — 2026-06-02

Generated from the specification corpus in [PR #1](https://github.com/AbdelStark/Lensemble/pull/1). Every implementable unit of work in the spec is filed below as a GitHub issue; each is independently shippable, and cross-issue dependencies are noted inline and in each issue body. Issue numbers link to [the tracker](https://github.com/AbdelStark/Lensemble/issues).

Totals: 94 issues (77 implementation + 17 tracking). Milestones: v0.1 (40), v0.2 (28), v0.3 (5), v1.0 (4).

## Milestone: v0.1 (Stage A — single-site upper bound + scaffolding)

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| [#31](https://github.com/AbdelStark/Lensemble/issues/31) | artifacts: safetensors checkpoint format with JSON header | artifacts | p0 | m | 0010 | Open |
| [#32](https://github.com/AbdelStark/Lensemble/issues/32) | artifacts: canonical-byte SHA-256 hashing and integrity checks | artifacts | p0 | s | 0010 | Open |
| [#66](https://github.com/AbdelStark/Lensemble/issues/66) | ci: GitHub Actions pipeline (lint, types, unit, coverage) | ci | p0 | m | 0009 | Open |
| [#65](https://github.com/AbdelStark/Lensemble/issues/65) | ci: pytest + hypothesis harness and CPU fixtures | ci | p0 | m | 0008 | Open |
| [#34](https://github.com/AbdelStark/Lensemble/issues/34) | config: Hydra structured config schema and validation | config | p0 | m | 0009 | Open |
| [#35](https://github.com/AbdelStark/Lensemble/issues/35) | config: deterministic seeding scheme | config | p0 | s | 0009 | Open |
| [#7](https://github.com/AbdelStark/Lensemble/issues/7) | contracts: implement ActionSpec and its validation | contracts | p0 | s | 0007 | Open |
| [#6](https://github.com/AbdelStark/Lensemble/issues/6) | contracts: implement LatentState and the WMCP conformance check | contracts | p0 | m | 0007 | Open |
| [#3](https://github.com/AbdelStark/Lensemble/issues/3) | core: implement the error taxonomy in lensemble/errors.py | core | p0 | s | 0001 | Open |
| [#2](https://github.com/AbdelStark/Lensemble/issues/2) | core: scaffold the lensemble package and module layout | core | p0 | m | 0001 | Open |
| [#21](https://github.com/AbdelStark/Lensemble/issues/21) | data: episode/transition schema and windowed loader | data | p0 | m | 0004 | Open |
| [#24](https://github.com/AbdelStark/Lensemble/issues/24) | data: public probe set, landmark targets, and hash pinning | data | p0 | m | 0004 | Open |
| [#23](https://github.com/AbdelStark/Lensemble/issues/23) | data: residency guard (INV-RESIDENCY, fail-closed) | data | p0 | m | 0004 | Open |
| [#52](https://github.com/AbdelStark/Lensemble/issues/52) | eval: evaluation harness | eval | p0 | m | 0005 | Open |
| [#51](https://github.com/AbdelStark/Lensemble/issues/51) | eval: latent MPC planner | eval | p0 | l | 0005 | Open |
| [#10](https://github.com/AbdelStark/Lensemble/issues/10) | model: video-ViT encoder with V-JEPA 2 warm-start | model | p0 | l | 0008 | Open |
| [#11](https://github.com/AbdelStark/Lensemble/issues/11) | model: action-conditioned latent predictor | model | p0 | m | 0008 | Open |
| [#12](https://github.com/AbdelStark/Lensemble/issues/12) | model: SIGReg implementation (shared sketch + Epps-Pulley statistic) | model | p0 | l | 0008 | Open |
| [#59](https://github.com/AbdelStark/Lensemble/issues/59) | observability: redaction guard (INV-RESIDENCY, fail-closed) | observability | p0 | s | 0015, 0004 | Open |
| [#71](https://github.com/AbdelStark/Lensemble/issues/71) | packaging: pyproject.toml and dependency pins | packaging | p0 | s | 0009 | Open |
| [#5](https://github.com/AbdelStark/Lensemble/issues/5) | cli: Typer app skeleton with config loading and RunManifest emission | cli | p1 | m | 0001 | Open |
| [#37](https://github.com/AbdelStark/Lensemble/issues/37) | config: reproducibility guarantee test | config | p1 | s | 0009 | Open |
| [#36](https://github.com/AbdelStark/Lensemble/issues/36) | config: RunManifest schema and emission | config | p1 | s | 0009 | Open |
| [#8](https://github.com/AbdelStark/Lensemble/issues/8) | contracts: define the ActionHead interface and cond_dim contract | contracts | p1 | s | 0007 | Open |
| [#4](https://github.com/AbdelStark/Lensemble/issues/4) | core: enforce module dependency layering (no cycles) | core | p1 | s | 0001 | Open |
| [#22](https://github.com/AbdelStark/Lensemble/issues/22) | data: lance / hdf5 / lerobot data adapters | data | p1 | l | 0004 | Open |
| [#70](https://github.com/AbdelStark/Lensemble/issues/70) | ci: documentation link-check gate | docs | p1 | s | 0001 | Open |
| [#73](https://github.com/AbdelStark/Lensemble/issues/73) | docs: CONTRIBUTING and the RFC process | docs | p1 | s | 0009 | Open |
| [#74](https://github.com/AbdelStark/Lensemble/issues/74) | docs: SECURITY.md and threat-model alignment | docs | p1 | s | 0006 | Open |
| [#53](https://github.com/AbdelStark/Lensemble/issues/53) | eval: metrics (success, planning cost, effective dim, comms) | eval | p1 | m | 0005 | Open |
| [#14](https://github.com/AbdelStark/Lensemble/issues/14) | model: numerical contract (dtype/device/determinism) | model | p1 | m | 0008 | Open |
| [#13](https://github.com/AbdelStark/Lensemble/issues/13) | model: the composite training objective | model | p1 | m | 0008 | Open |
| [#58](https://github.com/AbdelStark/Lensemble/issues/58) | observability: metric taxonomy emission | observability | p1 | s | 0015 | Open |
| [#57](https://github.com/AbdelStark/Lensemble/issues/57) | observability: structured JSON logging | observability | p1 | s | 0015 | Open |
| [#72](https://github.com/AbdelStark/Lensemble/issues/72) | packaging: license files (code/docs/data) | packaging | p1 | s | 0009 | Open |
| [#27](https://github.com/AbdelStark/Lensemble/issues/27) | provenance: canonical episode hashing | provenance | p1 | s | 0014 | Open |
| [#28](https://github.com/AbdelStark/Lensemble/issues/28) | provenance: Merkle tree, root, and inclusion proofs | provenance | p1 | m | 0014 | Open |
| [#26](https://github.com/AbdelStark/Lensemble/issues/26) | data: data-quality metadata and WMCP precondition declaration | data | p2 | s | 0004, 0007 | Open |
| [#76](https://github.com/AbdelStark/Lensemble/issues/76) | docs: changelog discipline (Keep a Changelog) | docs | p2 | s | 0009 | Open |
| [#75](https://github.com/AbdelStark/Lensemble/issues/75) | docs: CODE_OF_CONDUCT.md | docs | p2 | s | 0009 | Open |

## Milestone: v0.2 (Stage B — simulated federation + the gauge + paper experiments)

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| [#39](https://github.com/AbdelStark/Lensemble/issues/39) | federation: deterministic Nesterov outer step | federation | p0 | m | 0003 | Open |
| [#38](https://github.com/AbdelStark/Lensemble/issues/38) | federation: PseudoGradient contract | federation | p0 | s | 0003 | Open |
| [#42](https://github.com/AbdelStark/Lensemble/issues/42) | runtime: Coordinator orchestration | federation | p0 | l | 0013 | Open |
| [#43](https://github.com/AbdelStark/Lensemble/issues/43) | runtime: Participant local round | federation | p0 | m | 0013 | Open |
| [#41](https://github.com/AbdelStark/Lensemble/issues/41) | runtime: round state machine | federation | p0 | m | 0013 | Open |
| [#16](https://github.com/AbdelStark/Lensemble/issues/16) | gauge: Variant A landmark frame-anchor loss | gauge | p0 | m | 0002 | Open |
| [#19](https://github.com/AbdelStark/Lensemble/issues/19) | gauge: frame-drift diagnostic (the headline measurement) | gauge | p0 | m | 0002, 0005 | Open |
| [#15](https://github.com/AbdelStark/Lensemble/issues/15) | gauge: closed-form Procrustes alignment with degeneracy handling | gauge | p0 | m | 0002 | Open |
| [#49](https://github.com/AbdelStark/Lensemble/issues/49) | privacy: per-participant clip+noise mechanism | privacy | p0 | s | 0012 | Open |
| [#61](https://github.com/AbdelStark/Lensemble/issues/61) | verify: aggregation determinism self-check | verify | p0 | s | 0006, 0003 | Open |
| [#46](https://github.com/AbdelStark/Lensemble/issues/46) | aggregation: in-process simulated secure aggregation (Stage B) | aggregation | p1 | m | 0011 | Open |
| [#68](https://github.com/AbdelStark/Lensemble/issues/68) | ci: aggregation determinism gate | ci | p1 | s | 0006, 0003 | Open |
| [#67](https://github.com/AbdelStark/Lensemble/issues/67) | ci: ML-specific property tests (gauge invariance, anchor, SIGReg) | ci | p1 | m | 0008, 0002 | Open |
| [#9](https://github.com/AbdelStark/Lensemble/issues/9) | contracts: wmcp_version federation join gate and conformance errors | contracts | p1 | s | 0007 | Open |
| [#55](https://github.com/AbdelStark/Lensemble/issues/55) | eval: ablation ladder runner | eval | p1 | m | 0005 | Open |
| [#54](https://github.com/AbdelStark/Lensemble/issues/54) | eval: baseline configurations | eval | p1 | m | 0005 | Open |
| [#17](https://github.com/AbdelStark/Lensemble/issues/17) | gauge: Variant B rotational-drift anchor penalty | gauge | p1 | m | 0002 | Open |
| [#18](https://github.com/AbdelStark/Lensemble/issues/18) | gauge: Layer-3 Procrustes re-alignment backstop at aggregation | gauge | p1 | m | 0002 | Open |
| [#60](https://github.com/AbdelStark/Lensemble/issues/60) | observability: frame-drift diagnostic emission contract | observability | p1 | s | 0015, 0002 | Open |
| [#50](https://github.com/AbdelStark/Lensemble/issues/50) | privacy: (eps,delta) accountant | privacy | p1 | m | 0012 | Open |
| [#29](https://github.com/AbdelStark/Lensemble/issues/29) | provenance: DatasetCommitment and binding Delta_c to R_c | provenance | p1 | s | 0014 | Open |
| [#62](https://github.com/AbdelStark/Lensemble/issues/62) | verify: public recomputation of frame alignment | verify | p1 | s | 0006, 0002 | Open |
| [#33](https://github.com/AbdelStark/Lensemble/issues/33) | artifacts: schema versioning and migration | artifacts | p2 | s | 0010 | Open |
| [#69](https://github.com/AbdelStark/Lensemble/issues/69) | ci: performance smoke budget | ci | p2 | s | 0001 | Open |
| [#25](https://github.com/AbdelStark/Lensemble/issues/25) | data: probe versioning and re-anchoring event | data | p2 | s | 0004 | Open |
| [#56](https://github.com/AbdelStark/Lensemble/issues/56) | eval: non-IID severity and scale sweeps | eval | p2 | l | 0005 | Open |
| [#20](https://github.com/AbdelStark/Lensemble/issues/20) | gauge: Layer-4 function-space distillation fallback | gauge | p2 | l | 0002 | Open |
| [#30](https://github.com/AbdelStark/Lensemble/issues/30) | provenance: append-only contribution ledger | provenance | p2 | s | 0014, 0004 | Open |

## Milestone: v0.3 (Stage C — two real sovereign nodes)

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| [#47](https://github.com/AbdelStark/Lensemble/issues/47) | aggregation: pairwise-mask secure aggregation with dropout robustness | aggregation | p0 | l | 0011 | Open |
| [#44](https://github.com/AbdelStark/Lensemble/issues/44) | runtime: elasticity, churn, and rejoiner recovery | federation | p1 | l | 0013 | Open |
| [#45](https://github.com/AbdelStark/Lensemble/issues/45) | runtime: networked control-plane transport | federation | p1 | l | 0013 | Open |
| [#48](https://github.com/AbdelStark/Lensemble/issues/48) | aggregation: TEE-based aggregator backend | aggregation | p2 | l | 0011 | Open |
| [#40](https://github.com/AbdelStark/Lensemble/issues/40) | federation: optional int8 pseudo-gradient quantization | federation | p2 | s | 0003 | Open |

## Milestone: v1.0 (hardening, public-API freeze, Fork A, verified proof-readiness)

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| [#63](https://github.com/AbdelStark/Lensemble/issues/63) | verify: proof-readiness audit | verify | p1 | m | 0006 | Open |
| [#77](https://github.com/AbdelStark/Lensemble/issues/77) | docs: documentation site build and API docs | docs | p2 | m | 0009 | Open |
| [#78](https://github.com/AbdelStark/Lensemble/issues/78) | packaging: release automation and artifact release | packaging | p2 | m | 0009, 0010 | Open |
| [#64](https://github.com/AbdelStark/Lensemble/issues/64) | verify: Phase-2 prover interface stub | verify | p2 | s | 0006 | Open |

## Tracking issues

- [#79](https://github.com/AbdelStark/Lensemble/issues/79) [Tracking] Architecture & foundation — 4 children
- [#80](https://github.com/AbdelStark/Lensemble/issues/80) [Tracking] WMCP latent contract & embodiment adapters — 4 children
- [#81](https://github.com/AbdelStark/Lensemble/issues/81) [Tracking] Model, objective & numerical contracts — 5 children
- [#82](https://github.com/AbdelStark/Lensemble/issues/82) [Tracking] The latent gauge & frame-anchored aggregation — 6 children
- [#83](https://github.com/AbdelStark/Lensemble/issues/83) [Tracking] Data, sovereignty & provenance (data layer) — 6 children
- [#84](https://github.com/AbdelStark/Lensemble/issues/84) [Tracking] Provenance commitments & Merkle scheme — 4 children
- [#85](https://github.com/AbdelStark/Lensemble/issues/85) [Tracking] Checkpoint & artifact format — 3 children
- [#86](https://github.com/AbdelStark/Lensemble/issues/86) [Tracking] Configuration, run manifest & reproducibility — 4 children
- [#87](https://github.com/AbdelStark/Lensemble/issues/87) [Tracking] Federated training protocol — 3 children
- [#88](https://github.com/AbdelStark/Lensemble/issues/88) [Tracking] Coordinator & participant runtime — 5 children
- [#89](https://github.com/AbdelStark/Lensemble/issues/89) [Tracking] Secure aggregation protocol — 3 children
- [#90](https://github.com/AbdelStark/Lensemble/issues/90) [Tracking] Differential privacy accounting — 2 children
- [#91](https://github.com/AbdelStark/Lensemble/issues/91) [Tracking] Evaluation & benchmark protocol — 6 children
- [#92](https://github.com/AbdelStark/Lensemble/issues/92) [Tracking] Observability, diagnostics & telemetry — 4 children
- [#93](https://github.com/AbdelStark/Lensemble/issues/93) [Tracking] Verifiable contribution (proof-readiness) — 4 children
- [#94](https://github.com/AbdelStark/Lensemble/issues/94) [Tracking] Testing strategy & CI — 6 children
- [#95](https://github.com/AbdelStark/Lensemble/issues/95) [Tracking] Packaging, docs & release — 8 children

## Cross-cutting dependencies

Selected cross-subsystem edges (the full graph is encoded in each issue's Dependencies section). `A -> B` means A blocks B.

- #2 (core: scaffold the lensemble package and module layout) blocks #5 (cli: Typer app skeleton with config loading and RunManifest emission)
- #2 (core: scaffold the lensemble package and module layout) blocks #21 (data: episode/transition schema and windowed loader)
- #2 (core: scaffold the lensemble package and module layout) blocks #31 (artifacts: safetensors checkpoint format with JSON header)
- #2 (core: scaffold the lensemble package and module layout) blocks #57 (observability: structured JSON logging)
- #2 (core: scaffold the lensemble package and module layout) blocks #65 (ci: pytest + hypothesis harness and CPU fixtures)
- #2 (core: scaffold the lensemble package and module layout) blocks #71 (packaging: pyproject.toml and dependency pins)
- #2 (core: scaffold the lensemble package and module layout) blocks #72 (packaging: license files (code/docs/data))
- #2 (core: scaffold the lensemble package and module layout) blocks #73 (docs: CONTRIBUTING and the RFC process)
- #2 (core: scaffold the lensemble package and module layout) blocks #74 (docs: SECURITY.md and threat-model alignment)
- #2 (core: scaffold the lensemble package and module layout) blocks #75 (docs: CODE_OF_CONDUCT.md)
- #2 (core: scaffold the lensemble package and module layout) blocks #76 (docs: changelog discipline (Keep a Changelog))
- #3 (core: implement the error taxonomy in lensemble/errors.py) blocks #6 (contracts: implement LatentState and the WMCP conformance check)
- #3 (core: implement the error taxonomy in lensemble/errors.py) blocks #7 (contracts: implement ActionSpec and its validation)
- #3 (core: implement the error taxonomy in lensemble/errors.py) blocks #23 (data: residency guard (INV-RESIDENCY, fail-closed))
- #3 (core: implement the error taxonomy in lensemble/errors.py) blocks #34 (config: Hydra structured config schema and validation)
- #3 (core: implement the error taxonomy in lensemble/errors.py) blocks #59 (observability: redaction guard (INV-RESIDENCY, fail-closed))
- #6 (contracts: implement LatentState and the WMCP conformance check) blocks #10 (model: video-ViT encoder with V-JEPA 2 warm-start)
- #6 (contracts: implement LatentState and the WMCP conformance check) blocks #15 (gauge: closed-form Procrustes alignment with degeneracy handling)
- #7 (contracts: implement ActionSpec and its validation) blocks #26 (data: data-quality metadata and WMCP precondition declaration)
- #8 (contracts: define the ActionHead interface and cond_dim contract) blocks #11 (model: action-conditioned latent predictor)
- #10 (model: video-ViT encoder with V-JEPA 2 warm-start) blocks #16 (gauge: Variant A landmark frame-anchor loss)
- #10 (model: video-ViT encoder with V-JEPA 2 warm-start) blocks #24 (data: public probe set, landmark targets, and hash pinning)
- #10 (model: video-ViT encoder with V-JEPA 2 warm-start) blocks #51 (eval: latent MPC planner)
- #11 (model: action-conditioned latent predictor) blocks #51 (eval: latent MPC planner)
- … and 28 further cross-subsystem edges (see issue bodies).

