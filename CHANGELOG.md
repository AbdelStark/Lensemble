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
