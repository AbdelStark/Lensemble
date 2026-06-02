# RFC-0009 — Configuration, Run Manifest & Reproducibility

| | |
|---|---|
| **RFC** | 0009 |
| **Title** | Configuration, Run Manifest & Reproducibility |
| **Slug** | configuration-reproducibility |
| **Status** | Accepted |
| **Track** | Standards |
| **Authors** | @AbdelStark |
| **Created** | 2026-06-02 |
| **Target milestone** | v0.1 (Stage A) |
| **Area** | config |
| **Requires** | RFC-0001, RFC-0005 |
| **Informs** | RFC-0002, RFC-0003, RFC-0008, RFC-0010, RFC-0011, RFC-0012, RFC-0013, RFC-0015 |

## Summary

This RFC specifies the configuration system and the reproducibility guarantees of Lensemble. Every
runnable entry point — single-site training, a federated round, evaluation — is driven by one frozen,
validated configuration tree (`LensembleConfig`) composed from Hydra structured-config groups, and emits
one machine-readable reproducibility record (`RunManifest`). The configuration layer pins all of the
project's determinism contracts: the single root seed from which every component seed and per-round
sketch seed `s_t` derives, the deterministic-algorithm flags, the dependency/environment fingerprint, and
the content hashes (config, probe) that bind a run to its exact inputs.

The guarantee is twofold. (1) *Configuration reproducibility*: the same `LensembleConfig` plus the same
root seed produce an identical `config_hash` and identical derived seeds on any conforming host. (2)
*Aggregation reproducibility*: the federated outer step is a pure, bitwise-deterministic function of its
committed inputs (`INV-AGG-DETERMINISM`), making the central scientific figure (frame drift, [RFC-0005
§2](RFC-0005-evaluation.md)) and the Phase-1 proof-readiness disciplines (public recomputation of
alignment, [RFC-0006](RFC-0006-verifiable-contribution.md)) reconstructible from logs and committed
artifacts alone. Inner (intra-participant) training determinism is best-effort and seed-pinned, gated by
an opt-in flag, because full bitwise determinism of large-model forward/backward on GPU costs throughput.

This RFC owns the `config/` subsystem (`lensemble/config/schema.py`, `manifest.py`, `seed.py`). It does
not define the on-disk weight artifact (that is [RFC-0010](RFC-0010-artifact-checkpoint-format.md)) nor
the dataset commitment ([RFC-0014](RFC-0014-provenance-commitments.md)); it defines the configuration
that selects, and the manifest that records, both.

## Motivation

A federated, cross-silo training system makes reproducibility a correctness property rather than a
convenience:

- **The scientific claim depends on it.** The headline result — that anchored federation holds the latent
  frame pinned where naive `FedAvg` rotates apart ([RFC-0002](RFC-0002-gauge-and-aggregation.md),
  [RFC-0005 §2](RFC-0005-evaluation.md)) — is a *measurement*. A measurement that cannot be reproduced
  from a recorded configuration and seed is not evidence. The ablation ladder ([RFC-0005 §6](RFC-0005-evaluation.md))
  is realized as configuration overrides; each rung must differ from its neighbour by exactly one
  documented mechanism, and a reviewer must be able to re-run any rung from its manifest.

- **Proof-readiness depends on it.** The Phase-2 verifiable layer ([RFC-0006
  §3](RFC-0006-verifiable-contribution.md#3-phase-1-proof-ready-requirements-cheap-to-honor-now)) requires that the aggregation/outer step be a deterministic
  function whose inputs are committed and whose output is reproducible by a third party. That discipline
  must be enforced in Phase 1, from the configuration up, or Phase 2 needs rework. `INV-AGG-DETERMINISM`
  and the per-round sketch seed `s_t` (`INV-SKETCH-CONSISTENCY`) are configuration-level contracts.

- **Federation needs shared, validated state.** Participants in distinct administrative domains must agree
  on the objective weights, the sketch dimension, the probe, the DP budget, and the inner horizon. A typo
  or version skew in one silo silently corrupts an aggregate that no single party can inspect (secure
  aggregation, [RFC-0011](RFC-0011-secure-aggregation.md)). Configuration must be validated at the
  boundary and bound to a content hash that all parties can compare.

- **Heterogeneous experiments need composition, not copy-paste.** Non-IID and scale sweeps ([RFC-0005
  §7](RFC-0005-evaluation.md)) vary one axis at a time across a combinatorial space. A structured config
  with composable groups and command-line overrides is the difference between a documented sweep and an
  unauditable pile of YAML.

Without a single, validated, hash-bound configuration tree and a manifest that fingerprints every input,
the system can produce numbers it cannot defend.

## Goals

- **G1.** One frozen, fully-typed root configuration `LensembleConfig`, composed from Hydra structured-config
  groups, validated at load with typed `ConfigError` on any inconsistency. Testable: an invalid config
  raises `ConfigError` with a remediation string; a valid config round-trips to an identical resolved tree.
- **G2.** A `RunManifest` (pydantic v2, integer `schema_version`) emitted by every entry point, recording
  the config content-hash, root seed, derived component seeds, per-round sketch seeds, git SHA,
  environment fingerprint, pinned dependency versions, probe content-hash, and `wmcp_version`. Testable:
  every entry point writes a manifest; a manifest validates against its schema.
- **G3.** A deterministic seeding scheme: one root seed derives all component seeds (python/numpy/torch/cuda)
  and every per-round sketch seed `s_t = derive(root_seed, t)`. Testable: `derive` is pure and stable
  across processes and platforms for fixed inputs.
- **G4.** The configuration-reproducibility guarantee: identical `LensembleConfig` + identical `root_seed`
  ⇒ identical `config_hash` and identical derived seeds, on any conforming host. Testable: two independent
  resolutions produce equal hashes (the reproduce-run test).
- **G5.** The configuration layer enforces `INV-AGG-DETERMINISM` reproducibility preconditions: the
  aggregation/outer-step path runs in fp32 (or fp64) with fixed reduction order and no atomics, gated by
  configuration, and a determinism self-check runs each outer step. Testable: repeated outer steps on
  fixed inputs are bitwise-equal; a deliberately non-deterministic reduction trips the self-check.
- **G6.** Override semantics for the ablation ladder and sweeps: Hydra `key=value` overrides compose with
  group selection, with documented precedence and validation after override. Testable: an override changes
  exactly the targeted field and the manifest records the resolved (post-override) config.

## Non-Goals

- **Not** the on-disk model artifact format, hashing of weights, or the parent-hash chain — that is
  [RFC-0010](RFC-0010-artifact-checkpoint-format.md). This RFC defines `config_hash` and how the manifest
  references artifacts, not the artifact bytes.
- **Not** the dataset commitment / Merkle scheme — that is [RFC-0014](RFC-0014-provenance-commitments.md).
  The manifest records the dataset root by reference (`R_c`), it does not compute it.
- **Not** the experiment-tracking backend (TensorBoard/W&B). Those are optional observability sinks
  ([RFC-0015](RFC-0015-observability-diagnostics.md)); the canonical reproducibility record is the
  on-disk `RunManifest`, not a tracking dashboard.
- **Not** full bitwise determinism of the *inner* training loop. Inner determinism is best-effort and
  seed-pinned; only the aggregation/outer path is contractually bitwise-deterministic ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)).
- **Not** a configuration UI, a config server, or runtime config hot-reload. Configuration is resolved
  once at process start and frozen for the lifetime of the run.
- **Not** secret management. No credential, key, or token is ever placed in a config file or manifest
  ([06-security.md](../spec/06-security.md)); secrets are supplied out-of-band.

## Proposed Design

### 1. Module responsibilities (`lensemble/config/`)

| File | Responsibility | Public? |
|---|---|---|
| `schema.py` | The frozen-dataclass structured-config tree (`LensembleConfig` and group schemas); registration with the Hydra `ConfigStore`; load + validation entry point `load_config`. | Public (`LensembleConfig` re-exported at top level). |
| `manifest.py` | The `RunManifest` pydantic model, `config_hash` computation, environment/dependency fingerprinting, manifest serialization/load. | Public (`RunManifest` re-exported). |
| `seed.py` | The seeding scheme: `derive`, `seed_everything`, `round_sketch_seed`, determinism-flag application. | Internal (`_internal`-equivalent; used through `load_config`/runtime). |

Dependency direction ([conventions §1](../spec/conventions.md#1-repository-and-package-layout), [01-architecture.md](../spec/01-architecture.md)): `config` depends only on
`errors` and the standard ecosystem deps (hydra-core, omegaconf, pydantic). Every other subsystem may
depend on `config`; `config` depends on no model/federation/data code, so it has no cycles and can be
imported by tests cheaply.

### 2. Structured configuration tree

Python ≥ 3.11. Configs are **frozen dataclasses** (immutable after resolution) registered as Hydra
structured configs and validated by OmegaConf. The root tree is `LensembleConfig`, re-exported from
`lensemble` ([conventions §5](../spec/conventions.md#5-public-api-surface)).

```python
# lensemble/config/schema.py
from dataclasses import dataclass, field
from typing import Literal

@dataclass(frozen=True)
class ModelConfig:
    encoder: Literal["vjepa2-vit-l", "vjepa2-vit-h", "vjepa2-vit-g"] = "vjepa2-vit-l"
    warm_start_release: str = "vjepa2-2.0"   # pinned V-JEPA 2 release tag (RFC-0008)
    latent_dim: int = 1024                   # d ([conventions §2](../spec/conventions.md#2-mathematical-notation))
    num_tokens: int = 256                    # N latent tokens per clip ([conventions §2](../spec/conventions.md#2-mathematical-notation))
    predictor_depth: int = 12
    predictor_width: int = 1024
    wmcp_version: str = "wmcp-1.0.0"          # gated at federation join (INV-WMCP)

@dataclass(frozen=True)
class ObjectiveConfig:                        # the three loss terms ([conventions §2](../spec/conventions.md#2-mathematical-notation); RFC-0008)
    lambda_pred: float = 1.0                  # λ_pred
    lambda_sig: float = 0.1                   # λ_sig
    lambda_anc: float = 1.0                   # λ_anc  — the central gauge knob (RFC-0002 §7)
    sigreg_sketch_dim: int = 64               # SIGReg projection count (RFC-0008)
    sigreg_knots: int = 17                    # Epps–Pulley integration knots (RFC-0008)
    anchor_variant: Literal["landmark", "rotational"] = "landmark"   # Variant A / B (RFC-0002 §4)

@dataclass(frozen=True)
class GaugeConfig:
    frame_drift_threshold_deg: float = 15.0   # fires Layer-3 Procrustes backstop (RFC-0002 §5)
    procrustes_singular_floor: float = 1e-6   # condition guard -> DegenerateProcrustes (RFC-0002)
    anchor_landmark_count: int = 2048         # k >= d generic landmarks (RFC-0004)

@dataclass(frozen=True)
class FederationConfig:
    participant_count: int = 4                # C ([conventions §2](../spec/conventions.md#2-mathematical-notation))
    inner_horizon: int = 50                   # H — local steps per outer round ([conventions §2](../spec/conventions.md#2-mathematical-notation))
    num_rounds: int = 100
    outer_lr: float = 0.7                     # η_out, Nesterov ([conventions §2](../spec/conventions.md#2-mathematical-notation))
    outer_nesterov_momentum: float = 0.9
    quantize_pseudo_gradient: bool = False    # int8 Δ_c quantization (RFC-0003 §6)
    fault_tolerance_min_participants: int = 3 # below -> FaultToleranceExceeded (RFC-0013)
    transport: Literal["in_process", "network"] = "in_process"   # crosses a real trust boundary iff "network"

@dataclass(frozen=True)
class PrivacyConfig:
    enabled: bool = True
    clip_norm: float = 1.0                    # C_clip ([conventions §2](../spec/conventions.md#2-mathematical-notation); INV-DP-BOUND)
    noise_multiplier: float = 1.0             # σ ([conventions §2](../spec/conventions.md#2-mathematical-notation))
    epsilon: float = 8.0                      # ε target ([conventions §2](../spec/conventions.md#2-mathematical-notation))
    delta: float = 1e-5                       # δ ([conventions §2](../spec/conventions.md#2-mathematical-notation))
    accountant: Literal["rdp", "prv"] = "rdp" # RFC-0012

@dataclass(frozen=True)
class DataConfig:
    format: Literal["lance", "hdf5", "lerobot"] = "lance"   # RFC-0004
    residency_enforced: bool = True           # INV-RESIDENCY (fail-closed; never disabled in Stage C)
    probe_path: str | None = None             # public probe P (RFC-0004); content-hashed and pinned
    embodiment_id: str = "default"            # selects the per-embodiment ActionSpec / action head

@dataclass(frozen=True)
class EvalConfig:
    env_id: str = "stable-worldmodel://pusht"
    planner: Literal["cem", "icem", "mppi"] = "icem"        # RFC-0005 §3
    planning_samples: int = 512
    horizon: int = 16

@dataclass(frozen=True)
class ObservabilityConfig:
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_path: str = "run.log.jsonl"
    metrics_path: str = "metrics.jsonl"       # canonical metrics sink (RFC-0015)
    tensorboard: bool = False                 # optional adapter (RFC-0015)
    wandb: bool = False                       # optional adapter (RFC-0015)

@dataclass(frozen=True)
class DeterminismConfig:
    root_seed: int = 0                        # the single root seed ([conventions §9](../spec/conventions.md#9-determinism-dtype-device))
    deterministic_inner: bool = False         # torch.use_deterministic_algorithms (best-effort, gated)
    deterministic_aggregation: bool = True    # INV-AGG-DETERMINISM (always on for federated runs)
    aggregation_dtype: Literal["fp32", "fp64"] = "fp32"     # fixed-order reduction dtype ([conventions §9](../spec/conventions.md#9-determinism-dtype-device))

@dataclass(frozen=True)
class LensembleConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    gauge: GaugeConfig = field(default_factory=GaugeConfig)
    federation: FederationConfig = field(default_factory=FederationConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    determinism: DeterminismConfig = field(default_factory=DeterminismConfig)
    run_mode: Literal["train_local", "coordinator", "participant", "eval"] = "train_local"
```

The group names — `model`, `objective`, `gauge`, `federation`, `privacy`, `data`, `eval`,
`observability` (plus `determinism` for the seeding/determinism knobs) — are the canonical Hydra config
groups. Group files live under `configs/<group>/<variant>.yaml` ([conventions §1](../spec/conventions.md#1-repository-and-package-layout), the `configs/` directory). The
brief already includes `model`, `gauge`, `federation`, `privacy`, `data`, and `eval`; this RFC adds two
new first-class groups, `objective` and `determinism`, so that the objective weights and the
seeding/determinism contracts are each composable and overridable in isolation, which the ablation ladder
requires. `OPEN QUESTION:` whether `objective`
and `gauge` should be sub-groups of `model` rather than peers (owner @AbdelStark, resolution Stage A
config-schema review).

### 3. Load, validation, and override semantics

```python
# lensemble/config/schema.py
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

def load_config(
    config_name: str = "default",
    overrides: list[str] | None = None,
    *,
    config_dir: Path | None = None,
) -> LensembleConfig:
    """Compose a frozen LensembleConfig from Hydra groups + key=value overrides.

    Precedence (lowest to highest): structured-config defaults < the named config
    file (config_name) < group selections < `key=value` command-line overrides.

    Raises:
        ConfigError: composition failure, type/range violation, or a cross-field
                     inconsistency (see validate_config). Carries .remediation.
    """
```

**Override precedence**, lowest to highest: (1) the frozen-dataclass defaults registered in the
`ConfigStore`; (2) the named base config file; (3) group selections (`federation=cluster4`); (4)
command-line `key=value` overrides (`objective.lambda_anc=0.5`). The last writer wins. Hydra resolves the
tree, OmegaConf type-checks against the structured schema (rejecting unknown keys — `struct` mode is on),
and the result is converted to the immutable `LensembleConfig` dataclass instance. After this point the
object is frozen; any in-process mutation is a programming error.

**Validation** (`validate_config`, called inside `load_config`) enforces the cross-field rules that a
per-field type check cannot. Each violation raises `ConfigError` ([conventions §6](../spec/conventions.md#6-error-taxonomy)) with `.code` and
`.remediation`. The validation contract:

| Rule | Condition | Error / remediation |
|---|---|---|
| Latent dimension positive and consistent | `model.latent_dim == d` matches the warm-start release's emitted dimension | `ConfigError`: "latent_dim {x} != warm-start release dimension {y}; pin a matching release" |
| Landmark coverage | `gauge.anchor_landmark_count >= model.latent_dim` (the `k >= d` condition, [RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)) | `ConfigError`: "anchor needs k>=d landmarks to pin the frame; raise anchor_landmark_count" |
| Fault-tolerance floor | `0 < federation.fault_tolerance_min_participants <= federation.participant_count` | `ConfigError`: "min participants must be in (0, C]" |
| DP budget well-formed | if `privacy.enabled`: `clip_norm > 0`, `noise_multiplier >= 0`, `0 < delta < 1`, `epsilon > 0` | `ConfigError`: "DP budget malformed; see RFC-0012" |
| Aggregation determinism for federated runs | if `run_mode in {coordinator, participant}`: `determinism.deterministic_aggregation is True` | `ConfigError`: "federated runs require deterministic_aggregation=true (INV-AGG-DETERMINISM)" |
| Residency across a network boundary | if `federation.transport == "network"`: `data.residency_enforced is True` | `ConfigError`: "residency enforcement may not be disabled across a real trust boundary (INV-RESIDENCY)" |
| Variant/SVD coherence | if `objective.anchor_variant == "rotational"`: `gauge.procrustes_singular_floor > 0` | `ConfigError`: "Variant B needs a singular-value floor to guard SVD (RFC-0002)" |
| Probe presence | if `run_mode in {coordinator, participant, eval}` and a gauge anchor is active: `data.probe_path is not None` | `ConfigError`: "anchored federation requires a pinned public probe (RFC-0004)" |

`validate_config` runs at the configuration boundary — it is one of the four boundary-validation points
mandated by [conventions §6](../spec/conventions.md#6-error-taxonomy) (config load, message ingress, artifact load, dataset ingest). It never silently
coerces; an out-of-range value is an error, not a clamp.

### 4. Seeding scheme

One root seed drives everything ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)). The derivation is a pure function over `(root_seed, label)`,
stable across processes, hosts, and OS, with no dependence on wall-clock, PID, or hardware. Reference
implementation uses BLAKE3 over a domain-separated byte encoding, reduced to a 63-bit non-negative
integer; the algorithm identifier is recorded so a future change is a versioned migration, not a silent
reinterpretation.

```python
# lensemble/config/seed.py
import blake3

SEED_DERIVATION = "blake3-v1"   # recorded in RunManifest.env["seed_derivation"]

def derive(root_seed: int, label: str) -> int:
    """Deterministic, cross-platform child seed. Pure: no time, PID, or RNG state.

    derive is collision-resistant by label and order-independent by label, so
    `derive(s, "torch")` is stable regardless of when it is called.
    """
    h = blake3.blake3(f"{root_seed}:{label}".encode("utf-8")).digest(8)
    return int.from_bytes(h, "big") & ((1 << 63) - 1)

def round_sketch_seed(root_seed: int, round_index: int) -> int:
    """s_t = derive(root_seed, t). Realizes INV-SKETCH-CONSISTENCY: every participant
    in round t derives the identical projection matrix A from this seed (RFC-0002 §3)."""
    return derive(root_seed, f"sketch:{round_index}")

def seed_everything(cfg: "LensembleConfig") -> dict[str, int]:
    """Seed python/numpy/torch/cuda from the root seed and apply determinism flags.
    Returns the component-seed map recorded in the RunManifest."""
    root = cfg.determinism.root_seed
    seeds = {lib: derive(root, lib) for lib in ("python", "numpy", "torch", "cuda")}
    # ... apply to random / np.random / torch.manual_seed / torch.cuda.manual_seed_all ...
    if cfg.determinism.deterministic_inner:
        # torch.use_deterministic_algorithms(True); CUBLAS_WORKSPACE_CONFIG handled by caller
        ...
    return seeds
```

`INV-SKETCH-CONSISTENCY` ([conventions §7](../spec/conventions.md#7-named-invariants)) is enforced here: the round sketch seed `s_t` is a deterministic
function of `(root_seed, t)` only, so the coordinator broadcasts `s_t` in `GlobalState` ([03-data-model.md
§7](../spec/03-data-model.md#7-globalstate--the-broadcast-round-state)) and every participant reconstructs the identical SIGReg projection matrix
`A`. A participant that derives a different `A` produces incommensurable projection statistics; the
mismatch surfaces as elevated frame drift and is a configuration/seed bug, detected by the
sketch-consistency test (Testing Strategy, T6).

`deterministic_inner` is best-effort: setting `torch.use_deterministic_algorithms(True)` and the CUBLAS
workspace config makes the inner forward/backward reproducible where torch supports it, at a throughput
cost. It defaults off ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)). `deterministic_aggregation` is a different, stronger contract handled in
§5.

### 5. Determinism contracts and the aggregation self-check

Two determinism regimes, distinguished by the [conventions §9](../spec/conventions.md#9-determinism-dtype-device) contract:

- **Inner loop (intra-participant):** best-effort, seed-pinned. Controlled by `deterministic_inner`. Not
  bitwise-guaranteed across GPU architectures.
- **Outer step / aggregation (`INV-AGG-DETERMINISM`):** contractually bitwise-deterministic given
  (committed deltas, round seed, prior global params). Fixed reduction order, fp32 (default) or fp64
  accumulation, no atomic adds, no nondeterministic CUDA kernels on the aggregation path. This is enforced
  by configuration (`deterministic_aggregation` must be true for federated runs — see the validation
  table in §3) and verified at runtime: the coordinator runs a *determinism self-check* each outer step.

The self-check recomputes the outer step a second time over the same committed inputs and asserts
bitwise equality of the resulting global parameters. The check is the responsibility of the outer
optimizer ([RFC-0013](RFC-0013-coordinator-runtime.md), `outer_optimizer.py`); this RFC fixes its
configuration trigger and the error it raises:

```python
# pseudocode of the per-outer-step contract (enforced in federation.outer_optimizer)
new_state_a = outer_step(prior_global, committed_deltas, round_seed)
new_state_b = outer_step(prior_global, committed_deltas, round_seed)   # recompute
if not bitwise_equal(new_state_a, new_state_b):
    raise NonDeterministicAggregation(remediation=
        "aggregation path is nondeterministic; set determinism.aggregation_dtype, "
        "disable atomic reductions, and re-run the outer step")
```

`NonDeterministicAggregation` (an `AggregationError`, [conventions §6](../spec/conventions.md#6-error-taxonomy)) is in the never-swallow set ([conventions §6](../spec/conventions.md#6-error-taxonomy)): the
round aborts and recomputes; it is never caught-and-ignored, because a nondeterministic aggregate
silently invalidates both the frame-drift figure and the Phase-2 public-recomputation guarantee
([RFC-0006 §4](RFC-0006-verifiable-contribution.md#4-public-recomputation-phase-1-free-in-scope-now)). The system response is enumerated in
[04-error-model.md](../spec/04-error-model.md): abort the round, log the determinism failure with the
round/correlation id ([RFC-0015](RFC-0015-observability-diagnostics.md)), recompute. `RISK:` recompute is
a per-pair byte comparison of the model parameters and roughly doubles outer-step CPU cost; for ViT-L/
~300M this is negligible against the inner loop, but at the 1.2B target it should be sampled (every k-th
round) rather than run every round — resolution plan: a `determinism.self_check_every: int` knob added in
Stage B once outer-step cost is measured ([08-performance-budget.md](../spec/08-performance-budget.md)).

### 6. The `RunManifest`

Every entry point (`train_local`, `Coordinator.run`, `Participant.local_round`, `evaluate`) emits one
`RunManifest`: a pydantic v2 model serialized to JSON with an explicit integer `schema_version` ([conventions §8](../spec/conventions.md#8-core-data-types),
§10). The summary schema appears in [03-data-model.md §11](../spec/03-data-model.md); this RFC is the
authoritative definition.

```python
# lensemble/config/manifest.py
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field

MANIFEST_SCHEMA_VERSION = 1

class RunManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=MANIFEST_SCHEMA_VERSION)
    config_hash: str                       # SHA-256 over the canonical resolved-config bytes (§7)
    config_resolved: dict                  # the fully-resolved config tree (for re-instantiation)
    root_seed: int                         # the single root seed ([conventions §9](../spec/conventions.md#9-determinism-dtype-device))
    component_seeds: dict[str, int]        # {"python","numpy","torch","cuda"} = derive(root_seed, lib)
    round_sketch_seeds: dict[int, int]     # t -> s_t = round_sketch_seed(root_seed, t)
    git_sha: str                           # repo commit producing the run; "+dirty" suffix if uncommitted
    env: dict[str, str]                    # python/torch/CUDA/driver/OS/seed_derivation/determinism flags
    dependency_versions: dict[str, str]    # pinned versions ([conventions §11](../spec/conventions.md#11-external-dependencies)): torch, numpy, stable-worldmodel...
    probe_hash: str | None                 # pinned public probe content hash (federated/eval runs)
    dataset_roots: dict[str, str] = {}     # participant_id -> R_c (recorded by reference; RFC-0014)
    wmcp_version: str                       # the latent-contract version in force (INV-WMCP)
    run_mode: str                           # train_local | coordinator | participant | eval
    created_at: datetime
```

Field semantics:

| Field | Source | Reproducibility role |
|---|---|---|
| `config_hash` | SHA-256 over canonical config bytes (§7) | binds the run to the exact intended computation; the comparison key all silos use |
| `config_resolved` | the frozen `LensembleConfig` serialized to a plain dict | lets a third party re-instantiate the run without the original group files |
| `root_seed` / `component_seeds` / `round_sketch_seeds` | `seed.py` | the full deterministic seed lineage ([conventions §9](../spec/conventions.md#9-determinism-dtype-device)) |
| `git_sha` | `git rev-parse HEAD` (+`+dirty` if the tree is not clean) | the source state; a dirty tree is recorded honestly, never silently |
| `env` | runtime probe | python/torch/CUDA/driver/OS, `seed_derivation` algorithm id, the determinism flags |
| `dependency_versions` | `importlib.metadata` over the pinned deps ([conventions §11](../spec/conventions.md#11-external-dependencies)) | the dependency state needed to reproduce numerics |
| `probe_hash` | the pinned probe content hash ([RFC-0004](RFC-0004-data-provenance.md)) | `INV-PROBE-PIN`: the manifest records which probe defined the reference frame |
| `dataset_roots` | the `DatasetCommitment` roots ([RFC-0014](RFC-0014-provenance-commitments.md)) | `INV-COMMIT-BINDING`: which data each released delta was bound to |
| `wmcp_version` | `model.wmcp_version` | `INV-WMCP`: the latent-contract version gating federation join |

**Manifest invariants and validation:**

- *Reproducibility contract.* Same `LensembleConfig` + same `root_seed` ⇒ identical `config_hash` and
  identical `component_seeds`/`round_sketch_seeds`. A run that loads a committed manifest and cannot
  reproduce its `config_hash` from `config_resolved` raises `ConfigError` (the config/setup is not what
  the manifest claims). Verified by the reproduce-run test (Testing Strategy, T3).
- *Schema version.* A manifest whose `schema_version` exceeds the reader's `MANIFEST_SCHEMA_VERSION`, or
  is unknown, raises `SchemaVersionMismatch` (an `ArtifactError`, [conventions §6](../spec/conventions.md#6-error-taxonomy)) — never a best-effort parse
  ([09-release-and-versioning.md §4.1](../spec/09-release-and-versioning.md)). Lower versions are read
  through registered migration functions (§8 of this RFC's Migration / Rollout).
- *Residency.* `config_resolved` and the manifest as a whole are a boundary-crossing artifact in
  federated runs. The manifest carries hashes, seeds, versions, norms, counts — never any raw observation,
  action, or private embedding (`INV-RESIDENCY`, [03-data-model.md §12](../spec/03-data-model.md)). The
  redaction guard ([RFC-0015](RFC-0015-observability-diagnostics.md)) covers the manifest writer; a
  raw-tensor field reaching the manifest is a `ResidencyViolation` (fail-closed, never swallowed).
- *No secrets.* `extra="forbid"` plus a manifest-write guard rejects any field outside the schema, which
  prevents accidental inclusion of credentials. Secrets are never configuration ([06-security.md](../spec/06-security.md)).

### 7. Configuration content hashing (`config_hash`)

`config_hash` is SHA-256 (the canonical Phase-1 commitment hash, [conventions §11](../spec/conventions.md#11-external-dependencies)) over a *canonical* byte
serialization of the resolved configuration tree. Canonicalization is required so the hash is identical
across hosts and Python builds:

1. Resolve the full `LensembleConfig` to a plain nested dict (OmegaConf `to_container(resolve=True)`).
2. Drop fields explicitly marked non-semantic (e.g. `observability.log_path`, `observability.metrics_path`
   — output sinks do not change the computation). The excluded set is a fixed, documented allowlist; a new
   field defaults to *included* so omissions fail safe.
3. Serialize with `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=True)` so key
   ordering, whitespace, and unicode are deterministic.
4. SHA-256 over the UTF-8 bytes; hex digest.

The hash algorithm and the canonicalization version are recorded (`env["config_hash_algo"] =
"sha256-canon-v1"`) so the `OPEN QUESTION:` migration to a STARK-friendly hash (Poseidon2) for Phase 2
([conventions §11](../spec/conventions.md#11-external-dependencies); shared with [RFC-0006](RFC-0006-verifiable-contribution.md) and
[RFC-0014](RFC-0014-provenance-commitments.md)) is a forward-compatible versioned change, not a breaking
reinterpretation. `OPEN QUESTION:` the exact non-semantic exclusion allowlist (owner @AbdelStark,
resolution Stage A — finalized alongside the artifact canonicalization in
[RFC-0010](RFC-0010-artifact-checkpoint-format.md) so config and checkpoint hashing share one canonical
JSON encoder).

### 8. Data flow and lifecycle

A run's configuration lifecycle, as numbered prose:

1. **Compose.** The CLI ([02-public-api.md](../spec/02-public-api.md)) calls `load_config(config_name,
   overrides)`. Hydra composes groups; OmegaConf type-checks; `validate_config` runs the cross-field
   rules; the result freezes into a `LensembleConfig`. Any failure here raises `ConfigError` before any
   model, data, or network resource is touched (fail-fast at the boundary).
2. **Seed.** `seed_everything(cfg)` derives and applies `component_seeds`; `round_sketch_seed` is called
   per round during federation. Determinism flags are applied per `DeterminismConfig`.
3. **Fingerprint.** `manifest.py` computes `config_hash`, probes `env`, reads `dependency_versions` and
   `git_sha`, and assembles the `RunManifest`. The manifest is written to the run directory immediately,
   before training starts, so a crashed run still has a reproducibility record.
4. **Run.** The entry point executes. For federated runs the coordinator broadcasts the relevant config
   subset and `s_t` in `GlobalState`; participants validate the received config against their local
   `wmcp_version` and the pinned probe hash at the federation-join precondition
   ([RFC-0007](RFC-0007-wmcp-latent-contract.md)). A `config_hash` mismatch between participants is a join
   failure (`ConfigError`), not a silent average.
5. **Record.** Per-round sketch seeds and dataset roots are appended to the manifest as the run proceeds;
   the manifest is finalized and flushed at run end. The manifest references checkpoints by their content
   hash ([RFC-0010](RFC-0010-artifact-checkpoint-format.md)) and the contribution ledger by global-model
   hash ([RFC-0014](RFC-0014-provenance-commitments.md)).

### 9. Concurrency, determinism, error propagation

- **Concurrency.** Configuration is resolved once per process at start and is immutable thereafter; there
  is no shared mutable config across the coordinator and participant processes ([01-architecture.md
  §8](../spec/01-architecture.md)). Each process owns its frozen `LensembleConfig`. The only config state
  that crosses the boundary is the explicit config subset in `GlobalState`, compared by `config_hash`.
- **Determinism.** Covered by §4–§5: inner best-effort/seed-pinned; aggregation bitwise (`INV-AGG-DETERMINISM`).
  The seeding derivation is pure and platform-independent (§4).
- **Error propagation.** All configuration errors are `ConfigError` subtype-free typed errors raised at
  load/validate. Schema-version errors on manifest load are `SchemaVersionMismatch`. No configuration code
  uses a bare `except`; validation never returns `None`/`-1` to signal failure — it raises ([conventions §6](../spec/conventions.md#6-error-taxonomy)).

### 10. Failure modes and the system's response

| Failure mode | Trigger | Detection | Error ([conventions §6](../spec/conventions.md#6-error-taxonomy)) | System response |
|---|---|---|---|---|
| Malformed / inconsistent config | bad override, type/range/cross-field violation | `load_config` → `validate_config` at boundary | `ConfigError` | Fail-fast before any resource is acquired; print remediation; non-zero CLI exit. |
| Federated run without aggregation determinism | `deterministic_aggregation=false` for `coordinator`/`participant` | `validate_config` rule | `ConfigError` | Reject at load; remediation: set the flag (`INV-AGG-DETERMINISM`). |
| Residency disabled across a real boundary | `residency_enforced=false` with `federation.transport=network` | `validate_config` rule | `ConfigError` | Reject at load; `INV-RESIDENCY` may not be configured off across a real trust boundary. |
| Anchor under-coverage | `anchor_landmark_count < latent_dim` | `validate_config` rule | `ConfigError` | Reject at load; remediation: raise landmark count to `k >= d` ([RFC-0002 §4](RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)). |
| Cross-silo config skew | participant `config_hash` ≠ coordinator's | federation-join precondition | `ConfigError` | Refuse to join the federation; the silo re-pulls the canonical config. |
| Nondeterministic aggregation | reduction order/atomics make the outer step irreproducible | per-outer-step self-check (§5) | `NonDeterministicAggregation` | Never swallowed; abort round, log, recompute ([04-error-model.md](../spec/04-error-model.md)). |
| Manifest schema too new / unknown | `schema_version > MANIFEST_SCHEMA_VERSION` | manifest load validation | `SchemaVersionMismatch` | Reject; do not best-effort parse; remediation: upgrade reader. |
| Manifest cannot reproduce its `config_hash` | `config_resolved` re-hash ≠ stored `config_hash` | reproduce-run / manifest-load check | `ConfigError` | Treat the manifest as untrustworthy; refuse to reproduce; investigate hash-algo / dependency skew. |
| Raw tensor reaching the manifest | a residency-sensitive field assigned to a manifest field | redaction guard + `extra="forbid"` | `ResidencyViolation` | Fail-closed; never written; never swallowed (`INV-RESIDENCY`). |
| Probe hash absent for anchored run | `probe_path=None` with an active anchor | `validate_config` rule | `ConfigError` | Reject at load; an anchored federation requires a pinned probe (`INV-PROBE-PIN`). |

## Alternatives Considered

**Hydra structured configs (chosen).** *What it is:* composable, typed config groups with command-line
`key=value` overrides and a sweep mechanism, validated against frozen dataclasses by OmegaConf. *Why
considered:* the ablation ladder ([RFC-0005 §6](RFC-0005-evaluation.md)) and the non-IID/scale sweeps
([RFC-0005 §7](RFC-0005-evaluation.md)) are precisely group composition plus single-axis overrides;
Hydra's multirun expresses a sweep declaratively, and its structured-config mode gives compile-time-style
type checking. *Why chosen:* it is the only candidate that natively supports group composition,
overrides, and sweeps together, and it is already a pinned ecosystem dependency ([conventions §11](../spec/conventions.md#11-external-dependencies)).

**`argparse` + hand-written YAML loading.** *What it is:* flat command-line flags plus YAML files parsed
into dicts. *Why considered:* zero dependency, fully transparent, no global state. *Why rejected:* no
composition or override semantics — every ablation rung becomes a hand-edited full config file, which is
exactly the unauditable pile of YAML the sweep design must avoid; no typed validation without
re-implementing it; the combinatorial sweep space would be managed by shell scripts.

**`pydantic-settings` (pydantic BaseSettings) as the primary config layer.** *What it is:* env-var and
file-driven typed settings with pydantic validation. *Why considered:* pydantic v2 is already pinned for
on-disk metadata ([conventions §11](../spec/conventions.md#11-external-dependencies)), so a single validation library is attractive; its typed validation is
excellent. *Why rejected:* it has no first-class config-group composition or multirun sweep, so the
ablation/sweep ergonomics fall back to manual file management; it is oriented toward
twelve-factor/env-var deployment config, not research experiment composition. Decision: use pydantic for
the *on-disk metadata schemas* (`RunManifest`, commitments, artifact headers — where typed JSON
validation is the requirement) and Hydra/OmegaConf for the *structured experiment configuration tree*
(where composition and overrides are the requirement). The two libraries play to their strengths and the
manifest validates the config that Hydra produced.

**JSON vs YAML manifests.** *What it is:* the on-disk format of the `RunManifest`. *Why considered:* YAML
is human-friendlier and matches the Hydra config files. *Why rejected for the manifest:* the manifest is
a machine-verifiable record whose `config_hash` must be byte-canonical (§7); JSON with sorted keys and
fixed separators canonicalizes cleanly and deterministically, where YAML's flexible representations
(anchors, multiple scalar styles, ordering) make a stable canonical byte form harder. Config *inputs* are
YAML (authored by humans, composed by Hydra); the manifest *output* is canonical JSON
([03-data-model.md §14](../spec/03-data-model.md)).

**Per-process independent seeding vs a single root seed (chosen: single root).** *What it is:* seed each
library/process independently from its own seed vs deriving all seeds from one root. *Why considered:*
independent seeds are simpler to wire. *Why rejected:* a single root seed plus a pure `derive` function
(§4) makes the entire seed lineage reproducible from one recorded integer and makes `INV-SKETCH-CONSISTENCY`
trivial to guarantee (`s_t = derive(root_seed, t)` is computed identically by every silo); independent
seeds would require communicating and recording each one and cannot guarantee cross-silo sketch agreement.

## Drawbacks

- **Hydra's learning curve and global state.** Hydra changes the working directory by default, owns
  argument parsing, and uses a global `ConfigStore`; contributors unfamiliar with it can be surprised by
  the output-directory behavior and the override grammar. Mitigation: we disable Hydra's working-directory
  changes, document the override grammar in [02-public-api.md](../spec/02-public-api.md), and keep the
  `ConfigStore` registration in one module (`schema.py`).
- **Two configuration libraries.** Using Hydra/OmegaConf for the input tree and pydantic for the on-disk
  manifest is a deliberate split but is two mental models. Mitigation: the boundary is sharp — OmegaConf
  validates the in-memory experiment config; pydantic validates persisted JSON metadata — and the manifest
  is generated from the resolved config, never authored by hand.
- **Canonical-hash fragility.** `config_hash` is only as stable as the canonicalization (§7) and the
  non-semantic exclusion allowlist. A field added without classifying it defaults to *included*, which is
  safe (it can only change the hash, never silently leave a semantic field out), but a careless
  *exclusion* could let two semantically different runs collide on one hash. Mitigation: the exclusion
  allowlist is small, reviewed, and covered by a test that asserts a semantic field change always changes
  the hash.
- **Determinism cost.** The aggregation self-check (§5) roughly doubles outer-step compute, and
  `deterministic_inner` reduces inner throughput. Both are configuration-gated, and the self-check
  cadence is a planned knob (see the §5 `RISK:`).
- **Manifest can drift from reality if not written first.** A manifest assembled at run *end* would be lost
  on a crash. Mitigation: the manifest is written before training starts (§8 step 3) and appended to, so a
  partial run is still reproducible up to its last recorded round.

## Migration / Rollout

- **v0.1 (Stage A).** The full config tree, `load_config`/`validate_config`, the seeding scheme,
  `config_hash`, and the `RunManifest` ship for single-site `train_local` and `evaluate`. Stage A is the
  centralized upper bound ([conventions §12](../spec/conventions.md#12-milestones-and-stages)), so the federation/privacy groups exist and validate but are exercised
  by the inner-loop and eval paths.
- **v0.2 (Stage B).** The federated groups (`federation`, `gauge`, `privacy`) are exercised in simulated
  federation; the per-round sketch seed `s_t` and the aggregation determinism self-check go live; the
  config-hash cross-silo comparison gates the (simulated) federation join. The `self_check_every` cadence
  knob is added once outer-step cost is measured.
- **v0.3 (Stage C).** Residency enforcement is non-optional in configuration across a real boundary; the
  manifest records `dataset_roots` from real `DatasetCommitment`s.
- **v1.0.** The config schema and `RunManifest` schema freeze under SemVer ([conventions §10](../spec/conventions.md#10-versioning-and-schema-policy)).
- **Schema migration.** `MANIFEST_SCHEMA_VERSION` starts at 1. Each bump ships a `migrate_vN_to_vN+1`
  function; readers accept `schema_version <= current` and chain migrations; an unknown/too-new version is
  `SchemaVersionMismatch` ([09-release-and-versioning.md §4.1](../spec/09-release-and-versioning.md)).
  Pre-1.0, the config-group schemas may change with a manifest `schema_version` bump and a deprecation
  note ([conventions §10](../spec/conventions.md#10-versioning-and-schema-policy)).

## Testing Strategy

Tests run on CPU with tiny synthetic configs (no large downloads, [conventions §9](../spec/conventions.md#9-determinism-dtype-device), [07-testing-strategy.md
§7](../spec/07-testing-strategy.md)). Named tests with the property each checks:

- **T1 — config validation (valid).** A representative valid `LensembleConfig` loads and freezes; the
  resolved tree equals the expected structure. (`area:config`)
- **T2 — config validation (invalid).** Each row of the §3 validation table is exercised: the offending
  config raises `ConfigError` with a non-empty `.remediation`; no resource is acquired before the raise.
  Property: invalid configs fail fast at the boundary, never silently. (Maps to [07-testing-strategy.md
  §2](../spec/07-testing-strategy.md).)
- **T3 — reproduce-run (config-hash determinism).** Two independent resolutions of the same config with the
  same `root_seed` produce equal `config_hash`, equal `component_seeds`, and equal `round_sketch_seeds`.
  Property: the configuration-reproducibility guarantee G4; same seed ⇒ same manifest hash
  ([07-testing-strategy.md §2.11](../spec/07-testing-strategy.md)).
- **T4 — manifest round-trip.** A `RunManifest` serializes to canonical JSON and loads back equal;
  `extra="forbid"` rejects an unknown field; a `schema_version` above current raises
  `SchemaVersionMismatch`. (Maps to [07-testing-strategy.md §2.10](../spec/07-testing-strategy.md).)
- **T5 — seed-determinism.** `derive(root, label)` is pure and stable across processes and platforms
  (golden-value vectors checked); `seed_everything` produces a fixed `component_seeds` map for a fixed
  root. Property: the seeding scheme is platform-independent (G3).
- **T6 — sketch-consistency.** For a fixed `root_seed` and round `t`, `round_sketch_seed` is identical
  across simulated participants, so the derived projection matrix `A` is identical
  (`INV-SKETCH-CONSISTENCY`). Property: every participant in round `t` shares one `A` ([RFC-0002
  §3](RFC-0002-gauge-and-aggregation.md#3-layer-1--shared-sketch-matrix-objective-consistency-not-the-gauge-fix)).
- **T7 — override precedence.** A `key=value` override changes exactly the targeted field and nothing else;
  the resulting `config_hash` differs from the un-overridden one; the manifest records the resolved
  (post-override) config. Property: G6.
- **T8 — aggregation-determinism self-check.** On a toy aggregation, the self-check passes for a
  fixed-order fp32 reduction and raises `NonDeterministicAggregation` when a deliberately
  order-randomized reduction is injected. Property: `INV-AGG-DETERMINISM` is enforced and detected
  ([07-testing-strategy.md §2.5](../spec/07-testing-strategy.md)).
- **T9 — hash-allowlist sensitivity.** Changing any *semantic* field changes `config_hash`; changing only
  a non-semantic field (a log path) does not. Property: the §7 canonicalization neither under- nor
  over-includes; semantic changes can never collide.
- **T10 — residency redaction of the manifest.** Asserts no raw observation/action/embedding field can be
  written to a manifest; the redaction guard raises `ResidencyViolation` (`INV-RESIDENCY`,
  [RFC-0015](RFC-0015-observability-diagnostics.md)). Property: the manifest is a safe boundary-crossing
  artifact.

CI gates wire T1–T10 into the unit/property tier; T3, T5, T8 are part of the determinism gate
([07-testing-strategy.md §8](../spec/07-testing-strategy.md), [09-release-and-versioning.md
§5.2](../spec/09-release-and-versioning.md)).

## Open Questions

- `OPEN QUESTION:` The exact non-semantic exclusion allowlist for `config_hash` canonicalization (§7) —
  which fields are output-only and therefore excluded. Owner @AbdelStark; resolution: Stage A, finalized
  jointly with the canonical JSON encoder in [RFC-0010](RFC-0010-artifact-checkpoint-format.md) so config
  and checkpoint hashing share one canonicalization.
- `OPEN QUESTION:` How much of the environment to hash for reproducibility vs merely record. Hashing the
  full `env` (CUDA/driver/OS) into a stricter "environment fingerprint" would let a run *refuse* to claim
  reproduction on a divergent host, but is too strict for cross-platform CI on CPU. Owner @AbdelStark;
  resolution: Stage A — decide the recorded-vs-hashed split; default to *record everything, hash only the
  config and seeds* until numeric divergence across environments is characterized.
- `OPEN QUESTION:` Config-schema stability pre-1.0 — how aggressively the group schemas may change between
  minor versions while the corpus is still being written. Owner @AbdelStark; resolution: governed by the
  deprecation policy in [09-release-and-versioning.md §2](../spec/09-release-and-versioning.md); each
  breaking config change ships a manifest `schema_version` bump and a migration.
- `OPEN QUESTION:` Whether `objective` and `gauge` should be peer config groups or sub-groups of `model`
  (§2). Owner @AbdelStark; resolution: Stage A config-schema review.
- `OPEN QUESTION:` The `self_check_every` aggregation-determinism cadence at the 1.2B scale (§5 `RISK:`).
  Owner @AbdelStark; resolution: Stage B, once outer-step wall time is measured
  ([08-performance-budget.md](../spec/08-performance-budget.md)).
- `OPEN QUESTION:` Migration of the Phase-1 hash (SHA-256) used for `config_hash` to a STARK-friendly hash
  (Poseidon2) for cheap Phase-2 circuits ([conventions §11](../spec/conventions.md#11-external-dependencies); shared with
  [RFC-0006](RFC-0006-verifiable-contribution.md) and
  [RFC-0014](RFC-0014-provenance-commitments.md)). Owner @AbdelStark; resolution: Stage D / Phase 2 — kept
  forward-compatible via the recorded `config_hash_algo` version (§7).

## References

- [RFC-0001 — Architecture & System Overview](RFC-0001-architecture.md) — module map (§2), trust
  boundaries (§5/§6), process model (§8).
- [RFC-0002 — The Latent Gauge & Frame-Anchored Aggregation](RFC-0002-gauge-and-aggregation.md) — the
  sketch matrix `A` and `s_t` (§3), the anchor and `k >= d` landmarks (§4), the `λ_anc` knob (§7).
- [RFC-0003 — Federated Training Protocol](RFC-0003-federated-protocol.md) — DiLoCo outer loop, `H`,
  pseudo-gradient int8 quantization, the message table.
- [RFC-0005 — Evaluation & Benchmark Protocol](RFC-0005-evaluation.md) — §2 frame-drift diagnostic, §6
  ablation ladder, §7 sweeps, §8 reproducibility & reporting (the primary source for this RFC).
- [RFC-0006 — Verifiable Contribution](RFC-0006-verifiable-contribution.md) — §3 proof-readiness, §4
  public recomputation; deterministic aggregation as a Phase-1 discipline.
- [RFC-0008 — Model, Objective & Numerical Contracts](RFC-0008-model-objective-numerics.md) — the
  objective terms (`λ_pred`, `λ_sig`, `λ_anc`), SIGReg sketch dim / knots, bf16/fp32 numerics.
- [RFC-0010 — Checkpoint & Artifact Format](RFC-0010-artifact-checkpoint-format.md) — `config_hash`
  referenced by checkpoint headers; shared canonicalization.
- [RFC-0012 — Differential Privacy Accounting](RFC-0012-differential-privacy.md) — the DP budget config
  (`σ`, `C_clip`, `(ε,δ)`) and accountant selection.
- [RFC-0013 — Coordinator & Participant Runtime](RFC-0013-coordinator-runtime.md) — the outer optimizer
  that runs the aggregation determinism self-check; the round lifecycle.
- [RFC-0014 — Provenance Commitments & Merkle Scheme](RFC-0014-provenance-commitments.md) — the dataset
  roots `R_c` recorded by reference in the manifest.
- [RFC-0015 — Observability, Diagnostics & Telemetry](RFC-0015-observability-diagnostics.md) — the
  redaction guard covering the manifest writer; canonical metrics sink.
- [03-data-model.md](../spec/03-data-model.md) — `RunManifest` summary (§11), serialization rules (§14),
  schema versioning (§15), residency/redaction (§12).
- [02-public-api.md](../spec/02-public-api.md) — the CLI `--config` and `key=value` override surface; the
  `LensembleConfig`/`RunManifest` public symbols.
- [07-testing-strategy.md](../spec/07-testing-strategy.md) — the reproducibility and determinism tests.
- [09-release-and-versioning.md](../spec/09-release-and-versioning.md) — SemVer, schema versioning,
  release-blocking gates.
- External: hydra-core / omegaconf (structured config groups + overrides + multirun sweeps), pydantic v2
  (typed JSON metadata validation), blake3 / hashlib (seed derivation and SHA-256 config hashing) —
  [conventions §11](../spec/conventions.md#11-external-dependencies).
