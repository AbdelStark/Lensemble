# 02 — Public API

This section is the stable contract for everything a consumer imports from `lensemble` or invokes
through the `lensemble` CLI. It enumerates the public Python surface (signatures, contracts, examples),
the stability and versioning policy, the CLI surface, and the extension points for new embodiments and
data adapters. The rationale behind each contract lives in the owning RFC; this document holds the
shape, the preconditions/postconditions, the raised errors (from the taxonomy in
[04 — Error Model](04-error-model.md)), and the determinism guarantees. A new contributor should be
able to call any symbol here, or build a CLI invocation, without reading the implementation.

Scope notes:

- Python `>= 3.11` (structural typing, `tomllib`); see [09 — Release & Versioning](09-release-and-versioning.md).
- Every on-disk object named below (`RunManifest`, `DatasetCommitment`, `EvalReport`,
  `FrameDriftReport`, `Checkpoint`) is fully schematized in [03 — Data Model](03-data-model.md); this
  section references those types, it does not redefine them.
- Error classes (`ConfigError`, `ContractViolation`, `ResidencyViolation`, `GaugeError`,
  `AggregationError`, `PrivacyBudgetExceeded`, `ProvenanceError`, `ArtifactError`, `RoundError`,
  `ProbeError`, `EvaluationError`) are defined in [04 — Error Model](04-error-model.md). Every error
  carries `.code: LensembleErrorCode` and `.remediation: str`.

---

## 1. Public Python surface

The public surface is re-exported from `lensemble/__init__.py`. Anything in a module named `_internal`
or any symbol prefixed with `_` is private and unversioned (see [§3](#3-stability--versioning-policy)).
The canonical signature set is fixed by the [conventions document](conventions.md) and is frozen at `1.0`.

```python
import lensemble
lensemble.__version__  # str, SemVer
```

### 1.1 `LensembleConfig` and `RunManifest`

```python
from lensemble.config import LensembleConfig, RunManifest
```

`LensembleConfig` is the root structured-configuration tree: a frozen dataclass composed of the config
groups `model`, `data`, `federation`, `gauge`, `privacy`, `eval`, `observability`. It is constructed by
Hydra/OmegaConf from the config files in `configs/` plus `key=value` overrides, and validated on load.

- Type: `@dataclass(frozen=True) class LensembleConfig` — immutable; mutation raises `FrozenInstanceError`.
- Construction: `lensemble.config.load(path: Path, overrides: list[str] = []) -> LensembleConfig`.
- Precondition: every group conforms to its schema. Postcondition: the returned tree is fully
  validated and hashable; its content hash seeds the `RunManifest`.
- Errors: `ConfigError` on an unknown key, a type mismatch, an out-of-range value, or an inconsistent
  combination (for example a `gauge` variant that requires a probe with no probe configured). Validation
  happens at the boundary (config load), never lazily mid-run.

`RunManifest` is the pydantic v2 record every run emits: config content-hash, root seed and the derived
component seeds (python/numpy/torch/cuda), per-round sketch seeds `s_t`, git SHA, environment
(python/torch/CUDA/driver), pinned dependency versions, probe content-hash, and hardware. It carries an
integer `schema_version`. Its full schema is in [03 — Data Model](03-data-model.md); the reproducibility
contract is specified in [RFC-0009](../rfcs/RFC-0009-configuration-reproducibility.md).

```python
cfg = lensemble.config.load(Path("configs/stage_a.yaml"), overrides=["model.size=vit_l", "seed=7"])
result = lensemble.train_local(cfg)
result.manifest.config_hash  # stable across machines for the same cfg
```

Determinism: two loads of the same files with the same overrides produce byte-identical config hashes;
two runs with the same `cfg` and seed produce identical `RunManifest` config/seed hashes
(`INV-AGG-DETERMINISM` governs the aggregation path; inner-loop determinism is seed-pinned and best-effort,
gated by a config flag — see [§1.2](#12-train_local) and the [conventions document](conventions.md) determinism notes).

### 1.2 `train_local`

```python
def train_local(config: LensembleConfig) -> "RunResult": ...
```

Runs single-site / participant-local training: the warm-started encoder `f_θ`, predictor `g_φ`, and the
local action head `h_ψ^(c)` co-trained on the objective of
[RFC-0002 §3](../rfcs/RFC-0002-gauge-and-aggregation.md#3-layer-1--shared-sketch-matrix-objective-consistency-not-the-gauge-fix). This is the Stage-A path and also the standalone inner
loop a `Participant` runs per round (see [§1.3](#13-coordinator-and-participant)).

- Returns: `RunResult` — references to the produced `Checkpoint`, the `RunManifest`, and the final-step
  scalar metrics (`loss/pred`, `loss/sigreg`, `loss/anchor`, `grad_norm`); see
  [05 — Observability](05-observability.md).
- Preconditions: `config` validated; the V-JEPA 2 warm-start weights resolvable; the configured dataset
  ingestible by the data layer ([RFC-0004 §2](../rfcs/RFC-0004-data-provenance.md#2-residency-the-sovereignty-guarantee-inv-residency)). At round 0 the encoder weights
  are hash-identical to the pinned warm-start (`INV-WARMSTART-T0`, enforced in `model`/`gauge`).
- Postconditions: a schema-versioned, hash-committed `Checkpoint` (`INV-CHECKPOINT-HASH`); a complete
  `RunManifest`.
- Errors: `ConfigError` (invalid config), `ContractViolation` (a `LatentState` or `ActionSpec` that does
  not conform to the pinned `wmcp_version`, `INV-WMCP`), `ResidencyViolation` (fail-closed: any attempt
  to serialize raw observations/actions/private embeddings, `INV-RESIDENCY`), `ArtifactError`
  (`CheckpointIntegrityError` on a hash mismatch when writing/loading), `EvaluationError` (if an inline
  eval is configured and fails).
- Determinism: best-effort and seed-pinned; full determinism is enabled by the config flag that sets
  `torch.use_deterministic_algorithms(True)`. CUDA primary; a CPU path runs the small CI configs.

```python
cfg = lensemble.config.load(Path("configs/stage_a.yaml"))
result = lensemble.train_local(cfg)
print(result.checkpoint.content_hash, result.metrics["loss/pred"])
```

### 1.3 `Coordinator` and `Participant`

```python
from lensemble.federation import Coordinator, Participant, RoundState

class Coordinator:
    def run(self, num_rounds: int) -> None: ...

class Participant:
    def local_round(self, global_state: "GlobalState", round_seed: int) -> "PseudoGradient": ...
```

The federation roles (Stage B+). The `Coordinator` orchestrates the DiLoCo outer loop, holds the
canonical global model, drives the `RoundState` machine, and runs the outer Nesterov step
([RFC-0003 §3](../rfcs/RFC-0003-federated-protocol.md#3-the-pseudogradient-contract)). A `Participant` holds sovereign data, runs `H` inner steps
locally, and emits a `PseudoGradient`. The state machine, runtime classes, and fault tolerance are
specified in [RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md); secure aggregation in
[RFC-0011](../rfcs/RFC-0011-secure-aggregation.md); DP in [RFC-0012](../rfcs/RFC-0012-differential-privacy.md).

`Coordinator.run(num_rounds)`:

- Precondition: every joining participant is WMCP-conformant on the pinned `wmcp_version` and pinned the
  same probe content-hash (`INV-PROBE-PIN`). All round-`t` participants use the identical sketch matrix
  `A` derived from the broadcast seed `s_t` (`INV-SKETCH-CONSISTENCY`).
- Postcondition: after each outer step the new `(θ_{t+1}, φ_{t+1})` is hash-committed
  (`INV-CHECKPOINT-HASH`); the outer step is a pure, bitwise-reproducible function of (committed deltas,
  round seed, prior global params) (`INV-AGG-DETERMINISM`, enforced in `federation`/`aggregation` by a
  per-step determinism self-check).
- Errors: `RoundError` → `FaultToleranceExceeded` (too few participants remain to complete a round);
  `AggregationError` → `SecureAggregationError` (live set below the dropout threshold),
  `NonDeterministicAggregation` (the determinism self-check fails — abort and recompute, never swallow);
  `ProvenanceError` → `CommitmentMismatch` (a released delta is not bound to exactly one dataset Merkle
  root `R_c`, `INV-COMMIT-BINDING`; reject the update), `MerkleVerificationError`;
  `PrivacyBudgetExceeded` (the `(ε,δ)` budget over the planned rounds is spent — stop training);
  `GaugeError` → `FrameDriftExceeded` (drift past threshold triggers the Layer-3 Procrustes backstop;
  if the backstop cannot recover, the error propagates), `DegenerateProcrustes` (near-degenerate SVD;
  the implementation clamps/conditions before raising).

`Participant.local_round(global_state, round_seed)`:

- Precondition: `global_state.probe_hash` matches the participant's pinned probe; `round_seed` derives
  the sketch matrix shared by all participants this round.
- Postcondition: returns a `PseudoGradient` carrying the flat delta, its L2 norm, and the bound
  `dataset_root` (`R_c`). After clipping and before noising, `‖Δ_c‖ ≤ C_clip` (`INV-DP-BOUND`, enforced
  in `privacy.dp`). Per-embodiment action heads `h_ψ^(c)` are never included (`INV-ACTIONHEAD-LOCAL`).
- Errors: `ResidencyViolation` (fail-closed), `ContractViolation`, `PrivacyBudgetExceeded`,
  `ProbeError` (probe hash mismatch / under-coverage).

```python
participants = [Participant(cfg, dataset=ds_c) for ds_c in local_datasets]
coord = Coordinator(cfg, participants=participants)
coord.run(num_rounds=200)  # raw data never crosses a boundary; only Δ_c, under secure-agg + DP
```

### 1.4 `build_encoder`, `build_predictor`, `build_action_head`, `Objective`

```python
from lensemble.model import build_encoder, build_predictor, build_action_head, Objective
# build_encoder(cfg) -> Encoder      ; Encoder.__call__(clip: Tensor) -> LatentState
# build_predictor(cfg) -> Predictor  ; Predictor.__call__(latent: LatentState, action: Tensor) -> LatentState
# build_action_head(cfg, spec: ActionSpec) -> ActionHead
```

Model construction, WMCP-conformant. The encoder is a video ViT warm-started from released V-JEPA 2
weights and co-trained (Fork B); the predictor is the LeWM `ARPredictor`-shaped compact transformer that
predicts future latents action-conditioned; the action head is per-embodiment and local. Implementation
and numerical contracts are in [RFC-0008](../rfcs/RFC-0008-model-objective-numerics.md); the latent/action contract
in [RFC-0007](../rfcs/RFC-0007-wmcp-latent-contract.md).

- `Encoder.__call__(clip) -> LatentState`: emits a `LatentState` of shape `(N, d)` conforming to the
  pinned `wmcp_version` (`INV-WMCP`). Errors: `ContractViolation` on a shape/dtype/semantics mismatch.
- `build_action_head(cfg, spec)`: validates `spec` (an `ActionSpec`: dimensionality, bounds,
  discrete/continuous, embodiment id, units) before constructing the head (`INV-WMCP`). Errors:
  `ContractViolation` on an invalid or unsupported `ActionSpec`.
- `Objective`: computes the three-term loss `λ_pred · L_pred + λ_sig · SIGReg_A + λ_anc · L_anchor`
  ([conventions §2](conventions.md#2-mathematical-notation) notation; [RFC-0002 §3](../rfcs/RFC-0002-gauge-and-aggregation.md#3-layer-1--shared-sketch-matrix-objective-consistency-not-the-gauge-fix)) and returns the total plus each per-term
  scalar for logging. The SIGReg target uses the shared sketch matrix `A` (`INV-SKETCH-CONSISTENCY`); the
  anchor targets derive only from `f_ref` (`INV-PROBE-PIN`). The default prediction target remains the
  existing stop-gradiented JEPA-family path. Claim-grade LeWorldModel base runs set
  `objective.target_stop_gradient=false`, so `L_pred` compares against the live `f_theta(o_{t+1})` target
  branch without EMA/teacher/target stop-gradient (#191).
- Determinism: bf16 forward, fp32 accumulation for loss/statistics; deterministic mode via the config
  flag.

```python
enc = lensemble.model.build_encoder(cfg)
pred = lensemble.model.build_predictor(cfg)
obj = Objective(cfg)
z_t = enc(clip_t)                      # LatentState (N, d)
loss = obj(enc, pred, batch)           # returns total + {loss/pred, loss/sigreg, loss/anchor}
```

### 1.5 `evaluate` and `Planner`

```python
from lensemble.eval import evaluate, Planner
# evaluate(checkpoint: Path, env_id: str, *, cfg) -> "EvalReport"
```

Latent model-predictive control plus metrics, wrapping `stable-worldmodel`'s envs and `world.evaluate`.
The planner minimizes an `L1` goal-energy in latent space (CEM / iCEM / MPPI); see
[RFC-0005 §3](../rfcs/RFC-0005-evaluation.md#3-downstream-metric--planning-success).

- `evaluate(checkpoint, env_id, *, cfg) -> EvalReport`: loads the schema-versioned checkpoint, runs
  latent MPC on `env_id` over held-out factors-of-variation, returns an `EvalReport` (success rate,
  planning samples, time/action; full schema in [03 — Data Model](03-data-model.md)).
- Precondition: the checkpoint's `wmcp_version` matches the runtime; the env id is registered in the
  data layer.
- Errors: `ArtifactError` (`SchemaVersionMismatch` on an unknown/too-new checkpoint schema,
  `CheckpointIntegrityError` on a hash mismatch), `ContractViolation` (latent nonconformance),
  `EvaluationError` (planning/rollout failure or an unregistered env).
- Determinism: planning seeds are recorded in the returned report; with a fixed seed and pinned env the
  success-rate computation is reproducible.

```python
report = lensemble.eval.evaluate(Path("ckpts/round_200.safetensors"), env_id="pusht", cfg=cfg)
print(report.success_rate, report.time_per_action_ms)
```

### 1.6 `frame_drift` and `procrustes_align`

```python
from lensemble.gauge import frame_drift, procrustes_align
# frame_drift(embeddings: Mapping[str, Tensor]) -> "FrameDriftReport"
# procrustes_align(source: Tensor, target: Tensor) -> tuple[Tensor, float]  # (Q*, residual)
```

The gauge diagnostics. `frame_drift` is the implementation behind the headline empirical artifact
([RFC-0002 §9](../rfcs/RFC-0002-gauge-and-aggregation.md#9-the-frame-drift-diagnostic-the-headline-measurement), [RFC-0005 §2](../rfcs/RFC-0005-evaluation.md#2-headline-diagnostic--latent-frame-drift)); its emission schema
is in [05 — Observability](05-observability.md).

- `frame_drift(embeddings)`: `embeddings` maps a participant/global label to that model's probe
  embeddings `f_θ(P)`. Returns a `FrameDriftReport` with the per-pair Procrustes residual and mean
  rotation angle, plus drift-from-global. Deterministic given committed weights and the pinned probe
  (`INV-PROBE-PIN`); the per-pair cost is `O(C²)`.
- `procrustes_align(source, target) -> (Q*, residual)`: returns the optimal orthogonal `Q* ∈ O(d)`
  (closed form `Q* = V Uᵀ` from the SVD of `targetᵀ source`) and the post-alignment Frobenius residual.
- Errors: `GaugeError` → `DegenerateProcrustes` (near-degenerate singular values that cannot be
  conditioned), `FrameDriftExceeded` (only when called in the enforcing aggregation path with a
  threshold; the bare diagnostic does not raise on high drift, it reports it).

```python
embs = {"c0": f0(P), "c1": f1(P), "global": fg(P)}
report = lensemble.gauge.frame_drift(embs)
Q_star, residual = lensemble.gauge.procrustes_align(f1(P), fg(P))
```

### 1.7 `commit_dataset`, `DatasetCommitment`, `ContributionLedger`

```python
from lensemble.provenance import commit_dataset, DatasetCommitment, ContributionLedger
# commit_dataset(dataset) -> DatasetCommitment  # carries Merkle root R_c
```

Provenance. `commit_dataset` content-hashes each episode and builds the dataset Merkle root `R_c`,
returning a `DatasetCommitment` (root + episode count + WMCP metadata; schema in
[03 — Data Model](03-data-model.md)). The Merkle scheme and inclusion proofs are in
[RFC-0014](../rfcs/RFC-0014-provenance-commitments.md). The canonical Phase-1 commitment hash is SHA-256.

- Postcondition: `R_c` binds the dataset; a released `Δ_c` is associated with exactly one `R_c`
  (`INV-COMMIT-BINDING`, enforced in `provenance`/`federation`).
- `ContributionLedger`: an append-only log of `(round, contributing participants, their roots, resulting
  global-model hash)` — the Phase-1 audit substrate the Phase-2 layer formalizes.
- Errors: `ProvenanceError` → `CommitmentMismatch` (a delta bound to the wrong root — rejected, never
  swallowed), `MerkleVerificationError` (an inclusion proof fails).

```python
commitment = lensemble.provenance.commit_dataset(local_dataset)
ledger = ContributionLedger.open(Path("ledger.jsonl"))
commitment.root  # R_c, SHA-256 Merkle root
```

### 1.8 `recompute_alignment`

```python
from lensemble.verify import recompute_alignment
```

Phase-1 public recomputation. `recompute_alignment` reproduces the coordinator's frame alignment from
the public probe plus the committed weights, so anyone can recompute and check it without a ZK proof
([RFC-0002 §5](../rfcs/RFC-0002-gauge-and-aggregation.md#5-layer-3--procrustes-re-alignment-at-aggregation-backstop), [RFC-0006 §3](../rfcs/RFC-0006-verifiable-contribution.md#3-phase-1-proof-ready-requirements-cheap-to-honor-now)).

- Signature (proof-ready, Phase 1): `recompute_alignment(committed_weights: Path, probe: Path) ->
  FrameDriftReport`. Deterministic given the inputs (`INV-PROBE-PIN`, `INV-CHECKPOINT-HASH`).
- Errors: `ProbeError` (probe hash mismatch / under-coverage), `ArtifactError`
  (`CheckpointIntegrityError` / `SchemaVersionMismatch`), `GaugeError` → `DegenerateProcrustes`.
- Phase-2 proof entry points (the STARK prover/verifier) raise `NotImplementedError` in Phase 1; they
  are not part of the frozen `1.0` surface and are tracked in
  [RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md).

```python
report = lensemble.verify.recompute_alignment(
    committed_weights=Path("ckpts/round_200.safetensors"), probe=Path("probe/v1.lance")
)
```

---

## 2. Invariants touched by the public surface

The public API is where several named invariants are observable. Each is enforced in the module noted
and fires the listed error on violation; full statements are in [03 — Data Model](03-data-model.md) and
[04 — Error Model](04-error-model.md).

| Invariant | Where enforced | Error on violation |
|---|---|---|
| `INV-RESIDENCY` | `data.residency` (gates `train_local`, `Participant.local_round`) | `ResidencyViolation` (fail-closed) |
| `INV-WARMSTART-T0` | `model`/`gauge` (round-0 of `train_local`/`Coordinator`) | `GaugeError` / `ArtifactError` |
| `INV-SKETCH-CONSISTENCY` | `model.sigreg` (objective in `Coordinator`/`Participant`) | `ContractViolation` |
| `INV-AGG-DETERMINISM` | `federation`/`aggregation` (`Coordinator.run` self-check) | `NonDeterministicAggregation` |
| `INV-PROBE-PIN` | `data.probe` / `gauge` (`frame_drift`, `recompute_alignment`) | `ProbeError` |
| `INV-COMMIT-BINDING` | `provenance`/`federation` (`commit_dataset`, aggregation) | `CommitmentMismatch` |
| `INV-CHECKPOINT-HASH` | `artifacts` (checkpoint write/load in all paths) | `CheckpointIntegrityError` |
| `INV-DP-BOUND` | `privacy.dp` (`Participant.local_round`) | `PrivacyBudgetExceeded` / `ConfigError` |
| `INV-WMCP` | `contracts` (`build_*`, encoder/predictor calls) | `ContractViolation` |
| `INV-ACTIONHEAD-LOCAL` | `federation` (broadcast/aggregate paths) | `ContractViolation` |

---

## 3. Stability & versioning policy

The full SemVer and deprecation policy is in [09 — Release & Versioning](09-release-and-versioning.md);
this is the API-surface view, normative for what callers may rely on.

- **Public surface** = the symbols in [§1](#1-public-python-surface) re-exported from `lensemble`, the
  `lensemble` CLI commands in [§4](#4-cli-surface), and the on-disk schemas they read/write
  (versioned independently by `schema_version` / `wmcp_version`).
- **Private surface** = any module named `_internal` or any symbol whose name begins with `_`. These are
  unversioned and may change in any release without notice. Importing them is unsupported.
- **Pre-1.0** (minor tracks the milestone: `0.1`→Stage A, `0.2`→Stage B, `0.3`→Stage C): the public
  surface MAY change across minor versions, each change carrying a deprecation note and a changelog entry.
  A symbol is deprecated for one minor, then removed.
- **At `1.0`** the names and signatures in [§1](#1-public-python-surface) and the CLI commands in
  [§4](#4-cli-surface) are frozen under SemVer. Post-`1.0`, a deprecated symbol is kept for two minors.
- **Schema/contract versions** evolve separately: `schema_version: int` on every on-disk artifact
  (forward-compatible readers, explicit migrations, `SchemaVersionMismatch` on unknown/too-new), and
  `wmcp_version: str` on the latent contract (conformance gates on it).

Within a SemVer-compatible range (pre-`1.0`: same minor; post-`1.0`: same major) callers MAY rely on:
the listed symbol names and signatures; the documented error types and their `.code`/`.remediation`
fields; the determinism guarantee on the aggregation/outer-step path (`INV-AGG-DETERMINISM`,
bitwise-reproducible given its inputs); and read-compatibility of on-disk schemas at or below the current
`schema_version`. Callers may NOT rely on: private symbols; the exact wording of remediation strings;
inner-loop bitwise determinism (best-effort, seed-pinned, flag-gated); or the layout of any `_internal`
module.

---

## 4. CLI surface

The CLI is a Typer app exposed as `lensemble`. Every command accepts `--config PATH` and Hydra-style
`key=value` overrides (positional, after the command), composes the `LensembleConfig`, and emits a
`RunManifest`. The CLI mirrors the Python surface; the config system is specified in
[RFC-0009](../rfcs/RFC-0009-configuration-reproducibility.md).

**Override semantics.** `--config` selects the base config; trailing `key=value` tokens override config
groups using Hydra dot-paths (e.g. `model.size=vit_l federation.H=200`). An unknown key or invalid value
raises `ConfigError` and exits non-zero before any work begins (validation at the boundary).

**Stdout / exit-code contract.** Human-readable progress and tables go to stderr (rich-rendered);
machine-readable output (the `RunManifest` path, content hashes, report JSON) goes to stdout. Exit codes:
`0` success; `1` a `LensembleError` (the message includes `.code` and `.remediation`); `2` a CLI usage
error (Typer); `130` interrupted. Every successful run writes a `RunManifest` to the run directory and
prints its path to stdout.

| Command | Purpose | Key arguments / flags | Emits |
|---|---|---|---|
| `train` | Single-site / Stage-A training (`train_local`) | `--config`, `key=value`, `--out DIR` | `Checkpoint`, `RunManifest` |
| `federate coordinator` | Run the coordinator outer loop (`Coordinator.run`) | `--config`, `--num-rounds N`, `--listen ADDR` | per-round `Checkpoint`, `ContributionRecord`, `RunManifest` |
| `federate participant` | Run a participant inner loop (`Participant`) | `--config`, `--coordinator ADDR`, `--data PATH` | `PseudoGradient` (over the wire), `RunManifest` |
| `eval` | Latent-MPC evaluation (`evaluate`) | `--config`, `--checkpoint PATH`, `--env ENV_ID` | `EvalReport` |
| `probe build` | Build a probe set from public/licensed data | `--config`, `--source PATH`, `--out PATH` | probe artifact + content hash |
| `probe pin` | Pin/record a probe content hash (`INV-PROBE-PIN`) | `--probe PATH` | the pinned hash to stdout |
| `probe verify` | Verify a probe matches its pinned hash | `--probe PATH`, `--hash HEX` | exit `0`/`1`; `ProbeError` on mismatch |
| `commit dataset` | Build a dataset Merkle root (`commit_dataset`) | `--config`, `--data PATH` | `DatasetCommitment` (root `R_c`) |
| `drift` | Compute the frame-drift diagnostic (`frame_drift`) | `--config`, `--checkpoints PATHS`, `--probe PATH` | `FrameDriftReport` |
| `verify recompute` | Public alignment recomputation (`recompute_alignment`) | `--config`, `--checkpoint PATH`, `--probe PATH` | `FrameDriftReport` |
| `verify prove` | Phase-2 aggregation proof entry point | `--config` | exits non-zero (`NotImplementedError`) in Phase 1 |
| `doctor` | Environment / dependency / determinism self-check | `--config` | a diagnostic report; non-zero on a failed check |

Notes:

- `verify prove` is reserved for Phase 2 ([RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md)); in Phase 1 it
  exits non-zero with a remediation pointing at the proof-ready path.
- `federate participant` enforces `INV-RESIDENCY` fail-closed: no raw observation/action/embedding ever
  reaches the wire; only the DP-clipped, secure-agg-masked `PseudoGradient` does.
- `doctor` validates Python/torch versions, resolvable warm-start weights, the determinism flag, and the
  aggregation determinism self-check; it is the first thing a new contributor runs.

### 4.1 Invocation examples

```bash
# Stage-A single-site training with overrides; writes a RunManifest under runs/.
lensemble train --config configs/stage_a.yaml model.size=vit_l federation.H=200 --out runs/a1

# Commit a local dataset, then start a participant against a coordinator.
lensemble commit dataset --config configs/stage_b.yaml --data /data/silo_c0
lensemble federate participant --config configs/stage_b.yaml \
    --coordinator coord.internal:7000 --data /data/silo_c0

# Run the coordinator outer loop for 200 rounds.
lensemble federate coordinator --config configs/stage_b.yaml --num-rounds 200 --listen :7000

# Reproduce the headline frame-drift figure from committed weights + the pinned probe (no data needed).
lensemble drift --config configs/stage_b.yaml \
    --checkpoints runs/a1/ckpts/round_*.safetensors --probe probe/v1.lance
```

---

## 5. Extension points

Two extension surfaces are stable across the staged plan: new embodiments and new data adapters. Both
are gated by conformance checks so a nonconforming extension fails closed at registration, not mid-round.

### 5.1 Adding a new embodiment (action head + `ActionSpec`)

A new embodiment supplies an `ActionSpec` (dimensionality, bounds, discrete/continuous, embodiment id,
units) and a conforming action head `h_ψ^(c)` that maps that action space into the shared
latent-conditioning space. The contract is in [RFC-0007](../rfcs/RFC-0007-wmcp-latent-contract.md).

```python
spec = ActionSpec(embodiment_id="quadruped_v1", dim=12, kind="continuous",
                  low=..., high=..., units="rad")
head = lensemble.model.build_action_head(cfg, spec)   # validates spec (INV-WMCP), raises ContractViolation
```

- The head is per-participant and local: it is never broadcast or aggregated (`INV-ACTIONHEAD-LOCAL`,
  enforced in `federation`).
- `build_action_head` validates the `ActionSpec` against the pinned `wmcp_version` before constructing;
  an invalid or unsupported spec raises `ContractViolation`.
- The head must conform to the shared `LatentState` interface so its conditioning embedding is consumable
  by the predictor `g_φ` (`INV-WMCP`).

### 5.2 Registering a new data adapter

The data layer supports `lance` (default), `hdf5` (portable), the `lerobot://<repo_id>` Hub adapter, and
the local `lerobot-h5://<path>` LeRobot-layout HDF5 adapter
([RFC-0004 §1](../rfcs/RFC-0004-data-provenance.md#1-per-participant-data-layer)). A new adapter implements the dataset/loader protocol from
[RFC-0004](../rfcs/RFC-0004-data-provenance.md) (yields fixed `Window`s of `num_steps` over `Episode`s of
`(o_t, a_t, o_{t+1})` `Transition`s) and is registered with the data layer.

- A registered adapter MUST honor the residency flag and refuse to emit raw data across a boundary
  (`INV-RESIDENCY`, `data.residency`); a violation raises `ResidencyViolation` (fail-closed).
- Episodes ingested through any adapter are hashable for `commit_dataset` so the dataset Merkle root
  `R_c` is well-defined (`INV-COMMIT-BINDING`); a malformed or unhashable episode raises a
  `ProvenanceError` at ingest, not at commit time.
- Each adapter declares minimal data-quality metadata (modality, embodiment, `ActionSpec`, episode
  count); the federation MAY gate on declared quality ([RFC-0004 §6](../rfcs/RFC-0004-data-provenance.md#6-data-quality-metadata-and-the-wmcp-precondition)).

The built-in adapters resolve through one module-level registry in `lensemble.data.adapters`
keyed by `fmt` (or, for read-only LeRobot views, the `lerobot://` / `lerobot-h5://` URI schemes). `save_episodes` /
`load_episodes` are the dispatch entry points; `register_adapter` is the extension point:

```python
from lensemble.data import save_episodes, load_episodes, register_adapter

save_episodes(dataset, "silo_c0.lance", fmt="lance")    # default reference store
ds = load_episodes("silo_c0.lance")                       # fmt inferred from the .lance suffix
ds = load_episodes("lerobot://lerobot/pusht")             # read-only, conformance-checked on load
ds = load_episodes("lerobot-h5:///data/silo0.h5")         # local LeRobot-layout HDF5, read-only

register_adapter("myfmt", loader=my_load, saver=my_save)  # a new backend plugs in here
```

- `save_episodes(..., fmt="lerobot")` raises: the `lerobot://` view is read-only by construction
  (it registers no saver), so it never participates in commitment or egress.
- `save_episodes(..., fmt="lerobot-h5")` raises for the same reason: the adapter directly reads a local
  LeRobot-layout HDF5 export as resident raw data and registers no saver.
- A `loader` returns a read-back `EpisodeDataset` carrying the same `fmt`; an omitted `saver` makes the
  adapter read-only. Both run inside the trust boundary on local files only.

---

## 6. Open questions

OPEN QUESTION: the exact `RunResult` field set returned by `train_local` (beyond the `Checkpoint`,
`RunManifest`, and final-step scalar metrics named here) is fixed in
[RFC-0009](../rfcs/RFC-0009-configuration-reproducibility.md). Owner @AbdelStark; resolution in v0.1 / Stage A.

OPEN QUESTION: the transport for `federate coordinator|participant` (`ADDR` form, gRPC vs HTTP) is
deferred to [RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md); the CLI argument shape (`--coordinator ADDR`,
`--listen ADDR`) is stable, the wire transport is not yet pinned. Owner @AbdelStark; resolution in
Stage C.

OPEN QUESTION: the Phase-2 `verify prove` signature and its STARK prover/verifier surface are out of the
frozen `1.0` API; they are specified in [RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md). Owner @AbdelStark;
resolution post-`v1.0` (Stage D).

---

## References

- Public surface, stability policy, milestone↔stage mapping: the [conventions document](conventions.md) ([§5](conventions.md#5-public-api-surface), [§12](conventions.md#12-milestones-and-stages)).
- [03 — Data Model](03-data-model.md) — schemas for every type referenced here.
- [04 — Error Model](04-error-model.md) — the error taxonomy and `LensembleErrorCode`.
- [05 — Observability](05-observability.md) — metric names and the frame-drift emission schema.
- [09 — Release & Versioning](09-release-and-versioning.md) — SemVer, deprecation, changelog.
- [RFC-0001](../rfcs/RFC-0001-architecture.md) — module map and federation map.
- [RFC-0002](../rfcs/RFC-0002-gauge-and-aggregation.md) — the gauge, Procrustes, frame anchoring.
- [RFC-0003](../rfcs/RFC-0003-federated-protocol.md) — the round protocol, DP clip+noise, messages.
- [RFC-0004](../rfcs/RFC-0004-data-provenance.md) — data layer, probe, commitments.
- [RFC-0005](../rfcs/RFC-0005-evaluation.md) — evaluation, latent MPC, the frame-drift diagnostic.
- [RFC-0006](../rfcs/RFC-0006-verifiable-contribution.md) — Phase-1 proof-readiness, Phase-2 proofs.
- [RFC-0007](../rfcs/RFC-0007-wmcp-latent-contract.md) — `LatentState` / `ActionSpec` / action-head contract.
- [RFC-0008](../rfcs/RFC-0008-model-objective-numerics.md) — model and objective numerical contracts.
- [RFC-0009](../rfcs/RFC-0009-configuration-reproducibility.md) — config system and `RunManifest`.
- [RFC-0011](../rfcs/RFC-0011-secure-aggregation.md), [RFC-0012](../rfcs/RFC-0012-differential-privacy.md),
  [RFC-0013](../rfcs/RFC-0013-coordinator-runtime.md), [RFC-0014](../rfcs/RFC-0014-provenance-commitments.md) — federation
  runtime, secure aggregation, DP, provenance commitments.
