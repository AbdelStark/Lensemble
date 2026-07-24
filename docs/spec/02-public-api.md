# 02 — Public API

This page documents the Python and CLI behavior implemented by the current `0.1.x` package. It is an
API inventory, not a promise that every research workflow is a one-call product surface. In particular,
the low-level federation roles require an explicit transport, participant-local data, and round state;
the CLI still contains readiness checks and manifest-only stubs alongside commands that execute real
work.

The package requires Python `>=3.11`. Error classes and their structured `.code` and `.remediation`
fields are described in [04 — Error Model](04-error-model.md). On-disk models are described in
[03 — Data Model](03-data-model.md).

## 1. Public Python surface

`lensemble.__all__` is the authoritative set of top-level convenience exports in the current release:

```python
import lensemble

assert lensemble.__version__ == "0.1.0"
```

| Area | Top-level exports |
|---|---|
| Configuration | `LensembleConfig`, `RunManifest`, `load` |
| Training and federation | `train_local`, `Coordinator`, `Participant`, `RoundState` |
| Models | `build_encoder`, `build_predictor`, `build_action_head`, `Objective` |
| Evaluation | `evaluate`, `Planner` |
| Gauge diagnostics | `frame_drift`, `procrustes_align` |
| Provenance | `commit_dataset`, `DatasetCommitment`, `ContributionLedger` |
| Public recomputation | `recompute_alignment` |

The exports are resolved lazily, so `import lensemble` does not itself import PyTorch. Supporting types
such as `ActionSpec`, `EpisodeDataset`, `InProcessTransport`, `EvalReport`, and `ContributionRecord`
remain available from their owning subpackages.

### CPU-safe end-to-end example

This example uses the built-in, read-only `synthetic-dynamic://` data source, a tiny scratch encoder,
and the built-in `synthetic://toy` evaluation world. It downloads no checkpoint and is suitable for a
clean CPU checkout.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import lensemble

cfg = lensemble.load(
    overrides=[
        "model.encoder=scratch",
        "model.latent_dim=8",
        "model.num_tokens=4",
        "model.predictor_depth=1",
        "model.predictor_width=8",
        "model.num_frames=1",
        "model.tubelet=1",
        "model.image_size=4",
        "model.patch_size=2",
        "model.depth=1",
        "model.num_heads=2",
        "objective.lambda_anc=0.0",
        "objective.sigreg_sketch_dim=8",
        "gauge.anchor_landmark_count=8",
        "federation.inner_horizon=1",
        "data.format=synthetic-dynamic",
        (
            "data.data_source=synthetic-dynamic://swipe-dot"
            "?seed=0&n_episodes=1&steps=2&image_size=4"
        ),
        "eval.env_id=synthetic://toy",
        "eval.planning_samples=8",
        "eval.horizon=2",
    ]
)

with TemporaryDirectory() as tmp:
    trained = lensemble.train_local(cfg, run_dir=Path(tmp))
    assert (trained.checkpoint_dir / "weights.safetensors").is_file()
    assert len(trained.checkpoint_hash) == 64

    report = lensemble.evaluate(
        trained.checkpoint_dir,
        "synthetic://toy",
        cfg=cfg,
        num_episodes=2,
        planner_iters=1,
    )
    print(report.success_rate, report.effective_dim)
```

The toy world is a deterministic harness fixture, not robotics evidence. Use it to verify plumbing and
API behavior only.

### 1.1 `LensembleConfig`, `load`, and `RunManifest`

The implemented loader signature is:

```python
def load(
    config_name: str = "default",
    overrides: list[str] | None = None,
    *,
    config_dir: Path | None = None,
) -> LensembleConfig: ...
```

`load` takes a config **name**, not a config path. To load `configs/stage_a.yaml`, pass
`config_name="stage_a", config_dir=Path("configs")`. Calling `load()` composes the registered defaults.
Unknown override keys and invalid cross-field combinations raise `ConfigError`.

`LensembleConfig` and its nested groups are frozen dataclasses. The current groups are `model`,
`objective`, `gauge`, `federation`, `privacy`, `data`, `eval`, `observability`, and `determinism`, plus
the root `run_mode`.

`RunManifest` is a frozen Pydantic model. Callers normally obtain one with
`lensemble.config.build_manifest`; hand-authoring all required environment, seed, and hash fields is not
the intended path. `train_local` currently returns a deterministic `manifest_hash`, not a
`RunManifest` object or a `.manifest` field.

### 1.2 `train_local`

```python
def train_local(
    config: LensembleConfig,
    *,
    run_dir: Path | None = None,
) -> RunResult: ...
```

`train_local` resolves participant-local windows from `config.data.data_source`, trains for
`config.federation.inner_horizon` steps, and writes a checkpoint directory containing
`weights.safetensors` and `header.json`. If `run_dir` is omitted it creates a temporary run directory.

The returned frozen `RunResult` currently has exactly these fields:

```python
trained.checkpoint_dir   # Path
trained.checkpoint_hash  # str: checkpoint content hash
trained.manifest_hash    # str: hash of the generated manifest, excluding created_at
trained.final_loss       # float: final objective total
```

There is no `trained.checkpoint.content_hash`, `trained.manifest`, or `trained.metrics` field. With
`objective.lambda_anc > 0`, training also requires a pinned public probe at
`config.data.probe_path`. Missing local data raises `RoundError`; invalid model or action contracts
raise structured configuration/contract errors.

### 1.3 `Coordinator` and `Participant`

These are low-level federation roles, not a constructor-level orchestration API:

```python
class Coordinator:
    def __init__(
        self,
        config: LensembleConfig,
        *,
        transport: Transport,
        artifacts_dir: Path | None = None,
        enable_backstop: bool = False,
        warm_start: dict[str, Tensor] | None = None,
    ) -> None: ...

    def run(self, num_rounds: int) -> None: ...
    def try_round(self) -> RoundState: ...

class Participant:
    def __init__(
        self,
        config: LensembleConfig,
        *,
        participant_id: str,
        transport: Transport,
    ) -> None: ...

    def local_round(
        self,
        global_state: GlobalState,
        round_seed: int,
    ) -> PseudoGradient: ...
```

`Participant` does not accept a `dataset=` keyword. Its default data hooks resolve
`config.data.data_source` inside the participant boundary. `Coordinator` does not accept a
`participants=` keyword; it collects updates from the supplied `Transport`. Consequently,
`Coordinator.run()` is runnable only after the surrounding transport/service layer has staged the
round inputs. It returns `None`, while state and records are available through methods such as
`round_state()`, `global_state_hash()`, and `ledger_records()`.

For complete consortium execution, use the Phase 3 service/orchestration APIs in
`lensemble.federation` or the maintained launcher in `deploy/hfjobs/train_phase3_consortium.py`.
Constructing `Coordinator` and `Participant` objects in one list does not itself wire a federation.

### 1.4 Model builders and `Objective`

The model builders use the shape fields in `cfg.model`:

```python
encoder = lensemble.build_encoder(cfg)
predictor = lensemble.build_predictor(cfg)
action_head = lensemble.build_action_head(cfg, action_spec)
```

For a batched clip shaped `(B, T, C, H, W)`, the encoder returns a `LatentState` whose tokens are
`(B, N, d)`. The predictor consumes a batched `LatentState` and an action embedding shaped
`(B, cond_dim)`. `build_action_head` requires a validated `ActionSpec` and returns local,
per-embodiment state; the shared checkpoint contains encoder and predictor weights only.

`Objective` is configured explicitly with keyword arguments:

```python
objective = lensemble.Objective(
    lambda_pred=1.0,
    lambda_sig=0.1,
    lambda_anc=0.0,
    sketch_seed=0,
    sketch_dim=8,
    target_stop_gradient=True,
)

terms = objective(encoder, predictor, window, action_head.encode(window.actions))
terms.pred
terms.sigreg
terms.anchor
terms.total  # the scalar tensor used for backward()
```

`Objective(cfg)` is not a supported signature. The participant and local-training paths translate the
relevant config fields into the constructor shown above.

### 1.5 `evaluate` and `Planner`

```python
def evaluate(
    checkpoint: Path,
    env_id: str,
    *,
    cfg: LensembleConfig,
    num_episodes: int = 4,
    planner_iters: int = 4,
) -> EvalReport: ...
```

`checkpoint` is a checkpoint **directory**, not a standalone `.safetensors` file. `evaluate` verifies
the artifact, rebuilds the encoder and predictor, resolves `env_id`, and returns an `EvalReport` with
these metric fields:

- `success_rate`
- `planning_samples`
- `time_per_action_ms`
- `effective_dim`
- `probe_accuracy`
- `state_probe_r2`
- `checkpoint_hash`
- `run_manifest_hash`

`state_probe_r2` is `None` for worlds without resident state labels. The built-in
`synthetic://toy` and `kinematic://swipe-dot` worlds are CPU-safe. A
`stable-worldmodel://...` identifier currently fails with `EvaluationError` unless a deployment wires
that optional environment suite.

The standalone planner constructor is:

```python
planner = lensemble.Planner(
    family="icem",
    horizon=4,
    num_samples=32,
    action_dim=2,
    seed=0,
    num_iters=2,
)
result = planner.plan(dynamics, initial_latent, goal_latent)
```

`result` is a `PlanResult` with `actions`, `cost`, `planner`, `num_samples`, `num_iters`, and
`wall_time_s`.

### 1.6 `frame_drift` and `procrustes_align`

```python
def frame_drift(
    embeddings: Mapping[str, Tensor],
    *,
    round_index: int = 0,
    probe: PublicProbe | None = None,
    expected_probe_hash: str | None = None,
    degenerate_safe: bool = False,
) -> FrameDriftReport: ...

def procrustes_align(
    source: Tensor,
    target: Tensor,
    *,
    singular_floor: float = 1e-6,
) -> tuple[Tensor, float]: ...
```

A small geometry-only example:

```python
import torch

points = torch.tensor(
    [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
)
rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
rotated = points @ rotation

q_star, residual = lensemble.procrustes_align(points, rotated)
report = lensemble.frame_drift(
    {"site-a": points, "site-b": rotated},
    round_index=0,
)
pair = report.pairs[0]
print(pair.rotation_angle_deg, pair.procrustes_residual)
```

`FrameDriftReport` has `round_index`, `probe_hash`, `pairs`, and `drift_from_global`. Rotation angles
and residuals live on each `PairDrift`; there is no report-level `mean_rotation_angle` field. The
example above intentionally omits a probe and therefore is only a geometry check. Claim-grade
diagnostics pass both `probe` and `expected_probe_hash`.

### 1.7 Provenance

```python
def commit_dataset(dataset: EpisodeDataset) -> DatasetCommitment: ...
```

`commit_dataset` hashes the logical episode set and returns a frozen `DatasetCommitment`. Its root field
is named `merkle_root`, not `root`:

```python
from lensemble.data import load_episodes

dataset = load_episodes(
    (
        "synthetic-dynamic://swipe-dot"
        "?seed=0&n_episodes=1&steps=2&image_size=4"
    ),
    fmt="synthetic-dynamic",
)
commitment = lensemble.commit_dataset(dataset)
assert len(commitment.merkle_root) == 64
assert commitment.episode_count == 1
```

`ContributionLedger` is a JSONL-backed, append-only hash chain. Its constructor is
`ContributionLedger(path: Path, records: Sequence[ContributionRecord])`. The current implementation
also provides the classmethod `ContributionLedger.open(path)`, plus `append(record)`, the read-only
`records` property, and `verify_chain()`. These methods are current pre-1.0 behavior, not a separately
frozen protocol surface.

### 1.8 `recompute_alignment`

```python
def recompute_alignment(
    committed_weights: Path,
    probe: Path,
) -> FrameDriftReport: ...
```

Both arguments are artifact paths: `committed_weights` is a self-describing checkpoint directory and
`probe` is a probe artifact written by `save_probe`. The result records the committed encoder's
measured alignment to the pinned reference in `drift_from_global["committed"]`.

This is public recomputation of a measurement. It does not prove honest participant computation and
does not prove that an activation-space backstop was applied.

## 2. Invariants touched by the public surface

The public functions expose, but do not each independently establish, the repository's invariants:

| Invariant | Current enforcement point |
|---|---|
| `INV-RESIDENCY` | data egress guards, artifact schemas, and boundary payloads |
| `INV-SKETCH-CONSISTENCY` | per-round `Objective` construction |
| `INV-AGG-DETERMINISM` | deterministic aggregation and outer-step checks |
| `INV-PROBE-PIN` | probe verification in gauge/recomputation paths |
| `INV-COMMIT-BINDING` | provenance binding checks and coordinator update acceptance |
| `INV-CHECKPOINT-HASH` | checkpoint save/load |
| `INV-WMCP` | latent and action-contract gates |
| `INV-ACTIONHEAD-LOCAL` | shared checkpoint and aggregation parameter-group boundaries |

See the owning RFC and [03 — Data Model](03-data-model.md) for the precise statement of each invariant.

## 3. Stability & versioning policy

The package version is currently `0.1.0`. The top-level names above are the supported convenience
surface for this release, but they are **not frozen**:

- Before `1.0`, names, signatures, and return models may change in a minor release with release notes
  and the deprecation policy in [09 — Release & Versioning](09-release-and-versioning.md).
- Patch releases should preserve documented `0.1.x` behavior unless a correctness or security fix
  requires otherwise.
- A leading underscore denotes private, unversioned behavior. Public subpackages expose additional
  specialist APIs, but top-level re-export is the narrow compatibility surface documented here.
- On-disk `schema_version` values and the WMCP protocol version evolve independently from package
  SemVer. Use the owning parse/load function so too-new schemas fail explicitly.
- The literal `wmcp_version="wmcp-1.0.0"` is the latent protocol version; it does not mean the Python
  package API is at `1.0`.

The planned `1.0` freeze is future work. Documentation that describes the present `0.1.x` surface as
already frozen is incorrect.

## 4. CLI surface

Use `uv run lensemble --help` and `uv run lensemble <command> --help` for the installed command tree.
The current CLI mixes real execution, readiness/preflight commands, and explicit stubs:

| Command | Current behavior |
|---|---|
| `train` | Executes `train_local`; prints a skeleton manifest path, checkpoint directory, and checkpoint hash. |
| `eval` | Always validates config and writes a skeleton manifest; executes `evaluate` only when `--checkpoint` is supplied. |
| `federate coordinator` | Instantiates an in-process coordinator and reports readiness/initial hash; it does not run a connected federation. |
| `federate participant` | Instantiates an in-process participant and reports readiness; it does not execute a local round. |
| `federate coordinator-service` | Validates Phase 3 inputs and emits a control-plane report/trace; it is not a long-running network server by itself. |
| `federate participant-agent` | Runs Phase 3 participant preflight; assigned-round execution still requires the coordinator service/transport. |
| `commit dataset` | Manifest-only stub; use the Python `commit_dataset` function for a real commitment. |
| `drift` | Manifest-only stub; use the Python `frame_drift` function for a real diagnostic. |
| `probe build`, `probe pin`, `probe verify` | Execute the corresponding probe artifact operations. |
| `verify recompute` | Executes public alignment recomputation from a checkpoint directory and probe artifact. |
| `verify prove` | Deliberately unavailable; exits non-zero because the STARK prover is not implemented. |
| `doctor` | Executes environment, config, and lightweight determinism checks. |
| `demo federated` | Serves the local educational browser demo. |

Most config-driven commands accept `--config PATH`, `--run-dir PATH`, and trailing Hydra-style
`key=value` overrides. Probe and demo commands have their own option sets. The current CLI
`run_manifest.json` uses the transitional schema string `run-manifest/skeleton-0`; it is not the
Pydantic `RunManifest` schema documented in [03 — Data Model](03-data-model.md).

Machine-oriented values are written to stdout and explanatory status to stderr where the command
supports that split. `LensembleError` maps to exit code `1`, Typer usage errors to `2`, and an interrupt
to `130`.

## 5. Extension points

These extension seams live in subpackages and therefore are broader than the top-level compatibility
surface.

### 5.1 Adding an embodiment

Construct a complete `ActionSpec`; enum values and tuple-valued fields are required:

```python
from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec

spec = ActionSpec(
    embodiment_id="arm-2dof",
    kind=ActionKind.CONTINUOUS,
    dim=2,
    low=(-1.0, -1.0),
    high=(1.0, 1.0),
    num_classes=None,
    units=("rad", "rad"),
    wmcp_version=WMCP_VERSION,
)
head = lensemble.build_action_head(cfg, spec)
```

The head output width is the model conditioning dimension. Its parameters remain local and are not
part of shared checkpoints.

### 5.2 Registering a new data adapter

The runtime extension signature is:

```python
from lensemble.data import register_adapter

register_adapter(
    "my-format",
    loader=my_load,       # (str | Path) -> EpisodeDataset
    saver=my_save,        # (EpisodeDataset, Path) -> None; omit for read-only
)
```

The built-ins are `lance`, `hdf5`, read-only `lerobot`, read-only `lerobot-h5`, and read-only
`synthetic-dynamic`. A custom adapter runs inside the participant trust boundary and must preserve the
same episode/action contracts and residency rules. Registration is process-local and replacing a key
replaces its current adapter.

## 6. Error and determinism boundaries

All `LensembleError` subclasses carry `.code` and `.remediation`; see
[04 — Error Model](04-error-model.md). Plain `ValueError` is still used by some mathematical shape
checks and registry lookups, so callers should not assume every invalid argument has already been
migrated to the structured taxonomy.

Aggregation is designed to be bitwise reproducible for fixed inputs on its supported path. Local
training and planning are seed-pinned/best-effort and should not be described as bitwise identical
across hardware. Wall-clock fields such as `time_per_action_ms` and `PlanResult.wall_time_s` are
telemetry, not deterministic outputs.

## References

- [03 — Data Model](03-data-model.md)
- [04 — Error Model](04-error-model.md)
- [09 — Release & Versioning](09-release-and-versioning.md)
- [RFC-0002 — Gauge and Aggregation](../rfcs/RFC-0002-gauge-and-aggregation.md)
- [RFC-0004 — Data and Provenance](../rfcs/RFC-0004-data-provenance.md)
- [RFC-0005 — Evaluation](../rfcs/RFC-0005-evaluation.md)
- [RFC-0006 — Verifiable Contribution](../rfcs/RFC-0006-verifiable-contribution.md)
- [RFC-0007 — WMCP Latent Contract](../rfcs/RFC-0007-wmcp-latent-contract.md)
- [RFC-0008 — Model and Objective Numerics](../rfcs/RFC-0008-model-objective-numerics.md)
- [RFC-0009 — Configuration and Reproducibility](../rfcs/RFC-0009-configuration-reproducibility.md)
- [RFC-0013 — Coordinator Runtime](../rfcs/RFC-0013-coordinator-runtime.md)
- [RFC-0014 — Provenance Commitments](../rfcs/RFC-0014-provenance-commitments.md)
