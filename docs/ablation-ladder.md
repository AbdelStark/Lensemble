# The ablation ladder — the core experiment

The ablation ladder is the paper's central experiment ([RFC-0005
§6](rfcs/RFC-0005-evaluation.md#6-ablation-ladder-the-core-experiment)). It realizes the [RFC-0002
§4](rfcs/RFC-0002-gauge-and-aggregation.md#4-layer-2--frame-anchoring-on-a-public-probe-the-gauge-fix)
latent-gauge fix **additively**: rung 1 is the negative control, and each subsequent rung adds exactly
**one** gauge mechanism. At every rung the runner reports all three metric families — frame drift (§2),
MPC success (§3), and effective dimension / collapse (§4) — so three independent signals must agree.

The runner is `lensemble.eval.run_ablation_ladder`; it drives each rung through the live multi-round
federated-simulation harness `lensemble.federation.run_federated_simulation`.

## The five rungs → RFC-0002 §4 layer mapping

One mechanism is added per rung. The runner's `LADDER_RUNGS` table encodes each rung as objective-knob
overrides (the `lambda_sig` / `lambda_anc` weights) plus the mechanism-enablement flags it reads.

| # | Rung (`name`)          | Mechanism added                                   | RFC-0002 layer | Knobs / flags                                  |
|---|------------------------|---------------------------------------------------|----------------|------------------------------------------------|
| 1 | `naive-fedavg`         | — (negative control: end-to-end FedAvg)           | none (§2.1)    | `lambda_sig=0`, `lambda_anc=0`, backstop off   |
| 2 | `shared-sketch`        | + shared sketch matrix `A` (`INV-SKETCH-CONSISTENCY`) | Layer 1 (§3) | `lambda_sig>0`                                 |
| 3 | `procrustes-backstop`  | + Procrustes align-then-average backstop          | Layer 3 (§5)   | + coordinator backstop seam ON                 |
| 4 | `frame-anchor` ← **recommended** | + frame-anchor loss (Variant A landmark) | Layer 2 (§4)   | + `lambda_anc>0` (a real probe is pinned)      |
| 5 | `distillation`         | + function-space distillation                     | Layer 4 (§6)   | + gauge-invariant consensus on the probe frames |

Rung 2 adds the shared sketch — **objective consistency, not the gauge fix**: the SIGReg statistic agrees
across silos, but the latent frame still drifts. Rung 3 adds the Layer-3 backstop alone. Rung 4 — the
**recommended configuration** — adds the frame anchor: `k >= d` generic public-probe landmarks pinned to
fixed round-0 targets `t_i = f_ref(p_i)`, so the only orthogonal map satisfying all constraints is the
identity (*pin the frame, not the content*). Rung 5 adds the heterogeneity / instability fallback —
function-space distillation, gauge-invariant by construction because it compares functions on the shared
probe, never weights.

## What the ladder measures

The expected qualitative ordering (RFC-0005 §6) is **naive worst on drift; anchored flat**. On the
small-config CPU regression run (`tests/ml/test_ablation_ladder.py`), with genuinely different per-silo
data (so the naive frames actually diverge), the measured mean inter-silo frame drift is, for example:

```
naive-fedavg          ~25 deg   (no gauge control — the frame diverges)
shared-sketch         ~25 deg   (sketch fixes the objective, NOT the gauge)
procrustes-backstop   ~22 deg   (the backstop alone — bounded help)
frame-anchor          ~7  deg   (the anchor pins the frame — RECOMMENDED)
distillation          ~7  deg   (same drift; the consensus is gauge-invariant)
```

The regression test asserts the load-bearing claim — the naive rung's drift materially exceeds the
anchored rung's by a clear margin, and the anchored rung's drift is small — rather than a strict 5-rung
monotonic ordering, which is not reliable on a toy CPU budget.

## The `lambda_anc` sweep — the central hyperparameter

`lambda_anc` ([RFC-0002 §7](rfcs/RFC-0002-gauge-and-aggregation.md#7-the-central-hyperparameter)) trades
frame stability against representational freedom:

- **too high** (`>> 1`) → the encoder is clamped to the reference frame *and its quality*;
- **too low** (`-> 0`) → the frame drifts and weight-space averaging degrades.

The sweet spot is a **small positive** `lambda_anc`: warm-start + a small anchor keeps the frame pinned
cheaply, so Layers 3-4 rarely fire. `lensemble.eval.lambda_anc_sweep(base_cfg, values)` resolves each
swept value to a distinct, validated `LensembleConfig` (a config-group override over `objective.lambda_anc`);
the caller drives each through the ladder harness to plot frame drift and MPC success versus `lambda_anc`.

## The §7 sweeps — non-IID severity, C/H, and scale

The [RFC-0005 §7](rfcs/RFC-0005-evaluation.md#7-non-iid-severity--scale-sweeps) robustness sweeps (the
paper's **Claim 4**: the recipe holds across heterogeneity and scale) run **over** the ladder rungs,
reusing the same runner and harness. They split across the [RFC-0001
§3](rfcs/RFC-0001-architecture.md#3-dependency-layering-no-cycles) module band (eval may not import
federation): the **compose** side — the synthetic non-IID partition and the seeded pair-sampling — lives
in `lensemble.eval.sweeps`; the **drivers** that call the harness live one band up in
`lensemble.federation.sweeps`. Each sweep keeps the dims/rounds/silos tiny (CPU-fast) and runs only the
two load-bearing rungs per point — `naive-fedavg` (the negative control) and `frame-anchor` (the
recommended config) — since the directional claim each axis asserts is naive-vs-anchored.

### Non-IID severity — `non_iid_severity_sweep`

For each severity `s ∈ [0, 1]` the synthetic per-silo data is partitioned by a per-silo distribution
shift scaled by `s`, then the rungs run at each `s`; the per-rung drift-degradation curve vs severity is
reported.

- **`s = 0` (near-IID):** every silo draws the *same* synthetic distribution (identical per-silo
  windows), so there is no heterogeneity-induced inter-silo offset.
- **`s = 1` (strongly non-IID):** silo `c`'s observation mean is shifted by `c · unit` (a per-silo factor
  index scaled by the severity), so the per-silo distributions are pulled apart.

The expected trend (Claim 4): the **naive** rung's inter-silo frame drift *grows* with severity (the shift
pulls the unconstrained frames apart), while the **anchored** rung stays low (the Variant-A landmark
anchor pins each frame onto the round-0 reference regardless of the shift). On the small-config CPU run the
measured naive drift roughly *doubles* from near-IID to strong (e.g. ~17° → ~34°) while the anchored rung
holds at ~7°.

#### Partition-by-factor protocol and the deferred real-factors seam (#96)

RFC-0005 §7 calls for partitioning a multi-environment / multi-factor-of-variation corpus across silos by
`stable-worldmodel`'s **factors-of-variation** (controlled, reproducible heterogeneity). That suite is
**not vendored yet** (maintainer-gated, [#96](https://github.com/AbdelStark/Lensemble/issues/96)), so the
severity axis here is **synthetic**: the partition shifts each silo's synthetic toy distribution by a
per-silo mean offset. The real factors-of-variation path is wired as a **documented, fail-closed seam** —
`partition_synthetic_noniid(..., factor=...)` accepts only `factor="synthetic"`; any other value (a real
factor-of-variation name such as `"embodiment"`) raises `EvaluationError` (it never silently falls back to
the synthetic partition), exactly mirroring `lensemble.eval.world.resolve_env`'s `stable-worldmodel://`
fail-closed branch. When the suite is vendored, the real partition is wired behind this same seam.

### Participant count `C` and inner horizon `H` — `participant_horizon_sweep`

Varies `federation.participant_count` (`C`) and `federation.inner_horizon` (`H`) over a grid and runs the
rungs at each `(C, H)` (the round quorum is clamped to `C` so the round closes). A longer `H` rotates the
per-silo frames further apart before the outer step
([RFC-0002 §2.1](rfcs/RFC-0002-gauge-and-aggregation.md#21-three-failures-that-compound-it), DiLoCo
drift), so the **naive** rung's drift *grows with `H`* (e.g. ~20° at `H=8` → ~31° at `H=48`); varying `C`
characterizes the DiLoCo robustness in the JEPA setting.

### Scale — `scale_sweep`

Repeats the key rungs at increasing `model.latent_dim` to show the recipe holds as the encoder grows
(RFC-0005 §7 frames this as ViT-L → V-JEPA-2-class; the CPU step uses tiny dims, e.g. `8 → 16`). Each dim
is a coherent ViT shape via the [#166](https://github.com/AbdelStark/Lensemble/issues/166) bridge —
`num_heads` divides `latent_dim`, and `num_tokens` is a function of the patching geometry (independent of
the hidden dim), so `d` / `cond_dim` / `predictor_width` track `latent_dim` together. The **anchored** rung
stays below the **naive** rung at *every* scale (the anchor holds regardless of encoder width).

### `O(C²)` drift-pair sampling — `sample_drift_pairs`

Pairwise drift is `O(C²)` per round; full enumeration is the default for the paper's central figure, and at
large `C` the documented degrade ([RFC-0005 Alternatives
Considered](rfcs/RFC-0005-evaluation.md#alternatives-considered)) is a **seeded, bounded** sample of
participant pairs. `sample_drift_pairs(ids, max_pairs, seed)` returns at most `max_pairs` distinct
unordered pairs deterministically (same seed → same pairs; a different seed → a different set; capped at
the `C-choose-2` total). The sampled set is **recorded in the `RunManifest`** (RFC-0005 §8: pair sampling
at large `C` is seeded *and* recorded) — it serializes to a manifest-native list of `[a, b]` id lists, so
no tensor crosses the residency redaction guard and the figure stays reproducible.

## Reproducibility

The ladder is driven by config composition ([RFC-0009](rfcs/RFC-0009-configuration-reproducibility.md)):
each rung and each `lambda_anc` value is a config override resolving to a distinct, validated config. The
simulation is residency-safe (only pseudo-gradients cross the transport, `INV-RESIDENCY`) and the
frame-drift diagnostic is a deterministic function of committed weights + the pinned probe (`INV-PROBE-PIN`),
so it is publicly recomputable.
