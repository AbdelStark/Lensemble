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

## Reproducibility

The ladder is driven by config composition ([RFC-0009](rfcs/RFC-0009-configuration-reproducibility.md)):
each rung and each `lambda_anc` value is a config override resolving to a distinct, validated config. The
simulation is residency-safe (only pseudo-gradients cross the transport, `INV-RESIDENCY`) and the
frame-drift diagnostic is a deterministic function of committed weights + the pinned probe (`INV-PROBE-PIN`),
so it is publicly recomputable.
