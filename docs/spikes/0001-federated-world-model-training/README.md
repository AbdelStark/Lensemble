# Spike 0001: Federated training of the world model itself, end to end

Tracking issue: [#335](https://github.com/AbdelStark/Lensemble/issues/335). Status: complete.
Date: 2026-06-16. Scope: CPU experiments through the shipped stack, plus literature and feasibility
analysis. The deliverable is a decision, not a published checkpoint.

## Question

Can the world model itself (encoder plus predictor), not a frozen-checkpoint adapter, be trained
under federation end to end and produce a result that is both gauge-stable (no collapse) and useful
(passes the ground-truth probe gate), ideally with browser or phone participation?

## TL;DR

1. Naive DiLoCo federation of the full model at the schema default outer step does not reach the
   single-silo local-only baseline on the binding ground-truth probe. Fork B at `outer_lr=1.0`
   lands at `state_probe_r2` 0.531 mean vs local-only 0.659, a margin of `-0.128`, and only 1 of 3
   seeds clears the RFC-0017 `+0.05` gate. This reproduces the #259 prediction-parity gap in a
   controlled, multi-seed harness.
2. The single highest-leverage knob found is the outer learning rate. Dropping `outer_lr` from 1.0
   to 0.3 moves federated from margin `-0.128` (1 of 3) to `+0.062` (2 of 3), `state_probe_r2`
   0.531 to 0.720. The schema default (0.7) is too aggressive for a small model. This matches the
   Phase 3 note and the DiLoCo scaling-law literature.
3. Federated averaging compresses latent magnitude to about 30 percent of local-only in every
   configuration and seed (std ratio 0.30 to 0.35), and drops effective rank from 17.5 to 3.8 to
   6.3, even when the probe `state_probe_r2` is recovered. Magnitude shrinks under averaging
   regardless of the probe verdict.
4. `effective_rank` cannot see magnitude collapse: the collapsed control reads `effective_rank`
   127.6 out of 128 with `latent_std_mean` 0.000 and `state_probe_r2` -0.603. A scale-sensitive
   magnitude metric is necessary and is not in the Python eval today.
5. The frame anchor at `lambda_anc` up to 0.5 against a random-init round-0 reference does not
   control the gauge (frame drift saturates at 180 degrees in every federated run), and stronger
   anchoring hurts (`lambda_anc=0.5` gives the worst Fork B result). The anchor must pin to a good
   fixed reference (a warm-start), not a random init.
6. `state_probe_r2` alone is gauge-insensitive at this scale (the best federated run reads 0.720
   with 180 degree drift). The usefulness gate must be a triad: ground-truth probe plus absolute
   magnitude plus frame drift.
7. In-browser full-model training is infeasible by a wide, quantified margin: a full-model delta is
   183.5x over the 384 KB artifact byte budget and 1100.7x over the 16,384 parameter cap, and there
   is no in-browser autograd for the ViT. The shipped adapter is 0.069 percent of the model.

Decision: GO on a tightly scoped GPU follow-up, NO-GO on in-browser full-model training. Details in
[Recommendation](#recommendation-and-go-no-go).

## Method

Real experiments run through the same code the GPU launchers use: `lensemble.federation.Coordinator`
and `Participant` (the DiLoCo outer-optimizer path), `lensemble.model.Objective`, and
`lensemble.eval.state_probe_r2`, on the in-repo `synthetic-dynamic://swipe-dot` env (the epic #273
dynamic env with a resident ground-truth `(x, y)` state). CPU only, deterministic, three seeds per
configuration. Harness: [`run_spike.py`](run_spike.py). Raw results:
[`results/spike_results.json`](results/spike_results.json),
[`results/browser_feasibility.json`](results/browser_feasibility.json).

- Model: `latent_dim=128`, `depth=4`, `predictor_depth=3`, 48px frames, 9 tokens (the same shape
  the CPU gate proves reaches `state_probe_r2 >= 0.5`).
- Non-IID silos: three silos with distinct seeds (different trajectories and state coverage) and
  shared dynamics (`step_scale=0.7`), so a single global predictor can fit all silos and the pooled
  central run is a valid no-aggregation-penalty upper bound.
- Federation: 3 silos, `inner_horizon=100`, 3 rounds, DP off (to measure the gauge contrast, per
  the Phase 3 lesson that DP noise dominates at these dimensions).
- Metrics, all on a held-out env (a fourth seed): `state_probe_r2` (the RFC-0017 binding gate,
  encoder only), `effective_rank`, and two absolute magnitude metrics introduced here,
  `latent_std_mean` (mean per-dim std) and `latent_rms`. Frame drift is read from the coordinator.

## Results

### Controls (metric validation)

| Control | state_probe_r2 | effective_rank | latent_std_mean |
|---|---|---|---|
| random encoder | 0.261 | 17.2 | 0.0520 |
| magnitude-collapsed | -0.603 | 127.6 | 0.0000 |

The collapsed control is the point of the spike in one row: `effective_rank` reads 127.6 out of 128
(essentially full rank) on a representation whose absolute magnitude is zero and whose probe is
worse than the mean. Rank cannot detect magnitude collapse.

### Baselines (no federation, multi-seed mean [min, max])

| Baseline | state_probe_r2 | effective_rank | latent_std_mean |
|---|---|---|---|
| local-only (1 silo) | 0.659 [0.498, 0.754] | 17.5 | 0.0623 |
| central pooled (3 silos, upper bound) | 0.515 [0.409, 0.664] | 29.6 | 0.0906 |

### Federated variants (3 silos, DiLoCo, multi-seed mean)

| Config | state_probe_r2 | margin vs local | seeds passing +0.05 | latent_std ratio to local | effective_rank | frame drift |
|---|---|---|---|---|---|---|
| Fork B, outer_lr 1.0 (claim-grade) | 0.531 | -0.128 | 1/3 | 0.30 | 3.8 | 180 deg |
| Fork B, stop-gradient | 0.506 | -0.153 | 1/3 | 0.31 | 3.8 | 180 deg |
| Fork B, anchor lambda 0.5 | 0.383 | -0.276 | 0/3 | 0.30 | 3.8 | 180 deg |
| Fork B, outer_lr 0.3 | 0.720 | +0.062 | 2/3 | 0.35 | 6.3 | 180 deg |
| Fork A, frozen scratch encoder | 0.305 | -0.354 | 0/3 | 0.70 | 19.5 | 180 deg |

### Browser and on-device feasibility (measured from the real checkpoint)

The `quentinll/lewm-tworooms` model is 18,034,478 parameters (encoder 5.50M, predictor 10.79M),
72.14 MB in fp32.

| Quantity | Value |
|---|---|
| Full-model delta vs 384 KB byte budget | 183.5x over |
| Full-model params vs 16,384 parameter cap | 1100.7x over |
| Shipped adapter share of the model | 0.069 percent (12,512 params) |
| Shipped adapter vs byte budget | 12.7 percent of budget |
| In-browser ViT autograd | none (ONNX is inference-only; the adapter uses hand-coded JS gradients) |

## Findings

- F1. The magnitude-collapse blind spot is real and reproducible. `effective_rank` 127.6 with
  `latent_std_mean` 0.000 on the collapsed control. The Python eval emits only `val_pred`,
  `val_sigreg`, and `effective_rank` (all scale-invariant or collapse-prone), and no absolute
  magnitude metric. This is the exact #259 failure that read as healthy.
- F2. Naive DiLoCo federation does not reach local-only parity at the default outer step. Fork B at
  `outer_lr=1.0` sits at the pooled level (0.531 vs pooled 0.515) but below single-silo local-only
  (0.659), margin -0.128, 1 of 3 seeds passing.
- F3. The outer learning rate is the highest-leverage knob. `outer_lr` 1.0 to 0.3 moves margin
  -0.128 to +0.062 and `state_probe_r2` 0.531 to 0.720. The schema default of 0.7 is too aggressive
  for a small model.
- F4. Averaging compresses magnitude to about 30 percent of local-only and drops effective rank
  (17.5 to 3.8 to 6.3) in every configuration, including the winning `outer_lr=0.3` run. Magnitude
  shrinks under averaging even when the probe is recovered.
- F5. The anchor at `lambda_anc` up to 0.5 against a random-init reference does not control drift
  (180 degrees everywhere) and stronger anchoring hurts the most (0.383). The anchor needs a good
  fixed reference, that is, a warm-start, not a random init.
- F6. `state_probe_r2` is gauge-insensitive here (0.720 at 180 degrees drift). The usefulness gate
  must be a triad: ground-truth probe plus magnitude plus drift.
- F7. Fork A with a frozen scratch encoder cannot be useful (0.305) but preserves magnitude (ratio
  0.70) because a frozen random encoder does not collapse, it simply does not learn. Fork A needs a
  warm-start to provide a useful frozen reference.
- F8. In-browser full-model training is infeasible (183.5x byte budget, 1100.7x parameter cap, no
  ViT autograd). Split federated learning or a deliberately small model on capable devices are the
  only credible on-device routes.

## Recommendation and go-no-go

GO on a tightly scoped follow-up epic for federated full-model training, on GPU at real scale, with
the instrumentation and interventions below. The CPU spike proved the failure modes are real and
reproducible and found the highest-leverage knob, but the toy scale cannot settle the headline
usefulness margin. That needs the GPU launcher with a real warm-start.

NO-GO on in-browser full-model training. The numbers rule it out. If on-device contribution is a
requirement, scope split federated learning or a deliberately small model as a separate spike.

Suggested follow-up work, smallest and highest value first:

1. Phase 0 instrumentation (cheap, do first). Add the absolute magnitude metric
   (`latent_std_mean`, `latent_rms` on held-out) to `lensemble/eval/jepa_metrics.py`, and make the
   RFC-0017 usefulness gate a triad (probe plus magnitude plus drift) rather than `state_probe_r2`
   alone. Commit the missing `dynamic_env_benchmark_report.json` and `dynamic_env_evidence_bundle.json`
   so the existing federated result has a real artifact, not just prose.
2. Phase 1 GPU run. Run `deploy/hfjobs/train_federated_lewm.py` with a real V-JEPA warm-start
   (unblock #96), tune `outer_lr` downward (start at 0.3 and sweep), and measure the triad. Test the
   warm-start fixed anchor (not a random init) and a Fork-A-then-unfreeze staged schedule.
3. Phase 2 directions if Phase 1 still misses parity: a shared or EMA target encoder, function-space
   distillation to a shared teacher (RFC-0002 Layer 4, unimplemented), and low-rank or sparse
   updates so the DP and bandwidth costs scale with effective rank rather than parameter count.

## Limitations: what this spike does not prove

- It does not settle the headline usefulness margin for the real LeWorldModel. The CPU toy model
  reaches `state_probe_r2 >= 0.5` but cannot stand in for the 18M-parameter checkpoint on real data.
- The per-seed `state_probe_r2` spread is wide (for example Fork B claim-grade ranges 0.146 to
  0.813 across seeds), which is why the headline is the multi-seed margin and the pass count, not a
  single draw, and why the parity question escalates to GPU.
- DP is off in these runs by design (to read the gauge contrast). The DP-utility frontier for the
  full model is a separate open question (the sqrt(d) tax).
- Frame drift is measured between participant encoders each round; the 180 degree saturation is a
  strong qualitative signal, not a calibrated angle.

## Reproduce

```bash
uv run python docs/spikes/0001-federated-world-model-training/run_spike.py
```

Deterministic and CPU-only. Writes `results/spike_results.json` (about 13 minutes on a laptop) and
reuses `results/browser_feasibility.json`. The browser numbers are regenerated by the snippet in
[`results/browser_feasibility.json`](results/browser_feasibility.json)'s producer (instantiate
`lensemble.model.lewm_tworooms.LeWMTwoRooms` and count parameters).

## References

Internal: issues #249, #259, #273, #314, #332; RFC-0002, RFC-0005, RFC-0013, RFC-0017; code
`deploy/hfjobs/train_federated_lewm.py`, `lensemble/federation/outer_optimizer.py`,
`lensemble/federation/participant.py`, `lensemble/gauge/anchor.py`, `lensemble/model/objective.py`,
`lensemble/eval/metrics.py`, `lensemble/eval/jepa_metrics.py`; evidence
`docs/evidence/phase3_mvp_model_card.md`, `docs/roadmap/DYNAMIC_ENV.md`. The spike issue #335 carries
the external literature references (DiLoCo and variants, VICReg and SIGReg, non-IID FedSSL, gauge
alignment, world-model evaluation, DP and secure aggregation).
