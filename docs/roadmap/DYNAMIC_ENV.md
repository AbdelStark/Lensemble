# Dynamic Env Evidence Roadmap

Tracker [#273](https://github.com/AbdelStark/Lensemble/issues/273) is closed.
Issue [#335](https://github.com/AbdelStark/Lensemble/issues/335) tracks the
post-correction full-model research and any replacement claim-grade run.

RFC-0017 pivots the usefulness claim away from SO-100 proxy metrics and onto a
small synthetic control env with resident ground truth. The only calibrated
quantitative usefulness gate is held-out `state_probe_r2` on
`kinematic://swipe-dot`; magnitude and frame drift are mandatory diagnostics
for any positive full-model verdict.

## Historical audit baseline

The recorded dynamic-env metrics below were produced **before** correction of
the participant pseudo-gradient and outer-update direction. They are retained
only to audit that historical run. They do not validate the corrected runtime
and must not be presented as current performance.

The pre-correction run recorded:

| control | `state_probe_r2` | role |
|---|---:|---|
| federated scratch | `0.8885337114` | historical federated row |
| local-only | `0.8838405609` | historical no-aggregation control |

Within that historical artifact family, the federated row beat the recorded
random and naive-FedAvg controls but exceeded local-only by only
`0.0046931505`, below RFC-0017's required `0.05` margin. It therefore failed
the binding usefulness gate even before the direction bug was considered.

No post-correction claim-grade dynamic-env result is committed. The checked-in
dynamic-env artifacts also predate per-round
`latent_std_mean`/`latent_rms` instrumentation and do not contain those values.
This roadmap does not infer or retrofit them. A replacement run under
[#335](https://github.com/AbdelStark/Lensemble/issues/335) must use the corrected
runtime, bind the required controls, and report held-out magnitude and
frame-drift diagnostics before any positive claim is reconsidered. RFC-0017
intentionally leaves the calibrated magnitude/drift thresholds open; until
claim-grade runs establish them, they are required review inputs rather than
additional numeric wins.

The CPU/CI de-risking gate is now positive, but narrower than the publication
bar. `tests/ml/test_dynamic_env_cpu_gate.py` trains the real tiny scratch
objective on smooth swipe-dot data, checks both unanchored and pinned-anchor
regimes against held-out `state_probe_r2 >= 0.5`, and closes a two-silo local
federated round whose committed aggregate clears the same ground-truth gate. That
test proves the local recipe can learn a resident state representation and that
the anchored path does not re-collapse it; it is **not** a substitute for the
historical configured-noise benchmark. The deterministic replay path is not
effective DP.

A local configured-noise replay of the two-silo CPU-gate updates found the remaining
publication risk. A fresh one-round RDP calculation at target `epsilon=8` indicates that one
full-participation round needs roughly `noise_multiplier >= 0.7`; settings that satisfy that
one-round calculation in the
tested neighborhood sometimes cleared the absolute R2 floor (for example,
`noise_multiplier=0.7`, `clip_norm=0.2`, `state_probe_r2 ~= 0.553`) but all
tested settings reported `frame_drift_deg=180.0`, and the required local-only margin was not proven.
Because the seed is replayable and accounting is not cumulative or pre-release,
none is a budget-enforced DP result. Any replacement run must both wire
effective privacy controls and tune the noise/anchor/aggregation recipe, not
merely rerun the CPU proof on HF Jobs.

## Environment

- Dataset adapter: `synthetic-dynamic://swipe-dot?...`
- Eval world: `kinematic://swipe-dot`
- State: resident `(x, y)` true position carried inside `Window.state`
- Calibrated binding metric: `state_probe_r2`
- Mandatory full-model diagnostics: held-out `latent_std_mean`, `latent_rms`,
  and public-probe `frame_drift_deg` (exact pass thresholds remain open)
- Supporting-only metrics: closed-loop `success_rate`, `effective_rank`,
  `skill_vs_identity`, latent-MPC goal energy

`success_rate` is reported but non-binding because the planner objective can be
gameable. `effective_rank` is scale-invariant and cannot detect magnitude
collapse. `skill_vs_identity` is gameable. This is a synthetic control env, not
SO-100 and not paper-scale robotics evidence.

## Artifact Producers

| Artifact | Producer | Gate |
|---|---|---|
| Consortium manifest + dataset registry | `scripts/dynamic_env_silos.py` | Synthetic participant and held-out refs are deterministic, non-IID, disjoint, and published as placeholder/reproducible-from-seed metadata at HF dataset revision `abdelstark/lensemble-dynamic-env-silos@6b61bdc10ee3ce22b3239f7b8c9dbbc5062d7b0d`. |
| Long-run checkpoint/report | `deploy/hfjobs/train_phase3_consortium.py --data-format synthetic-dynamic --encoder scratch` | A replacement run manifest must record `scratch`, not `vjepa2-vit-l`; per-round reports include held-out `latent_std_mean` and `latent_rms` alongside rank and drift. |
| Dynamic downstream report | `scripts/phase3_inference_demo.py --dynamic-env` | Per-control `state_probe_r2` plus non-binding `success_rate`. |
| Observability/privacy report | `scripts/dynamic_env_observability_report.py` | Per-round one-round epsilon snapshot, secure-sum consumption/equivalence/fallback status, effective-DP status, communication bytes, and run-manifest hash binding; snapshots must not be presented as cumulative DP. |
| Benchmark/card/bundle | `scripts/dynamic_env_benchmark.py` | Requires the full artifact-kind set and rejects a failed R2 gate or model-card drift. |

## Replacement-run template

Representative launcher shape for a post-correction run:

```bash
hf jobs uv run --flavor a10g-large --timeout 2h --secrets HF_TOKEN \
  deploy/hfjobs/train_phase3_consortium.py \
  --data-format synthetic-dynamic \
  --data-source 'synthetic-dynamic://swipe-dot?seed=10&n_episodes=16&steps=64&image_size=48' \
  --data-source 'synthetic-dynamic://swipe-dot?seed=20&n_episodes=16&steps=64&image_size=48' \
  --data-source 'synthetic-dynamic://swipe-dot?seed=30&n_episodes=16&steps=64&image_size=48' \
  --data-source 'synthetic-dynamic://swipe-dot?seed=40&n_episodes=16&steps=64&image_size=48' \
  --heldout-source 'synthetic-dynamic://swipe-dot?seed=99&n_episodes=16&steps=64&image_size=48' \
  --encoder scratch \
  --latent-dim 128 \
  --depth 4 \
  --predictor-depth 4 \
  --num-heads 8 \
  --image-size 48 \
  --patch-size 16 \
  --num-frames 1 \
  --tubelet 1 \
  --num-rounds 12 \
  --inner-horizon 2 \
  --window-steps 1 \
  --lambda-anc 1.0 \
  --secure-agg-threshold 3 \
  --min-trainers 3 \
  --metric-windows 64 \
  --push \
  --out-repo abdelstark/lensemble-dynamic-env-swipe-dot
```

## Acceptance Matrix

| Claim | Required evidence | Status |
|---|---|---|
| Dynamic env data is resident and deterministic. | `tests/ml/test_synthetic_dynamic_backend.py`, `tests/ml/test_dynamic_env_silos.py`, `docs/evidence/dynamic_env_silo_plan.json` | Implemented locally; placeholder/reproducible-from-seed registry metadata is published at HF dataset revision `abdelstark/lensemble-dynamic-env-silos@6b61bdc10ee3ce22b3239f7b8c9dbbc5062d7b0d`. |
| The eval report exposes binding `state_probe_r2`. | `lensemble.eval.report.EvalReport.state_probe_r2`, `tests/ml/test_harness.py` | Implemented locally. |
| The CPU gate distinguishes binding R2 from scale-invariant collapse. | `tests/ml/test_dynamic_env_cpu_gate.py` | Implemented locally: the real tiny objective passes held-out `state_probe_r2 >= 0.5` in unanchored and pinned-anchor modes, and a two-silo local aggregate clears the same ground-truth gate while collapsed/random controls fail. |
| Full-model reports expose scale and drift diagnostics. | `lensemble/eval/jepa_metrics.py`, `lensemble/federation/phase3_orchestration.py`, `tests/unit/test_jepa_metrics.py`, `tests/integration/test_phase3_consortium_run.py` | Implemented for new runs: `latent_std_mean` and `latent_rms` make uniform scale collapse visible even when `effective_rank` remains healthy; `frame_drift_deg` remains paired with them. Historical dynamic-env evidence has no retrofitted values, and calibrated pass thresholds remain open. |
| The HF launcher records a true scratch architecture. | `--encoder scratch`, `tests/ml/test_phase3_consortium_launcher.py` | Implemented locally. |
| The historical checkpoint cleared the binding gate. | Pre-correction control report retained for audit. | Failed and superseded: historical federated `state_probe_r2=0.8885337114`, local-only `0.8838405609`, and margin `0.0046931505 < 0.05`; these values do not validate the corrected runtime. |
| The corrected runtime clears the binding gate. | New `dynamic_env_benchmark_report.json` from the corrected runtime: federated `state_probe_r2 >= 0.5`, margin at least `0.05` over random / naive-FedAvg / local-only, held-out magnitude and drift diagnostics, and an honestly labeled configured-noise control. | Blocked pending the claim-grade work in [#335](https://github.com/AbdelStark/Lensemble/issues/335). No post-correction result is committed. |
| A replacement model card is integrity chained. | `dynamic_env_evidence_bundle.json` with required artifact kinds and byte-identical card embedding. | Producer implemented; publication remains blocked until a post-correction run clears the binding gate. |

## Non-Claims

- No SO-100 downstream usefulness claim.
- No paper-scale LeWorldModel performance claim.
- No provenance ledger implementation.
- No cryptographic proof of honest participant computation.
- No claim that the pre-correction dynamic-env numbers validate the corrected
  runtime.
- No claim that a federated dynamic-env checkpoint materially beats local-only.
