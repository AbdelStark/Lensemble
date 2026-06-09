# Dynamic Env Evidence Roadmap

Tracker: [#273](https://github.com/AbdelStark/Lensemble/issues/273)

RFC-0017 pivots the usefulness claim away from SO-100 proxy metrics and onto a
small synthetic control env with resident ground truth. The only binding
usefulness gate is held-out `state_probe_r2` on `kinematic://swipe-dot`.

## Environment

- Dataset adapter: `synthetic-dynamic://swipe-dot?...`
- Eval world: `kinematic://swipe-dot`
- State: resident `(x, y)` true position carried inside `Window.state`
- Binding metric: `state_probe_r2`
- Supporting-only metrics: closed-loop `success_rate`, `effective_rank`,
  `skill_vs_identity`, latent-MPC goal energy

`success_rate` is reported but non-binding because the planner objective can be
gameable. `effective_rank` is scale-invariant and cannot detect magnitude
collapse. `skill_vs_identity` is gameable. This is a synthetic control env, not
SO-100 and not paper-scale robotics evidence.

## Artifact Producers

| Artifact | Producer | Gate |
|---|---|---|
| Consortium manifest + dataset registry | `scripts/dynamic_env_silos.py` | Synthetic participant and held-out refs are deterministic, non-IID, and disjoint. |
| Long-run checkpoint/report | `deploy/hfjobs/train_phase3_consortium.py --data-format synthetic-dynamic --encoder scratch` | Published run manifest records `scratch`, not `vjepa2-vit-l`. |
| Dynamic downstream report | `scripts/phase3_inference_demo.py --dynamic-env` | Per-control `state_probe_r2` plus non-binding `success_rate`. |
| Observability/privacy report | `scripts/dynamic_env_observability_report.py` | Per-round DP epsilon, secure-sum status, communication bytes, run-manifest hash binding. |
| Benchmark/card/bundle | `scripts/dynamic_env_benchmark.py` | Requires the full artifact-kind set and rejects a failed R2 gate or model-card drift. |
| Browser north-star | `scripts/dynamic_env_onnx_export.py` + `web/dynamic-env-demo/` | ONNX inference and JS/Canvas env-sim only; browser training is not claimed. |

## Launch Shape

The intended C11 job shape is:

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
| Dynamic env data is resident and deterministic. | `tests/ml/test_synthetic_dynamic_backend.py`, `tests/ml/test_dynamic_env_silos.py` | Implemented locally. |
| The eval report exposes binding `state_probe_r2`. | `lensemble.eval.report.EvalReport.state_probe_r2`, `tests/ml/test_harness.py` | Implemented locally. |
| The CPU gate distinguishes binding R2 from scale-invariant collapse. | `tests/ml/test_dynamic_env_cpu_gate.py` | Implemented locally. |
| The HF launcher records a true scratch architecture. | `--encoder scratch`, `tests/ml/test_phase3_consortium_launcher.py` | Implemented locally. |
| The published checkpoint clears the binding gate. | `dynamic_env_benchmark_report.json`: federated `state_probe_r2 >= 0.5` and margin over random / naive-FedAvg / local-only, DP-on | Pending the C11 GPU run. |
| The final model card is integrity chained. | `dynamic_env_evidence_bundle.json` with required artifact kinds and byte-identical card embedding | Producer implemented; final artifact pending C11. |
| Browser demo is scoped correctly. | `web/dynamic-env-demo/`, `scripts/dynamic_env_onnx_export.py`, `tests/ml/test_dynamic_env_browser_demo.py` | Implemented locally; exported ONNX artifact pending C11 checkpoint. |

## Non-Claims

- No SO-100 downstream usefulness claim.
- No paper-scale LeWorldModel performance claim.
- No provenance ledger implementation.
- No cryptographic proof of honest participant computation.
- No browser training claim.
