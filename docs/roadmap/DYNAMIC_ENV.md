# Dynamic Env Evidence Roadmap

Tracker: [#273](https://github.com/AbdelStark/Lensemble/issues/273)

RFC-0017 pivots the usefulness claim away from SO-100 proxy metrics and onto a
small synthetic control env with resident ground truth. The only binding
usefulness gate is held-out `state_probe_r2` on `kinematic://swipe-dot`.

## Current Status (June 2026)

The dynamic-env slice is good enough to continue as an educational end-to-end
demo of the federated JEPA world-model paradigm, but it does **not** clear the
binding benchmark gate.

The current published/control family records:

| control | `state_probe_r2` | role |
|---|---:|---|
| federated scratch | `0.8885337114` | headline demo row |
| local-only | `0.8838405609` | no-aggregation lower bound |
| random encoder | `0.8082002401` | representation sanity baseline |
| naive-FedAvg | `0.5502954721` | unanchored federation control |

The federated row clears the absolute R2 floor and beats random/naive-FedAvg, but
it beats local-only by only `0.0046931505`, below the RFC-0017 required margin
of `0.05`. The correct public framing is therefore: systems concept and demo
surface complete; material federated-over-local usefulness still blocked. The
final benchmark bundle/model card should not be published as a success artifact
until the local-only margin clears.

The CPU/CI de-risking gate is now positive, but narrower than the publication
bar. `tests/ml/test_dynamic_env_cpu_gate.py` trains the real tiny scratch
objective on smooth swipe-dot data, checks both unanchored and pinned-anchor
regimes against held-out `state_probe_r2 >= 0.5`, and closes a two-silo local
federated round whose committed aggregate clears the same ground-truth gate. That
test proves the local recipe can learn a resident state representation and that
the anchored path does not re-collapse it; it is **not** a substitute for the
published DP-on benchmark.

A local DP-on replay of the two-silo CPU-gate updates found the remaining
publication risk. Under the RDP accountant at `epsilon=8`, one full-participation
round needs roughly `noise_multiplier >= 0.7`; budget-valid settings in the
tested neighborhood sometimes cleared the absolute R2 floor (for example,
`noise_multiplier=0.7`, `clip_norm=0.2`, `state_probe_r2 ~= 0.553`) but all
tested budget-valid settings reported `frame_drift_deg=180.0`, and the required
local-only margin was not proven. So the next claim-grade run must tune the
DP/anchor/aggregation recipe, not merely rerun the CPU proof on HF Jobs.

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
| Consortium manifest + dataset registry | `scripts/dynamic_env_silos.py` | Synthetic participant and held-out refs are deterministic, non-IID, disjoint, and published as placeholder/reproducible-from-seed metadata at HF dataset revision `abdelstark/lensemble-dynamic-env-silos@6b61bdc10ee3ce22b3239f7b8c9dbbc5062d7b0d`. |
| Long-run checkpoint/report | `deploy/hfjobs/train_phase3_consortium.py --data-format synthetic-dynamic --encoder scratch` | Published run manifest records `scratch`, not `vjepa2-vit-l`. |
| Dynamic downstream report | `scripts/phase3_inference_demo.py --dynamic-env` | Per-control `state_probe_r2` plus non-binding `success_rate`. |
| Observability/privacy report | `scripts/dynamic_env_observability_report.py` | Per-round DP epsilon, secure-sum status, communication bytes, run-manifest hash binding. |
| Benchmark/card/bundle | `scripts/dynamic_env_benchmark.py` | Requires the full artifact-kind set and rejects a failed R2 gate or model-card drift. |
| Browser north-star | `scripts/dynamic_env_onnx_export.py` + `web/dynamic-env-demo/` | ONNX inference and JS/Canvas env-sim only; browser training is not claimed. |
| Browser federated demo | `uv run lensemble demo federated --port 8765` + `web/federated-demo/` | QR joins, backend lifecycle, browser-surrogate update metadata, aggregation, inference attachment, and evidence export; educational systems demo only. |

## Launch Shape

Representative federated launcher shape:

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
| The HF launcher records a true scratch architecture. | `--encoder scratch`, `tests/ml/test_phase3_consortium_launcher.py` | Implemented locally. |
| The published checkpoint clears the binding gate. | `dynamic_env_benchmark_report.json`: federated `state_probe_r2 >= 0.5` and margin over random / naive-FedAvg / local-only, DP-on | Blocked: published federated R2 is `0.8885337114`, local-only is `0.8838405609`, margin is `0.0046931505 < 0.05`; local budget-valid DP replays still fail the frame-drift/margin bar. |
| The final model card is integrity chained. | `dynamic_env_evidence_bundle.json` with required artifact kinds and byte-identical card embedding | Producer implemented; no success bundle/model card published while the binding margin fails. |
| Browser inference demo is scoped correctly. | `web/dynamic-env-demo/`, `scripts/dynamic_env_onnx_export.py`, `tests/ml/test_dynamic_env_browser_demo.py` | Implemented as ONNX inference + JS/Canvas env-sim; browser training is not claimed. |
| Browser federated demo is scoped correctly. | `web/federated-demo/`, `lensemble/demo/`, `tests/ml/test_federated_demo_app.py`, `docs/roadmap/BROWSER_FEDERATED_DEMO.md` | Implemented as a local educational run-orchestration demo with metadata-only browser-surrogate updates, not production browser training. |

## Non-Claims

- No SO-100 downstream usefulness claim.
- No paper-scale LeWorldModel performance claim.
- No provenance ledger implementation.
- No cryptographic proof of honest participant computation.
- No browser training claim.
- No claim that the current federated dynamic-env checkpoint materially beats
  local-only.
