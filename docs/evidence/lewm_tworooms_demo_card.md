# Tapestry-like LeWM TwoRooms Demo Card

This card is the public evidence surface for the **Tapestry-like browser-federated LeWorldModel
showcase** ([#314](https://github.com/AbdelStark/Lensemble/issues/314); contract in
[TAPESTRY_LEWM.md](../roadmap/TAPESTRY_LEWM.md)). It states exactly what was real, what the
evidence binds, and what is **not** claimed. It exists so the real-checkpoint path can never be
confused with the older `surrogate-swipe-dot` educational path
([BROWSER_FEDERATED_DEMO.md](../roadmap/BROWSER_FEDERATED_DEMO.md)).

## The claim

> Lensemble demonstrates a **Tapestry-like browser-local federated adaptation run** for a real
> LeWorldModel TwoRooms checkpoint. Browser participants keep local rollouts resident, train a
> bounded checkpoint-adaptation subset, submit bounded deltas, aggregate into global revisions,
> and inspect real inference, loss, anti-collapse, probe, planning, and evidence outputs.

"Tapestry-like" means the concepts — sovereign participants, a shared protocol, bounded update
artifacts, coordinator aggregation, observability, claim-bounded evidence — borrowed from Project
Tapestry (AI Alliance), not its primitives and not its scale.

## What was real

| Component | What actually happens | Binding evidence |
|---|---|---|
| Base model | The released `quentinll/lewm-tworooms` LeWorldModel checkpoint (MIT), pinned at revision `77adaae0bc31…`, reconstructed in-tree and strict-loaded (303/303 tensors, 18,034,478 params); parity vs the actual upstream implementation ≤ 2e-5 max abs diff | [checkpoint manifest](lewm_tworooms_checkpoint_manifest.json), [reference report](lewm_tworooms_reference_report.json) |
| Browser inference | Single-file ONNX graphs (encoder+projector, action encoder, predictor+pred_proj) exported deterministically from the pinned checkpoint, hash-bound, with ImageNet pixel normalization and the expert-dataset action z-score baked in; PyTorch-vs-onnxruntime parity ≤ 1e-4 | [export manifest](lewm_tworooms_browser_export_manifest.json), [action stats](lewm_tworooms_action_stats.json) |
| Pipeline correctness | On held-out official expert episodes, the exported pipeline's teacher-forced next-latent MSE is 0.070 vs 1.306 for the copy-last baseline (ratio 0.054) | [real-data check](lewm_tworooms_realdata_check.json) |
| TwoRooms environment | A deterministic JS port of the upstream env at the released default variations, validated pixel-level against the dataset frames (max abs diff 1/255); labeled a TwoRooms-compatible probe with its deviations documented | `web/federated-demo/tworooms_env.mjs` |
| Browser-local training | A bounded zero-init residual adapter (12,512 params) on the frozen predictor output, trained with manual-gradient Adam in the browser; SIGReg Epps–Pulley diagnostics ported exactly from `lensemble.model.sigreg` (torch parity ≤ 2e-7 rel); one-browser overfit on real latents: pred loss 0.0559 → 0.00061 | [adapter overfit](lewm_tworooms_adapter_overfit.json) |
| Federation | `lewm-adapter-delta/1` bounded clipped deltas, validated server-side (schema, byte budget, shapes, norms, checkpoint/export-hash freshness, replay, fabricated-metric rejection) and aggregated as a deterministic per-round mean over a shared deterministic adapter init | `tests/ml/test_lewm_federation.py` |
| Before/after probe (system-composed) | The headline number is produced by the **system the demo ships**: real node-trained adapter deltas flow through `FederatedDemoService.submit_update` (real validation) and `_close_round_lewm` (real deterministic-mean aggregation + hash-chained revisions), and the probe scores the **server-produced** final `modelRevisionId` — not an offline-recomputed mean. Held-out MSE improved 0.0604 → 0.0530 (**+12.3%**, verdict `improved`, 0 claim-audit violations) on the seed-`20260612` draw | [system-composed probe](lewm_tworooms_system_probe.json); offline math cross-check [federated probe](lewm_tworooms_probe_check.json) |
| Seed robustness | The headline is not a single favorable draw: across 5 independent seeds / episode splits the system-composed probe improves on **every** seed with **no collapse** — mean +16.8%, worst case **+5.4%** (seed 2), best +32.6% | [seed sweep](lewm_tworooms_probe_seedsweep.json) |
| Held-out collapse check | The gain is bias-correction, not the #259 held-out magnitude collapse: on the validation pairs the adapted latent std holds vs the frozen baseline (0.90 → 0.91) and effective rank is preserved (9.86 → 9.80), so an "improved" MSE cannot hide a collapse — a materially lower std/rank overrides the verdict to `collapse-risk` | held-out diagnostics in [system-composed probe](lewm_tworooms_system_probe.json) |
| Honest failure states | Round metrics carry server-side health flags (effective-rank collapse, latent-std magnitude collapse, flat/worsened loss, SIGReg outliers); the dashboard probe reports `flat`/`worse` verdicts instead of hiding them — the first probe configuration **was** flat and is recorded as such in the development history | round metrics in every evidence export |

## Privacy and residency

- Raw frames, actions, labels, latents, tensors, participant tokens, and base checkpoint weights
  never leave participant browsers; the coordinator rejects raw-data-like keys fail-closed.
- Secure aggregation (RFC-0011) and differential privacy (RFC-0012) exist in the main lensemble
  stack but are **absent in this demo path**; every evidence export states this explicitly
  instead of implying protection.

## Non-claims

This demo is **not**:

- full from-scratch LeWorldModel base training inside every browser — it is bounded
  **checkpoint adaptation** around a frozen exported base;
- production browser training;
- paper-scale TwoRooms or PushT benchmark parity (the +12.3% probe result is a fixed held-out
  validation probe for this demo, single local coordinator with mean-of-clipped-deltas and no
  robust aggregation / DP in this path, not a benchmark win);
- a cryptographic proof of honest computation;
- closed-loop physical robotics success;
- evidence that raw participant data ever leaves the browser (it does not, and the audit gate
  rejects evidence that contains any).

## Reproduce

```bash
uv run python scripts/lewm_tworooms_ingest.py
uv run --with onnx --with onnxscript --with onnxruntime python scripts/lewm_tworooms_export.py
uv run --with onnxruntime --with hdf5plugin python scripts/lewm_tworooms_realdata_check.py --h5 <tworoom.h5>
uv run --with onnxruntime --with hdf5plugin python scripts/lewm_adapter_overfit_check.py --h5 <tworoom.h5>
uv run --with onnxruntime --with hdf5plugin python scripts/lewm_probe_check.py --h5 <tworoom.h5>     # offline math cross-check
uv run --with onnxruntime --with hdf5plugin python scripts/lewm_system_probe.py --h5 <tworoom.h5>    # system-composed headline (#327)
uv run --with onnxruntime --with hdf5plugin python scripts/lewm_probe_seedsweep.py --h5 <tworoom.h5> # seed robustness (#330)
uv run lensemble demo federated --port 8765   # then create a real-lewm-tworooms run
```

Run-level evidence (`demo-evidence/1` with `runMode: real-lewm-tworooms`) is exported from the
host dashboard and validated by `lensemble.demo.evidence_audit` — the claim audit fails closed on
missing claim-boundary phrases, unbound revisions, missing hashes, raw participant data, or
unnegated overclaims.
