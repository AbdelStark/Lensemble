# Surprise-meter — Data Contracts

Surprise-meter is mostly **live** (computed per-step in the browser), so its contracts are lighter than Cartographer's. Two artifacts: an optional **recorded-trajectory fallback** and the **evidence JSON**.

---

## 1. `lewm_tworooms_surprise.json` — evidence (`lewm-surprise/1`)

Path: `docs/evidence/lewm_tworooms_surprise.json`. Producer: `scripts/lewm_surprise_check.py`. Validated by `tests/ml/test_lewm_surprise.py`.

```jsonc
{
  "schema": "lewm-surprise/1",
  "role": "surprise-meter-evidence",
  "seed": 20260618,
  "checkpoint": {
    "repoId": "quentinll/lewm-tworooms",
    "revision": "77adaae0bc31deab21c93740d1f8bb947cd0bdec",
    "weightsSha256": "566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd"
  },
  "trajectory": { "numSteps": 200, "source": "tworoom.h5 episode replay", "predictorWindow": 3 },
  "result": {
    "meanSurprisePre": 0.0604,            // == baseline pair MSE scale from system probe
    "meanSurprisePost": 0.0530,
    "surpriseDropRatio": 0.1227,          // (pre-post)/pre on in-distribution steps
    "perturbationSpikeRatio": 4.8,        // mean surprise on perturbed steps / on normal steps
    "frameDiffCorrelation": 0.12,         // corr(surprise, framediff); LOW => surprise != motion
    "oodActionSpikeRatio": 3.1,           // off-distribution action surprise / in-distribution
    "federatedRelativeImprovement": 0.1227,
    "federatedRelativeImprovementSource": "docs/evidence/lewm_tworooms_system_probe.json",
    "federatedSeedMean": 0.168,
    "federatedSeedMeanSource": "docs/evidence/lewm_tworooms_probe_seedsweep.json",
    "latentDim": 192,
    "stepLatencyMsCpu": 6.2
  },
  "passes": true,
  "nonClaims": [
    "Federated result is adapter continuation on a frozen checkpoint, not federated world-model training (spike #335; lewm_tworooms_system_probe.json).",
    "Surprise is a scalar per-frame next-latent prediction error (CLS-latent model); it is not a per-patch spatial map.",
    "Perturbation responses are illustrative on the TwoRooms backbone, not a calibrated anomaly detector.",
    "Single local coordinator, mean of clipped deltas; no secure aggregation or differential privacy in this path."
  ],
  "crossCheck": {
    "systemProbe": "docs/evidence/lewm_tworooms_system_probe.json",
    "seedSweep": "docs/evidence/lewm_tworooms_probe_seedsweep.json",
    "browserExport": "docs/evidence/lewm_tworooms_browser_export_manifest.json"
  }
}
```

### `passes` predicate (asserted by the test + producer)
`passes == true` iff **all**:
- `meanSurprisePost < meanSurprisePre` (federation reduces in-distribution surprise).
- `surpriseDropRatio > 0.02` (matches the system-probe verdict threshold).
- `perturbationSpikeRatio > 1.5` **or** `oodActionSpikeRatio > 1.5` (at least one perturbation channel spikes — guards R1; if neither, `passes=false` and the demo must lean on pre/post only).
- `abs(frameDiffCorrelation) < 0.6` (surprise is not merely tracking motion).
- `nonClaims` contains the four mandatory negations.

### Sourcing rule
- `federatedRelativeImprovement` / `federatedSeedMean` are **read from** the cited evidence files (not recomputed). Producer asserts they exist and match within 1e-6.
- All other `result` numbers are produced by this run, deterministic given `seed`.

---

## 2. `surprise_trajectory.json` — recorded-trajectory fallback (`cartographer-surprise-traj/1`)

Path: `web/surprise-meter/data/surprise_trajectory.json` (committed fallback so the meter runs without live env/ONNX quirks on stage).

```jsonc
{
  "schema": "cartographer-surprise-traj/1",
  "steps": [
    { "t": 0, "agentPos": [x,y], "action": [dx,dy], "surprisePre": 0.061, "surprisePost": 0.053,
      "frameDiff": 0.004, "perturbed": false }
    // …  cap: <= 600 steps
  ],
  "perturbations": [ { "t": 120, "kind": "teleport" }, { "t": 160, "kind": "ood-action" } ],
  "provenance": { "checkpointRevision": "77adaae…", "adapterOffsetFile": "runs/surprise/adapter_offset.json", "bakeSeed": 20260618 }
}
```
Used only as the fallback rung; the default path computes surprise **live** from the ONNX graphs.

---

## 3. `adapter_offset.json` — pre/post toggle input
Path: `runs/surprise/adapter_offset.json` — a JSON `list[float]` of length **12512** (the post-federation `adapterState` offset from SM-1). Consumed by `lewm_adapter.mjs::adapterFromInitAndOffset`. A pre-event committed copy is the fallback for the pre/post toggle.

---

## 4. Versioning & validation
- Schemas are `name/INT`; breaking change bumps the int + updates the consumer in the same PR.
- The viewer validates the trajectory schema string before rendering the fallback.
- The evidence test pins `schema == "lewm-surprise/1"` and the `passes` predicate.
