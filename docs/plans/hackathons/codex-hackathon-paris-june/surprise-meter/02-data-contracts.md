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
  "warmupSteps": 2,                       // ring-buffer HOLD: no surprise emitted for the first 2 model steps
  "result": {
    "meanSurprisePre": 0.0604,            // MEASURED this run; == baseline pair-MSE scale (mean over 192 dims)
    "meanSurprisePost": 0.0530,           // MEASURED this run
    "surpriseDropRatioLive": 0.1180,      // MEASURED (pre-post)/pre on THIS run's pairs — label "this run", not the headline
    "perturbationSpikeRatio": 4.8,        // PLACEHOLDER — emit MEASURED (mean surprise perturbed / normal)
    "oodActionSpikeRatio": 3.1,           // PLACEHOLDER — emit MEASURED
    "frameDiffCorrelation": 0.12,         // PLACEHOLDER — emit MEASURED corr(surprise, framediff); LOW => surprise != motion
    "stepLatencyMsCpu": 6.2,              // PLACEHOLDER — emit MEASURED, or omit if untimed (not passes-gated)
    // ── sourced from existing evidence at FULL precision (never recompute; never truncate) ──
    "federatedRelativeImprovement": 0.12275377883038366,
    "federatedRelativeImprovementSource": "docs/evidence/lewm_tworooms_system_probe.json#result.relativeImprovement",
    "federatedSeedMean": 0.16787180214169914,
    "federatedSeedMeanSource": "docs/evidence/lewm_tworooms_probe_seedsweep.json#distribution.relativeImprovementMean",
    "federatedSeedWorst": 0.054144202317108696,
    "federatedSeedWorstSource": "docs/evidence/lewm_tworooms_probe_seedsweep.json#distribution.relativeImprovementMin",
    "federatedSeedStdev": 0.10969915982049769,
    "latentDim": 192
  },
  "passes": true,
  "nonClaims": [
    "Federated result is adapter continuation on a frozen checkpoint, not federated world-model training (spike #335; lewm_tworooms_system_probe.json).",
    "Surprise is a scalar per-frame next-latent prediction error (CLS-latent model); it is not a per-patch spatial map.",
    "Perturbation responses are illustrative on the TwoRooms backbone, not a calibrated anomaly detector.",
    "Single local coordinator, mean of clipped deltas; no secure aggregation or differential privacy in this path.",
    "This is a TwoRooms educational demo, not paper-scale LeWorldModel performance.",
    "No closed-loop robot control is claimed; the agent follows a scripted/expert policy in a 2D sim."
  ],
  // The first four are MANDATORY (asserted by the test). The viewer renders the perturbation-screen
  // nonClaim ("illustrative … not a calibrated anomaly detector") VISIBLY on the perturbation footer,
  // not just inside the JSON.
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
- `surpriseDropRatioLive > 0.02` (matches the system-probe verdict threshold).
- `perturbationSpikeRatio > 1.5` **or** `oodActionSpikeRatio > 1.5` (at least one perturbation channel spikes — guards R1; if neither, `passes=false` and the demo must lean on pre/post only).
- `abs(frameDiffCorrelation) < 0.6` (surprise is not merely tracking motion).
- `nonClaims` contains the four mandatory negations.

### Sourcing rule (claim discipline — verified values)
- `federatedRelativeImprovement`, `federatedSeedMean`, `federatedSeedWorst`, `federatedSeedStdev` are **read from** the cited evidence files at **full float precision** (not recomputed, not truncated). The producer reads them at runtime and the test asserts equality to the source value within **1e-6**. Storing the display rounding (`0.1227`, `0.168`) fails this check (`|0.1227 - 0.1227537788...| ~= 5.4e-5`). Verified source values: `result.relativeImprovement = 0.12275377883038366`; `distribution.relativeImprovementMean = 0.16787180214169914`; `distribution.relativeImprovementMin = 0.054144202317108696`; `distribution.relativeImprovementStdev = 0.10969915982049769`.
- The `// PLACEHOLDER` numbers (`perturbationSpikeRatio`, `oodActionSpikeRatio`, `frameDiffCorrelation`, `stepLatencyMsCpu`) are **measured by this run**, not hardcoded — the literals above are illustrative scales only. `perturbation*`/`frameDiff*` come from the perturbed-replay channels (R1 is empirical and **only checkable with a runtime present** — onnxruntime offline, or in-browser).
- All other `result` numbers (`meanSurprisePre/Post`, `surpriseDropRatioLive`) are produced by this run, deterministic given `seed`. `surpriseDropRatioLive` is the **live** drop and is *not* the certified headline — the HUD labels it "this run".

---

## 2. `surprise_trajectory.json` — recorded-trajectory fallback (`lewm-surprise-traj/1`)

Path: `web/surprise-meter/data/surprise_trajectory.json` (committed fallback so the meter runs without live env/ONNX quirks on stage). Schema **`lewm-surprise-traj/1`** (renamed from the draft's stray `cartographer-surprise-traj/1`); the viewer validates this string before replaying. The current stage fallback is generated by `scripts/lewm_surprise_check.py` alongside `docs/evidence/lewm_tworooms_surprise.json` and `web/surprise-meter/data/result_card.json`.

```jsonc
{
  "schema": "lewm-surprise-traj/1",
  "steps": [
    { "t": 0, "agentPos": [x,y], "action": [dx,dy], "surprisePre": 0.061, "surprisePost": 0.053,
      "frameDiff": 0.004, "perturbed": false }
    // …  cap: <= 600 steps
  ],
  "perturbations": [ { "t": 120, "kind": "teleport" }, { "t": 160, "kind": "ood-action" } ],
  "provenance": { "checkpointRevision": "77adaae…", "adapterOffsetFile": "web/surprise-meter/fixtures/adapter_offset.json", "bakeSeed": 20260618 }
}
```
Used only as the fallback rung; the default path computes surprise **live** from the ONNX graphs.

---

## 3. `adapter_offset.json` — pre/post toggle input
A JSON `list[float]` of length **12512** (the post-federation `adapterState` offset from SM-1, captured via `service.model_revision(run_id, final_id)["adapterState"]`). Consumed by `lewm_adapter.mjs::adapterFromInitAndOffset({inputDim:192, hiddenDim:32, initSeed:42, offset})` (object arg — **not** a bare positional offset).
- **Working output (ephemeral, gitignored):** `runs/surprise/adapter_offset.json` (`runs/` is in `.gitignore`).
- **Committed fallback (tracked + served under `WEB_ROOT`):** `web/surprise-meter/fixtures/adapter_offset.json` — copy the run output here and `git add` it. The pre/post toggle and rehearsal load this path.
- The offset is **never** written into any `demo-evidence/1` bundle (the audit forbids the `"adapterState"` substring). It is a standalone sidecar file.
- **Loader guard:** assert `length === 12512` **and** `some(v => v !== 0)` (an all-zero offset = identity adapter = zero drop = dead demo).

---

## 4. Versioning & validation
- Schemas are `name/INT`; breaking change bumps the int + updates the consumer in the same PR.
- The viewer validates the trajectory schema string before rendering the fallback.
- The evidence test pins `schema == "lewm-surprise/1"` and the `passes` predicate.
