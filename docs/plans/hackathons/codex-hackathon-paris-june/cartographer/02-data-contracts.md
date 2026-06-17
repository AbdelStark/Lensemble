# Cartographer — Data Contracts

This is the **single source of truth** for the two JSON artifacts. Freezing these lets the viewer (CART-8) and the bake (CART-7) be built in parallel. Both are versioned; bump the version on any breaking field change and update the consumer.

All numbers are JSON numbers (float/int). All hashes are lowercase hex. Arrays of vectors are row-major. Coordinates are pre-projected to 3D by the bake — **the viewer does no dimensionality reduction**.

---

## 1. `manifold.json` — viewer payload (`cartographer-manifold/1`)

Path: `web/latent-manifold-viewer/data/manifold.json`. Target size ≤ ~5 MB (enforce via the caps below; the bake LOGs any downsampling — no silent truncation).

```jsonc
{
  "schema": "cartographer-manifold/1",
  "provenance": {
    "checkpointRepoId": "quentinll/lewm-tworooms",
    "checkpointRevision": "77adaae0bc31deab21c93740d1f8bb947cd0bdec",
    "checkpointWeightsSha256": "566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd",
    "exportGraphHashes": { "encoder": "…64hex", "action": "…", "predictor": "…" },
    "bakeSeed": 20260618,
    "gitSha": "…",
    "evidenceFile": "docs/evidence/lewm_tworooms_manifold.json"
  },

  "projection": {
    "method": "pca-3d",
    "fittedOn": "post",                  // basis fit on the post-federation cloud
    "varianceExplained": [0.41, 0.19, 0.11],   // per-axis ratio, len 3
    "center": [/* 192 */],               // mean subtracted before projection (for reproducibility)
    "procrustesResidualPrePost": 0.0123, // gauge.procrustes residual aligning pre→post
    "latentDim": 192
  },

  // Each cloud is a set of 3D points sharing the SAME basis+center (so toggles are comparable).
  "clouds": {
    "post":     { "label": "post-federation (healthy)", "synthetic": false,
                  "coords3d": [[x,y,z], …],   // cap: <= 4000 points
                  "pointMeta": { "episodeId": [int,…], "stepIdx": [int,…] } },
    "pre":      { "label": "pre-federation",            "synthetic": false,
                  "coords3d": [[x,y,z], …] }, // SAME point ordering/count as post (paired)
    "collapsed":{ "label": "collapsed (synthetic illustration)", "synthetic": true,
                  "mode": "rank1",
                  "coords3d": [[x,y,z], …] }
  },

  // 1–3 planning episodes. Latent trajectories are already projected to 3D.
  "plans": [
    {
      "goalCoord3d": [x,y,z],
      "startCoord3d": [x,y,z],
      "chosen": { "coords3d": [[x,y,z], …],   // winning rollout, len = horizon(+1)
                  "cost": 12.34, "wallTimeS": 0.08 },
      "iterations": [                          // cap: <= 4 iters
        { "iter": 0,
          "candidates": [                      // cap: <= 24 candidates/iter (LOGGED if more existed)
            { "coords3d": [[x,y,z], …], "cost": 18.7, "elite": false }, …
          ] }
      ],
      "plannerFamily": "icem", "horizon": 10, "numSamples": 128, "numIters": 4,
      "renderedCandidateCap": 24, "totalCandidatesPerIter": 128   // honesty: cap vs truth
    }
  ],

  "metrics": {
    "effectiveRankHealthy": 9.86,
    "effectiveRankCollapsed": 1.02,
    "effectiveDimHealthy": 7.4,
    "effectiveDimCollapsed": 1.0,
    "federatedRelativeImprovement": 0.1227,   // from system-composed probe (committed seed)
    "federatedRelativeImprovementSeedMean": 0.168,
    "planningLatencyMsPerStep": 6.2,          // from the feasibility spike (CPU)
    "plannerWallTimeS": 0.08,
    "goalReachSuccessRate": 0.8               // measured on harvested start/goal pairs
  },

  // Rendered verbatim in the viewer footer (claim discipline on the artifact itself).
  "nonClaims": [
    "Federated result is adapter continuation on a frozen checkpoint, not federated world-model training.",
    "The collapsed cloud is a synthetic illustration, not a trained model.",
    "The 3D layout is a PCA projection of a ~192-dimensional latent space; distances are approximate.",
    "No latent-vs-pixel compute comparison is claimed.",
    "Single local coordinator, mean of clipped deltas; no secure aggregation or differential privacy in this path."
  ]
}
```

### Field rules
- `clouds.pre.coords3d` and `clouds.post.coords3d` are **paired** (index i = same source latent) so a pre↔post tween is a per-point morph.
- `clouds.collapsed` need not be paired; it's an illustration.
- All `coords3d` are already centered/scaled by the bake to fit roughly in a unit cube (record any global scale in `projection` if applied).
- Every value in `metrics` MUST also appear in the evidence JSON (§2). The viewer cross-checks on load (warns in console if mismatch).
- Caps (`<= 4000` points, `<= 24` candidates/iter, `<= 4` iters, `<= 3` plans) keep the file small and the scene readable. The bake records actual-vs-cap in `plans[].totalCandidatesPerIter` / `renderedCandidateCap` and LOGs to stdout.

---

## 2. `lewm_tworooms_manifold.json` — evidence (`lewm-manifold/1`)

Path: `docs/evidence/lewm_tworooms_manifold.json`. Producer: `scripts/lewm_manifold_check.py`. Validated by `tests/ml/test_lewm_manifold.py`.

```jsonc
{
  "schema": "lewm-manifold/1",
  "role": "manifold-and-planning-evidence",
  "seed": 20260618,
  "checkpoint": {
    "repoId": "quentinll/lewm-tworooms",
    "revision": "77adaae0bc31deab21c93740d1f8bb947cd0bdec",
    "weightsSha256": "566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd"
  },
  "harvest": { "numEpisodes": 40, "numPoints": 2000, "h5": "tworoom.h5" },
  "result": {
    "effectiveRankHealthy": 9.86,
    "effectiveRankCollapsedRank1": 1.02,
    "effectiveRankCollapsedMagnitude": 9.7,
    "effectiveDimHealthy": 7.4,
    "effectiveDimCollapsedRank1": 1.0,
    "latentStdHealthy": 0.904,
    "latentStdCollapsedMagnitude": 1e-6,
    "pcaVarianceExplained3d": [0.41, 0.19, 0.11],
    "procrustesResidualPrePost": 0.0123,
    "federatedRelativeImprovement": 0.1227,
    "federatedRelativeImprovementSource": "docs/evidence/lewm_tworooms_system_probe.json",
    "federatedSeedMeanImprovement": 0.168,
    "federatedSeedMeanSource": "docs/evidence/lewm_tworooms_probe_seedsweep.json",
    "planningLatencyMsPerStep": 6.2,
    "plannerWallTimeS": 0.08,
    "plannerFamily": "icem",
    "goalReachSuccessRate": 0.8,
    "goalReachTrials": 20
  },
  "passes": true,
  "provenance": {
    "exportGraphHashes": { "encoder": "…", "action": "…", "predictor": "…" }
  },
  "nonClaims": [
    "Federated result is adapter continuation on a frozen checkpoint, not federated world-model training (see lewm_tworooms_system_probe.json).",
    "The collapsed representation is a synthetic illustration (rank-1 / magnitude collapse), not a trained model.",
    "The 3D projection (PCA) is a lossy view of a 192-dim latent space; reported variance-explained quantifies the loss.",
    "Planning latency is measured on CPU via the exported graphs; it is not a comparison against any pixel-space world model.",
    "Single local coordinator; no secure aggregation or differential privacy in this path."
  ],
  "crossCheck": {
    "systemProbe": "docs/evidence/lewm_tworooms_system_probe.json",
    "seedSweep": "docs/evidence/lewm_tworooms_probe_seedsweep.json",
    "browserExport": "docs/evidence/lewm_tworooms_browser_export_manifest.json"
  }
}
```

### `passes` predicate (asserted by the test and the producer)
`passes == true` iff **all** of:
- `effectiveRankHealthy >= 5.0` (sanity: not collapsed) **and** `effectiveDimCollapsedRank1 <= 1.5` (the contrast is real).
- `federatedRelativeImprovement > 0.02` (matches the system-probe verdict threshold).
- `0.0 <= goalReachSuccessRate <= 1.0` and `goalReachTrials >= 10`.
- `len(pcaVarianceExplained3d) == 3` and each ∈ [0,1].
- `nonClaims` contains the four mandatory negations (federated-adapter-not-training, synthetic-collapse, projection-lossy, no-pixel-comparison).

### Sourcing rule (claim discipline)
- `federatedRelativeImprovement` and `federatedSeedMeanImprovement` are **read from** the cited existing evidence files at **full float precision**, never recomputed and never truncated (Decision D3). Verified values: `result.relativeImprovement = 0.1227556578424805` (system probe); `distribution.relativeImprovementMean = 0.16787180214169914` (seed-sweep). The producer asserts the source files exist and the values match within 1e-6 — the contract's `0.1227`/`0.168` literals are display roundings only and would **fail** that check if stored.
- The example above lists **five** `nonClaims`; the `passes` predicate requires **at least the four** mandatory negations (federated-adapter-not-training, synthetic-collapse, projection-lossy, no-pixel-comparison) as substrings — the fifth (single-coordinator/no-DP) is recommended and matches the surprise-meter set.
- All other `result` numbers are produced by this run and are deterministic given `seed`.

---

## 3. Versioning & validation
- Both schemas are `name/INT`. A breaking change bumps the int and updates the consumer in the same PR.
- The viewer validates `schema === "cartographer-manifold/1"` on load and refuses to render otherwise (mirror `lewm_runtime.mjs` schema check).
- The evidence test pins `schema == "lewm-manifold/1"` and the `passes` predicate above.
