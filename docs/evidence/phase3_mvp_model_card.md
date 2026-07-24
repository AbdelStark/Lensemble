---
license: apache-2.0
library_name: lensemble
tags:
- federated-learning
- world-model
- jepa
- robotics
- phase3
- historical
---

# Lensemble MVP — Historical SO-100 Gauge Audit

## Historical Evidence Status

These runs predate the correction of the outer-update direction and the public-probe-v2 target-binding contract. Their metrics are retained for audit history only and do not validate the corrected runtime; GitHub issue #335 tracks the replacement run.

This card preserves the immutable Phase 3 MVP measurements for provenance and
failure analysis. It is **not current correctness evidence** and must not be
used to claim that the corrected distributed training loop converges.

## Recorded Run Shape

- Four participant silos and one disjoint held-out SO-100 split
- From-scratch latent model (`latent_dim=256`, `depth=8`, `image_size=224`)
- Simulated secure aggregation
- Relaxed-DP (DP-off) gauge-probe regime

## Historical Metrics

| Control | Final effective rank | Final prediction loss | Final frame drift |
|---|---:|---:|---:|
| Anchored federation | 14.815850257873535 | 22.19140625 | 83.59595115984035 |
| Naive FedAvg | 2.6838886737823486 | 203776.0 | 180.0 |
| Local only | 105.507 (mean) | 0.025432 (mean) | 180.0 |

Pinned immutable revisions: `anchored-federation` `a6f5a96174f76af90ef3f4975bdda101e5ce6d45` · `naive-fedavg` `cd8481c4c041d610008491f13e64bf092d3bc94d` · `local-only` `9345bc3cf831238508e540981a15e8884acedec1`

These numbers describe the superseded implementation. Under that implementation,
the anchored configuration exhibited less gauge collapse than naive FedAvg, but
the sign error in the outer update prevents treating the comparison as evidence
for the corrected runtime.

## Claim Boundaries

- Held-out magnitude collapse was approximately `~7.5e-6` latent variance; the
  checked-in downstream and inference reports preserve that audit trail. The
  associated central ceiling diagnostic did not establish downstream
  usefulness.
- In plain terms: skill_vs_identity is gameable; effective_rank is
  scale-invariant. Therefore,
  neither is a binding usefulness metric.
- Latent-MPC `success_rate=0.0` is a negative result.
- The SO-100 checkpoint is not a downstream-useful world model.
- No closed-loop physical SO-100 task success, paper-scale LeWorldModel
  performance, effective differential privacy, secure-sum optimizer integration,
  or cryptographic honest-computation proof is claimed.
- [Issue #335](https://github.com/AbdelStark/Lensemble/issues/335) tracks a clean
  rerun using the corrected outer update and target-bound public probe.

Machine-readable source: `phase3_mvp_benchmark_report.json`.
