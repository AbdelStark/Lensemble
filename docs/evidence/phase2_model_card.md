---
license: apache-2.0
library_name: lensemble
tags:
- federated-learning
- world-model
- jepa
- robotics
- phase2
---

# Lensemble Phase 2 SO-100 Federated JEPA World Model

This model repository contains the Phase 2 engineering evidence bundle for a
federated JEPA-style world-model run over two public SO-100 participant silos.

## Dataset Refs

Dataset repo: `hf://datasets/abdelstark/lensemble-phase2-so100-silos@97336927606fea6fbfda308bb7cee6e7b48999fa`

| Participant | File ref | Episodes | Windows | Dataset root |
|---|---|---:|---:|---|
| phase2-so100-a | hf://datasets/abdelstark/lensemble-phase2-so100-silos/phase2-so100-silo0.h5 | 25 | 3149 | `df4dceed9ee55b95f2827f8b02ec3aa6b86a02421052eb84cfd96b41d7947c0a` |
| phase2-so100-b | hf://datasets/abdelstark/lensemble-phase2-so100-silos/phase2-so100-silo1.h5 | 25 | 3210 | `ce6a42bab6edbdefd47f53f4cfc306cb4ed3db84d9f8ac8f7fcb2adc103c7b52` |

Split policy: `episode_modulo`. Held-out policy:
final_local_episode_per_silo for held-out evaluation for Phase 2 issue #206

## Training

- HF Job: [6a22ba68e6aa50b87b9ebef7](https://huggingface.co/jobs/abdelstark/6a22ba68e6aa50b87b9ebef7)
- Pinned code SHA: `4b446a558882f25e47ee6410a4c32982bbf33477`
- Checkpoint revision: `abdelstark/lensemble-phase2-so100-checkpoint@da52ef380ac87317c89e87f048d65bae65c16b9e`
- Config hash: `82296109ed452a1aaf494306b100bd4f2b7e3b968a8d4219bc89a3461b0294a3`
- Final global hash: `8f1494fd9e57b7496daf96e379a3de1457a435080b81b9e0ea1d20a52f4827c4`
- Committed rounds: 3
- Metrics: `val_pred=1.513671025633812`, `val_sigreg=0.15686095133423805`,
  `effective_rank=1.5215493440628052`,
  `frame_drift_deg=10.538757949205232`

## Downstream Eval

- HF Job: [6a22c9e3ece949d7b3dca25a](https://huggingface.co/jobs/abdelstark/6a22c9e3ece949d7b3dca25a)
- Env/planner: `synthetic://toy` / `icem`
- Success rate: 0.5
- Time per action: 91.15450602257624 ms
- Effective dimension: 1.0000066342911489
- Eval config hash: `f926ca5d1f230c960b6c10810a7f42620e99a05f144ac2a45991541940d014d4`

## Baselines And Curves

The generated curve report has 23 rows over
`anchored-federation`, `naive-fedavg`. Phase 2 baseline coverage is partial: the generated table includes only completed, hash-bound public runs. Blocked comparisons: local-only, centralized-pooled, fork-a. Blocked rows must not be described as completed comparisons.

## Claim Boundaries

- Engineering-scale evidence: published SO-100 participant silos, a GPU-backed three-round federated JEPA-style run, downstream synthetic planning eval, and a matched lambda_anc=0 control.
- Does not claim paper-scale LeWorldModel performance, SO-100 task success, broad robotics generalization, or completed RFC-0006 cryptographic contribution proofs.
- Baseline coverage is partial; blocked comparisons remain blocked until matched public runs exist.

## Known Gaps

- local-only: No matched local-only Phase 2 SO-100 run is published for the same silos, seed, model size, and eval budget.
- centralized-pooled: No centralized/pooled run is published; pooling these participant silos was not executed for this artifact.
- fork-a: The RFC-0005 Fork-A safe-degrade baseline has not been run on the Phase 2 SO-100 silos.

## Public Reports

- `reports/phase2_downstream_eval_report.json`
- `reports/phase2_baselines_curves_report.json`
- `reports/phase2_evidence_bundle.json`
