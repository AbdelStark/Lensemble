---
license: apache-2.0
library_name: lensemble
tags:
- federated-learning
- world-model
- jepa
- robotics
- phase3
---

# Lensemble Phase 3 Consortium JEPA World Model

This model repository records the Phase 3 consortium-training release-candidate
evidence for a federated JEPA / LeWorldModel-flavour world model.

## Consortium Runtime Evidence

- Consortium id: `lensemble-phase3-consortium`
- Run id: `phase3-consortium-v1`
- Participant agents: 4
- Coordinator: `lensemble-phase3-consortium-coordinator`
- Protocol: `phase3-consortium-v1`
- Public probe hash: `f1053ffdf6402f2fd1e8327fe1e0eb8b3f586b32448e6b77ab6ca4f0eb59faad`
- Secure aggregation backend: `simulated`
- DP accountant: `rdp`

## Training And Evaluation Scale

- Closed rounds: 10/10
- Tiny model shape: `latent_dim=256`,
  `num_tokens=196`
- Config hash: `27f2c77c9d47a7d053c01ab65f8d43aad79463b27d882f2d85ec28bc062cb2b2`
- Final checkpoint hash: `bb31c0922de639cb9220c4cc5fc35d79aec719eb6fcedb09159bdff8cfb8fd43`
- Run-manifest hash: `21819c9b936468ffc38f943b4ce13ec2ac150d328410f503fa73d9014e040c9d`
- Training evidence is a deterministic local consortium smoke, not a public
  HF Jobs robotics-scale result.

## Completed And Blocked Controls

- Completed controls: `anchored-federation`, `fork-a-frozen-encoder`, `local-only`, `naive-fedavg`
- Blocked controls: none
- Eval/control metric rows: 10

Phase 3 evaluation evidence covers the local deterministic consortium-runtime smoke (participant-agent updates, ten closed rounds, secure-sum reporting, and DP accounting) plus four real matched control runs published on HF Jobs (DP-off, latent_dim=256, 6 rounds, window_steps=4, simulated secure-agg, four participants phase3-so100-a..d, held-out silo4). Gauge finding: the frame anchor reduces inter-participant latent frame-drift at aggregation (anchored round-0 48.97 deg vs naive-FedAvg 180 deg); Fork-A's frozen encoder is the 0 deg safe-degrade baseline; and local-only silos train healthily (effective_rank ~120) but diverge maximally (180 deg) - the divergence federation is designed to close. Honest limitation: at the default outer-step (outer_lr=0.7) with a random-init warm-start (real V-JEPA weights remain unvendored, #96), the federated global representation collapses over rounds (effective_rank -> 1), so the clean anchored-vs-naive contrast is the round-0 measurement; sustained non-collapsing federated training is a documented follow-up. This report is consortium-engineering and training evidence, NOT a cryptographic proof of honest participant computation. Public task-scale SO-100 downstream evaluation remains blocked until the Phase 3 checkpoint and held-out eval data are published. Completed matched controls bound to published run hashes: naive-fedavg, fork-a-frozen-encoder, local-only. These are representation-gauge controls and must not be described as completed robotics performance comparisons.

## Privacy And Observability Controls

- Secure-sum rounds: 10
- DP-accounted rounds: 10
- Max per-round epsilon spent: 5.302585092994046
- Observability round summaries: 11
- Induced dropout outcomes: induced-dropout-close-with-quorum:closed
- Redaction contract: `phase3-observability-redaction-v1`

## Dataset And Publication Status

- Dataset registry: `lensemble-phase3-consortium:phase3-consortium-v1:dataset-probe-registry`
- Dataset run mode: `public_example`
- Participant data declarations: 4
- Raw data crosses participant boundary: `False`
- Model repo target: `hf://models/abdelstark/lensemble-phase3-consortium-checkpoint@828e210cba4870b2be4ab573a5f0dd4ee30bae29`
- Dataset repo target: `hf://datasets/abdelstark/lensemble-phase3-consortium-data@15f71911432b300dfdf41c998e27492e8c986be4`
- Publication status: `published`

## Claim Boundaries

- Consortium-runtime evidence: four sovereign participant agents on the union SO-100 action contract completed a governed Phase 3 run with ten closed federated rounds, secure-sum aggregation, and DP accounting.
- Training/eval scale: this is consortium-engineering and real-training evidence on tiny tokens/latent, not a public HF Jobs paper-scale robotics training result.
- Controls: anchored-federation, naive-FedAvg, Fork-A/frozen-encoder, and local-only controls are completed as representation-metric rows; no Phase 3 control rows remain blocked.
- Privacy controls: secure_sum aggregation status and DP accounting are exercised as operational controls, not cryptographic computation proofs (RFC-0006 honest-computation proofs remain out of scope).

## Non-Claims

- Phase 3 does not include a provenance ledger implementation.
- Phase 3 does not cryptographically prove honest participant computation.
- Phase 3 does not claim paper-scale LeWorldModel performance.
- Phase 3 does not claim public SO-100 robotics task success.
- Phase 3 is consortium-engineering and real-training evidence, not a cryptographic honest-computation proof; RFC-0006 honest-computation proofs are out of scope.

## Known Limitations

- DP-utility / federated-collapse (#244): the published checkpoints exhibit global-representation collapse over rounds under the DP noise/clipping budget, so latent quality degrades and downstream planning success would be uninformative on these checkpoints.
- Downstream task-success (stable-worldmodel #96): closed-loop physical SO-100 task success is deferred, not claimed; it requires the unvendored stable-worldmodel planner suite and a non-collapsing federated checkpoint, because a recorded held-out split is open-loop and cannot apply arbitrary planner actions to recorded frames.

## Reports In This Release Candidate

- `reports/phase3_evidence_bundle.json`
- `reports/phase3_long_run_smoke_report.json`
- `reports/phase3_eval_report.json`
- `reports/phase3_observability_report.json`
- `reports/phase3_long_run_manifest.json`
- `reports/phase3_long_run_dataset_registry.json`
- `artifacts/final/header.json`
- `artifacts/final/weights.safetensors`
