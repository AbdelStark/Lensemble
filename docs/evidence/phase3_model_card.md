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

- Consortium id: `lensemble-phase3-long-run-smoke`
- Run id: `phase3-long-run-smoke-v1`
- Participant agents: 4
- Coordinator: `phase3-long-run-coordinator`
- Protocol: `phase3-consortium-v1`
- Public probe hash: `ddc8349fccfec07a41847e880f19574969d2a00de85a9353dd8c87cdeb7dfea2`
- Secure aggregation backend: `simulated`
- DP accountant: `rdp`

## Training And Evaluation Scale

- Closed rounds: 10/10
- Tiny model shape: `latent_dim=8`,
  `num_tokens=4`
- Config hash: `a4f6bbabb919735c7679320c3f204cd0b1298e046e30cce8d11cbbbc8df07e57`
- Final checkpoint hash: `ed3081ee514af142a226443f113a37c24d7d5872bfb707f11abe10893a0ad50d`
- Run-manifest hash: `cfaf14dcbe7c3fe9e64f2257729c047df448bd35218b387412f6a5d479c3169a`
- Training evidence is a deterministic local consortium smoke, not a public
  HF Jobs robotics-scale result.

## Completed And Blocked Controls

- Completed controls: `anchored-federation`
- Blocked controls: `local-only`, `naive-fedavg`, `fork-a-frozen-encoder`
- Eval/control metric rows: 4

Phase 3 evaluation evidence is currently limited to the local deterministic consortium-runtime smoke: participant-agent updates, ten closed rounds, secure-sum reporting, and DP accounting. Public task-scale SO-100 downstream evaluation remains blocked until the Phase 3 checkpoint and held-out eval data are published. Blocked controls: local-only, naive-fedavg, fork-a-frozen-encoder. These rows must not be described as completed robotics performance comparisons.

## Privacy And Observability Controls

- Secure-sum rounds: 10
- DP-accounted rounds: 10
- Max per-round epsilon spent: 5.302585092994046
- Observability round summaries: 11
- Induced dropout outcomes: induced-dropout-close-with-quorum:closed
- Redaction contract: `phase3-observability-redaction-v1`

## Dataset And Publication Status

- Dataset registry: `lensemble-phase3-long-run-smoke:phase3-long-run-smoke-v1:dataset-probe-registry`
- Dataset run mode: `public_example`
- Participant data declarations: 4
- Raw data crosses participant boundary: `False`
- Model repo target: `hf://models/abdelstark/lensemble-phase3-consortium-checkpoint@local-smoke`
- Dataset repo target: `hf://datasets/abdelstark/lensemble-phase3-consortium-data@local-smoke`
- Publication status: `local_smoke`

## Claim Boundaries

- Consortium-runtime evidence: four simulated sovereign participant agents completed a governed local Phase 3 run with ten closed federated rounds.
- Training/eval scale: this is a deterministic tiny-model local smoke, not a public HF Jobs robotics-scale training result.
- Controls: anchored-federation runtime metrics are completed; local-only, naive-FedAvg, and Fork-A/frozen-encoder Phase 3 controls remain blocked rows.
- Privacy controls: secure_sum aggregation status and DP accounting are exercised as operational controls, not cryptographic computation proofs.

## Non-Claims

- Phase 3 does not include a provenance ledger implementation.
- Phase 3 does not cryptographically prove honest participant computation.
- Phase 3 does not claim paper-scale LeWorldModel performance.
- Phase 3 does not claim public SO-100 robotics task success.

## Known Limitations

- local-only: No matched local-only Phase 3 run is published for the same participant data refs, seed, model size, and eval budget.
- naive-fedavg: No matched lambda_anc=0 / unanchored Phase 3 consortium control is published for the same run shape and eval budget.
- fork-a-frozen-encoder: The RFC-0002 Fork A frozen-encoder safe-degrade baseline has not been run for the Phase 3 consortium manifest.
- Public task-scale SO-100 downstream evaluation is blocked until a public Phase 3 task checkpoint and held-out eval dataset are published.

## Reports In This Release Candidate

- `reports/phase3_evidence_bundle.json`
- `reports/phase3_long_run_smoke_report.json`
- `reports/phase3_eval_report.json`
- `reports/phase3_observability_report.json`
- `reports/phase3_long_run_manifest.json`
- `reports/phase3_long_run_dataset_registry.json`
- `artifacts/final/header.json`
- `artifacts/final/weights.safetensors`
