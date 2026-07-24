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

This model repository preserves historical Phase 3 consortium-training audit evidence for a federated JEPA / LeWorldModel-flavour world model.


## Historical Evidence Status

These training and gauge rows predate the outer-update direction correction and
the `public-probe-v2` target-binding contract. They are retained for audit
history only and **do not validate the corrected runtime**. GitHub issue #335
tracks the required rerun and replacement publication.


## Consortium Runtime Evidence

- Consortium id: `lensemble-phase3-long-run-smoke`
- Run id: `phase3-long-run-smoke-v1`
- Participant agents: 4
- Coordinator: `phase3-long-run-coordinator`
- Protocol: `phase3-consortium-v1`
- Public probe hash: `ddc8349fccfec07a41847e880f19574969d2a00de85a9353dd8c87cdeb7dfea2`
- Public probe hash contract: `legacy-unscoped`
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

- Completed controls: `anchored-federation`, `fork-a-frozen-encoder`, `local-only`, `naive-fedavg`
- Blocked controls: none
- Eval/control metric rows: 12

Historical evidence status: these training and gauge rows predate the outer-update direction correction and the public-probe-v2 target-binding contract. They are retained for audit history only and do not validate the corrected runtime; rerun tracking is recorded in GitHub issue #335. Phase 3 evaluation evidence covers the local deterministic consortium-runtime smoke (participant-agent updates, 10 closed rounds, post-commit sum cross-check reporting, and one-round deterministic mechanism-accounting snapshots). Matched gauge controls are included only when their immutable reports and run manifests are supplied to the producer. Hash-bound historical gauge finding: anchored round-0 frame drift 48.9695 deg versus naive-FedAvg 180 deg; Fork-A 0 deg; local-only drift 180 deg with mean effective rank 120.316. These representation metrics are not robotics task-performance results. Historical limitation: at the old default outer step (outer_lr=0.7), the federated global representation collapsed over rounds (effective_rank -> 1). That trajectory also predates the outer-update correction, so it is failure-analysis history rather than a clean anchored-vs-naive result. Closed-loop task-scale SO-100 evaluation remains unrun; the published recorded splits cannot apply arbitrary planner actions to a live environment. Completed matched controls bound to published run hashes: anchored-federation, naive-fedavg, fork-a-frozen-encoder, local-only. These are representation-gauge controls and must not be described as completed robotics performance comparisons.

## Privacy And Observability Controls

- Optimizer-consumed secure-sum rounds: 0
- Post-commit sum cross-check rounds: 10
- One-round DP mechanism-accounting snapshots: 10
- Claim-grade effective-DP rounds: 0
- Max one-round epsilon snapshot: 5.302585092994046
- Observability round summaries: 11
- Induced dropout outcomes: induced-dropout-close-with-quorum:closed
- Redaction contract: `phase3-observability-redaction-v1`

## Dataset And Publication Status

- Dataset registry: `lensemble-phase3-long-run-smoke:phase3-long-run-smoke-v1:dataset-probe-registry`
- Dataset run mode: `public_example`
- Participant data declarations: 4
- Raw data crosses participant boundary: `False`
- Model repo target: `hf://models/abdelstark/lensemble-phase3-consortium-checkpoint@828e210cba4870b2be4ab573a5f0dd4ee30bae29`
- Dataset repo target: `hf://datasets/abdelstark/lensemble-phase3-so100-silos@15f71911432b300dfdf41c998e27492e8c986be4`
- Dataset revision bound to this training run: `False`
- Publication status: `historical`

## Claim Boundaries

- Historical evidence status: these Phase 3 training and gauge rows predate the outer-update direction correction and the public-probe-v2 target-binding contract. They are audit history and do not validate the corrected runtime; issue #335 tracks the rerun.
- Consortium-runtime evidence: four simulated participant agents completed a deterministic tiny-model lifecycle smoke with ten closed rounds; the optimizer consumed plaintext participant updates, while simulated sums were post-commit equivalence cross-checks.
- Training/eval scale: the bundled training run uses placeholder synthetic smoke data and tiny tokens/latent. Separately hash-bound control reports contain SO-100 representation metrics; neither surface is a paper-scale robotics result.
- Controls: anchored-federation, naive-FedAvg, Fork-A/frozen-encoder, and local-only controls are completed as hash-bound representation-metric rows; no matched control rows remain blocked.
- Privacy controls: deterministic noise and fresh one-round epsilon snapshots exercise mechanism plumbing but are not claim-grade effective DP or cumulative budget enforcement.

## Non-Claims

- The historical Phase 3 metrics are not evidence for the corrected outer update or the target-bound probe contract.
- Phase 3 does not include a provenance ledger implementation.
- Phase 3 does not cryptographically prove honest participant computation.
- Phase 3 does not claim paper-scale LeWorldModel performance.
- Phase 3 does not claim public SO-100 robotics task success.
- The bundle combines a synthetic runtime smoke with separately hash-bound historical real-data control reports; neither surface is a cryptographic honest-computation proof.

## Known Limitations

- Corrected-runtime rerun (#335): every training/gauge metric in this bundle was produced before the outer-update sign and probe-target-binding fixes and must be regenerated before comparison or promotion.
- Secure aggregation is not integrated into the live optimizer path: current simulated/TEE sums are post-commit cross-checks over plaintext updates already consumed by the coordinator.
- Current DP noise is deterministically replayable from shared run configuration, and accounting is recreated per round; effective DP and cumulative enforcement are not claimed.
- DP-utility / federated-collapse (#244): the historical checkpoints exhibit global-representation collapse over rounds under the configured deterministic clipping/noise settings. This is a failure observation, not evidence of an effective or cumulative DP budget, and downstream planning success would be uninformative on these checkpoints.
- Downstream task-success (stable-worldmodel #96): closed-loop physical SO-100 task success is deferred, not claimed; it requires the unvendored stable-worldmodel planner suite and a non-collapsing federated checkpoint, because a recorded held-out split is open-loop and cannot apply arbitrary planner actions to recorded frames.

## Reports In This Historical Record

- `reports/phase3_evidence_bundle.json`
- `reports/phase3_long_run_smoke_report.json`
- `reports/phase3_eval_report.json`
- `reports/phase3_observability_report.json`
- `reports/phase3_long_run_smoke_manifest.json`
- `reports/phase3_long_run_smoke_dataset_registry.json`
- `artifacts/final/header.json`
- `artifacts/final/weights.safetensors`
