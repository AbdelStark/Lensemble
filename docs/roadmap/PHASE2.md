# Phase 2 Empirical Evidence Roadmap

Phase 2 turns the claim-MVP into a scaled evidence bundle. It is empirical by
default: larger participant silos, GPU-backed HF Jobs, downstream evaluation,
baselines, curves, and a model-card/report artifact. The cryptographic proof
layer remains governed by RFC-0006 and is not implicitly complete when this
roadmap closes.

Tracker: [#200](https://github.com/AbdelStark/Lensemble/issues/200)

## Baseline

The claim-MVP established the narrow end-to-end path:

- two LeRobot-H5 participant silos;
- claim-grade live-target LeWorldModel objective
  (`objective.target_stop_gradient=false`);
- one closed federated round;
- published dataset/checkpoint/report artifacts;
- final HF Job
  [`6a229653e52fdd2a02ed9125`](https://huggingface.co/jobs/abdelstark/6a229653e52fdd2a02ed9125);
- final global hash
  `cf1c99a7e94ca610daa3bfc00c99d9ee68e9e34a302a96d848508e88edf4c0d5`.

Phase 2 starts from that working substrate and raises the evidence bar.

## Workstreams

| Issue | Workstream | Exit gate |
|---|---|---|
| [#201](https://github.com/AbdelStark/Lensemble/issues/201) | Participant-silo dataset contract and refs | Publish or mount at least two non-toy silo refs, or record the exact blocker. |
| [#202](https://github.com/AbdelStark/Lensemble/issues/202) | GPU-backed multi-round HF Jobs | Completed by job [`6a22ba68e6aa50b87b9ebef7`](https://huggingface.co/jobs/abdelstark/6a22ba68e6aa50b87b9ebef7), which published checkpoint/report artifacts. |
| [#206](https://github.com/AbdelStark/Lensemble/issues/206) | Downstream planning/eval report | Completed by job [`6a22c9e3ece949d7b3dca25a`](https://huggingface.co/jobs/abdelstark/6a22c9e3ece949d7b3dca25a), which published a schema-valid downstream eval report. |
| [#205](https://github.com/AbdelStark/Lensemble/issues/205) | Baselines, ablations, and curves | Generated report [`phase2_baselines_curves_report.json`](../evidence/phase2_baselines_curves_report.json) ties completed points to run/config/checkpoint hashes and blocks missing controls. |
| [#204](https://github.com/AbdelStark/Lensemble/issues/204) | Evidence bundle and model card | Publish one model-card/report bundle with artifact refs and claim boundaries. |
| [#203](https://github.com/AbdelStark/Lensemble/issues/203) | README and roadmap docs | Keep public docs aligned with the tracker and known limitations. |

The machine-readable matrix backing these rows is rendered by:

```bash
uv run --extra dev python scripts/phase2_matrix.py --format markdown
uv run --extra dev python scripts/phase2_matrix.py --format json
```

Dataset refs must pass the participant-silo smoke gate before a GPU run starts:

```bash
uv run --extra dev python scripts/phase2_split_lerobot_h5.py \
  --input /data/source/svla_so100_pickplace.h5 \
  --output-dir /tmp/lensemble-phase2-silos \
  --prefix phase2-so100-silo \
  --num-silos 2

uv run --extra dev python scripts/phase2_dataset_smoke.py \
  --data-source lerobot-h5:///tmp/lensemble-phase2-silos/phase2-so100-silo0.h5 \
  --data-source lerobot-h5:///tmp/lensemble-phase2-silos/phase2-so100-silo1.h5 \
  --participant-id phase2-a \
  --participant-id phase2-b \
  --window-steps 4 \
  --output phase2_dataset_smoke.json
```

The split policy is deterministic episode-level modulo assignment: source
episode `k` goes to silo `k % num_silos`, frames are never duplicated, and each
output HDF5 remaps `episode_index` to local 0-based ids. The split manifest
records the source hash, output hashes, selected source episode ids, frame
counts, and file paths. The smoke JSON report records participant ids, adapter
format, episode/window counts, dataset Merkle roots, action specs, and
first-window tensor shapes. It does not serialize raw observations, raw actions,
or private embeddings.

## Published Phase 2 Data Refs

The first Phase 2 dataset refs are published at
[`abdelstark/lensemble-phase2-so100-silos`](https://huggingface.co/datasets/abdelstark/lensemble-phase2-so100-silos)
revision `97336927606fea6fbfda308bb7cee6e7b48999fa`:

| Participant | File ref | Episodes | Windows (`window_steps=4`) | Dataset root |
|---|---|---:|---:|---|
| `phase2-so100-a` | `hf://datasets/abdelstark/lensemble-phase2-so100-silos/phase2-so100-silo0.h5` | 25 | 3149 | `df4dceed9ee55b95f2827f8b02ec3aa6b86a02421052eb84cfd96b41d7947c0a` |
| `phase2-so100-b` | `hf://datasets/abdelstark/lensemble-phase2-so100-silos/phase2-so100-silo1.h5` | 25 | 3210 | `ce6a42bab6edbdefd47f53f4cfc306cb4ed3db84d9f8ac8f7fcb2adc103c7b52` |

Source/provenance:

- derived from `abdelstark/so100-pickplace-lewm-ready/svla_so100_pickplace.h5`
  at revision `c210cb2f37b42954d31a17027e142c4cbdc7f7f8`;
- upstream HDF5 attrs identify `lerobot/svla_so100_pickplace` at revision
  `3d6d687a25cdf1565cdf24550814f72d999a861d`;
- upstream Hub metadata is public, ungated, and tagged `license:apache-2.0`;
- source SHA-256:
  `9dbeba303311d61a0129a6dcf3d0196524e7d8f58bb823e05dde0101546535ed`;
- silo file SHA-256 values are recorded in
  `phase2_silo_manifest.json` in the dataset repo.

Data contract:

- accepted format: `lerobot-h5`;
- camera/windowing: the current adapter reads `observation/pixels_top`,
  decodes uint8 frames to `[0,1]` float clips of shape `(1, 3, 224, 224)`,
  and produces windows with observation shape `(5, 1, 3, 224, 224)` for
  `window_steps=4`;
- action spec: continuous `lerobot-6dof`, action shape `(4, 6)`;
- declared held-out split policy for #206: the final local episode in each
  silo is reserved for held-out evaluation (`source_episode=48` for
  `phase2-so100-a`, `source_episode=49` for `phase2-so100-b`); train/eval
  reports must record whether they honor or intentionally override this split.

## Published Phase 2 GPU Job

The first GPU-backed multi-round Phase 2 run completed on Hugging Face Jobs:

- HF Job:
  [`6a22ba68e6aa50b87b9ebef7`](https://huggingface.co/jobs/abdelstark/6a22ba68e6aa50b87b9ebef7);
- pinned code SHA:
  `4b446a558882f25e47ee6410a4c32982bbf33477`;
- HF flavor: `t4-small`;
- run shape: two SO-100 silos, `window_steps=4`, `inner_horizon=1`,
  `num_rounds=3`, `metric_windows=8`, `latent_dim=96`, `depth=4`;
- checkpoint repo:
  [`abdelstark/lensemble-phase2-so100-checkpoint`](https://huggingface.co/abdelstark/lensemble-phase2-so100-checkpoint)
  at revision `da52ef380ac87317c89e87f048d65bae65c16b9e`;
- report fields: `schema_version=2`, `round_state=closed`,
  `committed_rounds=3`, `publication.pushed=true`,
  `publication.blocker=None`;
- final global hash:
  `8f1494fd9e57b7496daf96e379a3de1457a435080b81b9e0ea1d20a52f4827c4`;
- config hash:
  `82296109ed452a1aaf494306b100bd4f2b7e3b968a8d4219bc89a3461b0294a3`;
- run manifest hash:
  `890057f5b22a8390f2d3e8b71f1081150694eba70cad0b0ad2b646f99ef75b67`.

The published `claim_mvp_report.json` records these scalar metrics:

| Metric | Value |
|---|---:|
| `val_pred` | 1.513671025633812 |
| `val_sigreg` | 0.15686095133423805 |
| `effective_rank` | 1.5215493440628052 |
| `frame_drift_deg` | 10.538757949205232 |

The schema v2 `round_metrics` series records curve-ready per-round hashes and
update norms:

| Round | Global hash | `phase2-so100-a` update L2 | `phase2-so100-b` update L2 |
|---:|---|---:|---:|
| 0 | `f13dd109b88fc0df26a19153d8406e14a69f1583037e41bb9712325c4ccbb26d` | 0.8818898797 | 0.8815023303 |
| 1 | `541b3c453116c3016d86ff20a7aa09af6860c77c7c6e892976c447818eeb4cd0` | 0.8853848577 | 0.8813801408 |
| 2 | `8f1494fd9e57b7496daf96e379a3de1457a435080b81b9e0ea1d20a52f4827c4` | 0.8827524185 | 0.8824784756 |

This satisfies the #202 engineering gate, but it is intentionally compact. It
does not replace downstream evaluation (#206), baseline/ablation evidence
(#205), or the final model-card/evidence bundle (#204).

## Published Phase 2 Downstream Eval

The first Phase 2 downstream eval report was generated on Hugging Face Jobs:

- HF Job:
  [`6a22c9e3ece949d7b3dca25a`](https://huggingface.co/jobs/abdelstark/6a22c9e3ece949d7b3dca25a);
- eval-runner code SHA:
  `b57aed3da3b6250dce540da25b0bd65c391e68f4`;
- evaluated checkpoint revision:
  `da52ef380ac87317c89e87f048d65bae65c16b9e`, artifact
  `artifacts/round-00003`;
- evaluated checkpoint hash:
  `8f1494fd9e57b7496daf96e379a3de1457a435080b81b9e0ea1d20a52f4827c4`;
- published report URI:
  `hf://models/abdelstark/lensemble-phase2-so100-checkpoint/reports/phase2_downstream_eval_report.json`;
- model repo revision after report upload:
  `021a461eb789700209fcb49e99bb9bcc5d84bfe5`;
- checked-in report copy:
  [`docs/evidence/phase2_downstream_eval_report.json`](../evidence/phase2_downstream_eval_report.json).

The embedded `EvalReport` records:

| Metric | Value |
|---|---:|
| `env_id` | `synthetic://toy` |
| `success_rate` | 0.5 |
| `effective_dim` | 1.0000066342911489 |
| `planning_samples` | 1 |
| `time_per_action_ms` | 91.15450602257624 |
| `run_manifest_hash` | `7092bdda71bb1c01510ce8486e2e99cd3065dfed7834faa3235e6d7a2c7c17fb` |

Planner budget and task boundary:

- planner: `icem`;
- horizon: 1;
- planner iterations: 1;
- held-out policy: two seed-pinned `synthetic://toy` episodes derived from
  `root_seed=0`, not drawn from the SO-100 training silos;
- action clipping: none in the current planner path; the report records
  continuous action bounds `[-1, 1]` for audit;
- claim boundary: checkpoint load, latent MPC execution, planner-cost
  reporting, and residency-safe `EvalReport` emission are evidenced. SO-100
  task success and paper-scale LeWorldModel performance are not claimed.

## Published Phase 2 Baselines And Curves

The Phase 2 baseline/curve artifact is generated by:

```bash
hf download abdelstark/lensemble-phase2-so100-checkpoint \
  claim_mvp_report.json \
  --repo-type model \
  --revision da52ef380ac87317c89e87f048d65bae65c16b9e \
  --local-dir /tmp/lensemble-phase2-hf

hf download abdelstark/lensemble-phase2-so100-naive-fedavg \
  claim_mvp_report.json \
  --repo-type model \
  --revision 8e90bbd09dea96d90c4ae70770e3d6614073971d \
  --local-dir /tmp/lensemble-phase2-naive-hf

uv run --extra dev python scripts/phase2_curves_report.py \
  --anchored-claim-report /tmp/lensemble-phase2-hf/claim_mvp_report.json \
  --naive-fedavg-claim-report /tmp/lensemble-phase2-naive-hf/claim_mvp_report.json \
  --naive-fedavg-job-id 6a22cd9eece949d7b3dca260 \
  --naive-fedavg-revision 8e90bbd09dea96d90c4ae70770e3d6614073971d \
  --output docs/evidence/phase2_baselines_curves_report.json
```

Checked-in report:
[`docs/evidence/phase2_baselines_curves_report.json`](../evidence/phase2_baselines_curves_report.json).
It records source-report hashes for the anchored training report and downstream
eval report, then emits hash-bound rows for:

- per-round update L2 norms for `phase2-so100-a` and `phase2-so100-b`;
- final anchored training scalars: `val_pred`, `val_sigreg`,
  `effective_rank`, and `frame_drift_deg`;
- matched naive-FedAvg / `lambda_anc=0` control rows from HF Job
  [`6a22cd9eece949d7b3dca260`](https://huggingface.co/jobs/abdelstark/6a22cd9eece949d7b3dca260)
  and checkpoint repo
  [`abdelstark/lensemble-phase2-so100-naive-fedavg`](https://huggingface.co/abdelstark/lensemble-phase2-so100-naive-fedavg)
  at revision `8e90bbd09dea96d90c4ae70770e3d6614073971d`;
- downstream eval scalars: `success_rate`, `time_per_action_ms`, and
  `effective_dim`.

Every row includes the source report URI/hash, config hash, checkpoint/global
model hash, and run-manifest or eval-config hash where applicable. The report is
residency-safe and contains no raw observations, actions, latents, embeddings,
or model deltas.

Matched `lambda_anc` comparison:

| Run | `lambda_anc` | Final hash | `val_pred` | `val_sigreg` | `effective_rank` | `frame_drift_deg` |
|---|---:|---|---:|---:|---:|---:|
| Anchored federation | 0.01 | `8f1494fd9e57b7496daf96e379a3de1457a435080b81b9e0ea1d20a52f4827c4` | 1.513671025633812 | 0.15686095133423805 | 1.5215493440628052 | 10.538757949205232 |
| Naive FedAvg control | 0.0 | `12f47b6e58f1ad95994aaa4bf3d15ba980791013d5064af42393ce5412347120` | 1.537665456533432 | 0.15522571466863155 | 1.2233589887619019 | 35.07377879884967 |

Coverage is intentionally conservative. The current generated report blocks
missing local-only, centralized/pooled, and Fork-A rows until matched public runs
exist for the same Phase 2 SO-100 silos, seeds/model shape where possible, and
downstream planner budget. Model-card language must describe this as partial
baseline coverage, not a completed comparative study.

## Minimum Evidence Contract

Before #200 closes, the final report must include:

- dataset repositories or immutable dataset refs;
- participant ids, dataset roots, action specs, frame skip/windowing, and held-out split policy;
- pinned git SHA and HF Job command;
- checkpoint repo and final model hash;
- per-round or curve-ready metrics for prediction loss, SIGReg, effective rank, and frame drift;
- downstream eval report when an environment is available;
- baseline/ablation table, or explicit blockers for missing controls;
- model-card text that distinguishes engineering evidence from paper-scale performance claims.

## Starter HF Jobs Shape

Use the federated launcher with a pinned commit and mounted silos. Prefer a
dry-run before any expensive run:

```bash
hf jobs uv run --flavor h200 --timeout 2h --secrets HF_TOKEN \
  --with 'lensemble @ git+https://github.com/AbdelStark/Lensemble.git@<SHA>' \
  -v hf://datasets/abdelstark/lensemble-phase2-so100-silos:/data/phase2 \
  -d https://raw.githubusercontent.com/AbdelStark/Lensemble/<SHA>/deploy/hfjobs/train_federated_lewm.py \
  --data-source lerobot-h5:///data/phase2/phase2-so100-silo0.h5 \
  --data-source lerobot-h5:///data/phase2/phase2-so100-silo1.h5 \
  --participant-id phase2-so100-a \
  --participant-id phase2-so100-b \
  --out-dir /tmp/lensemble-phase2 \
  --image-size 224 --patch-size 14 --latent-dim 192 \
  --depth 12 --predictor-depth 6 --num-heads 3 \
  --probe-points 1024 --inner-horizon 4 --window-steps 4 \
  --num-rounds 8 --metric-windows 256 \
  --push --out-repo <org>/lensemble-phase2-checkpoint
```

Adjust model scale and HF flavor only after a smaller dry-run verifies mounts,
window counts, probe generation, and report publication.

## Claim Boundary

Closing Phase 2 should support a stronger engineering statement: Lensemble can
train and evaluate a federated end-to-end JEPA-style world model on larger
participant-local robot silos with published artifacts. It should not claim
paper-scale LeWorldModel performance, broad robotics generalization, or
cryptographic contribution proofs unless those are separately evidenced.
