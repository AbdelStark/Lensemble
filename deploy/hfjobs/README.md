# HF Jobs — LeWorldModel training

[`train_federated_lewm.py`](train_federated_lewm.py) is the claim-MVP launcher: it runs the real
Lensemble federated runtime (`Coordinator` + default `Participant` hooks) over mounted `lerobot-h5://`
participant data sources, writes committed checkpoints plus `claim_mvp_report.json`, and can push the
dataset/checkpoint artifacts to the Hub when `HF_TOKEN` is available.

[`train_lewm.py`](train_lewm.py) is the older single-site trainer. It trains the in-tree LeWorldModel (video-ViT `f_θ` + action-conditioned
`g_φ` + the SIGReg-JEPA objective) **from scratch (no warm start)** on a real robot dataset, on a
Hugging Face Jobs GPU. The dataset is loaded through the official `lerobot-h5://` data source
(`lensemble.data.adapters`) — the same `EpisodeDataset` → `Window` contract the federated `Participant`
consumes — so this is the real pipeline, not a bespoke loader.

## Federated Claim-MVP Run

Each mounted HDF5 file is one participant silo:

```bash
hf jobs uv run --flavor h200 --secrets HF_TOKEN \
  -v hf://datasets/<org>/silo-a:/data/a \
  -v hf://datasets/<org>/silo-b:/data/b \
  https://raw.githubusercontent.com/AbdelStark/Lensemble/main/deploy/hfjobs/train_federated_lewm.py -- \
  --data-source lerobot-h5:///data/a/silo0.h5 \
  --data-source lerobot-h5:///data/b/silo1.h5 \
  --num-rounds 1 --inner-horizon 1 \
  --lambda-sig 0.1 --lambda-anc 0.01 \
  --out-repo <org>/lewm-federated-claim-mvp \
  --push
```

Use `--dry-run` first to validate the mounts, generate the probe, and emit a dry-run
`claim_mvp_report.json` without training or publishing. Omit `--push` to keep artifacts only in the job
filesystem. Add repeated `--dataset-repo <org>/<dataset>` values, one per `--data-source`, when the job
should publish the mounted HDF5 sources too.

By default the federated launcher runs claim-grade LeWorldModel target mode
(`objective.target_stop_gradient=false`). Pass `--target-stop-gradient` only for the legacy detached
target helper.

The final claim-MVP job
[`6a229653e52fdd2a02ed9125`](https://huggingface.co/jobs/abdelstark/6a229653e52fdd2a02ed9125) published
`claim_mvp_report.json` to `abdelstark/lensemble-claim-mvp-checkpoint` with a closed round, pushed
artifacts, final global hash
`cf1c99a7e94ca610daa3bfc00c99d9ee68e9e34a302a96d848508e88edf4c0d5`, and non-null
`frame_drift_deg`.

## Phase 2 Run Shape

Phase 2 is tracked in [#200](https://github.com/AbdelStark/Lensemble/issues/200) and
[`docs/roadmap/PHASE2.md`](../../docs/roadmap/PHASE2.md). It raises the evidence bar from a tiny
claim-MVP smoke to larger participant silos, GPU-backed multi-round jobs, downstream evaluation,
baselines/ablations, curves, and a model-card/evidence bundle.

Render the current experiment matrix with:

```bash
uv run --extra dev python scripts/phase2_matrix.py --format markdown
```

Split a single publishable LeRobot-H5 source into participant silos, then
smoke-test the mounted refs before starting a GPU job:

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

The split policy is deterministic episode-level modulo assignment (`episode k →
k % num_silos`) and writes a manifest with source/output hashes and selected
source episode ids. The smoke report is residency-safe metadata: participant
ids, adapter format, episode/window counts, Merkle roots, action specs, and
first-window tensor shapes. It contains no raw observations/actions.

The published Phase 2 SO-100 refs are in
[`abdelstark/lensemble-phase2-so100-silos`](https://huggingface.co/datasets/abdelstark/lensemble-phase2-so100-silos)
at revision `97336927606fea6fbfda308bb7cee6e7b48999fa`:

- `phase2-so100-a`: `phase2-so100-silo0.h5`, 25 episodes, 3149 windows at
  `window_steps=4`, dataset root
  `df4dceed9ee55b95f2827f8b02ec3aa6b86a02421052eb84cfd96b41d7947c0a`.
- `phase2-so100-b`: `phase2-so100-silo1.h5`, 25 episodes, 3210 windows at
  `window_steps=4`, dataset root
  `ce6a42bab6edbdefd47f53f4cfc306cb4ed3db84d9f8ac8f7fcb2adc103c7b52`.

The current adapter reads `observation/pixels_top`, decodes uint8 frames to
`[0,1]` float clips, and uses the continuous `lerobot-6dof` action spec. The
declared held-out policy reserves the final local episode in each silo for #206
evaluation (`source_episode=48` and `49` respectively).

The first published GPU-backed Phase 2 job completed on `t4-small`:

- job
  [`6a22ba68e6aa50b87b9ebef7`](https://huggingface.co/jobs/abdelstark/6a22ba68e6aa50b87b9ebef7);
- pinned commit `4b446a558882f25e47ee6410a4c32982bbf33477`;
- three closed federated rounds over the two SO-100 silos;
- checkpoint/report repo
  [`abdelstark/lensemble-phase2-so100-checkpoint`](https://huggingface.co/abdelstark/lensemble-phase2-so100-checkpoint)
  at revision `da52ef380ac87317c89e87f048d65bae65c16b9e`;
- final global hash
  `8f1494fd9e57b7496daf96e379a3de1457a435080b81b9e0ea1d20a52f4827c4`;
- report metrics: `val_pred=1.513671025633812`,
  `val_sigreg=0.15686095133423805`,
  `effective_rank=1.5215493440628052`, and
  `frame_drift_deg=10.538757949205232`.

The run used a compact Phase 2 shape (`latent_dim=96`, `depth=4`,
`inner_horizon=1`, `metric_windows=8`) to prove the GPU publication path before
larger sweeps. Its `claim_mvp_report.json` uses schema v2 and contains three
`round_metrics` rows with post-round global hashes, participant ids, dataset
roots, and update L2 norms.

The first published downstream eval job
[`6a22c9e3ece949d7b3dca25a`](https://huggingface.co/jobs/abdelstark/6a22c9e3ece949d7b3dca25a)
ran `scripts/phase2_eval_checkpoint.py` from
`b57aed3da3b6250dce540da25b0bd65c391e68f4`, downloaded checkpoint revision
`da52ef380ac87317c89e87f048d65bae65c16b9e`, and uploaded
`reports/phase2_downstream_eval_report.json` to the model repo at revision
`021a461eb789700209fcb49e99bb9bcc5d84bfe5`. The report is also checked in at
[`docs/evidence/phase2_downstream_eval_report.json`](../../docs/evidence/phase2_downstream_eval_report.json).
It records `synthetic://toy`, `success_rate=0.5`,
`effective_dim=1.0000066342911489`, `planning_samples=1`, horizon 1, one
planner iteration, and no action clipping beyond recording the continuous
`[-1, 1]` bounds.

Generate the Phase 2 baseline/curve report from local copies of the public
training and downstream reports:

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

The checked-in
[`phase2_baselines_curves_report.json`](../../docs/evidence/phase2_baselines_curves_report.json)
binds every curve row to source-report/config/checkpoint hashes. It includes the
matched `lambda_anc=0` naive-FedAvg control from HF Job
[`6a22cd9eece949d7b3dca260`](https://huggingface.co/jobs/abdelstark/6a22cd9eece949d7b3dca260)
and model repo
[`abdelstark/lensemble-phase2-so100-naive-fedavg`](https://huggingface.co/abdelstark/lensemble-phase2-so100-naive-fedavg)
revision `8e90bbd09dea96d90c4ae70770e3d6614073971d`. Missing local-only,
centralized/pooled, and Fork-A comparisons remain blocked until matched public
runs exist. Do not describe those blocked rows as completed baselines in
model-card text.

Generate the final Phase 2 evidence bundle and model card after the curves
report has been uploaded to the checkpoint repo:

```bash
uv run --extra dev python scripts/phase2_bundle.py \
  --dataset-smoke /tmp/lensemble-phase2-data-hf/phase2_dataset_smoke.json \
  --dataset-manifest /tmp/lensemble-phase2-data-hf/phase2_silo_manifest.json \
  --training-claim-report /tmp/lensemble-phase2-hf/claim_mvp_report.json \
  --curves-revision 8643d9f60eeb997afd5b254d525a145769d59c68 \
  --output docs/evidence/phase2_evidence_bundle.json \
  --model-card-output docs/evidence/phase2_model_card.md
```

The published checkpoint repo revision
`eaf13136b42cde324758a191c98e377636ded7f8` contains the generated `README.md`,
`reports/phase2_evidence_bundle.json`, `reports/phase2_model_card.md`, and
`reports/phase2_baselines_curves_report.json`.

Start every expensive run with the dataset smoke, `--dry-run`, and a pinned SHA.
A representative GPU command is:

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

## Phase 3 Consortium Run

[`train_phase3_consortium.py`](train_phase3_consortium.py) drives the **full** Phase 3 consortium
runtime — the networked `Phase3CoordinatorService` plus one sovereign `Phase3ParticipantAgent` per
mounted participant-local data ref — for `--num-rounds` closed federated rounds and emits REAL
residency-safe per-round JEPA metrics (`val_pred` / `val_sigreg` / `effective_rank` / `frame_drift_deg`)
measured off the committed global checkpoints and a disjoint held-out eval split. It builds the agreed
consortium manifest and dataset/probe registry **from the actual loaded data** (deriving the action and
observation contracts from each silo's `ActionSpec` and first-window shape), pins the public-probe hash,
and calls the frozen library entry point `lensemble.federation.run_phase3_consortium`. No raw participant
trajectory ever leaves a participant boundary; only pseudo-gradients and residency-safe metadata cross.

Each mounted store is one sovereign participant silo. Hold one disjoint split out for the residency-safe
per-round metrics via `--heldout-source` (required for a real run). Always validate first with
`--dry-run`, which pins the probe hash, builds + validates the manifest and registry, and preflights
every participant agent **without running any federated round or any training compute**, writing
`phase3_consortium_dry_run.json`:

The published Phase 3 silos + held-out split are
[`abdelstark/lensemble-phase3-so100-silos`](https://huggingface.co/datasets/abdelstark/lensemble-phase3-so100-silos)
(#242): four participant silos `phase3-so100-silo{0..3}.h5` and the disjoint held-out split
`phase3-so100-silo4.h5`. The model shape (`latent_dim=256`, `patch_size=16` → 196 tokens) and the
`--probe-points 512` / seed `20260608` pin match the public-probe hash recorded in the dataset registry.

```bash
hf jobs uv run --flavor h200 --timeout 2h --secrets HF_TOKEN \
  --with 'lensemble @ git+https://github.com/AbdelStark/Lensemble.git@<SHA>' \
  -v hf://datasets/abdelstark/lensemble-phase3-so100-silos:/data/phase3 \
  -d https://raw.githubusercontent.com/AbdelStark/Lensemble/<SHA>/deploy/hfjobs/train_phase3_consortium.py \
  --data-source lerobot-h5:///data/phase3/phase3-so100-silo0.h5 \
  --data-source lerobot-h5:///data/phase3/phase3-so100-silo1.h5 \
  --data-source lerobot-h5:///data/phase3/phase3-so100-silo2.h5 \
  --data-source lerobot-h5:///data/phase3/phase3-so100-silo3.h5 \
  --participant-id phase3-so100-a \
  --participant-id phase3-so100-b \
  --participant-id phase3-so100-c \
  --participant-id phase3-so100-d \
  --heldout-source lerobot-h5:///data/phase3/phase3-so100-silo4.h5 \
  --out-dir /tmp/lensemble-phase3 \
  --image-size 224 --patch-size 16 --latent-dim 256 \
  --depth 6 --predictor-depth 4 --num-heads 8 \
  --probe-points 512 --inner-horizon 2 --window-steps 4 \
  --num-rounds 10 --metric-windows 256 \
  --secure-agg-backend simulated --secure-agg-threshold 4 --min-trainers 3 \
  --privacy --dp-epsilon 8.0 --dp-delta 1e-5 --dp-clip-norm 0.5 \
  --dp-noise-multiplier 1.0 --dp-accountant rdp \
  --consortium-id lensemble-phase3-consortium --run-id phase3-consortium-v1 \
  --push --out-repo abdelstark/lensemble-phase3-consortium-checkpoint
```

Add `--dry-run` to the same command for the validation-only preflight. DP is **on by default** for the
real run; pass `--no-privacy` only for a non-private control. The launcher writes
`phase3_long_run_smoke_report.json`, `phase3_consortium_manifest.json`,
`phase3_dataset_probe_registry.json`, the pinned probe, the coordinator artifacts/ledger, and the run
manifest into `--out-dir`; with `--push` and `HF_TOKEN` it uploads that directory to `--out-repo`. The
per-round JEPA metrics are representation metrics only — downstream planner/task-success eval is deferred
to [#245](https://github.com/AbdelStark/Lensemble/issues/245). The evidence is real federated
consortium-engineering + training evidence, **not** a cryptographic honest-computation proof or a
paper-scale robotics performance result.

## Single-Site Run

The dataset is mounted read-only at `/data`; the script is a PEP-723 uv script with inline deps:

```bash
hf jobs uv run --flavor h200 --secrets HF_TOKEN \
  -v hf://datasets/abdelstark/so100-pickplace-lewm-ready:/data \
  https://raw.githubusercontent.com/AbdelStark/Lensemble/main/deploy/hfjobs/train_lewm.py -- \
  --data-source lerobot-h5:///data/svla_so100_pickplace.h5 \
  --steps 6000 --image-size 224 --latent-dim 384 --depth 8 \
  --lambda-anc 0.01 \
  --out-repo abdelstark/lewm-so100-jepa
```

`hf jobs hardware` lists flavors (`t4-small` … `h200x8`). On an H200, 6000 steps of a ~30M-param model
on the SO-100 set runs in ~5 min. Omit `--out-repo` to skip the checkpoint push.

## Reading the output

Each eval prints `val_pred` (held-out next-latent prediction), `val_sigreg`, and **`eff_rank`** — the
effective rank of the embedding covariance (`exp(entropy of eigenvalues)`). `eff_rank ≈ 1–3 / d` means
the representation has **collapsed** (prediction is trivially low); a healthy run keeps a large fraction
of `d`. Multi-round federated reports also include `round_metrics`: one record
per attempted round with the post-round global hash, participant ids, dataset
roots, and update L2 norms, so Phase 2 runs can generate curve-ready
round/update-norm tables without exposing raw data.

## Anti-collapse (#184) — use `--lambda-anc`

The bare SIGReg-JEPA objective is only a *gentle* anti-collapse on small datasets. The design's intended
mechanism is the **frame anchor** (`--lambda-anc > 0`): it pins `f_θ` on `latent_dim` generic landmarks
to the round-0 `f_ref` snapshot, holding the representation on the (high-rank) reference frame instead of
letting it collapse.

Empirical comparison on SO-100 (eff_rank / 384; the random-init reference is ≈ 13):

| objective | final `eff_rank` | |
|---|---|---|
| bare SIGReg-JEPA (`--lambda-anc 0`) | **3** | collapsed |
| isotropy-fixed SIGReg only (#185) | ~8 | partial |
| **frame anchor `--lambda-anc 0.01`** | **~12.5** | **holds the reference frame — no collapse** |
| direct decorrelation (covariance) term | ~20 | strongest, but a non-LeJEPA term |

`--lambda-anc 0.01` is the recommended default (`0.001` is too weak; `≥ 0.1` saturates at the reference
rank). The anchor adds `latent_dim` landmark forwards per step, so it is ~2–3× slower than bare. `val_pred`
stays low on SO-100 regardless because consecutive robot frames barely move (an easy target) — read
`eff_rank`, not `val_pred`, for collapse.
