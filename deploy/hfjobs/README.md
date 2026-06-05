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

Start every expensive run with `--dry-run` and a pinned SHA. A representative GPU command is:

```bash
hf jobs uv run --flavor h200 --timeout 2h --secrets HF_TOKEN \
  --with 'lensemble @ git+https://github.com/AbdelStark/Lensemble.git@<SHA>' \
  -v hf://datasets/<org>/<phase2-silo-a>:/data/a \
  -v hf://datasets/<org>/<phase2-silo-b>:/data/b \
  -d https://raw.githubusercontent.com/AbdelStark/Lensemble/<SHA>/deploy/hfjobs/train_federated_lewm.py \
  --data-source lerobot-h5:///data/a/<silo-a>.h5 \
  --data-source lerobot-h5:///data/b/<silo-b>.h5 \
  --out-dir /tmp/lensemble-phase2 \
  --image-size 224 --patch-size 14 --latent-dim 192 \
  --depth 12 --predictor-depth 6 --num-heads 3 \
  --probe-points 1024 --inner-horizon 4 --window-steps 4 \
  --num-rounds 8 --metric-windows 256 \
  --push --out-repo <org>/lensemble-phase2-checkpoint
```

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
of `d`.

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
