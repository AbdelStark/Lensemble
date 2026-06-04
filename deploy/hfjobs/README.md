# HF Jobs — end-to-end LeWorldModel training

[`train_lewm.py`](train_lewm.py) trains the in-tree LeWorldModel (video-ViT `f_θ` + action-conditioned
`g_φ` + the SIGReg-JEPA objective) **from scratch (no warm start)** on a real robot dataset, on a
Hugging Face Jobs GPU. The dataset is loaded through the official `lerobot-h5://` data source
(`lensemble.data.adapters`) — the same `EpisodeDataset` → `Window` contract the federated `Participant`
consumes — so this is the real pipeline, not a bespoke loader.

## Run

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
