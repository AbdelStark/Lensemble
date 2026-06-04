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
  --out-repo abdelstark/lewm-so100-jepa
```

`hf jobs hardware` lists flavors (`t4-small` … `h200x8`). On an H200, 6000 steps of a ~30M-param model
on the SO-100 set runs in ~5 min. Omit `--out-repo` to skip the checkpoint push.

## Reading the output

Each eval prints `val_pred` (held-out next-latent prediction), `val_sigreg`, and **`eff_rank`** — the
effective rank of the embedding covariance (`exp(entropy of eigenvalues)`). `eff_rank ≈ 1–3 / d` means
the representation has **collapsed** (prediction is trivially low); a healthy run keeps a large fraction
of `d`.

## Anti-collapse (#184)

The bare SIGReg-JEPA objective is a *gentle* anti-collapse on small datasets. For robust full-rank
representations, lean on the **frame anchor** (`lambda_anc > 0`, the design's gauge/anchor mechanism) via
the federated path, or raise `--lambda-sig`. The empirical comparison (SO-100, 6k steps): bare SIGReg
collapsed to `eff_rank ≈ 3/384`; the isotropy-fixed SIGReg (#185) reached `~8` and kept `val_pred`
non-trivial; a direct decorrelation term reached `~20`.
