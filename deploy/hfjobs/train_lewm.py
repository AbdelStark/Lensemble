# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "numpy",
#   "h5py",
#   "safetensors",
#   "huggingface-hub",
#   "lensemble",
# ]
# ///
"""End-to-end LeWorldModel training on HF Jobs — the official path, on a real robot dataset.

A reproducible HF Jobs entry that trains the in-tree LeWorldModel (video-ViT f_theta +
action-conditioned g_phi + the SIGReg-JEPA objective) from scratch (no warm start) on a LeRobot-layout
HDF5 dataset resolved through the official ``lerobot-h5://`` data source (``lensemble.data.adapters``).
Held-out episode split, cosine LR schedule, and an effective-rank collapse diagnostic; the trained
encoder+predictor checkpoint is optionally pushed to the Hub.

Launch (the dataset is mounted read-only at /data):

    hf jobs uv run --flavor h200 --secrets HF_TOKEN \\
      -v hf://datasets/<org>/<dataset>:/data \\
      https://raw.githubusercontent.com/AbdelStark/Lensemble/main/deploy/hfjobs/train_lewm.py -- \\
      --data-source lerobot-h5:///data/<file>.h5 --steps 6000 --out-repo <org>/<model>

Anti-collapse note (#184): the bare SIGReg-JEPA objective is a *gentle* anti-collapse on small
datasets — for robust full-rank representations lean on the frame anchor (``lambda_anc>0``, the design's
gauge/anchor mechanism) or raise ``--lambda-sig`` with more SIGReg projections. The eff_rank diagnostic
printed each eval makes collapse visible (≈1-3/d is collapsed; healthy is a large fraction of d).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

from lensemble.contracts import WMCP_VERSION, LatentState
from lensemble.data.adapters import load_episodes
from lensemble.data.episode import Window
from lensemble.model import (
    build_action_head,
    build_encoder,
    build_predictor,
    build_sketch,
    sigreg_statistic,
)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end LeWorldModel training on HF Jobs."
    )
    p.add_argument(
        "--data-source",
        required=True,
        help="lerobot-h5:///data/<file>.h5 (or any load_episodes source)",
    )
    p.add_argument(
        "--num-steps",
        type=int,
        default=8,
        help="window horizon (obs = num_steps+1 frames)",
    )
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--latent-dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--num-heads", type=int, default=6)
    p.add_argument("--batch", type=int, default=16, help="windows per step")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lambda-sig", type=float, default=0.5)
    p.add_argument(
        "--out-repo",
        default=None,
        help="optional HF model repo to push the checkpoint to",
    )
    return p.parse_args()


def _effective_rank(emb: torch.Tensor) -> float:
    x = emb.reshape(-1, emb.shape[-1]).float()
    x = x - x.mean(0, keepdim=True)
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov).clamp_min(1e-12)
    p = ev / ev.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def main() -> None:
    a = _args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"torch {torch.__version__} | device {dev} | data {a.data_source}", flush=True
    )

    # Official data path: real robot dataset -> EpisodeDataset -> Windows (no bespoke loader).
    t0 = time.time()
    ds = load_episodes(a.data_source)
    spec = ds.episodes[0].action_spec
    windows = list(ds.windows(a.num_steps))
    n_val = max(1, len(windows) // 10)
    val_w, train_w = windows[:n_val], windows[n_val:]
    print(
        f"loaded {len(ds)} episodes, {len(windows)} windows ({len(train_w)} train / {len(val_w)} val) "
        f"in {time.time() - t0:.1f}s | action dim {spec.dim}",
        flush=True,
    )

    tokens_n = (a.image_size // a.patch_size) ** 2
    model = SimpleNamespace(
        latent_dim=a.latent_dim,
        num_tokens=tokens_n,
        in_channels=3,
        num_frames=1,
        image_size=a.image_size,
        patch_size=a.patch_size,
        tubelet=1,
        depth=a.depth,
        num_heads=a.num_heads,
        cond_dim=a.latent_dim,
        predictor_width=a.latent_dim,
        predictor_depth=a.depth,
        mlp_ratio=4.0,
        wmcp_version=WMCP_VERSION,
    )
    cfg = SimpleNamespace(model=model)
    enc = build_encoder(cfg).to(dev)
    pred = build_predictor(cfg).to(dev)
    head = build_action_head(cfg, spec).to(dev)
    params = [*enc.parameters(), *pred.parameters(), *head.parameters()]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.05)
    print(
        f"params(enc+pred) {sum(p.numel() for p in (*enc.parameters(), *pred.parameters())) / 1e6:.1f}M",
        flush=True,
    )

    def lr_at(step: int) -> float:
        warm = min(300, a.steps // 10)
        if step < warm:
            return a.lr * step / max(1, warm)
        prog = (step - warm) / max(1, a.steps - warm)
        return 0.5 * a.lr * (1 + math.cos(math.pi * prog))

    def batch(
        pool: list[Window], rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        picks = [pool[int(i)] for i in rng.integers(0, len(pool), a.batch)]
        obs = torch.stack([w.obs for w in picks]).to(dev)  # (B, span, 1, 3, H, W)
        acts = torch.stack([w.actions for w in picks]).to(dev)  # (B, num_steps, dim)
        return obs, acts

    def step_loss(
        obs: torch.Tensor, acts: torch.Tensor, sk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, span = obs.shape[0], obs.shape[1]
        tok = enc(obs.reshape(b * span, *obs.shape[2:])).tokens
        d = tok.shape[-1]
        tok = tok.reshape(b, span, tokens_n, d)
        inp = LatentState(
            tok[:, :-1].reshape(b * a.num_steps, tokens_n, d), tokens_n, d, WMCP_VERSION
        )
        tgt = LatentState(
            tok[:, 1:].reshape(b * a.num_steps, tokens_n, d), tokens_n, d, WMCP_VERSION
        )
        aemb = head.encode(acts.reshape(b * a.num_steps, spec.dim))
        predv = pred.prediction_residual(inp, aemb, tgt).pow(2).mean().float()
        sig = sigreg_statistic(tok.reshape(-1, d), sk).float()
        return predv, sig, tok

    rng, vrng = np.random.default_rng(0), np.random.default_rng(7)
    hist, t0 = [], time.time()
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        sk = build_sketch(1000 + step, a.latent_dim, 128).to(dev)
        predv, sig, _ = step_loss(*batch(train_w, rng), sk)
        total = predv + a.lambda_sig * sig
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 500 == 0 or step == a.steps - 1:
            with torch.no_grad():
                enc.eval()
                pred.eval()
                vp, vs, embs = [], [], []
                for _ in range(6):
                    pv, sv, tk = step_loss(*batch(val_w, vrng), sk)
                    vp.append(pv.item())
                    vs.append(sv.item())
                    embs.append(tk.reshape(-1, tk.shape[-1]).cpu())
                enc.train()
                pred.train()
                er = _effective_rank(torch.cat(embs)[:6000])
            hist.append(
                {
                    "step": step,
                    "val_pred": float(np.mean(vp)),
                    "val_sigreg": float(np.mean(vs)),
                    "eff_rank": er,
                }
            )
            print(
                f"step {step:5d} | val_pred {np.mean(vp):.4f} | val_sigreg {np.mean(vs):.4f} | eff_rank {er:.1f}/{a.latent_dim}",
                flush=True,
            )
    print(
        f"=== trained {a.steps} steps in {(time.time() - t0) / 60:.1f} min on {dev.type} ===",
        flush=True,
    )

    if a.out_repo:
        from safetensors.torch import save_file as st_save

        out = "/tmp/lewm_ckpt"
        os.makedirs(out, exist_ok=True)
        st_save(
            {f"encoder.{k}": v.detach().cpu() for k, v in enc.state_dict().items()}
            | {
                f"predictor.{k}": v.detach().cpu() for k, v in pred.state_dict().items()
            },
            f"{out}/model.safetensors",
        )
        with open(f"{out}/training.json", "w") as fh:
            json.dump(
                {
                    "model": vars(model),
                    "steps": a.steps,
                    "warm_start": False,
                    "history": hist,
                },
                fh,
                indent=2,
            )
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(a.out_repo, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(folder_path=out, repo_id=a.out_repo, repo_type="model")
        print(
            f"=== pushed checkpoint to https://huggingface.co/{a.out_repo} (private) ===",
            flush=True,
        )


if __name__ == "__main__":
    main()
