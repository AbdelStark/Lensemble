#!/usr/bin/env python3
"""Validate the exported browser graphs end-to-end on real TwoRooms expert trajectories.

The strongest gate-G2 sanity check there is: encode real dataset frames with the exported
encoder graph, embed the real raw action blocks with the exported action graph (z-score baked
in), run the exported predictor teacher-forced, and compare the predicted next latent to the
encoder's latent of the actual next frame. The pipeline is correct only if the model beats the
copy-last-latent baseline by a wide margin — any normalization, frameskip, or windowing mistake
destroys that margin immediately.

Writes ``docs/evidence/lewm_tworooms_realdata_check.json``. Requires the exported graphs
(scripts/lewm_tworooms_export.py), the extracted dataset ``tworoom.h5``, onnxruntime, and
hdf5plugin:

  uv run --with onnxruntime --with hdf5plugin python scripts/lewm_tworooms_realdata_check.py \
    --h5 ~/.cache/lensemble-lewm/tworoom.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument(
        "--model-dir", type=Path, default=Path("web/federated-demo/model/lewm-tworooms")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/evidence/lewm_tworooms_realdata_check.json"),
    )
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260612)
    return parser.parse_args()


def main() -> None:
    import h5py
    import hdf5plugin  # noqa: F401 - registers the pixels compression plugin
    import onnxruntime as ort

    args = _args()
    providers = ["CPUExecutionProvider"]
    enc = ort.InferenceSession(
        args.model_dir / "lewm_tworooms_encoder.onnx", providers=providers
    )
    act = ort.InferenceSession(
        args.model_dir / "lewm_tworooms_action.onnx", providers=providers
    )
    pred = ort.InferenceSession(
        args.model_dir / "lewm_tworooms_predictor.onnx", providers=providers
    )
    manifest = json.loads((args.model_dir / "manifest.json").read_text())

    rng = np.random.default_rng(args.seed)
    model_mses: list[float] = []
    baseline_mses: list[float] = []
    with h5py.File(args.h5, "r") as f:
        offsets = np.asarray(cast(h5py.Dataset, f["ep_offset"])[:])
        lengths = np.asarray(cast(h5py.Dataset, f["ep_len"])[:])
        pixels_ds = cast(h5py.Dataset, f["pixels"])
        actions_ds = cast(h5py.Dataset, f["action"])
        candidates = np.flatnonzero(lengths >= 21)
        chosen = rng.choice(
            candidates, size=min(args.episodes, len(candidates)), replace=False
        )
        for ep in sorted(int(e) for e in chosen):
            start = int(offsets[ep])
            idx = [start, start + 5, start + 10, start + 15]
            frames = (pixels_ds[idx].astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
            latents = enc.run(None, {"pixels": frames})[0]  # (4, D)
            actions = (
                actions_ds[start : start + 15].astype(np.float32).reshape(1, 3, 10)
            )
            act_emb = act.run(None, {"actions": actions})[0]
            preds = pred.run(
                None, {"latents": latents[:3][None], "action_embeddings": act_emb}
            )[0]
            model_mses.append(float(((preds[0, -1] - latents[3]) ** 2).mean()))
            baseline_mses.append(float(((latents[2] - latents[3]) ** 2).mean()))

    model_mse = float(np.mean(model_mses))
    baseline_mse = float(np.mean(baseline_mses))
    report = {
        "schema": "lewm-realdata-check/1",
        "seed": args.seed,
        "episodes": len(model_mses),
        "protocol": "teacher-forced next-latent prediction over [t, t+5, t+10] frames + 3 raw "
        "frameskip-5 action blocks; target = exported-encoder latent of frame t+15; baseline = "
        "copy-last latent",
        "checkpoint": manifest["checkpoint"],
        "graphHashes": {k: v["sha256"] for k, v in manifest["files"].items()},
        "modelPredictionMse": round(model_mse, 6),
        "copyLastBaselineMse": round(baseline_mse, 6),
        "modelOverBaselineRatio": round(model_mse / baseline_mse, 6),
        "passes": model_mse < 0.5 * baseline_mse,
        "nonClaims": [
            "Pipeline-correctness probe for the Tapestry-like demo's exported inference graphs "
            "on the official expert dataset; not a paper-scale benchmark reproduction.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "episodes",
                    "modelPredictionMse",
                    "copyLastBaselineMse",
                    "modelOverBaselineRatio",
                    "passes",
                )
            },
            indent=2,
        )
    )
    if not report["passes"]:
        raise SystemExit(
            "real-data check FAILED: the exported pipeline does not beat copy-last"
        )


if __name__ == "__main__":
    main()
