#!/usr/bin/env python3
"""Gate G3: prove the browser adapter's loss decreases on REAL frozen-LeWM latents (#319).

Builds the exact training pairs the browser produces — frozen exported-graph teacher-forced
next-latent predictions vs the frozen encoder's latent of the actually-observed next frame —
from official expert-dataset trajectories, then runs the *shipping JS trainer* under node
(web/federated-demo/lewm_adapter_overfit.mjs) and records the loss curve as evidence.

Writes ``docs/evidence/lewm_tworooms_adapter_overfit.json``.

  uv run --with onnxruntime --with hdf5plugin python scripts/lewm_adapter_overfit_check.py \
    --h5 ~/.cache/lensemble-lewm/tworoom.h5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
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
        "--out", type=Path, default=Path("docs/evidence/lewm_tworooms_adapter_overfit.json")
    )
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260612)
    return parser.parse_args()


def main() -> None:
    import h5py
    import hdf5plugin  # noqa: F401
    import onnxruntime as ort

    args = _args()
    providers = ["CPUExecutionProvider"]
    enc = ort.InferenceSession(args.model_dir / "lewm_tworooms_encoder.onnx", providers=providers)
    act = ort.InferenceSession(args.model_dir / "lewm_tworooms_action.onnx", providers=providers)
    pred = ort.InferenceSession(
        args.model_dir / "lewm_tworooms_predictor.onnx", providers=providers
    )
    manifest = json.loads((args.model_dir / "manifest.json").read_text())

    rng = np.random.default_rng(args.seed)
    xs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with h5py.File(args.h5, "r") as f:
        offsets = np.asarray(cast(h5py.Dataset, f["ep_offset"])[:])
        lengths = np.asarray(cast(h5py.Dataset, f["ep_len"])[:])
        pixels_ds = cast(h5py.Dataset, f["pixels"])
        actions_ds = cast(h5py.Dataset, f["action"])
        candidates = np.flatnonzero(lengths >= 41)
        chosen = rng.choice(candidates, size=min(args.episodes, len(candidates)), replace=False)
        for ep in sorted(int(e) for e in chosen):
            start = int(offsets[ep])
            n_model_steps = min(7, (int(lengths[ep]) - 1) // 5)
            idx = [start + 5 * i for i in range(n_model_steps + 1)]
            frames = (pixels_ds[idx].astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
            latents = enc.run(None, {"pixels": frames})[0]  # (n+1, D)
            raw_actions = actions_ds[start : start + 5 * n_model_steps].astype(np.float32)
            blocks = raw_actions.reshape(1, n_model_steps, 10)
            act_emb = act.run(None, {"actions": blocks})[0][0]  # (n, D)
            for t in range(2, n_model_steps):
                hist_z = latents[t - 2 : t + 1][None]
                hist_a = act_emb[t - 2 : t + 1][None]
                preds = pred.run(None, {"latents": hist_z, "action_embeddings": hist_a})[0]
                xs.append(preds[0, -1])
                targets.append(latents[t + 1])

    dim = xs[0].shape[0]
    fixture = {
        "dim": dim,
        "count": len(xs),
        "x": np.concatenate(xs).astype(float).round(6).tolist(),
        "target": np.concatenate(targets).astype(float).round(6).tolist(),
        "steps": args.steps,
        "batchSize": 32,
        "seed": 1,
        "lambda": 0.1,
        "clipNorm": 3.0,
        "hiddenDim": 32,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(fixture, tmp)
        tmp_path = tmp.name
    result = subprocess.run(
        ["node", "web/federated-demo/lewm_adapter_overfit.mjs", tmp_path],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    evidence = {
        "schema": "lewm-adapter-overfit/1",
        "seed": args.seed,
        "protocol": "frozen exported-graph teacher-forced predictions vs frozen-encoder next "
        "latents over official expert episodes; trained with the shipping JS adapter "
        "(web/federated-demo/lewm_adapter.mjs) under node",
        "checkpoint": manifest["checkpoint"],
        "episodes": int(len(chosen)),
        "trainer": report,
        "passes": bool(report["lossDecreased"] and report["relativeImprovement"] > 0.2),
        "nonClaims": [
            "One-browser overfit gate for the bounded Tapestry-like adapter subset; "
            "checkpoint adaptation around a frozen base, not from-scratch browser LeWM "
            "pretraining and not a benchmark claim.",
        ],
    }
    args.out.write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        json.dumps(
            {
                "pairs": report["pairCount"],
                "firstPredLoss": report["firstPredLoss"],
                "lastPredLoss": report["lastPredLoss"],
                "relativeImprovement": report["relativeImprovement"],
                "sigreg": report["finalDiagnostics"]["sigregStatistic"],
                "effectiveRank": report["finalDiagnostics"]["effectiveRank"],
                "passes": evidence["passes"],
            },
            indent=2,
        )
    )
    if not evidence["passes"]:
        raise SystemExit("adapter overfit gate FAILED — do not proceed to federation")


if __name__ == "__main__":
    main()
