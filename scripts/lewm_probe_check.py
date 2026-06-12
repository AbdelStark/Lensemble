#!/usr/bin/env python3
"""Gate G5: federated before/after probe on REAL latents with the shipping JS federation math.

Builds disjoint per-participant training pairs and a held-out validation set through the exported
checkpoint graphs (official expert episodes), then drives the exact shipping federation shape
under node (web/federated-demo/lewm_probe_check.mjs): shared deterministic adapter init, per-round
local training from (init + global offset), clipped deltas, deterministic mean, offset
accumulation, and a final identity-vs-adapted comparison on the held-out pairs.

Writes ``docs/evidence/lewm_tworooms_probe_check.json``. A non-improving verdict fails the gate
and blocks public positive claims.

  uv run --with onnxruntime --with hdf5plugin python scripts/lewm_probe_check.py \
    --h5 ~/.cache/lensemble-lewm/tworoom.h5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument(
        "--model-dir", type=Path, default=Path("web/federated-demo/model/lewm-tworooms")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("docs/evidence/lewm_tworooms_probe_check.json")
    )
    parser.add_argument("--participants", type=int, default=2)
    # defaults validated by the #322 sweep: enough resident pairs that the adapter learns the
    # systematic predictor bias (which generalizes) instead of memorizing episodes
    parser.add_argument("--episodes-per-participant", type=int, default=8)
    parser.add_argument("--validation-episodes", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps-per-round", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260612)
    return parser.parse_args()


def _pairs_from_episodes(
    sessions: dict[str, Any], f: Any, episodes: list[int], offsets: np.ndarray, lengths: np.ndarray
) -> dict[str, Any]:
    xs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for ep in episodes:
        start = int(offsets[ep])
        n_model_steps = min(7, (int(lengths[ep]) - 1) // 5)
        idx = [start + 5 * i for i in range(n_model_steps + 1)]
        frames = (sessions["pixels"][idx].astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
        latents = sessions["enc"].run(None, {"pixels": frames})[0]
        raw_actions = sessions["actions"][start : start + 5 * n_model_steps].astype(np.float32)
        blocks = raw_actions.reshape(1, n_model_steps, 10)
        act_emb = sessions["act"].run(None, {"actions": blocks})[0][0]
        for t in range(2, n_model_steps):
            hist_z = latents[t - 2 : t + 1][None]
            hist_a = act_emb[t - 2 : t + 1][None]
            preds = sessions["pred"].run(
                None, {"latents": hist_z, "action_embeddings": hist_a}
            )[0]
            xs.append(preds[0, -1])
            targets.append(latents[t + 1])
    return {
        "count": len(xs),
        "x": np.concatenate(xs).astype(float).round(6).tolist(),
        "target": np.concatenate(targets).astype(float).round(6).tolist(),
    }


def main() -> None:
    import h5py
    import hdf5plugin  # noqa: F401
    import onnxruntime as ort

    args = _args()
    providers = ["CPUExecutionProvider"]
    manifest = json.loads((args.model_dir / "manifest.json").read_text())
    rng = np.random.default_rng(args.seed)

    with h5py.File(args.h5, "r") as f:
        offsets = np.asarray(cast(h5py.Dataset, f["ep_offset"])[:])
        lengths = np.asarray(cast(h5py.Dataset, f["ep_len"])[:])
        candidates = np.flatnonzero(lengths >= 41)
        total_needed = args.participants * args.episodes_per_participant + args.validation_episodes
        chosen = rng.choice(candidates, size=total_needed, replace=False)
        sessions = {
            "enc": ort.InferenceSession(args.model_dir / "lewm_tworooms_encoder.onnx", providers=providers),
            "act": ort.InferenceSession(args.model_dir / "lewm_tworooms_action.onnx", providers=providers),
            "pred": ort.InferenceSession(args.model_dir / "lewm_tworooms_predictor.onnx", providers=providers),
            "pixels": cast(h5py.Dataset, f["pixels"]),
            "actions": cast(h5py.Dataset, f["action"]),
        }
        participants = []
        cursor = 0
        for _ in range(args.participants):
            eps = sorted(int(e) for e in chosen[cursor : cursor + args.episodes_per_participant])
            cursor += args.episodes_per_participant
            participants.append(_pairs_from_episodes(sessions, f, eps, offsets, lengths))
        val_eps = sorted(int(e) for e in chosen[cursor:])
        validation = _pairs_from_episodes(sessions, f, val_eps, offsets, lengths)

    fixture = {
        "dim": 192,
        "adapterHidden": 32,
        "adapterInitSeed": 42,
        "rounds": args.rounds,
        "stepsPerRound": args.steps_per_round,
        "batchSize": args.batch_size,
        "clipNorm": 3.0,
        "participants": participants,
        "validation": validation,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(fixture, tmp)
        tmp_path = tmp.name
    result = subprocess.run(
        ["node", "web/federated-demo/lewm_probe_check.mjs", tmp_path],
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise SystemExit(f"probe driver failed: {result.stdout}{result.stderr}")
    report = json.loads(result.stdout.strip().splitlines()[-1])
    evidence = {
        "schema": "lewm-federated-probe/1",
        "seed": args.seed,
        "protocol": "disjoint per-participant expert episodes -> shipping JS federation "
        "(shared init + offset, clipped deltas, deterministic mean) -> held-out validation "
        "pairs; before = identity adapter (frozen predictor), after = final global revision",
        "checkpoint": manifest["checkpoint"],
        "trainPairsPerParticipant": [p["count"] for p in participants],
        "result": report,
        "passes": report["verdict"] == "improved",
        "nonClaims": [
            "Before/after validation probe for the Tapestry-like demo's federated adapter path; "
            "not paper-scale TwoRooms benchmark parity and not evidence of production browser "
            "training.",
        ],
    }
    args.out.write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "baselineMse": report["baselineMse"],
                "adaptedMse": report["adaptedMse"],
                "relativeImprovement": report["relativeImprovement"],
                "passes": evidence["passes"],
            },
            indent=2,
        )
    )
    if not evidence["passes"]:
        raise SystemExit(
            "federated probe verdict is not 'improved' — the negative result is recorded in "
            "evidence and blocks public positive claims"
        )


if __name__ == "__main__":
    main()
