"""Shared real-latent probe pairs for the TwoRooms LeWM federated demo (epic #314, #327).

Builds disjoint per-participant training pairs and a held-out validation set by running official
expert episodes through the *exported* checkpoint graphs (the same ONNX encoder/action/predictor
the browser ships). One model step = 5 raw env steps; for each teacher-forced window we record the
frozen predictor's next-latent prediction (``x``) and the frozen encoder's latent of the actually
observed next frame (``target``). The systematic residual ``target - x`` is the bias a bounded
adapter can learn to correct — and the only thing that generalises to held-out episodes.

Both the offline math cross-check (``scripts/lewm_probe_check.py``) and the system-composed gate
(``scripts/lewm_system_probe.py``) build their fixtures here so the two artifacts are scored on the
*same* real-latent protocol and differ only in the aggregation path under test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

ADAPTER_DIM = 192
DEFAULT_MODEL_DIR = Path("web/federated-demo/model/lewm-tworooms")


@dataclass(frozen=True)
class ProbeSplit:
    """Disjoint per-participant training pairs + a shared held-out validation set."""

    participants: list[dict[str, Any]]
    validation: dict[str, Any]
    checkpoint: dict[str, Any]
    dim: int = ADAPTER_DIM


def _pairs_from_episodes(
    sessions: dict[str, Any],
    episodes: list[int],
    offsets: np.ndarray,
    lengths: np.ndarray,
) -> dict[str, Any]:
    xs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for ep in episodes:
        start = int(offsets[ep])
        n_model_steps = min(7, (int(lengths[ep]) - 1) // 5)
        idx = [start + 5 * i for i in range(n_model_steps + 1)]
        frames = (sessions["pixels"][idx].astype(np.float32) / 255.0).transpose(
            0, 3, 1, 2
        )
        latents = sessions["enc"].run(None, {"pixels": frames})[0]
        raw_actions = sessions["actions"][start : start + 5 * n_model_steps].astype(
            np.float32
        )
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


def build_probe_split(
    *,
    h5_path: Path,
    model_dir: Path = DEFAULT_MODEL_DIR,
    seed: int,
    participants: int = 2,
    episodes_per_participant: int = 8,
    validation_episodes: int = 4,
) -> ProbeSplit:
    """Build a seeded disjoint train/validation split through the exported graphs.

    Lazily imports onnxruntime/h5py/hdf5plugin so importing this module stays cheap (the unit
    suite imports it without the heavyweight runtimes installed).
    """
    import h5py
    import hdf5plugin  # noqa: F401  # type: ignore[reportMissingImports]
    import onnxruntime as ort  # type: ignore[reportMissingImports]

    providers = ["CPUExecutionProvider"]
    manifest = json.loads((model_dir / "manifest.json").read_text())
    rng = np.random.default_rng(seed)

    with h5py.File(h5_path, "r") as f:
        offsets = np.asarray(cast("h5py.Dataset", f["ep_offset"])[:])
        lengths = np.asarray(cast("h5py.Dataset", f["ep_len"])[:])
        candidates = np.flatnonzero(lengths >= 41)
        total_needed = participants * episodes_per_participant + validation_episodes
        chosen = rng.choice(candidates, size=total_needed, replace=False)
        sessions = {
            "enc": ort.InferenceSession(
                model_dir / "lewm_tworooms_encoder.onnx", providers=providers
            ),
            "act": ort.InferenceSession(
                model_dir / "lewm_tworooms_action.onnx", providers=providers
            ),
            "pred": ort.InferenceSession(
                model_dir / "lewm_tworooms_predictor.onnx", providers=providers
            ),
            "pixels": cast("h5py.Dataset", f["pixels"]),
            "actions": cast("h5py.Dataset", f["action"]),
        }
        participant_pairs: list[dict[str, Any]] = []
        cursor = 0
        for _ in range(participants):
            eps = sorted(
                int(e) for e in chosen[cursor : cursor + episodes_per_participant]
            )
            cursor += episodes_per_participant
            participant_pairs.append(
                _pairs_from_episodes(sessions, eps, offsets, lengths)
            )
        val_eps = sorted(int(e) for e in chosen[cursor:])
        validation = _pairs_from_episodes(sessions, val_eps, offsets, lengths)

    return ProbeSplit(
        participants=participant_pairs,
        validation=validation,
        checkpoint=manifest["checkpoint"],
        dim=ADAPTER_DIM,
    )
