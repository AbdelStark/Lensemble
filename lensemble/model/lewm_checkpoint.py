"""lensemble.model.lewm_checkpoint — pinned TwoRooms checkpoint ingestion and manifest (gate G1).

Resolves the real ``quentinll/lewm-tworooms`` LeWorldModel artifact (``config.json`` + ``weights.pt``)
at a **pinned revision**, builds the in-tree reconstruction
(:mod:`lensemble.model.lewm_tworooms`), loads the weights strictly, and emits a
``lewm-checkpoint-manifest/1`` binding revision, file hashes, architecture fields, the full tensor
inventory, and license/source references (docs/roadmap/TAPESTRY_LEWM.md, gate G1).

Claim-grade rule: a manifest used for evidence must come from a pinned 40-hex revision — ``main`` or a
tag can move, which would unbind every downstream export/evidence hash. ``resolve_checkpoint`` fails
closed on unpinned revisions unless ``claim_grade=False`` is requested explicitly (dev only).

All checkpoint handling here is server-side/build-time. Participant browsers never receive
``weights.pt``; they receive the hash-bound exported inference graphs of #317.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lensemble.errors import (
    ArtifactError,
    ConfigError,
    LensembleErrorCode,
)
from lensemble.model.lewm_tworooms import (
    LeWMTwoRooms,
    LeWMTwoRoomsConfig,
    build_lewm_tworooms,
    load_lewm_state_dict,
)

__all__ = [
    "TWOROOMS_REPO_ID",
    "TWOROOMS_PINNED_REVISION",
    "MANIFEST_SCHEMA",
    "ResolvedCheckpoint",
    "resolve_checkpoint",
    "load_tworooms_model",
    "checkpoint_manifest",
    "reference_forward_report",
]

TWOROOMS_REPO_ID = "quentinll/lewm-tworooms"
# The pinned upstream revision (HF commit SHA) recorded in docs/roadmap/TAPESTRY_LEWM.md.
TWOROOMS_PINNED_REVISION = "77adaae0bc31deab21c93740d1f8bb947cd0bdec"
MANIFEST_SCHEMA = "lewm-checkpoint-manifest/1"

_CONFIG_FILE = "config.json"
_WEIGHTS_FILE = "weights.pt"


@dataclass(frozen=True)
class ResolvedCheckpoint:
    """A local, revision-bound view of the checkpoint files."""

    repo_id: str
    revision: str
    config_path: Path
    weights_path: Path
    claim_grade: bool


def _is_pinned(revision: str) -> bool:
    return len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)


def resolve_checkpoint(
    *,
    local_dir: str | Path | None = None,
    revision: str = TWOROOMS_PINNED_REVISION,
    repo_id: str = TWOROOMS_REPO_ID,
    claim_grade: bool = True,
) -> ResolvedCheckpoint:
    """Resolve ``config.json`` + ``weights.pt`` from a local snapshot dir or the HF Hub.

    ``local_dir`` short-circuits the network entirely (CI and air-gapped runs); otherwise
    ``huggingface_hub`` (an optional dependency of this path, not of ``lensemble``) downloads the
    pinned revision into its cache. In claim-grade mode an unpinned revision is a ``ConfigError``.
    """
    if claim_grade and not _is_pinned(revision):
        raise ConfigError(
            f"claim-grade checkpoint resolution requires a pinned 40-hex revision, got {revision!r}",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="pin the HF commit SHA (docs/roadmap/TAPESTRY_LEWM.md) or pass "
            "claim_grade=False for throwaway dev runs",
        )

    if local_dir is not None:
        base = Path(local_dir)
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ArtifactError(
                "huggingface_hub is required to download the TwoRooms checkpoint",
                code=LensembleErrorCode.ARTIFACT_INVALID,
                remediation="uv pip install huggingface_hub, or pass local_dir= to a "
                "pre-downloaded snapshot",
            ) from exc
        base = Path(
            snapshot_download(repo_id, revision=revision, allow_patterns=["*.json", "*.pt"])
        )

    config_path = base / _CONFIG_FILE
    weights_path = base / _WEIGHTS_FILE
    missing = [p.name for p in (config_path, weights_path) if not p.is_file()]
    if missing:
        raise ArtifactError(
            f"checkpoint snapshot at {base} is missing {missing}",
            code=LensembleErrorCode.ARTIFACT_INVALID,
            remediation="re-download the pinned revision; both config.json and weights.pt "
            "are required",
        )
    return ResolvedCheckpoint(
        repo_id=repo_id,
        revision=revision,
        config_path=config_path,
        weights_path=weights_path,
        claim_grade=claim_grade,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tworooms_model(resolved: ResolvedCheckpoint) -> tuple[LeWMTwoRooms, dict[str, Any]]:
    """Build the reconstruction from the resolved files and strictly load the weights.

    Returns ``(model_in_eval_mode, upstream_config_dict)``. Every failure mode is fail-closed:
    invalid config (``ConfigError``), unreadable/non-state-dict weights (``ArtifactError``), and
    unknown/missing/mismatched tensors (``CheckpointIntegrityError`` from the strict loader).
    """
    try:
        config = json.loads(resolved.config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            f"unreadable checkpoint config at {resolved.config_path}: {exc}",
            code=LensembleErrorCode.ARTIFACT_INVALID,
            remediation="re-download the pinned revision",
        ) from exc

    model = build_lewm_tworooms(config)

    try:
        state_dict = torch.load(
            resolved.weights_path, map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise ArtifactError(
            f"unreadable weights at {resolved.weights_path}: {exc}",
            code=LensembleErrorCode.ARTIFACT_INVALID,
            remediation="re-download the pinned revision; weights.pt must be a plain "
            "tensor state dict",
        ) from exc
    if not isinstance(state_dict, dict) or not all(
        isinstance(v, torch.Tensor) for v in state_dict.values()
    ):
        raise ArtifactError(
            "weights.pt did not deserialize to a tensor state dict",
            code=LensembleErrorCode.ARTIFACT_INVALID,
            remediation="the released artifact is a torch state dict; refuse pickled modules",
        )

    load_lewm_state_dict(model, state_dict)
    return model, config


def checkpoint_manifest(resolved: ResolvedCheckpoint, model: LeWMTwoRooms) -> dict[str, Any]:
    """The ``lewm-checkpoint-manifest/1`` evidence artifact for gate G1."""
    cfg: LeWMTwoRoomsConfig = model.cfg
    state = model.state_dict()
    tensors = [
        {"name": name, "shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch.")}
        for name, t in state.items()
    ]
    trainable = sum(p.numel() for p in model.parameters())
    return {
        "schema": MANIFEST_SCHEMA,
        "source": {
            "repoId": resolved.repo_id,
            "revision": resolved.revision,
            "revisionPinned": _is_pinned(resolved.revision),
            "claimGrade": resolved.claim_grade,
            "license": "mit",
            "upstreamModel": "https://github.com/lucas-maes/le-wm",
            "upstreamModules": "https://github.com/galilai-group/stable-worldmodel "
            "(stable_worldmodel.wm.lewm; reconstructed in-tree, not vendored)",
            "hubUrl": f"https://huggingface.co/{resolved.repo_id}",
        },
        "files": {
            "config.json": {"sha256": _sha256_file(resolved.config_path)},
            "weights.pt": {
                "sha256": _sha256_file(resolved.weights_path),
                "bytes": resolved.weights_path.stat().st_size,
            },
        },
        "architecture": {
            "family": "LeWorldModel (JEPA-style latent world model)",
            "encoder": {
                "kind": "hf-vit",
                "size": cfg.vit_size,
                "patchSize": cfg.patch_size,
                "imageSize": cfg.image_size,
                "hiddenDim": cfg.hidden_dim,
                "layers": cfg.vit_layers,
                "heads": cfg.vit_heads,
                "numPatches": cfg.num_patches,
            },
            "predictor": {
                "kind": "adaln-causal-transformer",
                "numFrames": cfg.pred_num_frames,
                "depth": cfg.pred_depth,
                "heads": cfg.pred_heads,
                "dimHead": cfg.pred_dim_head,
                "mlpDim": cfg.pred_mlp_dim,
                "dropout": cfg.pred_dropout,
            },
            "actionEncoder": {
                "kind": "conv1d-silu-mlp",
                "inputDim": cfg.action_input_dim,
                "embDim": cfg.action_emb_dim,
            },
            "projectionHeads": {
                "kind": "linear-batchnorm-gelu-linear",
                "hiddenDim": cfg.proj_hidden_dim,
            },
        },
        "tensors": tensors,
        "tensorCount": len(tensors),
        "parameterCount": trainable,
        "nonClaims": [
            "This manifest binds the real LeWorldModel TwoRooms checkpoint used by the "
            "Tapestry-like browser-federation demo. It is not evidence of from-scratch "
            "browser LeWM training, production browser training, or paper-scale benchmark "
            "parity.",
        ],
    }


def reference_forward_report(
    model: LeWMTwoRooms, *, seed: int = 20260612, rollout_steps: int = 2
) -> dict[str, Any]:
    """Deterministic CPU reference forwards over fixed fixtures (the parity targets of G1/G2).

    Fixed-seed fixtures exercise the three exported paths — frame encoding (encoder+projector),
    action encoding, and teacher-forced prediction (predictor+pred_proj) — plus an autoregressive
    rollout. Summaries carry rounded tensor statistics and a fingerprint over outputs rounded to
    1e-4; exact-equality parity across platforms is asserted by tests with explicit tolerances, the
    fingerprint is the cheap same-machine regression signal.
    """
    cfg = model.cfg
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    pixels = torch.rand(
        (1, cfg.pred_num_frames, 3, cfg.image_size, cfg.image_size), generator=gen
    )
    actions = (
        torch.rand((1, cfg.pred_num_frames + rollout_steps, cfg.action_input_dim), generator=gen)
        * 2
        - 1
    )

    with torch.no_grad():
        emb = model.encode_frames(pixels)
        act_emb = model.encode_actions(actions)
        preds = model.predict(emb, act_emb[:, : cfg.pred_num_frames])
        rollout = model.rollout(emb, act_emb)

    def _summary(name: str, t: torch.Tensor) -> dict[str, Any]:
        flat = t.reshape(-1)
        return {
            "name": name,
            "shape": list(t.shape),
            "mean": round(flat.mean().item(), 6),
            "std": round(flat.std().item(), 6),
            "l2Norm": round(flat.norm().item(), 4),
            "first8": [round(v, 5) for v in flat[:8].tolist()],
        }

    outputs = [
        _summary("frameEmbedding", emb),
        _summary("actionEmbedding", act_emb),
        _summary("teacherForcedPrediction", preds),
        _summary("autoregressiveRollout", rollout),
    ]
    fingerprint = hashlib.sha256(
        json.dumps(
            [[round(v, 4) for v in torch.cat([o.reshape(-1)[:64] for o in (emb, act_emb, preds, rollout)]).tolist()]],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if was_training:
        model.train()
    return {
        "seed": seed,
        "rolloutSteps": rollout_steps,
        "fixture": {
            "pixels": {"shape": list(pixels.shape), "distribution": "uniform[0,1)"},
            "actions": {"shape": list(actions.shape), "distribution": "uniform[-1,1)"},
        },
        "outputs": outputs,
        "fingerprintSha256": fingerprint,
    }
