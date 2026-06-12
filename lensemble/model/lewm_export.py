"""lensemble.model.lewm_export — checkpoint-backed LeWM browser inference graphs (gate G2).

Exports the minimal TwoRooms LeWM inference surface to ONNX for ONNX Runtime Web (WASM/WebGPU):

- ``lewm_tworooms_encoder.onnx`` — pixels ``(B, 3, 224, 224)`` → projected CLS latent ``(B, 192)``
  (HF-ViT encoder + BatchNorm ``projector``, eval-mode running statistics);
- ``lewm_tworooms_action.onnx`` — action blocks ``(B, T, 10)`` → embeddings ``(B, T, 192)``;
- ``lewm_tworooms_predictor.onnx`` — latents ``(B, T, 192)`` × action embeddings ``(B, T, 192)`` →
  next-latent predictions ``(B, T, 192)`` (AdaLN predictor + ``pred_proj``), ``T ≤ num_frames``.

Artifacts are generated from the pinned checkpoint (#316) and hash-bound: the
``lewm-browser-export/1`` manifest records every graph's SHA-256, byte size, IO contract, opset,
torch version, and the parent checkpoint revision/weights hash, so evidence can cite exact browser
model revisions (docs/roadmap/TAPESTRY_LEWM.md gate G2). Parity against the PyTorch reference runs
through ``onnxruntime`` (optional dependency; the blocking CPU gates skip when absent).

The browser receives these exported graphs only — never ``weights.pt``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from lensemble.model.lewm_checkpoint import ResolvedCheckpoint
from lensemble.model.lewm_tworooms import LeWMTwoRooms

__all__ = [
    "EXPORT_SCHEMA",
    "EXPORT_GRAPH_VERSION",
    "EncoderGraph",
    "ActionGraph",
    "PredictorGraph",
    "export_browser_graphs",
    "browser_export_manifest",
    "onnxruntime_parity",
]

EXPORT_SCHEMA = "lewm-browser-export/1"
# Bump when the exported graph decomposition or IO contract changes.
EXPORT_GRAPH_VERSION = 1

_ENCODER_FILE = "lewm_tworooms_encoder.onnx"
_ACTION_FILE = "lewm_tworooms_action.onnx"
_PREDICTOR_FILE = "lewm_tworooms_predictor.onnx"


# Input normalizations are baked INTO the graphs so the browser feeds raw values and cannot
# drift from the training pipeline: pixels in [0,1] get the ImageNet statistics the upstream
# ToImage transform applied; raw env actions get the expert-dataset z-score
# (docs/evidence/lewm_tworooms_action_stats.json).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class EncoderGraph(nn.Module):
    """Frame in [0,1] → ImageNet-normalize → projected CLS latent (the browser's ``encode`` op)."""

    def __init__(self, model: LeWMTwoRooms) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.projector = model.projector
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, pixels: Tensor) -> Tensor:
        normalized = (pixels - self.pixel_mean) / self.pixel_std
        cls = self.encoder(normalized)[:, 0]
        return self.projector(cls)


class ActionGraph(nn.Module):
    """Raw action blocks → z-score → action embeddings (the browser's ``embed_actions`` op).

    ``action_stats``: per-2D-dimension ``(mean, std)`` fitted on the expert dataset; the stats are
    tiled across the frameskip-5 block (the upstream pipeline z-scores before chunking).
    """

    def __init__(
        self,
        model: LeWMTwoRooms,
        action_stats: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
    ) -> None:
        super().__init__()
        self.action_encoder = model.action_encoder
        block = model.cfg.action_input_dim
        if action_stats is None:
            mean = torch.zeros(block)
            std = torch.ones(block)
        else:
            raw_mean, raw_std = action_stats
            repeats = block // len(raw_mean)
            mean = torch.tensor(raw_mean).repeat(repeats)
            std = torch.tensor(raw_std).repeat(repeats)
        self.register_buffer("action_mean", mean.view(1, 1, block))
        self.register_buffer("action_std", std.view(1, 1, block))

    def forward(self, actions: Tensor) -> Tensor:
        normalized = (actions - self.action_mean) / self.action_std
        return self.action_encoder(normalized)


class PredictorGraph(nn.Module):
    """Latent history × action embeddings → next-latent predictions (the browser's ``predict`` op)."""

    def __init__(self, model: LeWMTwoRooms) -> None:
        super().__init__()
        self.predictor = model.predictor
        self.pred_proj = model.pred_proj

    def forward(self, emb: Tensor, act_emb: Tensor) -> Tensor:
        preds = self.predictor(emb, act_emb)
        b, t, d = preds.shape
        return self.pred_proj(preds.reshape(b * t, d)).reshape(b, t, d)


def _fixtures(model: LeWMTwoRooms, seed: int = 20260612) -> dict[str, tuple[Tensor, ...]]:
    cfg = model.cfg
    gen = torch.Generator().manual_seed(seed)
    pixels = torch.rand((2, 3, cfg.image_size, cfg.image_size), generator=gen)
    actions = torch.rand((2, cfg.pred_num_frames, cfg.action_input_dim), generator=gen) * 2 - 1
    emb = torch.randn((2, cfg.pred_num_frames, cfg.hidden_dim), generator=gen)
    act_emb = torch.randn((2, cfg.pred_num_frames, cfg.hidden_dim), generator=gen)
    return {
        _ENCODER_FILE: (pixels,),
        _ACTION_FILE: (actions,),
        _PREDICTOR_FILE: (emb, act_emb),
    }


_IO_SPECS: dict[str, dict[str, Any]] = {
    _ENCODER_FILE: {
        "graph": "encoder",
        "inputs": {"pixels": ["batch", 3, "image_size", "image_size"]},
        "inputUnits": "RGB floats in [0,1]; ImageNet normalization applied inside the graph",
        "outputs": {"latent": ["batch", "hidden_dim"]},
        "components": [
            "imagenet-normalize",
            "encoder (HF-ViT)",
            "projector (Linear-BN-GELU-Linear)",
        ],
    },
    _ACTION_FILE: {
        "graph": "action_encoder",
        "inputs": {"actions": ["batch", "time", "action_dim"]},
        "inputUnits": "raw env action blocks (frameskip-5 x 2D, each in [-1,1]); expert-dataset "
        "z-score applied inside the graph",
        "outputs": {"action_embedding": ["batch", "time", "hidden_dim"]},
        "components": ["zscore-normalize", "action_encoder (Conv1d-SiLU-MLP)"],
    },
    _PREDICTOR_FILE: {
        "graph": "predictor",
        "inputs": {
            "latents": ["batch", "time<=num_frames", "hidden_dim"],
            "action_embeddings": ["batch", "time<=num_frames", "hidden_dim"],
        },
        "outputs": {"predicted_latents": ["batch", "time<=num_frames", "hidden_dim"]},
        "components": ["predictor (AdaLN causal transformer)", "pred_proj (Linear-BN-GELU-Linear)"],
    },
}


def export_browser_graphs(
    model: LeWMTwoRooms,
    out_dir: Path,
    *,
    opset: int = 18,
    action_stats: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
) -> dict[str, Path]:
    """Export the three inference graphs; returns ``{file_name: path}``.

    The model is forced to ``eval()`` (BatchNorm running statistics; no dropout) and every graph
    gets a dynamic batch axis. ``time`` is dynamic on the action/predictor graphs — parity tests
    cover T ∈ {1, 2, num_frames}; T > num_frames is out of contract (the position table ends).
    """
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    graphs: dict[str, nn.Module] = {
        _ENCODER_FILE: EncoderGraph(model),
        _ACTION_FILE: ActionGraph(model, action_stats),
        _PREDICTOR_FILE: PredictorGraph(model),
    }
    dynamic: dict[str, dict[str, dict[int, str]]] = {
        _ENCODER_FILE: {"pixels": {0: "batch"}, "latent": {0: "batch"}},
        _ACTION_FILE: {"actions": {0: "batch", 1: "time"}, "action_embedding": {0: "batch", 1: "time"}},
        _PREDICTOR_FILE: {
            "latents": {0: "batch", 1: "time"},
            "action_embeddings": {0: "batch", 1: "time"},
            "predicted_latents": {0: "batch", 1: "time"},
        },
    }
    names: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
        _ENCODER_FILE: (("pixels",), ("latent",)),
        _ACTION_FILE: (("actions",), ("action_embedding",)),
        _PREDICTOR_FILE: (("latents", "action_embeddings"), ("predicted_latents",)),
    }
    fixtures = _fixtures(model)
    paths: dict[str, Path] = {}
    for file_name, graph in graphs.items():
        graph.eval()
        path = out_dir / file_name
        input_names, output_names = names[file_name]
        torch.onnx.export(
            graph,
            fixtures[file_name],
            path,
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes=dynamic[file_name],
            opset_version=opset,
            dynamo=True,
            # single-file graphs: one artifact = one hash; ORT-web needs no externalData wiring
            external_data=False,
        )
        paths[file_name] = path
    return paths


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def onnxruntime_parity(
    model: LeWMTwoRooms,
    paths: dict[str, Path],
    *,
    atol: float = 1e-4,
    require: bool = False,
    action_stats: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
) -> dict[str, Any]:
    """PyTorch-vs-onnxruntime parity on fixed fixtures, including short-history predictor calls."""
    try:
        import numpy as np
        import onnxruntime as ort  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]
    except Exception as exc:
        if require:
            raise RuntimeError("onnxruntime is required for export parity validation") from exc
        return {
            "onnxruntimeAvailable": False,
            "status": "skipped",
            "reason": "onnxruntime is not installed",
        }

    model.eval()
    graphs: dict[str, nn.Module] = {
        _ENCODER_FILE: EncoderGraph(model).eval(),
        _ACTION_FILE: ActionGraph(model, action_stats).eval(),
        _PREDICTOR_FILE: PredictorGraph(model).eval(),
    }
    cases: dict[str, list[tuple[Tensor, ...]]] = {
        name: [fixture] for name, fixture in _fixtures(model).items()
    }
    # short-history predictor parity (the browser starts rollouts with < num_frames history)
    gen = torch.Generator().manual_seed(7)
    for t in range(1, model.cfg.pred_num_frames):
        cases[_PREDICTOR_FILE].append(
            (
                torch.randn((1, t, model.cfg.hidden_dim), generator=gen),
                torch.randn((1, t, model.cfg.hidden_dim), generator=gen),
            )
        )
    cases[_ACTION_FILE].append(
        (torch.rand((1, 7, model.cfg.action_input_dim), generator=gen),)
    )

    checks: list[dict[str, Any]] = []
    worst = 0.0
    for file_name, fixture_list in cases.items():
        session = ort.InferenceSession(
            str(paths[file_name]), providers=["CPUExecutionProvider"]
        )
        input_names = [i.name for i in session.get_inputs()]
        for fixture in fixture_list:
            with torch.no_grad():
                expected = graphs[file_name](*fixture)
            feeds = {
                name: tensor.numpy().astype(np.float32)
                for name, tensor in zip(input_names, fixture, strict=True)
            }
            (actual,) = session.run(None, feeds)
            max_abs = float(np.max(np.abs(actual - expected.numpy())))
            worst = max(worst, max_abs)
            checks.append(
                {
                    "graph": file_name,
                    "inputShapes": [list(t.shape) for t in fixture],
                    "maxAbsDiff": max_abs,
                }
            )
    return {
        "onnxruntimeAvailable": True,
        "status": "passed" if worst <= atol else "failed",
        "atol": atol,
        "maxAbsDiff": worst,
        "checks": checks,
    }


def browser_export_manifest(
    resolved: ResolvedCheckpoint,
    model: LeWMTwoRooms,
    paths: dict[str, Path],
    parity: dict[str, Any],
    *,
    opset: int,
    weights_sha256: str,
    action_stats_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``lewm-browser-export/1`` manifest binding graphs to the parent checkpoint (gate G2)."""
    cfg = model.cfg
    files = {
        name: {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
            **_IO_SPECS[name],
        }
        for name, path in paths.items()
    }
    return {
        "schema": EXPORT_SCHEMA,
        "graphVersion": EXPORT_GRAPH_VERSION,
        "opset": opset,
        "torchVersion": torch.__version__,
        "checkpoint": {
            "repoId": resolved.repo_id,
            "revision": resolved.revision,
            "weightsSha256": weights_sha256,
        },
        "architecture": {
            "hiddenDim": cfg.hidden_dim,
            "imageSize": cfg.image_size,
            "patchSize": cfg.patch_size,
            "numFrames": cfg.pred_num_frames,
            "actionDim": cfg.action_input_dim,
        },
        "files": files,
        "totalBytes": sum(f["bytes"] for f in files.values()),
        "normalization": {
            "pixels": {
                "where": "inside lewm_tworooms_encoder.onnx",
                "kind": "imagenet",
                "mean": list(IMAGENET_MEAN),
                "std": list(IMAGENET_STD),
                "input": "RGB floats in [0,1], CHW",
            },
            "actions": {
                "where": "inside lewm_tworooms_action.onnx",
                "kind": "expert-dataset z-score (per 2D dim, tiled over the frameskip-5 block)",
                "input": "raw env actions in [-1,1]",
                "stats": action_stats_record
                or {"status": "identity (no dataset stats baked — dev export only)"},
            },
        },
        "runtime": {
            "preferred": "onnxruntime-web webgpu",
            "fallback": "onnxruntime-web wasm",
            "unsupported": "participants report an explicit unsupported state; no silent "
            "surrogate fallback in the real mode",
        },
        "parity": parity,
        "nonClaims": [
            "Checkpoint-backed LeWorldModel inference graphs for the Tapestry-like demo; "
            "not browser training artifacts, not a paper-scale benchmark claim.",
        ],
    }
