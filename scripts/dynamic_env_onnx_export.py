#!/usr/bin/env python3
"""Export the tiny RFC-0017 dynamic-env model to ONNX and optionally check runtime parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec, LatentState
from lensemble.eval.jepa_metrics import load_round_models
from lensemble.model.action_head import build_action_head


class DynamicEnvOnnxModule(nn.Module):
    """Tensor-only wrapper around encoder + local action head + predictor."""

    def __init__(
        self, encoder: nn.Module, action_head: nn.Module, predictor: nn.Module
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.action_head = action_head
        self.predictor = predictor

    def forward(self, clip: Tensor, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        latent = self.encoder(clip)
        action_embedding = self.action_head(action)
        predicted = self.predictor(
            LatentState(
                tokens=latent.tokens,
                num_tokens=latent.num_tokens,
                dim=latent.dim,
                wmcp_version=latent.wmcp_version,
            ),
            action_embedding,
        )
        return latent.tokens, action_embedding, predicted.tokens


def _cfg(args: argparse.Namespace) -> SimpleNamespace:
    num_tokens = (args.num_frames // args.tubelet) * (
        args.image_size // args.patch_size
    ) ** 2
    model = SimpleNamespace(
        encoder="scratch",
        warm_start_release="none",
        latent_dim=args.latent_dim,
        num_tokens=num_tokens,
        predictor_depth=args.predictor_depth,
        predictor_width=args.latent_dim,
        wmcp_version=WMCP_VERSION,
        encoder_frozen=False,
        d=args.latent_dim,
        in_channels=3,
        num_frames=args.num_frames,
        image_size=args.image_size,
        patch_size=args.patch_size,
        tubelet=args.tubelet,
        depth=args.depth,
        num_heads=args.num_heads,
        cond_dim=args.latent_dim,
        mlp_ratio=4.0,
    )
    return SimpleNamespace(model=model)


def _action_spec() -> ActionSpec:
    return ActionSpec(
        embodiment_id="swipe-dot-2dof",
        kind=ActionKind.CONTINUOUS,
        dim=2,
        low=(-1.0, -1.0),
        high=(1.0, 1.0),
        num_classes=None,
        units=("dx", "dy"),
        wmcp_version=WMCP_VERSION,
    )


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("web/dynamic-env-demo/model")
    )
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--predictor-depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--tubelet", type=int, default=1)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument(
        "--require-onnxruntime",
        action="store_true",
        help="Fail if onnxruntime is unavailable instead of writing an export-only report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _args(argv)
    cfg = _cfg(args)
    encoder, predictor = load_round_models(cfg, args.checkpoint_dir)  # type: ignore[arg-type]
    action_head = build_action_head(cfg, _action_spec())
    model = DynamicEnvOnnxModule(
        encoder.eval(), action_head.eval(), predictor.eval()
    ).eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out_dir / "dynamic_env_world_model.onnx"
    report_path = args.out_dir / "dynamic_env_onnx_export_report.json"
    clip = torch.zeros(1, args.num_frames, 3, args.image_size, args.image_size)
    action = torch.tensor([[0.25, -0.25]], dtype=torch.float32)

    with torch.no_grad():
        torch_outputs = model(clip, action)
    torch.onnx.export(
        model,
        (clip, action),
        onnx_path,
        input_names=("clip", "action"),
        output_names=("latent_tokens", "action_embedding", "predicted_tokens"),
        dynamic_axes={
            "clip": {0: "batch"},
            "action": {0: "batch"},
            "latent_tokens": {0: "batch"},
            "action_embedding": {0: "batch"},
            "predicted_tokens": {0: "batch"},
        },
        opset_version=args.opset,
        dynamo=True,
    )

    parity = _runtime_parity(
        onnx_path=onnx_path,
        clip=clip,
        action=action,
        torch_outputs=torch_outputs,
        atol=args.atol,
        require=args.require_onnxruntime,
    )
    report = {
        "schema_version": 1,
        "artifact": str(onnx_path),
        "model_arch": "scratch",
        "task_env_id": "kinematic://swipe-dot",
        "input_shapes": {
            "clip": list(clip.shape),
            "action": list(action.shape),
        },
        "outputs": ["latent_tokens", "action_embedding", "predicted_tokens"],
        "parity": parity,
        "scope": (
            "Browser scope is ONNX inference plus JS/Canvas env-sim only. In-browser training is not claimed."
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def _runtime_parity(
    *,
    onnx_path: Path,
    clip: Tensor,
    action: Tensor,
    torch_outputs: tuple[Tensor, Tensor, Tensor],
    atol: float,
    require: bool,
) -> dict[str, Any]:
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except Exception as exc:
        if require:
            raise RuntimeError("onnxruntime is required for parity validation") from exc
        return {
            "onnxruntime_available": False,
            "status": "skipped",
            "reason": "onnxruntime is not installed",
        }

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.run(
        None,
        {
            "clip": clip.detach().cpu().numpy().astype(np.float32),
            "action": action.detach().cpu().numpy().astype(np.float32),
        },
    )
    max_abs = [
        float(np.max(np.abs(actual - expected.detach().cpu().numpy())))
        for actual, expected in zip(outputs, torch_outputs, strict=True)
    ]
    return {
        "onnxruntime_available": True,
        "status": "passed" if max(max_abs) <= atol else "failed",
        "atol": atol,
        "max_abs": max_abs,
    }


if __name__ == "__main__":
    main()
