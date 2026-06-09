#!/usr/bin/env python
"""Run the Phase-3 latent-space inference audit on a committed checkpoint (#265).

Downloads one or more committed checkpoints (a converged-federated / naive-FedAvg / local-only model repo
+ round) and the held-out SO-100 split, rebuilds the frozen encoder/predictor, and runs the two
simulator-free latent proxy signals from :mod:`lensemble.eval.inference_demo`:

- multi-step open-loop latent prediction quality vs predict-current / predict-random, and
- latent-MPC goal-reaching (the planner reduces the goal-energy below the zero-action baseline).

Writes a JSON inference report. Honest boundary: this is a corrected SO-100 proxy audit, not a usefulness
claim. ``skill_vs_identity`` is gameable, ``effective_rank`` is scale-invariant, and closed-loop physical
task-success stays gated on the unvendored ``stable-worldmodel`` simulator (#96).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from lensemble.config import load_config
from lensemble.contracts import WMCP_VERSION
from lensemble.data.adapters import load_episodes
from lensemble.eval import (
    DYNAMIC_ENV_DOWNSTREAM_REPORT_SCHEMA_VERSION,
    DynamicEnvCheckpointRef,
    DynamicEnvControlReport,
    DynamicEnvDownstreamEvalReport,
    evaluate,
    write_dynamic_env_downstream_eval_report,
)
from lensemble.eval.inference_demo import (
    latent_mpc_goal_reaching,
    multistep_prediction_report,
)
from lensemble.eval.jepa_metrics import load_round_models
from lensemble.model.action_head import build_action_head


def _cfg(args: argparse.Namespace) -> SimpleNamespace:
    num_tokens = (args.num_frames // args.tubelet) * (
        args.image_size // args.patch_size
    ) ** 2
    model = SimpleNamespace(
        encoder="vjepa2-vit-l",
        warm_start_release="vjepa2-2.0",
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


def _eval_cfg(args: argparse.Namespace):
    num_tokens = (args.num_frames // args.tubelet) * (
        args.image_size // args.patch_size
    ) ** 2
    return load_config(
        overrides=[
            "model.encoder=scratch",
            f"model.latent_dim={args.latent_dim}",
            f"model.num_tokens={num_tokens}",
            f"model.num_frames={args.num_frames}",
            f"model.tubelet={args.tubelet}",
            f"model.image_size={args.image_size}",
            f"model.patch_size={args.patch_size}",
            f"model.depth={args.depth}",
            f"model.num_heads={args.num_heads}",
            f"model.predictor_depth={args.predictor_depth}",
            f"model.predictor_width={args.latent_dim}",
            "eval.env_id=kinematic://swipe-dot",
            f"eval.horizon={args.horizon}",
            f"eval.planning_samples={args.planning_samples}",
        ]
    )


def _download_round(repo: str, round_index: int) -> tuple[Path, str]:
    """Download a committed round's checkpoint dir from a model repo; return (dir, pinned_revision_sha)."""
    from huggingface_hub import HfApi, hf_hub_download

    sha = HfApi().model_info(repo).sha or "main"
    sub = f"coordinator-artifacts/round-{round_index:05d}"
    header = hf_hub_download(
        repo, f"{sub}/header.json", repo_type="model", revision=sha
    )
    hf_hub_download(repo, f"{sub}/weights.safetensors", repo_type="model", revision=sha)
    return Path(header).parent, sha


def _evaluate_checkpoint(
    args: argparse.Namespace,
    *,
    label: str,
    repo: str,
    action_spec: Any | None,
    windows: Any | None,
) -> dict[str, Any]:
    cfg = _cfg(args)
    round_dir, sha = _download_round(repo, args.round)
    if args.dynamic_env:
        report = evaluate(
            round_dir,
            "kinematic://swipe-dot",
            cfg=_eval_cfg(args),
            num_episodes=args.dynamic_eval_episodes,
            planner_iters=args.planner_iters,
        )
        return {
            "label": label,
            "model_repo": repo,
            "revision": sha,
            "dynamic_env": {
                "checkpoint_hash": report.checkpoint_hash,
                "state_probe_r2": report.state_probe_r2,
                "success_rate": report.success_rate,
                "success_rate_role": "reported_non_binding",
                "effective_dim": report.effective_dim,
                "planner": report.planner,
                "planning_samples": report.planning_samples,
            },
        }
    if action_spec is None or windows is None:
        raise ValueError("SO-100 inference mode requires action_spec and windows")
    # cfg is a duck-typed namespace carrying .model; build_encoder/build_predictor read it via getattr
    # (the same shape the launcher's _JobModelConfig uses) — runtime-compatible with LensembleConfig.
    encoder, predictor = load_round_models(cfg, round_dir)  # type: ignore[arg-type]
    action_head = build_action_head(cfg, action_spec)
    prediction = multistep_prediction_report(
        encoder=encoder,
        predictor=predictor,
        action_head=action_head,
        windows=windows,
        horizon=args.horizon,
        max_windows=args.max_windows,
    )
    planning = latent_mpc_goal_reaching(
        encoder=encoder,
        predictor=predictor,
        action_head=action_head,
        windows=windows,
        horizon=args.horizon,
        planning_samples=args.planning_samples,
        planner_iters=args.planner_iters,
        max_episodes=args.max_episodes,
    )
    return {
        "label": label,
        "model_repo": repo,
        "revision": sha,
        "multistep_prediction": prediction,
        "latent_mpc": planning,
        "metric_boundary": (
            "Corrected SO-100 proxy audit: skill_vs_identity is gameable; effective_rank is "
            "scale-invariant; latent-MPC success_rate=0.0 is a negative result, not a near-static-video "
            "success story; held-out magnitude collapse (~7.5e-6 latent variance; "
            "thoughts/collapse_fix_probe.py) and the central ceiling probe "
            "(thoughts/central_ceiling_probe.py) show this is not downstream usefulness."
        ),
    }


def _args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase-3 latent inference demonstration (#265)."
    )
    p.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=REPO",
        help="A control to evaluate, e.g. converged-federated=abdelstark/lensemble-phase3-converged-checkpoint. Repeat.",
    )
    p.add_argument("--round", type=int, default=12)
    p.add_argument("--heldout-repo", default="abdelstark/lensemble-phase3-so100-silos")
    p.add_argument("--heldout-file", default="phase3-so100-silo4.h5")
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--max-windows", type=int, default=64)
    p.add_argument("--max-episodes", type=int, default=24)
    p.add_argument("--dynamic-eval-episodes", type=int, default=6)
    p.add_argument("--planning-samples", type=int, default=256)
    p.add_argument("--planner-iters", type=int, default=4)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--predictor-depth", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--num-frames", type=int, default=1)
    p.add_argument("--tubelet", type=int, default=1)
    p.add_argument(
        "--output", default="docs/evidence/phase3_inference_demo_report.json"
    )
    p.add_argument(
        "--dynamic-env",
        action="store_true",
        help="Evaluate checkpoints on kinematic://swipe-dot and emit the RFC-0017 dynamic report container.",
    )
    p.add_argument(
        "--dynamic-data-ref",
        default="synthetic-dynamic://swipe-dot?seed=0&n_episodes=8&steps=64&image_size=48",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _args(argv)
    windows = None
    action_spec = None
    if not args.dynamic_env:
        from huggingface_hub import hf_hub_download

        heldout = hf_hub_download(
            args.heldout_repo, args.heldout_file, repo_type="dataset"
        )
        dataset = load_episodes(f"lerobot-h5://{heldout}", fmt="lerobot-h5")
        windows = list(dataset.windows(args.horizon))
        action_spec = dataset.episodes[0].action_spec

    controls = []
    for spec in args.checkpoint:
        label, repo = spec.split("=", 1)
        torch.manual_seed(0)
        controls.append(
            _evaluate_checkpoint(
                args, label=label, repo=repo, action_spec=action_spec, windows=windows
            )
        )

    out = Path(args.output)
    if args.dynamic_env:
        dynamic_controls = []
        for control in controls:
            metrics = control["dynamic_env"]
            dynamic_controls.append(
                DynamicEnvControlReport(
                    label=control["label"],
                    checkpoint=DynamicEnvCheckpointRef(
                        repo_id=control["model_repo"],
                        revision=control["revision"],
                        checkpoint_hash=metrics["checkpoint_hash"],
                    ),
                    state_probe_r2=metrics["state_probe_r2"],
                    success_rate=metrics["success_rate"],
                    latent_goal_success_rate=None,
                    effective_rank=metrics["effective_dim"],
                    metric_boundary=(
                        "state_probe_r2 is binding; closed-loop success is reported non-binding; "
                        "latent metrics are supporting and gameable"
                    ),
                )
            )
        report_model = DynamicEnvDownstreamEvalReport(
            schema_version=DYNAMIC_ENV_DOWNSTREAM_REPORT_SCHEMA_VERSION,
            generated_at=datetime.now(timezone.utc),
            task_env_id="kinematic://swipe-dot",
            held_out_data_ref=args.dynamic_data_ref,
            controls=tuple(dynamic_controls),
            claim_boundary=(
                "synthetic control env; state_probe_r2 is the binding ground-truth metric; "
                "latent-MPC and skill metrics are gameable supporting signals; no paper-scale robotics claim"
            ),
            source_report_uri=str(out),
        )
        write_dynamic_env_downstream_eval_report(report_model, out)
        report = report_model.model_dump(mode="json")
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return report

    report = {
        "schema_version": 1,
        "task_env_id": "held-out-so100://phase3-so100-silo4",
        "held_out_data_ref": f"{args.heldout_repo}/{args.heldout_file}",
        "horizon": args.horizon,
        "windows_available": len(windows),
        "honest_boundary": (
            "Corrected SO-100 proxy audit on real held-out data: skill_vs_identity is gameable, "
            "effective_rank is scale-invariant, and latent-MPC success_rate=0.0 is a negative result, "
            "not a near-static-video success story. Held-out magnitude collapse (~7.5e-6 latent variance; "
            "thoughts/collapse_fix_probe.py) plus the central ceiling probe "
            "(thoughts/central_ceiling_probe.py) show this checkpoint is not downstream-useful. "
            "Closed-loop physical task-success is NOT claimed here; it stays gated on the unvendored "
            "stable-worldmodel simulator (#96). Not a cryptographic honest-computation proof."
        ),
        "controls": controls,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    main()
