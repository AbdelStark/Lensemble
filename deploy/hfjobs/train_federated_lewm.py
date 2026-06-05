# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "numpy",
#   "h5py",
#   "safetensors",
#   "huggingface-hub",
#   "lensemble @ git+https://github.com/AbdelStark/Lensemble.git@main",
# ]
# ///
"""HF Jobs launcher for the federated LeWorldModel claim MVP.

Runs the real Lensemble federated runtime over mounted participant-local data sources:
``Coordinator`` + default ``Participant`` hooks + ``lerobot-h5://`` data loading + claim-mode
``Objective(target_stop_gradient=False)``. The output directory contains hash-committed coordinator
checkpoints, the contribution ledger, and ``claim_mvp_report.json``. With ``--push`` and ``HF_TOKEN`` it
can also upload source HDF5 files to dataset repos and the checkpoint/report folder to a model repo.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from lensemble.artifacts import load_checkpoint
from lensemble.config import LensembleConfig
from lensemble.contracts import WMCP_VERSION, LatentState
from lensemble.data.adapters import load_episodes
from lensemble.data.probe import build_probe, load_probe, save_probe
from lensemble.eval import (
    ClaimMetricEvidence,
    ClaimPublicationEvidence,
    build_claim_mvp_report,
)
from lensemble.federation import (
    Coordinator,
    InProcessTransport,
    Participant,
    RoundState,
)
from lensemble.gauge import frame_drift
from lensemble.model import (
    build_action_head,
    build_predictor,
    build_sketch,
    sigreg_statistic,
)
from lensemble.model.encoder import build_encoder, snapshot_reference


@dataclass(frozen=True)
class _JobModelConfig:
    """Config shape consumed by build_encoder/build_predictor/model_arch_from_config."""

    encoder: str
    warm_start_release: str
    latent_dim: int
    num_tokens: int
    predictor_depth: int
    predictor_width: int
    wmcp_version: str
    encoder_frozen: bool
    d: int
    in_channels: int
    num_frames: int
    image_size: int
    patch_size: int
    tubelet: int
    depth: int
    num_heads: int
    cond_dim: int
    mlp_ratio: float = 4.0


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the federated LeWorldModel claim MVP on HF Jobs."
    )
    parser.add_argument(
        "--data-source",
        action="append",
        required=True,
        help="Participant data source. Repeat once per silo, e.g. lerobot-h5:///data/silo0.h5.",
    )
    parser.add_argument(
        "--participant-id",
        action="append",
        default=None,
        help="Participant id. Repeat to match --data-source; defaults to silo-0, silo-1, ...",
    )
    parser.add_argument("--data-format", default="lerobot-h5")
    parser.add_argument("--out-dir", default="/tmp/lensemble-claim-mvp")
    parser.add_argument("--probe-path", default=None)
    parser.add_argument("--probe-points", type=int, default=None)
    parser.add_argument("--num-rounds", type=int, default=1)
    parser.add_argument("--inner-horizon", type=int, default=1)
    parser.add_argument("--window-steps", type=int, default=1)
    parser.add_argument("--outer-lr", type=float, default=0.7)
    parser.add_argument("--lambda-sig", type=float, default=0.1)
    parser.add_argument("--lambda-anc", type=float, default=0.01)
    parser.add_argument(
        "--target-stop-gradient",
        action="store_true",
        help="Use the legacy detached target helper. Omit for claim-grade LeWorldModel base mode.",
    )
    parser.add_argument("--privacy", action="store_true", help="Enable DP clip/noise.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--tubelet", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--predictor-depth", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument(
        "--metric-windows",
        type=int,
        default=32,
        help="Maximum windows used for scalar report metrics.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument(
        "--out-repo", default=None, help="HF model repo for checkpoints/report."
    )
    parser.add_argument(
        "--dataset-repo",
        action="append",
        default=None,
        help="HF dataset repo to upload a matching --data-source file. Repeat to map by order.",
    )
    parser.add_argument("--public", action="store_true", help="Create public HF repos.")
    return parser.parse_args()


def _num_tokens(args: argparse.Namespace) -> int:
    if args.num_frames % args.tubelet != 0 or args.image_size % args.patch_size != 0:
        raise ValueError(
            "num_frames/tubelet and image_size/patch_size must divide exactly"
        )
    return (args.num_frames // args.tubelet) * (args.image_size // args.patch_size) ** 2


def _model_cfg(args: argparse.Namespace) -> _JobModelConfig:
    return _JobModelConfig(
        encoder="vjepa2-vit-l",
        warm_start_release="vjepa2-2.0",
        latent_dim=args.latent_dim,
        num_tokens=_num_tokens(args),
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
        mlp_ratio=args.mlp_ratio,
    )


def _cfg(
    args: argparse.Namespace,
    *,
    data_source: str | None,
    probe_path: Path,
    run_mode: str,
) -> LensembleConfig:
    base = LensembleConfig()
    participant_count = len(args.data_source)
    federation = dataclasses.replace(
        base.federation,
        participant_count=participant_count,
        num_rounds=args.num_rounds,
        inner_horizon=args.inner_horizon,
        outer_lr=args.outer_lr,
        fault_tolerance_min_participants=participant_count,
        secure_agg_threshold=participant_count,
    )
    data = dataclasses.replace(
        base.data,
        format=args.data_format,
        data_source=data_source,
        probe_path=str(probe_path),
        window_steps=args.window_steps,
    )
    objective = dataclasses.replace(
        base.objective,
        lambda_sig=args.lambda_sig,
        lambda_anc=args.lambda_anc,
        target_stop_gradient=bool(args.target_stop_gradient),
    )
    privacy = dataclasses.replace(base.privacy, enabled=bool(args.privacy))
    return dataclasses.replace(
        base,
        model=_model_cfg(args),  # type: ignore[arg-type]
        federation=federation,
        data=data,
        objective=objective,
        privacy=privacy,
        run_mode=run_mode,  # type: ignore[arg-type]
    )


def _participant_ids(args: argparse.Namespace) -> list[str]:
    if args.participant_id is None:
        return [f"silo-{i}" for i in range(len(args.data_source))]
    if len(args.participant_id) != len(args.data_source):
        raise ValueError("--participant-id count must match --data-source count")
    if len(set(args.participant_id)) != len(args.participant_id):
        raise ValueError("--participant-id values must be unique")
    return list(args.participant_id)


def _source_path(source: str) -> Path:
    text = source.removeprefix("lerobot-h5://")
    return Path(text)


def _validate_sources(args: argparse.Namespace) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in args.data_source:
        dataset = load_episodes(source, fmt=args.data_format)
        windows = list(dataset.windows(args.window_steps))
        if not windows:
            raise ValueError(
                f"{source!r} produced zero windows for window_steps={args.window_steps}"
            )
        counts[source] = len(windows)
    return counts


def _ensure_probe(args: argparse.Namespace, out_dir: Path) -> Path:
    if args.probe_path is not None:
        return Path(args.probe_path)
    probe_points = args.probe_points or args.latent_dim
    if probe_points < args.latent_dim:
        raise ValueError("--probe-points must be >= --latent-dim for the frame anchor")
    probe_path = out_dir / "probe.safetensors"
    cfg = _cfg(
        args,
        data_source=args.data_source[0],
        probe_path=probe_path,
        run_mode="participant",
    )
    points = torch.randn(
        probe_points,
        args.num_frames,
        3,
        args.image_size,
        args.image_size,
        generator=torch.Generator().manual_seed(20260605),
    )
    probe = build_probe(
        points,
        torch.arange(probe_points),
        snapshot_reference(build_encoder(cfg)),
    )
    save_probe(probe, probe_path)
    return probe_path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _publication(
    args: argparse.Namespace,
    out_dir: Path,
    *,
    pushed: bool = False,
    blocker: str | None = None,
) -> ClaimPublicationEvidence:
    return ClaimPublicationEvidence(
        dataset_repos=tuple(args.dataset_repo or ()),
        checkpoint_repo=args.out_repo,
        checkpoint_path=str(out_dir / "artifacts"),
        pushed=pushed,
        dry_run=bool(args.dry_run),
        blocker=blocker,
    )


def _push_outputs(
    args: argparse.Namespace, out_dir: Path, report_path: Path
) -> ClaimPublicationEvidence:
    if not args.push:
        return _publication(args, out_dir)
    token = os.environ.get("HF_TOKEN")
    if not token:
        return _publication(args, out_dir, blocker="HF_TOKEN is not set")
    if args.out_repo is None and not args.dataset_repo:
        return _publication(
            args, out_dir, blocker="--push requires --out-repo and/or --dataset-repo"
        )

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    private = not args.public
    if args.dataset_repo:
        if len(args.dataset_repo) != len(args.data_source):
            return _publication(
                args,
                out_dir,
                blocker="--dataset-repo count must match --data-source count",
            )
        for repo_id, source in zip(args.dataset_repo, args.data_source):
            path = _source_path(source)
            if not path.exists():
                return _publication(
                    args, out_dir, blocker=f"dataset source not local: {source}"
                )
            api.create_repo(
                repo_id, repo_type="dataset", private=private, exist_ok=True
            )
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path.name,
                repo_id=repo_id,
                repo_type="dataset",
            )
    if args.out_repo:
        api.create_repo(
            args.out_repo, repo_type="model", private=private, exist_ok=True
        )
        api.upload_folder(
            folder_path=str(out_dir),
            repo_id=args.out_repo,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj=str(report_path),
            path_in_repo=report_path.name,
            repo_id=args.out_repo,
            repo_type="model",
        )
    return _publication(args, out_dir, pushed=True)


def _push_report(args: argparse.Namespace, report_path: Path) -> None:
    """Upload the final report after its publication fields have been rewritten."""
    if not args.push or not args.out_repo:
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        return
    from huggingface_hub import HfApi

    HfApi(token=token).upload_file(
        path_or_fileobj=str(report_path),
        path_in_repo=report_path.name,
        repo_id=args.out_repo,
        repo_type="model",
    )


def _effective_rank(embeddings: torch.Tensor) -> float:
    x = embeddings.reshape(-1, embeddings.shape[-1]).to(torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov).clamp_min(1e-12)
    p = ev / ev.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def _launcher_inputs_hash(out_dir: Path) -> str | None:
    path = out_dir / "launcher_inputs.json"
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_checkpoint_groups(
    checkpoint_dir: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    weights, _header = load_checkpoint(checkpoint_dir)
    theta = {
        k.removeprefix("encoder."): v
        for k, v in weights.items()
        if k.startswith("encoder.")
    }
    phi = {
        k.removeprefix("predictor."): v
        for k, v in weights.items()
        if k.startswith("predictor.")
    }
    return theta, phi


def _load_final_models(cfg: LensembleConfig, checkpoint_dir: Path) -> tuple[Any, Any]:
    theta, phi = _load_checkpoint_groups(checkpoint_dir)
    encoder = build_encoder(cfg).eval()
    predictor = build_predictor(cfg).eval()
    encoder.load_state_dict(theta, strict=True)
    predictor.load_state_dict(phi, strict=True)
    return encoder, predictor


def _unflatten_update_delta(
    theta_template: Mapping[str, torch.Tensor],
    phi_template: Mapping[str, torch.Tensor],
    flat: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    theta: dict[str, torch.Tensor] = {}
    phi: dict[str, torch.Tensor] = {}
    offset = 0
    for group, template, out in (
        ("encoder", theta_template, theta),
        ("predictor", phi_template, phi),
    ):
        for name in sorted(template):
            ref = template[name]
            end = offset + ref.numel()
            if end > flat.numel():
                raise ValueError(
                    f"released delta ended inside {group}.{name}; expected {end} values, "
                    f"got {flat.numel()}"
                )
            out[name] = flat[offset:end].reshape(ref.shape)
            offset = end
    if offset != flat.numel():
        raise ValueError(
            f"released delta has {flat.numel() - offset} trailing values after encoder/predictor groups"
        )
    return theta, phi


def _apply_theta_delta(
    theta: Mapping[str, torch.Tensor], delta: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    updated: dict[str, torch.Tensor] = {}
    for name, base in theta.items():
        change = delta[name].to(device=base.device)
        if torch.is_floating_point(base):
            updated[name] = base + change.to(dtype=base.dtype)
        else:
            updated[name] = (
                (base.to(torch.float32) + change).round().to(dtype=base.dtype)
            )
    return updated


def _probe_embedding(encoder: Any, probe: Any) -> torch.Tensor:
    try:
        device = next(encoder.parameters()).device
    except StopIteration:  # pragma: no cover - encoders used here have parameters
        device = probe.points.device
    encoded = encoder(probe.points.to(device))
    return encoded.tokens.reshape(-1, encoded.tokens.shape[-1]).detach().cpu()


def _mean_frame_drift_deg(report: Any) -> float | None:
    angles = [pair.rotation_angle_deg for pair in report.pairs]
    if not angles:
        angles = list(report.drift_from_global.values())
    if not angles:
        return None
    return sum(float(angle) for angle in angles) / len(angles)


def _claim_frame_drift_deg(
    cfg: LensembleConfig,
    *,
    out_dir: Path,
    committed_rounds: int,
    participant_updates: Mapping[str, Any],
    final_encoder: Any,
) -> float | None:
    if committed_rounds <= 0 or not participant_updates:
        return None
    probe_path = getattr(cfg.data, "probe_path", None)
    if probe_path is None:
        return None

    prior_round = committed_rounds - 1
    prior_theta, prior_phi = _load_checkpoint_groups(
        out_dir / "artifacts" / f"round-{prior_round:05d}"
    )
    probe = load_probe(Path(probe_path))
    embeddings = {"global": _probe_embedding(final_encoder, probe)}
    for participant_id, update in sorted(participant_updates.items()):
        if int(update.round_index) != prior_round:
            raise ValueError(
                f"participant {participant_id!r} update is for round {update.round_index}, "
                f"but the latest committed round expects {prior_round}"
            )
        theta_delta, _phi_delta = _unflatten_update_delta(
            prior_theta, prior_phi, update.delta
        )
        local_encoder = build_encoder(cfg).eval()
        local_encoder.load_state_dict(
            _apply_theta_delta(prior_theta, theta_delta), strict=True
        )
        embeddings[participant_id] = _probe_embedding(local_encoder, probe)

    report = frame_drift(
        embeddings,
        round_index=prior_round,
        probe=probe,
        expected_probe_hash=probe.content_hash.hex(),
    )
    return _mean_frame_drift_deg(report)


def _claim_metrics(
    args: argparse.Namespace,
    cfg: LensembleConfig,
    *,
    out_dir: Path,
    committed_rounds: int,
    participant_updates: Mapping[str, Any],
) -> ClaimMetricEvidence:
    if args.dry_run or committed_rounds <= 0:
        return ClaimMetricEvidence(run_manifest_hash=_launcher_inputs_hash(out_dir))
    checkpoint_dir = out_dir / "artifacts" / f"round-{committed_rounds:05d}"
    encoder, predictor = _load_final_models(cfg, checkpoint_dir)
    sketch = build_sketch(0, int(args.latent_dim), 64)
    pred_losses: list[float] = []
    sigreg_losses: list[float] = []
    embeddings: list[torch.Tensor] = []
    remaining = max(0, int(args.metric_windows))
    with torch.no_grad():
        for source in args.data_source:
            if remaining <= 0:
                break
            dataset = load_episodes(source, fmt=args.data_format)
            action_head = build_action_head(cfg, dataset.episodes[0].action_spec).eval()
            for window in dataset.windows(args.window_steps):
                encoded = encoder(window.obs)
                tokens = encoded.tokens
                input_latent = LatentState(
                    tokens=tokens[:-1],
                    num_tokens=encoded.num_tokens,
                    dim=encoded.dim,
                    wmcp_version=encoded.wmcp_version,
                )
                target = tokens[1:]
                action_embedding = action_head.encode(window.actions)
                pred_tokens = predictor(input_latent, action_embedding).tokens
                pred_losses.append(float((pred_tokens - target).pow(2).mean()))
                sigreg_losses.append(
                    float(
                        sigreg_statistic(tokens.reshape(-1, tokens.shape[-1]), sketch)
                    )
                )
                embeddings.append(tokens.reshape(-1, tokens.shape[-1]).cpu())
                remaining -= 1
                if remaining <= 0:
                    break
    if not pred_losses or not embeddings:
        return ClaimMetricEvidence(run_manifest_hash=_launcher_inputs_hash(out_dir))
    frame_drift_deg = _claim_frame_drift_deg(
        cfg,
        out_dir=out_dir,
        committed_rounds=committed_rounds,
        participant_updates=participant_updates,
        final_encoder=encoder,
    )
    return ClaimMetricEvidence(
        val_pred=sum(pred_losses) / len(pred_losses),
        val_sigreg=sum(sigreg_losses) / len(sigreg_losses),
        effective_rank=_effective_rank(torch.cat(embeddings, dim=0)),
        frame_drift_deg=frame_drift_deg,
        run_manifest_hash=_launcher_inputs_hash(out_dir),
    )


def main() -> None:
    args = _args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    participant_ids = _participant_ids(args)
    source_window_counts = _validate_sources(args)
    probe_path = _ensure_probe(args, out_dir)
    coord_cfg = _cfg(
        args, data_source=None, probe_path=probe_path, run_mode="coordinator"
    )

    _write_json(
        out_dir / "launcher_inputs.json",
        {
            "participant_ids": participant_ids,
            "data_sources": args.data_source,
            "source_window_counts": source_window_counts,
            "dry_run": bool(args.dry_run),
            "objective_target_stop_gradient": bool(args.target_stop_gradient),
        },
    )

    transport = InProcessTransport()
    coordinator = Coordinator(
        coord_cfg, transport=transport, artifacts_dir=out_dir / "artifacts"
    )
    updates: dict[str, Any] = {}
    round_state: RoundState | str = "dry_run"
    if not args.dry_run:
        for _ in range(args.num_rounds):
            state = coordinator.global_state()
            updates = {}
            for participant_id, source in zip(participant_ids, args.data_source):
                cfg = _cfg(
                    args,
                    data_source=source,
                    probe_path=probe_path,
                    run_mode="participant",
                )
                participant = Participant(
                    cfg, participant_id=participant_id, transport=transport
                )
                update = participant.local_round(state, state.sketch_seed)
                updates[participant_id] = update
                transport.submit_update(
                    participant_id=participant_id,
                    round_index=state.round_index,
                    update=update,
                )
            round_state = coordinator.try_round()
            if round_state is not RoundState.CLOSED:
                break

    report_path = out_dir / "claim_mvp_report.json"
    sources = dict(zip(participant_ids, args.data_source))
    metrics = _claim_metrics(
        args,
        coord_cfg,
        out_dir=out_dir,
        committed_rounds=len(coordinator.ledger_records()),
        participant_updates=updates,
    )
    report = build_claim_mvp_report(
        cfg=coord_cfg,
        coordinator=coordinator,
        participant_updates=updates,
        participant_sources=sources,
        round_state=round_state,
        metrics=metrics,
        publication=_publication(args, out_dir),
    )
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    publication = _push_outputs(args, out_dir, report_path)
    if publication != report.publication:
        report = build_claim_mvp_report(
            cfg=coord_cfg,
            coordinator=coordinator,
            participant_updates=updates,
            participant_sources=sources,
            round_state=round_state,
            metrics=metrics,
            publication=publication,
        )
        report_path.write_text(
            report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        if publication.pushed and publication.blocker is None:
            _push_report(args, report_path)

    print(report.model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
