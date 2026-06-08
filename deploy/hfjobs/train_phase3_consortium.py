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
"""HF Jobs launcher for the FULL Phase 3 consortium runtime.

This drives the operational Phase 3 stack — the networked ``Phase3CoordinatorService`` plus one
sovereign ``Phase3ParticipantAgent`` per mounted participant-local data ref — for ``--num-rounds``
closed federated rounds and emits REAL residency-safe per-round JEPA metrics
(``val_pred``/``val_sigreg``/``effective_rank``/``frame_drift_deg``) measured off the committed global
checkpoints and a disjoint held-out eval split. No raw participant trajectory ever leaves a participant
boundary; only pseudo-gradients and residency-safe metadata cross.

The metric/orchestration LIBRARY is :func:`lensemble.federation.run_phase3_consortium`; this script only
loads each silo's residency-safe metadata, builds the agreed manifest + dataset/probe registry from the
actual loaded data, pins the public-probe hash, constructs the typed ``Phase3ConsortiumInputs``, and
(unless ``--dry-run``) runs the consortium and writes ``phase3_long_run_smoke_report.json``. With
``--push`` and ``HF_TOKEN`` it uploads the run directory to a model repo.

``--dry-run`` validates the manifest + dataset/probe registry, pins the public-probe hash, and preflights
every participant agent WITHOUT running any federated round or any training compute.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from lensemble.config import (
    LensembleConfig,
    Phase3ActionContract,
    Phase3ConsortiumManifest,
    Phase3Contact,
    Phase3DataDeclaration,
    Phase3DPPolicy,
    Phase3ModelAgreement,
    Phase3ObservationContract,
    Phase3ParticipantCapabilities,
    Phase3ParticipantDeclaration,
    Phase3PublicProbe,
    Phase3RuntimePolicy,
)
from lensemble.config.manifest import config_hash
from lensemble.contracts import WMCP_VERSION, ActionSpec
from lensemble.data.adapters import load_episodes
from lensemble.data.phase3 import (
    phase3_registry_from_consortium_manifest,
    validate_phase3_registry_against_manifest,
    write_phase3_dataset_registry,
)
from lensemble.data.probe import PublicProbe, build_probe, save_probe
from lensemble.federation import (
    InProcessTransport,
    Phase3ArtifactTargets,
    Phase3ConsortiumInputs,
    Phase3ParticipantAgent,
    run_phase3_consortium,
    write_phase3_long_run_report,
)
from lensemble.federation.phase3_orchestration import Phase3LongRunReport
from lensemble.model.encoder import build_encoder, snapshot_reference

# The honest scope of the evidence this launcher produces (RFC-0005 §4, RFC-0002): a real federated
# consortium-engineering + training run, NOT a cryptographic honest-computation proof.
_CLAIM_BOUNDARY = (
    "Real federated Phase 3 consortium run: sovereign participant-agents + a networked coordinator-service "
    "over mounted participant-local data refs produce committed global checkpoints, a contribution ledger, "
    "secure-aggregation/DP-accounting reports, and residency-safe per-round JEPA metrics. It is engineering "
    "and training evidence, NOT a cryptographic proof of honest computation, and not a paper-scale robotics "
    "performance result."
)

# Eval-planner budget is deferred to #245; the per-round JEPA metrics here are representation metrics, not
# downstream task-success planning.
_EVAL_BUDGET = (
    "Per-round JEPA representation metrics only (val_pred/val_sigreg/effective_rank/frame_drift_deg); "
    "downstream planner/task-success eval budget deferred to #245."
)

_OBS_DTYPE = "float32"


# --------------------------------------------------------------------------------------------------- #
# Model config shape (mirrors deploy/hfjobs/train_federated_lewm.py::_JobModelConfig).
# --------------------------------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
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


@dataclasses.dataclass(frozen=True)
class _SiloMetadata:
    """Residency-safe metadata loaded from one participant silo (no raw arrays)."""

    participant_id: str
    data_ref: str
    window_count: int
    first_obs_shape: tuple[int, ...]
    action_spec: ActionSpec
    embodiment_id: str


def _args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the FULL Phase 3 consortium runtime on HF Jobs."
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
    parser.add_argument(
        "--heldout-source",
        default=None,
        help="Held-out eval split (disjoint from every silo); required for a real run.",
    )
    parser.add_argument("--out-dir", default="/tmp/lensemble-phase3-consortium")
    parser.add_argument("--probe-path", default=None)
    parser.add_argument("--probe-points", type=int, default=None)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--inner-horizon", type=int, default=1)
    parser.add_argument("--window-steps", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--predictor-depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--tubelet", type=int, default=1)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--lambda-sig", type=float, default=0.1)
    parser.add_argument("--lambda-anc", type=float, default=0.01)
    parser.add_argument("--target-stop-gradient", action="store_true")
    parser.add_argument(
        "--privacy",
        dest="privacy",
        action="store_true",
        default=True,
        help="Enable DP clip/noise (default for the real run).",
    )
    parser.add_argument(
        "--no-privacy",
        dest="privacy",
        action="store_false",
        help="Disable DP clip/noise.",
    )
    parser.add_argument("--dp-epsilon", type=float, default=8.0)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--dp-clip-norm", type=float, default=0.5)
    parser.add_argument("--dp-noise-multiplier", type=float, default=1.0)
    parser.add_argument("--dp-accountant", default="rdp")
    parser.add_argument("--secure-agg-backend", default="simulated")
    parser.add_argument("--secure-agg-threshold", type=int, default=None)
    parser.add_argument("--min-trainers", type=int, default=3)
    parser.add_argument("--metric-windows", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--out-repo", default=None)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--consortium-id", default="lensemble-phase3-consortium")
    parser.add_argument("--run-id", default="phase3-consortium-v1")
    return parser.parse_args(argv)


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


def _participant_ids(args: argparse.Namespace) -> tuple[str, ...]:
    if args.participant_id is None:
        return tuple(f"silo-{i}" for i in range(len(args.data_source)))
    if len(args.participant_id) != len(args.data_source):
        raise ValueError("--participant-id count must match --data-source count")
    if len(set(args.participant_id)) != len(args.participant_id):
        raise ValueError("--participant-id values must be unique")
    return tuple(args.participant_id)


def _secure_agg_threshold(args: argparse.Namespace) -> int:
    if args.secure_agg_threshold is not None:
        return int(args.secure_agg_threshold)
    return len(args.data_source)


def _coordinator_cfg(
    args: argparse.Namespace, *, probe_path: Path, participant_count: int
) -> LensembleConfig:
    base = LensembleConfig()
    federation = dataclasses.replace(
        base.federation,
        participant_count=participant_count,
        num_rounds=args.num_rounds,
        inner_horizon=args.inner_horizon,
        fault_tolerance_min_participants=args.min_trainers,
        secure_agg_threshold=_secure_agg_threshold(args),
        aggregation_backend=args.secure_agg_backend,  # type: ignore[arg-type]
        transport="in_process",
    )
    objective = dataclasses.replace(
        base.objective,
        lambda_sig=args.lambda_sig,
        lambda_anc=args.lambda_anc,
        target_stop_gradient=bool(args.target_stop_gradient),
    )
    privacy = dataclasses.replace(
        base.privacy,
        enabled=bool(args.privacy),
        clip_norm=args.dp_clip_norm,
        noise_multiplier=args.dp_noise_multiplier,
        epsilon=args.dp_epsilon,
        delta=args.dp_delta,
        accountant=args.dp_accountant,  # type: ignore[arg-type]
    )
    data = dataclasses.replace(
        base.data,
        format=args.data_format,  # type: ignore[arg-type]
        data_source=None,
        probe_path=str(probe_path),
        window_steps=args.window_steps,
        residency_enforced=True,
    )
    return dataclasses.replace(
        base,
        model=_model_cfg(args),  # type: ignore[arg-type]
        federation=federation,
        objective=objective,
        privacy=privacy,
        data=data,
        run_mode="coordinator",
    )


def _participant_cfg(
    cfg: LensembleConfig, args: argparse.Namespace, *, data_source: str
) -> LensembleConfig:
    data = dataclasses.replace(
        cfg.data,
        data_source=data_source,
        format=args.data_format,  # type: ignore[arg-type]
        window_steps=args.window_steps,
        residency_enforced=True,
    )
    return dataclasses.replace(cfg, data=data, run_mode="participant")


def _load_silo_metadata(
    args: argparse.Namespace, *, participant_id: str, data_source: str
) -> _SiloMetadata:
    dataset = load_episodes(data_source, fmt=args.data_format)
    episodes = dataset.episodes
    if not episodes:
        raise ValueError(f"{data_source!r} has no episodes")
    action_spec = episodes[0].action_spec
    first_window = None
    window_count = 0
    for window in dataset.windows(args.window_steps):
        if first_window is None:
            first_window = window
        window_count += 1
    if first_window is None or window_count <= 0:
        raise ValueError(
            f"{data_source!r} produced zero windows for window_steps={args.window_steps}"
        )
    return _SiloMetadata(
        participant_id=participant_id,
        data_ref=data_source,
        window_count=window_count,
        first_obs_shape=tuple(int(x) for x in first_window.obs.shape),
        action_spec=action_spec,
        embodiment_id=first_window.embodiment_id,
    )


def _action_contract(spec: ActionSpec) -> Phase3ActionContract:
    return Phase3ActionContract(
        contract_id=f"phase3-consortium-action-{spec.embodiment_id}-v1",
        embodiment_id=spec.embodiment_id,
        kind=spec.kind.value,  # type: ignore[arg-type]
        dim=spec.dim,
        low=spec.low,
        high=spec.high,
        num_classes=spec.num_classes,
        units=spec.units,
        wmcp_version=spec.wmcp_version,
    )


def _observation_contract(
    shape: tuple[int, ...], *, wmcp_version: str
) -> Phase3ObservationContract:
    return Phase3ObservationContract(
        contract_id="phase3-consortium-window-v1",
        shape=shape,
        dtype=_OBS_DTYPE,
        frame_skip=1,
        wmcp_version=wmcp_version,
    )


def _smoke_report_sha256(meta: _SiloMetadata) -> str:
    payload = {
        "participant_id": meta.participant_id,
        "data_ref": meta.data_ref,
        "window_count": meta.window_count,
        "window_steps": int(meta.first_obs_shape[0]) - 1,
        "first_obs_shape": list(meta.first_obs_shape),
        "action_dim": meta.action_spec.dim,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _shared_contracts(
    metas: list[_SiloMetadata], *, wmcp_version: str
) -> tuple[Phase3ActionContract, Phase3ObservationContract]:
    first = metas[0]
    for meta in metas[1:]:
        if meta.first_obs_shape != first.first_obs_shape:
            raise ValueError(
                "all participant silos must share one observation/window shape; "
                f"{meta.participant_id!r} has {meta.first_obs_shape}, "
                f"{first.participant_id!r} has {first.first_obs_shape}"
            )
        if meta.action_spec != first.action_spec:
            raise ValueError(
                "all participant silos must share one ActionSpec; "
                f"{meta.participant_id!r} differs from {first.participant_id!r}"
            )
        if meta.embodiment_id != first.embodiment_id:
            raise ValueError(
                "all participant silos must share one window embodiment_id; "
                f"{meta.participant_id!r} has {meta.embodiment_id!r}, "
                f"{first.participant_id!r} has {first.embodiment_id!r}"
            )
    return (
        _action_contract(first.action_spec),
        _observation_contract(first.first_obs_shape, wmcp_version=wmcp_version),
    )


def _build_manifest(
    args: argparse.Namespace,
    cfg: LensembleConfig,
    *,
    metas: list[_SiloMetadata],
    public_probe_hash: str,
) -> Phase3ConsortiumManifest:
    wmcp_version = cfg.model.wmcp_version
    action, observation = _shared_contracts(metas, wmcp_version=wmcp_version)
    probe = Phase3PublicProbe(
        probe_id=f"{args.consortium_id}-public-probe",
        version=1,
        content_hash=public_probe_hash,
    )
    capabilities = Phase3ParticipantCapabilities(
        network_transport=False,
        secure_aggregation_backends=(args.secure_agg_backend,),
        dp_accountants=(args.dp_accountant,),
        max_model_latent_dim=int(cfg.model.latent_dim),
        resumable=True,
        private_data_mounts=True,
    )
    participants = tuple(
        Phase3ParticipantDeclaration(
            participant_id=meta.participant_id,
            role="trainer",
            contact=Phase3Contact(
                owner=f"Phase 3 trust domain {idx}",
                contact=f"phase3-consortium-{idx}@example.invalid",
            ),
            action_contract=action,
            observation_contract=observation,
            accepted_probe_hash=probe.content_hash,
            accepted_probe_version=probe.version,
            capabilities=capabilities,
            data=Phase3DataDeclaration(
                data_ref=meta.data_ref,
                format=args.data_format,
                smoke_report_uri=f"artifact://{args.consortium_id}/{meta.participant_id}/dataset_smoke.json",
                smoke_report_sha256=_smoke_report_sha256(meta),
                window_steps=int(args.window_steps),
                heldout_policy="disjoint held-out eval split mounted separately for residency-safe metrics",
                license="participant-declared",
                raw_data_crosses_boundary=False,
            ),
        )
        for idx, meta in enumerate(metas, start=1)
    )
    return Phase3ConsortiumManifest(
        consortium_id=args.consortium_id,
        run_id=args.run_id,
        coordinator_id=f"{args.consortium_id}-coordinator",
        created_at=datetime.now(timezone.utc),
        model=Phase3ModelAgreement(
            model_family="LeWorldModel-phase3-consortium",
            wmcp_version=wmcp_version,
            latent_dim=int(cfg.model.latent_dim),
            num_tokens=int(cfg.model.num_tokens),
            objective_target_stop_gradient=bool(cfg.objective.target_stop_gradient),
            lambda_anc=float(cfg.objective.lambda_anc),
            base_checkpoint_ref=None,
            config_hash=config_hash(asdict(cfg)),
        ),
        public_probe=probe,
        runtime=Phase3RuntimePolicy(
            transport="in_process",
            secure_aggregation_backend=args.secure_agg_backend,
            secure_aggregation_required=True,
            dp_required=bool(cfg.privacy.enabled),
            min_trainers=int(args.min_trainers),
            dropout_retry_budget=0,
        ),
        dp_policy=Phase3DPPolicy(
            enabled=bool(cfg.privacy.enabled),
            clip_norm=float(cfg.privacy.clip_norm),
            noise_multiplier=float(cfg.privacy.noise_multiplier),
            epsilon=float(cfg.privacy.epsilon),
            delta=float(cfg.privacy.delta),
            accountant=cfg.privacy.accountant,
        ),
        accepted_action_contracts=(action,),
        accepted_observation_contracts=(observation,),
        participants=participants,
        claim_boundary=_CLAIM_BOUNDARY,
    )


def _build_probe(args: argparse.Namespace, cfg: LensembleConfig) -> PublicProbe:
    probe_points = args.probe_points or args.latent_dim
    if probe_points < args.latent_dim:
        raise ValueError("--probe-points must be >= --latent-dim for the frame anchor")
    points = torch.randn(
        probe_points,
        args.num_frames,
        3,
        args.image_size,
        args.image_size,
        generator=torch.Generator().manual_seed(20260608),
    )
    return build_probe(
        points,
        torch.arange(probe_points),
        snapshot_reference(build_encoder(cfg)),
        probe_version=1,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _artifact_targets(args: argparse.Namespace) -> Phase3ArtifactTargets:
    return Phase3ArtifactTargets(
        model_repo=args.out_repo or f"hf://models/local/{args.consortium_id}",
        dataset_repo=f"hf://datasets/local/{args.consortium_id}-data",
        reports_prefix="reports/phase3/",
        publication_mode="hf_jobs_release",
    )


def _eval_windows(args: argparse.Namespace) -> tuple[Any, ...]:
    if args.heldout_source is None:
        raise ValueError("--heldout-source is required for a real (non --dry-run) run")
    dataset = load_episodes(args.heldout_source, fmt=args.data_format)
    windows = tuple(dataset.windows(args.window_steps))
    if not windows:
        raise ValueError(
            f"held-out source {args.heldout_source!r} produced zero windows "
            f"for window_steps={args.window_steps}"
        )
    return windows


def _dry_run(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    cfg: LensembleConfig,
    manifest: Phase3ConsortiumManifest,
    metas: list[_SiloMetadata],
    probe: PublicProbe,
    probe_path: Path,
    manifest_path: Path,
    registry_path: Path,
) -> dict[str, Any]:
    registry = phase3_registry_from_consortium_manifest(
        manifest,
        min_participant_count=len(metas),
        window_counts={meta.participant_id: meta.window_count for meta in metas},
    )
    validate_phase3_registry_against_manifest(registry, manifest)
    save_probe(probe, probe_path)
    _write_json(manifest_path, manifest.model_dump(mode="json"))
    write_phase3_dataset_registry(registry, registry_path)

    transport = InProcessTransport()
    coordinator_endpoint = f"in-process://{manifest.coordinator_id}"
    checks = [
        "manifest_validated",
        "dataset_registry_validated_against_manifest",
        f"public_probe_hash_pinned:{probe.content_hash.hex()}",
    ]
    for meta in metas:
        agent = Phase3ParticipantAgent(
            _participant_cfg(cfg, args, data_source=meta.data_ref),
            manifest=manifest,
            participant_id=meta.participant_id,
            transport=transport,
            state_dir=out_dir / "participant-agents-dry-run",
            coordinator_endpoint=coordinator_endpoint,
            registry=registry,
            participant_factory=None,
            emit_observability=False,
        )
        agent.preflight()
        checks.append(f"participant_preflighted:{meta.participant_id}")

    payload: dict[str, Any] = {
        "dry_run": True,
        "consortium_id": manifest.consortium_id,
        "run_id": manifest.run_id,
        "participant_ids": [meta.participant_id for meta in metas],
        "window_counts": {meta.participant_id: meta.window_count for meta in metas},
        "public_probe_hash": probe.content_hash.hex(),
        "manifest_path": str(manifest_path),
        "registry_path": str(registry_path),
        "checks": checks,
    }
    _write_json(out_dir / "phase3_consortium_dry_run.json", payload)
    return payload


def _real_run(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    cfg: LensembleConfig,
    manifest: Phase3ConsortiumManifest,
    metas: list[_SiloMetadata],
    probe: PublicProbe,
    probe_path: Path,
    manifest_path: Path,
    registry_path: Path,
) -> Phase3LongRunReport:
    registry = phase3_registry_from_consortium_manifest(
        manifest,
        min_participant_count=len(metas),
        window_counts={meta.participant_id: meta.window_count for meta in metas},
    )
    validate_phase3_registry_against_manifest(registry, manifest)
    _write_json(manifest_path, manifest.model_dump(mode="json"))
    write_phase3_dataset_registry(registry, registry_path)

    eval_windows = _eval_windows(args)
    inputs = Phase3ConsortiumInputs(
        cfg=cfg,
        participant_configs={
            meta.participant_id: _participant_cfg(cfg, args, data_source=meta.data_ref)
            for meta in metas
        },
        manifest=manifest,
        registry=registry,
        probe=probe,
        probe_path=probe_path,
        participant_ids=tuple(meta.participant_id for meta in metas),
        eval_windows=eval_windows,
        eval_action_spec=metas[0].action_spec,
        participant_factory=None,
    )
    report = run_phase3_consortium(
        inputs,
        run_dir=out_dir,
        rounds=args.num_rounds,
        generated_at=datetime.now(timezone.utc),
        metric_windows=args.metric_windows,
        compute_metrics=True,
        artifact_targets=_artifact_targets(args),
        eval_budget=_EVAL_BUDGET,
        claim_boundary=_CLAIM_BOUNDARY,
    )
    write_phase3_long_run_report(report, out_dir / "phase3_long_run_smoke_report.json")
    return report


def _push(args: argparse.Namespace, out_dir: Path) -> tuple[bool, str | None]:
    if not args.push:
        return False, None
    token = os.environ.get("HF_TOKEN")
    if not token:
        return False, "HF_TOKEN is not set; skipped push"
    if not args.out_repo:
        return False, "--out-repo is not set; skipped push"

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        args.out_repo,
        repo_type="model",
        private=not args.public,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=args.out_repo,
        repo_type="model",
    )
    return True, None


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    participant_ids = _participant_ids(args)

    metas = [
        _load_silo_metadata(args, participant_id=pid, data_source=source)
        for pid, source in zip(participant_ids, args.data_source)
    ]

    probe_path = (
        Path(args.probe_path)
        if args.probe_path is not None
        else out_dir / "phase3_public_probe.safetensors"
    )
    cfg = _coordinator_cfg(args, probe_path=probe_path, participant_count=len(metas))
    probe = _build_probe(args, cfg)
    manifest = _build_manifest(
        args, cfg, metas=metas, public_probe_hash=probe.content_hash.hex()
    )
    manifest_path = out_dir / "phase3_consortium_manifest.json"
    registry_path = out_dir / "phase3_dataset_probe_registry.json"

    if args.dry_run:
        payload = _dry_run(
            args,
            out_dir=out_dir,
            cfg=cfg,
            manifest=manifest,
            metas=metas,
            probe=probe,
            probe_path=probe_path,
            manifest_path=manifest_path,
            registry_path=registry_path,
        )
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return payload

    report = _real_run(
        args,
        out_dir=out_dir,
        cfg=cfg,
        manifest=manifest,
        metas=metas,
        probe=probe,
        probe_path=probe_path,
        manifest_path=manifest_path,
        registry_path=registry_path,
    )
    pushed, blocker = _push(args, out_dir)
    summary = {
        "dry_run": False,
        "consortium_id": report.consortium_id,
        "run_id": report.run_id,
        "closed_rounds": report.closed_rounds,
        "target_rounds": report.target_rounds,
        "completed_target": report.completed_target,
        "final_global_model_hash": report.final_global_model_hash,
        "config_hash": report.config_hash,
        "report_path": str(out_dir / "phase3_long_run_smoke_report.json"),
        "manifest_path": str(manifest_path),
        "registry_path": str(registry_path),
        "pushed": pushed,
        "push_blocker": blocker,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    main()
