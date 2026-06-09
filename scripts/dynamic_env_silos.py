#!/usr/bin/env python
"""Generate RFC-0017 dynamic-env silo metadata.

The generated registry uses ``publication_status="placeholder"`` for each participant: the episodes are
reproducible from seed via ``synthetic-dynamic://`` URIs, not published raw bytes. This is the honest
registry path until a later run materializes and publishes datasets to HF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from lensemble.config import (
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
    write_consortium_manifest,
)
from lensemble.data import (
    phase3_registry_from_consortium_manifest,
    validate_phase3_registry_against_manifest,
    write_phase3_dataset_registry,
)

_WMCP_VERSION = "wmcp-1.0.0"
_CONSORTIUM_ID = "lensemble-dynamic-env-consortium"
_RUN_ID = "dynamic-env-swipe-dot-seeded"
_PROBE_HASH = "7" * 64


def _uri(
    *, seed: int, steps: int, image_size: int, episode_count: int, step_scale: float
) -> str:
    return (
        "synthetic-dynamic://swipe-dot"
        f"?seed={seed}&n_episodes={episode_count}&steps={steps}&image_size={image_size}"
        f"&step_scale={step_scale:.5f}"
    )


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_dynamic_env_manifest(
    *,
    num_silos: int = 4,
    seed: int = 1000,
    steps: int = 64,
    image_size: int = 48,
    window_steps: int = 4,
    episode_count: int = 8,
) -> tuple[Phase3ConsortiumManifest, str]:
    """Build a dynamic-env consortium manifest plus a disjoint held-out source URI."""

    if num_silos <= 0:
        raise ValueError("num_silos must be positive")
    if steps < window_steps:
        raise ValueError("steps must be >= window_steps")
    action = Phase3ActionContract(
        contract_id="swipe-dot-2dof-continuous-v1",
        embodiment_id="swipe-dot-2dof",
        kind="continuous",
        dim=2,
        low=(-1.0, -1.0),
        high=(1.0, 1.0),
        num_classes=None,
        units=("u", "u"),
        wmcp_version=_WMCP_VERSION,
    )
    observation = Phase3ObservationContract(
        contract_id="swipe-dot-rgb-window-v1",
        shape=(window_steps + 1, 1, 3, image_size, image_size),
        dtype="float32",
        frame_skip=1,
        wmcp_version=_WMCP_VERSION,
    )
    probe = Phase3PublicProbe(
        probe_id="dynamic-env-public-probe-smoke",
        version=1,
        content_hash=_PROBE_HASH,
    )
    participants = []
    for idx in range(num_silos):
        participant_seed = seed + idx
        step_scale = 0.10 + 0.01 * idx
        data_ref = _uri(
            seed=participant_seed,
            steps=steps,
            image_size=image_size,
            episode_count=episode_count,
            step_scale=step_scale,
        )
        participant_id = f"dynamic-swipe-dot-silo{idx}"
        participants.append(
            Phase3ParticipantDeclaration(
                participant_id=participant_id,
                role="trainer",
                contact=Phase3Contact(
                    owner=f"Dynamic env synthetic trust domain {idx}",
                    contact=f"dynamic-env-silo{idx}@example.invalid",
                ),
                action_contract=action,
                observation_contract=observation,
                accepted_probe_hash=probe.content_hash,
                accepted_probe_version=probe.version,
                capabilities=Phase3ParticipantCapabilities(
                    network_transport=True,
                    secure_aggregation_backends=("masking", "simulated"),
                    dp_accountants=("rdp",),
                    max_model_latent_dim=128,
                    resumable=True,
                    private_data_mounts=False,
                ),
                data=Phase3DataDeclaration(
                    data_ref=data_ref,
                    format="synthetic-dynamic",
                    smoke_report_uri=f"artifact://local/dynamic-env/{participant_id}/dataset_smoke.json",
                    smoke_report_sha256=_sha(data_ref),
                    window_steps=window_steps,
                    heldout_policy="disjoint held-out synthetic-dynamic seed recorded separately",
                    license="generated; reproducible from seed; no raw bytes published in this registry",
                    raw_data_crosses_boundary=False,
                ),
            )
        )
    heldout_source = _uri(
        seed=seed + 10_000,
        steps=steps,
        image_size=image_size,
        episode_count=episode_count,
        step_scale=0.12,
    )
    manifest = Phase3ConsortiumManifest(
        consortium_id=_CONSORTIUM_ID,
        run_id=_RUN_ID,
        coordinator_id="dynamic-env-coordinator",
        created_at=datetime(2026, 6, 9, tzinfo=timezone.utc),
        model=Phase3ModelAgreement(
            model_family="LeWorldModel-dynamic-env-scratch",
            wmcp_version=_WMCP_VERSION,
            latent_dim=128,
            num_tokens=(1 // 1) * (image_size // 16) ** 2,
            objective_target_stop_gradient=False,
            lambda_anc=0.01,
            base_checkpoint_ref=None,
            config_hash=None,
        ),
        public_probe=probe,
        runtime=Phase3RuntimePolicy(
            transport="network",
            secure_aggregation_backend="masking",
            secure_aggregation_required=True,
            dp_required=True,
            min_trainers=min(3, num_silos),
            dropout_retry_budget=1,
        ),
        dp_policy=Phase3DPPolicy(
            enabled=True,
            clip_norm=0.5,
            noise_multiplier=1.0,
            epsilon=8.0,
            delta=1e-5,
            accountant="rdp",
        ),
        accepted_action_contracts=(action,),
        accepted_observation_contracts=(observation,),
        participants=tuple(participants),
        claim_boundary=(
            "Dynamic-env synthetic-control run agreement. Silo data are reproducible from seed and "
            "placeholder-registered until materialized HF dataset bytes are published; no raw resident "
            "(x,y), observations, or actions cross this registry."
        ),
    )
    return manifest, heldout_source


def build_dynamic_env_registry(
    manifest: Phase3ConsortiumManifest,
):
    """Build a placeholder dataset registry from the dynamic-env manifest."""

    registry = phase3_registry_from_consortium_manifest(
        manifest,
        run_mode="public_example",
        min_participant_count=len(manifest.participants),
        min_windows_per_participant=1,
        window_counts={
            participant.participant_id: 8 * 61 for participant in manifest.participants
        },
        episode_counts={
            participant.participant_id: 8 for participant in manifest.participants
        },
    )
    validate_phase3_registry_against_manifest(registry, manifest)
    return registry


def _args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dynamic-env seeded silo registry metadata."
    )
    parser.add_argument("--num-silos", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--window-steps", type=int, default=4)
    parser.add_argument("--episode-count", type=int, default=8)
    parser.add_argument(
        "--manifest-output",
        default="docs/evidence/dynamic_env_consortium_manifest.json",
    )
    parser.add_argument(
        "--registry-output", default="docs/evidence/dynamic_env_dataset_registry.json"
    )
    parser.add_argument(
        "--plan-output", default="docs/evidence/dynamic_env_silo_plan.json"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = _args(argv)
    manifest, heldout_source = build_dynamic_env_manifest(
        num_silos=args.num_silos,
        seed=args.seed,
        steps=args.steps,
        image_size=args.image_size,
        window_steps=args.window_steps,
        episode_count=args.episode_count,
    )
    registry = build_dynamic_env_registry(manifest)
    manifest_path = write_consortium_manifest(manifest, Path(args.manifest_output))
    registry_path = write_phase3_dataset_registry(registry, Path(args.registry_output))
    plan = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "registry": str(registry_path),
        "heldout_source": heldout_source,
        "silo_sources": [
            participant.data.data_ref
            for participant in manifest.participants
            if participant.data is not None
        ],
        "honest_boundary": (
            "Registry is placeholder/reproducible-from-seed metadata, not published raw dataset bytes."
        ),
    }
    plan_path = Path(args.plan_output)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(plan, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(plan, sort_keys=True, indent=2), flush=True)
    return plan


if __name__ == "__main__":
    main()
