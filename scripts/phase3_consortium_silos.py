#!/usr/bin/env python3
"""Generate the published Phase 3 consortium manifest + dataset/probe registry (#242).

Consumes the residency-safe dataset smoke report for the published SO-100 participant silos and the
held-out split, derives the agreed action/observation contracts and the deterministic public-probe pin,
builds the consortium manifest bound to the immutable ``hf://`` dataset refs, and regenerates the Phase 3
dataset/probe registry so every participant is ``published`` (zero placeholders) with its real window and
episode counts. The registry is validated against the manifest before either is written.

The public-probe content hash depends only on ``(seed, probe_points, num_frames, image_size)`` — not on
the model weights — so the launcher (`deploy/hfjobs/train_phase3_consortium.py`, same seed ``20260608``)
reproduces the identical pin at run time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

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
)
from lensemble.data.phase3 import (
    phase3_registry_from_consortium_manifest,
    validate_phase3_registry_against_manifest,
    write_phase3_dataset_registry,
)
from lensemble.data.probe import probe_content_hash

# --- the pinned run identity + shape (shared with deploy/hfjobs/train_phase3_consortium.py) --------- #
_CONSORTIUM_ID = "lensemble-phase3-consortium"
_RUN_ID = "phase3-consortium-v1"
_COORDINATOR_ID = "lensemble-phase3-consortium-coordinator"
_SILO_REPO = "abdelstark/lensemble-phase3-so100-silos"
_WMCP_VERSION = "wmcp-1.0.0"
_WINDOW_STEPS = 4
_LATENT_DIM = 256
_NUM_TOKENS = 196  # (image_size/patch_size)^2 = (224/16)^2 for num_frames=1, tubelet=1
_PROBE_POINTS = 512
_PROBE_SEED = 20260608
_IMAGE_SIZE = 224
_NUM_FRAMES = 1
_LICENSE = "Apache-2.0 (derived from abdelstark/so100-pickplace-lewm-ready)"
_PARTICIPANT_SILOS = (
    ("phase3-so100-a", "phase3-so100-silo0.h5"),
    ("phase3-so100-b", "phase3-so100-silo1.h5"),
    ("phase3-so100-c", "phase3-so100-silo2.h5"),
    ("phase3-so100-d", "phase3-so100-silo3.h5"),
)
_HELDOUT = ("phase3-so100-heldout", "phase3-so100-silo4.h5")
_HELDOUT_POLICY = (
    "Held-out eval split phase3-so100-heldout (silo4) is mounted separately and is disjoint from every "
    "participant silo by deterministic episode-modulo (k % 5) assignment."
)
_CLAIM_BOUNDARY = (
    "Dataset/probe registry for the real Phase 3 SO-100 consortium training run: published, immutable "
    "participant silos + a disjoint held-out eval split. Not a raw-data publication artifact, provenance "
    "ledger, or cryptographic proof."
)


def _hf_ref(filename: str) -> str:
    return f"hf://datasets/{_SILO_REPO}/{filename}"


def _public_probe() -> tuple[Phase3PublicProbe, dict[str, Any]]:
    gen = torch.Generator().manual_seed(_PROBE_SEED)
    points = torch.randn(
        _PROBE_POINTS, _NUM_FRAMES, 3, _IMAGE_SIZE, _IMAGE_SIZE, generator=gen
    )
    landmark_idx = torch.arange(_PROBE_POINTS)
    content_hash = probe_content_hash(points, landmark_idx).hex()
    pin = Phase3PublicProbe(
        probe_id=f"{_CONSORTIUM_ID}-public-probe", version=1, content_hash=content_hash
    )
    spec = {
        "probe_id": pin.probe_id,
        "version": pin.version,
        "content_hash": content_hash,
        "probe_points": _PROBE_POINTS,
        "num_frames": _NUM_FRAMES,
        "image_size": _IMAGE_SIZE,
        "seed": _PROBE_SEED,
        "reproduction": (
            "points = torch.randn(probe_points, num_frames, 3, image_size, image_size, "
            "generator=torch.Generator().manual_seed(seed)); "
            "content_hash = probe_content_hash(points, torch.arange(probe_points)).hex()"
        ),
    }
    return pin, spec


def _silo_meta(smoke: dict[str, Any], participant_id: str) -> dict[str, Any]:
    for silo in smoke["silos"]:
        if silo["participant_id"] == participant_id:
            return silo
    raise KeyError(f"{participant_id!r} missing from the dataset smoke report")


def _action_contract(action_spec: dict[str, Any]) -> Phase3ActionContract:
    return Phase3ActionContract(
        contract_id="phase3-consortium-so100-action-v1",
        embodiment_id=action_spec["embodiment_id"],
        kind=action_spec["kind"],
        dim=action_spec["dim"],
        low=tuple(action_spec["low"]) if action_spec["low"] is not None else None,
        high=tuple(action_spec["high"]) if action_spec["high"] is not None else None,
        num_classes=(
            tuple(action_spec["num_classes"])
            if action_spec["num_classes"] is not None
            else None
        ),
        units=tuple(action_spec["units"]),
        wmcp_version=action_spec["wmcp_version"],
    )


def _observation_contract(observation_shape: list[int]) -> Phase3ObservationContract:
    return Phase3ObservationContract(
        contract_id="phase3-consortium-so100-window-v1",
        shape=tuple(int(x) for x in observation_shape),
        dtype="float32",
        frame_skip=1,
        wmcp_version=_WMCP_VERSION,
    )


def _participant_smoke_report(meta: dict[str, Any], *, data_ref: str) -> dict[str, Any]:
    """A residency-safe per-participant smoke report (metadata only; the dataset Merkle root, no arrays)."""

    return {
        "schema_version": 1,
        "participant_id": meta["participant_id"],
        "data_ref": data_ref,
        "data_format": "lerobot-h5",
        "window_steps": _WINDOW_STEPS,
        "window_count": meta["window_count"],
        "episode_count": meta["episode_count"],
        "dataset_root": meta["dataset_root"],
        "dataset_commitment_schema_version": meta["dataset_commitment_schema_version"],
        "hash_algorithm": meta["hash_algorithm"],
        "wmcp_version": meta["wmcp_version"],
        "embodiment_ids": meta["embodiment_ids"],
        "observation_shape": meta["observation_shape"],
        "action_shape": meta["action_shape"],
    }


def _sha256_canonical(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _build_manifest(
    smoke: dict[str, Any],
    *,
    probe: Phase3PublicProbe,
    participant_smoke_sha: dict[str, str],
) -> Phase3ConsortiumManifest:
    first = _silo_meta(smoke, _PARTICIPANT_SILOS[0][0])
    action = _action_contract(first["action_spec"])
    observation = _observation_contract(first["observation_shape"])
    capabilities = Phase3ParticipantCapabilities(
        network_transport=False,
        secure_aggregation_backends=("simulated",),
        dp_accountants=("rdp",),
        max_model_latent_dim=_LATENT_DIM,
        resumable=True,
        private_data_mounts=True,
    )
    participants = tuple(
        Phase3ParticipantDeclaration(
            participant_id=participant_id,
            role="trainer",
            contact=Phase3Contact(
                owner=f"Phase 3 SO-100 trust domain {idx}",
                contact=f"phase3-consortium-{idx}@example.invalid",
            ),
            action_contract=action,
            observation_contract=observation,
            accepted_probe_hash=probe.content_hash,
            accepted_probe_version=probe.version,
            capabilities=capabilities,
            data=Phase3DataDeclaration(
                data_ref=_hf_ref(filename),
                format="lerobot-h5",
                smoke_report_uri=f"hf://datasets/{_SILO_REPO}/smoke/{participant_id}.json",
                smoke_report_sha256=participant_smoke_sha[participant_id],
                window_steps=_WINDOW_STEPS,
                heldout_policy=_HELDOUT_POLICY,
                license=_LICENSE,
                raw_data_crosses_boundary=False,
            ),
        )
        for idx, (participant_id, filename) in enumerate(_PARTICIPANT_SILOS, start=1)
    )
    return Phase3ConsortiumManifest(
        consortium_id=_CONSORTIUM_ID,
        run_id=_RUN_ID,
        coordinator_id=_COORDINATOR_ID,
        created_at=datetime(2026, 6, 8, 0, 0, 0, tzinfo=timezone.utc),
        model=Phase3ModelAgreement(
            model_family="LeWorldModel-phase3-consortium",
            wmcp_version=_WMCP_VERSION,
            latent_dim=_LATENT_DIM,
            num_tokens=_NUM_TOKENS,
            objective_target_stop_gradient=False,
            lambda_anc=0.01,
            base_checkpoint_ref=None,
            config_hash=None,
        ),
        public_probe=probe,
        runtime=Phase3RuntimePolicy(
            transport="in_process",
            secure_aggregation_backend="simulated",
            secure_aggregation_required=True,
            dp_required=True,
            min_trainers=3,
            dropout_retry_budget=0,
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
        participants=participants,
        claim_boundary=_CLAIM_BOUNDARY,
    )


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", type=Path, required=True, help="Combined dataset smoke report JSON."
    )
    parser.add_argument(
        "--registry-output",
        type=Path,
        default=Path("docs/evidence/phase3_long_run_dataset_registry.json"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("docs/evidence/phase3_consortium_manifest.json"),
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        default=None,
        help="Optional dir to also write per-participant smoke reports + probe spec for Hub upload.",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    smoke = json.loads(args.smoke.read_text(encoding="utf-8"))
    probe, probe_spec = _public_probe()

    participant_smoke: dict[str, dict[str, Any]] = {}
    participant_smoke_sha: dict[str, str] = {}
    for participant_id, filename in _PARTICIPANT_SILOS:
        meta = _silo_meta(smoke, participant_id)
        report = _participant_smoke_report(meta, data_ref=_hf_ref(filename))
        participant_smoke[participant_id] = report
        participant_smoke_sha[participant_id] = _sha256_canonical(report)

    manifest = _build_manifest(
        smoke, probe=probe, participant_smoke_sha=participant_smoke_sha
    )
    window_counts = {
        pid: _silo_meta(smoke, pid)["window_count"] for pid, _ in _PARTICIPANT_SILOS
    }
    episode_counts = {
        pid: _silo_meta(smoke, pid)["episode_count"] for pid, _ in _PARTICIPANT_SILOS
    }
    registry = phase3_registry_from_consortium_manifest(
        manifest,
        run_mode="public_example",
        min_participant_count=len(_PARTICIPANT_SILOS),
        min_windows_per_participant=1,
        window_counts=window_counts,
        episode_counts=episode_counts,
    )
    validate_phase3_registry_against_manifest(registry, manifest)

    published = sum(
        1 for p in registry.participants if p.publication_status == "published"
    )
    assert published == len(registry.participants), "all participants must be published"

    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_phase3_dataset_registry(registry, args.registry_output)

    if args.publish_dir is not None:
        publish = args.publish_dir
        (publish / "smoke").mkdir(parents=True, exist_ok=True)
        for participant_id, report in participant_smoke.items():
            (publish / "smoke" / f"{participant_id}.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        (publish / "phase3_public_probe_spec.json").write_text(
            json.dumps(probe_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    print(
        f"wrote {args.manifest_output} and {args.registry_output}: "
        f"{published}/{len(registry.participants)} participants published; "
        f"held-out ref {_hf_ref(_HELDOUT[1])}; probe {probe.content_hash[:16]}…"
    )


if __name__ == "__main__":
    main()
