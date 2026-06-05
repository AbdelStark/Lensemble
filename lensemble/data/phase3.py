"""Phase 3 dataset/public-probe registry.

The registry is the data-side admission artifact for consortium training. It
declares participant-local dataset smoke metadata, held-out policy, adapter
format, action/observation contracts, and the public-probe pin without making
raw trajectories portable. It is intentionally not a provenance ledger and not
a cryptographic proof surface.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lensemble.config.consortium import (
    DataFormatName,
    Phase3ActionContract,
    Phase3ConsortiumManifest,
    Phase3ObservationContract,
    Phase3ParticipantDeclaration,
    Phase3PublicProbe,
    validate_consortium_manifest,
)
from lensemble.errors import ConfigError, LensembleErrorCode, SchemaVersionMismatch

PHASE3_DATASET_REGISTRY_SCHEMA_VERSION = 1

RegistryRunMode = Literal["public_example", "private_consortium"]
RegistryPublicationStatus = Literal["published", "private", "placeholder"]


def _fail(key: str, value: object, expected: str, remediation: str) -> ConfigError:
    err = ConfigError(
        f"invalid Phase 3 dataset registry: {key}={value!r} ({expected})",
        code=LensembleErrorCode.CONFIG_INVALID,
        remediation=remediation,
    )
    err.key = key  # type: ignore[attr-defined]
    err.value = value  # type: ignore[attr-defined]
    err.expected = expected  # type: ignore[attr-defined]
    return err


def _same_contract(
    lhs: Phase3ActionContract | Phase3ObservationContract,
    rhs: Phase3ActionContract | Phase3ObservationContract,
) -> bool:
    return lhs.model_dump(mode="json") == rhs.model_dump(mode="json")


def _public_dataset_ref(value: str) -> bool:
    return value.startswith("hf://") or value.startswith(
        "https://huggingface.co/datasets/"
    )


def _raw_or_private_ref(value: str) -> bool:
    prefixes = (
        "/",
        "./",
        "../",
        "file://",
        "local://",
        "hdf5://",
        "lance://",
        "lerobot-h5://",
        "private://",
        "mount://",
    )
    return value.startswith(prefixes) or "://" not in value


class Phase3ProbeGovernance(BaseModel):
    """Governance rules for the pinned public probe in one registry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    probe_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_uri: str = Field(min_length=1)
    curator: str = Field(min_length=1)
    versioning_policy: str = Field(min_length=1)
    allowed_update_process: str = Field(min_length=1)
    model_card_implication: str = Field(min_length=1)


class Phase3DatasetParticipantDeclaration(BaseModel):
    """One participant's residency-safe dataset/probe registry declaration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    participant_id: str = Field(min_length=1)
    data_ref: str = Field(min_length=1)
    format: DataFormatName
    publication_status: RegistryPublicationStatus
    publication_blocker: str | None = Field(default=None, min_length=1)
    heldout_policy: str = Field(min_length=1)
    license: str = Field(min_length=1)
    window_steps: int = Field(ge=1)
    min_windows: int = Field(ge=0)
    window_count: int = Field(ge=0)
    episode_count: int = Field(ge=1)
    smoke_report_uri: str = Field(min_length=1)
    smoke_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    action_contract: Phase3ActionContract
    observation_contract: Phase3ObservationContract
    accepted_probe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    accepted_probe_version: int = Field(ge=1)
    raw_data_path_allowed: bool = False


class Phase3DatasetProbeRegistry(BaseModel):
    """Machine-readable Phase 3 dataset/probe registry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = PHASE3_DATASET_REGISTRY_SCHEMA_VERSION
    registry_id: str = Field(min_length=1)
    consortium_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    generated_at: datetime
    run_mode: RegistryRunMode
    min_participant_count: int = Field(default=4, ge=1)
    min_windows_per_participant: int = Field(default=1, ge=0)
    public_probe: Phase3PublicProbe
    probe_governance: Phase3ProbeGovernance
    accepted_action_contracts: tuple[Phase3ActionContract, ...] = Field(min_length=1)
    accepted_observation_contracts: tuple[Phase3ObservationContract, ...] = Field(
        min_length=1
    )
    participants: tuple[Phase3DatasetParticipantDeclaration, ...] = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)


def _contract_map(
    contracts: tuple[Phase3ActionContract, ...] | tuple[Phase3ObservationContract, ...],
    *,
    key: str,
) -> dict[str, Phase3ActionContract | Phase3ObservationContract]:
    mapping: dict[str, Phase3ActionContract | Phase3ObservationContract] = {}
    for contract in contracts:
        if contract.contract_id in mapping:
            raise _fail(
                key,
                contract.contract_id,
                "unique contract ids",
                "deduplicate accepted contract declarations in the registry",
            )
        mapping[contract.contract_id] = contract
    return mapping


def _participant_map(
    participants: tuple[Phase3DatasetParticipantDeclaration, ...],
) -> dict[str, Phase3DatasetParticipantDeclaration]:
    mapping: dict[str, Phase3DatasetParticipantDeclaration] = {}
    ids: list[str] = []
    for participant in participants:
        ids.append(participant.participant_id)
        if participant.participant_id in mapping:
            raise _fail(
                "participants.participant_id",
                ids,
                "unique participant ids",
                "deduplicate participant declarations in the dataset registry",
            )
        mapping[participant.participant_id] = participant
    return mapping


def _manifest_participant_map(
    participants: tuple[Phase3ParticipantDeclaration, ...],
) -> dict[str, Phase3ParticipantDeclaration]:
    return {participant.participant_id: participant for participant in participants}


def validate_phase3_dataset_registry(
    registry: Phase3DatasetProbeRegistry,
) -> None:
    """Validate registry-internal data/probe consistency."""

    if registry.schema_version != PHASE3_DATASET_REGISTRY_SCHEMA_VERSION:
        raise SchemaVersionMismatch(
            f"Phase 3 dataset registry schema_version {registry.schema_version!r} "
            f"exceeds reader max {PHASE3_DATASET_REGISTRY_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation="read with a build supporting this Phase 3 dataset registry schema",
        )
    if len(registry.participants) < registry.min_participant_count:
        raise _fail(
            "participants",
            len(registry.participants),
            f">= min_participant_count ({registry.min_participant_count})",
            "declare every participant dataset required by this consortium run",
        )
    if registry.probe_governance.probe_id != registry.public_probe.probe_id:
        raise _fail(
            "probe_governance.probe_id",
            registry.probe_governance.probe_id,
            f"== public_probe.probe_id ({registry.public_probe.probe_id})",
            "govern the same public probe declared in the registry",
        )
    if registry.probe_governance.version != registry.public_probe.version:
        raise _fail(
            "probe_governance.version",
            registry.probe_governance.version,
            f"== public_probe.version ({registry.public_probe.version})",
            "govern the same public probe version declared in the registry",
        )
    if registry.probe_governance.content_hash != registry.public_probe.content_hash:
        raise _fail(
            "probe_governance.content_hash",
            registry.probe_governance.content_hash,
            f"== public_probe.content_hash ({registry.public_probe.content_hash})",
            "pin the same public probe hash in governance metadata",
        )

    action_contracts = _contract_map(
        registry.accepted_action_contracts, key="accepted_action_contracts"
    )
    observation_contracts = _contract_map(
        registry.accepted_observation_contracts,
        key="accepted_observation_contracts",
    )
    for contract in registry.accepted_action_contracts:
        contract.check_shape()

    for participant in _participant_map(registry.participants).values():
        _validate_registry_participant(
            registry, participant, action_contracts, observation_contracts
        )


def _validate_registry_participant(
    registry: Phase3DatasetProbeRegistry,
    participant: Phase3DatasetParticipantDeclaration,
    action_contracts: dict[str, Phase3ActionContract | Phase3ObservationContract],
    observation_contracts: dict[str, Phase3ActionContract | Phase3ObservationContract],
) -> None:
    threshold = max(registry.min_windows_per_participant, participant.min_windows)
    if participant.window_count < threshold:
        raise _fail(
            f"participants.{participant.participant_id}.window_count",
            participant.window_count,
            f">= {threshold}",
            "publish a dataset smoke report with enough fixed-horizon windows, or mark the participant blocked",
        )
    if participant.publication_status in {"private", "placeholder"} and (
        participant.publication_blocker is None
    ):
        raise _fail(
            f"participants.{participant.participant_id}.publication_blocker",
            None,
            "a blocker for private or placeholder participant datasets",
            "record the exact publication blocker so public claims remain bounded",
        )
    if participant.publication_status == "published" and not _public_dataset_ref(
        participant.data_ref
    ):
        raise _fail(
            f"participants.{participant.participant_id}.data_ref",
            participant.data_ref,
            "an hf:// or Hugging Face dataset URL for published public examples",
            "use an immutable public dataset ref or mark the participant private/placeholder",
        )
    if (
        registry.run_mode == "public_example"
        and _raw_or_private_ref(participant.data_ref)
        and participant.publication_status != "placeholder"
    ):
        raise _fail(
            f"participants.{participant.participant_id}.data_ref",
            participant.data_ref,
            "no raw/private path in public_example mode unless declared as a placeholder",
            "replace the ref with hf://... or mark it as a placeholder with a blocker",
        )
    if (
        registry.run_mode == "private_consortium"
        and _raw_or_private_ref(participant.data_ref)
        and not participant.raw_data_path_allowed
    ):
        raise _fail(
            f"participants.{participant.participant_id}.raw_data_path_allowed",
            participant.raw_data_path_allowed,
            "True for raw/private refs in private_consortium mode",
            "use a governed private mount class or explicitly allow this local path",
        )
    if participant.accepted_probe_hash != registry.public_probe.content_hash:
        raise _fail(
            f"participants.{participant.participant_id}.accepted_probe_hash",
            participant.accepted_probe_hash,
            f"== public_probe.content_hash ({registry.public_probe.content_hash})",
            "pin the same public-probe hash in every participant registry row",
        )
    if participant.accepted_probe_version != registry.public_probe.version:
        raise _fail(
            f"participants.{participant.participant_id}.accepted_probe_version",
            participant.accepted_probe_version,
            f"== public_probe.version ({registry.public_probe.version})",
            "pin the same public-probe version in every participant registry row",
        )

    accepted_action = action_contracts.get(participant.action_contract.contract_id)
    if accepted_action is None or not _same_contract(
        participant.action_contract, accepted_action
    ):
        raise _fail(
            f"participants.{participant.participant_id}.action_contract",
            participant.action_contract.contract_id,
            "one of the registry accepted action contracts",
            "declare an action contract accepted by the dataset registry",
        )
    participant.action_contract.check_shape()
    accepted_observation = observation_contracts.get(
        participant.observation_contract.contract_id
    )
    if accepted_observation is None or not _same_contract(
        participant.observation_contract, accepted_observation
    ):
        raise _fail(
            f"participants.{participant.participant_id}.observation_contract",
            participant.observation_contract.contract_id,
            "one of the registry accepted observation contracts",
            "declare an observation contract accepted by the dataset registry",
        )


def validate_phase3_registry_against_manifest(
    registry: Phase3DatasetProbeRegistry,
    manifest: Phase3ConsortiumManifest,
) -> None:
    """Validate a dataset/probe registry against the consortium manifest."""

    validate_phase3_dataset_registry(registry)
    validate_consortium_manifest(manifest)
    if registry.consortium_id != manifest.consortium_id:
        raise _fail(
            "consortium_id",
            registry.consortium_id,
            f"== manifest.consortium_id ({manifest.consortium_id})",
            "use the dataset registry generated for this consortium",
        )
    if registry.run_id != manifest.run_id:
        raise _fail(
            "run_id",
            registry.run_id,
            f"== manifest.run_id ({manifest.run_id})",
            "use the dataset registry generated for this run",
        )
    if registry.public_probe != manifest.public_probe:
        raise _fail(
            "public_probe",
            registry.public_probe.model_dump(mode="json"),
            f"== manifest.public_probe ({manifest.public_probe.model_dump(mode='json')})",
            "use one shared public-probe pin across manifest and dataset registry",
        )
    _compare_contract_sets(
        registry.accepted_action_contracts,
        manifest.accepted_action_contracts,
        key="accepted_action_contracts",
    )
    _compare_contract_sets(
        registry.accepted_observation_contracts,
        manifest.accepted_observation_contracts,
        key="accepted_observation_contracts",
    )

    registry_participants = _participant_map(registry.participants)
    manifest_participants = _manifest_participant_map(manifest.participants)
    if set(registry_participants) != set(manifest_participants):
        raise _fail(
            "participants",
            sorted(registry_participants),
            f"exactly the manifest participant ids ({sorted(manifest_participants)})",
            "generate the registry from the same manifest the coordinator and agents use",
        )
    for participant_id, registry_participant in registry_participants.items():
        manifest_participant = manifest_participants[participant_id]
        _compare_registry_manifest_participant(
            registry_participant, manifest_participant
        )


def _compare_contract_sets(
    registry_contracts: tuple[Phase3ActionContract, ...]
    | tuple[Phase3ObservationContract, ...],
    manifest_contracts: tuple[Phase3ActionContract, ...]
    | tuple[Phase3ObservationContract, ...],
    *,
    key: str,
) -> None:
    registry_by_id = _contract_map(registry_contracts, key=key)
    manifest_by_id = _contract_map(manifest_contracts, key=key)
    if set(registry_by_id) != set(manifest_by_id):
        raise _fail(
            key,
            sorted(registry_by_id),
            f"exactly the manifest contract ids ({sorted(manifest_by_id)})",
            "generate the registry from the same manifest contract set",
        )
    for contract_id, registry_contract in registry_by_id.items():
        manifest_contract = manifest_by_id[contract_id]
        if not _same_contract(registry_contract, manifest_contract):
            raise _fail(
                f"{key}.{contract_id}",
                registry_contract.model_dump(mode="json"),
                "the matching manifest contract",
                "keep registry and manifest action/observation contracts identical",
            )


def _compare_registry_manifest_participant(
    registry_participant: Phase3DatasetParticipantDeclaration,
    manifest_participant: Phase3ParticipantDeclaration,
) -> None:
    data = manifest_participant.data
    if data is None:
        raise _fail(
            f"participants.{manifest_participant.participant_id}.data",
            None,
            "a manifest data declaration",
            "declare manifest data before attaching a dataset registry",
        )
    checks: tuple[tuple[str, object, object], ...] = (
        ("data_ref", registry_participant.data_ref, data.data_ref),
        ("format", registry_participant.format, data.format),
        (
            "smoke_report_uri",
            registry_participant.smoke_report_uri,
            data.smoke_report_uri,
        ),
        (
            "smoke_report_sha256",
            registry_participant.smoke_report_sha256,
            data.smoke_report_sha256,
        ),
        ("window_steps", registry_participant.window_steps, data.window_steps),
        ("heldout_policy", registry_participant.heldout_policy, data.heldout_policy),
        ("license", registry_participant.license, data.license),
        (
            "accepted_probe_hash",
            registry_participant.accepted_probe_hash,
            manifest_participant.accepted_probe_hash,
        ),
        (
            "accepted_probe_version",
            registry_participant.accepted_probe_version,
            manifest_participant.accepted_probe_version,
        ),
    )
    for field, got, expected in checks:
        if got != expected:
            raise _fail(
                f"participants.{registry_participant.participant_id}.{field}",
                got,
                f"== manifest value ({expected!r})",
                "generate the dataset registry from the same manifest participant declaration",
            )
    if not _same_contract(
        registry_participant.action_contract, manifest_participant.action_contract
    ):
        raise _fail(
            f"participants.{registry_participant.participant_id}.action_contract",
            registry_participant.action_contract.contract_id,
            "the matching manifest action contract",
            "keep registry and manifest participant action contracts identical",
        )
    if not _same_contract(
        registry_participant.observation_contract,
        manifest_participant.observation_contract,
    ):
        raise _fail(
            f"participants.{registry_participant.participant_id}.observation_contract",
            registry_participant.observation_contract.contract_id,
            "the matching manifest observation contract",
            "keep registry and manifest participant observation contracts identical",
        )


def validate_coordinator_registry_preflight(
    registry: Phase3DatasetProbeRegistry,
    manifest: Phase3ConsortiumManifest,
) -> Phase3DatasetProbeRegistry:
    """Coordinator-side preflight for a shared Phase 3 dataset/probe registry."""

    validate_phase3_registry_against_manifest(registry, manifest)
    return registry


def validate_participant_registry_preflight(
    registry: Phase3DatasetProbeRegistry,
    manifest: Phase3ConsortiumManifest,
    *,
    participant_id: str,
) -> Phase3DatasetParticipantDeclaration:
    """Participant-side preflight for the shared Phase 3 dataset/probe registry."""

    validate_phase3_registry_against_manifest(registry, manifest)
    participant = _participant_map(registry.participants).get(participant_id)
    if participant is None:
        raise _fail(
            "participant_id",
            participant_id,
            "one of the dataset registry participants",
            "join with a participant id declared in the shared dataset registry",
        )
    return participant


def parse_phase3_dataset_registry(raw: dict[str, Any]) -> Phase3DatasetProbeRegistry:
    """Parse raw registry JSON, gating future schema versions first."""

    version = raw.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version > PHASE3_DATASET_REGISTRY_SCHEMA_VERSION
    ):
        raise SchemaVersionMismatch(
            f"Phase 3 dataset registry schema_version {version!r} exceeds reader max "
            f"{PHASE3_DATASET_REGISTRY_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation="read with a build supporting this Phase 3 dataset registry schema",
        )
    registry = Phase3DatasetProbeRegistry.model_validate(raw)
    validate_phase3_dataset_registry(registry)
    return registry


def to_phase3_dataset_registry_json(registry: Phase3DatasetProbeRegistry) -> str:
    """Canonical JSON for a Phase 3 dataset/probe registry."""

    validate_phase3_dataset_registry(registry)
    return json.dumps(
        registry.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def write_phase3_dataset_registry(
    registry: Phase3DatasetProbeRegistry, path: Path
) -> Path:
    """Write a validated registry as canonical JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_phase3_dataset_registry_json(registry) + "\n", encoding="utf-8")
    return path


def load_phase3_dataset_registry(path: Path) -> Phase3DatasetProbeRegistry:
    """Load and validate a Phase 3 dataset/probe registry JSON file."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_phase3_dataset_registry(raw)


def phase3_registry_from_consortium_manifest(
    manifest: Phase3ConsortiumManifest,
    *,
    generated_at: datetime | None = None,
    registry_id: str | None = None,
    run_mode: RegistryRunMode = "public_example",
    min_participant_count: int = 4,
    min_windows_per_participant: int = 1,
    window_counts: dict[str, int] | None = None,
    episode_counts: dict[str, int] | None = None,
) -> Phase3DatasetProbeRegistry:
    """Generate a registry skeleton from a validated Phase 3 consortium manifest."""

    validate_consortium_manifest(manifest)
    counts = dict(window_counts or {})
    episodes = dict(episode_counts or {})
    participants = tuple(
        _registry_participant_from_manifest(
            participant,
            min_windows_per_participant=min_windows_per_participant,
            window_count=counts.get(participant.participant_id),
            episode_count=episodes.get(participant.participant_id),
        )
        for participant in manifest.participants
    )
    registry = Phase3DatasetProbeRegistry(
        registry_id=registry_id
        or f"{manifest.consortium_id}:{manifest.run_id}:dataset-probe-registry",
        consortium_id=manifest.consortium_id,
        run_id=manifest.run_id,
        generated_at=generated_at or manifest.created_at,
        run_mode=run_mode,
        min_participant_count=min_participant_count,
        min_windows_per_participant=min_windows_per_participant,
        public_probe=manifest.public_probe,
        probe_governance=Phase3ProbeGovernance(
            probe_id=manifest.public_probe.probe_id,
            version=manifest.public_probe.version,
            content_hash=manifest.public_probe.content_hash,
            artifact_uri=(
                f"artifact://phase3/public-probes/{manifest.public_probe.probe_id}"
                f"/v{manifest.public_probe.version}.safetensors"
            ),
            curator="phase3-consortium-operators",
            versioning_policy=(
                "Probe bytes are immutable for one run_id; any probe change requires "
                "a new version, new content hash, and model-card update."
            ),
            allowed_update_process=(
                "Submit a registry update PR that changes probe_id/version/hash, "
                "regenerates this registry, reruns participant and coordinator "
                "preflights, and states the claim-boundary impact."
            ),
            model_card_implication=(
                "Model cards must cite the exact probe hash and say whether all "
                "participant datasets are public or blocked/private."
            ),
        ),
        accepted_action_contracts=manifest.accepted_action_contracts,
        accepted_observation_contracts=manifest.accepted_observation_contracts,
        participants=participants,
        claim_boundary=(
            "Dataset/probe registry for Phase 3 operational consortium training; "
            "not a raw-data publication artifact, provenance ledger, or cryptographic proof."
        ),
    )
    validate_phase3_registry_against_manifest(registry, manifest)
    return registry


def _registry_participant_from_manifest(
    participant: Phase3ParticipantDeclaration,
    *,
    min_windows_per_participant: int,
    window_count: int | None,
    episode_count: int | None,
) -> Phase3DatasetParticipantDeclaration:
    data = participant.data
    if data is None:
        raise _fail(
            f"participants.{participant.participant_id}.data",
            None,
            "a data declaration",
            "generate dataset registries only from manifests with participant data declarations",
        )
    status: RegistryPublicationStatus = (
        "published" if _public_dataset_ref(data.data_ref) else "placeholder"
    )
    blocker = (
        None
        if status == "published"
        else (
            "placeholder dataset ref; replace with an immutable hf:// dataset ref "
            "or document the private consortium data blocker before launch"
        )
    )
    return Phase3DatasetParticipantDeclaration(
        participant_id=participant.participant_id,
        data_ref=data.data_ref,
        format=data.format,
        publication_status=status,
        publication_blocker=blocker,
        heldout_policy=data.heldout_policy,
        license=data.license,
        window_steps=data.window_steps,
        min_windows=min_windows_per_participant,
        window_count=window_count
        if window_count is not None
        else max(min_windows_per_participant, 1),
        episode_count=episode_count if episode_count is not None else 1,
        smoke_report_uri=data.smoke_report_uri,
        smoke_report_sha256=data.smoke_report_sha256,
        action_contract=participant.action_contract,
        observation_contract=participant.observation_contract,
        accepted_probe_hash=participant.accepted_probe_hash,
        accepted_probe_version=participant.accepted_probe_version,
        raw_data_path_allowed=False,
    )


__all__ = [
    "PHASE3_DATASET_REGISTRY_SCHEMA_VERSION",
    "Phase3DatasetParticipantDeclaration",
    "Phase3DatasetProbeRegistry",
    "Phase3ProbeGovernance",
    "RegistryPublicationStatus",
    "RegistryRunMode",
    "load_phase3_dataset_registry",
    "parse_phase3_dataset_registry",
    "phase3_registry_from_consortium_manifest",
    "to_phase3_dataset_registry_json",
    "validate_coordinator_registry_preflight",
    "validate_participant_registry_preflight",
    "validate_phase3_dataset_registry",
    "validate_phase3_registry_against_manifest",
    "write_phase3_dataset_registry",
]
