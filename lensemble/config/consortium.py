"""Phase 3 consortium manifest and shared run-agreement validation.

The consortium manifest is the operational admission contract for Phase 3: it
declares the participants, data/probe agreement, runtime capabilities, DP
policy, and claim boundary before a networked consortium run starts. It is a
governance/config artifact, not a provenance ledger and not a cryptographic
proof surface.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lensemble.errors import ConfigError, LensembleErrorCode, SchemaVersionMismatch

CONSORTIUM_MANIFEST_SCHEMA_VERSION = 1
PHASE3_CONSORTIUM_PROTOCOL_VERSION = "phase3-consortium-v1"

ParticipantRole = Literal["trainer", "evaluator", "observer"]
ActionKindName = Literal["continuous", "discrete"]
DataFormatName = Literal["lance", "hdf5", "lerobot", "lerobot-h5", "synthetic-dynamic"]
TransportMode = Literal["in_process", "network"]
SecureAggregationBackendName = Literal["simulated", "masking", "tee"]
DPAccountantName = Literal["rdp", "prv"]

_HEX64_LEN = 64


def _is_hex64(value: str) -> bool:
    return len(value) == _HEX64_LEN and all(c in "0123456789abcdef" for c in value)


def _fail(key: str, value: object, expected: str, remediation: str) -> ConfigError:
    err = ConfigError(
        f"invalid consortium manifest: {key}={value!r} ({expected})",
        code=LensembleErrorCode.CONFIG_INVALID,
        remediation=remediation,
    )
    err.key = key  # type: ignore[attr-defined]
    err.value = value  # type: ignore[attr-defined]
    err.expected = expected  # type: ignore[attr-defined]
    return err


class Phase3Contact(BaseModel):
    """Human-operational contact metadata for one consortium participant."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    owner: str = Field(min_length=1)
    contact: str = Field(min_length=1)


class Phase3ActionContract(BaseModel):
    """Accepted action contract declared at consortium join."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_id: str = Field(min_length=1)
    embodiment_id: str = Field(min_length=1)
    kind: ActionKindName
    dim: int = Field(ge=1)
    low: tuple[float, ...] | None = None
    high: tuple[float, ...] | None = None
    num_classes: tuple[int, ...] | None = None
    units: tuple[str, ...] = Field(min_length=1)
    wmcp_version: str = Field(min_length=1)

    @field_validator("units")
    @classmethod
    def _units_nonempty(cls, units: tuple[str, ...]) -> tuple[str, ...]:
        if any(not unit for unit in units):
            raise ValueError("action units must be non-empty")
        return units

    def check_shape(self) -> None:
        """Validate kind-specific action-space shape."""

        if len(self.units) != self.dim:
            raise _fail(
                "action_contract.units",
                self.units,
                f"len == dim ({self.dim})",
                "declare one unit per action dimension",
            )
        if self.kind == "continuous":
            if self.num_classes is not None:
                raise _fail(
                    "action_contract.num_classes",
                    self.num_classes,
                    "None for continuous action spaces",
                    "remove num_classes for continuous actions",
                )
            if self.low is None or self.high is None:
                raise _fail(
                    "action_contract.bounds",
                    (self.low, self.high),
                    "low/high bounds for continuous action spaces",
                    "declare continuous action bounds",
                )
            if len(self.low) != self.dim or len(self.high) != self.dim:
                raise _fail(
                    "action_contract.bounds",
                    (self.low, self.high),
                    f"len == dim ({self.dim})",
                    "declare one lower and upper bound per action dimension",
                )
            if any(lo >= hi for lo, hi in zip(self.low, self.high, strict=True)):
                raise _fail(
                    "action_contract.bounds",
                    (self.low, self.high),
                    "low_i < high_i for every action dimension",
                    "fix continuous action bounds",
                )
            return
        if self.low is not None or self.high is not None:
            raise _fail(
                "action_contract.bounds",
                (self.low, self.high),
                "None for discrete action spaces",
                "remove continuous bounds for discrete actions",
            )
        if self.num_classes is None or len(self.num_classes) != self.dim:
            raise _fail(
                "action_contract.num_classes",
                self.num_classes,
                f"len == dim ({self.dim}) for discrete action spaces",
                "declare one class count per discrete action dimension",
            )
        if any(n < 2 for n in self.num_classes):
            raise _fail(
                "action_contract.num_classes",
                self.num_classes,
                "every class count >= 2",
                "declare at least two classes per discrete action dimension",
            )


class Phase3ObservationContract(BaseModel):
    """Accepted observation/window contract declared at consortium join."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_id: str = Field(min_length=1)
    shape: tuple[int, ...] = Field(min_length=1)
    dtype: str = Field(min_length=1)
    frame_skip: int = Field(ge=1)
    wmcp_version: str = Field(min_length=1)

    @field_validator("shape")
    @classmethod
    def _shape_positive(cls, shape: tuple[int, ...]) -> tuple[int, ...]:
        if any(dim <= 0 for dim in shape):
            raise ValueError("observation shape dimensions must be positive")
        return shape


class Phase3PublicProbe(BaseModel):
    """Pinned public-probe agreement for frame anchoring."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    probe_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class Phase3ModelAgreement(BaseModel):
    """Model/objective agreement accepted by every participant."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_family: str = Field(min_length=1)
    wmcp_version: str = Field(min_length=1)
    latent_dim: int = Field(ge=1)
    num_tokens: int = Field(ge=1)
    objective_target_stop_gradient: bool
    lambda_anc: float = Field(ge=0.0)
    base_checkpoint_ref: str | None = Field(default=None, min_length=1)
    config_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class Phase3DPPolicy(BaseModel):
    """Operational DP policy for a consortium run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool
    clip_norm: float = Field(gt=0.0)
    noise_multiplier: float = Field(ge=0.0)
    epsilon: float = Field(gt=0.0)
    delta: float = Field(gt=0.0, lt=1.0)
    accountant: DPAccountantName


class Phase3RuntimePolicy(BaseModel):
    """Runtime admission policy shared by coordinator and participant agents."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol_version: str = PHASE3_CONSORTIUM_PROTOCOL_VERSION
    transport: TransportMode
    secure_aggregation_backend: SecureAggregationBackendName
    secure_aggregation_required: bool
    dp_required: bool
    min_trainers: int = Field(ge=1)
    dropout_retry_budget: int = Field(ge=0)


class Phase3ParticipantCapabilities(BaseModel):
    """Participant-declared runtime capabilities."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    network_transport: bool
    secure_aggregation_backends: tuple[SecureAggregationBackendName, ...] = Field(
        min_length=1
    )
    dp_accountants: tuple[DPAccountantName, ...] = Field(min_length=1)
    max_model_latent_dim: int = Field(ge=1)
    resumable: bool
    private_data_mounts: bool


class Phase3DataDeclaration(BaseModel):
    """Residency-safe participant-local data declaration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    data_ref: str = Field(min_length=1)
    format: DataFormatName
    smoke_report_uri: str = Field(min_length=1)
    smoke_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    window_steps: int = Field(ge=1)
    heldout_policy: str = Field(min_length=1)
    license: str = Field(min_length=1)
    raw_data_crosses_boundary: Literal[False] = False


class Phase3ParticipantDeclaration(BaseModel):
    """One participant's join declaration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    participant_id: str = Field(min_length=1)
    role: ParticipantRole
    contact: Phase3Contact
    action_contract: Phase3ActionContract
    observation_contract: Phase3ObservationContract
    accepted_probe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    accepted_probe_version: int = Field(ge=1)
    capabilities: Phase3ParticipantCapabilities
    data: Phase3DataDeclaration | None = None


class Phase3ConsortiumManifest(BaseModel):
    """Machine-readable run agreement for a Phase 3 consortium run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = CONSORTIUM_MANIFEST_SCHEMA_VERSION
    consortium_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    coordinator_id: str = Field(min_length=1)
    created_at: datetime
    model: Phase3ModelAgreement
    public_probe: Phase3PublicProbe
    runtime: Phase3RuntimePolicy
    dp_policy: Phase3DPPolicy
    accepted_action_contracts: tuple[Phase3ActionContract, ...] = Field(min_length=1)
    accepted_observation_contracts: tuple[Phase3ObservationContract, ...] = Field(
        min_length=1
    )
    participants: tuple[Phase3ParticipantDeclaration, ...] = Field(min_length=1)
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
                "deduplicate accepted contract declarations",
            )
        mapping[contract.contract_id] = contract
    return mapping


def _same_contract(
    lhs: Phase3ActionContract | Phase3ObservationContract,
    rhs: Phase3ActionContract | Phase3ObservationContract,
) -> bool:
    return lhs.model_dump(mode="json") == rhs.model_dump(mode="json")


def validate_consortium_manifest(manifest: Phase3ConsortiumManifest) -> None:
    """Validate cross-field run-agreement rules shared by every Phase 3 actor."""

    if manifest.schema_version != CONSORTIUM_MANIFEST_SCHEMA_VERSION:
        raise SchemaVersionMismatch(
            f"consortium manifest schema_version {manifest.schema_version!r} "
            f"exceeds reader max {CONSORTIUM_MANIFEST_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation="read with a build supporting this consortium manifest schema",
        )
    action_contracts = _contract_map(
        manifest.accepted_action_contracts, key="accepted_action_contracts"
    )
    observation_contracts = _contract_map(
        manifest.accepted_observation_contracts,
        key="accepted_observation_contracts",
    )
    for contract in manifest.accepted_action_contracts:
        contract.check_shape()
        if contract.wmcp_version != manifest.model.wmcp_version:
            raise _fail(
                "accepted_action_contracts.wmcp_version",
                contract.wmcp_version,
                f"== model.wmcp_version ({manifest.model.wmcp_version})",
                "all accepted action contracts must use the run WMCP version",
            )
    for contract in manifest.accepted_observation_contracts:
        if contract.wmcp_version != manifest.model.wmcp_version:
            raise _fail(
                "accepted_observation_contracts.wmcp_version",
                contract.wmcp_version,
                f"== model.wmcp_version ({manifest.model.wmcp_version})",
                "all accepted observation contracts must use the run WMCP version",
            )
    if manifest.runtime.protocol_version != PHASE3_CONSORTIUM_PROTOCOL_VERSION:
        raise _fail(
            "runtime.protocol_version",
            manifest.runtime.protocol_version,
            PHASE3_CONSORTIUM_PROTOCOL_VERSION,
            "use the supported Phase 3 consortium protocol version",
        )
    if manifest.runtime.dp_required and not manifest.dp_policy.enabled:
        raise _fail(
            "dp_policy.enabled",
            manifest.dp_policy.enabled,
            "True when runtime.dp_required is true",
            "enable DP or mark the run as not requiring DP",
        )
    participant_ids = [p.participant_id for p in manifest.participants]
    if len(set(participant_ids)) != len(participant_ids):
        raise _fail(
            "participants.participant_id",
            participant_ids,
            "unique participant ids",
            "assign each consortium participant a stable unique id",
        )
    trainers = [p for p in manifest.participants if p.role == "trainer"]
    if len(trainers) < manifest.runtime.min_trainers:
        raise _fail(
            "runtime.min_trainers",
            manifest.runtime.min_trainers,
            f"<= trainer count ({len(trainers)})",
            "lower min_trainers or add trainer participants",
        )
    for participant in manifest.participants:
        _validate_participant(
            manifest, participant, action_contracts, observation_contracts
        )


def _validate_participant(
    manifest: Phase3ConsortiumManifest,
    participant: Phase3ParticipantDeclaration,
    action_contracts: dict[str, Phase3ActionContract | Phase3ObservationContract],
    observation_contracts: dict[str, Phase3ActionContract | Phase3ObservationContract],
) -> None:
    if participant.data is None:
        raise _fail(
            f"participants.{participant.participant_id}.data",
            None,
            "a data declaration",
            "declare the participant-local data ref and smoke-report metadata",
        )
    if participant.accepted_probe_hash != manifest.public_probe.content_hash:
        raise _fail(
            f"participants.{participant.participant_id}.accepted_probe_hash",
            participant.accepted_probe_hash,
            f"== public_probe.content_hash ({manifest.public_probe.content_hash})",
            "pin the same public probe hash before joining the run",
        )
    if participant.accepted_probe_version != manifest.public_probe.version:
        raise _fail(
            f"participants.{participant.participant_id}.accepted_probe_version",
            participant.accepted_probe_version,
            f"== public_probe.version ({manifest.public_probe.version})",
            "pin the same public probe version before joining the run",
        )
    if participant.action_contract.wmcp_version != manifest.model.wmcp_version:
        raise _fail(
            f"participants.{participant.participant_id}.action_contract.wmcp_version",
            participant.action_contract.wmcp_version,
            f"== model.wmcp_version ({manifest.model.wmcp_version})",
            "join with the consortium WMCP version",
        )
    participant.action_contract.check_shape()
    accepted_action = action_contracts.get(participant.action_contract.contract_id)
    if accepted_action is None or not _same_contract(
        participant.action_contract, accepted_action
    ):
        raise _fail(
            f"participants.{participant.participant_id}.action_contract",
            participant.action_contract.contract_id,
            "one of the accepted action contracts",
            "declare an action contract accepted by the consortium manifest",
        )
    if participant.observation_contract.wmcp_version != manifest.model.wmcp_version:
        raise _fail(
            f"participants.{participant.participant_id}.observation_contract.wmcp_version",
            participant.observation_contract.wmcp_version,
            f"== model.wmcp_version ({manifest.model.wmcp_version})",
            "join with the consortium WMCP version",
        )
    accepted_observation = observation_contracts.get(
        participant.observation_contract.contract_id
    )
    if accepted_observation is None or not _same_contract(
        participant.observation_contract, accepted_observation
    ):
        raise _fail(
            f"participants.{participant.participant_id}.observation_contract",
            participant.observation_contract.contract_id,
            "one of the accepted observation contracts",
            "declare an observation contract accepted by the consortium manifest",
        )
    if (
        manifest.runtime.transport == "network"
        and not participant.capabilities.network_transport
    ):
        raise _fail(
            f"participants.{participant.participant_id}.capabilities.network_transport",
            participant.capabilities.network_transport,
            "True for network transport runs",
            "enable network transport capability or use in_process for a local smoke",
        )
    if (
        manifest.runtime.secure_aggregation_required
        and manifest.runtime.secure_aggregation_backend
        not in participant.capabilities.secure_aggregation_backends
    ):
        raise _fail(
            f"participants.{participant.participant_id}.capabilities.secure_aggregation_backends",
            participant.capabilities.secure_aggregation_backends,
            f"contains {manifest.runtime.secure_aggregation_backend!r}",
            "join only with a secure-aggregation backend every participant supports",
        )
    if (
        manifest.dp_policy.enabled
        and manifest.dp_policy.accountant not in participant.capabilities.dp_accountants
    ):
        raise _fail(
            f"participants.{participant.participant_id}.capabilities.dp_accountants",
            participant.capabilities.dp_accountants,
            f"contains {manifest.dp_policy.accountant!r}",
            "join only with a DP accountant every participant supports",
        )
    if manifest.model.latent_dim > participant.capabilities.max_model_latent_dim:
        raise _fail(
            f"participants.{participant.participant_id}.capabilities.max_model_latent_dim",
            participant.capabilities.max_model_latent_dim,
            f">= model.latent_dim ({manifest.model.latent_dim})",
            "use a smaller model or a participant runtime with enough capacity",
        )
    if participant.data.raw_data_crosses_boundary is not False:
        raise _fail(
            f"participants.{participant.participant_id}.data.raw_data_crosses_boundary",
            participant.data.raw_data_crosses_boundary,
            "False",
            "Phase 3 consortium manifests may not permit raw data to cross boundaries",
        )


def parse_consortium_manifest(raw: dict[str, Any]) -> Phase3ConsortiumManifest:
    """Parse a raw manifest dict, gating schema version before field validation."""

    version = raw.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version > CONSORTIUM_MANIFEST_SCHEMA_VERSION
    ):
        raise SchemaVersionMismatch(
            f"consortium manifest schema_version {version!r} exceeds reader max "
            f"{CONSORTIUM_MANIFEST_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation="read with a build supporting this consortium manifest schema",
        )
    manifest = Phase3ConsortiumManifest.model_validate(raw)
    validate_consortium_manifest(manifest)
    return manifest


def to_consortium_json(manifest: Phase3ConsortiumManifest) -> str:
    """Canonical JSON for a consortium manifest."""

    validate_consortium_manifest(manifest)
    return json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def write_consortium_manifest(manifest: Phase3ConsortiumManifest, path: Path) -> Path:
    """Write a validated consortium manifest as canonical JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_consortium_json(manifest) + "\n", encoding="utf-8")
    return path


def load_consortium_manifest(path: Path) -> Phase3ConsortiumManifest:
    """Load and validate a consortium manifest JSON file."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_consortium_manifest(raw)


def validate_coordinator_run_agreement(
    manifest: Phase3ConsortiumManifest,
) -> Phase3ConsortiumManifest:
    """Coordinator-side admission gate for a Phase 3 run agreement."""

    validate_consortium_manifest(manifest)
    return manifest


def validate_participant_join(
    manifest: Phase3ConsortiumManifest, *, participant_id: str
) -> Phase3ParticipantDeclaration:
    """Participant-side join gate using the same manifest validator as the coordinator."""

    validate_consortium_manifest(manifest)
    for participant in manifest.participants:
        if participant.participant_id == participant_id:
            return participant
    raise _fail(
        "participant_id",
        participant_id,
        "one of the manifest participants",
        "join with a participant id declared in the consortium manifest",
    )


def default_phase3_consortium_manifest(
    *, created_at: datetime | None = None
) -> Phase3ConsortiumManifest:
    """Generate the four-participant Phase 3 example manifest."""

    wmcp_version = "wmcp-1.0.0"
    action_contract = Phase3ActionContract(
        contract_id="so100-6dof-continuous-v1",
        embodiment_id="so100-arm-6dof",
        kind="continuous",
        dim=6,
        low=(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
        high=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        num_classes=None,
        units=("unitless", "unitless", "unitless", "unitless", "unitless", "unitless"),
        wmcp_version=wmcp_version,
    )
    observation_contract = Phase3ObservationContract(
        contract_id="so100-top-camera-window-v1",
        shape=(5, 1, 3, 224, 224),
        dtype="float32",
        frame_skip=1,
        wmcp_version=wmcp_version,
    )
    probe = Phase3PublicProbe(
        probe_id="phase3-public-probe-smoke",
        version=1,
        content_hash="1" * 64,
    )
    participants = tuple(
        Phase3ParticipantDeclaration(
            participant_id=f"phase3-so100-{suffix}",
            role="trainer",
            contact=Phase3Contact(
                owner=f"Phase 3 simulated trust domain {suffix.upper()}",
                contact=f"phase3-{suffix}@example.invalid",
            ),
            action_contract=action_contract,
            observation_contract=observation_contract,
            accepted_probe_hash=probe.content_hash,
            accepted_probe_version=probe.version,
            capabilities=Phase3ParticipantCapabilities(
                network_transport=True,
                secure_aggregation_backends=("masking", "simulated"),
                dp_accountants=("rdp",),
                max_model_latent_dim=256,
                resumable=True,
                private_data_mounts=True,
            ),
            data=Phase3DataDeclaration(
                data_ref=f"lerobot-h5://phase3/simulated-trust-domain-{suffix}.h5",
                format="lerobot-h5",
                smoke_report_uri=f"artifact://phase3/examples/{suffix}/dataset_smoke.json",
                smoke_report_sha256=f"{idx:064x}",
                window_steps=4,
                heldout_policy="last local episode held out for downstream eval",
                license="example-only; replace with dataset card license before launch",
                raw_data_crosses_boundary=False,
            ),
        )
        for idx, suffix in enumerate(("a", "b", "c", "d"), start=10)
    )
    manifest = Phase3ConsortiumManifest(
        consortium_id="lensemble-phase3-consortium-example",
        run_id="phase3-consortium-rc-smoke",
        coordinator_id="phase3-coordinator",
        created_at=created_at or datetime(2026, 6, 5, 0, 0, 0, tzinfo=timezone.utc),
        model=Phase3ModelAgreement(
            model_family="LeWorldModel-claim-mode",
            wmcp_version=wmcp_version,
            latent_dim=192,
            num_tokens=256,
            objective_target_stop_gradient=False,
            lambda_anc=0.01,
            base_checkpoint_ref=(
                "hf://models/abdelstark/lensemble-phase2-so100-checkpoint"
                "@eaf13136b42cde324758a191c98e377636ded7f8"
            ),
            config_hash=None,
        ),
        public_probe=probe,
        runtime=Phase3RuntimePolicy(
            transport="network",
            secure_aggregation_backend="masking",
            secure_aggregation_required=True,
            dp_required=True,
            min_trainers=4,
            dropout_retry_budget=1,
        ),
        dp_policy=Phase3DPPolicy(
            enabled=True,
            clip_norm=1.0,
            noise_multiplier=1.0,
            epsilon=8.0,
            delta=1e-5,
            accountant="rdp",
        ),
        accepted_action_contracts=(action_contract,),
        accepted_observation_contracts=(observation_contract,),
        participants=participants,
        claim_boundary=(
            "Operational Phase 3 consortium-training agreement only: this manifest "
            "does not implement a provenance ledger and does not cryptographically "
            "prove honest participant computation."
        ),
    )
    validate_consortium_manifest(manifest)
    return manifest
