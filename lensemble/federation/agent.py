"""Phase 3 sovereign participant-agent runtime.

The agent is the participant-side Phase 3 process boundary: it validates the
consortium run agreement and local data/probe/model contracts before sending
any message, then delegates the actual LeWorldModel local objective to
``Participant.local_round``. It persists only the released pseudo-gradient
delta plus redacted metadata, so resume/rejoin can replay the same committed
update hash without serializing resident observations, actions, latents,
embeddings, or private action heads.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import load_file, save_file
from safetensors.torch import save as st_save

from lensemble.config.consortium import (
    Phase3ConsortiumManifest,
    Phase3ParticipantDeclaration,
    validate_participant_join,
)
from lensemble.config.seed import derive
from lensemble.data.residency import guard_egress
from lensemble.errors import (
    CheckpointIntegrityError,
    ConfigError,
    LensembleErrorCode,
    ResidencyViolation,
    RoundError,
)
from lensemble.federation.participant import Participant
from lensemble.federation.pseudogradient import PseudoGradient
from lensemble.observability.logging import LogLevel, emit_log
from lensemble.observability.metrics import emit_metric

if TYPE_CHECKING:
    from lensemble.config.schema import LensembleConfig
    from lensemble.contracts import ActionSpec
    from lensemble.data.episode import Window
    from lensemble.federation.transport import Transport

PARTICIPANT_AGENT_PREFLIGHT_SCHEMA_VERSION = 1
PARTICIPANT_AGENT_ROUND_STATE_SCHEMA_VERSION = 1

ParticipantFactory = Callable[["LensembleConfig", str, "Transport"], Participant]

_ROUND_STATE_FILE = "round_state.json"
_DELTA_FILE = "delta.safetensors"
_FORBIDDEN_STATE_KEYS = frozenset(
    {
        "obs",
        "observation",
        "observations",
        "action",
        "actions",
        "latent",
        "latents",
        "embedding",
        "embeddings",
        "action_head",
        "private_action_head",
    }
)


def _fail(key: str, value: object, expected: str, remediation: str) -> ConfigError:
    err = ConfigError(
        f"invalid participant-agent preflight: {key}={value!r} ({expected})",
        code=LensembleErrorCode.CONFIG_INVALID,
        remediation=remediation,
    )
    err.key = key  # type: ignore[attr-defined]
    err.value = value  # type: ignore[attr-defined]
    err.expected = expected  # type: ignore[attr-defined]
    return err


def _default_participant_factory(
    config: "LensembleConfig", participant_id: str, transport: "Transport"
) -> Participant:
    return Participant(config, participant_id=participant_id, transport=transport)


def _safe_path_segment(kind: str, value: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise _fail(
            kind,
            value,
            "a non-empty path segment with no separators",
            "use a stable id that cannot traverse the participant-agent state directory",
        )
    return value


def _float_close(lhs: float, rhs: float) -> bool:
    return abs(lhs - rhs) <= 1e-9 + 1e-9 * max(abs(lhs), abs(rhs), 1.0)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _delta_hash(delta: torch.Tensor) -> str:
    raw = st_save({"delta": delta.detach().cpu().contiguous().to(torch.float32)})
    return hashlib.sha256(raw).hexdigest()


def _released_update_hash(
    *,
    delta_sha256: str,
    dataset_root_hex: str,
    round_index: int,
    clipped: bool,
    quantized: bool,
) -> str:
    payload = {
        "clipped": clipped,
        "dataset_root_hex": dataset_root_hex,
        "delta_sha256": delta_sha256,
        "quantized": quantized,
        "round_index": round_index,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


class ParticipantAgentPreflight(BaseModel):
    """Residency-safe result of a participant-agent preflight."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = PARTICIPANT_AGENT_PREFLIGHT_SCHEMA_VERSION
    participant_id: str = Field(min_length=1)
    consortium_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    data_ref: str = Field(min_length=1)
    data_format: str = Field(min_length=1)
    window_steps: int = Field(ge=1)
    local_window_count: int = Field(ge=1)
    first_window_obs_shape: tuple[int, ...] = Field(min_length=1)
    action_contract_id: str = Field(min_length=1)
    observation_contract_id: str = Field(min_length=1)
    public_probe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_probe_version: int = Field(ge=1)
    model_wmcp_version: str = Field(min_length=1)
    transport: str = Field(min_length=1)
    secure_aggregation_backend: str = Field(min_length=1)
    dp_accountant: str = Field(min_length=1)
    residency_checked: Literal[True] = True
    checks: tuple[str, ...] = Field(min_length=1)


class ParticipantAgentRoundState(BaseModel):
    """On-disk, residency-safe state for replaying a released update."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = PARTICIPANT_AGENT_ROUND_STATE_SCHEMA_VERSION
    state_kind: Literal["phase3_participant_agent_round"] = (
        "phase3_participant_agent_round"
    )
    participant_id: str = Field(min_length=1)
    consortium_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    round_index: int = Field(ge=0)
    round_seed: int
    wmcp_version: str = Field(min_length=1)
    update_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    update_delta_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    delta_path: str = Field(min_length=1)
    delta_numel: int = Field(ge=0)
    delta_dtype: Literal["float32"] = "float32"
    dataset_root_hex: str = Field(pattern=r"^[0-9a-f]{64}$")
    l2_norm: float = Field(ge=0.0)
    clipped: bool
    quantized: bool
    submitted: bool = False


class ParticipantAgentRoundResult(BaseModel):
    """Machine-readable result of one participant-agent round."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    preflight: ParticipantAgentPreflight
    state: ParticipantAgentRoundState
    state_path: str = Field(min_length=1)
    delta_path: str = Field(min_length=1)
    resumed: bool


class Phase3ParticipantAgent:
    """Manifest-aware participant runtime over the existing ``Participant`` local round."""

    def __init__(
        self,
        config: "LensembleConfig",
        *,
        manifest: Phase3ConsortiumManifest,
        participant_id: str,
        transport: "Transport",
        state_dir: Path,
        coordinator_endpoint: str,
        participant_factory: ParticipantFactory | None = None,
        emit_observability: bool = True,
    ) -> None:
        self.config = config
        self.manifest = manifest
        self.participant_id = participant_id
        self.transport = transport
        self.state_dir = Path(state_dir)
        self.coordinator_endpoint = coordinator_endpoint
        self.participant_factory = participant_factory or _default_participant_factory
        self.emit_observability = emit_observability
        self._run_segment = _safe_path_segment("run_id", manifest.run_id)
        self._participant_segment = _safe_path_segment("participant_id", participant_id)

    def _participant(self) -> Participant:
        return self.participant_factory(
            self.config, self.participant_id, self.transport
        )

    def preflight(self) -> ParticipantAgentPreflight:
        """Validate local join preconditions without contacting the coordinator."""

        declaration = validate_participant_join(
            self.manifest, participant_id=self.participant_id
        )
        data = declaration.data
        if (
            data is None
        ):  # validate_participant_join already enforces this; kept defensive.
            raise _fail(
                "participant.data",
                None,
                "a data declaration",
                "declare the participant-local data ref before joining",
            )
        if self.config.run_mode != "participant":
            raise _fail(
                "run_mode",
                self.config.run_mode,
                "participant",
                "participant-agent runs require a participant-mode config",
            )
        if self.config.data.data_source is None:
            raise _fail(
                "data.data_source",
                None,
                "a participant-local data ref",
                "pass --data-source or set cfg.data.data_source to the local private store",
            )
        if self.config.data.data_source != data.data_ref:
            raise _fail(
                "data.data_source",
                self.config.data.data_source,
                f"== manifest data_ref ({data.data_ref})",
                "run with the local data ref declared in the consortium manifest",
            )
        if self.config.data.format != data.format:
            raise _fail(
                "data.format",
                self.config.data.format,
                f"== manifest format ({data.format})",
                "select the data adapter declared in the consortium manifest",
            )
        if int(self.config.data.window_steps) != data.window_steps:
            raise _fail(
                "data.window_steps",
                self.config.data.window_steps,
                f"== manifest window_steps ({data.window_steps})",
                "use the exact window horizon declared for this participant",
            )
        if self.config.data.residency_enforced is not True:
            raise _fail(
                "data.residency_enforced",
                self.config.data.residency_enforced,
                "True",
                "participant-agent residency checks may not be disabled",
            )

        self._validate_model_runtime_agreement(declaration)

        participant = self._participant()
        probe = participant._pinned_probe()
        if probe.content_hash.hex() != self.manifest.public_probe.content_hash:
            raise _fail(
                "data.probe_path.content_hash",
                probe.content_hash.hex(),
                f"== manifest public_probe.content_hash ({self.manifest.public_probe.content_hash})",
                "pin the consortium public probe before joining",
            )
        if probe.probe_version != self.manifest.public_probe.version:
            raise _fail(
                "data.probe_path.probe_version",
                probe.probe_version,
                f"== manifest public_probe.version ({self.manifest.public_probe.version})",
                "pin the consortium public probe version before joining",
            )

        action_spec = participant._action_spec()
        self._validate_action_spec(action_spec, declaration)

        windows = self._load_windows(participant)
        first = windows[0]
        self._validate_first_window(first, declaration)

        return ParticipantAgentPreflight(
            participant_id=self.participant_id,
            consortium_id=self.manifest.consortium_id,
            run_id=self.manifest.run_id,
            data_ref=data.data_ref,
            data_format=data.format,
            window_steps=data.window_steps,
            local_window_count=len(windows),
            first_window_obs_shape=tuple(int(v) for v in first.obs.shape),
            action_contract_id=declaration.action_contract.contract_id,
            observation_contract_id=declaration.observation_contract.contract_id,
            public_probe_hash=probe.content_hash.hex(),
            public_probe_version=probe.probe_version,
            model_wmcp_version=self.config.model.wmcp_version,
            transport=self.manifest.runtime.transport,
            secure_aggregation_backend=self.manifest.runtime.secure_aggregation_backend,
            dp_accountant=self.manifest.dp_policy.accountant,
            checks=(
                "manifest_join",
                "data_ref",
                "data_adapter_windows",
                "action_contract",
                "observation_contract",
                "public_probe_pin",
                "model_runtime_policy",
                "residency_boundary",
            ),
        )

    def run_assigned_round(
        self, *, resume: bool = False
    ) -> ParticipantAgentRoundResult:
        """Join, run one assigned local round, persist and submit only the released update."""

        preflight = self.preflight()
        participant = self._participant()
        global_state = participant.join(self.coordinator_endpoint)
        if global_state.wmcp_version != self.manifest.model.wmcp_version:
            raise _fail(
                "global_state.wmcp_version",
                global_state.wmcp_version,
                f"== manifest model.wmcp_version ({self.manifest.model.wmcp_version})",
                "join only rounds opened under the accepted WMCP contract",
            )

        if resume:
            existing = self._load_existing_state(global_state.round_index)
            if existing is not None:
                update = self._load_update(existing)
                guard_egress(update, boundary="participant-agent->coordinator")
                self.transport.submit_update(
                    participant_id=self.participant_id,
                    round_index=existing.round_index,
                    update=update,
                )
                submitted = existing.model_copy(update={"submitted": True})
                state_path = self._write_state(submitted)
                self._emit_round_observability(submitted, resumed=True)
                return ParticipantAgentRoundResult(
                    preflight=preflight,
                    state=submitted,
                    state_path=str(state_path),
                    delta_path=str(self._round_dir(existing.round_index) / _DELTA_FILE),
                    resumed=True,
                )

        torch.manual_seed(
            derive(
                self.config.determinism.root_seed,
                f"participant-agent:{self.participant_id}:{global_state.round_index}",
            )
            % (2**63)
        )
        update = participant.local_round(
            global_state, round_seed=global_state.sketch_seed
        )
        guard_egress(update, boundary="participant-agent->coordinator")
        state = self._persist_update(
            update,
            preflight=preflight,
            round_seed=global_state.sketch_seed,
            submitted=False,
        )
        self.transport.submit_update(
            participant_id=self.participant_id,
            round_index=update.round_index,
            update=update,
        )
        submitted = state.model_copy(update={"submitted": True})
        state_path = self._write_state(submitted)
        self._emit_round_observability(submitted, resumed=False)
        return ParticipantAgentRoundResult(
            preflight=preflight,
            state=submitted,
            state_path=str(state_path),
            delta_path=str(self._round_dir(update.round_index) / _DELTA_FILE),
            resumed=False,
        )

    def _validate_model_runtime_agreement(
        self, declaration: Phase3ParticipantDeclaration
    ) -> None:
        model = self.manifest.model
        runtime = self.manifest.runtime
        dp = self.manifest.dp_policy
        if self.config.model.wmcp_version != model.wmcp_version:
            raise _fail(
                "model.wmcp_version",
                self.config.model.wmcp_version,
                f"== manifest model.wmcp_version ({model.wmcp_version})",
                "join with the consortium WMCP version",
            )
        if int(self.config.model.latent_dim) != model.latent_dim:
            raise _fail(
                "model.latent_dim",
                self.config.model.latent_dim,
                f"== manifest model.latent_dim ({model.latent_dim})",
                "run the model shape accepted by the consortium manifest",
            )
        if int(self.config.model.num_tokens) != model.num_tokens:
            raise _fail(
                "model.num_tokens",
                self.config.model.num_tokens,
                f"== manifest model.num_tokens ({model.num_tokens})",
                "run the token shape accepted by the consortium manifest",
            )
        if bool(self.config.objective.target_stop_gradient) != (
            model.objective_target_stop_gradient
        ):
            raise _fail(
                "objective.target_stop_gradient",
                self.config.objective.target_stop_gradient,
                f"== manifest value ({model.objective_target_stop_gradient})",
                "claim-mode Phase 3 participants must use the accepted LeWorldModel objective branch",
            )
        if not _float_close(float(self.config.objective.lambda_anc), model.lambda_anc):
            raise _fail(
                "objective.lambda_anc",
                self.config.objective.lambda_anc,
                f"== manifest value ({model.lambda_anc})",
                "use the accepted public-probe anchor strength for this run",
            )
        if self.config.federation.transport != runtime.transport:
            raise _fail(
                "federation.transport",
                self.config.federation.transport,
                f"== manifest runtime.transport ({runtime.transport})",
                "select the transport mode accepted by the consortium manifest",
            )
        if (
            self.config.federation.aggregation_backend
            != runtime.secure_aggregation_backend
        ):
            raise _fail(
                "federation.aggregation_backend",
                self.config.federation.aggregation_backend,
                f"== manifest secure_aggregation_backend ({runtime.secure_aggregation_backend})",
                "select the secure-aggregation backend accepted by the consortium manifest",
            )
        if self.config.privacy.enabled != dp.enabled:
            raise _fail(
                "privacy.enabled",
                self.config.privacy.enabled,
                f"== manifest dp_policy.enabled ({dp.enabled})",
                "match the consortium DP policy",
            )
        if not _float_close(float(self.config.privacy.clip_norm), dp.clip_norm):
            raise _fail(
                "privacy.clip_norm",
                self.config.privacy.clip_norm,
                f"== manifest clip_norm ({dp.clip_norm})",
                "match the consortium DP clipping bound",
            )
        if not _float_close(
            float(self.config.privacy.noise_multiplier), dp.noise_multiplier
        ):
            raise _fail(
                "privacy.noise_multiplier",
                self.config.privacy.noise_multiplier,
                f"== manifest noise_multiplier ({dp.noise_multiplier})",
                "match the consortium DP noise multiplier",
            )
        if not _float_close(float(self.config.privacy.epsilon), dp.epsilon):
            raise _fail(
                "privacy.epsilon",
                self.config.privacy.epsilon,
                f"== manifest epsilon ({dp.epsilon})",
                "match the consortium DP budget",
            )
        if not _float_close(float(self.config.privacy.delta), dp.delta):
            raise _fail(
                "privacy.delta",
                self.config.privacy.delta,
                f"== manifest delta ({dp.delta})",
                "match the consortium DP budget",
            )
        if self.config.privacy.accountant != dp.accountant:
            raise _fail(
                "privacy.accountant",
                self.config.privacy.accountant,
                f"== manifest accountant ({dp.accountant})",
                "match the consortium DP accountant",
            )
        if not declaration.capabilities.resumable:
            raise _fail(
                "participant.capabilities.resumable",
                declaration.capabilities.resumable,
                "True",
                "Phase 3 participant agents must support deterministic resume/rejoin state",
            )

    def _validate_action_spec(
        self, action_spec: "ActionSpec", declaration: Phase3ParticipantDeclaration
    ) -> None:
        contract = declaration.action_contract
        kind = getattr(action_spec.kind, "value", action_spec.kind)
        checks: tuple[tuple[str, object, object], ...] = (
            ("embodiment_id", action_spec.embodiment_id, contract.embodiment_id),
            ("kind", kind, contract.kind),
            ("dim", action_spec.dim, contract.dim),
            ("low", action_spec.low, contract.low),
            ("high", action_spec.high, contract.high),
            ("num_classes", action_spec.num_classes, contract.num_classes),
            ("units", action_spec.units, contract.units),
            ("wmcp_version", action_spec.wmcp_version, contract.wmcp_version),
        )
        for field, got, expected in checks:
            if got != expected:
                raise _fail(
                    f"action_spec.{field}",
                    got,
                    f"== manifest action_contract.{field} ({expected!r})",
                    "load data whose local ActionSpec matches the accepted consortium action contract",
                )

    def _load_windows(self, participant: Participant) -> Sequence["Window"]:
        try:
            windows = participant._local_windows()
        except ValueError as exc:
            raise _fail(
                "data.adapter",
                str(exc),
                "a loadable participant-local data adapter",
                "fix cfg.data.format/data_source or register the missing data adapter",
            ) from exc
        if not windows:
            raise RoundError(
                "participant-agent preflight found zero local training windows",
                code=LensembleErrorCode.ROUND_FAILED,
                remediation="point cfg.data.data_source at a non-empty local episode store",
            )
        return windows

    def _validate_first_window(
        self, window: "Window", declaration: Phase3ParticipantDeclaration
    ) -> None:
        observation = declaration.observation_contract
        action = declaration.action_contract
        obs_shape = tuple(int(v) for v in window.obs.shape)
        if obs_shape != observation.shape:
            raise _fail(
                "window.obs.shape",
                obs_shape,
                f"== manifest observation_contract.shape ({observation.shape})",
                "use data with the accepted observation/window shape",
            )
        obs_dtype = _dtype_name(window.obs.dtype)
        if obs_dtype != observation.dtype:
            raise _fail(
                "window.obs.dtype",
                obs_dtype,
                f"== manifest observation_contract.dtype ({observation.dtype})",
                "use data with the accepted observation dtype",
            )
        if int(window.num_steps) != int(self.config.data.window_steps):
            raise _fail(
                "window.num_steps",
                window.num_steps,
                f"== cfg.data.window_steps ({self.config.data.window_steps})",
                "load windows with the configured horizon",
            )
        action_shape = tuple(int(v) for v in window.actions.shape)
        expected_action_shape = (int(self.config.data.window_steps), action.dim)
        if action_shape != expected_action_shape:
            raise _fail(
                "window.actions.shape",
                action_shape,
                f"== {expected_action_shape}",
                "use data whose action rows match the accepted action contract",
            )
        if window.embodiment_id != action.embodiment_id:
            raise _fail(
                "window.embodiment_id",
                window.embodiment_id,
                f"== manifest action_contract.embodiment_id ({action.embodiment_id})",
                "use windows from the accepted embodiment",
            )

    def _round_dir(self, round_index: int) -> Path:
        return (
            self.state_dir
            / self._run_segment
            / self._participant_segment
            / f"round-{round_index:05d}"
        )

    def _persist_update(
        self,
        update: PseudoGradient,
        *,
        preflight: ParticipantAgentPreflight,
        round_seed: int,
        submitted: bool,
    ) -> ParticipantAgentRoundState:
        round_dir = self._round_dir(update.round_index)
        round_dir.mkdir(parents=True, exist_ok=True)
        delta = update.delta.detach().cpu().contiguous().to(torch.float32)
        digest = _delta_hash(delta)
        save_file(
            {"delta": delta},
            str(round_dir / _DELTA_FILE),
            metadata={
                "update_delta_sha256": digest,
                "participant_id": self.participant_id,
                "run_id": self.manifest.run_id,
                "round_index": str(update.round_index),
            },
        )
        state = ParticipantAgentRoundState(
            participant_id=self.participant_id,
            consortium_id=self.manifest.consortium_id,
            run_id=self.manifest.run_id,
            round_index=update.round_index,
            round_seed=round_seed,
            wmcp_version=preflight.model_wmcp_version,
            update_sha256=_released_update_hash(
                delta_sha256=digest,
                dataset_root_hex=update.dataset_root.hex(),
                round_index=update.round_index,
                clipped=update.clipped,
                quantized=update.quantized,
            ),
            update_delta_sha256=digest,
            delta_path=_DELTA_FILE,
            delta_numel=int(delta.numel()),
            dataset_root_hex=update.dataset_root.hex(),
            l2_norm=float(update.l2_norm),
            clipped=update.clipped,
            quantized=update.quantized,
            submitted=submitted,
        )
        self._validate_state_keys(state)
        self._write_state(state)
        return state

    def _write_state(self, state: ParticipantAgentRoundState) -> Path:
        self._validate_state_keys(state)
        path = self._round_dir(state.round_index) / _ROUND_STATE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    def _load_existing_state(
        self, round_index: int
    ) -> ParticipantAgentRoundState | None:
        path = self._round_dir(round_index) / _ROUND_STATE_FILE
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8")
        state = ParticipantAgentRoundState.model_validate_json(raw)
        if state.schema_version != PARTICIPANT_AGENT_ROUND_STATE_SCHEMA_VERSION:
            raise CheckpointIntegrityError(
                f"participant-agent round_state schema_version {state.schema_version!r} is unsupported",
                code=LensembleErrorCode.CHECKPOINT_INTEGRITY,
                remediation="resume with a build that understands this participant-agent round state",
            )
        expected = {
            "participant_id": self.participant_id,
            "consortium_id": self.manifest.consortium_id,
            "run_id": self.manifest.run_id,
            "wmcp_version": self.manifest.model.wmcp_version,
        }
        for field, value in expected.items():
            got = getattr(state, field)
            if got != value:
                raise CheckpointIntegrityError(
                    f"participant-agent resume state {field} mismatch: expected {value!r}, got {got!r}",
                    code=LensembleErrorCode.CHECKPOINT_INTEGRITY,
                    remediation="resume only from state produced for the same participant/run/WMCP contract",
                )
        self._validate_state_keys(state)
        return state

    def _load_update(self, state: ParticipantAgentRoundState) -> PseudoGradient:
        delta_path = self._round_dir(state.round_index) / state.delta_path
        tensors = load_file(str(delta_path))
        delta = tensors["delta"].detach().cpu().contiguous().to(torch.float32)
        actual = _delta_hash(delta)
        if actual != state.update_delta_sha256:
            raise CheckpointIntegrityError(
                "participant-agent persisted update hash mismatch: "
                f"expected {state.update_delta_sha256}, got {actual}",
                code=LensembleErrorCode.CHECKPOINT_INTEGRITY,
                remediation="discard the corrupted local round state and recompute the round from committed inputs",
            )
        payload_hash = _released_update_hash(
            delta_sha256=actual,
            dataset_root_hex=state.dataset_root_hex,
            round_index=state.round_index,
            clipped=state.clipped,
            quantized=state.quantized,
        )
        if payload_hash != state.update_sha256:
            raise CheckpointIntegrityError(
                "participant-agent released update hash does not match the recorded state: "
                f"expected {state.update_sha256}, got {payload_hash}",
                code=LensembleErrorCode.CHECKPOINT_INTEGRITY,
                remediation="discard the corrupted local round state and recompute the round",
            )
        update = PseudoGradient(
            delta=delta,
            l2_norm=float(delta.norm()),
            dataset_root=bytes.fromhex(state.dataset_root_hex),
            round_index=state.round_index,
            clipped=state.clipped,
            quantized=state.quantized,
        )
        if abs(update.l2_norm - state.l2_norm) > 1e-6 + 1e-5 * state.l2_norm:
            raise CheckpointIntegrityError(
                "participant-agent persisted update norm does not match the recorded state",
                code=LensembleErrorCode.CHECKPOINT_INTEGRITY,
                remediation="discard the corrupted local round state and recompute the round",
            )
        return update

    def _emit_round_observability(
        self, state: ParticipantAgentRoundState, *, resumed: bool
    ) -> None:
        if not self.emit_observability:
            return
        run_dir = self._round_dir(state.round_index)
        correlation_id = (
            f"{self.manifest.run_id}:{self.participant_id}:round-{state.round_index}"
        )
        emit_log(
            LogLevel.INFO,
            "participant_agent.round_submitted",
            run_dir=run_dir,
            correlation_id=correlation_id,
            round=state.round_index,
            participant_id=self.participant_id,
            update_sha256=state.update_sha256,
            update_delta_sha256=state.update_delta_sha256,
            delta_numel=state.delta_numel,
            l2_norm=state.l2_norm,
            resumed=resumed,
        )
        emit_metric(
            "grad_norm",
            state.l2_norm,
            run_dir=run_dir,
            correlation_id=correlation_id,
            round=state.round_index,
            participant_id=self.participant_id,
        )
        emit_metric(
            "fed/comm_bytes",
            float(state.delta_numel * 4),
            run_dir=run_dir,
            correlation_id=correlation_id,
            round=state.round_index,
            participant_id=self.participant_id,
        )

    @staticmethod
    def _validate_state_keys(state: ParticipantAgentRoundState) -> None:
        payload = state.model_dump(mode="json")
        lower_keys = {str(key).lower() for key in payload}
        leaked = sorted(lower_keys & _FORBIDDEN_STATE_KEYS)
        if leaked:
            raise ResidencyViolation(
                f"participant-agent state contains forbidden residency field(s): {leaked}",
                code=LensembleErrorCode.RESIDENCY_VIOLATION,
                remediation="persist only released pseudo-gradient metadata, never raw data or local heads",
            )


__all__ = [
    "PARTICIPANT_AGENT_PREFLIGHT_SCHEMA_VERSION",
    "PARTICIPANT_AGENT_ROUND_STATE_SCHEMA_VERSION",
    "ParticipantAgentPreflight",
    "ParticipantAgentRoundState",
    "ParticipantAgentRoundResult",
    "Phase3ParticipantAgent",
]
