"""Phase 2 participant-silo data smoke reports.

The Phase 2 dataset gate is intentionally evidence-oriented: load each
participant-local source through the public data adapter, count training
windows, commit a Merkle root, and emit only residency-safe metadata. Raw
observations/actions remain inside the local process.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lensemble.contracts import ActionSpec
from lensemble.data.adapters import load_episodes
from lensemble.data.dataset import Format
from lensemble.errors import ContractViolation, LensembleErrorCode
from lensemble.provenance import commit_dataset

PHASE2_DATASET_SMOKE_SCHEMA_VERSION = 1


class Phase2ActionSpecEvidence(BaseModel):
    """Residency-safe action-space metadata for one Phase 2 silo."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    embodiment_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    dim: int = Field(ge=1)
    low: tuple[float, ...] | None
    high: tuple[float, ...] | None
    num_classes: tuple[int, ...] | None
    units: tuple[str, ...]
    wmcp_version: str = Field(min_length=1)


class Phase2SiloSmokeEvidence(BaseModel):
    """One participant silo's load/commit/window-count evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    participant_id: str = Field(min_length=1)
    data_source: str = Field(min_length=1)
    data_format: str = Field(min_length=1)
    episode_count: int = Field(ge=1)
    window_count: int = Field(ge=0)
    dataset_root: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_commitment_schema_version: int = Field(ge=1)
    hash_algorithm: str = Field(min_length=1)
    wmcp_version: str = Field(min_length=1)
    embodiment_ids: tuple[str, ...] = Field(min_length=1)
    action_spec: Phase2ActionSpecEvidence
    observation_shape: tuple[int, ...] | None
    action_shape: tuple[int, ...] | None


class Phase2DatasetSmokeReport(BaseModel):
    """Machine-readable Phase 2 participant-silo data gate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=PHASE2_DATASET_SMOKE_SCHEMA_VERSION, ge=1)
    generated_at: datetime
    window_steps: int = Field(ge=1)
    min_windows_per_silo: int = Field(ge=0)
    participant_count: int = Field(ge=1)
    silos: tuple[Phase2SiloSmokeEvidence, ...] = Field(min_length=1)
    blocker: str | None = None

    @model_validator(mode="after")
    def _validate_report(self) -> "Phase2DatasetSmokeReport":
        if self.participant_count != len(self.silos):
            raise ValueError(
                f"participant_count {self.participant_count} does not match "
                f"silos length {len(self.silos)}"
            )
        participant_ids = [silo.participant_id for silo in self.silos]
        if len(set(participant_ids)) != len(participant_ids):
            raise ValueError(f"duplicate participant ids in report: {participant_ids}")
        if self.blocker is None:
            underfilled = [
                silo.participant_id
                for silo in self.silos
                if silo.window_count < self.min_windows_per_silo
            ]
            if underfilled:
                raise ValueError(
                    "silos below min_windows_per_silo "
                    f"{self.min_windows_per_silo}: {underfilled}"
                )
        return self


def _action_spec_evidence(action_spec: ActionSpec) -> Phase2ActionSpecEvidence:
    raw: dict[str, Any] = asdict(action_spec)
    raw["kind"] = str(action_spec.kind.value)
    return Phase2ActionSpecEvidence.model_validate(raw)


def _count_windows(
    data_source: str | Path,
    *,
    participant_id: str,
    data_format: Format | None,
    window_steps: int,
) -> Phase2SiloSmokeEvidence:
    dataset = load_episodes(data_source, fmt=data_format)
    commitment = commit_dataset(dataset)

    window_count = 0
    observation_shape: tuple[int, ...] | None = None
    action_shape: tuple[int, ...] | None = None
    for window in dataset.windows(window_steps):
        if window_count == 0:
            observation_shape = tuple(int(dim) for dim in window.obs.shape)
            action_shape = tuple(int(dim) for dim in window.actions.shape)
        window_count += 1

    first_episode = dataset.episodes[0]
    return Phase2SiloSmokeEvidence(
        participant_id=participant_id,
        data_source=str(data_source),
        data_format=dataset.fmt,
        episode_count=commitment.episode_count,
        window_count=window_count,
        dataset_root=commitment.merkle_root,
        dataset_commitment_schema_version=commitment.schema_version,
        hash_algorithm=commitment.hash_algorithm,
        wmcp_version=commitment.wmcp_version,
        embodiment_ids=commitment.embodiment_ids,
        action_spec=_action_spec_evidence(first_episode.action_spec),
        observation_shape=observation_shape,
        action_shape=action_shape,
    )


def build_phase2_dataset_smoke_report(
    data_sources: Sequence[str | Path],
    *,
    participant_ids: Sequence[str] | None = None,
    data_format: Format | None = None,
    window_steps: int = 4,
    min_windows_per_silo: int = 1,
) -> Phase2DatasetSmokeReport:
    """Load, commit, and summarize Phase 2 participant data sources.

    Raises :class:`ContractViolation` before loading when the request cannot
    produce a coherent participant report. Adapter/provenance failures propagate
    from the lower layers so callers see the canonical error code.
    """
    if window_steps <= 0:
        raise ContractViolation(
            f"window_steps must be > 0, got {window_steps}",
            code=LensembleErrorCode.WMCP_CONTRACT_VIOLATION,
            remediation="set --window-steps to a positive fixed horizon",
        )
    if min_windows_per_silo < 0:
        raise ContractViolation(
            f"min_windows_per_silo must be >= 0, got {min_windows_per_silo}",
            code=LensembleErrorCode.WMCP_CONTRACT_VIOLATION,
            remediation="set --min-windows-per-silo to zero or a positive integer",
        )
    if not data_sources:
        raise ContractViolation(
            "at least one data source is required",
            code=LensembleErrorCode.WMCP_CONTRACT_VIOLATION,
            remediation="pass one --data-source per participant silo",
        )

    resolved_participant_ids = (
        tuple(participant_ids)
        if participant_ids is not None
        else tuple(f"participant-{idx}" for idx, _ in enumerate(data_sources))
    )
    if len(resolved_participant_ids) != len(data_sources):
        raise ContractViolation(
            "participant id count does not match data source count",
            code=LensembleErrorCode.WMCP_CONTRACT_VIOLATION,
            remediation="pass one --participant-id per --data-source, or omit participant ids",
        )

    silos = tuple(
        _count_windows(
            source,
            participant_id=participant_id,
            data_format=data_format,
            window_steps=window_steps,
        )
        for participant_id, source in zip(resolved_participant_ids, data_sources)
    )
    return Phase2DatasetSmokeReport(
        schema_version=PHASE2_DATASET_SMOKE_SCHEMA_VERSION,
        generated_at=datetime.now(timezone.utc),
        window_steps=window_steps,
        min_windows_per_silo=min_windows_per_silo,
        participant_count=len(silos),
        silos=silos,
    )
