"""Claim-MVP evidence report for federated LeWorldModel runs.

The report is a residency-safe JSON artifact for the narrow claim-MVP path: a LeWorldModel-style
objective, real participant data sources, one or more federated rounds, and optional Hugging Face
publication metadata. It records only hashes, counts, scalar norms/losses, config fields, and repository
ids; raw observations, actions, windows, embeddings, and deltas stay out of the artifact.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from lensemble.config.manifest import config_hash
from lensemble.errors import LensembleErrorCode, SchemaVersionMismatch

CLAIM_MVP_REPORT_SCHEMA_VERSION = 2
ReportRoundState = Literal["closed", "aborted", "dry_run"]


class ClaimParticipantEvidence(BaseModel):
    """One participant's scalar/hash evidence for the claim-MVP report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    participant_id: str
    data_source: str
    data_format: str
    dataset_root: str = Field(pattern=r"^[0-9a-f]{64}$")
    update_l2_norm: float = Field(ge=0.0)
    clipped: bool
    quantized: bool


class ClaimPublicationEvidence(BaseModel):
    """Optional publication state for the dataset/checkpoint artifacts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_repos: tuple[str, ...] = ()
    checkpoint_repo: str | None = None
    checkpoint_path: str | None = None
    pushed: bool = False
    dry_run: bool = False
    blocker: str | None = None


class ClaimMetricEvidence(BaseModel):
    """Scalar metric evidence for the claim-MVP report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    val_pred: float | None = Field(default=None, ge=0.0)
    val_sigreg: float | None = Field(default=None, ge=0.0)
    effective_rank: float | None = Field(default=None, ge=0.0)
    frame_drift_deg: float | None = Field(default=None, ge=0.0)
    run_manifest_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ClaimRoundMetricEvidence(BaseModel):
    """Curve-ready per-round evidence for Phase 2 HF Jobs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    round_index: int = Field(ge=0)
    round_state: ReportRoundState
    global_model_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    participant_ids: tuple[str, ...]
    dataset_roots: dict[str, str]
    update_l2_norms: dict[str, float] = Field(default_factory=dict)


class ClaimMVPReport(BaseModel):
    """Schema-versioned, residency-safe report for the end-to-end federated claim MVP."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = CLAIM_MVP_REPORT_SCHEMA_VERSION
    claim: Literal["federated-leworldmodel-claim-mvp"] = (
        "federated-leworldmodel-claim-mvp"
    )
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    wmcp_version: str
    round_state: ReportRoundState
    committed_rounds: int = Field(ge=0)
    final_global_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    objective_target_stop_gradient: bool
    lambda_sig: float
    lambda_anc: float
    participant_count_configured: int = Field(ge=1)
    participants: tuple[ClaimParticipantEvidence, ...]
    ledger_records: tuple[dict[str, Any], ...]
    metrics: ClaimMetricEvidence = Field(default_factory=ClaimMetricEvidence)
    round_metrics: tuple[ClaimRoundMetricEvidence, ...] = ()
    publication: ClaimPublicationEvidence = Field(
        default_factory=ClaimPublicationEvidence
    )
    created_at: datetime


def build_claim_mvp_report(
    *,
    cfg: Any,
    coordinator: Any,
    participant_updates: Mapping[str, Any],
    participant_sources: Mapping[str, str],
    round_state: Any,
    metrics: ClaimMetricEvidence | None = None,
    round_metrics: Sequence[ClaimRoundMetricEvidence] | None = None,
    publication: ClaimPublicationEvidence | None = None,
    created_at: datetime | None = None,
) -> ClaimMVPReport:
    """Build a claim-MVP report from the live federated runtime objects.

    ``participant_updates`` is the exact mapping of participant id to ``PseudoGradient`` staged for the
    round. The function extracts the 32-byte dataset roots and scalar update norms, then cross-checks them
    against the coordinator's latest contribution-ledger record when a round closed.
    """
    records = tuple(coordinator.ledger_records())
    participants: list[ClaimParticipantEvidence] = []
    for participant_id in sorted(participant_updates):
        update = participant_updates[participant_id]
        participants.append(
            ClaimParticipantEvidence(
                participant_id=participant_id,
                data_source=participant_sources[participant_id],
                data_format=str(cfg.data.format),
                dataset_root=update.dataset_root.hex(),
                update_l2_norm=float(update.l2_norm),
                clipped=bool(update.clipped),
                quantized=bool(update.quantized),
            )
        )

    round_state_value = _report_round_state(round_state)

    if round_state_value == "closed" and records:
        latest_roots = records[-1].dataset_roots
        for participant in participants:
            if latest_roots.get(participant.participant_id) != participant.dataset_root:
                raise ValueError(
                    "participant update dataset root does not match the coordinator ledger "
                    f"for {participant.participant_id!r}"
                )
    return ClaimMVPReport(
        config_hash=config_hash(asdict(cfg)),
        wmcp_version=str(cfg.model.wmcp_version),
        round_state=round_state_value,
        committed_rounds=len(records),
        final_global_hash=coordinator.global_state_hash(),
        objective_target_stop_gradient=bool(cfg.objective.target_stop_gradient),
        lambda_sig=float(cfg.objective.lambda_sig),
        lambda_anc=float(cfg.objective.lambda_anc),
        participant_count_configured=int(cfg.federation.participant_count),
        participants=tuple(participants),
        ledger_records=tuple(r.model_dump(mode="json") for r in records),
        metrics=metrics or ClaimMetricEvidence(),
        round_metrics=tuple(round_metrics)
        if round_metrics is not None
        else _round_metrics_from_records(records),
        publication=publication or ClaimPublicationEvidence(),
        created_at=created_at or datetime.now(timezone.utc),
    )


def _round_metrics_from_records(
    records: Sequence[Any],
) -> tuple[ClaimRoundMetricEvidence, ...]:
    """Derive a minimal curve-ready series from contribution-ledger records."""
    return tuple(
        ClaimRoundMetricEvidence(
            round_index=int(record.round_index),
            round_state="closed",
            global_model_hash=str(record.global_model_hash),
            participant_ids=tuple(record.participants),
            dataset_roots=dict(record.dataset_roots),
        )
        for record in records
    )


def _report_round_state(round_state: Any) -> ReportRoundState:
    value = getattr(round_state, "value", round_state)
    if value == "closed":
        return "closed"
    if value == "aborted":
        return "aborted"
    if value == "dry_run":
        return "dry_run"
    raise ValueError(
        f"claim-MVP reports can only be emitted for closed, aborted, or dry-run states; got {round_state!r}"
    )


def parse_claim_mvp_report(raw: dict[str, Any]) -> ClaimMVPReport:
    """Parse a claim-MVP report, gating ``schema_version`` before body validation."""
    version = raw.get("schema_version")
    if not isinstance(version, int) or version > CLAIM_MVP_REPORT_SCHEMA_VERSION:
        raise SchemaVersionMismatch(
            f"claim-MVP report schema_version {version!r} exceeds reader max "
            f"{CLAIM_MVP_REPORT_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation=f"read with a build supporting schema_version <= {version!r}",
        )
    return ClaimMVPReport.model_validate(raw)
