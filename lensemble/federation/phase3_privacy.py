"""Phase 3 secure-aggregation and DP runtime reports.

This module records the operational privacy controls exercised by a Phase 3
coordinator service. It deliberately reports secure aggregation and DP
accounting status without exposing per-participant update values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field

from lensemble.aggregation import (
    FieldParams,
    SimulatedSecureAggregator,
    TEEAggregator,
    assert_no_wrap,
    encode_delta,
    flat_content_hash,
)
from lensemble.errors import (
    LensembleErrorCode,
    PrivacyBudgetExceeded,
    SecureAggregationError,
)
from lensemble.privacy import build_accountant

if TYPE_CHECKING:
    from lensemble.config.consortium import Phase3ConsortiumManifest
    from lensemble.config.schema import LensembleConfig
    from lensemble.federation.pseudogradient import PseudoGradient

PHASE3_AGGREGATION_PRIVACY_REPORT_SCHEMA_VERSION = 1

AggregationBackendStatus = Literal["secure_sum", "explicit_fallback"]
DPAccountingStatus = Literal["accounted", "disabled", "noise_disabled"]

_FIELD_MODULUS = 2**61 - 1
_FIELD_SCALE = 2.0**20
_TEE_SMOKE_CODE_HASH = "ab" * 32


class Phase3SecureAggregationReport(BaseModel):
    """Residency-safe report for the aggregation path used in a Phase 3 round."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: str = Field(min_length=1)
    backend_status: AggregationBackendStatus
    secure_sum_consumed: bool
    fallback_used: bool
    fallback_reason: str | None = Field(default=None, min_length=1)
    threshold: int = Field(ge=1)
    contributing_count: int = Field(ge=1)
    field_modulus: int = Field(ge=1)
    field_scale: float = Field(gt=0.0)
    fixed_point_tolerance: float = Field(ge=0.0)
    secure_sum_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    averaged_update_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    secure_sum_max_abs_error: float | None = Field(default=None, ge=0.0)
    secure_sum_matches_plaintext: bool | None = None


class Phase3DPAccountingReport(BaseModel):
    """Residency-safe DP accounting report for one successful Phase 3 round."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool
    effective_dp: bool
    status: DPAccountingStatus
    accountant: str = Field(min_length=1)
    clip_norm: float = Field(gt=0.0)
    noise_multiplier: float = Field(ge=0.0)
    target_epsilon: float = Field(gt=0.0)
    target_delta: float = Field(gt=0.0, lt=1.0)
    sample_rate: float = Field(gt=0.0, le=1.0)
    rounds_accounted: int = Field(ge=0)
    epsilon_spent: float | None = Field(default=None, ge=0.0)


class Phase3AggregationPrivacyReport(BaseModel):
    """Machine-readable Phase 3 aggregation/privacy report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = PHASE3_AGGREGATION_PRIVACY_REPORT_SCHEMA_VERSION
    consortium_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    round_index: int = Field(ge=0)
    participant_count: int = Field(ge=1)
    secure_aggregation: Phase3SecureAggregationReport
    dp_accounting: Phase3DPAccountingReport
    redaction_policy: str = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)


def build_phase3_aggregation_privacy_report(
    config: "LensembleConfig",
    manifest: "Phase3ConsortiumManifest",
    updates: "dict[str, PseudoGradient]",
    *,
    round_index: int,
) -> Phase3AggregationPrivacyReport:
    """Build a Phase 3 report for one successful aggregation/privacy round.

    The report consumes a secure-sum backend when the selected local backend is
    runnable in-process. For backends whose production control plane is not
    available in a local smoke, the report records an explicit fallback instead
    of silently treating visible individual updates as secure aggregation.
    """

    if not updates:
        raise SecureAggregationError(
            "cannot build a Phase 3 aggregation/privacy report with zero updates",
            code=LensembleErrorCode.SECURE_AGG_FAILED,
            remediation="collect at least one participant update before reporting aggregation/privacy controls",
        )
    _validate_updates(updates, round_index=round_index)
    secure_report = _secure_aggregation_report(config, updates, round_index=round_index)
    dp_report = _dp_accounting_report(
        config,
        contributing_count=len(updates),
        participant_count=max(1, int(config.federation.participant_count)),
        round_index=round_index,
    )
    return Phase3AggregationPrivacyReport(
        consortium_id=manifest.consortium_id,
        run_id=manifest.run_id,
        round_index=round_index,
        participant_count=int(config.federation.participant_count),
        secure_aggregation=secure_report,
        dp_accounting=dp_report,
        redaction_policy=(
            "Report records aggregate counts, hashes, backend status, and DP spend only; "
            "participant ids, raw data, raw actions, latents, embeddings, private action-head "
            "weights, and individual update values are omitted."
        ),
        claim_boundary=(
            "Operational secure-aggregation and DP accounting report only; not a "
            "provenance ledger and not a cryptographic proof of honest participant computation."
        ),
    )


def _validate_updates(
    updates: "dict[str, PseudoGradient]", *, round_index: int
) -> None:
    dims = {int(update.delta.numel()) for update in updates.values()}
    if len(dims) != 1:
        raise SecureAggregationError(
            f"participant updates have incompatible flat dimensions: {sorted(dims)}",
            code=LensembleErrorCode.SECURE_AGG_FAILED,
            remediation="aggregate only updates from the same model/config contract",
        )
    bad_rounds = {
        pid: update.round_index
        for pid, update in updates.items()
        if update.round_index != round_index
    }
    if bad_rounds:
        raise SecureAggregationError(
            f"participant updates target the wrong round: {bad_rounds}",
            code=LensembleErrorCode.SECURE_AGG_FAILED,
            remediation="report only updates collected for the round being closed",
        )


def _secure_aggregation_report(
    config: "LensembleConfig",
    updates: "dict[str, PseudoGradient]",
    *,
    round_index: int,
) -> Phase3SecureAggregationReport:
    backend = config.federation.aggregation_backend
    threshold = int(config.federation.secure_agg_threshold)
    contributing = len(updates)
    field = _field_for_updates(config, updates)
    direct_sum = _direct_sum(updates)
    fixed_point_tolerance = contributing / field.scale

    if backend in {"simulated", "tee"}:
        masked = {
            pid: encode_delta(
                update.delta,
                field,
                participant_id=pid,
                round_index=round_index,
                dataset_root=update.dataset_root,
            )
            for pid, update in updates.items()
        }
        aggregator = (
            SimulatedSecureAggregator()
            if backend == "simulated"
            else TEEAggregator(_TEE_SMOKE_CODE_HASH)
        )
        secure_sum = aggregator.aggregate(
            masked,
            field=field,
            round_index=round_index,
            threshold=threshold,
        )
        error = float((secure_sum - direct_sum).abs().max())
        if error > fixed_point_tolerance + 1e-6:
            raise SecureAggregationError(
                "secure-aggregation revealed sum exceeded the fixed-point tolerance: "
                f"{error} > {fixed_point_tolerance}",
                code=LensembleErrorCode.SECURE_AGG_FAILED,
                remediation="increase the fixed-point scale or inspect the secure-aggregation backend",
            )
        return Phase3SecureAggregationReport(
            backend=backend,
            backend_status="secure_sum",
            secure_sum_consumed=True,
            fallback_used=False,
            fallback_reason=None,
            threshold=threshold,
            contributing_count=contributing,
            field_modulus=field.modulus,
            field_scale=field.scale,
            fixed_point_tolerance=fixed_point_tolerance,
            secure_sum_sha256=flat_content_hash(secure_sum),
            averaged_update_sha256=flat_content_hash(secure_sum / contributing),
            secure_sum_max_abs_error=error,
            secure_sum_matches_plaintext=True,
        )

    return Phase3SecureAggregationReport(
        backend=backend,
        backend_status="explicit_fallback",
        secure_sum_consumed=False,
        fallback_used=True,
        fallback_reason=(
            "masking backend requires pairwise key-routing and dropout-recovery "
            "shares from the production transport; this local smoke records the "
            "fallback explicitly instead of claiming a masked secure-sum reveal"
        ),
        threshold=threshold,
        contributing_count=contributing,
        field_modulus=field.modulus,
        field_scale=field.scale,
        fixed_point_tolerance=fixed_point_tolerance,
        secure_sum_sha256=None,
        averaged_update_sha256=flat_content_hash(direct_sum / contributing),
        secure_sum_max_abs_error=None,
        secure_sum_matches_plaintext=None,
    )


def _field_for_updates(
    config: "LensembleConfig", updates: "dict[str, PseudoGradient]"
) -> FieldParams:
    dim = next(iter(updates.values())).delta.numel()
    max_l2 = max(
        float(config.privacy.clip_norm),
        *(float(update.delta.norm()) for update in updates.values()),
    )
    field = FieldParams(modulus=_FIELD_MODULUS, scale=_FIELD_SCALE, dim=int(dim))
    assert_no_wrap(len(updates), max_l2, field)
    return field


def _direct_sum(updates: "dict[str, PseudoGradient]") -> torch.Tensor:
    ordered = [updates[pid].delta.to(torch.float32) for pid in sorted(updates)]
    total = torch.zeros_like(ordered[0])
    for delta in ordered:
        total = total + delta
    return total


def _dp_accounting_report(
    config: "LensembleConfig",
    *,
    contributing_count: int,
    participant_count: int,
    round_index: int,
) -> Phase3DPAccountingReport:
    privacy = config.privacy
    sample_rate = min(1.0, max(1, contributing_count) / max(1, participant_count))
    if not privacy.enabled:
        return Phase3DPAccountingReport(
            enabled=False,
            effective_dp=False,
            status="disabled",
            accountant=privacy.accountant,
            clip_norm=float(privacy.clip_norm),
            noise_multiplier=float(privacy.noise_multiplier),
            target_epsilon=float(privacy.epsilon),
            target_delta=float(privacy.delta),
            sample_rate=sample_rate,
            rounds_accounted=0,
            epsilon_spent=0.0,
        )
    if float(privacy.noise_multiplier) <= 0.0:
        return Phase3DPAccountingReport(
            enabled=True,
            effective_dp=False,
            status="noise_disabled",
            accountant=privacy.accountant,
            clip_norm=float(privacy.clip_norm),
            noise_multiplier=float(privacy.noise_multiplier),
            target_epsilon=float(privacy.epsilon),
            target_delta=float(privacy.delta),
            sample_rate=sample_rate,
            rounds_accounted=0,
            epsilon_spent=None,
        )

    accountant = build_accountant(privacy.accountant)
    if accountant.would_exceed(
        target_epsilon=float(privacy.epsilon),
        target_delta=float(privacy.delta),
        noise_multiplier=float(privacy.noise_multiplier),
        sample_rate=sample_rate,
    ):
        err = PrivacyBudgetExceeded(
            f"Phase 3 DP budget would be exceeded before round {round_index}",
            code=LensembleErrorCode.DP_BUDGET_EXCEEDED,
            remediation="raise the DP budget, increase the noise multiplier, or stop the run before release",
        )
        err.round = round_index  # type: ignore[attr-defined]
        err.epsilon_budget = float(privacy.epsilon)  # type: ignore[attr-defined]
        err.epsilon_spent = accountant.spent(  # type: ignore[attr-defined]
            target_delta=float(privacy.delta)
        )
        raise err
    accountant.step(
        noise_multiplier=float(privacy.noise_multiplier),
        sample_rate=sample_rate,
    )
    return Phase3DPAccountingReport(
        enabled=True,
        effective_dp=True,
        status="accounted",
        accountant=privacy.accountant,
        clip_norm=float(privacy.clip_norm),
        noise_multiplier=float(privacy.noise_multiplier),
        target_epsilon=float(privacy.epsilon),
        target_delta=float(privacy.delta),
        sample_rate=sample_rate,
        rounds_accounted=1,
        epsilon_spent=accountant.spent(target_delta=float(privacy.delta)),
    )


__all__ = [
    "PHASE3_AGGREGATION_PRIVACY_REPORT_SCHEMA_VERSION",
    "Phase3AggregationPrivacyReport",
    "Phase3DPAccountingReport",
    "Phase3SecureAggregationReport",
    "build_phase3_aggregation_privacy_report",
]
