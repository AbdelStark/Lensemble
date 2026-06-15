"""Phase 2 final evidence bundle and model-card generation.

The bundle is a residency-safe, reviewer-facing aggregate over the Phase 2
dataset, training, downstream-eval, and baselines/curves reports. It validates
that referenced Hub artifacts exist before it emits the final model-card text.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lensemble.errors import ConfigError, LensembleErrorCode, SchemaVersionMismatch
from lensemble.eval.claim_mvp import ClaimMVPReport
from lensemble.eval.phase2_curves import Phase2BaselinesCurvesReport
from lensemble.eval.phase2_downstream import Phase2DownstreamEvalReport

PHASE2_EVIDENCE_BUNDLE_SCHEMA_VERSION = 1

Phase2ArtifactKind = Literal[
    "dataset-smoke-report",
    "dataset-split-manifest",
    "dataset-silo",
    "training-claim-report",
    "training-checkpoint-header",
    "downstream-eval-report",
    "baselines-curves-report",
]
RepoType = Literal["model", "dataset"]


class Phase2HubArtifactCheck(BaseModel):
    """Existence check for one referenced Hub artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Phase2ArtifactKind
    label: str = Field(min_length=1)
    repo_type: RepoType
    repo_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    path_in_repo: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    checked_at: datetime
    exists: bool
    status_code: int | None = None
    error: str | None = None


class Phase2ParticipantBundleSummary(BaseModel):
    """One participant's public dataset evidence in the final bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    participant_id: str = Field(min_length=1)
    hf_file_ref: str = Field(min_length=1)
    episode_count: int = Field(ge=1)
    window_count: int = Field(ge=0)
    dataset_root: str = Field(pattern=r"^[0-9a-f]{64}$")
    silo_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    action_dim: int = Field(ge=1)
    observation_shape: tuple[int, ...]
    action_shape: tuple[int, ...]


class Phase2DatasetBundleSummary(BaseModel):
    """Dataset refs, split policy, and residency-safe data metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_repo_id: str = Field(min_length=1)
    dataset_revision: str = Field(min_length=1)
    source_repo_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_policy: str = Field(min_length=1)
    heldout_split_policy: str = Field(min_length=1)
    participants: tuple[Phase2ParticipantBundleSummary, ...] = Field(min_length=1)


class Phase2TrainingBundleSummary(BaseModel):
    """Published Phase 2 federated training evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str = Field(min_length=1)
    job_url: str = Field(min_length=1)
    code_sha: str = Field(min_length=1)
    checkpoint_repo_id: str = Field(min_length=1)
    checkpoint_revision: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    final_global_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    committed_rounds: int = Field(ge=1)
    val_pred: float
    val_sigreg: float
    effective_rank: float
    frame_drift_deg: float
    run_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class Phase2DownstreamBundleSummary(BaseModel):
    """Published downstream eval evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str = Field(min_length=1)
    job_url: str = Field(min_length=1)
    report_revision: str = Field(min_length=1)
    env_id: str = Field(min_length=1)
    planner: str = Field(min_length=1)
    success_rate: float
    time_per_action_ms: float
    effective_dim: float
    eval_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = Field(min_length=1)


class Phase2CurvesBundleSummary(BaseModel):
    """Baseline/curve coverage and blockers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report_revision: str = Field(min_length=1)
    curve_point_count: int = Field(ge=1)
    run_roles: tuple[str, ...] = Field(min_length=1)
    blocked_comparisons: tuple[str, ...]
    model_card_baseline_text: str = Field(min_length=1)


class Phase2EvidenceBundle(BaseModel):
    """Final Phase 2 evidence bundle consumed by the model card."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    generated_at: datetime
    bundle: Literal["phase2-federated-leworldmodel-evidence"]
    artifact_checks: tuple[Phase2HubArtifactCheck, ...] = Field(min_length=1)
    dataset: Phase2DatasetBundleSummary
    training: Phase2TrainingBundleSummary
    downstream_eval: Phase2DownstreamBundleSummary
    baselines_curves: Phase2CurvesBundleSummary
    claim_boundaries: tuple[str, ...] = Field(min_length=1)
    known_gaps: tuple[str, ...]
    raw_data_in_report: Literal[False] = False
    model_card_markdown: str = Field(min_length=1)

    @model_validator(mode="after")
    def _cross_check(self) -> "Phase2EvidenceBundle":
        if self.schema_version != PHASE2_EVIDENCE_BUNDLE_SCHEMA_VERSION:
            raise SchemaVersionMismatch(
                f"phase2 evidence bundle schema_version {self.schema_version!r} "
                f"exceeds reader max {PHASE2_EVIDENCE_BUNDLE_SCHEMA_VERSION}",
                code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
                remediation="read with a build supporting this phase2 bundle schema",
            )
        missing = [check for check in self.artifact_checks if not check.exists]
        if missing:
            labels = ", ".join(check.label for check in missing)
            raise ConfigError(
                f"phase2 evidence bundle has missing artifact checks: {labels}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="publish or fix the referenced artifacts before claiming Phase 2 success",
            )
        required_kinds = {
            "dataset-smoke-report",
            "dataset-split-manifest",
            "dataset-silo",
            "training-claim-report",
            "training-checkpoint-header",
            "downstream-eval-report",
            "baselines-curves-report",
        }
        seen_kinds = {check.kind for check in self.artifact_checks}
        missing_kinds = required_kinds - seen_kinds
        if missing_kinds:
            raise ConfigError(
                "phase2 evidence bundle is missing artifact kinds: "
                + ", ".join(sorted(missing_kinds)),
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="include every data/train/eval/curve artifact class",
            )
        if "does not claim" not in self.model_card_markdown.lower():
            raise ConfigError(
                "model-card text must include explicit non-claims",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="state paper-scale and exhaustive-baseline boundaries",
            )
        return self


def parse_phase2_evidence_bundle(raw: dict[str, Any]) -> Phase2EvidenceBundle:
    """Parse a Phase 2 evidence bundle, gating schema version first."""

    version = raw.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version > PHASE2_EVIDENCE_BUNDLE_SCHEMA_VERSION
    ):
        raise SchemaVersionMismatch(
            f"phase2 evidence bundle schema_version {version!r} exceeds reader max "
            f"{PHASE2_EVIDENCE_BUNDLE_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation="read with a build supporting this phase2 bundle schema",
        )
    return Phase2EvidenceBundle.model_validate(raw)


def build_phase2_evidence_bundle(
    *,
    dataset_smoke: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    training_report: ClaimMVPReport,
    downstream_report: Phase2DownstreamEvalReport,
    curves_report: Phase2BaselinesCurvesReport,
    artifact_checks: Sequence[Phase2HubArtifactCheck],
    dataset_revision: str,
    checkpoint_revision: str,
    curves_revision: str,
    generated_at: datetime | None = None,
) -> Phase2EvidenceBundle:
    """Build the final Phase 2 evidence bundle from validated inputs."""

    dataset = _dataset_summary(
        dataset_smoke, dataset_manifest, dataset_revision=dataset_revision
    )
    training = _training_summary(
        training_report,
        downstream_report=downstream_report,
        checkpoint_revision=checkpoint_revision,
    )
    downstream = _downstream_summary(downstream_report)
    curves = _curves_summary(curves_report, curves_revision=curves_revision)
    claim_boundaries = (
        "Engineering-scale evidence: published SO-100 participant silos, a GPU-backed three-round federated JEPA-style run, downstream synthetic planning eval, and a matched lambda_anc=0 control.",
        "Does not claim paper-scale LeWorldModel performance, SO-100 task success, broad robotics generalization, or completed RFC-0006 cryptographic contribution proofs.",
        "Baseline coverage is partial; blocked comparisons remain blocked until matched public runs exist.",
    )
    known_gaps = tuple(
        f"{item.comparison}: {item.reason}"
        for item in curves_report.blocked_comparisons
    )
    model_card = render_phase2_model_card(
        dataset=dataset,
        training=training,
        downstream=downstream,
        curves=curves,
        claim_boundaries=claim_boundaries,
        known_gaps=known_gaps,
    )
    return Phase2EvidenceBundle(
        schema_version=PHASE2_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        generated_at=generated_at or datetime.now(timezone.utc),
        bundle="phase2-federated-leworldmodel-evidence",
        artifact_checks=tuple(artifact_checks),
        dataset=dataset,
        training=training,
        downstream_eval=downstream,
        baselines_curves=curves,
        claim_boundaries=claim_boundaries,
        known_gaps=known_gaps,
        model_card_markdown=model_card,
    )


def check_hf_artifact_exists(
    *,
    kind: Phase2ArtifactKind,
    label: str,
    repo_type: RepoType,
    repo_id: str,
    revision: str,
    path_in_repo: str,
    checked_at: datetime | None = None,
    timeout: float = 20.0,
) -> Phase2HubArtifactCheck:
    """Check a public Hub file by HTTP HEAD, falling back to a 1-byte GET."""

    uri = _hf_uri(repo_type, repo_id, revision, path_in_repo)
    url = _hf_resolve_url(repo_type, repo_id, revision, path_in_repo)
    checked = checked_at or datetime.now(timezone.utc)
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return Phase2HubArtifactCheck(
                kind=kind,
                label=label,
                repo_type=repo_type,
                repo_id=repo_id,
                revision=revision,
                path_in_repo=path_in_repo,
                uri=uri,
                checked_at=checked,
                exists=200 <= int(response.status) < 400,
                status_code=int(response.status),
            )
    except urllib.error.HTTPError as exc:
        if exc.code not in {403, 405}:
            return _failed_check(
                kind,
                label,
                repo_type,
                repo_id,
                revision,
                path_in_repo,
                uri,
                checked,
                exc.code,
                str(exc),
            )
    except urllib.error.URLError as exc:
        return _failed_check(
            kind,
            label,
            repo_type,
            repo_id,
            revision,
            path_in_repo,
            uri,
            checked,
            None,
            str(exc),
        )

    get_request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(get_request, timeout=timeout) as response:
            return Phase2HubArtifactCheck(
                kind=kind,
                label=label,
                repo_type=repo_type,
                repo_id=repo_id,
                revision=revision,
                path_in_repo=path_in_repo,
                uri=uri,
                checked_at=checked,
                exists=200 <= int(response.status) < 400,
                status_code=int(response.status),
            )
    except urllib.error.HTTPError as exc:
        return _failed_check(
            kind,
            label,
            repo_type,
            repo_id,
            revision,
            path_in_repo,
            uri,
            checked,
            exc.code,
            str(exc),
        )
    except urllib.error.URLError as exc:
        return _failed_check(
            kind,
            label,
            repo_type,
            repo_id,
            revision,
            path_in_repo,
            uri,
            checked,
            None,
            str(exc),
        )


def local_artifact_check(
    *,
    kind: Phase2ArtifactKind,
    label: str,
    repo_type: RepoType,
    repo_id: str,
    revision: str,
    path_in_repo: str,
) -> Phase2HubArtifactCheck:
    """Build a passing artifact check without network access for tests."""

    return Phase2HubArtifactCheck(
        kind=kind,
        label=label,
        repo_type=repo_type,
        repo_id=repo_id,
        revision=revision,
        path_in_repo=path_in_repo,
        uri=_hf_uri(repo_type, repo_id, revision, path_in_repo),
        checked_at=datetime.now(timezone.utc),
        exists=True,
        status_code=None,
    )


def render_phase2_model_card(
    *,
    dataset: Phase2DatasetBundleSummary,
    training: Phase2TrainingBundleSummary,
    downstream: Phase2DownstreamBundleSummary,
    curves: Phase2CurvesBundleSummary,
    claim_boundaries: Sequence[str],
    known_gaps: Sequence[str],
) -> str:
    """Render the Phase 2 checkpoint repo README/model card."""

    participants = "\n".join(
        "| "
        + " | ".join(
            (
                item.participant_id,
                item.hf_file_ref,
                str(item.episode_count),
                str(item.window_count),
                f"`{item.dataset_root}`",
            )
        )
        + " |"
        for item in dataset.participants
    )
    gaps = "\n".join(f"- {gap}" for gap in known_gaps) or "- none"
    boundaries = "\n".join(f"- {boundary}" for boundary in claim_boundaries)
    return f"""---
license: apache-2.0
library_name: lensemble
tags:
- federated-learning
- world-model
- jepa
- robotics
- phase2
---

# Lensemble Phase 2 SO-100 Federated JEPA World Model

This model repository contains the Phase 2 engineering evidence bundle for a
federated JEPA-style world-model run over two public SO-100 participant silos.

## Dataset Refs

Dataset repo: `hf://datasets/{dataset.dataset_repo_id}@{dataset.dataset_revision}`

| Participant | File ref | Episodes | Windows | Dataset root |
|---|---|---:|---:|---|
{participants}

Split policy: `{dataset.split_policy}`. Held-out policy:
{dataset.heldout_split_policy}

## Training

- HF Job: [{training.job_id}]({training.job_url})
- Pinned code SHA: `{training.code_sha}`
- Checkpoint revision: `{training.checkpoint_repo_id}@{training.checkpoint_revision}`
- Config hash: `{training.config_hash}`
- Final global hash: `{training.final_global_hash}`
- Committed rounds: {training.committed_rounds}
- Metrics: `val_pred={training.val_pred}`, `val_sigreg={training.val_sigreg}`,
  `effective_rank={training.effective_rank}`,
  `frame_drift_deg={training.frame_drift_deg}`

## Downstream Eval

- HF Job: [{downstream.job_id}]({downstream.job_url})
- Env/planner: `{downstream.env_id}` / `{downstream.planner}`
- Success rate: {downstream.success_rate}
- Time per action: {downstream.time_per_action_ms} ms
- Effective dimension: {downstream.effective_dim}
- Eval config hash: `{downstream.eval_config_hash}`

## Baselines And Curves

The generated curve report has {curves.curve_point_count} rows over
{", ".join(f"`{role}`" for role in curves.run_roles)}. {curves.model_card_baseline_text}

## Claim Boundaries

{boundaries}

## Known Gaps

{gaps}

## Public Reports

- `reports/phase2_downstream_eval_report.json`
- `reports/phase2_baselines_curves_report.json`
- `reports/phase2_evidence_bundle.json`
"""


def write_phase2_bundle_outputs(
    bundle: Phase2EvidenceBundle,
    *,
    bundle_path: Path,
    model_card_path: Path,
) -> None:
    """Write bundle JSON and model-card markdown to disk."""

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    model_card_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(
        json.dumps(bundle.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    model_card_path.write_text(bundle.model_card_markdown, encoding="utf-8")


def _dataset_summary(
    smoke: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    dataset_revision: str,
) -> Phase2DatasetBundleSummary:
    manifest_by_file = {
        str(item["filename"]): item for item in _as_sequence(manifest["silos"])
    }
    participants: list[Phase2ParticipantBundleSummary] = []
    for silo in _as_sequence(smoke["silos"]):
        hf_ref = str(silo["hf_file_ref"])
        filename = hf_ref.rsplit("/", 1)[-1]
        manifest_item = manifest_by_file.get(filename, {})
        participants.append(
            Phase2ParticipantBundleSummary(
                participant_id=str(silo["participant_id"]),
                hf_file_ref=hf_ref,
                episode_count=int(silo["episode_count"]),
                window_count=int(silo["window_count"]),
                dataset_root=str(silo["dataset_root"]),
                silo_sha256=(
                    str(manifest_item["sha256"]) if "sha256" in manifest_item else None
                ),
                action_dim=int(_as_mapping(silo["action_spec"])["dim"]),
                observation_shape=tuple(int(dim) for dim in silo["observation_shape"]),
                action_shape=tuple(int(dim) for dim in silo["action_shape"]),
            )
        )
    heldout = _as_mapping(manifest["heldout_split_policy"])
    return Phase2DatasetBundleSummary(
        dataset_repo_id=str(smoke["dataset_repo_id"]),
        dataset_revision=dataset_revision,
        source_repo_id=str(manifest["source_repo_id"]),
        source_file=str(manifest["source_file"]),
        source_sha256=str(manifest["source_sha256"]),
        split_policy=str(manifest["policy"]),
        heldout_split_policy=f"{heldout['policy']} for {heldout['purpose']}",
        participants=tuple(participants),
    )


def _training_summary(
    report: ClaimMVPReport,
    *,
    downstream_report: Phase2DownstreamEvalReport,
    checkpoint_revision: str,
) -> Phase2TrainingBundleSummary:
    metrics = report.metrics
    if (
        metrics.val_pred is None
        or metrics.val_sigreg is None
        or metrics.effective_rank is None
        or metrics.frame_drift_deg is None
        or metrics.run_manifest_hash is None
    ):
        raise ConfigError(
            "Phase 2 training report is missing final scalar metrics",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="use the published Phase 2 claim report with metrics populated",
        )
    checkpoint = downstream_report.checkpoint
    if checkpoint.checkpoint_hash != report.final_global_hash:
        raise ConfigError(
            "downstream checkpoint hash does not match training final global hash",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="bundle the downstream report for the same checkpoint",
        )
    return Phase2TrainingBundleSummary(
        job_id=str(checkpoint.training_job_id),
        job_url=str(checkpoint.training_job_url),
        code_sha=str(checkpoint.code_sha),
        checkpoint_repo_id=str(report.publication.checkpoint_repo),
        checkpoint_revision=checkpoint_revision,
        config_hash=report.config_hash,
        final_global_hash=report.final_global_hash,
        committed_rounds=report.committed_rounds,
        val_pred=metrics.val_pred,
        val_sigreg=metrics.val_sigreg,
        effective_rank=metrics.effective_rank,
        frame_drift_deg=metrics.frame_drift_deg,
        run_manifest_hash=metrics.run_manifest_hash,
    )


def _downstream_summary(
    report: Phase2DownstreamEvalReport,
) -> Phase2DownstreamBundleSummary:
    return Phase2DownstreamBundleSummary(
        job_id="6a22c9e3ece949d7b3dca25a",
        job_url="https://huggingface.co/jobs/abdelstark/6a22c9e3ece949d7b3dca25a",
        report_revision="021a461eb789700209fcb49e99bb9bcc5d84bfe5",
        env_id=report.eval_report.env_id,
        planner=report.eval_report.planner,
        success_rate=report.eval_report.success_rate,
        time_per_action_ms=report.eval_report.time_per_action_ms,
        effective_dim=report.eval_report.effective_dim,
        eval_config_hash=report.eval_config_hash,
        claim_boundary=report.claim_boundary,
    )


def _curves_summary(
    report: Phase2BaselinesCurvesReport,
    *,
    curves_revision: str,
) -> Phase2CurvesBundleSummary:
    return Phase2CurvesBundleSummary(
        report_revision=curves_revision,
        curve_point_count=len(report.curve_points),
        run_roles=tuple(sorted({point.run_role for point in report.curve_points})),
        blocked_comparisons=tuple(
            item.comparison for item in report.blocked_comparisons
        ),
        model_card_baseline_text=report.model_card_baseline_text,
    )


def _hf_resolve_url(
    repo_type: RepoType, repo_id: str, revision: str, path_in_repo: str
) -> str:
    prefix = "datasets/" if repo_type == "dataset" else ""
    encoded_path = "/".join(
        urllib.parse.quote(part) for part in path_in_repo.split("/")
    )
    return f"https://huggingface.co/{prefix}{repo_id}/resolve/{revision}/{encoded_path}"


def _hf_uri(repo_type: RepoType, repo_id: str, revision: str, path_in_repo: str) -> str:
    prefix = "datasets" if repo_type == "dataset" else "models"
    return f"hf://{prefix}/{repo_id}@{revision}/{path_in_repo}"


def _failed_check(
    kind: Phase2ArtifactKind,
    label: str,
    repo_type: RepoType,
    repo_id: str,
    revision: str,
    path_in_repo: str,
    uri: str,
    checked_at: datetime,
    status_code: int | None,
    error: str,
) -> Phase2HubArtifactCheck:
    return Phase2HubArtifactCheck(
        kind=kind,
        label=label,
        repo_type=repo_type,
        repo_id=repo_id,
        revision=revision,
        path_in_repo=path_in_repo,
        uri=uri,
        checked_at=checked_at,
        exists=False,
        status_code=status_code,
        error=error,
    )


def _as_sequence(value: object) -> Sequence[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"expected JSON array, got {type(value).__name__}")
    return [dict(_as_mapping(item)) for item in value]


def _as_mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"expected JSON object, got {type(value).__name__}")
    return value
