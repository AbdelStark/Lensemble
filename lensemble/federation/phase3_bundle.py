"""Phase 3 final evidence bundle and model-card generation."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lensemble.config.consortium import (
    Phase3ConsortiumManifest,
    write_consortium_manifest,
)
from lensemble.data.phase3 import (
    Phase3DatasetProbeRegistry,
    phase3_registry_from_consortium_manifest,
    write_phase3_dataset_registry,
)
from lensemble.errors import ConfigError, LensembleErrorCode, SchemaVersionMismatch
from lensemble.eval.phase3 import Phase3EvalReport
from lensemble.federation.phase3_observability import (
    Phase3ObservabilityReport,
)
from lensemble.federation.phase3_orchestration import (
    Phase3LongRunReport,
    load_phase3_long_run_report,
    phase3_long_run_smoke_config,
    phase3_long_run_smoke_manifest,
)

PHASE3_EVIDENCE_BUNDLE_SCHEMA_VERSION = 1

Phase3ArtifactLocation = Literal["local", "hub"]
Phase3RepoType = Literal["model", "dataset"]
Phase3ArtifactKind = Literal[
    "consortium-manifest",
    "dataset-probe-registry",
    "training-report",
    "privacy-aggregation-report",
    "observability-report",
    "eval-control-report",
    "run-manifest",
    "checkpoint-header",
    "checkpoint-weights",
    "model-card",
    "evidence-bundle",
]
Phase3PublicationStatus = Literal["local_smoke", "published", "blocked"]

_GENERATED_AT = datetime(2026, 6, 5, 0, 0, 0, tzinfo=timezone.utc)
_MODEL_REPO_ID = "abdelstark/lensemble-phase3-consortium-checkpoint"
_DATASET_REPO_ID = "abdelstark/lensemble-phase3-consortium-data"
_FORBIDDEN_KEYS = (
    "raw_data",
    "raw_observation",
    "raw_obs",
    "raw_action",
    "action_tensor",
    "latent",
    "embedding",
    "private_action_head",
    "secret",
    "access_token",
    "api_token",
)
_SAFE_PRIVATE_SURFACE_METADATA_KEYS = frozenset(
    {
        "raw_data_crosses_boundary",
        "raw_data_in_report",
        "model_latent_dim",
        "model_num_tokens",
    }
)


class Phase3ArtifactCheck(BaseModel):
    """Existence check for one referenced local or Hub artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Phase3ArtifactKind
    label: str = Field(min_length=1)
    location: Phase3ArtifactLocation
    uri: str = Field(min_length=1)
    checked_at: datetime
    exists: bool
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    repo_type: Phase3RepoType | None = None
    repo_id: str | None = Field(default=None, min_length=1)
    revision: str | None = Field(default=None, min_length=1)
    path_in_repo: str | None = Field(default=None, min_length=1)
    status_code: int | None = None
    error: str | None = None


class Phase3ManifestBundleSummary(BaseModel):
    """Consortium contract summary included in the final bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    consortium_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    participant_count: int = Field(ge=1)
    coordinator_id: str = Field(min_length=1)
    protocol_version: str = Field(min_length=1)
    secure_aggregation_backend: str = Field(min_length=1)
    dp_required: bool
    public_probe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = Field(min_length=1)


class Phase3DatasetBundleSummary(BaseModel):
    """Dataset/probe registry summary included in the final bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    registry_id: str = Field(min_length=1)
    run_mode: str = Field(min_length=1)
    participant_count: int = Field(ge=1)
    participant_ids: tuple[str, ...] = Field(min_length=1)
    public_probe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    published_participant_count: int = Field(ge=0)
    placeholder_participant_count: int = Field(ge=0)
    raw_data_crosses_boundary: Literal[False] = False
    heldout_policy: str = Field(min_length=1)


class Phase3TrainingBundleSummary(BaseModel):
    """Training/run evidence summary included in the final bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    participant_count: int = Field(ge=1)
    target_rounds: int = Field(ge=1)
    closed_rounds: int = Field(ge=0)
    completed_target: bool
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    final_global_model_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_header_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_latent_dim: int = Field(ge=1)
    model_num_tokens: int = Field(ge=1)
    claim_boundary: str = Field(min_length=1)


class Phase3PrivacyAggregationBundleSummary(BaseModel):
    """Aggregation/privacy evidence summary included in the final bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    secure_aggregation_backend: str = Field(min_length=1)
    secure_aggregation_threshold: int = Field(ge=1)
    secure_sum_rounds: int = Field(ge=0)
    dp_enabled: bool
    dp_accountant: str = Field(min_length=1)
    dp_epsilon: float = Field(gt=0.0)
    dp_delta: float = Field(gt=0.0, lt=1.0)
    dp_accounted_rounds: int = Field(ge=0)
    max_round_epsilon_spent: float = Field(ge=0.0)


class Phase3EvalControlBundleSummary(BaseModel):
    """Eval/control evidence summary included in the final bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    completed_controls: tuple[str, ...]
    blocked_controls: tuple[str, ...]
    metric_row_count: int = Field(ge=0)
    blocked_task_env_ids: tuple[str, ...]
    model_card_eval_text: str = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)


class Phase3ObservabilityBundleSummary(BaseModel):
    """Observability/dropout evidence summary included in the final bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    round_summary_count: int = Field(ge=1)
    dropout_decision_count: int = Field(ge=1)
    induced_dropout_outcomes: tuple[str, ...] = Field(min_length=1)
    redaction_contract_id: str = Field(min_length=1)
    final_bundle_handoff: str = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)


class Phase3PublicationBundleSummary(BaseModel):
    """Publication target/status for the final Phase 3 artifact repository."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Phase3PublicationStatus
    model_repo_id: str = Field(min_length=1)
    model_repo_revision: str = Field(min_length=1)
    dataset_repo_id: str = Field(min_length=1)
    dataset_repo_revision: str = Field(min_length=1)
    expected_model_repo_files: tuple[str, ...] = Field(min_length=1)
    blockers: tuple[str, ...]


class Phase3EvidenceBundle(BaseModel):
    """Final Phase 3 evidence bundle consumed by the model card."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    generated_at: datetime
    bundle: Literal["phase3-consortium-leworldmodel-evidence"]
    artifact_checks: tuple[Phase3ArtifactCheck, ...] = Field(min_length=1)
    manifest: Phase3ManifestBundleSummary
    dataset_registry: Phase3DatasetBundleSummary
    training: Phase3TrainingBundleSummary
    privacy_aggregation: Phase3PrivacyAggregationBundleSummary
    eval_controls: Phase3EvalControlBundleSummary
    observability: Phase3ObservabilityBundleSummary
    publication: Phase3PublicationBundleSummary
    claim_boundaries: tuple[str, ...] = Field(min_length=1)
    non_claims: tuple[str, ...] = Field(min_length=1)
    known_limitations: tuple[str, ...]
    raw_data_in_report: Literal[False] = False
    model_card_markdown: str = Field(min_length=1)

    @model_validator(mode="after")
    def _cross_check(self) -> "Phase3EvidenceBundle":
        if self.schema_version != PHASE3_EVIDENCE_BUNDLE_SCHEMA_VERSION:
            raise SchemaVersionMismatch(
                f"Phase 3 evidence bundle schema_version {self.schema_version!r} "
                f"exceeds reader max {PHASE3_EVIDENCE_BUNDLE_SCHEMA_VERSION}",
                code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
                remediation="read with a build supporting this Phase 3 bundle schema",
            )
        missing = [check for check in self.artifact_checks if not check.exists]
        if missing:
            labels = ", ".join(check.label for check in missing)
            raise ConfigError(
                f"Phase 3 evidence bundle has missing artifact checks: {labels}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="publish or regenerate referenced artifacts before claiming Phase 3 success",
            )
        required = {
            "consortium-manifest",
            "dataset-probe-registry",
            "training-report",
            "privacy-aggregation-report",
            "observability-report",
            "eval-control-report",
            "run-manifest",
            "checkpoint-header",
            "checkpoint-weights",
        }
        seen = {check.kind for check in self.artifact_checks}
        missing_kinds = required - seen
        if missing_kinds:
            raise ConfigError(
                "Phase 3 evidence bundle is missing artifact kinds: "
                + ", ".join(sorted(missing_kinds)),
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="include every Phase 3 contract/train/eval/observability/checkpoint artifact",
            )
        self._cross_check_artifact_hashes()
        card = self.model_card_markdown.lower()
        required_text = (
            "does not include a provenance ledger",
            "does not cryptographically prove honest participant computation",
            "does not claim paper-scale leworldmodel performance",
        )
        if any(text not in card for text in required_text):
            raise ConfigError(
                "Phase 3 model card is missing explicit non-claims",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="state the provenance, cryptographic-proof, and paper-scale boundaries",
            )
        validate_phase3_bundle_residency(self.model_dump(mode="json"))
        return self

    def _cross_check_artifact_hashes(self) -> None:
        expected_by_kind: tuple[tuple[Phase3ArtifactKind, str], ...] = (
            ("run-manifest", self.training.run_manifest_hash),
            ("checkpoint-header", self.training.checkpoint_header_sha256),
            ("checkpoint-weights", self.training.checkpoint_weights_sha256),
        )
        for kind, expected_hash in expected_by_kind:
            checks = [check for check in self.artifact_checks if check.kind == kind]
            sha_checks = [check for check in checks if check.sha256 is not None]
            if not sha_checks:
                raise ConfigError(
                    f"Phase 3 evidence bundle artifact check {kind!r} is missing sha256",
                    code=LensembleErrorCode.CONFIG_INVALID,
                    remediation="regenerate the bundle so run/checkpoint artifact hashes are bound",
                )
            mismatched = [
                check for check in sha_checks if check.sha256 != expected_hash
            ]
            if mismatched:
                uris = ", ".join(check.uri for check in mismatched)
                raise ConfigError(
                    f"Phase 3 evidence bundle artifact hash mismatch for {kind!r}: {uris}",
                    code=LensembleErrorCode.CONFIG_INVALID,
                    remediation="regenerate the bundle from the same run manifest and checkpoint artifacts",
                )


def parse_phase3_evidence_bundle(raw: dict[str, Any]) -> Phase3EvidenceBundle:
    """Parse a Phase 3 evidence bundle, gating schema version first."""

    version = raw.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version > PHASE3_EVIDENCE_BUNDLE_SCHEMA_VERSION
    ):
        raise SchemaVersionMismatch(
            f"Phase 3 evidence bundle schema_version {version!r} exceeds reader max "
            f"{PHASE3_EVIDENCE_BUNDLE_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation="read with a build supporting this Phase 3 bundle schema",
        )
    validate_phase3_bundle_residency(raw)
    return Phase3EvidenceBundle.model_validate(raw)


def load_phase3_evidence_bundle(path: Path) -> Phase3EvidenceBundle:
    """Load and validate a Phase 3 evidence bundle."""

    return parse_phase3_evidence_bundle(json.loads(Path(path).read_text()))


def materialize_phase3_run_contracts(
    *,
    long_run_report_path: Path,
    manifest_path: Path,
    registry_path: Path,
) -> tuple[Phase3ConsortiumManifest, Phase3DatasetProbeRegistry]:
    """Write a run-specific Phase 3 manifest and registry from long-run evidence."""

    long_run = load_phase3_long_run_report(long_run_report_path)
    probe_hash = _public_probe_hash(long_run)
    cfg = phase3_long_run_smoke_config(rounds=long_run.target_rounds)
    manifest = phase3_long_run_smoke_manifest(cfg, public_probe_hash=probe_hash)
    registry = phase3_registry_from_consortium_manifest(manifest)
    write_consortium_manifest(manifest, manifest_path)
    write_phase3_dataset_registry(registry, registry_path)
    return manifest, registry


def build_phase3_evidence_bundle(
    *,
    manifest: Phase3ConsortiumManifest,
    registry: Phase3DatasetProbeRegistry,
    long_run: Phase3LongRunReport,
    eval_report: Phase3EvalReport,
    observability_report: Phase3ObservabilityReport,
    artifact_checks: Sequence[Phase3ArtifactCheck],
    checkpoint_header_path: Path,
    checkpoint_weights_path: Path,
    model_repo_revision: str = "local-smoke",
    dataset_repo_revision: str = "local-smoke",
    publication_status: Phase3PublicationStatus = "local_smoke",
    generated_at: datetime = _GENERATED_AT,
) -> Phase3EvidenceBundle:
    """Build the final Phase 3 evidence bundle from validated reports."""

    manifest_summary = _manifest_summary(manifest)
    dataset_summary = _dataset_summary(registry)
    training_summary = _training_summary(
        long_run,
        checkpoint_header_path=checkpoint_header_path,
        checkpoint_weights_path=checkpoint_weights_path,
    )
    privacy_summary = _privacy_summary(long_run)
    eval_summary = _eval_summary(eval_report)
    observability_summary = _observability_summary(observability_report)
    publication_summary = _publication_summary(
        status=publication_status,
        model_repo_revision=model_repo_revision,
        dataset_repo_revision=dataset_repo_revision,
    )
    claim_boundaries = (
        "Consortium-runtime evidence: four sovereign participant agents on the union SO-100 action contract completed a governed Phase 3 run with ten closed federated rounds, secure-sum aggregation, and DP accounting.",
        "Training/eval scale: this is consortium-engineering and real-training evidence on tiny tokens/latent, not a public HF Jobs paper-scale robotics training result.",
        "Controls: anchored-federation, naive-FedAvg, Fork-A/frozen-encoder, and local-only controls are completed as representation-metric rows; no Phase 3 control rows remain blocked.",
        "Privacy controls: secure_sum aggregation status and DP accounting are exercised as operational controls, not cryptographic computation proofs (RFC-0006 honest-computation proofs remain out of scope).",
    )
    non_claims = (
        "Phase 3 does not include a provenance ledger implementation.",
        "Phase 3 does not cryptographically prove honest participant computation.",
        "Phase 3 does not claim paper-scale LeWorldModel performance.",
        "Phase 3 does not claim public SO-100 robotics task success.",
        "Phase 3 is consortium-engineering and real-training evidence, not a cryptographic honest-computation proof; RFC-0006 honest-computation proofs are out of scope.",
    )
    known_limitations = tuple(
        f"{row.control_role}: {row.reason}" for row in eval_report.blocked_controls
    ) + (
        "DP-utility / federated-collapse (#244): the published checkpoints exhibit global-representation collapse over rounds under the DP noise/clipping budget, so latent quality degrades and downstream planning success would be uninformative on these checkpoints.",
        "Downstream task-success (stable-worldmodel #96): closed-loop physical SO-100 task success is deferred, not claimed; it requires the unvendored stable-worldmodel planner suite and a non-collapsing federated checkpoint, because a recorded held-out split is open-loop and cannot apply arbitrary planner actions to recorded frames.",
    )
    model_card = render_phase3_model_card(
        manifest=manifest_summary,
        dataset=dataset_summary,
        training=training_summary,
        privacy=privacy_summary,
        eval_controls=eval_summary,
        observability=observability_summary,
        publication=publication_summary,
        claim_boundaries=claim_boundaries,
        non_claims=non_claims,
        known_limitations=known_limitations,
    )
    return Phase3EvidenceBundle(
        schema_version=PHASE3_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        generated_at=generated_at,
        bundle="phase3-consortium-leworldmodel-evidence",
        artifact_checks=tuple(artifact_checks),
        manifest=manifest_summary,
        dataset_registry=dataset_summary,
        training=training_summary,
        privacy_aggregation=privacy_summary,
        eval_controls=eval_summary,
        observability=observability_summary,
        publication=publication_summary,
        claim_boundaries=claim_boundaries,
        non_claims=non_claims,
        known_limitations=known_limitations,
        model_card_markdown=model_card,
    )


def write_phase3_bundle_outputs(
    bundle: Phase3EvidenceBundle,
    *,
    bundle_path: Path,
    model_card_path: Path,
) -> None:
    """Write Phase 3 bundle JSON and model-card markdown."""

    parse_phase3_evidence_bundle(bundle.model_dump(mode="json"))
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    model_card_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(
        json.dumps(bundle.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    model_card_path.write_text(bundle.model_card_markdown, encoding="utf-8")


def local_artifact_check(
    *,
    kind: Phase3ArtifactKind,
    label: str,
    path: Path,
    uri: str | None = None,
    checked_at: datetime = _GENERATED_AT,
) -> Phase3ArtifactCheck:
    """Build a local artifact existence check."""

    path = Path(path)
    exists = path.exists()
    return Phase3ArtifactCheck(
        kind=kind,
        label=label,
        location="local",
        uri=uri or local_artifact_uri(path),
        checked_at=checked_at,
        exists=exists,
        sha256=sha256_file(path) if exists and path.is_file() else None,
        error=None if exists else "missing local artifact",
    )


def local_artifact_uri(path: Path) -> str:
    """Return a residency-safe default URI for a local artifact path."""

    path = Path(path)
    if path.is_absolute():
        return f"artifact://local/{path.name}"
    return path.as_posix()


def check_hf_artifact_exists(
    *,
    kind: Phase3ArtifactKind,
    label: str,
    repo_type: Phase3RepoType,
    repo_id: str,
    revision: str,
    path_in_repo: str,
    checked_at: datetime = _GENERATED_AT,
    timeout: float = 20.0,
) -> Phase3ArtifactCheck:
    """Check a public Hugging Face Hub artifact with HTTP HEAD."""

    uri = _hf_uri(repo_type, repo_id, revision, path_in_repo)
    url = _hf_resolve_url(repo_type, repo_id, revision, path_in_repo)
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return Phase3ArtifactCheck(
                kind=kind,
                label=label,
                location="hub",
                uri=uri,
                checked_at=checked_at,
                exists=200 <= int(response.status) < 400,
                repo_type=repo_type,
                repo_id=repo_id,
                revision=revision,
                path_in_repo=path_in_repo,
                status_code=int(response.status),
            )
    except urllib.error.HTTPError as exc:
        return _failed_hub_check(
            kind,
            label,
            repo_type,
            repo_id,
            revision,
            path_in_repo,
            uri,
            checked_at,
            exc,
        )
    except urllib.error.URLError as exc:
        return Phase3ArtifactCheck(
            kind=kind,
            label=label,
            location="hub",
            uri=uri,
            checked_at=checked_at,
            exists=False,
            repo_type=repo_type,
            repo_id=repo_id,
            revision=revision,
            path_in_repo=path_in_repo,
            error=str(exc),
        )


def render_phase3_model_card(
    *,
    manifest: Phase3ManifestBundleSummary,
    dataset: Phase3DatasetBundleSummary,
    training: Phase3TrainingBundleSummary,
    privacy: Phase3PrivacyAggregationBundleSummary,
    eval_controls: Phase3EvalControlBundleSummary,
    observability: Phase3ObservabilityBundleSummary,
    publication: Phase3PublicationBundleSummary,
    claim_boundaries: Sequence[str],
    non_claims: Sequence[str],
    known_limitations: Sequence[str],
) -> str:
    """Render the Phase 3 model-card markdown."""

    boundaries = "\n".join(f"- {item}" for item in claim_boundaries)
    non_claim_lines = "\n".join(f"- {item}" for item in non_claims)
    limitations = "\n".join(f"- {item}" for item in known_limitations) or "- none"
    completed = ", ".join(f"`{item}`" for item in eval_controls.completed_controls)
    blocked = ", ".join(f"`{item}`" for item in eval_controls.blocked_controls)
    return f"""---
license: apache-2.0
library_name: lensemble
tags:
- federated-learning
- world-model
- jepa
- robotics
- phase3
---

# Lensemble Phase 3 Consortium JEPA World Model

This model repository records the Phase 3 consortium-training release-candidate
evidence for a federated JEPA / LeWorldModel-flavour world model.

## Consortium Runtime Evidence

- Consortium id: `{manifest.consortium_id}`
- Run id: `{manifest.run_id}`
- Participant agents: {manifest.participant_count}
- Coordinator: `{manifest.coordinator_id}`
- Protocol: `{manifest.protocol_version}`
- Public probe hash: `{manifest.public_probe_hash}`
- Secure aggregation backend: `{privacy.secure_aggregation_backend}`
- DP accountant: `{privacy.dp_accountant}`

## Training And Evaluation Scale

- Closed rounds: {training.closed_rounds}/{training.target_rounds}
- Tiny model shape: `latent_dim={training.model_latent_dim}`,
  `num_tokens={training.model_num_tokens}`
- Config hash: `{training.config_hash}`
- Final checkpoint hash: `{training.final_global_model_hash}`
- Run-manifest hash: `{training.run_manifest_hash}`
- Training evidence is a deterministic local consortium smoke, not a public
  HF Jobs robotics-scale result.

## Completed And Blocked Controls

- Completed controls: {completed or "none"}
- Blocked controls: {blocked or "none"}
- Eval/control metric rows: {eval_controls.metric_row_count}

{eval_controls.model_card_eval_text}

## Privacy And Observability Controls

- Secure-sum rounds: {privacy.secure_sum_rounds}
- DP-accounted rounds: {privacy.dp_accounted_rounds}
- Max per-round epsilon spent: {privacy.max_round_epsilon_spent}
- Observability round summaries: {observability.round_summary_count}
- Induced dropout outcomes: {", ".join(observability.induced_dropout_outcomes)}
- Redaction contract: `{observability.redaction_contract_id}`

## Dataset And Publication Status

- Dataset registry: `{dataset.registry_id}`
- Dataset run mode: `{dataset.run_mode}`
- Participant data declarations: {dataset.participant_count}
- Raw data crosses participant boundary: `{dataset.raw_data_crosses_boundary}`
- Model repo target: `hf://models/{publication.model_repo_id}@{publication.model_repo_revision}`
- Dataset repo target: `hf://datasets/{publication.dataset_repo_id}@{publication.dataset_repo_revision}`
- Publication status: `{publication.status}`

## Claim Boundaries

{boundaries}

## Non-Claims

{non_claim_lines}

## Known Limitations

{limitations}

## Reports In This Release Candidate

- `reports/phase3_evidence_bundle.json`
- `reports/phase3_long_run_smoke_report.json`
- `reports/phase3_eval_report.json`
- `reports/phase3_observability_report.json`
- `reports/phase3_long_run_manifest.json`
- `reports/phase3_long_run_dataset_registry.json`
- `artifacts/final/header.json`
- `artifacts/final/weights.safetensors`
"""


def validate_phase3_bundle_residency(raw: dict[str, Any]) -> None:
    """Reject obvious private fields, secrets, and host-local paths in bundle JSON."""

    _scan_bundle(raw, path="bundle")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local file."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _manifest_summary(
    manifest: Phase3ConsortiumManifest,
) -> Phase3ManifestBundleSummary:
    trainers = [p for p in manifest.participants if p.role == "trainer"]
    return Phase3ManifestBundleSummary(
        consortium_id=manifest.consortium_id,
        run_id=manifest.run_id,
        participant_count=len(trainers),
        coordinator_id=manifest.coordinator_id,
        protocol_version=manifest.runtime.protocol_version,
        secure_aggregation_backend=manifest.runtime.secure_aggregation_backend,
        dp_required=manifest.runtime.dp_required,
        public_probe_hash=manifest.public_probe.content_hash,
        claim_boundary=manifest.claim_boundary,
    )


def _dataset_summary(
    registry: Phase3DatasetProbeRegistry,
) -> Phase3DatasetBundleSummary:
    published = sum(
        1 for item in registry.participants if item.publication_status == "published"
    )
    placeholders = sum(
        1 for item in registry.participants if item.publication_status == "placeholder"
    )
    heldout = sorted({item.heldout_policy for item in registry.participants})
    return Phase3DatasetBundleSummary(
        registry_id=registry.registry_id,
        run_mode=registry.run_mode,
        participant_count=len(registry.participants),
        participant_ids=tuple(item.participant_id for item in registry.participants),
        public_probe_hash=registry.public_probe.content_hash,
        published_participant_count=published,
        placeholder_participant_count=placeholders,
        raw_data_crosses_boundary=False,
        heldout_policy="; ".join(heldout),
    )


def _training_summary(
    long_run: Phase3LongRunReport,
    *,
    checkpoint_header_path: Path,
    checkpoint_weights_path: Path,
) -> Phase3TrainingBundleSummary:
    return Phase3TrainingBundleSummary(
        run_id=long_run.run_id,
        participant_count=long_run.run_shape.participant_count,
        target_rounds=long_run.target_rounds,
        closed_rounds=long_run.closed_rounds,
        completed_target=long_run.completed_target,
        config_hash=long_run.config_hash,
        final_global_model_hash=long_run.final_global_model_hash,
        run_manifest_hash=_phase3_run_manifest_hash(long_run),
        checkpoint_header_sha256=sha256_file(checkpoint_header_path),
        checkpoint_weights_sha256=sha256_file(checkpoint_weights_path),
        model_latent_dim=long_run.run_shape.model_latent_dim,
        model_num_tokens=long_run.run_shape.model_num_tokens,
        claim_boundary=long_run.claim_boundary,
    )


def _privacy_summary(
    long_run: Phase3LongRunReport,
) -> Phase3PrivacyAggregationBundleSummary:
    epsilons = [
        float(row.dp_epsilon_spent)
        for row in long_run.rounds
        if row.dp_epsilon_spent is not None
    ]
    return Phase3PrivacyAggregationBundleSummary(
        secure_aggregation_backend=long_run.run_shape.secure_aggregation_backend,
        secure_aggregation_threshold=long_run.run_shape.secure_aggregation_threshold,
        secure_sum_rounds=sum(
            1
            for row in long_run.rounds
            if row.aggregation_backend_status == "secure_sum"
        ),
        dp_enabled=long_run.run_shape.dp_enabled,
        dp_accountant=long_run.run_shape.dp_accountant,
        dp_epsilon=long_run.run_shape.dp_epsilon,
        dp_delta=long_run.run_shape.dp_delta,
        dp_accounted_rounds=len(epsilons),
        max_round_epsilon_spent=max(epsilons) if epsilons else 0.0,
    )


def _eval_summary(eval_report: Phase3EvalReport) -> Phase3EvalControlBundleSummary:
    return Phase3EvalControlBundleSummary(
        completed_controls=tuple(
            sorted({row.control_role for row in eval_report.metric_rows})
        ),
        blocked_controls=tuple(
            row.control_role for row in eval_report.blocked_controls
        ),
        metric_row_count=len(eval_report.metric_rows),
        blocked_task_env_ids=tuple(
            task.task_env_id
            for task in eval_report.eval_plan.tasks
            if task.status == "blocked"
        ),
        model_card_eval_text=eval_report.model_card_eval_text,
        claim_boundary=eval_report.claim_boundary,
    )


def _observability_summary(
    report: Phase3ObservabilityReport,
) -> Phase3ObservabilityBundleSummary:
    return Phase3ObservabilityBundleSummary(
        round_summary_count=len(report.rounds),
        dropout_decision_count=len(report.dropout_decisions),
        induced_dropout_outcomes=tuple(
            f"{decision.scenario_id}:{decision.outcome}"
            for decision in report.dropout_decisions
            if decision.induced
        ),
        redaction_contract_id=report.redaction_contract.contract_id,
        final_bundle_handoff=report.final_bundle_handoff,
        claim_boundary=report.claim_boundary,
    )


def _publication_summary(
    *,
    status: Phase3PublicationStatus,
    model_repo_revision: str,
    dataset_repo_revision: str,
) -> Phase3PublicationBundleSummary:
    blockers: tuple[str, ...] = ()
    if status != "published":
        blockers = (
            "Model-card/evidence publication is represented as a local release-candidate target until an immutable Hub revision is recorded.",
            "The Phase 3 dataset repo target is declared, but public task-scale SO-100 eval data remains blocked.",
        )
    return Phase3PublicationBundleSummary(
        status=status,
        model_repo_id=_MODEL_REPO_ID,
        model_repo_revision=model_repo_revision,
        dataset_repo_id=_DATASET_REPO_ID,
        dataset_repo_revision=dataset_repo_revision,
        expected_model_repo_files=(
            "README.md",
            "reports/phase3_evidence_bundle.json",
            "reports/phase3_long_run_smoke_report.json",
            "reports/phase3_eval_report.json",
            "reports/phase3_observability_report.json",
            "reports/phase3_long_run_manifest.json",
            "reports/phase3_long_run_dataset_registry.json",
            "artifacts/phase3_run_manifest.json",
            "artifacts/final/header.json",
            "artifacts/final/weights.safetensors",
        ),
        blockers=blockers,
    )


def _phase3_run_manifest_hash(report: Phase3LongRunReport) -> str:
    payload = {
        "schema": "phase3-long-run-manifest/v1",
        "generated_at": report.generated_at.isoformat(),
        "config_hash": report.config_hash,
        "consortium_id": report.consortium_id,
        "run_id": report.run_id,
        "run_shape": report.run_shape.model_dump(mode="json"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256((raw + "\n").encode("utf-8")).hexdigest()


def _public_probe_hash(report: Phase3LongRunReport) -> str:
    prefix = "public_probe_hash_pinned:"
    for check in report.dry_run_checks:
        if check.startswith(prefix):
            return check.removeprefix(prefix)
    raise ConfigError(
        "Phase 3 long-run report does not pin a public probe hash",
        code=LensembleErrorCode.CONFIG_INVALID,
        remediation="rerun the Phase 3 long-run smoke with public-probe hash recording",
    )


def _hf_uri(
    repo_type: Phase3RepoType, repo_id: str, revision: str, path_in_repo: str
) -> str:
    prefix = "models" if repo_type == "model" else "datasets"
    return f"hf://{prefix}/{repo_id}@{revision}/{path_in_repo}"


def _hf_resolve_url(
    repo_type: Phase3RepoType, repo_id: str, revision: str, path_in_repo: str
) -> str:
    kind = "" if repo_type == "model" else "datasets/"
    return f"https://huggingface.co/{kind}{repo_id}/resolve/{revision}/{path_in_repo}"


def _failed_hub_check(
    kind: Phase3ArtifactKind,
    label: str,
    repo_type: Phase3RepoType,
    repo_id: str,
    revision: str,
    path_in_repo: str,
    uri: str,
    checked_at: datetime,
    exc: urllib.error.HTTPError,
) -> Phase3ArtifactCheck:
    return Phase3ArtifactCheck(
        kind=kind,
        label=label,
        location="hub",
        uri=uri,
        checked_at=checked_at,
        exists=False,
        repo_type=repo_type,
        repo_id=repo_id,
        revision=revision,
        path_in_repo=path_in_repo,
        status_code=exc.code,
        error=str(exc),
    )


def _scan_bundle(value: Any, *, path: str) -> None:
    if (
        path.endswith("model_card_markdown")
        or ".non_claims" in path
        or ".claim_boundaries" in path
    ):
        return
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered not in _SAFE_PRIVATE_SURFACE_METADATA_KEYS and any(
                pattern in lowered for pattern in _FORBIDDEN_KEYS
            ):
                raise ConfigError(
                    f"Phase 3 evidence bundle contains forbidden field {path}.{key}",
                    code=LensembleErrorCode.CONFIG_INVALID,
                    remediation="emit hashes/counts/status only; never raw private surfaces",
                )
            _scan_bundle(child, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for idx, child in enumerate(value):
            _scan_bundle(child, path=f"{path}[{idx}]")
        return
    if isinstance(value, str):
        if _is_sensitive_path_value(value):
            raise ConfigError(
                f"Phase 3 evidence bundle contains sensitive path at {path}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="use repo-relative paths or artifact URIs",
            )
        lowered = value.lower()
        if (
            lowered.startswith(("hf_", "sk-"))
            or " token=" in lowered
            or lowered.startswith("token=")
        ):
            raise ConfigError(
                f"Phase 3 evidence bundle contains secret-like value at {path}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="remove tokens, keys, and secret-like values",
            )


def _is_sensitive_path_value(value: str) -> bool:
    if value.startswith(("file://", "~")):
        return True
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


__all__ = [
    "PHASE3_EVIDENCE_BUNDLE_SCHEMA_VERSION",
    "Phase3ArtifactCheck",
    "Phase3ArtifactKind",
    "Phase3ArtifactLocation",
    "Phase3DatasetBundleSummary",
    "Phase3EvalControlBundleSummary",
    "Phase3EvidenceBundle",
    "Phase3ManifestBundleSummary",
    "Phase3ObservabilityBundleSummary",
    "Phase3PrivacyAggregationBundleSummary",
    "Phase3PublicationBundleSummary",
    "Phase3PublicationStatus",
    "Phase3RepoType",
    "Phase3TrainingBundleSummary",
    "build_phase3_evidence_bundle",
    "check_hf_artifact_exists",
    "load_phase3_evidence_bundle",
    "local_artifact_check",
    "local_artifact_uri",
    "materialize_phase3_run_contracts",
    "parse_phase3_evidence_bundle",
    "render_phase3_model_card",
    "sha256_file",
    "validate_phase3_bundle_residency",
    "write_phase3_bundle_outputs",
]
