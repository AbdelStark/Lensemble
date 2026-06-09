"""RFC-0017 dynamic-env observability, benchmark, and evidence bundle contracts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lensemble.errors import (
    ConfigError,
    EvaluationError,
    LensembleErrorCode,
    SchemaVersionMismatch,
)
from lensemble.eval.dynamic_downstream import (
    DynamicEnvControlReport,
    DynamicEnvDownstreamEvalReport,
)
from lensemble.federation.phase3_bundle import (
    Phase3ArtifactCheck,
    Phase3ArtifactKind,
    local_artifact_check,
    sha256_file,
)
from lensemble.federation.phase3_orchestration import (
    Phase3LongRunReport,
    load_phase3_long_run_report,
)

DYNAMIC_ENV_OBSERVABILITY_REPORT_SCHEMA_VERSION = 1
DYNAMIC_ENV_BENCHMARK_REPORT_SCHEMA_VERSION = 1
DYNAMIC_ENV_EVIDENCE_BUNDLE_SCHEMA_VERSION = 1

_GENERATED_AT = datetime(2026, 6, 9, 0, 0, 0, tzinfo=timezone.utc)
_REQUIRED_ARTIFACT_KINDS: set[Phase3ArtifactKind] = {
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
_REQUIRED_CONTROL_LABELS = ("federated", "naive-fedavg", "local-only", "random-encoder")
_FORBIDDEN_KEY_PATTERNS = (
    "raw_data",
    "raw_observation",
    "raw_obs",
    "raw_action",
    "action_tensor",
    "private_weight",
    "private_action_head",
    "secret",
    "access_token",
    "api_token",
)
_SAFE_PRIVATE_SURFACE_METADATA_KEYS = frozenset({"raw_data_in_report"})
_SECRET_VALUE_RE = re.compile(
    r"(sk-[A-Za-z0-9]{8,}|hf_[A-Za-z0-9]{8,}|AKIA[0-9A-Z]{16}|"
    r"BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY|"
    r"(?:password|secret|token)\s*[=:])",
    re.IGNORECASE,
)
_SENSITIVE_PATH_RE = re.compile(
    r"(^~(?:/|$)|^/(?:Users|home|var/folders|private/var|tmp)(?:/|$)|"
    r"^file://|(?:^|/)\.(?:ssh|aws|config)(?:/|$))"
)


class DynamicEnvSourceArtifactRef(BaseModel):
    """Residency-safe reference to one dynamic-env evidence input."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_name: str = Field(min_length=1)
    schema_version: int = Field(ge=1)


class DynamicEnvRoundObservability(BaseModel):
    """One round's privacy/aggregation observability for the dynamic-env run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    round_index: int = Field(ge=0)
    state: Literal["closed"]
    contributing_count: int = Field(ge=1)
    aggregation_backend_status: str = Field(min_length=1)
    dp_epsilon_spent: float | None = Field(default=None, ge=0.0)
    estimated_update_bytes: int = Field(ge=0)
    global_model_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class DynamicEnvObservabilityReport(BaseModel):
    """Dynamic-env observability and privacy-aggregation report (#286)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    generated_at: datetime
    task_env_id: Literal["kinematic://swipe-dot"]
    run_id: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_artifacts: tuple[DynamicEnvSourceArtifactRef, ...] = Field(min_length=1)
    rounds: tuple[DynamicEnvRoundObservability, ...] = Field(min_length=1)
    secure_sum_rounds: int = Field(ge=0)
    dp_enabled: bool
    dp_accountant: str = Field(min_length=1)
    dp_epsilon: float = Field(gt=0.0)
    dp_delta: float = Field(gt=0.0, lt=1.0)
    dp_accounted_rounds: int = Field(ge=0)
    max_round_epsilon_spent: float = Field(ge=0.0)
    dropout_decisions: tuple[str, ...]
    artifact_kinds_satisfied: tuple[
        Literal["observability-report", "privacy-aggregation-report"], ...
    ] = (
        "observability-report",
        "privacy-aggregation-report",
    )
    redaction_contract_id: str = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)
    raw_data_in_report: Literal[False] = False

    @model_validator(mode="after")
    def _cross_check(self) -> "DynamicEnvObservabilityReport":
        if self.schema_version != DYNAMIC_ENV_OBSERVABILITY_REPORT_SCHEMA_VERSION:
            raise SchemaVersionMismatch(
                f"dynamic-env observability schema_version {self.schema_version!r} exceeds reader max "
                f"{DYNAMIC_ENV_OBSERVABILITY_REPORT_SCHEMA_VERSION}",
                code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
                remediation="read with a build supporting this dynamic-env observability schema",
            )
        required = ("synthetic control env", "DP", "secure", "residency", "paper-scale")
        missing = [phrase for phrase in required if phrase not in self.claim_boundary]
        if missing:
            raise ConfigError(
                f"dynamic-env observability claim_boundary missing required phrases: {missing}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="keep the DP/secure-aggregation observability boundary explicit",
            )
        source_hashes = {artifact.sha256 for artifact in self.source_artifacts}
        if self.run_manifest_sha256 not in source_hashes:
            raise ConfigError(
                "dynamic-env observability report is not bound to the run manifest hash",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="include the run manifest as a source artifact",
            )
        if self.secure_sum_rounds > len(self.rounds):
            raise ConfigError(
                "dynamic-env secure_sum_rounds exceeds emitted round count",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="regenerate observability from the matching long-run report",
            )
        if self.dp_accounted_rounds > len(self.rounds):
            raise ConfigError(
                "dynamic-env dp_accounted_rounds exceeds emitted round count",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="regenerate observability from the matching long-run report",
            )
        validate_dynamic_env_residency(self.model_dump(mode="json"))
        return self


class DynamicEnvBenchmarkControl(BaseModel):
    """One dynamic-env control row summarized for the benchmark report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(min_length=1)
    repo_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    round_index: int = Field(ge=0)
    checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_probe_r2: float
    success_rate: float = Field(ge=0.0, le=1.0)
    success_rate_role: Literal["reported_non_binding"] = "reported_non_binding"
    effective_rank: float | None = Field(default=None, ge=0.0)
    val_pred: float | None = None
    frame_drift_deg: float | None = Field(default=None, ge=0.0)
    metric_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def _ranges(self) -> "DynamicEnvBenchmarkControl":
        for value in (self.state_probe_r2, self.success_rate):
            if not math.isfinite(value):
                raise EvaluationError(
                    "dynamic-env benchmark controls must use finite metrics",
                    code=LensembleErrorCode.EVALUATION_FAILED,
                    remediation="remove NaN/Inf before publishing dynamic-env evidence",
                )
        if self.state_probe_r2 > 1.0:
            raise EvaluationError(
                f"state_probe_r2 must be <= 1, got {self.state_probe_r2}",
                code=LensembleErrorCode.EVALUATION_FAILED,
                remediation="R2 is upper-bounded by 1; inspect the probe computation",
            )
        if (
            "gameable" not in self.metric_boundary
            or "supporting" not in self.metric_boundary
        ):
            raise ConfigError(
                "dynamic-env benchmark control must label non-R2 metrics as supporting/gameable",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="state that closed-loop/latent proxy metrics are non-binding supporting signals",
            )
        return self


class DynamicEnvBenchmarkReport(BaseModel):
    """Benchmark report centered on the binding ground-truth state probe (#287)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    generated_at: datetime
    task_env_id: Literal["kinematic://swipe-dot"]
    held_out_data_ref: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_arch: Literal["scratch"]
    binding_metric: Literal["state_probe_r2"] = "state_probe_r2"
    binding_control_label: str = Field(min_length=1)
    r2_gate: float = Field(default=0.5, ge=-1.0, le=1.0)
    absolute_margin: float = Field(default=0.05, gt=0.0)
    controls: tuple[DynamicEnvBenchmarkControl, ...] = Field(min_length=1)
    closed_rounds: int = Field(ge=0)
    target_rounds: int = Field(ge=1)
    dp_enabled: bool
    dp_epsilon: float = Field(gt=0.0)
    claim_boundary: str = Field(min_length=1)
    raw_data_in_report: Literal[False] = False

    @model_validator(mode="after")
    def _cross_check(self) -> "DynamicEnvBenchmarkReport":
        if self.schema_version != DYNAMIC_ENV_BENCHMARK_REPORT_SCHEMA_VERSION:
            raise SchemaVersionMismatch(
                f"dynamic-env benchmark schema_version {self.schema_version!r} exceeds reader max "
                f"{DYNAMIC_ENV_BENCHMARK_REPORT_SCHEMA_VERSION}",
                code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
                remediation="read with a build supporting this dynamic-env benchmark schema",
            )
        required = (
            "synthetic control env",
            "state_probe_r2",
            "binding",
            "gameable",
            "scale-invariant",
            "success_rate is reported non-binding",
            "paper-scale",
            "scratch",
            "not vjepa2-vit-l",
        )
        missing = [phrase for phrase in required if phrase not in self.claim_boundary]
        if missing:
            raise ConfigError(
                f"dynamic-env benchmark claim_boundary missing required phrases: {missing}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="preserve RFC-0017's claim boundary and scratch-architecture note",
            )
        by_label = {control.label: control for control in self.controls}
        if len(by_label) != len(self.controls):
            raise ConfigError(
                "dynamic-env benchmark control labels must be unique",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="deduplicate control labels before publishing",
            )
        missing_controls = set(_REQUIRED_CONTROL_LABELS) - set(by_label)
        if missing_controls:
            raise ConfigError(
                "dynamic-env benchmark is missing required controls: "
                + ", ".join(sorted(missing_controls)),
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="include federated, naive-fedavg, local-only, and random-encoder rows",
            )
        if self.binding_control_label not in by_label:
            raise ConfigError(
                "dynamic-env benchmark binding control label is missing",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="set binding_control_label to the federated row",
            )
        headline = by_label[self.binding_control_label]
        if headline.state_probe_r2 < self.r2_gate:
            raise EvaluationError(
                f"dynamic-env binding state_probe_r2 {headline.state_probe_r2} is below gate {self.r2_gate}",
                code=LensembleErrorCode.EVALUATION_FAILED,
                remediation="do not publish a usefulness claim until the held-out ground-truth R2 gate clears",
            )
        for label, control in by_label.items():
            if label == self.binding_control_label:
                continue
            if headline.state_probe_r2 < control.state_probe_r2 + self.absolute_margin:
                raise EvaluationError(
                    f"dynamic-env binding control does not beat {label!r} by margin {self.absolute_margin}",
                    code=LensembleErrorCode.EVALUATION_FAILED,
                    remediation="publish the negative result or rerun before claiming the dynamic-env gate",
                )
        validate_dynamic_env_residency(self.model_dump(mode="json"))
        return self


class DynamicEnvEvidenceBundle(BaseModel):
    """Integrity-chained dynamic-env evidence bundle and model-card contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    generated_at: datetime
    bundle: Literal["dynamic-env-swipe-dot-evidence"]
    artifact_checks: tuple[Phase3ArtifactCheck, ...] = Field(min_length=1)
    benchmark: DynamicEnvBenchmarkReport
    observability: DynamicEnvObservabilityReport
    run_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_header_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observability_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    publication_status: Literal["local_smoke", "published", "blocked"]
    model_repo_revision: str = Field(min_length=1)
    claim_boundaries: tuple[str, ...] = Field(min_length=1)
    non_claims: tuple[str, ...] = Field(min_length=1)
    model_card_markdown: str = Field(min_length=1)
    raw_data_in_report: Literal[False] = False

    @model_validator(mode="after")
    def _cross_check(self) -> "DynamicEnvEvidenceBundle":
        if self.schema_version != DYNAMIC_ENV_EVIDENCE_BUNDLE_SCHEMA_VERSION:
            raise SchemaVersionMismatch(
                f"dynamic-env evidence bundle schema_version {self.schema_version!r} exceeds reader max "
                f"{DYNAMIC_ENV_EVIDENCE_BUNDLE_SCHEMA_VERSION}",
                code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
                remediation="read with a build supporting this dynamic-env bundle schema",
            )
        missing = [check for check in self.artifact_checks if not check.exists]
        if missing:
            raise ConfigError(
                "dynamic-env evidence bundle has missing artifact checks: "
                + ", ".join(check.label for check in missing),
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="publish or regenerate every referenced dynamic-env artifact",
            )
        seen = {check.kind for check in self.artifact_checks}
        missing_kinds = _REQUIRED_ARTIFACT_KINDS - seen
        if missing_kinds:
            raise ConfigError(
                "dynamic-env evidence bundle is missing artifact kinds: "
                + ", ".join(sorted(missing_kinds)),
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="include training, eval, observability/privacy, manifest, registry, and checkpoint artifacts",
            )
        self._cross_check_artifact_hashes()
        card = self.model_card_markdown.lower()
        required_text = (
            "synthetic control env",
            "state_probe_r2",
            "binding",
            "gameable",
            "scale-invariant",
            "closed-loop success_rate is reported non-binding",
            "scratch",
            "not vjepa2-vit-l",
            "does not include a provenance ledger",
            "does not cryptographically prove honest participant computation",
            "does not claim paper-scale leworldmodel performance",
        )
        missing_text = [text for text in required_text if text not in card]
        if missing_text:
            raise ConfigError(
                "dynamic-env model card is missing required honest-boundary text: "
                + ", ".join(missing_text),
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="regenerate the model card from the dynamic-env bundle renderer",
            )
        validate_dynamic_env_residency(self.model_dump(mode="json"))
        return self

    def _cross_check_artifact_hashes(self) -> None:
        expected: tuple[tuple[Phase3ArtifactKind, str], ...] = (
            ("run-manifest", self.run_manifest_sha256),
            ("checkpoint-header", self.checkpoint_header_sha256),
            ("checkpoint-weights", self.checkpoint_weights_sha256),
            ("eval-control-report", self.benchmark_report_sha256),
            ("observability-report", self.observability_report_sha256),
            ("privacy-aggregation-report", self.observability_report_sha256),
        )
        for kind, sha in expected:
            checks = [
                check
                for check in self.artifact_checks
                if check.kind == kind and check.sha256 is not None
            ]
            if not checks:
                raise ConfigError(
                    f"dynamic-env evidence bundle artifact check {kind!r} is missing sha256",
                    code=LensembleErrorCode.CONFIG_INVALID,
                    remediation="regenerate the bundle with local artifact hashes",
                )
            mismatched = [check for check in checks if check.sha256 != sha]
            if mismatched:
                raise ConfigError(
                    f"dynamic-env evidence bundle artifact hash mismatch for {kind!r}",
                    code=LensembleErrorCode.CONFIG_INVALID,
                    remediation="regenerate the bundle from the same run artifacts",
                )


def build_dynamic_env_observability_report(
    *,
    long_run_report_path: Path,
    run_manifest_path: Path,
    generated_at: datetime = _GENERATED_AT,
) -> DynamicEnvObservabilityReport:
    """Build the dynamic-env observability/privacy report from a long-run report."""

    long_run_path = Path(long_run_report_path)
    manifest_path = Path(run_manifest_path)
    long_run = load_phase3_long_run_report(long_run_path)
    long_run_sha = sha256_file(long_run_path)
    manifest_sha = sha256_file(manifest_path)
    epsilons = [
        float(row.dp_epsilon_spent)
        for row in long_run.rounds
        if row.dp_epsilon_spent is not None
    ]
    per_update_numel = int(long_run.run_shape.model_latent_dim) * int(
        long_run.run_shape.model_num_tokens
    )
    rounds = tuple(
        DynamicEnvRoundObservability(
            round_index=row.round_index,
            state=row.state,
            contributing_count=row.contributing_count,
            aggregation_backend_status=row.aggregation_backend_status,
            dp_epsilon_spent=row.dp_epsilon_spent,
            estimated_update_bytes=int(row.contributing_count) * per_update_numel * 4,
            global_model_hash=row.global_model_hash,
        )
        for row in long_run.rounds
    )
    report = DynamicEnvObservabilityReport(
        schema_version=DYNAMIC_ENV_OBSERVABILITY_REPORT_SCHEMA_VERSION,
        generated_at=generated_at,
        task_env_id="kinematic://swipe-dot",
        run_id=long_run.run_id,
        config_hash=long_run.config_hash,
        checkpoint_hash=long_run.final_global_model_hash,
        run_manifest_sha256=manifest_sha,
        source_artifacts=(
            DynamicEnvSourceArtifactRef(
                label="dynamic_env_long_run_report",
                uri=_artifact_uri(long_run_path, "dynamic-env-training"),
                sha256=long_run_sha,
                schema_name="phase3_long_run_report",
                schema_version=long_run.schema_version,
            ),
            DynamicEnvSourceArtifactRef(
                label="dynamic_env_run_manifest",
                uri=_artifact_uri(manifest_path, "dynamic-env-run-manifest"),
                sha256=manifest_sha,
                schema_name="phase3-long-run-manifest",
                schema_version=1,
            ),
        ),
        rounds=rounds,
        secure_sum_rounds=sum(
            1 for row in rounds if row.aggregation_backend_status == "secure_sum"
        ),
        dp_enabled=long_run.run_shape.dp_enabled,
        dp_accountant=long_run.run_shape.dp_accountant,
        dp_epsilon=long_run.run_shape.dp_epsilon,
        dp_delta=long_run.run_shape.dp_delta,
        dp_accounted_rounds=len(epsilons),
        max_round_epsilon_spent=max(epsilons) if epsilons else 0.0,
        dropout_decisions=(
            "no induced dropout in this dynamic-env evidence row; runtime dropout semantics remain covered by the Phase 3 coordinator-service tests",
        ),
        redaction_contract_id="dynamic-env-observability-redaction-v1",
        claim_boundary=(
            "synthetic control env observability report: DP accounting, secure aggregation status, "
            "communication byte estimates, and residency-clean artifact hashes for the dynamic-env run. "
            "This does not claim paper-scale performance."
        ),
    )
    return parse_dynamic_env_observability_report(report.model_dump(mode="json"))


def build_dynamic_env_benchmark_report(
    *,
    downstream_report: DynamicEnvDownstreamEvalReport,
    long_run: Phase3LongRunReport,
    model_arch: Literal["scratch"] = "scratch",
    binding_control_label: str = "federated",
    r2_gate: float = 0.5,
    absolute_margin: float = 0.05,
    generated_at: datetime = _GENERATED_AT,
) -> DynamicEnvBenchmarkReport:
    """Build the dynamic-env benchmark report from downstream rows and run metadata."""

    final_round = long_run.rounds[-1] if long_run.rounds else None
    controls = tuple(
        _benchmark_control(
            control,
            final_val_pred=final_round.val_pred if final_round is not None else None,
            final_frame_drift=final_round.frame_drift_deg
            if final_round is not None
            else None,
        )
        for control in downstream_report.controls
    )
    report = DynamicEnvBenchmarkReport(
        schema_version=DYNAMIC_ENV_BENCHMARK_REPORT_SCHEMA_VERSION,
        generated_at=generated_at,
        task_env_id=downstream_report.task_env_id,
        held_out_data_ref=downstream_report.held_out_data_ref,
        run_id=long_run.run_id,
        config_hash=long_run.config_hash,
        model_arch=model_arch,
        binding_control_label=binding_control_label,
        r2_gate=r2_gate,
        absolute_margin=absolute_margin,
        controls=controls,
        closed_rounds=long_run.closed_rounds,
        target_rounds=long_run.target_rounds,
        dp_enabled=long_run.run_shape.dp_enabled,
        dp_epsilon=long_run.run_shape.dp_epsilon,
        claim_boundary=(
            "synthetic control env benchmark; state_probe_r2 is the binding ground-truth metric; "
            "closed-loop success_rate is reported non-binding; latent-MPC skill metrics are gameable "
            "supporting signals; effective_rank is scale-invariant and non-binding; scratch architecture, "
            "not vjepa2-vit-l; no paper-scale robotics claim"
        ),
    )
    return parse_dynamic_env_benchmark_report(report.model_dump(mode="json"))


def build_dynamic_env_evidence_bundle(
    *,
    benchmark: DynamicEnvBenchmarkReport,
    observability: DynamicEnvObservabilityReport,
    artifact_checks: Sequence[Phase3ArtifactCheck],
    run_manifest_path: Path,
    checkpoint_header_path: Path,
    checkpoint_weights_path: Path,
    benchmark_report_path: Path,
    observability_report_path: Path,
    publication_status: Literal["local_smoke", "published", "blocked"] = "local_smoke",
    model_repo_revision: str = "local-smoke",
    generated_at: datetime = _GENERATED_AT,
) -> DynamicEnvEvidenceBundle:
    """Build the integrity-chained dynamic-env evidence bundle."""

    run_manifest_sha = sha256_file(run_manifest_path)
    header_sha = sha256_file(checkpoint_header_path)
    weights_sha = sha256_file(checkpoint_weights_path)
    benchmark_sha = sha256_file(benchmark_report_path)
    observability_sha = sha256_file(observability_report_path)
    claim_boundaries = (
        "Usefulness is ground-truth-measured by the single binding state_probe_r2 gate on kinematic://swipe-dot.",
        "Closed-loop success_rate is reported non-binding because the planner objective can be gameable.",
        "effective_rank and latent proxy scores are supporting only; effective_rank is scale-invariant and blind to magnitude collapse.",
        "This is a synthetic control env, not SO-100 and not paper-scale robotics evidence.",
        "The published dynamic-env architecture is scratch, not vjepa2-vit-l.",
    )
    non_claims = (
        "Dynamic-env evidence does not include a provenance ledger.",
        "Dynamic-env evidence does not cryptographically prove honest participant computation.",
        "Dynamic-env evidence does not claim paper-scale LeWorldModel performance.",
        "Dynamic-env evidence does not claim browser training; browser work is inference and env-sim only.",
    )
    model_card = render_dynamic_env_model_card(
        benchmark=benchmark,
        observability=observability,
        publication_status=publication_status,
        model_repo_revision=model_repo_revision,
        claim_boundaries=claim_boundaries,
        non_claims=non_claims,
    )
    bundle = DynamicEnvEvidenceBundle(
        schema_version=DYNAMIC_ENV_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        generated_at=generated_at,
        bundle="dynamic-env-swipe-dot-evidence",
        artifact_checks=tuple(artifact_checks),
        benchmark=benchmark,
        observability=observability,
        run_manifest_sha256=run_manifest_sha,
        checkpoint_header_sha256=header_sha,
        checkpoint_weights_sha256=weights_sha,
        benchmark_report_sha256=benchmark_sha,
        observability_report_sha256=observability_sha,
        publication_status=publication_status,
        model_repo_revision=model_repo_revision,
        claim_boundaries=claim_boundaries,
        non_claims=non_claims,
        model_card_markdown=model_card,
    )
    return parse_dynamic_env_evidence_bundle(bundle.model_dump(mode="json"))


def render_dynamic_env_model_card(
    *,
    benchmark: DynamicEnvBenchmarkReport,
    observability: DynamicEnvObservabilityReport,
    publication_status: str,
    model_repo_revision: str,
    claim_boundaries: Sequence[str],
    non_claims: Sequence[str],
) -> str:
    """Render the dynamic-env model card markdown."""

    controls = "\n".join(
        "- `{label}` round {round_index}: state_probe_r2={r2:.4f}, success_rate={success:.4f} ({role})".format(
            label=control.label,
            round_index=control.round_index,
            r2=control.state_probe_r2,
            success=control.success_rate,
            role=control.success_rate_role.replace("_", "-"),
        )
        for control in benchmark.controls
    )
    boundaries = "\n".join(f"- {item}" for item in claim_boundaries)
    non_claim_lines = "\n".join(f"- {item}" for item in non_claims)
    return f"""---
license: apache-2.0
library_name: lensemble
tags:
- federated-learning
- world-model
- dynamic-env
- state-probe
- phase3
---

# Lensemble Dynamic-Env Swipe-Dot World Model

This model card records the RFC-0017 dynamic-env evidence for a synthetic
control env. The binding usefulness metric is `state_probe_r2`; closed-loop
success_rate is reported non-binding because the planner objective can be
gameable.

## Binding Result

- Task env: `{benchmark.task_env_id}`
- Held-out data ref: `{benchmark.held_out_data_ref}`
- Model architecture: `scratch`, not vjepa2-vit-l
- Run id: `{benchmark.run_id}`
- Closed rounds: {benchmark.closed_rounds}/{benchmark.target_rounds}
- DP enabled: `{benchmark.dp_enabled}`
- DP epsilon: {benchmark.dp_epsilon}
- Binding gate: `state_probe_r2 >= {benchmark.r2_gate}`
- Required margin over controls: `{benchmark.absolute_margin}`

{controls}

## Supporting Signals

`effective_rank` is scale-invariant and non-binding. Latent proxy metrics such
as skill-vs-identity or latent goal energy are supporting only and gameable.

## Observability And Privacy

- Secure-sum rounds: {observability.secure_sum_rounds}
- DP-accounted rounds: {observability.dp_accounted_rounds}
- Max per-round epsilon spent: {observability.max_round_epsilon_spent}
- Redaction contract: `{observability.redaction_contract_id}`
- Run-manifest hash: `{observability.run_manifest_sha256}`

## Publication

- Publication status: `{publication_status}`
- Model repo revision: `{model_repo_revision}`

## Claim Boundaries

{boundaries}

## Non-Claims

{non_claim_lines}

The dynamic-env evidence does not include a provenance ledger, does not
cryptographically prove honest participant computation, and does not claim
paper-scale LeWorldModel performance.
"""


def parse_dynamic_env_observability_report(
    raw: dict[str, Any],
) -> DynamicEnvObservabilityReport:
    """Parse a dynamic-env observability report, gating future schemas first."""

    version = raw.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version > DYNAMIC_ENV_OBSERVABILITY_REPORT_SCHEMA_VERSION
    ):
        raise SchemaVersionMismatch(
            f"dynamic-env observability schema_version {version!r} exceeds reader max "
            f"{DYNAMIC_ENV_OBSERVABILITY_REPORT_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation="read with a build supporting this dynamic-env observability schema",
        )
    validate_dynamic_env_residency(raw)
    return DynamicEnvObservabilityReport.model_validate(raw)


def parse_dynamic_env_benchmark_report(
    raw: dict[str, Any],
) -> DynamicEnvBenchmarkReport:
    """Parse a dynamic-env benchmark report, gating future schemas first."""

    version = raw.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version > DYNAMIC_ENV_BENCHMARK_REPORT_SCHEMA_VERSION
    ):
        raise SchemaVersionMismatch(
            f"dynamic-env benchmark schema_version {version!r} exceeds reader max "
            f"{DYNAMIC_ENV_BENCHMARK_REPORT_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation="read with a build supporting this dynamic-env benchmark schema",
        )
    validate_dynamic_env_residency(raw)
    return DynamicEnvBenchmarkReport.model_validate(raw)


def parse_dynamic_env_evidence_bundle(raw: dict[str, Any]) -> DynamicEnvEvidenceBundle:
    """Parse a dynamic-env evidence bundle, gating future schemas first."""

    version = raw.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version > DYNAMIC_ENV_EVIDENCE_BUNDLE_SCHEMA_VERSION
    ):
        raise SchemaVersionMismatch(
            f"dynamic-env evidence bundle schema_version {version!r} exceeds reader max "
            f"{DYNAMIC_ENV_EVIDENCE_BUNDLE_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation="read with a build supporting this dynamic-env bundle schema",
        )
    validate_dynamic_env_residency(raw)
    return DynamicEnvEvidenceBundle.model_validate(raw)


def load_dynamic_env_observability_report(path: Path) -> DynamicEnvObservabilityReport:
    """Load and validate a dynamic-env observability report."""

    return parse_dynamic_env_observability_report(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def load_dynamic_env_benchmark_report(path: Path) -> DynamicEnvBenchmarkReport:
    """Load and validate a dynamic-env benchmark report."""

    return parse_dynamic_env_benchmark_report(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def load_dynamic_env_evidence_bundle(path: Path) -> DynamicEnvEvidenceBundle:
    """Load and validate a dynamic-env evidence bundle."""

    return parse_dynamic_env_evidence_bundle(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def to_dynamic_env_observability_report_json(
    report: DynamicEnvObservabilityReport,
) -> str:
    """Canonical JSON for a dynamic-env observability report."""

    return _canonical_json(report.model_dump(mode="json"))


def to_dynamic_env_benchmark_report_json(report: DynamicEnvBenchmarkReport) -> str:
    """Canonical JSON for a dynamic-env benchmark report."""

    return _canonical_json(report.model_dump(mode="json"))


def to_dynamic_env_evidence_bundle_json(bundle: DynamicEnvEvidenceBundle) -> str:
    """Canonical JSON for a dynamic-env evidence bundle."""

    return json.dumps(
        bundle.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=True
    )


def write_dynamic_env_observability_report(
    report: DynamicEnvObservabilityReport, path: Path
) -> Path:
    """Write a validated dynamic-env observability report."""

    parse_dynamic_env_observability_report(report.model_dump(mode="json"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        to_dynamic_env_observability_report_json(report) + "\n", encoding="utf-8"
    )
    return path


def write_dynamic_env_benchmark_report(
    report: DynamicEnvBenchmarkReport, path: Path
) -> Path:
    """Write a validated dynamic-env benchmark report."""

    parse_dynamic_env_benchmark_report(report.model_dump(mode="json"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        to_dynamic_env_benchmark_report_json(report) + "\n", encoding="utf-8"
    )
    return path


def write_dynamic_env_evidence_bundle_outputs(
    bundle: DynamicEnvEvidenceBundle,
    *,
    bundle_path: Path,
    model_card_path: Path,
) -> None:
    """Write a dynamic-env evidence bundle and byte-identical model card."""

    parse_dynamic_env_evidence_bundle(bundle.model_dump(mode="json"))
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    model_card_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(
        to_dynamic_env_evidence_bundle_json(bundle) + "\n", encoding="utf-8"
    )
    model_card_path.write_text(bundle.model_card_markdown, encoding="utf-8")


def dynamic_env_artifact_checks(
    *,
    manifest_path: Path,
    registry_path: Path,
    training_report_path: Path,
    observability_report_path: Path,
    benchmark_report_path: Path,
    run_manifest_path: Path,
    checkpoint_header_path: Path,
    checkpoint_weights_path: Path,
    checked_at: datetime = _GENERATED_AT,
) -> tuple[Phase3ArtifactCheck, ...]:
    """Build the required artifact-kind checks for a dynamic-env evidence bundle."""

    return (
        local_artifact_check(
            kind="consortium-manifest",
            label="dynamic-env consortium manifest",
            path=manifest_path,
            checked_at=checked_at,
        ),
        local_artifact_check(
            kind="dataset-probe-registry",
            label="dynamic-env dataset/probe registry",
            path=registry_path,
            checked_at=checked_at,
        ),
        local_artifact_check(
            kind="training-report",
            label="dynamic-env training report",
            path=training_report_path,
            checked_at=checked_at,
        ),
        local_artifact_check(
            kind="privacy-aggregation-report",
            label="dynamic-env privacy aggregation report",
            path=observability_report_path,
            checked_at=checked_at,
        ),
        local_artifact_check(
            kind="observability-report",
            label="dynamic-env observability report",
            path=observability_report_path,
            checked_at=checked_at,
        ),
        local_artifact_check(
            kind="eval-control-report",
            label="dynamic-env benchmark/control report",
            path=benchmark_report_path,
            checked_at=checked_at,
        ),
        local_artifact_check(
            kind="run-manifest",
            label="dynamic-env run manifest",
            path=run_manifest_path,
            checked_at=checked_at,
        ),
        local_artifact_check(
            kind="checkpoint-header",
            label="dynamic-env checkpoint header",
            path=checkpoint_header_path,
            checked_at=checked_at,
        ),
        local_artifact_check(
            kind="checkpoint-weights",
            label="dynamic-env checkpoint weights",
            path=checkpoint_weights_path,
            checked_at=checked_at,
        ),
    )


def validate_dynamic_env_residency(raw: dict[str, Any]) -> None:
    """Reject obvious private fields, secrets, and host-local paths."""

    _scan_for_sensitive(raw, path="dynamic_env")


def _benchmark_control(
    control: DynamicEnvControlReport,
    *,
    final_val_pred: float | None,
    final_frame_drift: float | None,
) -> DynamicEnvBenchmarkControl:
    return DynamicEnvBenchmarkControl(
        label=control.label,
        repo_id=control.checkpoint.repo_id,
        revision=control.checkpoint.revision,
        round_index=control.checkpoint.round_index,
        checkpoint_hash=control.checkpoint.checkpoint_hash,
        state_probe_r2=control.state_probe_r2,
        success_rate=control.success_rate,
        effective_rank=control.effective_rank,
        val_pred=final_val_pred if control.label == "federated" else None,
        frame_drift_deg=final_frame_drift if control.label == "federated" else None,
        metric_boundary=control.metric_boundary,
    )


def _artifact_uri(path: Path, namespace: str) -> str:
    path = Path(path)
    return f"artifact://{namespace}/{path.name}"


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _scan_for_sensitive(value: Any, *, path: str) -> None:
    if (
        path.endswith("model_card_markdown")
        or ".claim_boundaries" in path
        or ".non_claims" in path
    ):
        return
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered not in _SAFE_PRIVATE_SURFACE_METADATA_KEYS and any(
                pattern in lowered for pattern in _FORBIDDEN_KEY_PATTERNS
            ):
                raise ConfigError(
                    f"dynamic-env evidence contains forbidden field {path}.{key}",
                    code=LensembleErrorCode.CONFIG_INVALID,
                    remediation="emit hashes/counts/status only; never raw private surfaces",
                )
            _scan_for_sensitive(child, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for idx, child in enumerate(value):
            _scan_for_sensitive(child, path=f"{path}[{idx}]")
        return
    if isinstance(value, str):
        if _SECRET_VALUE_RE.search(value):
            raise ConfigError(
                f"dynamic-env evidence contains secret-like value at {path}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="remove tokens, keys, passwords, or secret-like strings from reports",
            )
        if _SENSITIVE_PATH_RE.search(value):
            raise ConfigError(
                f"dynamic-env evidence contains sensitive host-local path at {path}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="publish artifact URIs or repo-relative evidence paths instead",
            )


__all__ = [
    "DYNAMIC_ENV_BENCHMARK_REPORT_SCHEMA_VERSION",
    "DYNAMIC_ENV_EVIDENCE_BUNDLE_SCHEMA_VERSION",
    "DYNAMIC_ENV_OBSERVABILITY_REPORT_SCHEMA_VERSION",
    "DynamicEnvBenchmarkControl",
    "DynamicEnvBenchmarkReport",
    "DynamicEnvEvidenceBundle",
    "DynamicEnvObservabilityReport",
    "DynamicEnvRoundObservability",
    "DynamicEnvSourceArtifactRef",
    "build_dynamic_env_benchmark_report",
    "build_dynamic_env_evidence_bundle",
    "build_dynamic_env_observability_report",
    "dynamic_env_artifact_checks",
    "load_dynamic_env_benchmark_report",
    "load_dynamic_env_evidence_bundle",
    "load_dynamic_env_observability_report",
    "parse_dynamic_env_benchmark_report",
    "parse_dynamic_env_evidence_bundle",
    "parse_dynamic_env_observability_report",
    "render_dynamic_env_model_card",
    "to_dynamic_env_benchmark_report_json",
    "to_dynamic_env_evidence_bundle_json",
    "to_dynamic_env_observability_report_json",
    "validate_dynamic_env_residency",
    "write_dynamic_env_benchmark_report",
    "write_dynamic_env_evidence_bundle_outputs",
    "write_dynamic_env_observability_report",
]
