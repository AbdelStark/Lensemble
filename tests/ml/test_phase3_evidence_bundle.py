"""Phase 3 final evidence bundle and model-card contract (#230)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from lensemble.config.consortium import load_consortium_manifest
from lensemble.data.phase3 import load_phase3_dataset_registry
from lensemble.errors import ConfigError, SchemaVersionMismatch
from lensemble.eval import build_phase3_eval_report, write_phase3_eval_report
from lensemble.federation import (
    PHASE3_EVIDENCE_BUNDLE_SCHEMA_VERSION,
    Phase3EvidenceBundle,
    build_phase3_observability_report,
    load_phase3_evidence_bundle,
    local_artifact_check,
    parse_phase3_evidence_bundle,
    run_phase3_long_run_smoke,
    validate_phase3_bundle_residency,
    write_phase3_long_run_report,
    write_phase3_observability_report,
)


def test_checked_in_phase3_evidence_bundle_and_model_card_are_consistent() -> None:
    bundle_path = Path("docs/evidence/phase3_evidence_bundle.json")
    model_card_path = Path("docs/evidence/phase3_model_card.md")

    bundle = parse_phase3_evidence_bundle(json.loads(bundle_path.read_text()))

    assert bundle.schema_version == PHASE3_EVIDENCE_BUNDLE_SCHEMA_VERSION
    assert bundle.raw_data_in_report is False
    assert all(check.exists for check in bundle.artifact_checks)
    assert bundle.publication.status == "historical"
    assert (
        bundle.publication.model_repo_revision
        == "828e210cba4870b2be4ab573a5f0dd4ee30bae29"
    )
    assert (
        bundle.publication.dataset_repo_revision
        == "15f71911432b300dfdf41c998e27492e8c986be4"
    )
    assert bundle.publication.blockers
    assert bundle.publication.dataset_bound_to_training is False
    assert bundle.manifest.consortium_id == "lensemble-phase3-long-run-smoke"
    assert bundle.manifest.run_id == "phase3-long-run-smoke-v1"
    assert bundle.manifest.public_probe_hash_contract == "legacy-unscoped"
    assert bundle.dataset_registry.public_probe_hash_contract == "legacy-unscoped"
    assert bundle.training.run_id == "phase3-long-run-smoke-v1"
    assert bundle.training.closed_rounds == 10
    assert bundle.training.completed_target is True
    assert (
        bundle.training.final_global_model_hash
        == "ed3081ee514af142a226443f113a37c24d7d5872bfb707f11abe10893a0ad50d"
    )
    assert bundle.privacy_aggregation.secure_sum_rounds == 0
    assert bundle.privacy_aggregation.post_commit_cross_check_rounds == 10
    assert bundle.privacy_aggregation.legacy_reported_secure_sum_rounds == 0
    assert bundle.privacy_aggregation.dp_accounted_rounds == 10
    assert bundle.privacy_aggregation.effective_dp_rounds == 0
    assert bundle.observability.dropout_decision_count == 1
    assert set(bundle.eval_controls.completed_controls) == {
        "anchored-federation",
        "naive-fedavg",
        "fork-a-frozen-encoder",
        "local-only",
    }
    assert bundle.eval_controls.blocked_controls == ()
    assert model_card_path.read_text() == bundle.model_card_markdown
    assert bundle.training.evidence_status == "historical_pre_correctness_fix"
    assert "## Historical Evidence Status" in bundle.model_card_markdown
    assert "do not validate the corrected runtime" in bundle.model_card_markdown
    assert "placeholder synthetic smoke data" in bundle.model_card_markdown
    assert "does not include a provenance ledger" in bundle.model_card_markdown
    assert (
        "does not cryptographically prove honest participant computation"
        in bundle.model_card_markdown
    )
    assert "does not claim paper-scale LeWorldModel performance" in (
        bundle.model_card_markdown
    )
    assert _artifact_sha(bundle, "run-manifest") == bundle.training.run_manifest_hash
    assert (
        _artifact_sha(bundle, "checkpoint-header")
        == bundle.training.checkpoint_header_sha256
    )
    assert (
        _artifact_sha(bundle, "checkpoint-weights")
        == bundle.training.checkpoint_weights_sha256
    )


def test_checked_in_phase3_run_contracts_are_standalone_historical_records() -> None:
    manifest = load_consortium_manifest(
        Path("docs/evidence/phase3_long_run_smoke_manifest.json")
    )
    registry = load_phase3_dataset_registry(
        Path("docs/evidence/phase3_long_run_smoke_dataset_registry.json")
    )

    assert manifest.public_probe.hash_contract == "legacy-unscoped"
    assert "Historical pre-correctness-fix" in manifest.claim_boundary
    assert "optimizer consumed plaintext participant deltas" in manifest.claim_boundary
    assert "neither effective nor cumulative DP" in manifest.claim_boundary
    assert "does not validate the corrected runtime" in manifest.claim_boundary
    assert registry.public_probe.hash_contract == "legacy-unscoped"
    assert "Historical pre-correctness-fix" in registry.claim_boundary
    assert "not a current operational-training contract" in registry.claim_boundary
    assert "#335" in registry.claim_boundary


def test_phase3_evidence_bundle_rejects_missing_artifact() -> None:
    raw = json.loads(Path("docs/evidence/phase3_evidence_bundle.json").read_text())
    raw["artifact_checks"][0]["exists"] = False
    raw["artifact_checks"][0]["error"] = "missing in test"

    with pytest.raises(ConfigError):
        parse_phase3_evidence_bundle(raw)


def test_phase3_evidence_bundle_rejects_artifact_hash_drift() -> None:
    raw = json.loads(Path("docs/evidence/phase3_evidence_bundle.json").read_text())
    _raw_artifact(raw, "checkpoint-header")["sha256"] = "0" * 64

    with pytest.raises(ConfigError, match="artifact hash mismatch"):
        parse_phase3_evidence_bundle(raw)


def test_phase3_evidence_bundle_rejects_missing_bound_artifact_hash() -> None:
    raw = json.loads(Path("docs/evidence/phase3_evidence_bundle.json").read_text())
    _raw_artifact(raw, "run-manifest")["sha256"] = None

    with pytest.raises(ConfigError, match="missing sha256"):
        parse_phase3_evidence_bundle(raw)


def test_phase3_evidence_bundle_rejects_future_schema_first() -> None:
    with pytest.raises(SchemaVersionMismatch):
        parse_phase3_evidence_bundle(
            {"schema_version": PHASE3_EVIDENCE_BUNDLE_SCHEMA_VERSION + 1}
        )


def test_phase3_bundle_residency_rejects_private_surfaces() -> None:
    with pytest.raises(ConfigError):
        validate_phase3_bundle_residency({"raw_actions": [1, 2, 3]})
    with pytest.raises(ConfigError):
        validate_phase3_bundle_residency({"artifact": "/Users/alice/private.h5"})
    with pytest.raises(ConfigError):
        validate_phase3_bundle_residency({"artifact": "/private/var/tmp/private.h5"})
    with pytest.raises(ConfigError):
        validate_phase3_bundle_residency({"token_value": "hf_secretToken1234"})


def test_phase3_local_artifact_check_redacts_absolute_default_uri(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")

    check = local_artifact_check(
        kind="training-report",
        label="test artifact",
        path=artifact,
    )

    assert check.uri == "artifact://local/artifact.json"
    assert check.sha256 is not None
    assert str(tmp_path) not in check.uri


def test_phase3_bundle_script_generates_and_validates(tmp_path: Path) -> None:
    output = tmp_path / "phase3_evidence_bundle.json"
    model_card = tmp_path / "phase3_model_card.md"
    manifest = tmp_path / "phase3_long_run_manifest.json"
    registry = tmp_path / "phase3_long_run_dataset_registry.json"
    long_run_report = tmp_path / "phase3_long_run_report.json"
    eval_report_path = tmp_path / "phase3_eval_report.json"
    observability_report_path = tmp_path / "phase3_observability_report.json"
    run_dir = tmp_path / "phase3-long-run-smoke"
    long_run = run_phase3_long_run_smoke(run_dir=run_dir, rounds=10)
    write_phase3_long_run_report(long_run, long_run_report)
    eval_report = build_phase3_eval_report(long_run_report)
    write_phase3_eval_report(eval_report, eval_report_path)
    observability_report = build_phase3_observability_report(
        long_run_report_path=long_run_report,
        eval_report_path=eval_report_path,
        run_dir=tmp_path / "phase3-observability-smoke",
        output_uri="artifact://test/phase3_observability_report.json",
    )
    write_phase3_observability_report(observability_report, observability_report_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase3_bundle.py",
            "--long-run-report",
            str(long_run_report),
            "--eval-report",
            str(eval_report_path),
            "--observability-report",
            str(observability_report_path),
            "--manifest-output",
            str(manifest),
            "--registry-output",
            str(registry),
            "--run-manifest",
            str(run_dir / "phase3_run_manifest.json"),
            "--checkpoint-header",
            str(run_dir / "coordinator-artifacts" / "round-00010" / "header.json"),
            "--checkpoint-weights",
            str(
                run_dir
                / "coordinator-artifacts"
                / "round-00010"
                / "weights.safetensors"
            ),
            "--output",
            str(output),
            "--model-card-output",
            str(model_card),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote" in result.stdout
    bundle = load_phase3_evidence_bundle(output)
    assert bundle.schema_version == PHASE3_EVIDENCE_BUNDLE_SCHEMA_VERSION
    assert bundle.privacy_aggregation.secure_sum_rounds == 0
    assert bundle.privacy_aggregation.post_commit_cross_check_rounds == 10
    assert bundle.privacy_aggregation.dp_accounted_rounds == 10
    assert bundle.privacy_aggregation.effective_dp_rounds == 0
    assert model_card.read_text() == bundle.model_card_markdown
    assert manifest.exists()
    assert registry.exists()
    assert not any(str(tmp_path) in check.uri for check in bundle.artifact_checks)

    validate = subprocess.run(
        [
            sys.executable,
            "scripts/phase3_bundle.py",
            "--validate",
            str(output),
            "--model-card",
            str(model_card),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "validated" in validate.stdout


def _artifact_sha(bundle: Phase3EvidenceBundle, kind: str) -> str | None:
    for check in bundle.artifact_checks:
        if check.kind == kind:
            return check.sha256
    raise AssertionError(f"missing artifact check: {kind}")


def _raw_artifact(raw: dict[str, Any], kind: str) -> dict[str, Any]:
    checks = raw["artifact_checks"]
    assert isinstance(checks, list)
    for check in checks:
        assert isinstance(check, dict)
        if check["kind"] == kind:
            return check
    raise AssertionError(f"missing artifact check: {kind}")
