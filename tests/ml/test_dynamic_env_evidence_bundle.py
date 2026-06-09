"""RFC-0017 dynamic-env observability, benchmark, and evidence bundle contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from lensemble.errors import ConfigError, EvaluationError, SchemaVersionMismatch
from lensemble.eval import (
    DYNAMIC_ENV_DOWNSTREAM_REPORT_SCHEMA_VERSION,
    DynamicEnvCheckpointRef,
    DynamicEnvControlReport,
    DynamicEnvDownstreamEvalReport,
    write_dynamic_env_downstream_eval_report,
)
from lensemble.federation import (
    DYNAMIC_ENV_BENCHMARK_REPORT_SCHEMA_VERSION,
    DYNAMIC_ENV_EVIDENCE_BUNDLE_SCHEMA_VERSION,
    DYNAMIC_ENV_OBSERVABILITY_REPORT_SCHEMA_VERSION,
    build_dynamic_env_benchmark_report,
    build_dynamic_env_evidence_bundle,
    build_dynamic_env_observability_report,
    dynamic_env_artifact_checks,
    load_dynamic_env_evidence_bundle,
    parse_dynamic_env_benchmark_report,
    parse_dynamic_env_evidence_bundle,
    parse_dynamic_env_observability_report,
    run_phase3_long_run_smoke,
    write_dynamic_env_benchmark_report,
    write_dynamic_env_evidence_bundle_outputs,
    write_dynamic_env_observability_report,
    write_phase3_long_run_report,
)


def _control(label: str, r2: float, success_rate: float) -> DynamicEnvControlReport:
    return DynamicEnvControlReport(
        label=label,
        checkpoint=DynamicEnvCheckpointRef(
            repo_id=f"abdelstark/lensemble-dynamic-{label}",
            revision="0123456789abcdef0123456789abcdef01234567",
            checkpoint_hash=f"{len(label):064x}",
        ),
        state_probe_r2=r2,
        success_rate=success_rate,
        effective_rank=16.0,
        metric_boundary="state_probe_r2 is binding; latent metrics are supporting and gameable",
    )


def _downstream_report() -> DynamicEnvDownstreamEvalReport:
    return DynamicEnvDownstreamEvalReport(
        schema_version=DYNAMIC_ENV_DOWNSTREAM_REPORT_SCHEMA_VERSION,
        generated_at=datetime(2026, 6, 9, tzinfo=timezone.utc),
        task_env_id="kinematic://swipe-dot",
        held_out_data_ref="synthetic-dynamic://swipe-dot?seed=99&n_episodes=8&steps=64&image_size=48",
        controls=(
            _control("federated", 0.72, 0.45),
            _control("naive-fedavg", 0.41, 0.16),
            _control("local-only", 0.33, 0.12),
            _control("random-encoder", 0.02, 0.08),
        ),
        claim_boundary=(
            "synthetic control env; state_probe_r2 is the binding ground-truth metric; "
            "latent-MPC and skill metrics are gameable supporting signals; no paper-scale robotics claim"
        ),
        source_report_uri="artifact://dynamic-env/downstream.json",
    )


def _write_text_artifact(path: Path, payload: str = "{}\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def _bundle_fixture(tmp_path: Path) -> dict[str, Any]:
    run_dir = tmp_path / "run"
    long_run = run_phase3_long_run_smoke(run_dir=run_dir, rounds=2)
    long_run_path = tmp_path / "dynamic_env_long_run_report.json"
    write_phase3_long_run_report(long_run, long_run_path)

    downstream_path = tmp_path / "dynamic_env_downstream_eval_report.json"
    downstream = _downstream_report()
    write_dynamic_env_downstream_eval_report(downstream, downstream_path)

    observability = build_dynamic_env_observability_report(
        long_run_report_path=long_run_path,
        run_manifest_path=run_dir / "phase3_run_manifest.json",
    )
    observability_path = tmp_path / "dynamic_env_observability_report.json"
    write_dynamic_env_observability_report(observability, observability_path)

    benchmark = build_dynamic_env_benchmark_report(
        downstream_report=downstream,
        long_run=long_run,
    )
    benchmark_path = tmp_path / "dynamic_env_benchmark_report.json"
    write_dynamic_env_benchmark_report(benchmark, benchmark_path)

    manifest_path = _write_text_artifact(
        tmp_path / "dynamic_env_consortium_manifest.json"
    )
    registry_path = _write_text_artifact(tmp_path / "dynamic_env_dataset_registry.json")
    header = run_dir / "coordinator-artifacts" / "round-00002" / "header.json"
    weights = run_dir / "coordinator-artifacts" / "round-00002" / "weights.safetensors"

    checks = dynamic_env_artifact_checks(
        manifest_path=manifest_path,
        registry_path=registry_path,
        training_report_path=long_run_path,
        observability_report_path=observability_path,
        benchmark_report_path=benchmark_path,
        run_manifest_path=run_dir / "phase3_run_manifest.json",
        checkpoint_header_path=header,
        checkpoint_weights_path=weights,
    )
    bundle = build_dynamic_env_evidence_bundle(
        benchmark=benchmark,
        observability=observability,
        artifact_checks=checks,
        run_manifest_path=run_dir / "phase3_run_manifest.json",
        checkpoint_header_path=header,
        checkpoint_weights_path=weights,
        benchmark_report_path=benchmark_path,
        observability_report_path=observability_path,
    )
    bundle_path = tmp_path / "dynamic_env_evidence_bundle.json"
    card_path = tmp_path / "dynamic_env_model_card.md"
    write_dynamic_env_evidence_bundle_outputs(
        bundle, bundle_path=bundle_path, model_card_path=card_path
    )
    return {
        "run_dir": run_dir,
        "long_run_path": long_run_path,
        "downstream_path": downstream_path,
        "observability_path": observability_path,
        "benchmark_path": benchmark_path,
        "manifest_path": manifest_path,
        "registry_path": registry_path,
        "header": header,
        "weights": weights,
        "bundle_path": bundle_path,
        "card_path": card_path,
        "bundle": bundle,
    }


def test_dynamic_env_observability_and_bundle_round_trip(tmp_path: Path) -> None:
    fixture = _bundle_fixture(tmp_path)
    bundle = load_dynamic_env_evidence_bundle(fixture["bundle_path"])

    assert bundle.schema_version == DYNAMIC_ENV_EVIDENCE_BUNDLE_SCHEMA_VERSION
    assert (
        bundle.benchmark.schema_version == DYNAMIC_ENV_BENCHMARK_REPORT_SCHEMA_VERSION
    )
    assert (
        bundle.observability.schema_version
        == DYNAMIC_ENV_OBSERVABILITY_REPORT_SCHEMA_VERSION
    )
    assert bundle.raw_data_in_report is False
    assert all(check.exists for check in bundle.artifact_checks)
    assert (
        fixture["card_path"].read_text(encoding="utf-8") == bundle.model_card_markdown
    )
    assert bundle.benchmark.model_arch == "scratch"
    assert bundle.benchmark.controls[0].state_probe_r2 >= 0.5
    card = bundle.model_card_markdown.lower()
    assert "closed-loop success_rate is reported non-binding" in card
    assert "not vjepa2-vit-l" in bundle.model_card_markdown
    assert "does not cryptographically prove honest participant computation" in (
        bundle.model_card_markdown
    )


def test_dynamic_env_bundle_rejects_missing_required_kind(tmp_path: Path) -> None:
    fixture = _bundle_fixture(tmp_path)
    raw = json.loads(fixture["bundle_path"].read_text(encoding="utf-8"))
    raw["artifact_checks"] = [
        check
        for check in raw["artifact_checks"]
        if check["kind"] != "privacy-aggregation-report"
    ]

    with pytest.raises(ConfigError, match="missing artifact kinds"):
        parse_dynamic_env_evidence_bundle(raw)


def test_dynamic_env_bundle_rejects_model_card_drift(tmp_path: Path) -> None:
    fixture = _bundle_fixture(tmp_path)
    raw = json.loads(fixture["bundle_path"].read_text(encoding="utf-8"))
    raw["model_card_markdown"] = raw["model_card_markdown"].replace(
        "not vjepa2-vit-l", "warm-started vjepa2-vit-l"
    )

    with pytest.raises(ConfigError, match="model card"):
        parse_dynamic_env_evidence_bundle(raw)


def test_dynamic_env_benchmark_rejects_failed_binding_gate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    long_run = run_phase3_long_run_smoke(run_dir=run_dir, rounds=1)
    report = _downstream_report()
    raw = report.model_dump(mode="json")
    raw["controls"][0]["state_probe_r2"] = 0.49
    weak = DynamicEnvDownstreamEvalReport.model_validate(raw)

    with pytest.raises(EvaluationError, match="below gate"):
        build_dynamic_env_benchmark_report(
            downstream_report=weak,
            long_run=long_run,
        )


def test_dynamic_env_reports_reject_future_schemas_first() -> None:
    with pytest.raises(SchemaVersionMismatch):
        parse_dynamic_env_observability_report(
            {"schema_version": DYNAMIC_ENV_OBSERVABILITY_REPORT_SCHEMA_VERSION + 1}
        )
    with pytest.raises(SchemaVersionMismatch):
        parse_dynamic_env_benchmark_report(
            {"schema_version": DYNAMIC_ENV_BENCHMARK_REPORT_SCHEMA_VERSION + 1}
        )
    with pytest.raises(SchemaVersionMismatch):
        parse_dynamic_env_evidence_bundle(
            {"schema_version": DYNAMIC_ENV_EVIDENCE_BUNDLE_SCHEMA_VERSION + 1}
        )


def test_dynamic_env_scripts_generate_and_validate(tmp_path: Path) -> None:
    fixture = _bundle_fixture(tmp_path)
    obs_out = tmp_path / "script_observability.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/dynamic_env_observability_report.py",
            "--long-run-report",
            str(fixture["long_run_path"]),
            "--run-manifest",
            str(fixture["run_dir"] / "phase3_run_manifest.json"),
            "--output",
            str(obs_out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    bench_out = tmp_path / "script_benchmark.json"
    bundle_out = tmp_path / "script_bundle.json"
    card_out = tmp_path / "script_model_card.md"
    subprocess.run(
        [
            sys.executable,
            "scripts/dynamic_env_benchmark.py",
            "--downstream-report",
            str(fixture["downstream_path"]),
            "--long-run-report",
            str(fixture["long_run_path"]),
            "--observability-report",
            str(obs_out),
            "--manifest",
            str(fixture["manifest_path"]),
            "--registry",
            str(fixture["registry_path"]),
            "--run-manifest",
            str(fixture["run_dir"] / "phase3_run_manifest.json"),
            "--checkpoint-header",
            str(fixture["header"]),
            "--checkpoint-weights",
            str(fixture["weights"]),
            "--output",
            str(bench_out),
            "--bundle-output",
            str(bundle_out),
            "--model-card-output",
            str(card_out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    validate = subprocess.run(
        [
            sys.executable,
            "scripts/dynamic_env_benchmark.py",
            "--validate",
            str(bundle_out),
            "--model-card",
            str(card_out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "validated" in validate.stdout
