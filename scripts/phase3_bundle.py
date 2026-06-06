#!/usr/bin/env python3
"""Generate or validate the final Phase 3 evidence bundle and model card."""

from __future__ import annotations

import argparse
from pathlib import Path

from lensemble.config.consortium import load_consortium_manifest
from lensemble.data.phase3 import load_phase3_dataset_registry
from lensemble.eval.phase3 import load_phase3_eval_report
from lensemble.federation import (
    build_phase3_evidence_bundle,
    check_hf_artifact_exists,
    load_phase3_evidence_bundle,
    load_phase3_long_run_report,
    load_phase3_observability_report,
    local_artifact_check,
    materialize_phase3_run_contracts,
    write_phase3_bundle_outputs,
)
from lensemble.federation.phase3_bundle import (
    Phase3ArtifactCheck,
    Phase3ArtifactKind,
    local_artifact_uri,
)

_MODEL_REPO_ID = "abdelstark/lensemble-phase3-consortium-checkpoint"
_DATASET_REPO_ID = "abdelstark/lensemble-phase3-consortium-data"
_REMOTE_MODEL_ARTIFACTS: tuple[tuple[Phase3ArtifactKind, str, str], ...] = (
    ("model-card", "Phase 3 model card", "README.md"),
    (
        "evidence-bundle",
        "Phase 3 evidence bundle",
        "reports/phase3_evidence_bundle.json",
    ),
    (
        "training-report",
        "Phase 3 published long-run report",
        "reports/phase3_long_run_smoke_report.json",
    ),
    (
        "observability-report",
        "Phase 3 published observability report",
        "reports/phase3_observability_report.json",
    ),
    (
        "eval-control-report",
        "Phase 3 published eval report",
        "reports/phase3_eval_report.json",
    ),
    (
        "checkpoint-header",
        "Phase 3 published checkpoint header",
        "artifacts/final/header.json",
    ),
    (
        "checkpoint-weights",
        "Phase 3 published checkpoint weights",
        "artifacts/final/weights.safetensors",
    ),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long-run-report",
        type=Path,
        default=Path("docs/evidence/phase3_long_run_smoke_report.json"),
    )
    parser.add_argument(
        "--eval-report",
        type=Path,
        default=Path("docs/evidence/phase3_eval_report.json"),
    )
    parser.add_argument(
        "--observability-report",
        type=Path,
        default=Path("docs/evidence/phase3_observability_report.json"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("docs/evidence/phase3_long_run_manifest.json"),
    )
    parser.add_argument(
        "--registry-output",
        type=Path,
        default=Path("docs/evidence/phase3_long_run_dataset_registry.json"),
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=Path("runs/phase3-long-run-smoke/phase3_run_manifest.json"),
    )
    parser.add_argument(
        "--checkpoint-header",
        type=Path,
        default=Path(
            "runs/phase3-long-run-smoke/coordinator-artifacts/round-00010/header.json"
        ),
    )
    parser.add_argument(
        "--checkpoint-weights",
        type=Path,
        default=Path(
            "runs/phase3-long-run-smoke/coordinator-artifacts/round-00010/weights.safetensors"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/phase3_evidence_bundle.json"),
    )
    parser.add_argument(
        "--model-card-output",
        type=Path,
        default=Path("docs/evidence/phase3_model_card.md"),
    )
    parser.add_argument(
        "--publication-status",
        choices=("local_smoke", "published", "blocked"),
        default="local_smoke",
    )
    parser.add_argument("--model-revision", default="local-smoke")
    parser.add_argument("--dataset-revision", default="local-smoke")
    parser.add_argument(
        "--remote-check",
        action="store_true",
        help="Also check published model-repo artifacts on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        default=None,
        help="Validate an existing bundle instead of generating one.",
    )
    parser.add_argument(
        "--model-card",
        type=Path,
        default=Path("docs/evidence/phase3_model_card.md"),
        help="Model-card path to compare when --validate is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.validate is not None:
        bundle = load_phase3_evidence_bundle(args.validate)
        if args.model_card.exists():
            model_card = args.model_card.read_text(encoding="utf-8")
            if model_card != bundle.model_card_markdown:
                raise SystemExit(
                    f"model card {args.model_card} does not match bundle markdown"
                )
        print(
            "validated "
            f"{args.validate}: {len(bundle.artifact_checks)} artifact checks, "
            f"publication_status={bundle.publication.status}"
        )
        return

    materialize_phase3_run_contracts(
        long_run_report_path=args.long_run_report,
        manifest_path=args.manifest_output,
        registry_path=args.registry_output,
    )
    manifest = load_consortium_manifest(args.manifest_output)
    registry = load_phase3_dataset_registry(args.registry_output)
    long_run = load_phase3_long_run_report(args.long_run_report)
    eval_report = load_phase3_eval_report(args.eval_report)
    observability = load_phase3_observability_report(args.observability_report)
    artifact_checks = _artifact_checks(args)
    bundle = build_phase3_evidence_bundle(
        manifest=manifest,
        registry=registry,
        long_run=long_run,
        eval_report=eval_report,
        observability_report=observability,
        artifact_checks=artifact_checks,
        checkpoint_header_path=args.checkpoint_header,
        checkpoint_weights_path=args.checkpoint_weights,
        model_repo_revision=args.model_revision,
        dataset_repo_revision=args.dataset_revision,
        publication_status=args.publication_status,
    )
    write_phase3_bundle_outputs(
        bundle,
        bundle_path=args.output,
        model_card_path=args.model_card_output,
    )
    print(
        f"wrote {args.output} and {args.model_card_output}: "
        f"{len(bundle.artifact_checks)} artifact checks, "
        f"publication_status={bundle.publication.status}"
    )


def _artifact_checks(args: argparse.Namespace) -> tuple[Phase3ArtifactCheck, ...]:
    checks: list[Phase3ArtifactCheck] = [
        local_artifact_check(
            kind="consortium-manifest",
            label="Phase 3 long-run consortium manifest",
            path=args.manifest_output,
        ),
        local_artifact_check(
            kind="dataset-probe-registry",
            label="Phase 3 long-run dataset/probe registry",
            path=args.registry_output,
        ),
        local_artifact_check(
            kind="training-report",
            label="Phase 3 long-run training report",
            path=args.long_run_report,
        ),
        local_artifact_check(
            kind="privacy-aggregation-report",
            label="Phase 3 aggregation/privacy rows embedded in long-run report",
            path=args.long_run_report,
            uri=f"{local_artifact_uri(args.long_run_report)}#rounds",
        ),
        local_artifact_check(
            kind="observability-report",
            label="Phase 3 observability/dropout report",
            path=args.observability_report,
        ),
        local_artifact_check(
            kind="eval-control-report",
            label="Phase 3 eval/control report",
            path=args.eval_report,
        ),
        local_artifact_check(
            kind="run-manifest",
            label="Phase 3 deterministic run manifest",
            path=args.run_manifest,
        ),
        local_artifact_check(
            kind="checkpoint-header",
            label="Phase 3 final checkpoint header",
            path=args.checkpoint_header,
        ),
        local_artifact_check(
            kind="checkpoint-weights",
            label="Phase 3 final checkpoint weights",
            path=args.checkpoint_weights,
        ),
    ]
    if args.remote_check:
        for kind, label, path_in_repo in _REMOTE_MODEL_ARTIFACTS:
            checks.append(
                check_hf_artifact_exists(
                    kind=kind,
                    label=label,
                    repo_type="model",
                    repo_id=_MODEL_REPO_ID,
                    revision=args.model_revision,
                    path_in_repo=path_in_repo,
                )
            )
    return tuple(checks)


if __name__ == "__main__":
    main()
