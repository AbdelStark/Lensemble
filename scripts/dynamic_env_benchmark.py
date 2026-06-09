#!/usr/bin/env python3
"""Generate or validate the RFC-0017 dynamic-env benchmark, bundle, and model card."""

from __future__ import annotations

import argparse
from pathlib import Path

from lensemble.eval import load_dynamic_env_downstream_eval_report
from lensemble.federation import (
    build_dynamic_env_benchmark_report,
    build_dynamic_env_evidence_bundle,
    dynamic_env_artifact_checks,
    load_dynamic_env_benchmark_report,
    load_dynamic_env_evidence_bundle,
    load_dynamic_env_observability_report,
    load_phase3_long_run_report,
    write_dynamic_env_benchmark_report,
    write_dynamic_env_evidence_bundle_outputs,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--downstream-report",
        type=Path,
        default=Path("docs/evidence/dynamic_env_downstream_eval_report.json"),
        help="Dynamic-env downstream report with per-control state_probe_r2 rows.",
    )
    parser.add_argument(
        "--long-run-report",
        type=Path,
        default=Path("docs/evidence/dynamic_env_long_run_report.json"),
        help="Dynamic-env long-run training report.",
    )
    parser.add_argument(
        "--observability-report",
        type=Path,
        default=Path("docs/evidence/dynamic_env_observability_report.json"),
        help="Dynamic-env observability/privacy report from scripts/dynamic_env_observability_report.py.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/evidence/dynamic_env_consortium_manifest.json"),
        help="Dynamic-env consortium manifest.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("docs/evidence/dynamic_env_dataset_registry.json"),
        help="Dynamic-env dataset/probe registry.",
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=Path("runs/dynamic-env/phase3_run_manifest.json"),
        help="Run manifest emitted by the dynamic-env launcher run.",
    )
    parser.add_argument(
        "--checkpoint-header",
        type=Path,
        default=Path("runs/dynamic-env/coordinator-artifacts/round-00010/header.json"),
        help="Final dynamic-env checkpoint header.",
    )
    parser.add_argument(
        "--checkpoint-weights",
        type=Path,
        default=Path(
            "runs/dynamic-env/coordinator-artifacts/round-00010/weights.safetensors"
        ),
        help="Final dynamic-env checkpoint weights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/dynamic_env_benchmark_report.json"),
        help="Where to write the dynamic-env benchmark report.",
    )
    parser.add_argument(
        "--bundle-output",
        type=Path,
        default=Path("docs/evidence/dynamic_env_evidence_bundle.json"),
        help="Where to write the dynamic-env evidence bundle.",
    )
    parser.add_argument(
        "--model-card-output",
        type=Path,
        default=Path("docs/evidence/dynamic_env_model_card.md"),
        help="Where to write the dynamic-env model card.",
    )
    parser.add_argument(
        "--publication-status",
        choices=("local_smoke", "published", "blocked"),
        default="local_smoke",
        help="Publication status recorded in the dynamic-env bundle.",
    )
    parser.add_argument(
        "--model-repo-revision",
        default="local-smoke",
        help="Immutable model repo revision when publication-status=published.",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        default=None,
        help="Validate an existing dynamic-env evidence bundle instead of generating one.",
    )
    parser.add_argument(
        "--model-card",
        type=Path,
        default=Path("docs/evidence/dynamic_env_model_card.md"),
        help="Model-card path to compare when --validate is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.validate is not None:
        bundle = load_dynamic_env_evidence_bundle(args.validate)
        if args.model_card.exists():
            card = args.model_card.read_text(encoding="utf-8")
            if card != bundle.model_card_markdown:
                raise SystemExit(
                    f"model card {args.model_card} does not match bundle markdown"
                )
        print(
            f"validated {args.validate}: {len(bundle.benchmark.controls)} controls, "
            f"model card bytes={len(bundle.model_card_markdown)}"
        )
        return

    downstream = load_dynamic_env_downstream_eval_report(args.downstream_report)
    long_run = load_phase3_long_run_report(args.long_run_report)
    observability = load_dynamic_env_observability_report(args.observability_report)
    benchmark = build_dynamic_env_benchmark_report(
        downstream_report=downstream,
        long_run=long_run,
    )
    benchmark_path = write_dynamic_env_benchmark_report(benchmark, args.output)
    load_dynamic_env_benchmark_report(benchmark_path)

    checks = dynamic_env_artifact_checks(
        manifest_path=args.manifest,
        registry_path=args.registry,
        training_report_path=args.long_run_report,
        observability_report_path=args.observability_report,
        benchmark_report_path=benchmark_path,
        run_manifest_path=args.run_manifest,
        checkpoint_header_path=args.checkpoint_header,
        checkpoint_weights_path=args.checkpoint_weights,
    )
    bundle = build_dynamic_env_evidence_bundle(
        benchmark=benchmark,
        observability=observability,
        artifact_checks=checks,
        run_manifest_path=args.run_manifest,
        checkpoint_header_path=args.checkpoint_header,
        checkpoint_weights_path=args.checkpoint_weights,
        benchmark_report_path=benchmark_path,
        observability_report_path=args.observability_report,
        publication_status=args.publication_status,
        model_repo_revision=args.model_repo_revision,
    )
    write_dynamic_env_evidence_bundle_outputs(
        bundle,
        bundle_path=args.bundle_output,
        model_card_path=args.model_card_output,
    )
    print(
        f"wrote {benchmark_path}, {args.bundle_output}, {args.model_card_output}: "
        f"{len(bundle.benchmark.controls)} controls"
    )


if __name__ == "__main__":
    main()
