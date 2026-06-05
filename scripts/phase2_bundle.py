#!/usr/bin/env python3
"""Generate the final Phase 2 evidence bundle and model card."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lensemble.eval import (
    parse_claim_mvp_report,
    parse_phase2_baselines_curves_report,
    parse_phase2_downstream_eval_report,
)
from lensemble.eval.phase2_bundle import (
    Phase2HubArtifactCheck,
    build_phase2_evidence_bundle,
    check_hf_artifact_exists,
    local_artifact_check,
    write_phase2_bundle_outputs,
)

_DATASET_REPO = "abdelstark/lensemble-phase2-so100-silos"
_DATASET_REVISION = "97336927606fea6fbfda308bb7cee6e7b48999fa"
_CHECKPOINT_REPO = "abdelstark/lensemble-phase2-so100-checkpoint"
_CHECKPOINT_REVISION = "da52ef380ac87317c89e87f048d65bae65c16b9e"
_DOWNSTREAM_REVISION = "021a461eb789700209fcb49e99bb9bcc5d84bfe5"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-smoke",
        type=Path,
        required=True,
        help="Local copy of phase2_dataset_smoke.json.",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        required=True,
        help="Local copy of phase2_silo_manifest.json.",
    )
    parser.add_argument(
        "--training-claim-report",
        type=Path,
        required=True,
        help="Local copy of the anchored claim_mvp_report.json.",
    )
    parser.add_argument(
        "--downstream-report",
        type=Path,
        default=Path("docs/evidence/phase2_downstream_eval_report.json"),
    )
    parser.add_argument(
        "--curves-report",
        type=Path,
        default=Path("docs/evidence/phase2_baselines_curves_report.json"),
    )
    parser.add_argument("--dataset-revision", default=_DATASET_REVISION)
    parser.add_argument("--checkpoint-revision", default=_CHECKPOINT_REVISION)
    parser.add_argument("--downstream-revision", default=_DOWNSTREAM_REVISION)
    parser.add_argument(
        "--curves-revision",
        default="main",
        help="Model repo revision containing reports/phase2_baselines_curves_report.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/phase2_evidence_bundle.json"),
    )
    parser.add_argument(
        "--model-card-output",
        type=Path,
        default=Path("docs/evidence/phase2_model_card.md"),
    )
    parser.add_argument(
        "--no-remote-check",
        action="store_true",
        help="Use passing local artifact checks instead of HTTP Hub checks.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(
    args: argparse.Namespace,
    *,
    kind: str,
    label: str,
    repo_type: str,
    repo_id: str,
    revision: str,
    path_in_repo: str,
) -> Phase2HubArtifactCheck:
    kwargs = {
        "kind": kind,
        "label": label,
        "repo_type": repo_type,
        "repo_id": repo_id,
        "revision": revision,
        "path_in_repo": path_in_repo,
    }
    if args.no_remote_check:
        return local_artifact_check(**kwargs)  # type: ignore[arg-type]
    return check_hf_artifact_exists(**kwargs)  # type: ignore[arg-type]


def _artifact_checks(
    args: argparse.Namespace, manifest: dict[str, object]
) -> list[Phase2HubArtifactCheck]:
    checks = [
        _check(
            args,
            kind="dataset-smoke-report",
            label="Phase 2 dataset smoke report",
            repo_type="dataset",
            repo_id=_DATASET_REPO,
            revision=args.dataset_revision,
            path_in_repo="phase2_dataset_smoke.json",
        ),
        _check(
            args,
            kind="dataset-split-manifest",
            label="Phase 2 dataset split manifest",
            repo_type="dataset",
            repo_id=_DATASET_REPO,
            revision=args.dataset_revision,
            path_in_repo="phase2_silo_manifest.json",
        ),
        _check(
            args,
            kind="training-claim-report",
            label="Phase 2 anchored training report",
            repo_type="model",
            repo_id=_CHECKPOINT_REPO,
            revision=args.checkpoint_revision,
            path_in_repo="claim_mvp_report.json",
        ),
        _check(
            args,
            kind="training-checkpoint-header",
            label="Phase 2 final checkpoint header",
            repo_type="model",
            repo_id=_CHECKPOINT_REPO,
            revision=args.checkpoint_revision,
            path_in_repo="artifacts/round-00003/header.json",
        ),
        _check(
            args,
            kind="downstream-eval-report",
            label="Phase 2 downstream eval report",
            repo_type="model",
            repo_id=_CHECKPOINT_REPO,
            revision=args.downstream_revision,
            path_in_repo="reports/phase2_downstream_eval_report.json",
        ),
        _check(
            args,
            kind="baselines-curves-report",
            label="Phase 2 baselines/curves report",
            repo_type="model",
            repo_id=_CHECKPOINT_REPO,
            revision=args.curves_revision,
            path_in_repo="reports/phase2_baselines_curves_report.json",
        ),
    ]
    for silo in manifest["silos"]:  # type: ignore[index]
        item = dict(silo)  # type: ignore[arg-type]
        checks.append(
            _check(
                args,
                kind="dataset-silo",
                label=f"Phase 2 dataset silo {item['filename']}",
                repo_type="dataset",
                repo_id=_DATASET_REPO,
                revision=args.dataset_revision,
                path_in_repo=str(item["filename"]),
            )
        )
    return checks


def main() -> None:
    args = _args()
    dataset_smoke = _read_json(args.dataset_smoke)
    dataset_manifest = _read_json(args.dataset_manifest)
    training_report = parse_claim_mvp_report(_read_json(args.training_claim_report))
    downstream_report = parse_phase2_downstream_eval_report(
        _read_json(args.downstream_report)
    )
    curves_report = parse_phase2_baselines_curves_report(_read_json(args.curves_report))
    bundle = build_phase2_evidence_bundle(
        dataset_smoke=dataset_smoke,
        dataset_manifest=dataset_manifest,
        training_report=training_report,
        downstream_report=downstream_report,
        curves_report=curves_report,
        artifact_checks=tuple(_artifact_checks(args, dataset_manifest)),
        dataset_revision=args.dataset_revision,
        checkpoint_revision=args.checkpoint_revision,
        curves_revision=args.curves_revision,
    )
    write_phase2_bundle_outputs(
        bundle,
        bundle_path=args.output,
        model_card_path=args.model_card_output,
    )
    print(json.dumps(bundle.model_dump(mode="json"), indent=2), flush=True)


if __name__ == "__main__":
    main()
