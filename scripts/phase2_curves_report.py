#!/usr/bin/env python3
"""Generate the Phase 2 baselines/curves evidence report.

Inputs are already residency-safe JSON reports. Download public Hub artifacts
with ``hf download`` first, then pass those local copies here so the generator
can hash the exact bytes it consumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lensemble.eval import (
    Phase2ClaimCurveInput,
    build_phase2_baselines_curves_report,
    parse_claim_mvp_report,
    parse_phase2_downstream_eval_report,
    phase2_source_report_ref_from_path,
)

_ANCHOR_REPO = "abdelstark/lensemble-phase2-so100-checkpoint"
_ANCHOR_REVISION = "da52ef380ac87317c89e87f048d65bae65c16b9e"
_ANCHOR_JOB_ID = "6a22ba68e6aa50b87b9ebef7"
_ANCHOR_JOB_URL = f"https://huggingface.co/jobs/abdelstark/{_ANCHOR_JOB_ID}"
_DOWNSTREAM_REVISION = "021a461eb789700209fcb49e99bb9bcc5d84bfe5"
_DOWNSTREAM_JOB_ID = "6a22c9e3ece949d7b3dca25a"
_DOWNSTREAM_JOB_URL = f"https://huggingface.co/jobs/abdelstark/{_DOWNSTREAM_JOB_ID}"
_NAIVE_REPO = "abdelstark/lensemble-phase2-so100-naive-fedavg"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchored-claim-report",
        type=Path,
        required=True,
        help="Local copy of the anchored Phase 2 claim_mvp_report.json.",
    )
    parser.add_argument(
        "--downstream-report",
        type=Path,
        default=Path("docs/evidence/phase2_downstream_eval_report.json"),
        help="Local copy of reports/phase2_downstream_eval_report.json.",
    )
    parser.add_argument(
        "--naive-fedavg-claim-report",
        type=Path,
        default=None,
        help="Optional local copy of a matched lambda_anc=0 claim report.",
    )
    parser.add_argument(
        "--naive-fedavg-job-id",
        default=None,
        help="HF Job id for the optional naive FedAvg report.",
    )
    parser.add_argument(
        "--naive-fedavg-revision",
        default=None,
        help="HF model repo revision for the optional naive FedAvg report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/phase2_baselines_curves_report.json"),
        help="Path to write the generated report JSON.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = _args()
    anchored_report = parse_claim_mvp_report(_read_json(args.anchored_claim_report))
    downstream_report = parse_phase2_downstream_eval_report(
        _read_json(args.downstream_report)
    )
    anchored_ref = phase2_source_report_ref_from_path(
        args.anchored_claim_report,
        label="anchored Phase 2 SO-100 claim report",
        schema_name="claim_mvp_report",
        schema_version=anchored_report.schema_version,
        uri=f"hf://models/{_ANCHOR_REPO}@{_ANCHOR_REVISION}/claim_mvp_report.json",
        repo_id=_ANCHOR_REPO,
        repo_type="model",
        revision=_ANCHOR_REVISION,
        path_in_repo="claim_mvp_report.json",
        job_id=_ANCHOR_JOB_ID,
        job_url=_ANCHOR_JOB_URL,
    )
    downstream_ref = phase2_source_report_ref_from_path(
        args.downstream_report,
        label="anchored Phase 2 downstream eval report",
        schema_name="phase2_downstream_eval_report",
        schema_version=downstream_report.schema_version,
        uri=(
            f"hf://models/{_ANCHOR_REPO}@{_DOWNSTREAM_REVISION}/"
            "reports/phase2_downstream_eval_report.json"
        ),
        repo_id=_ANCHOR_REPO,
        repo_type="model",
        revision=_DOWNSTREAM_REVISION,
        path_in_repo="reports/phase2_downstream_eval_report.json",
        job_id=_DOWNSTREAM_JOB_ID,
        job_url=_DOWNSTREAM_JOB_URL,
    )

    controls: list[Phase2ClaimCurveInput] = []
    if args.naive_fedavg_claim_report is not None:
        naive_report = parse_claim_mvp_report(
            _read_json(args.naive_fedavg_claim_report)
        )
        revision = args.naive_fedavg_revision
        uri = f"hf://models/{_NAIVE_REPO}/claim_mvp_report.json"
        if revision:
            uri = f"hf://models/{_NAIVE_REPO}@{revision}/claim_mvp_report.json"
        job_url = (
            f"https://huggingface.co/jobs/abdelstark/{args.naive_fedavg_job_id}"
            if args.naive_fedavg_job_id
            else None
        )
        controls.append(
            Phase2ClaimCurveInput(
                run_role="naive-fedavg",
                run_label="naive FedAvg control (lambda_anc=0)",
                report=naive_report,
                source_ref=phase2_source_report_ref_from_path(
                    args.naive_fedavg_claim_report,
                    label="naive FedAvg Phase 2 SO-100 claim report",
                    schema_name="claim_mvp_report",
                    schema_version=naive_report.schema_version,
                    uri=uri,
                    repo_id=_NAIVE_REPO,
                    repo_type="model",
                    revision=revision,
                    path_in_repo="claim_mvp_report.json",
                    job_id=args.naive_fedavg_job_id,
                    job_url=job_url,
                ),
                ablation_axis="lambda_anc",
            )
        )

    report = build_phase2_baselines_curves_report(
        anchored=Phase2ClaimCurveInput(
            run_role="anchored-federation",
            run_label="anchored federation (lambda_anc=0.01)",
            report=anchored_report,
            source_ref=anchored_ref,
        ),
        downstream_report=downstream_report,
        downstream_source_ref=downstream_ref,
        control_reports=tuple(controls),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
