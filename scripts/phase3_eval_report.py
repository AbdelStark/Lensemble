#!/usr/bin/env python3
"""Generate or validate the Phase 3 eval and matched-control report.

The ``--*-report`` / ``--*-revision`` flags bind the four published Phase 3
matched control runs (see GitHub issue #244). When the previously-blocked
controls (``local-only``, ``naive-fedavg``, ``fork-a-frozen-encoder``) are
supplied, they are flipped from blocked rows to completed metric rows bound to
the published checkpoint revision, final global hash, config hash,
run-manifest hash (or report sha256 for the no-aggregation local-only
control), and the residency-safe per-round JEPA gauge metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from lensemble.eval import (
    Phase3CompletedControlInput,
    Phase3ControlGaugeValue,
    build_phase3_eval_report,
    load_phase3_eval_report,
    write_phase3_eval_report,
)

_CONTROL_ENV_PREFIX = "phase3://consortium-control"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long-run-report",
        type=Path,
        default=Path("docs/evidence/phase3_long_run_smoke_report.json"),
        help="Phase 3 long-run report produced by scripts/phase3_consortium_smoke.py.",
    )
    parser.add_argument(
        "--anchored-report",
        type=Path,
        default=None,
        help=(
            "Published anchored-federation control run report (lambda_anc=0.01). "
            "Used as the round-0 frame-drift reference for the gauge contrast."
        ),
    )
    parser.add_argument("--anchored-revision", type=str, default=None)
    parser.add_argument(
        "--naive-report",
        type=Path,
        default=None,
        help="Published naive-fedavg (lambda_anc=0) control run report.",
    )
    parser.add_argument("--naive-revision", type=str, default=None)
    parser.add_argument("--naive-run-manifest", type=Path, default=None)
    parser.add_argument(
        "--fork-a-report",
        type=Path,
        default=None,
        help="Published fork-a frozen-encoder safe-degrade control run report.",
    )
    parser.add_argument("--fork-a-revision", type=str, default=None)
    parser.add_argument("--fork-a-run-manifest", type=Path, default=None)
    parser.add_argument(
        "--local-only-report",
        type=Path,
        default=None,
        help="Published no-aggregation local-only control report.",
    )
    parser.add_argument("--local-only-revision", type=str, default=None)
    parser.add_argument(
        "--anchored-run-manifest",
        type=Path,
        default=None,
        help="Anchored control run manifest used only to report the contrast reference.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/phase3_eval_report.json"),
        help="Where to write the generated Phase 3 eval report.",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        default=None,
        help="Validate an existing eval report instead of generating one.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _round0(report: dict[str, object], key: str) -> float:
    rounds = report["rounds"]
    assert isinstance(rounds, list) and rounds, "control report has no rounds"
    first = rounds[0]
    assert isinstance(first, dict)
    return float(first[key])


def _federated_control_input(
    *,
    control_role: str,
    report_path: Path,
    revision: str,
    run_manifest_path: Path,
    note: str,
) -> Phase3CompletedControlInput:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    run_shape = report["run_shape"]
    assert isinstance(run_shape, dict)
    model_repo = run_shape["artifact_targets"]["model_repo"]
    return Phase3CompletedControlInput(
        control_role=control_role,  # type: ignore[arg-type]
        task_env_id=f"{_CONTROL_ENV_PREFIX}-{control_role}",
        repo=str(model_repo),
        revision=revision,
        checkpoint_hash=str(report["final_global_model_hash"]),
        config_hash=str(report["config_hash"]),
        run_manifest_hash=_sha256(run_manifest_path),
        seed=int(run_shape.get("root_seed", 0)),
        source_label=f"Phase 3 {control_role} control run report",
        source_uri=str(model_repo),
        source_report_sha256=_sha256(report_path),
        source_schema_name="phase3_long_run_report",
        source_schema_version=int(report["schema_version"]),
        gauges=(
            Phase3ControlGaugeValue(
                metric="latent_frame_drift_deg",
                value=_round0(report, "frame_drift_deg"),
                notes="round-0 inter-participant latent frame-drift at aggregation",
            ),
            Phase3ControlGaugeValue(
                metric="effective_rank",
                value=_round0(report, "effective_rank"),
                notes="round-0 global-representation effective rank",
            ),
        ),
        note=note,
    )


def _local_only_control_input(
    *,
    report_path: Path,
    revision: str,
) -> Phase3CompletedControlInput:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    per_participant = report["per_participant"]
    assert isinstance(per_participant, list) and per_participant
    mean_rank = sum(float(p["effective_rank"]) for p in per_participant) / len(
        per_participant
    )
    report_sha = _sha256(report_path)
    return Phase3CompletedControlInput(
        control_role="local-only",
        task_env_id=f"{_CONTROL_ENV_PREFIX}-local-only",
        repo="abdelstark/lensemble-phase3-consortium-local-only",
        revision=revision,
        # No-aggregation control: there is no committed global checkpoint, so the
        # immutable report sha256 binds checkpoint, run-manifest, and source.
        checkpoint_hash=report_sha,
        config_hash=report_sha,
        run_manifest_hash=report_sha,
        seed=0,
        source_label="Phase 3 local-only (no-aggregation) control report",
        source_uri="abdelstark/lensemble-phase3-consortium-local-only",
        source_report_sha256=report_sha,
        source_schema_name="phase3_local_only_report",
        source_schema_version=1,
        gauges=(
            Phase3ControlGaugeValue(
                metric="latent_frame_drift_deg",
                value=float(report["frame_drift_deg"]),
                notes="inter-participant latent frame-drift across isolated silos",
            ),
            Phase3ControlGaugeValue(
                metric="effective_rank",
                value=mean_rank,
                notes="mean per-participant held-out effective rank (healthy local training)",
            ),
        ),
        note=(
            "No-aggregation local-only control: silos train healthily but diverge "
            "maximally; bound to the published report sha256 and revision, not a "
            "global checkpoint."
        ),
    )


def _completed_controls(args: argparse.Namespace) -> list[Phase3CompletedControlInput]:
    controls: list[Phase3CompletedControlInput] = []
    if args.naive_report is not None:
        if args.naive_revision is None or args.naive_run_manifest is None:
            raise SystemExit(
                "--naive-report requires --naive-revision and --naive-run-manifest"
            )
        controls.append(
            _federated_control_input(
                control_role="naive-fedavg",
                report_path=args.naive_report,
                revision=args.naive_revision,
                run_manifest_path=args.naive_run_manifest,
                note=(
                    "Naive-FedAvg (lambda_anc=0) unanchored control: round-0 frame-drift "
                    "180 deg, the maximal divergence the frame anchor reduces to 48.97 deg."
                ),
            )
        )
    if args.fork_a_report is not None:
        if args.fork_a_revision is None or args.fork_a_run_manifest is None:
            raise SystemExit(
                "--fork-a-report requires --fork-a-revision and --fork-a-run-manifest"
            )
        controls.append(
            _federated_control_input(
                control_role="fork-a-frozen-encoder",
                report_path=args.fork_a_report,
                revision=args.fork_a_revision,
                run_manifest_path=args.fork_a_run_manifest,
                note=(
                    "Fork-A frozen-encoder safe-degrade baseline: constant 0 deg frame-drift "
                    "and constant effective rank, the structural lower bound on divergence."
                ),
            )
        )
    if args.local_only_report is not None:
        if args.local_only_revision is None:
            raise SystemExit("--local-only-report requires --local-only-revision")
        controls.append(
            _local_only_control_input(
                report_path=args.local_only_report,
                revision=args.local_only_revision,
            )
        )
    return controls


def main() -> None:
    args = _args()
    if args.validate is not None:
        report = load_phase3_eval_report(args.validate)
        print(
            "validated "
            f"{args.validate}: {len(report.metric_rows)} metric rows, "
            f"{len(report.blocked_controls)} blocked controls"
        )
        return

    completed: Sequence[Phase3CompletedControlInput] = _completed_controls(args)
    report = build_phase3_eval_report(
        args.long_run_report, completed_controls=completed
    )
    path = write_phase3_eval_report(report, args.output)
    load_phase3_eval_report(path)
    print(
        f"wrote {path}: {len(report.metric_rows)} metric rows, "
        f"{len(report.blocked_controls)} blocked controls"
    )


if __name__ == "__main__":
    main()
