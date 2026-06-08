#!/usr/bin/env python3
"""Generate or validate the Phase 3 downstream latent-MPC eval report (#245).

The only prior downstream number was a ``synthetic://toy``, 1-sample,
``success_rate=0.5`` placeholder. This generator produces an honest, bounded
Phase 3 downstream eval report that:

1. Goes beyond ``synthetic://toy`` by binding the REAL held-out SO-100 latent
   metrics (the final-round ``effective_rank``/``val_pred`` computed by the
   headline consortium run on the disjoint held-out split
   ``phase3-so100-silo4.h5``, #242).
2. Records a NON-TOY latent-MPC planner budget (icem, horizon 16, 512
   planning samples, 8 iterations, 20 episodes, action_dim 6) — the budget a
   closed-loop run WOULD use — without executing a planner.
3. Documents the two specific blockers on a real closed-loop task-success
   number: the unvendored ``stable-worldmodel`` suite (#96) and the collapsing
   federated checkpoints (#244). It does NOT fabricate a task-success pass.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lensemble.eval import (
    Phase3DownstreamCheckpointRef,
    Phase3DownstreamPlannerBudget,
    Phase3TaskSuccessBlocker,
    build_phase3_downstream_eval_report,
    load_phase3_downstream_eval_report,
    write_phase3_downstream_eval_report,
)

_CHECKPOINT_REPO = "abdelstark/lensemble-phase3-consortium-checkpoint"
_CHECKPOINT_REVISION = "828e210cba4870b2be4ab573a5f0dd4ee30bae29"
_CHECKPOINT_HASH = "bb31c0922de639cb9220c4cc5fc35d79aec719eb6fcedb09159bdff8cfb8fd43"
_CONFIG_HASH = "27f2c77c9d47a7d053c01ab65f8d43aad79463b27d882f2d85ec28bc062cb2b2"

_TASK_ENV_ID = "so100-heldout://phase3-public-task-scale"
_HELD_OUT_DATA_REF = (
    "hf://datasets/abdelstark/lensemble-phase3-so100-silos/phase3-so100-silo4.h5"
)
_HELD_OUT_WINDOWS = 1216
_WINDOW_STEPS = 4

_CLAIM_BOUNDARY = (
    "Real held-out SO-100 latent evidence only: the final-round effective_rank "
    "and val_pred were measured by the headline federated consortium run on the "
    "disjoint held-out split phase3-so100-silo4.h5 (#242), so this goes beyond "
    "the prior synthetic://toy placeholder. Closed-loop physical task-success is "
    "DEFERRED, not claimed: it requires the unvendored stable-worldmodel suite "
    "(#96), since a recorded held-out dataset is open-loop and cannot apply "
    "arbitrary planner actions to recorded frames; and it requires a "
    "non-collapsing federated checkpoint (#244), since latent-MPC planning "
    "success would be uninformative on the published checkpoints whose global "
    "representation collapses over rounds. This is engineering and training "
    "evidence, NOT a cryptographic proof of honest participant computation, and "
    "does not claim paper-scale LeWorldModel performance or SO-100 robotics "
    "task success."
)


def _planner_budget() -> Phase3DownstreamPlannerBudget:
    return Phase3DownstreamPlannerBudget(
        planner="icem",
        horizon=16,
        planning_samples=512,
        planner_iterations=8,
        eval_episodes=20,
        action_dim=6,
        note=(
            "Non-toy latent-MPC budget a closed-loop run WOULD use over the SO-100 "
            "6-DoF action space; no planner is executed here because closed-loop "
            "task-success is blocked (see task_success blockers #96 and #244)."
        ),
    )


def _blockers() -> tuple[Phase3TaskSuccessBlocker, ...]:
    return (
        Phase3TaskSuccessBlocker(
            blocker_ref="#96",
            reason=(
                "Closed-loop physical task-success is blocked by the unvendored "
                "stable-worldmodel suite (#96, maintainer-gated): "
                "lensemble/eval/world.py::resolve_env raises EvaluationError for any "
                "stable-worldmodel:// id because the suite is not vendored. A recorded "
                "held-out dataset is open-loop (you cannot apply arbitrary planner "
                "actions to recorded frames), so a faithful physical task-success "
                "score requires the simulator."
            ),
        ),
        Phase3TaskSuccessBlocker(
            blocker_ref="#244",
            reason=(
                "Latent-MPC planning success would be uninformative on the published "
                "federated checkpoints because they collapse (#244): at the default "
                "outer-step with a random-init warm-start the federated global "
                "representation collapses over rounds (eff_rank -> 1 for the DP-off "
                "probes); the DP-on headline holds rank at ~36 but val_pred grows "
                "monotonically, so a closed-loop number on this checkpoint would not "
                "reflect a usable world model."
            ),
        ),
    )


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--consortium-run-report",
        type=Path,
        default=Path("docs/evidence/phase3_consortium_run_report.json"),
        help=(
            "Headline Phase 3 consortium run report whose final round carries the "
            "real held-out SO-100 latent metrics (effective_rank / val_pred)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/phase3_downstream_eval_report.json"),
        help="Where to write the generated Phase 3 downstream eval report.",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        default=None,
        help="Validate an existing downstream eval report instead of generating one.",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.validate is not None:
        report = load_phase3_downstream_eval_report(args.validate)
        blockers = ", ".join(f"{b.blocker_ref}" for b in report.task_success.blockers)
        print(
            f"validated {args.validate}: task_success.status="
            f"{report.task_success.status}, blockers=[{blockers}], "
            f"held_out effective_rank={report.held_out_latent_metrics.effective_rank}, "
            f"val_pred={report.held_out_latent_metrics.val_pred}"
        )
        return

    report = build_phase3_downstream_eval_report(
        args.consortium_run_report,
        checkpoint=Phase3DownstreamCheckpointRef(
            repo_id=_CHECKPOINT_REPO,
            revision=_CHECKPOINT_REVISION,
            checkpoint_hash=_CHECKPOINT_HASH,
            config_hash=_CONFIG_HASH,
        ),
        task_env_id=_TASK_ENV_ID,
        held_out_data_ref=_HELD_OUT_DATA_REF,
        planner_budget=_planner_budget(),
        blockers=_blockers(),
        claim_boundary=_CLAIM_BOUNDARY,
        source_report_uri=str(args.consortium_run_report),
        held_out_windows=_HELD_OUT_WINDOWS,
        window_steps=_WINDOW_STEPS,
    )
    path = write_phase3_downstream_eval_report(report, args.output)
    load_phase3_downstream_eval_report(path)
    print(
        f"wrote {path}: held_out effective_rank="
        f"{report.held_out_latent_metrics.effective_rank}, "
        f"val_pred={report.held_out_latent_metrics.val_pred}, "
        f"task_success={report.task_success.status}"
    )


if __name__ == "__main__":
    main()
