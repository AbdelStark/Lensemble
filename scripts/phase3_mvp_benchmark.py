#!/usr/bin/env python
"""Assemble the historical Phase-3 MVP benchmark record (#266).

Fetches the three real HF Jobs run reports (anchored-federation M1 / naive-FedAvg / local-only) plus the
latent-space inference report (#265) and consolidates them into one benchmark JSON: the per-round
convergence series (effective_rank / val_pred / frame_drift_deg) for each control, the headline contrast,
the inference numbers, the pinned immutable HF revisions, and the honest boundaries. These runs predate
the outer-update direction correction and are retained as historical audit evidence, not as validation of
the corrected runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

_EVIDENCE_STATUS = "historical_pre_correctness_fix"
_SUPERSEDED_REASON = (
    "These runs predate the correction of the outer-update direction and the "
    "public-probe-v2 target-binding contract. Their metrics are retained for audit "
    "history only and do not validate the corrected runtime; GitHub issue #335 "
    "tracks the replacement run."
)
_HEADLINE = (
    "Historical SO-100 gauge audit: under the superseded runtime, the M1 anchored "
    "configuration exhibited less latent-gauge collapse than naive FedAvg. The "
    "trajectory does not validate the corrected optimizer. The held-out audit also "
    "found magnitude collapse (~7.5e-6 latent variance) and no downstream-usefulness "
    "result; skill_vs_identity is gameable and effective_rank is scale-invariant."
)
_BOUNDARY = (
    "Historical SO-100 gauge-only boundary: these values were produced before the "
    "outer-update direction and public-probe-v2 target-binding fixes, so they are "
    "failure-analysis evidence rather than a current anchored-federation result. The "
    "held-out representation has magnitude collapse (~7.5e-6 latent variance), and "
    "the associated central ceiling audit did not establish downstream usefulness. "
    "skill_vs_identity is gameable; effective_rank is scale-invariant and blind to "
    "magnitude collapse; latent-MPC success_rate=0.0 is a negative result. The #259 "
    "SO-100 checkpoint is not a useful downstream world model. The run used a "
    "relaxed-DP (DP-off) probe regime and simulated secure aggregation. This is not "
    "a cryptographic honest-computation proof or a paper-scale robotics claim. "
    "GitHub issue #335 tracks the corrected rerun."
)


def _run_report(
    repo: str, *, candidates: tuple[str, ...]
) -> tuple[dict[str, Any], str]:
    api = HfApi()
    sha = api.model_info(repo).sha or "main"
    files = set(api.list_repo_files(repo, repo_type="model"))
    for name in candidates:
        if name in files:
            return json.load(
                open(hf_hub_download(repo, name, repo_type="model", revision=sha))
            ), sha
    raise FileNotFoundError(f"no run report among {candidates} in {repo}")


def _series(rows: list[dict[str, Any]], key: str) -> list[float | None]:
    out: list[float | None] = []
    for r in rows:
        v = r.get(key)
        out.append(None if v is None else round(float(v), 4))
    return out


def _federated_control(repo: str, role: str) -> dict[str, Any]:
    report, sha = _run_report(
        repo,
        candidates=(
            "phase3_long_run_smoke_report.json",
            "phase3_consortium_run_report.json",
        ),
    )
    rows = report.get("rounds", [])
    final = rows[-1] if rows else {}
    return {
        "control_role": role,
        "model_repo": repo,
        "revision": sha,
        "run_id": report.get("run_id"),
        "closed_rounds": report.get("closed_rounds"),
        "final_global_model_hash": report.get("final_global_model_hash"),
        "effective_rank_series": _series(rows, "effective_rank"),
        "val_pred_series": _series(rows, "val_pred"),
        "frame_drift_deg_series": _series(rows, "frame_drift_deg"),
        "final_effective_rank": final.get("effective_rank"),
        "final_val_pred": final.get("val_pred"),
        "final_frame_drift_deg": final.get("frame_drift_deg"),
    }


def _local_only_control(repo: str) -> dict[str, Any]:
    report, sha = _run_report(repo, candidates=("phase3_local_only_report.json",))
    per = report.get("per_participant", [])
    ranks = [p["effective_rank"] for p in per if p.get("effective_rank") is not None]
    preds = [p["val_pred"] for p in per if p.get("val_pred") is not None]
    return {
        "control_role": "local-only",
        "model_repo": repo,
        "revision": sha,
        "run_id": report.get("run_id"),
        "per_silo_effective_rank": ranks,
        "per_silo_val_pred": preds,
        "mean_effective_rank": round(sum(ranks) / len(ranks), 4) if ranks else None,
        "mean_val_pred": round(sum(preds) / len(preds), 6) if preds else None,
        "inter_silo_frame_drift_deg": report.get("frame_drift_deg"),
    }


def render_model_card(report: dict[str, Any]) -> str:
    """Render the producer-owned historical MVP model card."""

    controls = {
        str(control["control_role"]): control
        for control in report["convergence_controls"]
    }
    anchored = controls["anchored-federation"]
    naive = controls["naive-fedavg"]
    local = controls["local-only"]
    revisions = " · ".join(
        f"`{control['control_role']}` `{control['revision']}`"
        for control in report["convergence_controls"]
    )
    return f"""---
license: apache-2.0
library_name: lensemble
tags:
- federated-learning
- world-model
- jepa
- robotics
- phase3
- historical
---

# Lensemble MVP — Historical SO-100 Gauge Audit

## Historical Evidence Status

{report["superseded_reason"]}

This card preserves the immutable Phase 3 MVP measurements for provenance and
failure analysis. It is **not current correctness evidence** and must not be
used to claim that the corrected distributed training loop converges.

## Recorded Run Shape

- Four participant silos and one disjoint held-out SO-100 split
- From-scratch latent model (`latent_dim=256`, `depth=8`, `image_size=224`)
- Simulated secure aggregation
- Relaxed-DP (DP-off) gauge-probe regime

## Historical Metrics

| Control | Final effective rank | Final prediction loss | Final frame drift |
|---|---:|---:|---:|
| Anchored federation | {anchored["final_effective_rank"]} | {anchored["final_val_pred"]} | {anchored["final_frame_drift_deg"]} |
| Naive FedAvg | {naive["final_effective_rank"]} | {naive["final_val_pred"]} | {naive["final_frame_drift_deg"]} |
| Local only | {local["mean_effective_rank"]} (mean) | {local["mean_val_pred"]} (mean) | {local["inter_silo_frame_drift_deg"]} |

Pinned immutable revisions: {revisions}

These numbers describe the superseded implementation. Under that implementation,
the anchored configuration exhibited less gauge collapse than naive FedAvg, but
the sign error in the outer update prevents treating the comparison as evidence
for the corrected runtime.

## Claim Boundaries

- Held-out magnitude collapse was approximately `~7.5e-6` latent variance; the
  checked-in downstream and inference reports preserve that audit trail. The
  associated central ceiling diagnostic did not establish downstream
  usefulness.
- In plain terms: skill_vs_identity is gameable; effective_rank is
  scale-invariant. Therefore,
  neither is a binding usefulness metric.
- Latent-MPC `success_rate=0.0` is a negative result.
- The SO-100 checkpoint is not a downstream-useful world model.
- No closed-loop physical SO-100 task success, paper-scale LeWorldModel
  performance, effective differential privacy, secure-sum optimizer integration,
  or cryptographic honest-computation proof is claimed.
- [Issue #335](https://github.com/AbdelStark/Lensemble/issues/335) tracks a clean
  rerun using the corrected outer update and target-bound public probe.

Machine-readable source: `phase3_mvp_benchmark_report.json`.
"""


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(
        description="Phase-3 MVP consolidated benchmark report (#266)."
    )
    p.add_argument(
        "--anchored-repo", default="abdelstark/lensemble-phase3-converged-checkpoint"
    )
    p.add_argument("--naive-repo", default="abdelstark/lensemble-phase3-naive-control")
    p.add_argument(
        "--local-only-repo", default="abdelstark/lensemble-phase3-local-only-control"
    )
    p.add_argument(
        "--inference-report", default="docs/evidence/phase3_inference_demo_report.json"
    )
    p.add_argument("--output", default="docs/evidence/phase3_mvp_benchmark_report.json")
    p.add_argument(
        "--model-card-output",
        default="docs/evidence/phase3_mvp_model_card.md",
    )
    args = p.parse_args(argv)

    controls = [
        _federated_control(args.anchored_repo, "anchored-federation"),
        _federated_control(args.naive_repo, "naive-fedavg"),
        _local_only_control(args.local_only_repo),
    ]
    inference = None
    inf_path = Path(args.inference_report)
    if inf_path.exists():
        inference = json.loads(inf_path.read_text(encoding="utf-8"))

    report = {
        "schema_version": 2,
        "report": "phase3-mvp-benchmark",
        "epic": "#259",
        "evidence_status": _EVIDENCE_STATUS,
        "superseded_reason": _SUPERSEDED_REASON,
        "run_shape": {
            "from_scratch": True,
            "warm_start": None,
            "latent_dim": 256,
            "depth": 8,
            "predictor_depth": 6,
            "image_size": 224,
            "patch_size": 16,
            "num_tokens": 196,
            "participants": 4,
            "held_out_split": "phase3-so100-silo4.h5",
            "secure_aggregation_backend": "simulated",
            "dp_regime": "relaxed (DP-off probe regime for the gauge measurement)",
            "hardware": "HF Jobs a10g-large",
        },
        "headline": _HEADLINE,
        "convergence_controls": controls,
        "inference": inference,
        "honest_boundary": _BOUNDARY,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    model_card_out = Path(args.model_card_output)
    model_card_out.parent.mkdir(parents=True, exist_ok=True)
    model_card_out.write_text(render_model_card(report), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    print(f"wrote {model_card_out}", flush=True)
    return report


if __name__ == "__main__":
    main()
