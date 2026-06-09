#!/usr/bin/env python
"""Assemble the Phase-3 MVP consolidated benchmark report (#266).

Fetches the three real HF Jobs run reports (anchored-federation M1 / naive-FedAvg / local-only) plus the
latent-space inference report (#265) and consolidates them into one benchmark JSON: the per-round
convergence series (effective_rank / val_pred / frame_drift_deg) for each control, the headline contrast,
the inference numbers, the pinned immutable HF revisions, and the honest boundaries. This is the public
"results story" the MVP milestone-4 publishes (RFC-0010).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

_HEADLINE = (
    "The M1 anchored federation prevents the naive-FedAvg gauge collapse, but this is now a corrected "
    "SO-100 gauge-only result: held-out magnitude collapse (~7.5e-6 latent variance; "
    "thoughts/collapse_fix_probe.py) and the central ceiling probe (thoughts/central_ceiling_probe.py) "
    "show the checkpoint is not useful downstream. skill_vs_identity is gameable, and effective_rank is "
    "scale-invariant."
)
_BOUNDARY = (
    "Corrected SO-100 gauge-only boundary: M1 prevents the naive-FedAvg gauge collapse, but the held-out "
    "representation has magnitude collapse (~7.5e-6 latent variance; thoughts/collapse_fix_probe.py) and "
    "the central ceiling probe (thoughts/central_ceiling_probe.py) shows the downstream ceiling is not "
    "cleared. skill_vs_identity is gameable; effective_rank is scale-invariant and blind to magnitude "
    "collapse; latent-MPC success_rate=0.0 is a negative result, not a near-static-video success story. "
    "The #259 SO-100 checkpoint is not a useful downstream world model; the dynamic-env RFC-0017 pivot is "
    "the ground-truth usefulness path. Relaxed-DP (DP-off) probe regime for the gauge measurement; "
    "secure-aggregation simulated. Consortium-engineering + from-scratch federated-training evidence, "
    "NOT a cryptographic honest-computation proof and NOT a paper-scale robotics claim."
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
        "schema_version": 1,
        "report": "phase3-mvp-benchmark",
        "epic": "#259",
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
    print(f"wrote {out}", flush=True)
    return report


if __name__ == "__main__":
    main()
