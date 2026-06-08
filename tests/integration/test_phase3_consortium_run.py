"""Phase 3 consortium run with real per-round learning metrics (#241).

These exercise the shared ``run_phase3_consortium`` driver that the HF Jobs entry point calls: the same
deterministic four-participant runtime as the long-run smoke, but with ``compute_metrics=True`` so each
closed round carries real ``val_pred``/``val_sigreg``/``effective_rank``/``frame_drift_deg`` measured
off the committed global checkpoints and the public probe. No raw trajectory may appear in the report.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from lensemble.eval.jepa_metrics import effective_rank
from lensemble.federation import (
    Phase3ArtifactTargets,
    run_phase3_consortium,
    run_phase3_long_run_smoke,
    to_phase3_long_run_report_json,
)
from lensemble.federation.phase3_orchestration import phase3_long_run_smoke_inputs

_GENERATED_AT = datetime(2026, 6, 8, 0, 0, 0, tzinfo=timezone.utc)


def test_consortium_smoke_emits_real_per_round_metrics(tmp_path: Path) -> None:
    report = run_phase3_long_run_smoke(
        run_dir=tmp_path / "run", rounds=3, compute_metrics=True
    )

    assert report.closed_rounds == 3
    assert report.completed_target is True
    for round_summary in report.rounds:
        # Every learning metric is measured (not the unset ``None`` of the metric-free smoke).
        assert round_summary.val_pred is not None
        assert round_summary.val_sigreg is not None
        assert round_summary.effective_rank is not None
        assert round_summary.frame_drift_deg is not None
        # effective_rank lives in [1, d] for d = latent_dim = 8; drift is a rotation angle in [0, 180].
        assert 0.0 <= round_summary.effective_rank <= 8.0 + 1e-6
        assert 0.0 <= round_summary.frame_drift_deg <= 180.0 + 1e-6
        assert round_summary.dp_epsilon_spent is not None


def test_consortium_metrics_are_deterministic(tmp_path: Path) -> None:
    first = run_phase3_long_run_smoke(
        run_dir=tmp_path / "a", rounds=2, compute_metrics=True
    )
    second = run_phase3_long_run_smoke(
        run_dir=tmp_path / "b", rounds=2, compute_metrics=True
    )

    metrics = lambda report: [  # noqa: E731 — terse local extractor for the comparison
        (r.val_pred, r.val_sigreg, r.effective_rank, r.frame_drift_deg)
        for r in report.rounds
    ]
    assert metrics(first) == metrics(second)


def test_consortium_release_targets_and_residency(tmp_path: Path) -> None:
    inputs = phase3_long_run_smoke_inputs(run_dir=tmp_path / "run", rounds=2)
    targets = Phase3ArtifactTargets(
        model_repo="hf://models/abdelstark/lensemble-phase3-consortium-checkpoint",
        dataset_repo="hf://datasets/abdelstark/lensemble-phase3-consortium-data",
        reports_prefix="reports/phase3/",
        publication_mode="hf_jobs_release",
    )
    report = run_phase3_consortium(
        inputs,
        run_dir=tmp_path / "run",
        rounds=2,
        generated_at=_GENERATED_AT,
        metric_windows=2,
        compute_metrics=True,
        artifact_targets=targets,
        eval_budget="real-run consortium eval budget",
        claim_boundary="real HF Jobs consortium run",
    )

    assert report.run_shape.artifact_targets.publication_mode == "hf_jobs_release"
    assert report.run_shape.eval_budget == "real-run consortium eval budget"
    # Residency: the serialized report must not leak raw observation/action arrays.
    payload = to_phase3_long_run_report_json(report)
    assert "obs" not in json.loads(payload)
    assert all(
        key not in payload for key in ("observation", "raw_action", "trajectory")
    )


def test_effective_rank_isotropic_vs_collapsed() -> None:
    gen = torch.Generator().manual_seed(0)
    isotropic = torch.randn(512, 8, generator=gen)
    collapsed = torch.randn(512, 1, generator=gen) * torch.ones(1, 8)

    assert effective_rank(isotropic) > 4.0
    assert effective_rank(collapsed) < 1.5
