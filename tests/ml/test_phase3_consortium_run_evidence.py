"""The checked-in real HF Jobs Phase 3 consortium run report is a published 10-round result (#243).

Validates `docs/evidence/phase3_consortium_run_report.json` — the per-round report the headline anchored
run pushed to the checkpoint repo — against the Phase 3 long-run schema: ten closed rounds at non-toy
model size, a real `hf_jobs_release` publication, and per-round secure-aggregation + DP accounting plus
the four real learning metrics. No raw trajectory may appear in the report.
"""

from __future__ import annotations

import json
from pathlib import Path

from lensemble.federation import load_phase3_long_run_report

_REPORT = Path("docs/evidence/phase3_consortium_run_report.json")


def test_real_run_report_is_a_ten_round_release() -> None:
    report = load_phase3_long_run_report(_REPORT)

    assert report.consortium_id == "lensemble-phase3-consortium"
    assert report.run_id == "phase3-consortium-v1"
    assert report.closed_rounds == 10
    assert report.target_rounds == 10
    assert report.completed_target is True
    assert report.blockers == ()
    assert report.run_shape.model_latent_dim == 256  # non-toy
    assert report.run_shape.artifact_targets.publication_mode == "hf_jobs_release"
    assert len(report.participants) == 4
    assert all(
        p.submitted_rounds == 10 and p.dropped_rounds == 0 for p in report.participants
    )


def test_real_run_rounds_carry_metrics_and_accounting() -> None:
    report = load_phase3_long_run_report(_REPORT)

    assert len(report.rounds) == 10
    for round_summary in report.rounds:
        assert round_summary.state == "closed"
        # Secure-aggregation status + DP (ε) accounting recorded per round.
        assert round_summary.aggregation_backend_status == "secure_sum"
        assert round_summary.dp_epsilon_spent is not None
        # All four real learning metrics present.
        assert round_summary.val_pred is not None
        assert round_summary.val_sigreg is not None
        assert round_summary.effective_rank is not None
        assert round_summary.frame_drift_deg is not None
    # The frame anchor holds representation rank well above the collapse floor on every round.
    assert all(
        r.effective_rank is not None and r.effective_rank > 8.0 for r in report.rounds
    )


def test_real_run_report_is_residency_safe() -> None:
    raw = json.loads(_REPORT.read_text(encoding="utf-8"))

    def _keys(node: object):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from _keys(value)
        elif isinstance(node, list):
            for item in node:
                yield from _keys(item)

    leaked = {"obs", "observation", "observations", "actions", "trajectory", "pixels"}
    assert leaked.isdisjoint(set(_keys(raw)))
