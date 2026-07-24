"""The checked-in Phase 3 observability report is one coherent historical record.

The report is regenerated from the historical local lifecycle smoke and the
hash-bound eval/control report. It must carry that same smoke identity and
checkpoint hash, disclose its superseded evidence status, capture at least one
induced dropout with a quorum-preserving close, and leak no raw trajectory (the
``phase3-observability-redaction-v1`` contract).
"""

from __future__ import annotations

import json
from pathlib import Path

from lensemble.federation.phase3_observability import (
    load_phase3_observability_report,
)

_REPORT = Path("docs/evidence/phase3_observability_report.json")


def test_observability_report_binds_the_historical_smoke() -> None:
    report = load_phase3_observability_report(_REPORT)

    assert report.consortium_id == "lensemble-phase3-long-run-smoke"
    assert report.run_id == "phase3-long-run-smoke-v1"
    assert report.checkpoint_hash == (
        "ed3081ee514af142a226443f113a37c24d7d5872bfb707f11abe10893a0ad50d"
    )
    assert report.training_evidence_status == "historical_pre_correctness_fix"
    assert report.training_evidence_superseded_reason is not None
    assert "#335" in report.training_evidence_superseded_reason
    assert "do not validate the corrected runtime" in report.claim_boundary
    assert len(report.participants) == 4


def test_observability_captures_a_quorum_preserving_dropout() -> None:
    report = load_phase3_observability_report(_REPORT)

    induced = [d for d in report.dropout_decisions if d.induced]
    assert induced, "must capture at least one real induced-dropout decision"
    # Quorum (3 of 4) preserved so the round still closes.
    assert any(d.effective_quorum == 3 for d in induced)


def test_observability_report_is_residency_safe() -> None:
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
