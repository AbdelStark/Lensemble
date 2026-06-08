"""The checked-in Phase 3 observability report is bound to the real HF Jobs run (#246).

`docs/evidence/phase3_observability_report.json` is regenerated from the real headline run report
(`phase3_consortium_run_report.json`) + the real eval/control report. It must carry the real run identity
and checkpoint hash, link the anchored-vs-naive gauge contrast, capture at least one induced dropout with
a quorum-preserving close, and leak no raw trajectory (the `phase3-observability-redaction-v1` contract).
"""

from __future__ import annotations

import json
from pathlib import Path

from lensemble.federation.phase3_observability import (
    load_phase3_observability_report,
)

_REPORT = Path("docs/evidence/phase3_observability_report.json")


def test_observability_report_binds_the_real_run() -> None:
    report = load_phase3_observability_report(_REPORT)

    assert report.consortium_id == "lensemble-phase3-consortium"
    assert report.run_id == "phase3-consortium-v1"
    # The real headline run's final global-model hash.
    assert report.checkpoint_hash == (
        "bb31c0922de639cb9220c4cc5fc35d79aec719eb6fcedb09159bdff8cfb8fd43"
    )
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
