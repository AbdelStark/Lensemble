"""Evidence bundle, demo card, and claim audit for the real-LeWM mode (#323, epic #314).

A clean real-mode run's evidence export must pass the audit; targeted mutations — missing
claim-boundary phrases, missing hashes, unbound revisions, raw participant data, dishonest
privacy status, missing health flags, unnegated overclaims — must each be rejected. The demo
card must repeat the Tapestry-like claim boundary and link the binding evidence artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lensemble.demo import (
    FederatedDemoError,
    audit_real_lewm_evidence,
    require_clean_evidence,
)
from lensemble.demo.federated import FederatedDemoService

from .test_lewm_federation import _delta_artifact, _real_run, _service, _submit


def _completed_real_evidence() -> dict[str, Any]:
    service = _service()
    state = _real_run(service)
    run = state["run"]
    p0, p1 = state["joins"]
    _submit(service, run, p0, _delta_artifact(run, p0, fill=2e-4, hash_suffix="aa"))
    result = _submit(
        service, run, p1, _delta_artifact(run, p1, fill=4e-4, hash_suffix="bb")
    )
    round2 = result["run"]
    _submit(
        service, round2, p0, _delta_artifact(round2, p0, fill=1e-4, hash_suffix="cc")
    )
    _submit(
        service, round2, p1, _delta_artifact(round2, p1, fill=1e-4, hash_suffix="dd")
    )
    return service.export_evidence(run["id"])


def test_clean_real_run_evidence_passes_the_audit() -> None:
    evidence = _completed_real_evidence()
    violations = audit_real_lewm_evidence(evidence)
    assert violations == []
    require_clean_evidence(evidence)  # must not raise


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (
            lambda e: e.__setitem__(
                "nonClaimText", "A great Tapestry-like bounded demo."
            ),
            "negate",
        ),
        (
            lambda e: e.__setitem__(
                "nonClaimText", e["claimBoundary"].replace("Tapestry-like", "fancy")
            ),
            "Tapestry-like",
        ),
        (
            lambda e: e["lewmBinding"]["checkpoint"].__setitem__("revision", "main"),
            "pinned 40-hex",
        ),
        (
            lambda e: e["lewmBinding"]["checkpoint"].__setitem__("weightsSha256", None),
            "weightsSha256",
        ),
        (
            lambda e: e["lewmBinding"].__setitem__("exportGraphHashes", {}),
            "exportGraphHashes",
        ),
        (lambda e: e["modelRevisions"][0].__setitem__("sha256", "short"), "sha256"),
        (
            lambda e: e["modelRevisions"][1].__setitem__(
                "parentModelRevisionId", "rev-unknown"
            ),
            "unbound parent",
        ),
        (
            lambda e: e["modelRevisions"][0]["baseCheckpoint"].__setitem__(
                "revision", "f" * 40
            ),
            "not bound to the pinned checkpoint",
        ),
        (lambda e: e["roundMetrics"][0].pop("healthFlags"), "health flags"),
        (
            lambda e: e["roundMetrics"][0].pop("sigregStatisticMean"),
            "sigregStatisticMean",
        ),
        (
            lambda e: e["privacy"].__setitem__(
                "secureAggregation", "secure aggregation protects all updates"
            ),
            "privacy.secureAggregation",
        ),
        (lambda e: e.__setitem__("leak", {"delta": [1.0, 2.0]}), "forbidden content"),
        (
            lambda e: e.__setitem__("note", "this is a benchmark win over local-only"),
            "overclaim",
        ),
        (lambda e: e.__setitem__("runMode", "surrogate-swipe-dot"), "runMode"),
    ],
)
def test_audit_rejects_violations(mutate, needle) -> None:
    evidence = _completed_real_evidence()
    mutate(evidence)
    violations = audit_real_lewm_evidence(evidence)
    assert violations, "mutation must be caught"
    assert any(needle in violation for violation in violations), violations
    with pytest.raises(FederatedDemoError, match="claim_audit_failed|"):
        require_clean_evidence(evidence)


def test_negated_overclaims_are_allowed() -> None:
    evidence = _completed_real_evidence()
    evidence["note"] = "this is not a benchmark win over local-only training"
    assert audit_real_lewm_evidence(evidence) == []


def test_surrogate_evidence_is_out_of_scope_for_this_audit() -> None:
    service = FederatedDemoService()
    run = service.create_run({"maxParticipants": 1, "quorum": 1, "rounds": 1})
    evidence = service.export_evidence(run["id"])
    violations = audit_real_lewm_evidence(evidence)
    assert any("runMode" in violation for violation in violations)


# ---------------------------------------------------------------------------
# the demo card
# ---------------------------------------------------------------------------

CARD = Path("docs/evidence/lewm_tworooms_demo_card.md")


def test_demo_card_repeats_the_claim_boundary() -> None:
    text = CARD.read_text(encoding="utf-8")
    assert text.count("Tapestry-like") >= 4  # "clearly and often"
    for needle in (
        "Tapestry-like browser-local federated adaptation run",
        "full from-scratch LeWorldModel base training",
        "production browser training",
        "paper-scale TwoRooms or PushT benchmark parity",
        "cryptographic proof of honest computation",
        "absent in this demo path",
        "surrogate-swipe-dot",
    ):
        assert needle in text, needle


def test_demo_card_links_every_gate_artifact() -> None:
    text = CARD.read_text(encoding="utf-8")
    for artifact in (
        "lewm_tworooms_checkpoint_manifest.json",
        "lewm_tworooms_reference_report.json",
        "lewm_tworooms_browser_export_manifest.json",
        "lewm_tworooms_action_stats.json",
        "lewm_tworooms_realdata_check.json",
        "lewm_tworooms_adapter_overfit.json",
        "lewm_tworooms_probe_check.json",
        # epic #332: the system-composed headline + seed-robustness artifacts
        "lewm_tworooms_system_probe.json",
        "lewm_tworooms_probe_seedsweep.json",
        "lewm_tworooms_surprise.json",
    ):
        assert artifact in text, artifact
        assert (Path("docs/evidence") / artifact).is_file(), artifact


def test_demo_card_numbers_match_the_committed_evidence() -> None:
    text = CARD.read_text(encoding="utf-8")
    probe = json.loads(Path("docs/evidence/lewm_tworooms_probe_check.json").read_text())
    assert probe["result"]["verdict"] == "improved"
    rel = probe["result"]["relativeImprovement"]
    assert f"+{rel * 100:.1f}%" in text
    realdata = json.loads(
        Path("docs/evidence/lewm_tworooms_realdata_check.json").read_text()
    )
    assert f"{realdata['modelPredictionMse']:.3f}" in text
    # the system-composed headline must match the committed evidence (it IS the headline now)
    system = json.loads(
        Path("docs/evidence/lewm_tworooms_system_probe.json").read_text()
    )
    assert system["role"] == "system-composed-headline"
    assert f"+{system['result']['relativeImprovement'] * 100:.1f}%" in text
    # the seed sweep's worst case is quoted (the headline cites the worst draw, not the best)
    sweep = json.loads(
        Path("docs/evidence/lewm_tworooms_probe_seedsweep.json").read_text()
    )
    worst = sweep["distribution"]["worstCaseRelativeImprovement"]
    assert f"+{worst * 100:.1f}%" in text
