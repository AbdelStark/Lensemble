"""Real-lewm mode integration into the federated demo flow (#321, epic #314).

The autonomous participant round, runtime caching, mode dispatch, and explicit-failure paths are
node-tested in ``web/federated-demo/lewm_adapter_selftest.mjs`` (driven by
``tests/ml/test_lewm_adapter.py``); the coordinator-side mode behavior is covered by
``tests/ml/test_lewm_federation.py``. This file pins the app wiring: mode selection on the host
form, the no-silent-fallback contract in the learner dispatch, mode-aware dashboard metrics,
claim-boundary banners, and the no-artificial-sleeps rule for the real path.
"""

from __future__ import annotations

from pathlib import Path

WEB_DIR = Path("web/federated-demo")


def test_host_form_creates_real_mode_runs_only() -> None:
    """The UI exposes only the Tapestry-like real path; the surrogate stays API/test-only."""
    app = (WEB_DIR / "app.mjs").read_text(encoding="utf-8")
    assert 'mode: "real-lewm-tworooms"' in app  # hardcoded in the create config
    assert "runModeSelect" not in app  # no learner-path selector
    assert "frontend-simulator" not in app.split("function renderHome")[1].split("function ")[0]


def test_learner_dispatch_has_no_silent_fallback() -> None:
    app = (WEB_DIR / "app.mjs").read_text(encoding="utf-8")
    assert 'run.runMode === "real-lewm-tworooms"' in app
    assert "runRealLewmRound" in app
    participant = (WEB_DIR / "lewm_participant.mjs").read_text(encoding="utf-8")
    assert "NO fallback to the surrogate learner" in participant


def test_real_round_driver_has_no_artificial_sleeps() -> None:
    participant = (WEB_DIR / "lewm_participant.mjs").read_text(encoding="utf-8")
    assert "setTimeout" not in participant
    assert "setInterval" not in participant


def test_dashboard_renders_real_mode_metrics() -> None:
    app = (WEB_DIR / "app.mjs").read_text(encoding="utf-8")
    for needle in (
        "predLossLastMean",
        "sigregStatisticMean",
        "effectiveRankMean",
        "adapterStateNorm",
        "lossDecreasedCount",
    ):
        assert needle in app, needle


def test_index_links_the_claim_boundary() -> None:
    """The UI carries product copy; the full claim boundary lives in the linked demo card and in
    every evidence export (gated by lensemble.demo.evidence_audit)."""
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert "lewm_tworooms_demo_card.md" in html
    assert "claim boundary" in html
    assert "rollouts never leave the browser" in html
    # the old disclaimer banners are gone by design
    assert "Simulated demo" not in html


def test_participant_view_explains_real_mode_residency() -> None:
    app = (WEB_DIR / "app.mjs").read_text(encoding="utf-8")
    assert "never leave this browser" in app
    assert "fails visibly" in app
