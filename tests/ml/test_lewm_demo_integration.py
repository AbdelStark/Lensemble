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


def test_host_form_offers_both_run_modes() -> None:
    app = (WEB_DIR / "app.mjs").read_text(encoding="utf-8")
    assert 'value: "surrogate-swipe-dot"' in app
    assert 'value: "real-lewm-tworooms"' in app
    assert "mode: runModeSelect.value" in app


def test_learner_dispatch_has_no_silent_fallback() -> None:
    app = (WEB_DIR / "app.mjs").read_text(encoding="utf-8")
    assert 'run.runMode === "real-lewm-tworooms"' in app
    assert "runRealLewmRound" in app
    assert "no surrogate fallback" in app
    participant = (WEB_DIR / "lewm_participant.mjs").read_text(encoding="utf-8")
    assert "NO fallback to the surrogate learner" in participant
    # the simulator cannot host real-mode runs
    assert "needs the backend API mode" in app


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


def test_index_banner_keeps_claim_boundary() -> None:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert "real-lewm-tworooms" in html
    assert "not full from-scratch LeWM browser pretraining" in html
    # the original surrogate-scope banner stays intact
    assert "Simulated demo" in html


def test_participant_view_explains_real_mode_residency() -> None:
    app = (WEB_DIR / "app.mjs").read_text(encoding="utf-8")
    assert "never leave this browser" in app
    assert "fails visibly" in app
