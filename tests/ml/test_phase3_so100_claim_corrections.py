"""Regression guards for the corrected #259 SO-100 claim boundary (#288)."""

from __future__ import annotations

import json
from pathlib import Path

from lensemble.eval import load_phase3_downstream_eval_report

_DISCLOSURES = (
    "magnitude collapse",
    "~7.5e-6",
    "thoughts/collapse_fix_probe.py",
    "central ceiling",
    "thoughts/central_ceiling_probe.py",
    "skill_vs_identity is gameable",
    "effective_rank is scale-invariant",
)


def _text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _assert_disclosures(surface: str) -> None:
    normalized = " ".join(surface.split())
    missing = [phrase for phrase in _DISCLOSURES if phrase not in normalized]
    assert missing == []


def test_corrected_so100_model_card_readme_roadmap_and_nav() -> None:
    card = _text("docs/evidence/phase3_mvp_model_card.md")
    readme = _text("README.md")
    roadmap = _text("docs/roadmap/PHASE3.md")
    nav = _text("mkdocs.yml")

    for surface in (card, readme, roadmap):
        _assert_disclosures(surface)
        assert (
            "not a downstream-useful world model" in surface
            or "not downstream-useful" in surface
        )

    assert "The converged model is then **used**" not in readme
    assert "MVP Model Card (converged)" not in nav
    assert "MVP Model Card (corrected SO-100)" in nav


def test_corrected_so100_benchmark_json_and_generator() -> None:
    report = json.loads(_text("docs/evidence/phase3_mvp_benchmark_report.json"))
    generator = _text("scripts/phase3_mvp_benchmark.py")
    surface = f"{report['headline']}\n{report['honest_boundary']}\n{generator}"

    _assert_disclosures(surface)
    assert "gauge-only boundary" in report["honest_boundary"]
    assert "not a useful downstream world model" in report["honest_boundary"]


def test_corrected_so100_downstream_json_and_generator() -> None:
    report = load_phase3_downstream_eval_report(
        Path("docs/evidence/phase3_downstream_eval_report.json")
    )
    generator = _text("scripts/phase3_downstream_eval_report.py")
    surface = (
        report.claim_boundary
        + "\n"
        + report.held_out_latent_metrics.note
        + "\n"
        + "\n".join(blocker.reason for blocker in report.task_success.blockers)
        + "\n"
        + generator
    )

    _assert_disclosures(surface)
    assert "correction of the prior SO-100 overclaim" in report.claim_boundary
    assert report.task_success.success_rate is None


def test_corrected_so100_inference_json_and_generator() -> None:
    report = json.loads(_text("docs/evidence/phase3_inference_demo_report.json"))
    generator = _text("scripts/phase3_inference_demo.py")
    control_boundaries = "\n".join(
        control["metric_boundary"] for control in report["controls"]
    )
    surface = report["honest_boundary"] + "\n" + control_boundaries + "\n" + generator

    _assert_disclosures(surface)
    assert "success_rate=0.0 is a negative result" in surface
    assert "near-static-video success story" in surface
    assert all("metric_boundary" in control for control in report["controls"])
