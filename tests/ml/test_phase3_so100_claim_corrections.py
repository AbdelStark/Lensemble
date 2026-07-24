"""Regression guards for the corrected #259 SO-100 claim boundary (#288)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from lensemble.eval import load_phase3_downstream_eval_report

_DISCLOSURES = (
    "magnitude collapse",
    "~7.5e-6",
    "central ceiling",
    "skill_vs_identity is gameable",
    "effective_rank is scale-invariant",
)


def _text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _assert_disclosures(surface: str) -> None:
    normalized = " ".join(surface.split())
    missing = [phrase for phrase in _DISCLOSURES if phrase not in normalized]
    assert missing == []


def test_historical_so100_model_card_readme_roadmap_and_nav() -> None:
    card = _text("docs/evidence/phase3_mvp_model_card.md")
    readme = _text("README.md")
    roadmap = _text("docs/roadmap/PHASE3.md")
    nav = _text("mkdocs.yml")

    _assert_disclosures(card)
    assert "not a downstream-useful world model" in card

    for summary_surface in (readme, roadmap):
        assert "#335" in summary_surface
        assert "historical" in summary_surface.casefold()
        assert "corrected" in summary_surface.casefold()
        assert "thoughts/" not in summary_surface
    assert "not downstream-useful" in roadmap
    assert "7.5e-6" in roadmap
    assert "central-ceiling" in roadmap

    assert "Historical Evidence Status" in card
    assert "do not validate the corrected runtime" in card
    assert "#335" in card
    assert "thoughts/" not in card
    assert "The converged model is then **used**" not in readme
    assert "MVP Model Card (converged)" not in nav
    assert "MVP Model Card (historical SO-100)" in nav


def test_historical_so100_benchmark_json_and_generator() -> None:
    report = json.loads(_text("docs/evidence/phase3_mvp_benchmark_report.json"))
    generator = _text("scripts/phase3_mvp_benchmark.py")
    surface = f"{report['headline']}\n{report['honest_boundary']}\n{generator}"

    _assert_disclosures(surface)
    assert report["schema_version"] == 2
    assert report["evidence_status"] == "historical_pre_correctness_fix"
    assert "#335" in report["superseded_reason"]
    assert "thoughts/" not in surface
    assert "gauge-only boundary" in report["honest_boundary"]
    assert "not a useful downstream world model" in report["honest_boundary"]


def test_historical_so100_downstream_json_and_generator() -> None:
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
    assert report.schema_version == 2
    assert report.evidence_status == "historical_pre_correctness_fix"
    assert "#335" in report.superseded_reason
    assert "do not validate the corrected runtime" in surface
    assert "thoughts/" not in surface


def test_historical_so100_inference_json_and_generator(tmp_path: Path) -> None:
    report = json.loads(_text("docs/evidence/phase3_inference_demo_report.json"))
    generator = _text("scripts/phase3_inference_demo.py")
    control_boundaries = "\n".join(
        control["metric_boundary"] for control in report["controls"]
    )
    surface = report["honest_boundary"] + "\n" + control_boundaries + "\n" + generator

    _assert_disclosures(surface)
    assert report["schema_version"] == 2
    assert report["evidence_status"] == "historical_pre_correctness_fix"
    assert "#335" in report["superseded_reason"]
    assert set(report["blocker_refs"]) == {"#96", "#244"}
    assert "do not validate the corrected runtime" in surface
    assert "thoughts/" not in surface
    assert "success_rate=0.0 is a negative result" in surface
    assert "near-static-video success story" in surface
    assert all("metric_boundary" in control for control in report["controls"])

    normalized = tmp_path / "phase3_inference_demo_report.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/phase3_inference_demo.py",
            "--normalize-historical",
            "docs/evidence/phase3_inference_demo_report.json",
            "--output",
            str(normalized),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    round_trip = json.loads(normalized.read_text(encoding="utf-8"))
    assert round_trip == report

    validated = subprocess.run(
        [
            sys.executable,
            "scripts/phase3_inference_demo.py",
            "--validate-historical",
            str(normalized),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "validated" in validated.stdout
