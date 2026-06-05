"""Phase 2 final evidence bundle contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lensemble.artifacts.phase2_bundle import parse_phase2_evidence_bundle
from lensemble.errors import ConfigError, SchemaVersionMismatch


def test_checked_in_phase2_evidence_bundle_is_schema_valid() -> None:
    bundle_path = Path("docs/evidence/phase2_evidence_bundle.json")
    model_card_path = Path("docs/evidence/phase2_model_card.md")

    bundle = parse_phase2_evidence_bundle(json.loads(bundle_path.read_text()))

    assert bundle.raw_data_in_report is False
    assert all(check.exists for check in bundle.artifact_checks)
    assert bundle.training.final_global_hash == (
        "8f1494fd9e57b7496daf96e379a3de1457a435080b81b9e0ea1d20a52f4827c4"
    )
    assert bundle.baselines_curves.curve_point_count == 23
    assert bundle.baselines_curves.run_roles == (
        "anchored-federation",
        "naive-fedavg",
    )
    assert bundle.baselines_curves.blocked_comparisons == (
        "local-only",
        "centralized-pooled",
        "fork-a",
    )
    assert model_card_path.read_text() == bundle.model_card_markdown
    assert "Does not claim paper-scale" in bundle.model_card_markdown


def test_phase2_evidence_bundle_rejects_missing_artifact_check() -> None:
    raw = json.loads(Path("docs/evidence/phase2_evidence_bundle.json").read_text())
    raw["artifact_checks"][0]["exists"] = False
    raw["artifact_checks"][0]["error"] = "missing in test"

    with pytest.raises(ConfigError):
        parse_phase2_evidence_bundle(raw)


def test_parse_phase2_evidence_bundle_rejects_future_schema() -> None:
    with pytest.raises(SchemaVersionMismatch):
        parse_phase2_evidence_bundle({"schema_version": 2})
