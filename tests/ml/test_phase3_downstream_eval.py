"""Phase 3 downstream latent-MPC eval report contract (#245).

Asserts the checked-in report binds the published checkpoint, carries the REAL
held-out SO-100 latent metrics (not the synthetic 0.5/1.0 placeholders), records
a non-toy planner budget, documents both blockers (#96, #244), and stays
residency-safe.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lensemble.errors import ConfigError, SchemaVersionMismatch
from lensemble.eval import (
    PHASE3_DOWNSTREAM_REPORT_SCHEMA_VERSION,
    Phase3DownstreamEvalReport,
    load_phase3_downstream_eval_report,
    parse_phase3_downstream_eval_report,
)

_REPORT_PATH = Path("docs/evidence/phase3_downstream_eval_report.json")

_CHECKPOINT_REPO = "abdelstark/lensemble-phase3-consortium-checkpoint"
_CHECKPOINT_REVISION = "828e210cba4870b2be4ab573a5f0dd4ee30bae29"
_CHECKPOINT_HASH = "bb31c0922de639cb9220c4cc5fc35d79aec719eb6fcedb09159bdff8cfb8fd43"
_CONFIG_HASH = "27f2c77c9d47a7d053c01ab65f8d43aad79463b27d882f2d85ec28bc062cb2b2"

# The final round (index 9) held-out SO-100 latent metrics from the headline
# consortium run report.
_FINAL_EFFECTIVE_RANK = 35.79561996459961
_FINAL_VAL_PRED = 32476.08585357666


def _load() -> Phase3DownstreamEvalReport:
    return load_phase3_downstream_eval_report(_REPORT_PATH)


def test_checked_in_phase3_downstream_report_is_schema_valid() -> None:
    report = _load()
    assert report.schema_version == PHASE3_DOWNSTREAM_REPORT_SCHEMA_VERSION
    # Round-trips through the parser identically.
    assert parse_phase3_downstream_eval_report(report.model_dump(mode="json")) == report


def test_checked_in_report_binds_published_checkpoint() -> None:
    report = _load()
    assert report.checkpoint.repo_id == _CHECKPOINT_REPO
    assert report.checkpoint.repo_type == "model"
    assert report.checkpoint.revision == _CHECKPOINT_REVISION
    assert report.checkpoint.checkpoint_hash == _CHECKPOINT_HASH
    assert report.checkpoint.config_hash == _CONFIG_HASH


def test_report_carries_real_held_out_so100_latent_metrics_not_synthetic() -> None:
    report = _load()
    metrics = report.held_out_latent_metrics
    # Real final-round held-out SO-100 latent signal, beyond synthetic://toy.
    assert metrics.effective_rank == pytest.approx(_FINAL_EFFECTIVE_RANK)
    assert metrics.val_pred == pytest.approx(_FINAL_VAL_PRED)
    assert metrics.round_index == 9
    assert metrics.latent_dim == 256
    assert metrics.held_out_windows == 1216
    assert metrics.window_steps == 4
    # NOT the synthetic 0.5 / 1.0 placeholders.
    assert metrics.effective_rank != 0.5
    assert metrics.val_pred != 1.0
    assert metrics.effective_rank > 1.0
    # Measured on the disjoint held-out SO-100 split (silo4), not synthetic toy.
    assert "phase3-so100-silo4.h5" in metrics.measured_on
    assert report.held_out_data_ref.endswith("phase3-so100-silo4.h5")
    assert report.task_env_id == "so100-heldout://phase3-public-task-scale"
    assert report.task_env_id != "synthetic://toy"


def test_report_records_non_toy_planner_budget() -> None:
    report = _load()
    budget = report.planner_budget
    assert budget.planner == "icem"
    assert budget.horizon == 16
    assert budget.planning_samples == 512
    assert budget.planner_iterations == 8
    assert budget.eval_episodes == 20
    assert budget.action_dim == 6
    assert budget.executed is False
    # Demonstrably beyond the Phase 2 synthetic toy budget.
    assert budget.horizon > 2
    assert budget.planning_samples > 8
    assert budget.eval_episodes > 4
    assert budget.planner_iterations > 4


def test_report_documents_both_blockers_and_no_task_success_pass() -> None:
    report = _load()
    success = report.task_success
    assert success.status == "blocked"
    assert success.success_rate is None
    refs = {blocker.blocker_ref for blocker in success.blockers}
    assert {"#96", "#244"}.issubset(refs)
    by_ref = {blocker.blocker_ref: blocker.reason for blocker in success.blockers}
    # Blocker 1: unvendored stable-worldmodel suite (#96).
    assert "stable-worldmodel" in by_ref["#96"]
    assert "resolve_env" in by_ref["#96"]
    assert "open-loop" in by_ref["#96"]
    # Blocker 2: collapsing federated checkpoints (#244).
    assert "collapse" in by_ref["#244"]
    assert "val_pred" in by_ref["#244"]


def test_report_claim_boundary_distinguishes_from_cryptographic_proof() -> None:
    report = _load()
    boundary = report.claim_boundary
    assert "held-out SO-100 latent" in boundary
    assert "#96" in boundary
    assert "#244" in boundary
    assert "paper-scale" in boundary
    assert "cryptographic proof" in boundary


def test_report_is_residency_safe() -> None:
    report = _load()
    assert report.raw_data_in_report is False
    raw = json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
    # No raw observation/action array keys leak into the report (the report
    # carries only scalars, hashes, ids, and configuration counts).
    for forbidden_key in (
        "raw_actions",
        "raw_observations",
        "observations",
        "frames",
        "trajectory",
        "episode_data",
    ):
        assert forbidden_key not in raw
    # No private filesystem surfaces or secret-token markers in any string value.
    serialized = json.dumps(raw)
    for forbidden in ("hf_", "/Users/", "/private/", "/home/"):
        assert forbidden not in serialized


def test_parse_rejects_future_schema_first() -> None:
    with pytest.raises(SchemaVersionMismatch):
        parse_phase3_downstream_eval_report(
            {"schema_version": PHASE3_DOWNSTREAM_REPORT_SCHEMA_VERSION + 1}
        )


def test_parse_rejects_synthetic_toy_latent_placeholder() -> None:
    raw = json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
    raw["held_out_latent_metrics"]["effective_rank"] = 0.5
    with pytest.raises(ConfigError):
        parse_phase3_downstream_eval_report(raw)

    raw = json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
    raw["held_out_latent_metrics"]["val_pred"] = 1.0
    with pytest.raises(ConfigError):
        parse_phase3_downstream_eval_report(raw)


def test_parse_rejects_toy_planner_budget() -> None:
    raw = json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
    raw["planner_budget"]["horizon"] = 2
    raw["planner_budget"]["planning_samples"] = 8
    raw["planner_budget"]["eval_episodes"] = 4
    with pytest.raises(ConfigError):
        parse_phase3_downstream_eval_report(raw)


def test_parse_rejects_fabricated_task_success_pass() -> None:
    raw = json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
    raw["task_success"]["success_rate"] = 0.5
    with pytest.raises(ConfigError):
        parse_phase3_downstream_eval_report(raw)


def test_parse_rejects_dropping_a_required_blocker() -> None:
    raw = json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
    raw["task_success"]["blockers"] = [raw["task_success"]["blockers"][0]]
    with pytest.raises(ConfigError):
        parse_phase3_downstream_eval_report(raw)


def test_generator_script_generates_and_validates(tmp_path: Path) -> None:
    output = tmp_path / "phase3_downstream_eval_report.json"

    generate = subprocess.run(
        [
            sys.executable,
            "scripts/phase3_downstream_eval_report.py",
            "--consortium-run-report",
            "docs/evidence/phase3_consortium_run_report.json",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "wrote" in generate.stdout

    report = load_phase3_downstream_eval_report(output)
    assert report.held_out_latent_metrics.effective_rank == pytest.approx(
        _FINAL_EFFECTIVE_RANK
    )
    assert report.task_success.status == "blocked"

    validate = subprocess.run(
        [
            sys.executable,
            "scripts/phase3_downstream_eval_report.py",
            "--validate",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "validated" in validate.stdout
    assert "#96" in validate.stdout
    assert "#244" in validate.stdout
