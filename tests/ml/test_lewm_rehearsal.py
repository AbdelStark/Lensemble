"""End-to-end rehearsal gate for the real-LeWM adapter demo (#324, epic #314).

Runs the rehearsal script in-process: the two-participant smoke run (auto + manual), the
four-participant dropout/reconnect run with stale-round rejection, and a longer configurable
gate — each must complete, export evidence, and pass the fail-closed claim audit. The runbook
must exist, make the system-composed probe and seed sweep binding, retain the
frozen-checkpoint/non-privacy boundary, and block positive claims on a
non-improving probe.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

RUNBOOK = Path("docs/roadmap/TAPESTRY_LEWM_RUNBOOK.md")


def _load_rehearsal_module():
    spec = importlib.util.spec_from_file_location(
        "lewm_demo_rehearsal", Path("scripts/lewm_demo_rehearsal.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rehearsal_gate_completes_end_to_end() -> None:
    rehearsal = _load_rehearsal_module()
    from lensemble.demo import FederatedDemoService

    service = FederatedDemoService(
        public_base_url="https://rehearsal.example/web/federated-demo",
        deployment_target="rehearsal",
        transport_mode="websocket-primary",
        lewm_export_manifest=rehearsal._rehearsal_manifest(),
    )
    smoke = rehearsal._smoke_two_participants(service, 2)
    assert smoke["state"] == "completed"
    assert smoke["claimAuditViolations"] == 0
    assert smoke["finalRevision"].startswith("lewmrev-")

    four = rehearsal._four_with_dropout_and_reconnect(service)
    assert four["state"] == "completed"
    assert four["staleRoundRejected"] is True
    assert len(four["dropped"]) == 1
    assert four["claimAuditViolations"] == 0

    longer = rehearsal._smoke_two_participants(service, 8)
    assert longer["rounds"] == 8
    assert longer["state"] == "completed"


def test_rehearsal_script_runs_as_a_command() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/lewm_demo_rehearsal.py", "--rounds", "1"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["schema"] == "lewm-demo-rehearsal/1"
    assert report["smokeTwoParticipants"]["state"] == "completed"
    assert report["fourWithDropoutAndReconnect"]["state"] == "completed"


def test_runbook_carries_the_claim_boundary_and_gates() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    for needle in (
        "scripts/lewm_demo_rehearsal.py",
        "scripts/lewm_system_probe.py",
        "scripts/lewm_probe_seedsweep.py",
        "scripts/lewm_probe_check.py",
        "optional and non-binding",
        "Do not use it as the headline",
        "12,512-parameter (0.069%) residual",
        "frozen checkpoint",
        "One trusted local coordinator",
        "secure aggregation and differential privacy are not wired",
        "full-model federated-training quality",
        "production readiness",
        "paper-scale",
        "real_lewm_mode=unavailable",
        "do not switch the run to the surrogate mode",
    ):
        assert needle in normalized, needle
    for retired_event_phrase in (
        "Four-phone stage runbook",
        "presenter",
        "public-demo",
    ):
        assert retired_event_phrase not in text


def test_runbook_commands_reference_existing_files() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    for path in (
        "scripts/lewm_tworooms_ingest.py",
        "scripts/lewm_tworooms_export.py",
        "scripts/lewm_tworooms_realdata_check.py",
        "scripts/lewm_adapter_overfit_check.py",
        "scripts/lewm_probe_check.py",
        "scripts/lewm_system_probe.py",
        "scripts/lewm_probe_seedsweep.py",
        "scripts/lewm_demo_rehearsal.py",
        "tests/ml/test_lewm_evidence_audit.py",
    ):
        assert path in text, path
        assert Path(path).is_file(), path
