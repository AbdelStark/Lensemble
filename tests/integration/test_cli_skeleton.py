"""CLI skeleton: command tree, config-load + override, manifest emission, exit-code contract.

Exercises the 02-public-api §4 surface via Typer's ``CliRunner``: a valid ``--config``/override
composes a ``LensembleConfig`` and emits a manifest (path -> stdout, exit 0); an unknown override exits
1 with the ``CONFIG_INVALID`` code + remediation on stderr; a usage error exits 2; ``doctor`` exits
non-zero when a check fails and 0 when all pass; machine output is on stdout, human on stderr. Issue #5.

Placed under ``tests/integration`` (multi-module wiring: cli + config + filesystem) rather than the
``tests/cli`` path named in the issue, so the §8 CI gate (#66) collects it.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from lensemble.cli import app

runner = CliRunner()

# A harmless override that satisfies the cross-field rules (raising C keeps min-participants <= C).
_VALID_OVERRIDE = "federation.participant_count=8"


def test_help_lists_the_public_command_tree() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "train",
        "federate",
        "eval",
        "probe",
        "commit",
        "drift",
        "verify",
        "doctor",
    ):
        assert command in result.stdout, command


def test_valid_override_emits_manifest_and_exits_zero(tmp_path: Path) -> None:
    # `eval` with no --checkpoint is the config-validation + manifest-emission path (exit 0) — `train` now
    # runs the real single-site loop and needs a data_source (covered end-to-end in tests/e2e), so this
    # boundary test uses `eval` to exercise override-application + manifest emission without training (#167).
    result = runner.invoke(app, ["eval", "--run-dir", str(tmp_path), _VALID_OVERRIDE])
    assert result.exit_code == 0, result.stderr
    # machine output: the manifest path is the FIRST stdout line (a dry eval emits only the manifest path).
    manifest_path = Path(result.stdout.strip().splitlines()[0])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["command"] == "eval"
    assert (
        manifest["config"]["federation"]["participant_count"] == 8
    )  # override applied
    assert manifest["seed_derivation"]  # reproducibility tag recorded


def test_stdout_is_machine_readable_stderr_is_human(tmp_path: Path) -> None:
    # `eval` with no --checkpoint is the dry config-validation path (#167): stdout is exactly the manifest
    # path (machine), the human note goes to stderr.
    result = runner.invoke(app, ["eval", "--run-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert result.stdout.strip().endswith("run_manifest.json")
    assert "pass --checkpoint" in result.stderr  # the human note about the dry path
    assert "pass --checkpoint" not in result.stdout


def test_unknown_override_exits_one_with_code_and_remediation(tmp_path: Path) -> None:
    result = runner.invoke(app, ["train", "--run-dir", str(tmp_path), "no_such.key=1"])
    assert result.exit_code == 1
    assert "config_invalid" in result.stderr
    assert "remediation" in result.stderr
    assert result.stdout.strip() == ""  # no machine output when work never began


def test_missing_config_file_exits_one(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(tmp_path / "absent.yaml"),
            "--run-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "config_invalid" in result.stderr


def test_usage_error_exits_two() -> None:
    result = runner.invoke(app, ["train", "--definitely-not-an-option"])
    assert result.exit_code == 2


def test_doctor_passes_on_defaults() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["ok"] is True
    names = {c["name"] for c in report["checks"]}
    assert names == {
        "python",
        "torch",
        "warm_start",
        "determinism",
        "aggregation_determinism",
    }


def test_doctor_fails_when_determinism_flag_is_off() -> None:
    result = runner.invoke(
        app, ["doctor", "determinism.deterministic_aggregation=false"]
    )
    assert result.exit_code == 1
    report = json.loads(result.stdout)  # the report is still machine-readable on stdout
    assert report["ok"] is False
    det = next(c for c in report["checks"] if c["name"] == "determinism")
    assert det["ok"] is False


def test_verify_prove_is_phase_two_and_exits_nonzero() -> None:
    result = runner.invoke(app, ["verify", "prove"])
    assert result.exit_code == 1
    assert "Phase-2" in result.stderr
    assert "recompute" in result.stderr  # remediation points at the Phase-1 path


def test_federate_subcommands_emit_manifests(tmp_path: Path) -> None:
    # The federate commands now instantiate the REAL Coordinator/Participant and report readiness (#167):
    # stdout's FIRST line is the manifest path, the SECOND is the initial global hash / participant id.
    for sub, addr_flag in (
        ("coordinator", "--listen"),
        ("participant", "--coordinator"),
    ):
        result = runner.invoke(
            app, ["federate", sub, addr_flag, "in-process", "--run-dir", str(tmp_path)]
        )
        assert result.exit_code == 0, result.stderr
        lines = result.stdout.strip().splitlines()
        manifest = json.loads(Path(lines[0]).read_text())
        assert manifest["command"] == f"federate {sub}"
        # readiness line: the coordinator echoes a 64-hex global hash; the participant echoes its id.
        if sub == "coordinator":
            assert len(lines[1]) == 64 and all(
                c in "0123456789abcdef" for c in lines[1]
            )
        else:
            assert lines[1] == "participant-0"
