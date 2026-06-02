"""Phase-2 prover/verifier seam (RFC-0006 §7; #64).

The Stage-D prover is reserved, not built: every `lensemble.verify.stark` entry point raises
`NotImplementedError` with a Stage-D remediation, the `lensemble verify prove` CLI maps that to a
non-zero exit, and none of these names are part of the frozen 1.0 public surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import lensemble.verify as verify
from lensemble.cli import app
from lensemble.verify import stark

runner = CliRunner()


def test_prove_outer_step_stub_raises_with_stage_d_message() -> None:
    with pytest.raises(NotImplementedError) as exc:
        stark.prove_outer_step(Path("prior"), Path("committed"), round_index=0)
    assert "Stage D" in str(exc.value)
    assert str(exc.value)  # non-empty


def test_verify_outer_step_proof_stub_raises() -> None:
    with pytest.raises(NotImplementedError) as exc:
        stark.verify_outer_step_proof(
            b"", Path("prior"), Path("committed"), round_index=0
        )
    assert "Stage D" in str(exc.value)


def test_prove_round_stub_raises() -> None:
    with pytest.raises(NotImplementedError) as exc:
        stark.prove_round(cfg=object())
    assert "Stage D" in str(exc.value)
    assert "recompute" in str(exc.value)  # points at the free Phase-1 path


def test_stark_is_outside_the_frozen_public_surface() -> None:
    # The prover seam is reached as lensemble.verify.stark.*, never re-exported from lensemble.verify.
    assert "stark" not in getattr(verify, "__all__", [])
    assert "prove_outer_step" not in getattr(verify, "__all__", [])


def test_cli_verify_prove_exits_nonzero_with_stage_d_message() -> None:
    result = runner.invoke(app, ["verify", "prove"])
    assert result.exit_code != 0
    assert "Stage D" in result.stderr
    assert "recompute" in result.stderr  # remediation points at the Phase-1 path
