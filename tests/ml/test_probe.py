"""Public probe set, landmark targets, hash pinning (RFC-0004 3). Issue #24. INV-PROBE-PIN."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from typer.testing import CliRunner

from lensemble.cli import app
from lensemble.data.probe import build_probe, verify_probe_pin
from lensemble.errors import LensembleErrorCode, ProbeError
from lensemble.model.encoder import build_encoder, snapshot_reference

_runner = CliRunner()


def _f_ref():
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            d=4,
            num_frames=2,
            image_size=4,
            patch_size=2,
            tubelet=2,
            depth=2,
            num_heads=2,
        )
    )
    enc = build_encoder(cfg)  # d=4, num_tokens = (2//2)*(4//2)**2 = 4
    return enc, snapshot_reference(enc)


def _points(p: int = 6) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(p, 2, 3, 4, 4)


def test_verify_accepts_pinned_hash() -> None:
    _, f_ref = _f_ref()
    probe = build_probe(_points(), torch.arange(4), f_ref)  # k=4, d=4
    assert verify_probe_pin(probe, probe.content_hash) is None
    assert tuple(probe.landmark_targets.shape) == (4, 4, 4)  # (k, N, d)


def test_verify_rejects_mismatched_hash() -> None:
    _, f_ref = _f_ref()
    probe = build_probe(_points(), torch.arange(4), f_ref)
    with pytest.raises(ProbeError) as exc:
        verify_probe_pin(probe, b"\x00" * 32)
    assert exc.value.code == LensembleErrorCode.PROBE_INVALID
    assert exc.value.expected_hash == probe.content_hash  # type: ignore[attr-defined]


def test_under_coverage_k_lt_d_rejected() -> None:
    _, f_ref = _f_ref()
    probe = build_probe(_points(), torch.arange(2), f_ref)  # k=2 < d=4
    with pytest.raises(ProbeError) as exc:
        verify_probe_pin(probe, probe.content_hash)
    assert exc.value.num_landmarks == 2 and exc.value.d == 4  # type: ignore[attr-defined]


def test_targets_derive_only_from_f_ref() -> None:
    enc, f_ref = _f_ref()
    points = _points()
    probe = build_probe(points, torch.arange(4), f_ref)
    targets0 = probe.landmark_targets.clone()
    # a later "training step" mutates the live encoder; f_ref (and the probe) must be unaffected
    with torch.no_grad():
        enc.pos_embed.add_(1.0)
    later = snapshot_reference(enc)(points[torch.arange(4)]).tokens
    assert torch.equal(probe.landmark_targets, targets0)  # unchanged (INV-PROBE-PIN)
    assert not torch.allclose(
        later, targets0
    )  # the current encoder would give different targets


def test_cli_build_pin_verify(tmp_path: Path) -> None:
    points_path = tmp_path / "points.safetensors"
    save_file({"points": _points()}, str(points_path))
    out = tmp_path / "probe.safetensors"
    res = _runner.invoke(
        app,
        [
            "probe",
            "build",
            "--points",
            str(points_path),
            "--out",
            str(out),
            "--d",
            "4",
            "--num-frames",
            "2",
            "--image-size",
            "4",
            "--patch-size",
            "2",
            "--tubelet",
            "2",
            "--num-heads",
            "2",
            "--k",
            "4",
        ],
    )
    assert res.exit_code == 0, res.output
    assert '"d": 4' in res.output and out.exists()

    pin = _runner.invoke(app, ["probe", "pin", str(out)])
    assert pin.exit_code == 0
    content_hash_hex = pin.output.strip().splitlines()[0]

    ok = _runner.invoke(app, ["probe", "verify", str(out), "--hash", content_hash_hex])
    assert ok.exit_code == 0 and "ok" in ok.output

    bad = _runner.invoke(app, ["probe", "verify", str(out), "--hash", "00" * 32])
    assert bad.exit_code == 1 and "probe_invalid" in bad.output
