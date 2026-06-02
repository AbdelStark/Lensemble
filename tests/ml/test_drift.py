"""Frame-drift diagnostic — the headline measurement (RFC-0002 9 / RFC-0005 2). Issue #19.

Synthetically-rotated silos: each silo's probe embeddings are a base set rotated by a known Q_c.
frame_drift recovers the inter-frame angle within tolerance, reports ~0 for an unrotated pair,
distinguishes a diverged (naive) regime from a held-low (anchored) one, is bitwise-reproducible, and
refuses to run on a probe-hash mismatch (INV-PROBE-PIN).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from lensemble.data.probe import probe_content_hash
from lensemble.errors import LensembleErrorCode, ProbeError
from lensemble.gauge import FrameDriftReport, frame_drift

_D = 8


def _rot(angle_deg: float) -> torch.Tensor:
    """A single-plane rotation by ``angle_deg`` in the (0, 1) plane of R^d."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    q = torch.eye(_D)
    q[0, 0], q[0, 1], q[1, 0], q[1, 1] = c, -s, s, c
    return q


def _silos(angles: dict[str, float]) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    base = torch.randn(64, _D)
    return {pid: base @ _rot(deg).T for pid, deg in angles.items()}


def test_recovers_pairwise_angle_and_zero_for_unrotated(tol: object) -> None:
    angle_tol: float = tol.ANGLE_TOL_DEG  # type: ignore[attr-defined]
    report = frame_drift(_silos({"p0": 0.0, "p1": 10.0, "p2": 0.0}), round_index=3)
    assert report.round_index == 3
    by_pair = {(p.participant_a, p.participant_b): p for p in report.pairs}

    assert abs(by_pair[("p0", "p1")].rotation_angle_deg - 10.0) < angle_tol
    assert abs(by_pair[("p1", "p2")].rotation_angle_deg - 10.0) < angle_tol
    assert by_pair[("p0", "p2")].rotation_angle_deg < angle_tol  # unrotated pair ~ 0

    for pair in report.pairs:  # pure rotations -> alignment residual ~ 0
        assert pair.procrustes_residual < 1e-3


def test_naive_diverges_while_anchored_holds_low(tol: object) -> None:
    angle_tol: float = tol.ANGLE_TOL_DEG  # type: ignore[attr-defined]
    naive = frame_drift(_silos({"p0": 0.0, "p1": 30.0})).pairs[0].rotation_angle_deg
    anchored = frame_drift(_silos({"p0": 0.0, "p1": 2.0})).pairs[0].rotation_angle_deg
    assert naive > anchored
    assert naive > 10.0  # the naive (un-gauged) regime diverges
    assert anchored < 2.0 + angle_tol  # the anchored regime holds the angle low


def test_drift_from_global(tol: object) -> None:
    angle_tol: float = tol.ANGLE_TOL_DEG  # type: ignore[attr-defined]
    report = frame_drift(_silos({"p0": 0.0, "p1": 10.0, "global": 5.0}))
    assert set(report.drift_from_global) == {
        "p0",
        "p1",
    }  # "global" is not a participant pair
    assert abs(report.drift_from_global["p0"] - 5.0) < angle_tol  # p0 vs global(5)
    assert abs(report.drift_from_global["p1"] - 5.0) < angle_tol  # p1(10) vs global(5)


def test_report_is_bitwise_reproducible() -> None:
    silos = _silos({"p0": 0.0, "p1": 12.0, "p2": 7.0})
    assert frame_drift(silos) == frame_drift(silos)  # deterministic on the same inputs


def test_probe_hash_mismatch_raises(synthetic_probe: SimpleNamespace) -> None:
    silos = _silos({"p0": 0.0, "p1": 5.0})
    with pytest.raises(ProbeError) as exc:
        frame_drift(silos, probe=synthetic_probe, expected_probe_hash="00" * 32)
    assert exc.value.code == LensembleErrorCode.PROBE_INVALID
    assert exc.value.remediation


def test_probe_hash_is_recorded_on_match(synthetic_probe: SimpleNamespace) -> None:
    pinned = probe_content_hash(
        synthetic_probe.points, synthetic_probe.landmark_idx
    ).hex()
    report = frame_drift(
        _silos({"p0": 0.0, "p1": 5.0}),
        probe=synthetic_probe,
        expected_probe_hash=pinned,
    )
    assert isinstance(report, FrameDriftReport)
    assert report.probe_hash == pinned
