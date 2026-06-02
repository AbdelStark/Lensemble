"""Closed-form Procrustes alignment correctness + degeneracy (RFC-0002 5 / 07 §2.3). Issue #15.

Q* recovers the rotation relating source to target; its residual is <= any small-angle candidate's
(it is the optimum), Q* is orthogonal with det = +1, and a near-rank-deficient M raises
DegenerateProcrustes rather than returning a NaN / non-orthogonal matrix.
"""

from __future__ import annotations

import math

import pytest
import torch

from lensemble.errors import DegenerateProcrustes, LensembleErrorCode
from lensemble.gauge import procrustes_align


def _rot_z(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )


def _rot_x(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float32
    )


def test_procrustes_matches_brute_force(tol: object) -> None:
    rtol_proc: float = tol.RTOL_PROC  # type: ignore[attr-defined]
    atol_ortho: float = tol.ATOL_ORTHO  # type: ignore[attr-defined]

    torch.manual_seed(0)
    source = torch.randn(64, 3)
    rotation = _rot_z(0.4) @ _rot_x(
        -0.25
    )  # the known rotation relating source -> target
    target = source @ rotation

    q_star, residual = procrustes_align(source, target)

    # Q* recovers the rotation and the residual is ~0 (target is an exact rotation of source)
    assert torch.allclose(q_star, rotation, atol=1e-4)
    assert residual < rtol_proc

    # Q* is a proper rotation: orthogonal (Q^T Q = I) with det = +1
    identity = torch.eye(3)
    assert torch.linalg.norm(q_star.T @ q_star - identity) < atol_ortho
    assert abs(float(torch.det(q_star)) - 1.0) < atol_ortho

    # residual is <= every small-angle brute-force candidate (Q* is the optimum)
    for axis_rot in (_rot_x, _rot_z):
        for delta in (-0.05, -0.01, 0.01, 0.05):
            candidate = axis_rot(delta) @ q_star
            cand_residual = float(torch.linalg.norm(source @ candidate - target))
            assert residual <= cand_residual + rtol_proc


def test_near_rank_deficient_raises_degenerate() -> None:
    # A source/target with a zeroed column makes M = T^T S rank-deficient (a zero singular value).
    source = torch.randn(32, 3)
    source[:, 2] = 0.0
    with pytest.raises(DegenerateProcrustes) as exc:
        procrustes_align(source, source)
    assert exc.value.code == LensembleErrorCode.PROCRUSTES_DEGENERATE
    assert exc.value.min_singular_value < exc.value.tol  # type: ignore[attr-defined]
    assert exc.value.remediation


def test_fp64_inputs_are_kept_and_reproducible() -> None:
    torch.manual_seed(1)
    source = torch.randn(40, 3, dtype=torch.float64)
    target = source @ _rot_z(0.2).to(torch.float64)
    q1, r1 = procrustes_align(source, target)
    q2, r2 = procrustes_align(source, target)
    assert q1.dtype == torch.float64
    assert (
        torch.equal(q1, q2) and r1 == r2
    )  # bitwise-reproducible (INV-AGG-DETERMINISM)


def test_shape_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        procrustes_align(torch.randn(4, 3), torch.randn(4, 2))
    with pytest.raises(ValueError):
        procrustes_align(torch.randn(4, 3, 2), torch.randn(4, 3, 2))  # not 2-D
