"""Variant A landmark frame anchor pins the frame (RFC-0002 4 / 07 §2.2). Issue #16.

With k >= d generic landmarks, minimizing L_anchor over a free output correction drives the probe
Procrustes rotation back to the identity (< ANGLE_TOL_DEG); k < d is rejected at construction
(FrameDriftExceeded), and a probe-hash mismatch raises ProbeError.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from lensemble.contracts import WMCP_VERSION, LatentState
from lensemble.data.probe import probe_content_hash
from lensemble.errors import (
    DegenerateProcrustes,
    FrameDriftExceeded,
    LensembleErrorCode,
    ProbeError,
)
from lensemble.gauge import FrameAnchor, procrustes_align

_D, _N = 8, 2


class _RefEncoder(nn.Module):
    """A fixed tiny f_ref: maps probe points (k, d) -> a LatentState (k, N, d)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(_D, _N * _D)

    def forward(self, points: Tensor) -> LatentState:
        tokens = self.lin(points).reshape(points.shape[0], _N, _D)
        return LatentState(
            tokens=tokens, num_tokens=_N, dim=_D, wmcp_version=WMCP_VERSION
        )


class _CorrectedEncoder(nn.Module):
    """f_theta = f_ref with a learnable d x d output correction (init away from identity)."""

    def __init__(self, base: _RefEncoder, init: Tensor) -> None:
        super().__init__()
        self.base = base
        self.weight = nn.Parameter(init.clone())

    def forward(self, points: Tensor) -> LatentState:
        ls = self.base(points)
        return LatentState(
            tokens=ls.tokens @ self.weight.T,
            num_tokens=ls.num_tokens,
            dim=ls.dim,
            wmcp_version=ls.wmcp_version,
        )


def _rot15() -> Tensor:
    a = torch.tensor(0.2618)  # ~15 degrees, in the (0,1) plane
    c, s = torch.cos(a), torch.sin(a)
    q = torch.eye(_D)
    q[0, 0], q[0, 1], q[1, 0], q[1, 1] = c, -s, s, c
    return q


def _hash(probe: SimpleNamespace) -> str:
    return probe_content_hash(probe.points, probe.landmark_idx).hex()


def test_landmark_anchor_recovers_identity(
    synthetic_probe: SimpleNamespace, tol: object
) -> None:
    angle_tol: float = tol.ANGLE_TOL_DEG  # type: ignore[attr-defined]
    torch.manual_seed(0)
    f_ref = _RefEncoder().eval()
    landmarks = synthetic_probe.points[synthetic_probe.landmark_idx]
    targets = f_ref(
        landmarks
    ).tokens.detach()  # t_i = f_ref(p_i), the fixed round-0 targets

    anchor = FrameAnchor(
        synthetic_probe, targets, "landmark", probe_hash=_hash(synthetic_probe)
    )

    # f_theta starts rotated ~15 deg off the round-0 frame -> the anchor penalizes it
    f_theta = _CorrectedEncoder(f_ref, _rot15())
    initial_loss = float(anchor.loss(f_theta).detach())

    optimizer = torch.optim.Adam([f_theta.weight], lr=0.05)
    for _ in range(400):
        optimizer.zero_grad()
        loss = anchor.loss(f_theta)
        loss.backward()
        optimizer.step()

    final_loss = float(anchor.loss(f_theta).detach())
    assert (
        final_loss < initial_loss * 1e-2
    )  # the anchor pulls the frame back onto the targets

    with torch.no_grad():
        aligned = f_theta(landmarks).tokens.reshape(-1, _D)
    rotation, _ = procrustes_align(aligned, targets.reshape(-1, _D))
    d = rotation.shape[-1]
    cos = max(-1.0, min(1.0, (float(torch.trace(rotation)) - (d - 2)) / 2.0))
    angle_deg = torch.rad2deg(torch.arccos(torch.tensor(cos)))
    assert float(angle_deg) < angle_tol  # the probe rotation is driven back to identity


def test_loss_is_zero_dim_fp32(synthetic_probe: SimpleNamespace) -> None:
    f_ref = _RefEncoder().eval()
    targets = f_ref(
        synthetic_probe.points[synthetic_probe.landmark_idx]
    ).tokens.detach()
    anchor = FrameAnchor(synthetic_probe, targets, probe_hash=_hash(synthetic_probe))
    value = anchor.loss(f_ref)
    assert value.ndim == 0 and value.dtype == torch.float32


def test_loss_accepts_bfloat16_probe_points(synthetic_probe: SimpleNamespace) -> None:
    f_ref = _RefEncoder().eval()
    probe = SimpleNamespace(
        points=synthetic_probe.points.to(torch.bfloat16),
        landmark_idx=synthetic_probe.landmark_idx,
    )
    targets = f_ref(probe.points[probe.landmark_idx].to(torch.float32)).tokens.detach()
    anchor = FrameAnchor(probe, targets, probe_hash=_hash(probe))

    value = anchor.loss(f_ref)

    assert value.ndim == 0 and value.dtype == torch.float32


def test_k_less_than_d_is_rejected() -> None:
    # only 3 landmarks for d=8 -> the frame is under-determined -> fail closed at construction
    probe = SimpleNamespace(points=torch.randn(16, _D), landmark_idx=torch.arange(3))
    targets = torch.randn(3, _N, _D)
    with pytest.raises(FrameDriftExceeded) as exc:
        FrameAnchor(probe, targets, probe_hash=_hash(probe))
    assert exc.value.code == LensembleErrorCode.FRAME_DRIFT_EXCEEDED


def test_probe_hash_mismatch_is_rejected(synthetic_probe: SimpleNamespace) -> None:
    f_ref = _RefEncoder().eval()
    targets = f_ref(
        synthetic_probe.points[synthetic_probe.landmark_idx]
    ).tokens.detach()
    with pytest.raises(ProbeError) as exc:
        FrameAnchor(synthetic_probe, targets, probe_hash="00" * 32)
    assert exc.value.code == LensembleErrorCode.PROBE_INVALID


# --- Variant B rotational-drift anchor (RFC-0002 4), issue #17 ---


class _RankDeficientEncoder(nn.Module):
    """An encoder whose output zeroes the last latent dim -> a rank-deficient Procrustes M."""

    def forward(self, points: Tensor) -> LatentState:
        tokens = torch.randn(points.shape[0], _N, _D)
        tokens[..., -1] = 0.0
        return LatentState(
            tokens=tokens, num_tokens=_N, dim=_D, wmcp_version=WMCP_VERSION
        )


def test_rotational_anchor_recovers_identity(
    synthetic_probe: SimpleNamespace, tol: object
) -> None:
    angle_tol: float = tol.ANGLE_TOL_DEG  # type: ignore[attr-defined]
    torch.manual_seed(0)
    f_ref = _RefEncoder().eval()
    landmarks = synthetic_probe.points[synthetic_probe.landmark_idx]
    targets = f_ref(landmarks).tokens.detach()

    anchor = FrameAnchor(
        synthetic_probe, targets, "rotational", probe_hash=_hash(synthetic_probe)
    )
    f_theta = _CorrectedEncoder(f_ref, _rot15())
    initial_loss = float(
        anchor.loss(f_theta).detach()
    )  # ||Q* - I||_F^2 > 0 while drifted

    optimizer = torch.optim.Adam([f_theta.weight], lr=0.05)
    for _ in range(400):
        optimizer.zero_grad()
        loss = anchor.loss(f_theta)  # gradients flow through the differentiable SVD
        loss.backward()
        optimizer.step()

    final_loss = float(anchor.loss(f_theta).detach())
    assert final_loss < initial_loss * 1e-2  # the rotation penalty drives Q* toward I

    with torch.no_grad():
        aligned = f_theta(landmarks).tokens.reshape(-1, _D)
    rotation, _ = procrustes_align(aligned, targets.reshape(-1, _D))
    cos = max(-1.0, min(1.0, (float(torch.trace(rotation)) - (_D - 2)) / 2.0))
    angle_deg = torch.rad2deg(torch.arccos(torch.tensor(cos)))
    assert float(angle_deg) < angle_tol


def test_rotational_near_degenerate_raises(synthetic_probe: SimpleNamespace) -> None:
    f_ref = _RefEncoder().eval()
    targets = f_ref(
        synthetic_probe.points[synthetic_probe.landmark_idx]
    ).tokens.detach()
    anchor = FrameAnchor(
        synthetic_probe, targets, "rotational", probe_hash=_hash(synthetic_probe)
    )
    # a rank-deficient encoder output makes M near-degenerate -> raise, never a NaN gradient
    with pytest.raises(DegenerateProcrustes):
        anchor.loss(_RankDeficientEncoder())
