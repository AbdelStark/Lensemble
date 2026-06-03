"""Layer-4 function-space distillation fallback (RFC-0002 §6 / RFC-0005 §6). Issue #20.

The top rung of the ablation ladder: aggregate participant *behaviors* on the public probe instead of
weights. Each participant emits ``f_c(P)`` — a common reference ``E_ref`` composed with a per-participant
gauge rotation ``Q_c`` (``f_c(P) = E_ref @ Q_c``). ``distill_consensus(..., align=True)`` Procrustes-aligns
every participant back onto a deterministic reference frame before the mean, so the consensus is
*gauge-invariant by construction*: it depends only on ``E_ref`` (up to the reference's own frame), never on
which ``Q_c`` each participant happened to draw. ``align=False`` (the plain mean of the rotated frames) is
the degraded baseline and visibly degrades. A global student then distills against the aligned consensus on
the probe and recovers it within tolerance. Pure function of public-probe outputs only — no private data
crosses (``INV-RESIDENCY`` not at stake; the probe is public).
"""

from __future__ import annotations

import pytest
import torch

from lensemble.errors import DegenerateProcrustes, LensembleErrorCode
from lensemble.gauge import distill_consensus, distill_to_consensus

_K, _D, _C = (
    24,
    4,
    5,
)  # k landmarks (k >> d so the frame is well-determined), d dims, C participants


def _random_rotation(d: int, gen: torch.Generator) -> torch.Tensor:
    """A proper rotation ``Q in SO(d)`` (Haar-distributed via QR with a determinant fix-up)."""
    a = torch.randn(d, d, generator=gen)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(
        0
    )  # fix the QR sign ambiguity -> O(d)
    if torch.det(q) < 0:  # force a proper rotation (det = +1), never a reflection
        q[:, -1] = -q[:, -1]
    return q


def _rotated_silos(
    ref: torch.Tensor, rotations: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Per-participant probe predictions ``f_c(P) = E_ref @ Q_c`` for a common reference ``E_ref``."""
    return {pid: ref @ q for pid, q in rotations.items()}


def _make_case(
    seed: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """A common reference ``E_ref`` and ``C`` silos rotated by known ``Q_c`` (one identity reference)."""
    gen = torch.Generator().manual_seed(seed)
    ref = torch.randn(_K, _D, generator=gen)
    pids = [f"p{i}" for i in range(_C)]
    rotations = {
        pids[0]: torch.eye(_D)
    }  # the first (sorted) id is the reference frame: Q = I
    for pid in pids[1:]:
        rotations[pid] = _random_rotation(_D, gen)
    return ref, rotations, _rotated_silos(ref, rotations)


def test_aligned_consensus_recovers_the_reference_frame() -> None:
    # The aligned consensus equals E_ref in the reference's (first sorted id's) own frame: the reference
    # carries Q = I, so the recovered frame is E_ref itself (every other silo aligns back onto it).
    ref, _rotations, silos = _make_case(seed=0)
    consensus = distill_consensus(silos, align=True)
    assert consensus.shape == (_K, _D)
    assert torch.allclose(consensus, ref, atol=1e-4, rtol=1e-4)


def test_aligned_consensus_is_invariant_to_the_gauge_rotations() -> None:
    # Gauge invariance by construction: permuting WHICH rotation each non-reference participant carries
    # must not change the aligned consensus beyond fp32 tolerance (the reference p0 stays the identity).
    ref, rotations, silos = _make_case(seed=1)
    consensus_a = distill_consensus(silos, align=True)

    non_ref = sorted(k for k in rotations if k != "p0")
    rolled = {"p0": rotations["p0"]}
    for i, pid in enumerate(
        non_ref
    ):  # cyclically permute which Q_c each non-reference id carries
        rolled[pid] = rotations[non_ref[(i + 1) % len(non_ref)]]
    consensus_b = distill_consensus(_rotated_silos(ref, rolled), align=True)

    assert torch.allclose(consensus_a, consensus_b, atol=1e-4, rtol=1e-4)


def test_misaligned_mean_visibly_degrades() -> None:
    # align=False is the plain mean of the ROTATED frames: averaging across gauges shrinks the embeddings
    # toward an inconsistent average, so its residual against the reference frame is materially larger
    # than the aligned consensus's (which sits ~on the reference). Assert a clear order-of-magnitude gap.
    ref, _rotations, silos = _make_case(seed=2)
    aligned = distill_consensus(silos, align=True)
    misaligned = distill_consensus(silos, align=False)

    aligned_residual = float(torch.linalg.norm(aligned - ref))
    misaligned_residual = float(torch.linalg.norm(misaligned - ref))
    assert (
        misaligned_residual > 10.0 * aligned_residual
    )  # the un-gauged mean is far from any one frame


def test_single_participant_consensus_is_that_participant() -> None:
    # One participant: the consensus is its own embeddings (nothing to align/average against).
    gen = torch.Generator().manual_seed(3)
    only = torch.randn(_K, _D, generator=gen)
    assert torch.equal(distill_consensus({"solo": only}, align=True), only)
    assert torch.equal(distill_consensus({"solo": only}, align=False), only)


def test_empty_predictions_rejected() -> None:
    with pytest.raises((ValueError, Exception)) as exc:
        distill_consensus({}, align=True)
    assert exc.value  # a clear error, not a silent empty/garbage consensus


def test_degenerate_alignment_surfaces_not_swallowed() -> None:
    # k < d underdetermines the frame -> the Procrustes SVD is rank-deficient. The error must SURFACE
    # (DegenerateProcrustes) rather than yield a silent garbage consensus.
    gen = torch.Generator().manual_seed(4)
    ref = torch.randn(2, _D, generator=gen)  # k=2 < d=4 -> M = T^T S is rank-deficient
    silos = {"p0": ref, "p1": ref @ _random_rotation(_D, gen)}
    with pytest.raises(DegenerateProcrustes) as exc:
        distill_consensus(silos, align=True)
    assert exc.value.code == LensembleErrorCode.PROCRUSTES_DEGENERATE


def test_student_distills_to_match_the_aligned_consensus() -> None:
    # The function-space distillation: a global student GD-fit against the aligned consensus on the probe
    # recovers it within tolerance (an L2 function-space distillation loss).
    _ref, _rotations, silos = _make_case(seed=5)
    consensus = distill_consensus(silos, align=True)
    student_pred = distill_to_consensus(consensus, steps=100, lr=0.4)
    assert student_pred.shape == consensus.shape
    assert torch.allclose(student_pred, consensus, atol=1e-3, rtol=1e-3)


def test_fp64_consensus_is_kept_and_reproducible() -> None:
    # fp64 inputs are kept (upcast policy of procrustes_align) and the consensus is bitwise-reproducible.
    gen = torch.Generator().manual_seed(6)
    ref = torch.randn(_K, _D, dtype=torch.float64, generator=gen)
    silos = {
        "p0": ref,
        "p1": ref @ _random_rotation(_D, gen).to(torch.float64),
    }
    c1 = distill_consensus(silos, align=True)
    c2 = distill_consensus(silos, align=True)
    assert c1.dtype == torch.float64
    assert torch.equal(c1, c2)  # deterministic on the same inputs (INV-AGG-DETERMINISM)
