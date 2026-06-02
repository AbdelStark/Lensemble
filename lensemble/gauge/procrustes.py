"""lensemble.gauge.procrustes — closed-form orthogonal Procrustes alignment (RFC-0002 5).

:func:`procrustes_align` solves ``Q* = argmin_{Q in SO(d)} ||S Q - T||_F`` via the SVD of
``M = T^T S = U Sigma V^T``, returning ``Q* = V U^T`` with a Kabsch sign-correction so ``det Q* = +1``
— a frame re-alignment is a *proper* rotation, never a reflection — and the Frobenius residual
``||S Q* - T||_F``.

Determinism (``INV-AGG-DETERMINISM``, conventions 9): bf16/fp16 inputs are upcast to fp32 (fp64 inputs
are kept) before the SVD and the reduction order is fixed, so the result is bitwise-reproducible on the
aggregation path. This is the primitive ``recompute_alignment`` re-runs publicly (RFC-0006 4), so its
determinism on the fp32/fp64 path is load-bearing for the Phase-2 verifiability story.

A near-degenerate ``M`` (smallest singular value below the floor) raises :class:`DegenerateProcrustes`
rather than return a NaN / non-orthogonal matrix.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lensemble.errors import DegenerateProcrustes, LensembleErrorCode

# Matches config ``gauge.procrustes_singular_floor`` (conventions 9); the frame is under-determined below.
_DEFAULT_SINGULAR_FLOOR = 1e-6


def procrustes_align(
    source: Tensor, target: Tensor, *, singular_floor: float = _DEFAULT_SINGULAR_FLOOR
) -> tuple[Tensor, float]:
    """Closed-form proper-rotation Procrustes: ``(source, target) -> (Q*, residual)`` (RFC-0002 5).

    Finds the rotation ``Q* in SO(d)`` that best maps ``source`` onto ``target`` and the Frobenius
    residual ``||source @ Q* - target||_F``. ``source``/``target`` are ``(n, d)``; ``Q*`` is ``(d, d)``.
    Inputs are upcast to fp32 (fp64 kept) before the SVD for determinism. Raises
    :class:`~lensemble.errors.DegenerateProcrustes` (``PROCRUSTES_DEGENERATE``) when ``M = T^T S`` is
    near rank-deficient (the error carries ``min_singular_value``, ``condition_number``, ``tol``).
    """
    if source.shape != target.shape:
        raise ValueError(
            f"source {tuple(source.shape)} and target {tuple(target.shape)} must have equal shape"
        )
    if source.ndim != 2:
        raise ValueError(f"source/target must be 2-D (n, d), got rank {source.ndim}")

    work = torch.float64 if source.dtype == torch.float64 else torch.float32
    s = source.to(work)
    t = target.to(work)

    m = t.transpose(-2, -1) @ s  # (d, d) = T^T S
    u, sigma, vh = torch.linalg.svd(m)

    # The singular-value diagnostics are reported scalars, not graph nodes (detach so the differentiable
    # path — used by the Variant B rotational anchor — does not warn on float() of a grad tensor).
    min_sv = float(sigma.detach().min())
    max_sv = float(sigma.detach().max())
    if min_sv < singular_floor:
        condition_number = float("inf") if min_sv == 0.0 else max_sv / min_sv
        err = DegenerateProcrustes(
            f"Procrustes SVD near-degenerate: min singular value {min_sv:.3e} "
            f"below floor {singular_floor:.3e}",
            code=LensembleErrorCode.PROCRUSTES_DEGENERATE,
            remediation="increase landmark coverage (k >> d) or condition the inputs; "
            "the latent frame is under-determined",
        )
        err.min_singular_value = min_sv  # type: ignore[attr-defined]
        err.condition_number = condition_number  # type: ignore[attr-defined]
        err.tol = float(singular_floor)  # type: ignore[attr-defined]
        raise err

    v = vh.transpose(-2, -1)
    q = v @ u.transpose(-2, -1)
    if (
        torch.det(q) < 0
    ):  # Kabsch: flip the reflected axis to force a proper rotation (det = +1)
        v = v.clone()
        v[:, -1] = -v[:, -1]
        q = v @ u.transpose(-2, -1)

    residual = float(torch.linalg.norm(s @ q - t).detach())
    return q, residual
