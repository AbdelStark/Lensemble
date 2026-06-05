"""lensemble.gauge.anchor — Variant A (landmark) frame anchoring, the gauge fix (RFC-0002 4).

The latent gauge (the ``O(d)`` rotational freedom of the SIGReg-JEPA objective) is closed by *manufacturing*
the missing frame: pin ``k >= d`` generic public-probe landmarks to fixed absolute targets
``t_i = f_ref(p_i)`` taken only from the round-0 encoder. With ``k >= d`` generic landmarks the only
rotation satisfying all ``k`` constraints is ``Q = I`` — so the frame is pinned while every non-landmark
probe point and every private point stays free ("pin the frame, not the content").

``INV-WARMSTART-T0`` / ``INV-PROBE-PIN``: the targets derive only from ``f_ref`` (never a later ``f_t``)
and the probe is hash-pinned. Construction fails closed with ``FrameDriftExceeded`` when ``k < d`` (the
frame would be under-determined) and with ``ProbeError`` on a probe-hash mismatch. ``FrameAnchor.loss``
is the unweighted ``L_anchor`` the composite ``Objective`` injects (weighted by ``lambda_anc``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor

from lensemble.data.probe import probe_content_hash
from lensemble.errors import FrameDriftExceeded, LensembleErrorCode, ProbeError
from lensemble.gauge.procrustes import procrustes_align
from lensemble.model.numerics import module_input_tensor

if TYPE_CHECKING:
    from lensemble.model.objective import _EncoderLike  # structural encoder protocol


class FrameAnchor:
    """Variant A landmark frame anchor (RFC-0002 4). Constructed once per run from a pinned probe + f_ref.

    Args:
        probe: the hash-pinned public probe (carries ``points`` and ``landmark_idx``).
        ref_embeddings: the fixed landmark targets ``t_i = f_ref(p_i)``, shape ``(k, ...)`` aligned with
            ``probe.landmark_idx``; detached, never a gradient source (``INV-WARMSTART-T0``).
        variant: ``"landmark"`` (Variant B is a separate issue).
        probe_hash: the pinned probe content hash; the probe is re-hashed and compared at construction.
    """

    def __init__(
        self,
        probe: Any,
        ref_embeddings: Tensor,
        variant: Literal["landmark", "rotational"] = "landmark",
        *,
        probe_hash: str,
    ) -> None:
        if variant not in ("landmark", "rotational"):
            raise ValueError(
                f"FrameAnchor variant must be 'landmark' or 'rotational', got {variant!r}"
            )
        k = ref_embeddings.shape[0]
        d = ref_embeddings.shape[-1]
        if k < d:
            err = FrameDriftExceeded(
                f"landmark anchor under-determined: k={k} < d={d}; the frame is not pinned",
                code=LensembleErrorCode.FRAME_DRIFT_EXCEEDED,
                remediation="provide at least d generic landmarks (k >= d) to pin all O(d) gauge dofs",
            )
            err.k = k  # type: ignore[attr-defined]
            err.d = d  # type: ignore[attr-defined]
            raise err

        recomputed = probe_content_hash(probe.points, probe.landmark_idx).hex()
        if recomputed != probe_hash:
            raise ProbeError(
                "probe content hash does not match the pinned hash; refusing to anchor",
                code=LensembleErrorCode.PROBE_INVALID,
                remediation="anchor only against the pinned probe (INV-PROBE-PIN)",
            )

        self.probe = probe
        self.variant = variant
        self.probe_hash = recomputed
        # Fixed targets t_i = f_ref(p_i): detached so no gradient ever flows into f_ref (INV-WARMSTART-T0).
        self.targets: Tensor = ref_embeddings.detach().to(torch.float32)
        self._k = int(k)
        self._d = int(d)

    def loss(self, encoder: _EncoderLike) -> Tensor:
        """The unweighted ``L_anchor`` (0-dim fp32, RFC-0002 4), differentiable w.r.t. ``f_theta``.

        ``variant="landmark"`` (Variant A): ``(1/k) * sum_i ||f_theta(p_i) - t_i||^2`` — pull the encoder's
        frame on the ``k`` landmarks back onto the fixed round-0 targets; the only rotation that zeroes it
        (for ``k >= d`` generic landmarks) is the identity.

        ``variant="rotational"`` (Variant B): ``||Q* - I||_F^2`` where ``Q* = procrustes_align(f_theta(P),
        E_ref)`` — penalize *only* the Procrustes rotation, leaving the post-alignment content residual
        free, with gradients flowing through the differentiable SVD into ``f_theta``. A near-degenerate
        SVD raises :class:`~lensemble.errors.DegenerateProcrustes` rather than a NaN gradient.
        """
        landmarks = module_input_tensor(
            encoder, self.probe.points[self.probe.landmark_idx]
        )
        predicted = encoder(landmarks).tokens.to(torch.float32)
        if self.variant == "landmark":
            return (predicted - self.targets).pow(2).sum() / self._k
        # rotational (Variant B): penalize the recovered rotation only
        rotation, _ = procrustes_align(
            predicted.reshape(-1, self._d), self.targets.reshape(-1, self._d)
        )
        identity = torch.eye(self._d, dtype=rotation.dtype, device=rotation.device)
        return (rotation - identity).pow(2).sum()
