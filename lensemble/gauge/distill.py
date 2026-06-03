"""lensemble.gauge.distill — Layer-4 function-space distillation fallback (RFC-0002 §6 / RFC-0005 §6).

The top rung of the gauge ablation ladder. Instead of averaging *weights*, aggregate participant
*behaviors* on the pinned public probe: each participant ``c`` emits ``f_c(P)`` (its probe embeddings),
the coordinator forms a frame-aligned consensus, and a global student distills against that consensus.
This compares *functions on shared inputs, never weights*, so it is gauge-invariant by construction and
admits participants with different encoder sizes — at a higher per-round cost (an extra distillation pass
over the probe). It is held in reserve, enabled only if the Stage-B drift-vs-quality curves show
Variant A + the Layer-3 backstop insufficient at scale (RFC-0002 Migration / Rollout).

:func:`distill_consensus` builds the align-then-mean consensus target; :func:`distill_to_consensus` runs
the global-student distillation pass against it.

Gauge invariance (the load-bearing property). The SIGReg-JEPA objective leaves an ``O(d)`` rotational
gauge free, so a participant's probe embeddings are a common reference composed with a per-participant
rotation: ``f_c(P) = E_ref @ Q_c``. With ``align=True`` every participant is Procrustes-aligned back onto
one deterministic reference frame *before* the mean (Layer 3), so the consensus depends only on ``E_ref``
(up to the reference's own frame) and is invariant to which ``Q_c`` each participant drew. With
``align=False`` the plain mean of the misaligned frames averages across incompatible gauges and collapses
toward an inconsistent average — the degraded baseline the ablation ladder measures against.

Residency (``INV-RESIDENCY`` not at stake): this is a pure function of public-probe outputs only — no
raw observation, action, or private embedding crosses a trust boundary, because the probe is public
(04-error-model §5.3). Determinism (``INV-AGG-DETERMINISM``, conventions §9): inputs are upcast to
fp32 (fp64 inputs kept) on the alignment path exactly as :func:`~lensemble.gauge.procrustes.procrustes_align`
does, and a near-degenerate SVD raises :class:`~lensemble.errors.DegenerateProcrustes` (surfaced, never
swallowed) rather than emit a meaningless consensus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lensemble.errors import GaugeError, LensembleErrorCode
from lensemble.gauge.procrustes import procrustes_align

if TYPE_CHECKING:
    from collections.abc import Mapping


def distill_consensus(
    probe_predictions: Mapping[str, Tensor], *, align: bool = True
) -> Tensor:
    """Frame-aligned consensus target on the probe for global-student distillation (RFC-0002 §6).

    ``probe_predictions`` maps ``participant_id -> f_c(P)``, each ``(k, d)`` (the participant's embeddings
    on the pinned public probe). Returns the consensus embeddings ``(k, d)``.

    Align-then-mean contract. When ``align=True``, the participant first in sorted id order is the
    deterministic reference frame; every *other* participant ``c`` is Procrustes-aligned onto it via
    ``procrustes_align(f_c(P), f_ref(P)) -> Q_c*`` and mapped through ``f_c(P) @ Q_c*`` before the mean.
    Because each ``f_c(P) = E_ref @ Q_c`` for a common reference ``E_ref`` and a per-participant gauge
    rotation ``Q_c``, the aligned frames all coincide with the reference and the mean is **gauge-invariant**
    — it recovers ``E_ref`` in the reference's own frame, independent of which ``Q_c`` each participant
    drew. When ``align=False`` the result is the plain mean of the raw (misaligned) predictions: the
    degraded baseline the ablation ladder measures against (RFC-0005 §6).

    This is a pure function of public-probe outputs only — no private data crosses (``INV-RESIDENCY`` not
    at stake; the probe is public). Inputs are upcast to fp32 (fp64 kept) for determinism, matching
    :func:`~lensemble.gauge.procrustes.procrustes_align` (``INV-AGG-DETERMINISM``). A degenerate alignment
    surfaces :class:`~lensemble.errors.DegenerateProcrustes` (``PROCRUSTES_DEGENERATE``) rather than a
    silent garbage consensus. Requires at least one participant.
    """
    participants = sorted(probe_predictions)
    if not participants:
        raise GaugeError(
            "distill_consensus requires at least one participant's probe predictions",
            code=LensembleErrorCode.GAUGE_FAILED,
            remediation="pass a non-empty mapping participant_id -> f_c(P) (each (k, d))",
        )

    # Upcast for determinism exactly as procrustes_align does: fp64 inputs are kept, everything else
    # (fp32/bf16/fp16) becomes fp32 (INV-AGG-DETERMINISM, conventions §9).
    reference = probe_predictions[participants[0]]
    work = torch.float64 if reference.dtype == torch.float64 else torch.float32

    if not align:
        # The degraded baseline: the plain mean of the raw, misaligned frames (no Layer-3 alignment).
        stacked = torch.stack([probe_predictions[pid].to(work) for pid in participants])
        return stacked.mean(dim=0)

    # The reference frame is the first sorted id; it contributes unaltered (it aligns to itself with Q=I).
    ref = reference.to(work)
    aligned = [ref]
    for pid in participants[1:]:
        f_c = probe_predictions[pid].to(work)
        # source @ Q* best maps f_c(P) onto f_ref(P); a degenerate SVD raises DegenerateProcrustes here
        # and is deliberately NOT swallowed (the frame is under-determined, e.g. k < d).
        q_star, _residual = procrustes_align(f_c, ref)
        aligned.append(f_c @ q_star)

    return torch.stack(aligned).mean(dim=0)


def distill_to_consensus(
    consensus_target: Tensor, *, steps: int = 100, lr: float = 0.4
) -> Tensor:
    """Distill a global student against the frame-aligned consensus on the probe (RFC-0002 §6).

    The student is the simplest object the function-space distillation needs: a free learnable table of
    the student's probe predictions — a parameter of the same ``(k, d)`` shape as ``consensus_target`` —
    standing in for ``f_student(P)``. It is fit by plain gradient descent to minimize the L2 (squared
    Frobenius) function-space distillation loss ``||f_student(P) - consensus||_F^2`` on the probe, and the
    final predictions are returned. The objective is a convex quadratic whose optimum is the consensus
    itself, so the student matches it within tolerance; the free table keeps the distillation CPU-fast
    while exercising the real function-space objective (a small MLP over the probe would distill against
    the identical target — the consensus is the teacher's behavior on the public probe, not its weights).

    ``steps`` / ``lr`` size the descent (``lr < 0.5`` keeps the per-element fixed point ``s = t`` a stable
    contraction). The target is detached (the consensus is a fixed teacher) and the student starts from
    zero, so the run is deterministic in the target's working dtype (``INV-AGG-DETERMINISM``).
    """
    target = consensus_target.detach()
    student = torch.zeros_like(target, requires_grad=True)
    optimizer = torch.optim.SGD([student], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        # The L2 (squared Frobenius) function-space distillation loss ||f_student(P) - consensus||_F^2.
        loss = (student - target).pow(2).sum()
        loss.backward()
        optimizer.step()
    return student.detach()
