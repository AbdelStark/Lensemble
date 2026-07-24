"""lensemble.gauge.backstop — Layer-3 Procrustes re-alignment backstop at aggregation (RFC-0002 §5).

Immediately before the outer step, for each participant whose latent frame drift on the public probe
exceeds the configured threshold, recompute the hard alignment ``Q_c* = procrustes_align(f_c(P), E_ref)``
where row-space embeddings satisfy ``f_c(P) @ Q_c* ~= E_ref``. Reconstruct the participant's full local
gauge-bearing weights from ``W_local = W_global + Delta``, transform those weights into the reference
frame, then return ``Delta_aligned = W_aligned - W_global``. Transforming the delta alone is incorrect for
a nonzero global model because the coordinate change is affine in ``Delta`` at fixed ``W_global``.

PyTorch ``Linear`` uses row activations ``y = x @ W.T + b``. Therefore the row-space ``Q_c*`` returned by
Procrustes induces these full-local-weight transforms (#262):

- encoder ``frame_proj.weight``: ``W_aligned = Q.T @ W_local``;
- predictor ``in_proj.weight``: ``W_in_aligned = W_in_local @ Q``;
- predictor ``out_proj.weight``: ``W_out_aligned = Q.T @ W_out_local``;
- predictor ``out_proj.bias``: ``b_out_aligned = Q.T @ b_out_local``.

Every OTHER encoder param (``patch_embed``/``pos_embed``/``blocks``/``norm``) and every other predictor param
is returned **byte-identical**. The current global weights are public coordinator state, so this
reconstruct-transform-re-difference operation remains deterministic and publicly recomputable
(``INV-AGG-DETERMINISM``; RFC-0006 §3 ``recompute_alignment``).

Determinism / dtype (``INV-AGG-DETERMINISM``, conventions §9): the rotation is computed in fp32 (fp64 kept)
exactly like :func:`~lensemble.gauge.procrustes.procrustes_align`, then cast back to the delta's dtype, so a
re-run with identical inputs is byte-for-byte reproducible. The backstop is order-independent — each
participant's delta is realigned from its own ``Q_c*`` alone.

Degeneracy (RFC-0002 §5). When ``procrustes_align`` raises :class:`~lensemble.errors.DegenerateProcrustes`
the backstop clamp-and-retries ONCE with a relaxed singular floor; if it still degenerates, the participant's
backstop is SKIPPED — its UNALIGNED delta survives into the reduction — and ``gauge/procrustes_residual`` is
logged at WARN (RFC-0015). The round is never aborted by the backstop: a drift above threshold is handled
in-round (realign, or skip-and-warn), not raised.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from lensemble.gauge.drift import _rotation_angle_deg
from lensemble.gauge.procrustes import _DEFAULT_SINGULAR_FLOOR, procrustes_align

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

__all__ = [
    "realign_predictor_delta",
    "realign_encoder_frame_delta",
    "procrustes_backstop",
]

_log = logging.getLogger(__name__)

# The four gauge-bearing params. Q maps LOCAL row-space latents to the reference frame (`z_local @ Q`);
# each comment gives the transform of the reconstructed full local weight, not of the delta alone.
_IN_PROJ_WEIGHT = "predictor.in_proj.weight"  # (width, d): W_local @ Q
_OUT_PROJ_WEIGHT = "predictor.out_proj.weight"  # (d, width): Q.T @ W_local
_OUT_PROJ_BIAS = "predictor.out_proj.bias"  # (d,): Q.T @ b_local
_FRAME_PROJ_WEIGHT = "encoder.frame_proj.weight"  # (d, d): Q.T @ W_local

# The clamp-and-retry singular floor for the degenerate path: one relaxed retry before skipping the backstop
# for that participant (RFC-0002 §5 "the caller clamps/conditions and re-tries, or skips ... and logs").
_RELAXED_FLOOR_FACTOR = 1e-3


def realign_predictor_delta(
    predictor_delta: Mapping[str, Tensor],
    q_star: Tensor,
    *,
    global_weights: Mapping[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    """Align reconstructed predictor I/O weights by row-space ``Q*`` and re-difference (RFC-0002 §5).

    For each gauge-bearing parameter, reconstructs ``W_local = W_global + Delta``, applies
    ``W_in_local @ Q`` or ``Q.T @ W_out_local`` / ``Q.T @ b_out_local``, then returns
    ``Delta_aligned = W_aligned - W_global``. Every other predictor delta is copied byte-identically.
    ``global_weights=None`` means an all-zero baseline for backward-compatible standalone use; the
    coordinator always supplies its current grouped global weights.
    """
    if q_star.ndim != 2 or q_star.shape[-1] != q_star.shape[-2]:
        raise ValueError(
            f"q_star must be a square (d, d) rotation, got {tuple(q_star.shape)}"
        )
    out: dict[str, Tensor] = {}
    for name, tensor in predictor_delta.items():
        if name == _IN_PROJ_WEIGHT:
            local, global_weight, q = _local_gauge_weight(
                name, tensor, global_weights, q_star
            )
            aligned_local = local @ q
            out[name] = (aligned_local - global_weight).to(tensor.dtype)
        elif name == _OUT_PROJ_WEIGHT:
            local, global_weight, q = _local_gauge_weight(
                name, tensor, global_weights, q_star
            )
            aligned_local = q.transpose(-2, -1) @ local
            out[name] = (aligned_local - global_weight).to(tensor.dtype)
        elif name == _OUT_PROJ_BIAS:
            local, global_weight, q = _local_gauge_weight(
                name, tensor, global_weights, q_star
            )
            aligned_local = q.transpose(-2, -1) @ local
            out[name] = (aligned_local - global_weight).to(tensor.dtype)
        else:
            # Everything else (the encoder-less predictor remainder) is byte-identical (a copy, not a view).
            out[name] = tensor.clone()
    return out


def realign_encoder_frame_delta(
    encoder_delta: Mapping[str, Tensor],
    q_star: Tensor,
    *,
    global_weights: Mapping[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    """Align the reconstructed encoder terminal weight by row-space ``Q*`` and re-difference (#262).

    Reconstructs ``W_local = W_global + Delta`` for ``encoder.frame_proj.weight``, applies
    ``W_aligned = Q.T @ W_local``, then returns ``Delta_aligned = W_aligned - W_global``. Every other
    encoder delta is copied byte-identically. ``global_weights=None`` means an all-zero baseline for
    backward-compatible standalone use; the coordinator always supplies the current global weights.
    """
    if q_star.ndim != 2 or q_star.shape[-1] != q_star.shape[-2]:
        raise ValueError(
            f"q_star must be a square (d, d) rotation, got {tuple(q_star.shape)}"
        )
    out: dict[str, Tensor] = {}
    for name, tensor in encoder_delta.items():
        if name == _FRAME_PROJ_WEIGHT:
            local, global_weight, q = _local_gauge_weight(
                name, tensor, global_weights, q_star
            )
            aligned_local = q.transpose(-2, -1) @ local
            out[name] = (aligned_local - global_weight).to(tensor.dtype)
        else:
            # Everything else (patch_embed/pos_embed/blocks/norm) is byte-identical (a copy, not a view).
            out[name] = tensor.clone()
    return out


def procrustes_backstop(
    deltas: Mapping[str, Mapping[str, Tensor]],
    embeddings: Mapping[str, Tensor],
    e_ref: Tensor,
    *,
    global_weights: Mapping[str, Tensor] | None = None,
    threshold_deg: float,
    singular_floor: float = _DEFAULT_SINGULAR_FLOOR,
) -> dict[str, dict[str, Tensor]]:
    """Realign each over-threshold participant's full local gauge weights before the outer step.

    For each participant id in ``deltas``: recompute ``Q_c*, residual = procrustes_align(embeddings[pid],
    e_ref)`` and ``angle = _rotation_angle_deg(Q_c*)``. When ``angle > threshold_deg`` the participant's
    **encoder** terminal frame (``encoder.frame_proj.weight``) AND **predictor** I/O are reconstructed from
    ``global_weights + delta``, transformed by ``Q_c*`` in the row-space convention, and re-differenced from
    ``global_weights``. When ``angle <= threshold_deg`` the backstop is un-fired and the participant's delta
    is returned **byte-identical**.

    ``deltas`` maps ``participant_id -> {group.name -> Δ}`` (the un-flattened ``encoder.*``/``predictor.*``
    grouped delta); ``embeddings`` maps ``participant_id -> f_c(P)`` ``(n, d)``; ``e_ref`` is the reference
    frame ``(n, d)`` (e.g. the round-0 ``E_ref``); ``global_weights`` is the current grouped global
    encoder/predictor state. Omitting it means an all-zero baseline for standalone compatibility. Returns a
    NEW nested dict; inputs are not mutated.

    Degenerate handling (RFC-0002 §5). When :func:`~lensemble.gauge.procrustes.procrustes_align` raises
    :class:`~lensemble.errors.DegenerateProcrustes`, the backstop clamp-and-retries ONCE with a relaxed
    singular floor; if it still degenerates, the participant's backstop is SKIPPED (its UNALIGNED delta is
    returned byte-identical) and ``gauge/procrustes_residual`` is logged at WARN (RFC-0015). The round is NOT
    aborted — a high drift is handled in-round (realign / skip-and-warn), never raised
    (``FrameDriftExceeded`` is informational only, per #18, and is deliberately not raised here).

    Deterministic + order-independent (``INV-AGG-DETERMINISM``): each participant's result depends only on its
    own ``(embeddings[pid], e_ref)``, so permuting ``deltas`` yields the identical per-participant result.
    """
    from lensemble.errors import DegenerateProcrustes

    aligned: dict[str, dict[str, Tensor]] = {}
    for pid in deltas:
        participant_delta = deltas[pid]
        try:
            q_star, residual = procrustes_align(
                embeddings[pid], e_ref, singular_floor=singular_floor
            )
        except DegenerateProcrustes:
            # Clamp-and-retry ONCE with a relaxed floor before giving up on this participant.
            relaxed_floor = singular_floor * _RELAXED_FLOOR_FACTOR
            try:
                q_star, residual = procrustes_align(
                    embeddings[pid], e_ref, singular_floor=relaxed_floor
                )
            except DegenerateProcrustes as exc:
                # Still degenerate: SKIP the backstop for this participant (keep its UNALIGNED delta) and
                # warn. The round stays alive — the backstop never aborts (RFC-0002 §5 / #18).
                _log.warning(
                    "gauge/procrustes_residual: backstop skipped for participant %s "
                    "(degenerate SVD, min_singular_value=%.3e, floor=%.3e); keeping the unaligned delta",
                    pid,
                    getattr(exc, "min_singular_value", float("nan")),
                    relaxed_floor,
                )
                aligned[pid] = _copy_grouped_delta(participant_delta)
                continue

        angle = _rotation_angle_deg(q_star)
        if angle > threshold_deg:
            # Above threshold: align BOTH reconstructed full-local gauge surfaces, then re-difference them
            # from the current global. Every other delta passes through byte-identical.
            aligned[pid] = _apply_realignment(
                participant_delta, q_star, global_weights=global_weights
            )
        else:
            # Un-fired: the delta is byte-identical (no realignment).
            aligned[pid] = _copy_grouped_delta(participant_delta)
    return aligned


def _apply_realignment(
    participant_delta: Mapping[str, Tensor],
    q_star: Tensor,
    *,
    global_weights: Mapping[str, Tensor] | None,
) -> dict[str, Tensor]:
    """Align both full-local gauge surfaces by ``q_star`` and return grouped deltas (#262).

    Splits the grouped delta into encoder/predictor params, delegates the full-weight reconstruction and
    re-difference to :func:`realign_encoder_frame_delta` / :func:`realign_predictor_delta`, then reassembles
    it. A delta missing either group's gauge surface simply has nothing folded there.
    """
    encoder_sub = {
        name: tensor
        for name, tensor in participant_delta.items()
        if name.split(".", 1)[0] == "encoder"
    }
    predictor_sub = {
        name: tensor
        for name, tensor in participant_delta.items()
        if name.split(".", 1)[0] == "predictor"
    }
    realigned = realign_encoder_frame_delta(
        encoder_sub, q_star, global_weights=global_weights
    )
    realigned.update(
        realign_predictor_delta(predictor_sub, q_star, global_weights=global_weights)
    )
    out: dict[str, Tensor] = {}
    for name, tensor in participant_delta.items():
        # Any group that is neither encoder nor predictor (none today) passes through byte-identical.
        out[name] = realigned[name] if name in realigned else tensor.clone()
    return out


def _copy_grouped_delta(
    participant_delta: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """A byte-identical copy of a grouped delta (every tensor cloned; no realignment)."""
    return {name: tensor.clone() for name, tensor in participant_delta.items()}


def _local_gauge_weight(
    name: str,
    delta: Tensor,
    global_weights: Mapping[str, Tensor] | None,
    q_star: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reconstruct one local gauge weight and return it with global/Q in a stable work dtype."""
    if global_weights is None:
        global_weight = torch.zeros_like(delta)
    elif name not in global_weights:
        raise ValueError(
            f"global_weights is missing gauge-bearing parameter {name!r}; "
            "cannot align a delta without its current global baseline"
        )
    else:
        global_weight = global_weights[name]
    if global_weight.shape != delta.shape:
        raise ValueError(
            f"global weight {name!r} shape {tuple(global_weight.shape)} does not match "
            f"delta shape {tuple(delta.shape)}"
        )
    work = (
        torch.float64
        if torch.float64 in {q_star.dtype, global_weight.dtype, delta.dtype}
        else torch.float32
    )
    global_work = global_weight.to(device=delta.device, dtype=work)
    local = global_work + delta.to(work)
    q = q_star.to(device=delta.device, dtype=work)
    return local, global_work, q
