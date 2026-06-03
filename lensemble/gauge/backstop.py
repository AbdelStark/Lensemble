"""lensemble.gauge.backstop — Layer-3 Procrustes re-alignment backstop at aggregation (RFC-0002 §5).

Immediately before the outer step, for each participant whose latent frame drift on the public probe
exceeds the configured threshold, recompute the hard alignment ``Q_c* = procrustes_align(f_c(P), E_ref)``
and apply it to the participant's *released* delta as a **pure linear operation** — so the result stays
bitwise-deterministic and publicly recomputable (``INV-AGG-DETERMINISM``; RFC-0006 §3 ``recompute_alignment``).

Activation-space realization (the recorded #18 decision). RFC-0002 §5 folds ``Q_c*`` into the encoder's
terminal linear map and conjugates the predictor I/O (``g_phi -> Q g_phi Q^T``). The encoder here ends in a
``LayerNorm`` with **no** terminal ``(d, d)`` linear to absorb ``Q`` into, and the maintainer chose **not**
to add one, so only the *weight-expressible* part — the predictor conjugation — is applied to the committed
delta. The encoder-frame component is bounded by the Layer-2 anchor (RFC-0002 §4) and lives in *activation
space*, NOT in the committed weights — which is exactly why ``recompute_alignment`` (#62) measures residual
encoder drift rather than verifying a weight-fold (the recorded #18/#62 verifiability tradeoff).

Concretely, the predictor conjugation ``g_phi -> Q g_phi Q^T`` is linear in the predictor delta and rotates
exactly three params (latent dim ``d``, predictor width ``width``); every other predictor param and the
whole encoder delta are returned **byte-identical**:

- ``predictor.in_proj.weight`` ``(width, d)``: ``Δ <- Δ @ Q^T`` (rotate the latent *input* axis);
- ``predictor.out_proj.weight`` ``(d, width)``: ``Δ <- Q @ Δ`` (rotate the latent *output* axis);
- ``predictor.out_proj.bias``  ``(d,)``:        ``Δ <- Q @ Δ``.

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

__all__ = ["realign_predictor_delta", "procrustes_backstop"]

_log = logging.getLogger(__name__)

# The three predictor params the conjugation g_phi -> Q g_phi Q^T touches (the bare state_dict keys with the
# `predictor.` group prefix, matching build_pseudogradient's `predictor.*` keys). Everything else — the whole
# encoder delta and predictor.cond_proj/pos_embed/blocks/norm/in_proj.bias — is returned byte-identical.
_IN_PROJ_WEIGHT = "predictor.in_proj.weight"  # (width, d): Δ <- Δ @ Q^T
_OUT_PROJ_WEIGHT = "predictor.out_proj.weight"  # (d, width): Δ <- Q @ Δ
_OUT_PROJ_BIAS = "predictor.out_proj.bias"  # (d,):       Δ <- Q @ Δ

# The clamp-and-retry singular floor for the degenerate path: one relaxed retry before skipping the backstop
# for that participant (RFC-0002 §5 "the caller clamps/conditions and re-tries, or skips ... and logs").
_RELAXED_FLOOR_FACTOR = 1e-3


def realign_predictor_delta(
    predictor_delta: Mapping[str, Tensor], q_star: Tensor
) -> dict[str, Tensor]:
    """Conjugate a predictor-group delta by ``Q*`` — the weight-expressible Layer-3 fold-in (RFC-0002 §5).

    Applies ``g_phi -> Q g_phi Q^T`` directly to the released predictor delta (linear in the delta), rotating
    exactly ``predictor.in_proj.weight`` (``Δ <- Δ @ Q^T``), ``predictor.out_proj.weight`` (``Δ <- Q @ Δ``),
    and ``predictor.out_proj.bias`` (``Δ <- Q @ Δ``). Every other key (``cond_proj``/``pos_embed``/``blocks``/
    ``norm``/``in_proj.bias``) is copied through **byte-identical**. Returns a NEW dict; the input is not
    mutated.

    ``q_star`` is the ``(d, d)`` rotation; ``d`` is the predictor latent dim (the second axis of
    ``in_proj.weight`` and the first axis of ``out_proj.weight``/``out_proj.bias``). The rotation is computed
    in fp32 (fp64 kept) and cast back to each delta's dtype so the operation is dtype-preserving and
    bitwise-reproducible (``INV-AGG-DETERMINISM``). ``Q = I`` is a no-op.
    """
    if q_star.ndim != 2 or q_star.shape[-1] != q_star.shape[-2]:
        raise ValueError(
            f"q_star must be a square (d, d) rotation, got {tuple(q_star.shape)}"
        )
    work = torch.float64 if q_star.dtype == torch.float64 else torch.float32
    q = q_star.to(work)
    qt = q.transpose(-2, -1)

    out: dict[str, Tensor] = {}
    for name, tensor in predictor_delta.items():
        if name == _IN_PROJ_WEIGHT:
            # (width, d) @ (d, d) — rotate the latent INPUT axis (contract the d input dim by Q^T).
            rotated = tensor.to(work) @ qt
            out[name] = rotated.to(tensor.dtype)
        elif name == _OUT_PROJ_WEIGHT:
            # (d, d) @ (d, width) — rotate the latent OUTPUT axis (rotate the d output rows by Q).
            rotated = q @ tensor.to(work)
            out[name] = rotated.to(tensor.dtype)
        elif name == _OUT_PROJ_BIAS:
            # (d, d) @ (d,) — rotate the output bias by Q.
            rotated = q @ tensor.to(work)
            out[name] = rotated.to(tensor.dtype)
        else:
            # Everything else (the encoder-less predictor remainder) is byte-identical (a copy, not a view).
            out[name] = tensor.clone()
    return out


def procrustes_backstop(
    deltas: Mapping[str, Mapping[str, Tensor]],
    embeddings: Mapping[str, Tensor],
    e_ref: Tensor,
    *,
    threshold_deg: float,
    singular_floor: float = _DEFAULT_SINGULAR_FLOOR,
) -> dict[str, dict[str, Tensor]]:
    """Layer-3 backstop: realign each over-threshold participant's predictor delta before the outer step.

    For each participant id in ``deltas``: recompute ``Q_c*, residual = procrustes_align(embeddings[pid],
    e_ref)`` and ``angle = _rotation_angle_deg(Q_c*)``. When ``angle > threshold_deg`` the participant's
    **predictor** delta is conjugated by ``Q_c*`` (:func:`realign_predictor_delta`); its encoder delta is left
    UNCHANGED (the activation-space decision — there is no terminal encoder linear to absorb ``Q`` into).
    When ``angle <= threshold_deg`` the backstop is un-fired and the participant's delta is returned
    **byte-identical**.

    ``deltas`` maps ``participant_id -> {group.name -> Δ}`` (the un-flattened ``encoder.*``/``predictor.*``
    grouped delta); ``embeddings`` maps ``participant_id -> f_c(P)`` ``(n, d)``; ``e_ref`` is the reference
    frame ``(n, d)`` (e.g. the round-0 ``E_ref``). Returns a NEW nested dict; inputs are not mutated.

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
            # Above threshold: conjugate the predictor delta; the encoder delta passes through byte-identical.
            aligned[pid] = _apply_predictor_realignment(participant_delta, q_star)
        else:
            # Un-fired: the delta is byte-identical (no realignment).
            aligned[pid] = _copy_grouped_delta(participant_delta)
    return aligned


def _apply_predictor_realignment(
    participant_delta: Mapping[str, Tensor], q_star: Tensor
) -> dict[str, Tensor]:
    """Realign the ``predictor.*`` sub-delta by ``q_star``; pass the encoder (and any other group) through.

    Splits the grouped delta into its predictor params (conjugated by :func:`realign_predictor_delta`) and the
    encoder params (byte-identical, the activation-space decision), then reassembles the full grouped delta.
    """
    predictor_sub = {
        name: tensor
        for name, tensor in participant_delta.items()
        if name.split(".", 1)[0] == "predictor"
    }
    realigned_predictor = realign_predictor_delta(predictor_sub, q_star)
    out: dict[str, Tensor] = {}
    for name, tensor in participant_delta.items():
        if name in realigned_predictor:
            out[name] = realigned_predictor[name]
        else:
            # Encoder (and any non-predictor group): byte-identical copy (the encoder frame is NOT folded).
            out[name] = tensor.clone()
    return out


def _copy_grouped_delta(
    participant_delta: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """A byte-identical copy of a grouped delta (every tensor cloned; no realignment)."""
    return {name: tensor.clone() for name, tensor in participant_delta.items()}
