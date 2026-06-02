"""lensemble.gauge.drift — the frame-drift diagnostic, the headline measurement (RFC-0002 9 / RFC-0005 2).

:func:`frame_drift` measures the inter-participant latent frame drift on the pinned public probe: for
each participant pair it Procrustes-aligns their probe embeddings and reports the recovered rotation
angle (the headline figure) and the alignment residual (RFC-0002 5). The diagnostic is a deterministic
function of committed weights + the pinned probe — inputs derive only from hash-committed checkpoints
(``INV-CHECKPOINT-HASH``) and the hash-pinned probe (``INV-PROBE-PIN``) — so it is publicly recomputable
and needs no proof (RFC-0006 4). A probe-hash mismatch raises ``ProbeError``; a degenerate SVD raises
``DegenerateProcrustes`` rather than emit a meaningless angle.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from lensemble.data.probe import probe_content_hash
from lensemble.errors import LensembleErrorCode, ProbeError
from lensemble.gauge.procrustes import procrustes_align

if TYPE_CHECKING:
    from collections.abc import Mapping

FRAME_DRIFT_SCHEMA_VERSION = 1
_GLOBAL_KEY = "global"  # the reserved participant id for the aggregated/global model


class PairDrift(BaseModel):
    """The drift between one participant pair's frames on the probe (03 13.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    participant_a: str
    participant_b: str
    rotation_angle_deg: (
        float  # recovered inter-frame rotation angle on P (the headline figure)
    )
    procrustes_residual: (
        float  # the optimal-Procrustes residual ||Q* A - B||_F (RFC-0002 5)
    )


class FrameDriftReport(BaseModel):
    """Per-round latent-frame-drift record — the central reproducible figure of the paper (03 13.2).

    Deterministic given committed weights + the pinned probe; recomputing it reproduces it bit-for-bit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = FRAME_DRIFT_SCHEMA_VERSION
    round_index: int
    probe_hash: str  # the pinned probe these embeddings came from (INV-PROBE-PIN)
    pairs: tuple[PairDrift, ...]  # all C-choose-2 participant pairs
    drift_from_global: dict[
        str, float
    ]  # participant_id -> rotation angle vs the global model


def _rotation_angle_deg(rotation: Tensor) -> float:
    """The rotation angle of ``rotation in SO(d)`` in degrees, from its trace (exact for one plane)."""
    d = rotation.shape[-1]
    cos = (float(torch.trace(rotation)) - (d - 2)) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def frame_drift(
    embeddings: Mapping[str, Tensor],
    *,
    round_index: int = 0,
    probe: Any = None,
    expected_probe_hash: str | None = None,
) -> FrameDriftReport:
    """Compute the frame-drift diagnostic from per-participant probe embeddings (RFC-0002 9).

    ``embeddings`` maps ``participant_id -> f_c(P)`` of shape ``(|P|*N, d)``; the reserved key
    ``"global"`` is the aggregated model. For each participant pair, :func:`procrustes_align` recovers
    the inter-frame rotation; ``rotation_angle_deg`` is the headline figure and ``procrustes_residual``
    the alignment residual. ``drift_from_global`` is each participant's angle against ``"global"``.

    Probe pin (``INV-PROBE-PIN``): when ``probe`` is given its content hash is recomputed; if
    ``expected_probe_hash`` is given and differs, raises :class:`~lensemble.errors.ProbeError` and
    refuses to run. The verified hash populates ``probe_hash``.
    """
    if probe is not None:
        recomputed = probe_content_hash(probe.points, probe.landmark_idx).hex()
        if expected_probe_hash is not None and recomputed != expected_probe_hash:
            raise ProbeError(
                "probe content hash does not match the pinned hash; refusing to measure drift",
                code=LensembleErrorCode.PROBE_INVALID,
                remediation="re-pin the probe or pass the probe these embeddings were produced from",
            )
        probe_hash = recomputed
    else:
        probe_hash = expected_probe_hash if expected_probe_hash is not None else ""

    participants = sorted(pid for pid in embeddings if pid != _GLOBAL_KEY)

    pairs: list[PairDrift] = []
    for i, a in enumerate(participants):
        for b in participants[i + 1 :]:
            rotation, residual = procrustes_align(embeddings[a], embeddings[b])
            pairs.append(
                PairDrift(
                    participant_a=a,
                    participant_b=b,
                    rotation_angle_deg=_rotation_angle_deg(rotation),
                    procrustes_residual=residual,
                )
            )

    drift_from_global: dict[str, float] = {}
    if _GLOBAL_KEY in embeddings:
        for c in participants:
            rotation, _ = procrustes_align(embeddings[c], embeddings[_GLOBAL_KEY])
            drift_from_global[c] = _rotation_angle_deg(rotation)

    return FrameDriftReport(
        round_index=round_index,
        probe_hash=probe_hash,
        pairs=tuple(pairs),
        drift_from_global=drift_from_global,
    )
