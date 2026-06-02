"""lensemble.data.probe — the shared, hash-pinned public probe set P (docs/rfcs/RFC-0004 3).

The probe is the one shared, agreed artifact in an otherwise data-sovereign system: every participant
embeds the *same* probe so their latent frames are comparable against a common reference. It carries the
``k >= d`` landmark targets ``t_i = f_ref(p_i)`` the frame anchor consumes (RFC-0002 4). It is a public
artifact (no resident data) and may cross boundaries freely.

``INV-PROBE-PIN``: the content hash is pinned and landmark targets derive **only** from the round-0
reference encoder ``f_ref``; a probe change is a versioned re-anchoring event. ``k >= d`` is necessary —
``k`` generic landmarks in general position pin all ``d`` degrees of the ``O(d)`` rotational gauge.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from lensemble.errors import LensembleErrorCode, ProbeError

if TYPE_CHECKING:
    from lensemble.model.encoder import ReferenceEncoder


@dataclass(frozen=True)
class PublicProbe:
    """The fixed, hash-pinned, public probe set ``P`` and its landmark targets (RFC-0004 3).

    Public artifact: contains no resident data and may cross boundaries freely.
    """

    points: Tensor  # (P, *obs_shape) — the probe inputs p_i (public)
    landmark_idx: Tensor  # (k,) indices into points marking the k >= d landmarks
    landmark_targets: (
        Tensor  # (k, N, d) — t_i = f_ref(p_i), derived ONLY from f_ref (INV-PROBE-PIN)
    )
    content_hash: bytes  # SHA-256 over canonical bytes of points + landmark_idx
    probe_version: (
        int  # bumped on any content change; a re-anchoring event (RFC-0004 3.1)
    )


def probe_content_hash(points: Tensor, landmark_idx: Tensor) -> bytes:
    """SHA-256 over the canonical (safetensors) bytes of ``points + landmark_idx``."""
    raw = save_file_bytes(
        {
            "points": points.detach().cpu().contiguous(),
            "landmark_idx": landmark_idx.detach().cpu().contiguous().to(torch.int64),
        }
    )
    return hashlib.sha256(raw).digest()


def save_file_bytes(tensors: dict[str, Tensor]) -> bytes:
    from safetensors.torch import save as _save

    return _save(tensors)


def build_probe(
    points: Tensor,
    landmark_idx: Tensor,
    f_ref: "ReferenceEncoder",
    *,
    probe_version: int = 1,
) -> PublicProbe:
    """Build a :class:`PublicProbe`: derive landmark targets from the round-0 ``f_ref`` (``INV-PROBE-PIN``).

    ``landmark_targets`` of shape ``(k, N, d)`` is computed only from ``f_ref`` — never a later-round
    encoder — and the content hash is pinned over ``points + landmark_idx``.
    """
    landmarks = points[landmark_idx]
    targets = f_ref(landmarks).tokens.detach()
    return PublicProbe(
        points=points,
        landmark_idx=landmark_idx,
        landmark_targets=targets,
        content_hash=probe_content_hash(points, landmark_idx),
        probe_version=probe_version,
    )


def verify_probe_pin(probe: PublicProbe, broadcast_hash: bytes) -> None:
    """Check the ``RoundOpen`` broadcast probe hash equals the pinned content hash (``INV-PROBE-PIN``).

    Raises :class:`~lensemble.errors.ProbeError` (code ``PROBE_INVALID``, fail-closed) on a hash mismatch
    (re-anchoring required) or landmark under-coverage (``k < d``). No-op return on success.
    """
    k = int(probe.landmark_targets.shape[0])
    d = int(probe.landmark_targets.shape[-1])

    def fail(msg: str, remediation: str) -> ProbeError:
        err = ProbeError(
            msg, code=LensembleErrorCode.PROBE_INVALID, remediation=remediation
        )
        err.expected_hash = probe.content_hash  # type: ignore[attr-defined]
        err.got_hash = broadcast_hash  # type: ignore[attr-defined]
        err.num_landmarks = k  # type: ignore[attr-defined]
        err.d = d  # type: ignore[attr-defined]
        return err

    if k < d:
        raise fail(
            f"probe under-coverage: k={k} landmarks < d={d}; the anchor under-determines the O(d) frame",
            "supply at least d landmarks in general position (k >= d)",
        )
    if probe.content_hash != broadcast_hash:
        raise fail(
            "probe content hash does not match the RoundOpen broadcast hash; the federation pins a "
            "different probe (a probe change is a re-anchoring event, RFC-0004 3.1)",
            "re-pin to the federation's probe or refuse the round (INV-PROBE-PIN)",
        )


def save_probe(probe: PublicProbe, path: Path) -> None:
    """Write a :class:`PublicProbe` to ``path`` (safetensors tensors + pinned metadata)."""
    path = Path(path)
    save_file(
        {
            "points": probe.points.detach().cpu().contiguous(),
            "landmark_idx": probe.landmark_idx.detach()
            .cpu()
            .contiguous()
            .to(torch.int64),
            "landmark_targets": probe.landmark_targets.detach().cpu().contiguous(),
        },
        str(path),
        metadata={
            "content_hash": probe.content_hash.hex(),
            "probe_version": str(probe.probe_version),
        },
    )


def load_probe(path: Path) -> PublicProbe:
    """Load a :class:`PublicProbe` written by :func:`save_probe`."""
    path = Path(path)
    tensors: dict[str, Tensor] = {}
    with safe_open(str(path), framework="pt") as f:  # type: ignore[no-untyped-call]
        meta = f.metadata() or {}
        for key in ("points", "landmark_idx", "landmark_targets"):
            tensors[key] = f.get_tensor(key)
    return PublicProbe(
        points=tensors["points"],
        landmark_idx=tensors["landmark_idx"],
        landmark_targets=tensors["landmark_targets"],
        content_hash=bytes.fromhex(meta["content_hash"]),
        probe_version=int(meta["probe_version"]),
    )


def probe_record(probe: PublicProbe) -> str:
    """A minimal JSON record of a probe (content hash + version + sizes).

    A precursor to the full ``RunManifest`` probe fields (#36) the CLI will emit once available.
    """
    return json.dumps(
        {
            "content_hash": probe.content_hash.hex(),
            "probe_version": probe.probe_version,
            "num_points": int(probe.points.shape[0]),
            "num_landmarks": int(probe.landmark_targets.shape[0]),
            "d": int(probe.landmark_targets.shape[-1]),
        },
        sort_keys=True,
    )
