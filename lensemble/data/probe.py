"""lensemble.data.probe — the shared, hash-pinned public probe set P (docs/rfcs/RFC-0004 3).

The probe is the one shared, agreed artifact in an otherwise data-sovereign system: every participant
embeds the *same* probe so their latent frames are comparable against a common reference. It carries the
``k >= d`` landmark targets ``t_i = f_ref(p_i)`` the frame anchor consumes (RFC-0002 4). It is a public
artifact (no resident data) and may cross boundaries freely.

``INV-PROBE-PIN``: the versioned content hash binds the points, landmark indices, **and** landmark
targets; the targets derive only from the round-0 reference encoder ``f_ref``. A probe or target change
is therefore a versioned re-anchoring event. ``k >= d`` is necessary — ``k`` generic landmarks in
general position pin all ``d`` degrees of the ``O(d)`` rotational gauge.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from lensemble.errors import LensembleErrorCode, ProbeError

if TYPE_CHECKING:
    from lensemble.model.encoder import ReferenceEncoder

PUBLIC_PROBE_HASH_VERSION = 2
PUBLIC_PROBE_HASH_ALGORITHM = "sha256-lensemble-public-probe-v2"
PROBE_SOURCE_HASH_VERSION = 1
PROBE_SOURCE_HASH_ALGORITHM = "sha256-lensemble-probe-source-v1"

_PUBLIC_PROBE_HASH_DOMAIN = b"lensemble.public-probe\x00v2\x00"
_PROBE_SOURCE_HASH_DOMAIN = b"lensemble.probe-source\x00v1\x00"
_HASH_BYTES = 32
_HASH_HEX_CHARS = _HASH_BYTES * 2


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
    content_hash: (
        bytes  # versioned SHA-256 over points + landmark_idx + landmark_targets
    )
    probe_version: (
        int  # bumped on any content change; a re-anchoring event (RFC-0004 3.1)
    )


def probe_content_hash(
    points: Tensor, landmark_idx: Tensor, landmark_targets: Tensor
) -> bytes:
    """Return the v2 full :class:`PublicProbe` content hash.

    The preimage is domain/version separated and uses canonical safetensors bytes for ``points``,
    int64 ``landmark_idx``, and ``landmark_targets``. Tensors are detached, made contiguous, and moved
    to CPU first, so an identical probe has one digest regardless of its runtime device or layout.
    """
    if landmark_idx.ndim != 1 or landmark_idx.dtype != torch.int64:
        raise ValueError("landmark_idx must be a rank-1 torch.int64 tensor")
    raw = save_file_bytes(
        {
            "points": points.detach().cpu().contiguous(),
            "landmark_idx": landmark_idx.detach().cpu().contiguous().to(torch.int64),
            "landmark_targets": landmark_targets.detach().cpu().contiguous(),
        }
    )
    return hashlib.sha256(_PUBLIC_PROBE_HASH_DOMAIN + raw).digest()


def probe_source_hash(points: Tensor, landmark_idx: Tensor) -> bytes:
    """Return the v1 points/index source fingerprint used before targets exist.

    This is deliberately *not* a :class:`PublicProbe` content hash and MUST NOT satisfy
    ``INV-PROBE-PIN``: it does not bind ``landmark_targets``. The separate name, domain, and algorithm
    identify the narrower Phase-3 registry/pre-target contract.
    """
    if landmark_idx.ndim != 1 or landmark_idx.dtype != torch.int64:
        raise ValueError("landmark_idx must be a rank-1 torch.int64 tensor")
    raw = save_file_bytes(
        {
            "points": points.detach().cpu().contiguous(),
            "landmark_idx": landmark_idx.detach().cpu().contiguous().to(torch.int64),
        }
    )
    return hashlib.sha256(_PROBE_SOURCE_HASH_DOMAIN + raw).digest()


def save_file_bytes(tensors: dict[str, Tensor]) -> bytes:
    from safetensors.torch import save as _save

    return _save(tensors)


def _probe_error(
    message: str,
    *,
    remediation: str,
    expected_hash: bytes | None = None,
    got_hash: bytes | None = None,
) -> ProbeError:
    err = ProbeError(
        message,
        code=LensembleErrorCode.PROBE_INVALID,
        remediation=remediation,
    )
    if expected_hash is not None:
        err.expected_hash = expected_hash  # type: ignore[attr-defined]
    if got_hash is not None:
        err.got_hash = got_hash  # type: ignore[attr-defined]
    return err


def _same_tensor(lhs: Tensor, rhs: Tensor) -> bool:
    lhs_cpu = lhs.detach().cpu().contiguous()
    rhs_cpu = rhs.detach().cpu().contiguous()
    return (
        lhs_cpu.dtype == rhs_cpu.dtype
        and lhs_cpu.shape == rhs_cpu.shape
        and torch.equal(lhs_cpu, rhs_cpu)
    )


def verify_probe_content(
    probe: Any,
    *,
    expected_hash: bytes | None = None,
    landmark_targets: Tensor | None = None,
) -> bytes:
    """Recompute and verify a full probe commitment, returning its verified digest.

    ``landmark_targets`` supports the historical :class:`FrameAnchor` shape where targets are passed
    separately from a points/index holder. If the holder also carries ``landmark_targets``, the two
    tensors must be exactly equal. A stored ``content_hash``, when present, is always checked before an
    external expected pin. Missing targets, malformed hashes, target disagreement, stored-hash tamper,
    and expected-pin mismatch all fail closed with :class:`ProbeError`.
    """
    bound_targets = getattr(probe, "landmark_targets", None)
    if bound_targets is None:
        bound_targets = landmark_targets
    elif landmark_targets is not None and not _same_tensor(
        bound_targets, landmark_targets
    ):
        raise _probe_error(
            "FrameAnchor targets differ from the landmark_targets bound by the PublicProbe",
            remediation=(
                "use the exact landmark_targets carried by the pinned PublicProbe "
                "(INV-PROBE-PIN)"
            ),
        )
    if not isinstance(bound_targets, Tensor):
        raise _probe_error(
            "full PublicProbe verification requires landmark_targets; a points-only source "
            "fingerprint cannot satisfy INV-PROBE-PIN",
            remediation=(
                "build/re-anchor a PublicProbe from the round-0 reference encoder before "
                "pinning it"
            ),
        )

    try:
        points = probe.points
        landmark_idx = probe.landmark_idx
    except AttributeError as exc:
        raise _probe_error(
            "PublicProbe verification requires points and landmark_idx",
            remediation="supply a complete PublicProbe artifact",
        ) from exc
    if not isinstance(points, Tensor) or not isinstance(landmark_idx, Tensor):
        raise _probe_error(
            "PublicProbe points, landmark_idx, and landmark_targets must be tensors",
            remediation="rebuild the probe through build_probe",
        )
    if points.ndim < 1 or landmark_idx.ndim != 1 or bound_targets.ndim != 3:
        raise _probe_error(
            "PublicProbe tensor ranks are invalid",
            remediation=(
                "use points shaped (P, ...), landmark_idx shaped (k,), and "
                "landmark_targets shaped (k, N, d)"
            ),
        )
    if int(points.shape[0]) == 0:
        raise _probe_error(
            "PublicProbe points must be nonempty",
            remediation="rebuild the probe with at least one public point",
        )
    if landmark_idx.dtype != torch.int64:
        raise _probe_error(
            f"PublicProbe landmark_idx must use canonical torch.int64, got {landmark_idx.dtype}",
            remediation="rebuild the probe with a rank-1 torch.int64 landmark index tensor",
        )
    if int(landmark_idx.shape[0]) != int(bound_targets.shape[0]):
        raise _probe_error(
            "PublicProbe landmark_idx and landmark_targets disagree on landmark count",
            remediation="rebuild the probe so every landmark index has exactly one target",
        )
    if landmark_idx.numel():
        idx = landmark_idx.detach().cpu().to(torch.int64)
        if int(idx.min()) < 0 or int(idx.max()) >= int(points.shape[0]):
            raise _probe_error(
                "PublicProbe landmark_idx contains an out-of-range point index",
                remediation="rebuild the probe with indices inside [0, num_points)",
            )

    recomputed = probe_content_hash(points, landmark_idx, bound_targets)
    stored_hash = getattr(probe, "content_hash", None)
    if stored_hash is not None:
        if not isinstance(stored_hash, bytes) or len(stored_hash) != _HASH_BYTES:
            raise _probe_error(
                "PublicProbe content_hash must be a 32-byte SHA-256 digest",
                remediation="rebuild/re-anchor the probe with the current hash contract",
            )
        if stored_hash != recomputed:
            raise _probe_error(
                "PublicProbe stored content_hash does not match the recomputed v2 hash over "
                "points, landmark_idx, and landmark_targets",
                remediation=(
                    "reject the artifact and rebuild/re-anchor it; legacy points-only hashes "
                    "do not bind targets"
                ),
                expected_hash=stored_hash,
                got_hash=recomputed,
            )
    elif expected_hash is None:
        raise _probe_error(
            "PublicProbe has no stored content_hash and no external full-probe pin",
            remediation="supply a complete hash-pinned PublicProbe",
        )

    if expected_hash is not None:
        if not isinstance(expected_hash, bytes) or len(expected_hash) != _HASH_BYTES:
            raise _probe_error(
                "expected PublicProbe hash must be a 32-byte SHA-256 digest",
                remediation="supply the 64-hex v2 full PublicProbe commitment",
            )
        if recomputed != expected_hash:
            raise _probe_error(
                "PublicProbe content does not match the expected v2 full-probe pin",
                remediation=(
                    "re-pin to the federation's full PublicProbe or refuse the round "
                    "(INV-PROBE-PIN)"
                ),
                expected_hash=expected_hash,
                got_hash=recomputed,
            )
    return recomputed


def build_probe(
    points: Tensor,
    landmark_idx: Tensor,
    f_ref: "ReferenceEncoder",
    *,
    probe_version: int = 1,
) -> PublicProbe:
    """Build a :class:`PublicProbe`: derive landmark targets from the round-0 ``f_ref`` (``INV-PROBE-PIN``).

    ``landmark_targets`` of shape ``(k, N, d)`` is computed only from ``f_ref`` — never a later-round
    encoder — and the v2 content hash binds ``points + landmark_idx + landmark_targets``.

    Device-follows ``f_ref`` (#188): ``points`` may be CPU-built while the round-0 encoder snapshot lives
    on the compute device (CUDA), so the target forward runs on ``f_ref``'s device and the returned probe
    is single-device. ``save_probe`` / ``probe_content_hash`` canonicalize via ``.cpu()``, so storage
    and the pinned hash (``INV-PROBE-PIN``) are device-invariant and the CPU fallback is a no-op.
    """
    try:
        device = next(f_ref.parameters()).device
    except StopIteration:  # pragma: no cover - f_ref always carries parameters
        device = points.device
    if points.ndim < 1 or int(points.shape[0]) == 0:
        raise _probe_error(
            "public probe points must have a nonempty leading point dimension",
            remediation="supply points shaped (P, ...) with P >= 1",
        )
    points = points.to(device)
    if landmark_idx.ndim != 1 or landmark_idx.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise _probe_error(
            "landmark_idx must be a rank-1 integer tensor",
            remediation="supply integer landmark indices into the public probe points",
        )
    landmark_idx = landmark_idx.detach().to(device=device, dtype=torch.int64)
    if landmark_idx.numel() and (
        int(landmark_idx.min()) < 0 or int(landmark_idx.max()) >= int(points.shape[0])
    ):
        raise _probe_error(
            "landmark_idx contains an out-of-range public-probe point index",
            remediation="use indices inside [0, num_points)",
        )
    landmarks = _reference_input_tensor(f_ref, points[landmark_idx])
    targets = f_ref(landmarks).tokens.detach()
    probe = PublicProbe(
        points=points,
        landmark_idx=landmark_idx,
        landmark_targets=targets,
        content_hash=probe_content_hash(points, landmark_idx, targets),
        probe_version=probe_version,
    )
    verify_probe_content(probe)
    return probe


def _reference_input_tensor(module: Any, tensor: Tensor) -> Tensor:
    """Cast temporary reference-forward inputs without changing the pinned probe tensor."""
    try:
        ref = next(
            parameter
            for parameter in module.parameters()
            if parameter.is_floating_point()
        )
    except StopIteration:
        return tensor
    if torch.is_floating_point(tensor):
        return tensor.to(device=ref.device, dtype=ref.dtype)
    return tensor.to(device=ref.device)


def reanchor_probe(
    previous: PublicProbe,
    f_ref: "ReferenceEncoder",
    *,
    points: Tensor | None = None,
    landmark_idx: Tensor | None = None,
) -> PublicProbe:
    """Re-anchor a probe — a deliberate, recorded, federation-wide event (RFC-0004 §3.1).

    Changing the probe's content (``points`` / ``landmark_idx``) or the reference encoder ``f_ref`` is not
    a free edit: it redefines the reference frame and forces re-anchoring. In one operation this **bumps**
    ``probe_version`` (``previous.probe_version + 1``), **recomputes** the landmark targets
    ``t_i = f_ref(p_i)`` against the *current* ``f_ref``, and **recomputes** ``content_hash`` over the
    (possibly new) ``points + landmark_idx + landmark_targets`` (``INV-PROBE-PIN``). It cannot happen
    mid-run; a new ``probe_version`` invalidates comparability with runs that used an earlier one and
    must be recorded in the run manifest. Defaults reuse the previous ``points`` / ``landmark_idx`` so
    a pure warm-start (``f_ref``) change is expressible and produces a distinct full-probe hash.
    """
    new_points = previous.points if points is None else points
    new_idx = previous.landmark_idx if landmark_idx is None else landmark_idx
    return build_probe(
        new_points, new_idx, f_ref, probe_version=previous.probe_version + 1
    )


def verify_probe_pin(probe: PublicProbe, broadcast_hash: bytes) -> None:
    """Check the ``RoundOpen`` broadcast probe hash equals the pinned content hash (``INV-PROBE-PIN``).

    Raises :class:`~lensemble.errors.ProbeError` (code ``PROBE_INVALID``, fail-closed) on a hash mismatch
    (re-anchoring required) or landmark under-coverage (``k < d``). No-op return on success.
    """
    recomputed = verify_probe_content(probe, expected_hash=broadcast_hash)
    k = int(probe.landmark_targets.shape[0])
    d = int(probe.landmark_targets.shape[-1])

    def fail(msg: str, remediation: str) -> ProbeError:
        err = ProbeError(
            msg, code=LensembleErrorCode.PROBE_INVALID, remediation=remediation
        )
        err.expected_hash = recomputed  # type: ignore[attr-defined]
        err.got_hash = broadcast_hash  # type: ignore[attr-defined]
        err.num_landmarks = k  # type: ignore[attr-defined]
        err.d = d  # type: ignore[attr-defined]
        return err

    if k < d:
        raise fail(
            f"probe under-coverage: k={k} landmarks < d={d}; the anchor under-determines the O(d) frame",
            "supply at least d landmarks in general position (k >= d)",
        )


def save_probe(probe: PublicProbe, path: Path) -> None:
    """Write a self-consistent v2 :class:`PublicProbe` artifact.

    The stored digest is recomputed first; a caller cannot persist modified targets under a stale pin.
    """
    path = Path(path)
    verified_hash = verify_probe_content(probe)
    if probe.probe_version < 1:
        raise _probe_error(
            "PublicProbe probe_version must be >= 1",
            remediation="use a positive, monotonically increasing probe version",
        )
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
            "content_hash": verified_hash.hex(),
            "content_hash_algorithm": PUBLIC_PROBE_HASH_ALGORITHM,
            "content_hash_version": str(PUBLIC_PROBE_HASH_VERSION),
            "probe_version": str(probe.probe_version),
        },
    )


def load_probe(path: Path) -> PublicProbe:
    """Load and integrity-check a current v2 :class:`PublicProbe`.

    Unversioned/legacy metadata is rejected explicitly: the historical points-only digest did not bind
    ``landmark_targets`` and therefore cannot be upgraded safely without rebuilding from ``f_ref``.
    """
    path = Path(path)
    tensors: dict[str, Tensor] = {}
    with safe_open(str(path), framework="pt") as f:  # type: ignore[no-untyped-call]
        meta = f.metadata() or {}
        hash_version = meta.get("content_hash_version")
        hash_algorithm = meta.get("content_hash_algorithm")
        if (
            hash_version != str(PUBLIC_PROBE_HASH_VERSION)
            or hash_algorithm != PUBLIC_PROBE_HASH_ALGORITHM
        ):
            raise _probe_error(
                "unsupported or legacy PublicProbe hash metadata: expected "
                f"{PUBLIC_PROBE_HASH_ALGORITHM} version {PUBLIC_PROBE_HASH_VERSION}, got "
                f"algorithm={hash_algorithm!r}, version={hash_version!r}; legacy points-only "
                "hashes do not bind landmark_targets",
                remediation=(
                    "rebuild/re-anchor the probe from the pinned round-0 reference encoder and "
                    "publish a new probe version"
                ),
            )
        content_hash_hex = meta.get("content_hash")
        if (
            content_hash_hex is None
            or len(content_hash_hex) != _HASH_HEX_CHARS
            or any(char not in "0123456789abcdef" for char in content_hash_hex)
        ):
            raise _probe_error(
                "PublicProbe metadata content_hash must be exactly 64 lowercase hex characters",
                remediation="reject and rebuild the probe artifact",
            )
        probe_version_raw = meta.get("probe_version")
        try:
            probe_version = int(probe_version_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise _probe_error(
                f"PublicProbe metadata probe_version is invalid: {probe_version_raw!r}",
                remediation="rebuild the probe with an integer probe_version >= 1",
            ) from exc
        if probe_version < 1 or str(probe_version) != probe_version_raw:
            raise _probe_error(
                f"PublicProbe metadata probe_version is invalid: {probe_version_raw!r}",
                remediation="rebuild the probe with a canonical integer probe_version >= 1",
            )
        for key in ("points", "landmark_idx", "landmark_targets"):
            if key not in f.keys():
                raise _probe_error(
                    f"PublicProbe artifact is missing required tensor {key!r}",
                    remediation="reject and rebuild the complete probe artifact",
                )
            tensors[key] = f.get_tensor(key)
    probe = PublicProbe(
        points=tensors["points"],
        landmark_idx=tensors["landmark_idx"],
        landmark_targets=tensors["landmark_targets"],
        content_hash=bytes.fromhex(content_hash_hex),
        probe_version=probe_version,
    )
    verify_probe_content(probe)
    return probe


def probe_record(probe: PublicProbe) -> str:
    """A minimal JSON record of a probe (content hash + version + sizes).

    A precursor to the full ``RunManifest`` probe fields (#36) the CLI will emit once available.
    """
    return json.dumps(
        {
            "content_hash": probe.content_hash.hex(),
            "content_hash_algorithm": PUBLIC_PROBE_HASH_ALGORITHM,
            "content_hash_version": PUBLIC_PROBE_HASH_VERSION,
            "probe_version": probe.probe_version,
            "num_points": int(probe.points.shape[0]),
            "num_landmarks": int(probe.landmark_targets.shape[0]),
            "d": int(probe.landmark_targets.shape[-1]),
        },
        sort_keys=True,
    )
