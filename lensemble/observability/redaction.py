"""lensemble.observability.redaction — the allow-list redaction guard (docs/rfcs/RFC-0015 5).

Enforces ``INV-RESIDENCY`` on the observability path: no raw observation, action, or private embedding
ever reaches a log line, metric sample, or diagnostic record. ``redact`` is an **allow-list** (a
block-list of "things that look private" is unwinnable; the allow-list fails toward over-rejection, the
safe direction). A disallowed value raises :class:`~lensemble.errors.ResidencyViolation` (fail-closed,
never caught-and-ignored) and the record is not written.

Emittable: ``bool`` / ``int`` / finite ``float`` / ``str``, hex-hash ``bytes`` of a digest length, shape
tuples (``tuple[int, ...]``), and dtype strings; mappings/sequences are recursed. Rejected: any
``torch.Tensor`` / ``numpy.ndarray`` or tensor-like buffer (``__array__`` / ``__torch_function__``),
non-hex ``bytes``, ``NaN``/``Inf`` floats, and any object not on the allow-list. To log "what an
embedding looked like", emit its shape, dtype, and L2 norm and/or content hash — never the tensor.

This module imports neither torch nor numpy: tensor-likes are detected by duck typing, keeping the guard
importable on any path.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from lensemble.errors import LensembleErrorCode, ResidencyViolation

_HEX_DIGEST_LENGTHS = frozenset(
    {32, 40, 56, 64, 96, 128}
)  # md5/sha1/.../sha512 hex lengths
_HEX_RE = re.compile(rb"^[0-9a-f]+$")


def _is_tensor_like(value: object) -> bool:
    """True for torch/numpy tensors and anything exposing a tensor buffer — without importing them."""
    if hasattr(value, "__array__") or hasattr(value, "__torch_function__"):
        return True
    module = type(value).__module__.split(".", 1)[0]
    return module in {"torch", "numpy"}


def _is_hex_hash(value: bytes | bytearray) -> bool:
    return len(value) in _HEX_DIGEST_LENGTHS and bool(
        _HEX_RE.match(bytes(value).lower())
    )


def _reject(field: str, reason: str) -> ResidencyViolation:
    err = ResidencyViolation(
        f"observability field {field!r} is not emittable: {reason}",
        code=LensembleErrorCode.RESIDENCY_VIOLATION,
        remediation="emit shape/dtype/L2-norm/hash, never the tensor (INV-RESIDENCY)",
    )
    err.tensor_role = "observability_field"  # type: ignore[attr-defined]
    err.boundary = "observability-sink"  # type: ignore[attr-defined]
    return err


def redact(value: object, *, field: str) -> Any:
    """Return ``value`` iff it is on the emittable allow-list; otherwise fail closed (RFC-0015 5).

    Raises :class:`~lensemble.errors.ResidencyViolation` (``RESIDENCY_VIOLATION``) on a tensor-like,
    non-hex ``bytes``, a non-finite float, or any value not on the allow-list. Mappings/sequences are
    recursed; a single disallowed leaf rejects the whole record.
    """
    # Tensor-likes are rejected first (a tensor scalar would otherwise pass an isinstance(float) check).
    if _is_tensor_like(value):
        raise _reject(
            field, "a tensor / array (raw or derived embedding/observation/action)"
        )
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _reject(field, "a non-finite float (NaN/Inf)")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        if _is_hex_hash(value):
            return value
        raise _reject(field, "raw bytes that are not a hex-encoded digest")
    if isinstance(value, Mapping):
        return {k: redact(v, field=f"{field}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return type(value)(redact(v, field=f"{field}[]") for v in value)
    raise _reject(
        field, f"type {type(value).__name__!r} is not on the emittable allow-list"
    )


def redact_record(record: Mapping[str, object]) -> dict[str, Any]:
    """Redact every field of a record, all-or-nothing.

    Returns a redacted copy if every field is emittable; raises :class:`ResidencyViolation` on the first
    disallowed leaf so a caller never writes a partially-redacted record (fail-closed). The emit/sink
    facades (``emit_log`` / ``emit_metric``, the diagnostic facade) route through this before any write.
    """
    return {k: redact(v, field=k) for k, v in record.items()}
