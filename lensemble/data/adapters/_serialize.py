"""lensemble.data.adapters._serialize — backend-agnostic tensor + ActionSpec (de)serialization (RFC-0004 §1).

The on-disk encoding is **not canonical** — ``lance`` is only the default format, never the reference
encoding (RFC-0004 §1 / RFC-0009). What every backend MUST guarantee is *exact* round-trip: a tensor is
stored as its raw little-endian bytes plus a recorded dtype string and shape tuple, so the read reshapes
to the byte-identical tensor (``torch.equal`` holds, dtype and all). This module centralizes that
encoding so the ``lance`` and ``hdf5`` backends share one tested code path.

Residency (``INV-RESIDENCY``): these helpers run inside the trust boundary on local files only; they
never touch an egress path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from lensemble.contracts import ActionKind, ActionSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

# A torch dtype <-> stable string label, so the read reconstructs the exact dtype (RFC-0009 reproducibility).
_DTYPE_TO_STR: dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
_STR_TO_DTYPE: dict[str, torch.dtype] = {v: k for k, v in _DTYPE_TO_STR.items()}


def dtype_label(dtype: torch.dtype) -> str:
    """The stable string label for a torch dtype. Raises ``ValueError`` on an unsupported dtype."""
    label = _DTYPE_TO_STR.get(dtype)
    if label is None:
        raise ValueError(
            f"unsupported tensor dtype for on-disk serialization: {dtype}; "
            f"supported: {sorted(_DTYPE_TO_STR.values())}"
        )
    return label


def tensor_to_bytes(tensor: torch.Tensor) -> tuple[bytes, str, tuple[int, ...]]:
    """Flatten ``tensor`` to raw little-endian bytes + its dtype label + shape (for an exact read-back).

    ``bfloat16`` has no numpy dtype, so it is bit-cast to ``uint16`` for the byte view and restored on
    read; every other dtype maps 1:1 to numpy. The tensor is detached and moved to CPU first (a local,
    resident operation — never an egress).
    """
    label = dtype_label(tensor.dtype)
    shape = tuple(int(s) for s in tensor.shape)
    cpu = tensor.detach().cpu().contiguous()
    if tensor.dtype is torch.bfloat16:
        raw = cpu.view(torch.uint16).numpy().tobytes()
    else:
        raw = cpu.numpy().tobytes()
    return raw, label, shape


def tensor_from_bytes(raw: bytes, label: str, shape: tuple[int, ...]) -> torch.Tensor:
    """Reconstruct the byte-identical tensor from ``tensor_to_bytes`` output (the read side of the round-trip)."""
    dtype = _STR_TO_DTYPE.get(label)
    if dtype is None:
        raise ValueError(f"unknown on-disk dtype label {label!r}")
    if dtype is torch.bfloat16:
        flat = torch.frombuffer(bytearray(raw), dtype=torch.uint16).view(torch.bfloat16)
    else:
        np_arr = np.frombuffer(raw, dtype=np.dtype(label)).copy()
        flat = torch.from_numpy(np_arr)
    return flat.reshape(shape)


# --- ActionSpec <-> a flat string map (the columns/attrs each backend persists) ---


def action_spec_to_meta(spec: ActionSpec) -> dict[str, str]:
    """Encode an ``ActionSpec`` as a flat ``str -> str`` map (backend columns/attrs).

    Tuples are stored comma-joined; an absent (``None``) bound/class field is the empty string. The
    encoding is total over the validated ``ActionSpec`` shape and exactly reversed by
    :func:`action_spec_from_meta`.
    """

    def _floats(xs: "tuple[float, ...] | None") -> str:
        return "" if xs is None else ",".join(repr(float(x)) for x in xs)

    def _ints(xs: "tuple[int, ...] | None") -> str:
        return "" if xs is None else ",".join(str(int(x)) for x in xs)

    return {
        "embodiment_id": spec.embodiment_id,
        "kind": spec.kind.value,
        "dim": str(spec.dim),
        "low": _floats(spec.low),
        "high": _floats(spec.high),
        "num_classes": _ints(spec.num_classes),
        "units": ",".join(spec.units),
        "wmcp_version": spec.wmcp_version,
    }


def action_spec_from_meta(meta: "Mapping[str, str]") -> ActionSpec:
    """Decode the flat map produced by :func:`action_spec_to_meta` back to an ``ActionSpec``."""

    def _floats(text: str) -> "tuple[float, ...] | None":
        return None if text == "" else tuple(float(x) for x in text.split(","))

    def _ints(text: str) -> "tuple[int, ...] | None":
        return None if text == "" else tuple(int(x) for x in text.split(","))

    return ActionSpec(
        embodiment_id=meta["embodiment_id"],
        kind=ActionKind(meta["kind"]),
        dim=int(meta["dim"]),
        low=_floats(meta["low"]),
        high=_floats(meta["high"]),
        num_classes=_ints(meta["num_classes"]),
        units=tuple(meta["units"].split(",")),
        wmcp_version=meta["wmcp_version"],
    )
