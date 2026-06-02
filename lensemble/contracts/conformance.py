"""lensemble.contracts.conformance — total WMCP conformance checks (docs/rfcs/RFC-0007 4).

``check_latent_state`` is pure (no I/O, no mutation) and order-independent. It is the only sanctioned
validation path for a ``LatentState``; callers must not re-implement ad hoc shape checks. A failure is
a hard reject — the check never reshapes or coerces, because a silent reshape would mask a real model
bug or interface disagreement. Enforces ``INV-WMCP``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from lensemble.contracts.latent import WMCP_VERSION, LatentState
from lensemble.errors import ContractViolation, LensembleErrorCode

if TYPE_CHECKING:
    from typing import Any

_ALLOWED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _fail(
    field: str,
    remediation: str,
    *,
    got_shape: tuple[int, ...],
    wmcp_version: str,
    expected_shape: "tuple[Any, ...] | None" = None,
) -> None:
    """Raise a ``ContractViolation`` carrying the diagnostic fields (RFC-0007 7)."""
    err = ContractViolation(
        f"LatentState conformance failed: {field}",
        code=LensembleErrorCode.WMCP_CONTRACT_VIOLATION,
        remediation=remediation,
    )
    # Diagnostic context (RFC-0007 7 sub-cases are distinguished by `.field`/`.remediation`).
    err.field = field  # type: ignore[attr-defined]
    err.wmcp_version = wmcp_version  # type: ignore[attr-defined]
    err.got_shape = got_shape  # type: ignore[attr-defined]
    err.expected_shape = expected_shape  # type: ignore[attr-defined]
    raise err


def check_latent_state(
    state: LatentState,
    *,
    expected_dim: int | None = None,
    expected_num_tokens: int | None = None,
) -> None:
    """Validate a ``LatentState`` against the WMCP contract (``INV-WMCP``, RFC-0007 4).

    Checks, in order: ``wmcp_version == WMCP_VERSION``; rank in ``{2, 3}``; the trailing axis equals
    ``state.dim`` and the tokens axis equals ``state.num_tokens``; the optional ``expected_dim`` /
    ``expected_num_tokens`` when supplied; dtype in ``{bfloat16, float16, float32}``; all-finite tokens.
    Raises :class:`~lensemble.errors.ContractViolation` (code ``WMCP_CONTRACT_VIOLATION``) on the first
    failing clause; no-op return on success.
    """
    got: tuple[int, ...] = tuple(state.tokens.shape)
    ver = state.wmcp_version

    if ver != WMCP_VERSION:
        _fail(
            "wmcp_version",
            f"expected wmcp_version == {WMCP_VERSION!r}, got {ver!r}; "
            "re-encode with the pinned WMCP version or refuse the join",
            got_shape=got,
            wmcp_version=ver,
        )

    ndim = state.tokens.ndim
    if ndim not in (2, 3):
        _fail(
            "rank",
            f"expected tokens rank in {{2, 3}} ((N, d) or (B, N, d)), got rank {ndim} with shape {got}",
            got_shape=got,
            wmcp_version=ver,
        )

    if got[-1] != state.dim:
        _fail(
            "dim",
            f"expected the last axis == dim ({state.dim}), got {got[-1]} in shape {got}",
            got_shape=got,
            wmcp_version=ver,
            expected_shape=("...", state.dim),
        )

    tokens_axis = got[0] if ndim == 2 else got[1]
    if tokens_axis != state.num_tokens:
        _fail(
            "num_tokens",
            f"expected the tokens axis == num_tokens ({state.num_tokens}), got {tokens_axis} in shape {got}",
            got_shape=got,
            wmcp_version=ver,
        )

    if expected_dim is not None and state.dim != expected_dim:
        _fail(
            "expected_dim",
            f"expected dim == {expected_dim} (the size this consumer was built for), got {state.dim}",
            got_shape=got,
            wmcp_version=ver,
        )

    if expected_num_tokens is not None and state.num_tokens != expected_num_tokens:
        _fail(
            "expected_num_tokens",
            f"expected num_tokens == {expected_num_tokens}, got {state.num_tokens}",
            got_shape=got,
            wmcp_version=ver,
        )

    if state.tokens.dtype not in _ALLOWED_DTYPES:
        _fail(
            "dtype",
            f"expected dtype in {{bfloat16, float16, float32}}, got {state.tokens.dtype}; "
            "cast the encoder output before emitting a LatentState",
            got_shape=got,
            wmcp_version=ver,
        )

    if not bool(torch.isfinite(state.tokens).all()):
        _fail(
            "finiteness",
            "expected all-finite tokens (no NaN/Inf); a non-finite latent indicates an upstream "
            "numerical fault and is rejected before it can poison aggregation",
            got_shape=got,
            wmcp_version=ver,
        )
