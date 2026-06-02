"""lensemble.contracts.conformance — total WMCP conformance checks (docs/rfcs/RFC-0007 4).

``check_latent_state`` is pure (no I/O, no mutation) and order-independent. It is the only sanctioned
validation path for a ``LatentState``; callers must not re-implement ad hoc shape checks. A failure is
a hard reject — the check never reshapes or coerces, because a silent reshape would mask a real model
bug or interface disagreement. Enforces ``INV-WMCP``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, NoReturn

import torch

from lensemble.contracts.action import ActionKind, ActionSpec
from lensemble.contracts.latent import WMCP_VERSION, LatentState
from lensemble.errors import ContractViolation, LensembleErrorCode

if TYPE_CHECKING:
    from typing import Any

_ALLOWED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_EMBODIMENT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _fail(
    field: str,
    remediation: str,
    *,
    got_shape: tuple[int, ...],
    wmcp_version: str,
    expected_shape: "tuple[Any, ...] | None" = None,
) -> NoReturn:
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


def _spec_fail(field: str, remediation: str) -> NoReturn:
    """Raise a ``ContractViolation`` for an ``ActionSpec`` rule (RFC-0007 7)."""
    err = ContractViolation(
        f"ActionSpec validation failed: {field}",
        code=LensembleErrorCode.WMCP_CONTRACT_VIOLATION,
        remediation=remediation,
    )
    err.field = field  # type: ignore[attr-defined]
    raise err


def validate_action_spec(spec: ActionSpec) -> None:
    """Validate an ``ActionSpec`` before action-head construction (``INV-WMCP``, RFC-0007 3/4).

    Raises :class:`~lensemble.errors.ContractViolation` (code ``WMCP_CONTRACT_VIOLATION``) on any rule;
    no-op return on success. Pure: no I/O, no mutation.
    """
    if spec.wmcp_version != WMCP_VERSION:
        _spec_fail(
            "wmcp_version",
            f"expected wmcp_version == {WMCP_VERSION!r}, got {spec.wmcp_version!r}",
        )
    if not spec.embodiment_id or not _EMBODIMENT_ID_RE.match(spec.embodiment_id):
        _spec_fail(
            "embodiment_id",
            "expected a non-empty embodiment_id matching ^[a-z0-9][a-z0-9._-]*$ "
            f"(safe as a key and a log/file label), got {spec.embodiment_id!r}",
        )
    if spec.dim <= 0:
        _spec_fail("dim", f"expected dim > 0, got {spec.dim}")
    if len(spec.units) != spec.dim:
        _spec_fail(
            "units",
            f"expected len(units) == dim ({spec.dim}), got {len(spec.units)}",
        )

    if spec.kind is ActionKind.CONTINUOUS:
        if spec.num_classes is not None:
            _spec_fail("num_classes", "continuous spec must have num_classes is None")
        if spec.low is None or spec.high is None:
            _spec_fail("bounds", "continuous spec must provide low and high bounds")
        if len(spec.low) != spec.dim or len(spec.high) != spec.dim:
            _spec_fail(
                "bounds",
                f"expected len(low) == len(high) == dim ({spec.dim}), "
                f"got {len(spec.low)} and {len(spec.high)}",
            )
        for i, (lo, hi) in enumerate(zip(spec.low, spec.high, strict=True)):
            if not lo < hi:
                _spec_fail("bounds", f"expected low[{i}] < high[{i}], got {lo} >= {hi}")
    elif spec.kind is ActionKind.DISCRETE:
        if spec.low is not None or spec.high is not None:
            _spec_fail("bounds", "discrete spec must have low and high as None")
        if spec.num_classes is None:
            _spec_fail("num_classes", "discrete spec must provide per-dim num_classes")
        if len(spec.num_classes) != spec.dim:
            _spec_fail(
                "num_classes",
                f"expected len(num_classes) == dim ({spec.dim}), got {len(spec.num_classes)}",
            )
        for i, n in enumerate(spec.num_classes):
            if n < 2:
                _spec_fail("num_classes", f"expected num_classes[{i}] >= 2, got {n}")
    else:  # pragma: no cover - exhaustive over ActionKind
        _spec_fail("kind", f"unknown ActionKind: {spec.kind!r}")


def check_wmcp_join(
    advertised_version: str, participant_version: str = WMCP_VERSION
) -> None:
    """Exact-equality WMCP federation-join gate (``INV-WMCP``, RFC-0007 6).

    Refuses a participant whose installed ``WMCP_VERSION`` differs from the federation's advertised
    version *before* it can contribute to an aggregation — conformance is a precondition for joining.
    Raises :class:`~lensemble.errors.ContractViolation` (code ``WMCP_CONTRACT_VIOLATION``) on mismatch,
    naming both versions and the required lockstep upgrade; no-op return when equal. Fail-closed: a
    non-conforming participant is detected at join, never after it has corrupted an aggregation. v0.1 is
    exact equality on the full SemVer string (no compatible-minor negotiation).
    """
    if advertised_version != participant_version:
        err = ContractViolation(
            f"WMCP version mismatch at federation join: federation advertises "
            f"{advertised_version!r}, participant has {participant_version!r}",
            code=LensembleErrorCode.WMCP_CONTRACT_VIOLATION,
            remediation=(
                "upgrade in lockstep so the participant and the federation run the same WMCP_VERSION "
                f"(federation={advertised_version!r}, participant={participant_version!r})"
            ),
        )
        err.field = "wmcp_version"  # type: ignore[attr-defined]
        raise err
