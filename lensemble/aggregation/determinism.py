"""lensemble.aggregation.determinism — the per-outer-step aggregation determinism self-check.

Enforces ``INV-AGG-DETERMINISM`` — the outer step is a pure, bitwise-reproducible function of (committed
deltas, round seed, prior global params). This is the first of the five Phase-1 proof-ready disciplines
(RFC-0006 §3): honoring it now lets the Phase-2 aggregation STARK attach with no rework. It is a
security-critical, fail-closed check, **not** a performance optimization.

:func:`assert_outer_step_deterministic` recomputes the outer step a second time on the identical inputs
and compares the two flat results byte-for-byte (``torch.equal`` plus an identical canonical content
hash). On any mismatch it raises :class:`~lensemble.errors.NonDeterministicAggregation`
(``AGG_NONDETERMINISTIC``) carrying ``round``, ``expected_hash``, ``got_hash`` — the step does not commit
and the round aborts (to the runtime ``ABORTED`` state, RFC-0013) for recompute; the error is never
caught-and-ignored. The check takes the recomputation as a thunk so this module does not import the
``federation`` outer optimizer (the dependency points inward only).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lensemble.errors import LensembleErrorCode, NonDeterministicAggregation

if TYPE_CHECKING:
    from collections.abc import Callable


def flat_content_hash(tensor: Tensor) -> str:
    """SHA-256 (64 hex) over the canonical little-endian bytes of a flat result tensor.

    Platform-stable: the tensor is detached, moved to CPU, made contiguous, and viewed little-endian at
    its stored dtype, so the hash is identical on a big-endian or little-endian host.
    """
    array = tensor.detach().cpu().contiguous().numpy()
    little_endian = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return hashlib.sha256(little_endian.tobytes()).hexdigest()


def assert_outer_step_deterministic(
    compute: "Callable[[], Tensor]", *, round_index: int
) -> Tensor:
    """Recompute the outer step on identical inputs and assert byte-for-byte reproducibility.

    ``compute`` MUST be a pure, side-effect-free recomputation of the outer-step result for a single round
    (it is invoked twice); a stateful optimizer must be reconstructed per call so its velocity is not
    advanced twice. The two flat results are compared with ``torch.equal`` **and** an identical
    :func:`flat_content_hash`.

    On any mismatch raises :class:`~lensemble.errors.NonDeterministicAggregation`
    (``AGG_NONDETERMINISTIC``) carrying ``round``, ``expected_hash``, ``got_hash`` and a populated
    ``.remediation``. This is security-critical and fail-closed (``INV-AGG-DETERMINISM``, RFC-0006 §3):
    the step does not commit, the round aborts to ``ABORTED`` for recompute, and the error is **never**
    swallowed. Returns the verified result on success.
    """
    first = compute()
    second = compute()
    expected = flat_content_hash(first)
    got = flat_content_hash(second)
    if expected != got or not torch.equal(first, second):
        err = NonDeterministicAggregation(
            f"outer step at round {round_index} was not bitwise-reproducible; "
            "aborting the round (INV-AGG-DETERMINISM)",
            code=LensembleErrorCode.AGG_NONDETERMINISTIC,
            remediation="use a fixed fp32/fp64 reduction order with no atomics or nondeterministic "
            "kernels; abort the round to ABORTED and recompute",
        )
        err.round = round_index  # type: ignore[attr-defined]
        err.expected_hash = expected  # type: ignore[attr-defined]
        err.got_hash = got  # type: ignore[attr-defined]
        raise err
    return first
