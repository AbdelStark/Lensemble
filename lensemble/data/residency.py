"""lensemble.data.residency — the single fail-closed egress checkpoint (docs/rfcs/RFC-0004 2).

``guard_egress`` is the one place a boundary-crossing payload is inspected before it leaves a
participant. It enforces ``INV-RESIDENCY``: raw observations, raw actions, and private embeddings
``f_theta(x)`` never cross a trust boundary. A resident tensor on the egress path raises
:class:`~lensemble.errors.ResidencyViolation` — security-critical, fail-closed, never
caught-and-ignored, never downgraded to a warning (04-error-model 1 principle 3, conventions 6).

Permitted to cross (06-security 3):

- a ``PseudoGradient``'s ``delta`` over ``(theta, phi)`` only — never action heads
  (``INV-ACTIONHEAD-LOCAL``);
- a ``DatasetCommitment`` (the root ``R_c`` + counts + WMCP metadata);
- coordination scalars/hashes (sketch seed ``s_t``, probe hash, global-model hash);
- redacted metrics (hashes, L2 norms, shapes, counts, scalars).

A vetted, cross-boundary-safe object opts in by setting a class attribute ``__egress_role__`` to one of
:class:`EgressRole`'s permitted values; ``PseudoGradient`` and ``DatasetCommitment`` do so when they land
(#38, #29). Any unmarked bare tensor, or any known resident object, is rejected by default (fail-closed):
the guard is only as strong as the discipline of routing every egress through it, so it errs toward
refusal.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Final

import torch

from lensemble.contracts import LatentState
from lensemble.data.dataset import EpisodeDataset
from lensemble.data.episode import Episode, Transition, Window
from lensemble.errors import LensembleErrorCode, ResidencyViolation

_PRIMITIVES: Final = (type(None), bool, int, float, complex, str, bytes, bytearray)


class EgressRole:
    """Markers an egress-safe object sets as ``__egress_role__`` (06-security 3)."""

    PSEUDO_GRADIENT: Final = "pseudo_gradient"
    DATASET_COMMITMENT: Final = "dataset_commitment"
    COORDINATION: Final = "coordination"
    REDACTED_METRIC: Final = "redacted_metric"
    ACTION_HEAD: Final = (
        "action_head"  # FORBIDDEN: a per-embodiment head must never cross
    )


_PERMITTED_ROLES: Final = frozenset(
    {
        EgressRole.PSEUDO_GRADIENT,
        EgressRole.DATASET_COMMITMENT,
        EgressRole.COORDINATION,
        EgressRole.REDACTED_METRIC,
    }
)

# Known resident types: their tensors are raw observations/actions or private embeddings.
_RESIDENT_ROLE: Final = {
    LatentState: "private_embedding",
    Transition: "raw_transition",
    Episode: "raw_episode",
    Window: "raw_window",
    EpisodeDataset: "sovereign_dataset",
}


def _violate(role: str, *, boundary: str, dataset_id: str | None) -> ResidencyViolation:
    """Build a ``ResidencyViolation`` carrying only non-tensor context (never tensor data)."""
    err = ResidencyViolation(
        f"resident data ({role}) found on an egress payload; refusing to cross boundary {boundary!r}",
        code=LensembleErrorCode.RESIDENCY_VIOLATION,
        remediation="only a privatized PseudoGradient.delta, a DatasetCommitment, coordination "
        "scalars/hashes, and redacted metrics may cross a trust boundary (INV-RESIDENCY)",
    )
    err.tensor_role = role  # type: ignore[attr-defined]
    err.boundary = boundary  # type: ignore[attr-defined]
    err.dataset_id = dataset_id  # type: ignore[attr-defined]
    return err


def _iter_fields(obj: object) -> list[tuple[str, Any]]:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return [(f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj)]
    inner = getattr(obj, "__dict__", None)
    if isinstance(inner, dict):
        return list(inner.items())
    return []


def _inspect(node: object, *, boundary: str, tensor_ok: bool) -> None:
    if isinstance(node, _PRIMITIVES):
        return
    if isinstance(node, torch.Tensor):
        if tensor_ok:
            return
        raise _violate("raw_tensor_or_embedding", boundary=boundary, dataset_id=None)

    # Known resident objects are refused outright (the guard reads EpisodeDataset.exportable for context).
    for resident_type, role in _RESIDENT_ROLE.items():
        if isinstance(node, resident_type):
            dataset_id = getattr(node, "path", None)
            raise _violate(
                role,
                boundary=boundary,
                dataset_id=None if dataset_id is None else str(dataset_id),
            )

    role = getattr(node, "__egress_role__", None)
    if role == EgressRole.ACTION_HEAD:
        raise _violate("action_head", boundary=boundary, dataset_id=None)

    if isinstance(node, Mapping):
        for value in node.values():
            _inspect(value, boundary=boundary, tensor_ok=False)
        return
    if isinstance(node, (list, tuple, set, frozenset)):
        for item in node:
            _inspect(item, boundary=boundary, tensor_ok=False)
        return

    if role in _PERMITTED_ROLES:
        # A vetted carrier may cross. Only a PseudoGradient's `delta` field may carry a tensor;
        # every other field (including any action-head group) is inspected with tensors forbidden.
        for name, value in _iter_fields(node):
            allow_tensor = role == EgressRole.PSEUDO_GRADIENT and name == "delta"
            _inspect(value, boundary=boundary, tensor_ok=allow_tensor)
        return

    # Unknown object: fail-closed — walk its fields so any nested resident tensor is caught.
    for _name, value in _iter_fields(node):
        _inspect(value, boundary=boundary, tensor_ok=False)


def guard_egress(
    payload: object, *, boundary: str = "participant->coordinator"
) -> None:
    """Inspect an outbound, boundary-crossing payload; raise if it carries resident data.

    Raises :class:`~lensemble.errors.ResidencyViolation` (code ``RESIDENCY_VIOLATION``) on any raw
    observation/action tensor, any private embedding ``f_theta(x)`` (a ``LatentState``), any known
    resident object (``Transition``/``Episode``/``Window``/``EpisodeDataset``), any per-embodiment
    action-head group (``INV-ACTIONHEAD-LOCAL``), or any unmarked bare tensor. Fail-closed and never
    caught-and-ignored. No-op return when the payload is clean.
    """
    _inspect(payload, boundary=boundary, tensor_ok=False)
