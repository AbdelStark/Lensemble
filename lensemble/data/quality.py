"""lensemble.data.quality — declared data-quality metadata and the WMCP join precondition (RFC-0004 §6).

Each participant publishes a minimal, **declared, non-resident** description of its dataset — modality,
embodiment id, the per-embodiment ``ActionSpec``, episode count, and collection conditions — carried
alongside the ``DatasetCommitment`` and the contribution record. No raw observation/action tensor is part
of this declaration (``INV-RESIDENCY``); the metadata is *declared*, not verified — provenance proves
origin, not data quality.

Conformance to the pinned ``wmcp_version`` is the **precondition for joining a federation**
(``INV-WMCP``, RFC-0007 §6): :func:`validate_join_precondition` validates the declared ``ActionSpec``
(``validate_action_spec``) and gates the version on exact SemVer equality (``check_wmcp_join``), raising
``ContractViolation`` on a nonconforming spec or version. It is a hard reject on a boundary path and is
never caught-and-ignored. Quality *enforcement* beyond this declaration (weighting/gating) is an Open
Question deferred past v0.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lensemble.contracts import (
    WMCP_VERSION,
    ActionSpec,
    check_wmcp_join,
    validate_action_spec,
)
from lensemble.errors import ContractViolation, LensembleErrorCode

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class DataQualityMetadata:
    """A participant's declared, non-resident dataset description (RFC-0004 §6).

    All fields are scalar metadata or the (scalar) ``ActionSpec`` — no raw observation/action tensor is
    carried (``INV-RESIDENCY``). ``collection_conditions`` are declared, non-private conditions (e.g.
    ``{"site": "lab-a", "fps": "30"}``), never raw data.
    """

    modality: str  # e.g. "rgb-video"
    embodiment_id: str  # must equal action_spec.embodiment_id
    action_spec: ActionSpec  # the per-embodiment action contract (RFC-0007 §3)
    episode_count: int  # number of committed episodes (>= 1)
    collection_conditions: "Mapping[str, str]"  # declared, non-private (RFC-0004 §6)


def _reject(field: str, message: str, remediation: str) -> ContractViolation:
    err = ContractViolation(
        message,
        code=LensembleErrorCode.WMCP_CONTRACT_VIOLATION,
        remediation=remediation,
    )
    err.field = field  # type: ignore[attr-defined]
    return err


def validate_join_precondition(
    metadata: DataQualityMetadata,
    *,
    federation_wmcp_version: str = WMCP_VERSION,
) -> None:
    """Validate a participant's declared metadata as the federation-join precondition (``INV-WMCP``).

    Checks, in order: the declared ``ActionSpec`` is valid (:func:`validate_action_spec`); the
    participant's ``wmcp_version`` equals the federation's advertised version (:func:`check_wmcp_join`,
    exact SemVer equality for v0.1); the declared ``embodiment_id`` matches the ``ActionSpec``'s; and the
    episode count is ``>= 1``. Raises :class:`~lensemble.errors.ContractViolation`
    (``WMCP_CONTRACT_VIOLATION``) on the first failing clause — a hard reject, never swallowed; no-op
    return on success.
    """
    validate_action_spec(metadata.action_spec)
    check_wmcp_join(federation_wmcp_version, metadata.action_spec.wmcp_version)
    if metadata.embodiment_id != metadata.action_spec.embodiment_id:
        raise _reject(
            "embodiment_id",
            f"declared embodiment_id {metadata.embodiment_id!r} does not match the ActionSpec's "
            f"{metadata.action_spec.embodiment_id!r}",
            "declare the embodiment_id the ActionSpec describes",
        )
    if metadata.episode_count < 1:
        raise _reject(
            "episode_count",
            f"declared episode_count {metadata.episode_count} < 1; an empty dataset cannot join",
            "commit at least one episode before declaring a join",
        )
