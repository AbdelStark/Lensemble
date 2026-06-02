"""lensemble.provenance.commit — DatasetCommitment, commit_dataset, and the Δ_c↔R_c binding.

Implements the dataset commitment and the binding check that ties every released pseudo-gradient
``Δ_c`` to exactly one dataset root ``R_c`` (``INV-COMMIT-BINDING``), per RFC-0014 §4, §6.

`commit_dataset` is a **pure function of the episode set + the commitment scheme**: it hashes every
episode (``episode_leaf_hash``), re-hashes each under the ``LEAF`` domain, builds the Merkle tree
(``merkle_root``), and records the root, episode count, and WMCP metadata. No RNG, no I/O-ordering
dependence (leaves are sorted), so two honest builds over the same logical episodes on any platform
produce the byte-identical ``merkle_root`` (the determinism postcondition; only ``created_at`` varies).
The full episodes never leave the boundary (``INV-RESIDENCY``) — the participant publishes only the
32-byte root.

`verify_binding` is the fail-closed binding check. ``CommitmentMismatch`` is **security-critical** and
is **never** caught-and-ignored (04 §Error Model): a mismatched delta is rejected and excluded from the
sum, so a participant cannot launder an unattributed delta into the global model. This module proves
**origin**, not data quality and not honest computation (RFC-0014 §8).

Encoding boundary: the on-disk ``DatasetCommitment.merkle_root`` is 64-hex SHA-256 (03 §9), while
``verify_binding`` operates on the raw 32-byte digest carried by ``PseudoGradient.dataset_root``; the
hex/bytes boundary is kept explicit.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lensemble.errors import (
    CommitmentMismatch,
    LensembleErrorCode,
    MerkleVerificationError,
    ProvenanceError,
    SchemaVersionMismatch,
)
from lensemble.provenance.merkle import (
    COMMITMENT_HASH,
    CommitmentScheme,
    HashDomain,
    _h,
    episode_leaf_hash,
    merkle_root,
)

if TYPE_CHECKING:
    from lensemble.data.dataset import EpisodeDataset

DATASET_COMMITMENT_SCHEMA_VERSION = 1
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class DatasetCommitment(BaseModel):
    """Tamper-evident commitment binding a participant's dataset to a Merkle root ``R_c`` (03 §9).

    On-disk metadata: pydantic v2 JSON, ``frozen``, ``extra="forbid"``, with an explicit integer
    ``schema_version``. ``merkle_root`` is ``R_c`` as 64-char lowercase hex (Phase-1 SHA-256). Validation
    rules: ``len(merkle_root) == 64`` and hex; ``episode_count >= 1``; ``hash_algorithm`` is the pinned
    algorithm. Load via :func:`parse_dataset_commitment` so an unknown/too-new ``schema_version`` raises
    :class:`~lensemble.errors.SchemaVersionMismatch`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=DATASET_COMMITMENT_SCHEMA_VERSION, ge=1)
    merkle_root: str  # R_c as 64-char lowercase-hex SHA-256 (Phase 1)
    episode_count: int = Field(ge=1)  # number of committed leaves
    hash_algorithm: Literal["sha256"]  # Phase-1 canonical commitment hash
    wmcp_version: str  # the latent-contract version the data targets
    embodiment_ids: tuple[str, ...]  # declared embodiments present in the dataset
    created_at: datetime  # commitment time (UTC, RFC 3339)

    @field_validator("merkle_root")
    @classmethod
    def _root_is_hex64(cls, v: str) -> str:
        if not _HEX64.fullmatch(v):
            raise ValueError(
                f"merkle_root must be 64-char lowercase hex SHA-256, got {v!r}"
            )
        return v


def parse_dataset_commitment(data: str | dict[str, Any]) -> DatasetCommitment:
    """Load a ``DatasetCommitment`` from JSON, gating ``schema_version`` first (fail-closed).

    Raises :class:`~lensemble.errors.SchemaVersionMismatch` on a missing/non-integer/too-new version
    before any field validation, mirroring the artifact-header and metric loaders (RFC-0010 §7).
    """
    raw = json.loads(data) if isinstance(data, str) else dict(data)
    version = raw.get("schema_version")
    if not isinstance(version, int) or version > DATASET_COMMITMENT_SCHEMA_VERSION:
        raise SchemaVersionMismatch(
            f"DatasetCommitment schema_version {version!r} exceeds reader max "
            f"{DATASET_COMMITMENT_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation="upgrade lensemble to read this commitment, or re-emit at the supported schema",
        )
    return DatasetCommitment.model_validate(raw)


def commit_dataset(dataset: EpisodeDataset) -> DatasetCommitment:
    """Build the Merkle commitment ``R_c`` for a participant-local dataset (RFC-0014 §4).

    Pure function of the episode set + the :class:`CommitmentScheme`; deterministic ``merkle_root`` across
    platforms (leaves are sorted, no RNG). An empty dataset raises
    :class:`~lensemble.errors.ProvenanceError` (a dataset with zero episodes cannot commit). A dataset
    whose episodes declare conflicting ``wmcp_version`` values raises ``ProvenanceError`` rather than
    silently recording one (prefer an explicit failure over a coerced commitment).
    """
    episodes = tuple(dataset.episodes)
    leaves = [_h(HashDomain.LEAF, episode_leaf_hash(ep)) for ep in episodes]
    root = merkle_root(leaves)  # raises ProvenanceError on an empty dataset

    wmcp_versions = {ep.action_spec.wmcp_version for ep in episodes}
    if len(wmcp_versions) != 1:
        raise ProvenanceError(
            f"dataset spans conflicting wmcp_versions {sorted(wmcp_versions)}; cannot commit one root",
            code=LensembleErrorCode.PROVENANCE_FAILED,
            remediation="commit episodes that conform to a single pinned wmcp_version",
        )
    embodiment_ids = tuple(sorted({ep.embodiment_id for ep in episodes}))

    return DatasetCommitment(
        schema_version=DATASET_COMMITMENT_SCHEMA_VERSION,
        merkle_root=root.hex(),
        episode_count=len(episodes),
        hash_algorithm=COMMITMENT_HASH,
        wmcp_version=next(iter(wmcp_versions)),
        embodiment_ids=embodiment_ids,
        created_at=datetime.now(timezone.utc),
    )


def verify_binding(
    committed_root: bytes, declared_root: bytes, scheme: CommitmentScheme
) -> None:
    """Raise unless a released delta's ``declared_root`` matches the participant's ``committed_root``.

    A pure binding check with no dataset access (membership is the Phase-2 inclusion-proof job). A
    declared root of the wrong length raises :class:`~lensemble.errors.MerkleVerificationError`
    (``MERKLE_VERIFY_FAILED``); a well-formed root that does not equal the committed one raises
    :class:`~lensemble.errors.CommitmentMismatch` (``COMMITMENT_MISMATCH``) — security-critical, rejected,
    excluded from the sum, **never** swallowed (``INV-COMMIT-BINDING``); a match returns ``None``.
    """
    if len(declared_root) != scheme.digest_size:
        raise MerkleVerificationError(
            f"declared dataset_root is {len(declared_root)} bytes, expected {scheme.digest_size}",
            code=LensembleErrorCode.MERKLE_VERIFY_FAILED,
            remediation="reject the Update/Commitment; the root is not a valid digest under the scheme",
        )
    if declared_root != committed_root:
        raise CommitmentMismatch(
            "released delta's dataset_root does not match the participant's committed R_c",
            code=LensembleErrorCode.COMMITMENT_MISMATCH,
            remediation="reject the update and exclude it from the sum (INV-COMMIT-BINDING); never swallow",
        )
