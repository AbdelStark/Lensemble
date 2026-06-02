"""lensemble.provenance.ledger — the append-only, hash-chained contribution ledger (RFC-0014 §7).

The ledger is the audit substrate (RFC-0004 §5): one :class:`ContributionRecord` per round, recording the
contributing participants, the dataset roots ``R_c`` they were bound to (``INV-COMMIT-BINDING``), and the
resulting global-model content hash (``INV-CHECKPOINT-HASH``, owned by RFC-0010 — the ledger records it,
it does not define it).

Each record carries ``prev_record_hash``, the content hash of the prior record, so the log is a hash
chain: rewriting any past round changes its hash and breaks every subsequent link. In **Phase 1 this is
tamper-evidence** — :meth:`ContributionLedger.verify_chain` detects a break; the chain plus the per-round
commitments are what the Phase-2 layer upgrades to a tamper-*proof* statement (RFC-0006 §2). The
provenance boundary still holds (RFC-0014 §8): the ledger proves data **origin**, not data quality and
not honest computation.

The on-disk store is JSON Lines — one canonical ``ContributionRecord`` per line. The content hash is
domain-separated and taken over the canonical (sorted-key) JSON form, so it is independent of the on-disk
key order and stable across platforms.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from lensemble.errors import (
    LensembleErrorCode,
    MerkleVerificationError,
    ProvenanceError,
    SchemaVersionMismatch,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

LEDGER_SCHEMA_VERSION = 1
_RECORD_DOMAIN = b"lensemble/provenance/ledger-record/v1\x00"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ContributionRecord(BaseModel):
    """One append-only ledger entry per round (03 §13.3 / RFC-0014 §7).

    On-disk metadata: pydantic v2, ``frozen``, ``extra="forbid"``, integer ``schema_version``. Hash
    fields are lowercase-hex strings (``global_model_hash`` is the checkpoint ``content_hash``;
    ``prev_record_hash`` is the prior record's content hash, ``None`` at round 0). The record is **not**
    hex-validated at construction so a hand-corrupted on-disk digest is caught by ``verify_chain`` as a
    structural corruption rather than silently rejected at load.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=LEDGER_SCHEMA_VERSION, ge=1)
    round_index: int = Field(ge=0)  # the outer round t
    participants: tuple[str, ...]  # contributing participant ids this round
    dataset_roots: dict[str, str]  # participant_id -> R_c (hex) bound this round
    global_model_hash: str  # resulting (theta_{t+1}, phi_{t+1}) content_hash (hex)
    prev_record_hash: str | None = None  # the append-only chain link (hex)


def parse_contribution_record(data: str | dict[str, Any]) -> ContributionRecord:
    """Load a ``ContributionRecord`` from JSON, gating ``schema_version`` first (fail-closed).

    Raises :class:`~lensemble.errors.SchemaVersionMismatch` on a missing/non-integer/too-new version
    before field validation; a structurally invalid line raises
    :class:`~lensemble.errors.MerkleVerificationError`.
    """
    try:
        raw = json.loads(data) if isinstance(data, str) else dict(data)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise MerkleVerificationError(
            f"ledger line is not valid JSON: {exc}",
            code=LensembleErrorCode.MERKLE_VERIFY_FAILED,
            remediation="the ledger file is structurally corrupt; restore it from a trusted copy",
        ) from exc
    version = raw.get("schema_version")
    if not isinstance(version, int) or version > LEDGER_SCHEMA_VERSION:
        raise SchemaVersionMismatch(
            f"ContributionRecord schema_version {version!r} exceeds reader max {LEDGER_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation="upgrade lensemble to read this ledger, or re-emit at the supported schema",
        )
    try:
        return ContributionRecord.model_validate(raw)
    except (
        Exception
    ) as exc:  # pydantic ValidationError -> typed structural corruption (fail-closed)
        raise MerkleVerificationError(
            f"ledger record failed validation: {exc}",
            code=LensembleErrorCode.MERKLE_VERIFY_FAILED,
            remediation="the ledger file is structurally corrupt; restore it from a trusted copy",
        ) from exc


def _record_hash(record: ContributionRecord) -> bytes:
    """Domain-separated SHA-256 over the record's canonical (sorted-key) JSON -> 32 bytes.

    Independent of the on-disk key order, so re-reading a record and re-hashing reproduces the same chain
    link cross-platform.
    """
    payload = json.dumps(
        record.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(_RECORD_DOMAIN + payload).digest()


class ContributionLedger:
    """Append-only, hash-chained log of :class:`ContributionRecord` entries (RFC-0014 §7).

    Backed by a JSON Lines file. :meth:`append` enforces the append-only invariant (strictly increasing
    ``round_index`` and a ``prev_record_hash`` that chains to the tail); :meth:`verify_chain` re-validates
    the whole chain. Phase-1 tamper-evidence: a rewritten past record breaks every subsequent link.
    """

    def __init__(self, path: Path, records: Sequence[ContributionRecord]) -> None:
        self._path = Path(path)
        self._records: list[ContributionRecord] = list(records)

    @classmethod
    def open(cls, path: str | Path) -> ContributionLedger:
        """Open (or initialize) a JSONL ledger at ``path``, loading and parsing any existing records.

        A missing file is an empty ledger. A structurally corrupt line raises
        :class:`~lensemble.errors.MerkleVerificationError`; a too-new ``schema_version`` raises
        :class:`~lensemble.errors.SchemaVersionMismatch`.
        """
        path = Path(path)
        records: list[ContributionRecord] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    records.append(parse_contribution_record(line))
        return cls(path, records)

    @property
    def records(self) -> tuple[ContributionRecord, ...]:
        """The records appended so far, in order (read-only snapshot)."""
        return tuple(self._records)

    def append(self, record: ContributionRecord) -> bytes:
        """Chain ``record`` to the tail, persist it, and return its 32-byte content hash.

        Sets the stored record's ``prev_record_hash`` to the current tail's content hash. Raises
        :class:`~lensemble.errors.ProvenanceError` (``PROVENANCE_FAILED``) if ``round_index`` is not
        strictly greater than the tail's, or if the supplied ``prev_record_hash`` is non-``None`` and does
        not match the tail (a caller asserting the wrong link). The first record must chain to ``None``.
        """
        expected_prev = _record_hash(self._records[-1]).hex() if self._records else None
        if (
            record.prev_record_hash is not None
            and record.prev_record_hash != expected_prev
        ):
            raise ProvenanceError(
                "supplied prev_record_hash does not chain to the ledger tail",
                code=LensembleErrorCode.PROVENANCE_FAILED,
                remediation="leave prev_record_hash unset or set it to the current tail's content hash",
            )
        if self._records and record.round_index <= self._records[-1].round_index:
            raise ProvenanceError(
                f"round_index {record.round_index} must be strictly greater than the tail's "
                f"{self._records[-1].round_index} (append-only / monotone)",
                code=LensembleErrorCode.PROVENANCE_FAILED,
                remediation="append rounds in strictly increasing order; the ledger is append-only",
            )
        stored = record.model_copy(update={"prev_record_hash": expected_prev})
        digest = _record_hash(stored)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(stored.model_dump_json() + "\n")
        self._records.append(stored)
        return digest

    def verify_chain(self) -> bool:
        """Walk the chain: each ``prev_record_hash`` equals the prior record's content hash and
        ``round_index`` strictly increases. Returns ``False`` (not raise) on any break.

        A *structurally corrupt* record — a ``prev_record_hash`` or ``global_model_hash`` that is not a
        64-char lowercase-hex digest — raises :class:`~lensemble.errors.MerkleVerificationError` (the
        chain cannot be meaningfully evaluated), distinct from a well-formed-but-broken link's ``False``.
        """
        expected_prev: str | None = None
        for i, rec in enumerate(self._records):
            for digest in (rec.prev_record_hash, rec.global_model_hash):
                if digest is not None and not _HEX64.fullmatch(digest):
                    raise MerkleVerificationError(
                        f"record {i} carries a malformed digest {digest!r}",
                        code=LensembleErrorCode.MERKLE_VERIFY_FAILED,
                        remediation="the ledger is structurally corrupt; restore it from a trusted copy",
                    )
            if rec.prev_record_hash != expected_prev:
                return False
            if i > 0 and rec.round_index <= self._records[i - 1].round_index:
                return False
            expected_prev = _record_hash(rec).hex()
        return True
