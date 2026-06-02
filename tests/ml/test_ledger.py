"""Append-only, hash-chained contribution ledger (RFC-0014 §7; #30).

Exact-byte tests, no numerical tolerance: the append-only / monotone invariant, hash-chain integrity
(intact -> True, rewritten past record -> False), the structural-corruption raise vs broken-link False
distinction, and the schema round-trip / version gate. Tests live under `tests/ml/` so the CI gate runs
them (the issue's illustrative `tests/provenance/` path is collected by no gate).
"""

from __future__ import annotations

import json

import pytest

from lensemble.errors import (
    MerkleVerificationError,
    ProvenanceError,
    SchemaVersionMismatch,
)
from lensemble.provenance.ledger import (
    ContributionLedger,
    ContributionRecord,
    _record_hash,
    parse_contribution_record,
)

_ROOT_A = "a" * 64
_ROOT_B = "b" * 64
_MODEL_HASH = "c" * 64


def _record(round_index: int, participant: str, root: str) -> ContributionRecord:
    return ContributionRecord(
        round_index=round_index,
        participants=(participant,),
        dataset_roots={participant: root},
        global_model_hash=_MODEL_HASH,
    )


def _ledger_path(tmp_path):
    return tmp_path / "contributions.jsonl"


# --- append + persistence + chain links ---


def test_append_chains_and_persists(tmp_path) -> None:
    led = ContributionLedger.open(_ledger_path(tmp_path))
    h0 = led.append(_record(0, "p0", _ROOT_A))
    h1 = led.append(_record(1, "p1", _ROOT_B))
    assert isinstance(h0, bytes) and len(h0) == 32
    assert led.records[0].prev_record_hash is None
    assert led.records[1].prev_record_hash == h0.hex()
    assert h0 != h1
    # Persisted: a fresh open sees both records and an intact chain.
    reopened = ContributionLedger.open(_ledger_path(tmp_path))
    assert len(reopened.records) == 2
    assert reopened.verify_chain() is True


def test_append_fills_prev_hash_when_unset(tmp_path) -> None:
    led = ContributionLedger.open(_ledger_path(tmp_path))
    led.append(_record(0, "p0", _ROOT_A))
    tail_hash = _record_hash(led.records[0]).hex()
    led.append(_record(5, "p1", _ROOT_B))  # prev unset -> filled to the tail hash
    assert led.records[1].prev_record_hash == tail_hash


# --- append-only invariant ---


def test_append_rejects_non_monotone_round_index(tmp_path) -> None:
    led = ContributionLedger.open(_ledger_path(tmp_path))
    led.append(_record(3, "p0", _ROOT_A))
    with pytest.raises(ProvenanceError):
        led.append(_record(3, "p1", _ROOT_B))  # not strictly greater
    with pytest.raises(ProvenanceError):
        led.append(_record(2, "p2", _ROOT_B))  # lower


def test_append_rejects_wrong_supplied_prev_hash(tmp_path) -> None:
    led = ContributionLedger.open(_ledger_path(tmp_path))
    led.append(_record(0, "p0", _ROOT_A))
    bad = ContributionRecord(
        round_index=1,
        participants=("p1",),
        dataset_roots={"p1": _ROOT_B},
        global_model_hash=_MODEL_HASH,
        prev_record_hash="f" * 64,  # asserts the wrong link
    )
    with pytest.raises(ProvenanceError):
        led.append(bad)


def test_first_record_with_nonnull_prev_is_rejected(tmp_path) -> None:
    led = ContributionLedger.open(_ledger_path(tmp_path))
    first = ContributionRecord(
        round_index=0,
        participants=("p0",),
        dataset_roots={"p0": _ROOT_A},
        global_model_hash=_MODEL_HASH,
        prev_record_hash="d" * 64,  # there is no tail to chain to
    )
    with pytest.raises(ProvenanceError):
        led.append(first)


# --- chain integrity: intact True, rewritten past record False ---


def test_verify_chain_true_on_intact(tmp_path) -> None:
    led = ContributionLedger.open(_ledger_path(tmp_path))
    for i in range(4):
        led.append(_record(i, f"p{i}", _ROOT_A))
    assert led.verify_chain() is True


def test_verify_chain_false_on_rewritten_past_record(tmp_path) -> None:
    path = _ledger_path(tmp_path)
    led = ContributionLedger.open(path)
    led.append(_record(0, "p0", _ROOT_A))
    led.append(_record(1, "p1", _ROOT_B))
    led.append(_record(2, "p2", _ROOT_A))
    # Tamper the first on-disk record (change a dataset root) but keep its prev/chain fields.
    lines = path.read_text().splitlines()
    rec0 = json.loads(lines[0])
    rec0["dataset_roots"] = {"p0": _ROOT_B}  # rewrite history
    lines[0] = json.dumps(rec0)
    path.write_text("\n".join(lines) + "\n")
    reopened = ContributionLedger.open(path)
    # rec0's recomputed hash now differs from rec1.prev_record_hash -> broken link.
    assert reopened.verify_chain() is False


def test_verify_chain_false_on_non_monotone_on_disk(tmp_path) -> None:
    # A correctly-chained pair whose round_index does not increase: a clean False, not a raise.
    rec0 = _record(5, "p0", _ROOT_A)
    h0 = _record_hash(rec0).hex()
    rec1 = ContributionRecord(
        round_index=5,  # equal -> not strictly increasing
        participants=("p1",),
        dataset_roots={"p1": _ROOT_B},
        global_model_hash=_MODEL_HASH,
        prev_record_hash=h0,
    )
    path = _ledger_path(tmp_path)
    path.write_text(rec0.model_dump_json() + "\n" + rec1.model_dump_json() + "\n")
    assert ContributionLedger.open(path).verify_chain() is False


# --- structural corruption raises (distinct from a broken-link False) ---


def test_verify_chain_raises_on_malformed_digest(tmp_path) -> None:
    rec = ContributionRecord(
        round_index=0,
        participants=("p0",),
        dataset_roots={"p0": _ROOT_A},
        global_model_hash="not-a-valid-hash",  # structurally corrupt digest
    )
    path = _ledger_path(tmp_path)
    path.write_text(rec.model_dump_json() + "\n")
    with pytest.raises(MerkleVerificationError):
        ContributionLedger.open(path).verify_chain()


def test_open_raises_on_corrupt_json_line(tmp_path) -> None:
    path = _ledger_path(tmp_path)
    path.write_text("{not valid json\n")
    with pytest.raises(MerkleVerificationError):
        ContributionLedger.open(path)


def test_open_raises_on_valid_json_that_fails_validation(tmp_path) -> None:
    # Parses as JSON, version ok, but missing required fields -> structural corruption (fail-closed).
    path = _ledger_path(tmp_path)
    path.write_text(json.dumps({"schema_version": 1, "round_index": 0}) + "\n")
    with pytest.raises(MerkleVerificationError):
        ContributionLedger.open(path)


def test_open_skips_blank_lines(tmp_path) -> None:
    rec = _record(0, "p0", _ROOT_A)
    path = _ledger_path(tmp_path)
    path.write_text("\n" + rec.model_dump_json() + "\n\n")  # leading/trailing blanks
    led = ContributionLedger.open(path)
    assert len(led.records) == 1
    assert led.verify_chain() is True


# --- schema round-trip + version gate ---


def test_record_round_trips_through_json() -> None:
    rec = _record(7, "p7", _ROOT_A)
    again = parse_contribution_record(rec.model_dump_json())
    assert again == rec
    assert _record_hash(again) == _record_hash(rec)  # exact byte equality


def test_parse_rejects_future_and_noninteger_schema() -> None:
    raw = json.loads(_record(0, "p0", _ROOT_A).model_dump_json())
    with pytest.raises(SchemaVersionMismatch):
        parse_contribution_record({**raw, "schema_version": 99})
    with pytest.raises(SchemaVersionMismatch):
        parse_contribution_record({**raw, "schema_version": "1"})


def test_open_missing_file_is_empty_ledger(tmp_path) -> None:
    led = ContributionLedger.open(_ledger_path(tmp_path))
    assert led.records == ()
    assert led.verify_chain() is True
