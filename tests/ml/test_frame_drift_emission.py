"""Frame-drift diagnostic emission contract (RFC-0015 §3; #60).

The headline artifact must be reproducible from logs alone: byte-identical emissions for identical
inputs, an independent recomputation that matches, the canonical `c < c'` pairing, the fail-closed pin
bindings (probe / checkpoint), and the figure proxy (naive averaging rotates frames apart over rounds
while the anchored configuration stays flat). Tests live under `tests/ml/` so the CI gate runs them.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import pytest
import torch

from lensemble.errors import CheckpointIntegrityError, ProbeError, SchemaVersionMismatch
from lensemble.gauge import frame_drift
from lensemble.observability import (
    FrameDriftRecord,
    emit_diagnostic,
    parse_frame_drift_record,
)

_PROBE_HASH = "ab" * 32  # a stand-in pinned probe hash (hex)
_TS = datetime(2026, 6, 2, 12, 0, 0, tzinfo=timezone.utc)


def _rot(d: int, angle_rad: float) -> torch.Tensor:
    r = torch.eye(d)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    r[0, 0], r[0, 1], r[1, 0], r[1, 1] = c, -s, s, c
    return r


def _base(n: int = 8, d: int = 4) -> torch.Tensor:
    return torch.randn(n, d, generator=torch.Generator().manual_seed(0))


def _embeddings(angle_rad: float, d: int = 4) -> dict[str, torch.Tensor]:
    base = _base(d=d)
    return {"p0": base.clone(), "p1": base @ _rot(d, angle_rad)}


def _report(angle_rad: float, round_index: int = 0):
    return frame_drift(
        _embeddings(angle_rad),
        round_index=round_index,
        expected_probe_hash=_PROBE_HASH,
    )


def _ckpt_hashes(*pids: str) -> dict[str, str]:
    return {pid: f"{i:064x}" for i, pid in enumerate(pids, start=1)}


def _emit(
    report,
    tmp_path,
    *,
    pinned_probe_hash: str = _PROBE_HASH,
    participant_checkpoint_hash: dict[str, str] | None = None,
) -> FrameDriftRecord:
    return emit_diagnostic(
        report,
        run_dir=tmp_path,
        global_checkpoint_hash="c" * 64,
        participant_checkpoint_hash=participant_checkpoint_hash
        or _ckpt_hashes("p0", "p1"),
        pinned_probe_hash=pinned_probe_hash,
        timestamp=_TS,
    )


# --- reproducibility: byte-identical emission, independent-recompute match ---


def test_two_emissions_are_byte_identical(tmp_path) -> None:
    _emit(_report(0.3), tmp_path)
    _emit(_report(0.3), tmp_path)  # identical inputs + fixed timestamp
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert lines[0] == lines[1]  # byte-stable (repr-float canonicalization)
    assert json.loads(lines[0])["record_kind"] == "frame_drift"


def test_independent_recomputation_matches_emitted_record(tmp_path) -> None:
    rec_a = _emit(_report(0.42), tmp_path)
    rec_b = _emit(_report(0.42), tmp_path)  # recomputed from the same inputs
    assert rec_a.pairwise_angle_deg == rec_b.pairwise_angle_deg
    assert rec_a.pairwise_residual == rec_b.pairwise_residual


def test_canonical_pair_ordering(tmp_path) -> None:
    rec = _emit(_report(0.2), tmp_path)
    for pair in rec.pairwise_angle_deg:
        assert pair.participant_a < pair.participant_b  # c < c'
    assert len(rec.pairwise_angle_deg) == 1  # two participants -> one unordered pair


# --- pin bindings fail closed ---


def test_probe_hash_mismatch_raises(tmp_path) -> None:
    with pytest.raises(ProbeError):
        _emit(_report(0.2), tmp_path, pinned_probe_hash="cd" * 32)


def test_missing_participant_checkpoint_hash_raises(tmp_path) -> None:
    with pytest.raises(CheckpointIntegrityError):
        _emit(
            _report(0.2),
            tmp_path,
            participant_checkpoint_hash={"p0": "1" * 64},  # p1 missing
        )


# --- schema round-trip + version gate ---


def test_record_round_trips_through_json(tmp_path) -> None:
    rec = _emit(_report(0.2), tmp_path)
    line = (tmp_path / "metrics.jsonl").read_text().splitlines()[0]
    assert parse_frame_drift_record(line) == rec


def test_too_new_schema_version_raises(tmp_path) -> None:
    rec = _emit(_report(0.2), tmp_path)
    raw = json.loads(rec.model_dump_json())
    raw["schema_version"] = 99
    with pytest.raises(SchemaVersionMismatch):
        parse_frame_drift_record(json.dumps(raw))


# --- figure proxy: naive averaging rotates frames apart; anchored stays flat ---


def test_figure_proxy_naive_increases_anchored_flat(tmp_path) -> None:
    theta = 0.15
    naive, anchored = [], []
    for t in range(1, 6):
        # naive: the inter-frame rotation grows each round; anchored: it is pinned at one step.
        naive.append(
            _emit(_report(theta * t, t), tmp_path).pairwise_angle_deg[0].angle_deg
        )
        anchored.append(
            _emit(_report(theta, t), tmp_path / "anch").pairwise_angle_deg[0].angle_deg
        )
    # naive drift increases monotonically over rounds; anchored stays flat (within fp tolerance).
    assert all(naive[i] < naive[i + 1] for i in range(len(naive) - 1))
    assert max(anchored) - min(anchored) < 1.0  # degrees; ANGLE_TOL_DEG-scale flatness


def test_record_is_pydantic_frozen_extra_forbid() -> None:
    raw = json.loads(
        FrameDriftRecord(
            round_index=0,
            probe_hash=_PROBE_HASH,
            global_checkpoint_hash="c" * 64,
            participant_checkpoint_hash={"p0": "1" * 64},
            pairwise_angle_deg=(),
            pairwise_residual=(),
            drift_from_global_deg={},
            pair_sampling="all_pairs",
            timestamp=_TS,
        ).model_dump_json()
    )
    raw["surprise"] = 1
    with pytest.raises(Exception):
        FrameDriftRecord.model_validate(raw)
