"""RunManifest schema, canonical config_hash, serialize/load (RFC-0009 6/7). Issue #36.

T4: canonical-JSON round-trip equality, extra="forbid", and a too-new schema_version raise.
T9: a semantic config field changes config_hash; a non-semantic one (a log path) does not.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from lensemble.config import load
from lensemble.config.manifest import (
    MANIFEST_SCHEMA_VERSION,
    RunManifest,
    build_manifest,
    config_hash,
    load_manifest,
    write_manifest,
)
from lensemble.errors import ConfigError, SchemaVersionMismatch

_FIXED = datetime(2026, 6, 2, 12, 0, 0, tzinfo=timezone.utc)


def _manifest() -> RunManifest:
    return build_manifest(load(), run_mode="train_local", created_at=_FIXED)


def test_manifest_round_trips_through_canonical_json(tmp_path: Path) -> None:
    manifest = _manifest()
    path = write_manifest(manifest, tmp_path / "run_manifest.json")
    assert (
        load_manifest(path) == manifest
    )  # field-for-field equal after a JSON round-trip


def test_extra_field_is_forbidden() -> None:
    payload = _manifest().model_dump()
    payload["smuggled_secret"] = "x"
    with pytest.raises(ValidationError):
        RunManifest.model_validate(payload)  # extra="forbid"


def test_schema_version_above_reader_max_raises(tmp_path: Path) -> None:
    raw = json.loads(write_manifest(_manifest(), tmp_path / "m.json").read_text())
    raw["schema_version"] = MANIFEST_SCHEMA_VERSION + 1
    future = tmp_path / "future.json"
    future.write_text(json.dumps(raw))
    with pytest.raises(SchemaVersionMismatch):
        load_manifest(future)


def test_tampered_config_resolved_fails_to_reproduce_hash(tmp_path: Path) -> None:
    raw = json.loads(write_manifest(_manifest(), tmp_path / "m.json").read_text())
    raw["config_resolved"]["federation"]["num_rounds"] += (
        1  # alter the tree, keep the old hash
    )
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(raw))
    with pytest.raises(ConfigError):
        load_manifest(tampered)


def test_semantic_field_changes_hash_non_semantic_does_not() -> None:
    from dataclasses import asdict

    resolved = json.loads(json.dumps(asdict(load())))
    base = config_hash(resolved)

    semantic = copy.deepcopy(resolved)
    semantic["federation"]["participant_count"] += 1
    assert config_hash(semantic) != base  # a semantic field moves the hash

    non_semantic = copy.deepcopy(resolved)
    non_semantic["observability"]["log_path"] = "somewhere/else.jsonl"
    non_semantic["observability"]["metrics_path"] = "other/metrics.jsonl"
    assert config_hash(non_semantic) == base  # output sinks are excluded (RFC-0009 7)


def test_config_hash_algo_is_recorded() -> None:
    assert _manifest().env["config_hash_algo"] == "sha256-canon-v1"
