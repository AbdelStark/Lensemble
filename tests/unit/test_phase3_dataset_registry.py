"""Phase 3 dataset/public-probe registry contract (#225)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lensemble.config import default_phase3_consortium_manifest
from lensemble.data import (
    PHASE3_DATASET_REGISTRY_SCHEMA_VERSION,
    load_phase3_dataset_registry,
    parse_phase3_dataset_registry,
    phase3_registry_from_consortium_manifest,
    to_phase3_dataset_registry_json,
    validate_phase3_registry_against_manifest,
    write_phase3_dataset_registry,
)
from lensemble.errors import ConfigError, SchemaVersionMismatch


def _raw() -> dict:
    manifest = default_phase3_consortium_manifest()
    return phase3_registry_from_consortium_manifest(manifest).model_dump(mode="json")


def test_default_phase3_dataset_registry_is_valid_against_manifest() -> None:
    manifest = default_phase3_consortium_manifest()
    registry = phase3_registry_from_consortium_manifest(manifest)

    assert registry.schema_version == PHASE3_DATASET_REGISTRY_SCHEMA_VERSION
    assert len(registry.participants) == 4
    assert registry.run_mode == "public_example"
    assert registry.public_probe == manifest.public_probe
    assert {
        participant.publication_status for participant in registry.participants
    } == {"placeholder"}
    assert all(participant.publication_blocker for participant in registry.participants)

    validate_phase3_registry_against_manifest(registry, manifest)


def test_phase3_dataset_registry_round_trips_canonical_json(tmp_path: Path) -> None:
    registry = phase3_registry_from_consortium_manifest(
        default_phase3_consortium_manifest()
    )
    path = write_phase3_dataset_registry(registry, tmp_path / "registry.json")

    assert load_phase3_dataset_registry(path) == registry
    assert json.loads(to_phase3_dataset_registry_json(registry)) == json.loads(
        path.read_text()
    )


def test_parse_phase3_dataset_registry_gates_future_schema_first() -> None:
    raw = _raw()
    raw["schema_version"] = PHASE3_DATASET_REGISTRY_SCHEMA_VERSION + 1
    raw["participants"] = "not-a-participant-list"

    with pytest.raises(SchemaVersionMismatch):
        parse_phase3_dataset_registry(raw)


def test_phase3_dataset_registry_rejects_duplicate_participant_ids() -> None:
    raw = _raw()
    raw["participants"][1]["participant_id"] = raw["participants"][0]["participant_id"]

    with pytest.raises(ConfigError) as exc:
        parse_phase3_dataset_registry(raw)
    assert exc.value.key == "participants.participant_id"  # type: ignore[attr-defined]


def test_phase3_dataset_registry_rejects_incompatible_action_contract() -> None:
    raw = _raw()
    raw["participants"][0]["action_contract"] = copy.deepcopy(
        raw["participants"][0]["action_contract"]
    )
    raw["participants"][0]["action_contract"]["contract_id"] = "not-accepted"

    with pytest.raises(ConfigError) as exc:
        parse_phase3_dataset_registry(raw)
    assert "action_contract" in exc.value.key  # type: ignore[attr-defined]


def test_phase3_dataset_registry_rejects_underfilled_silo() -> None:
    raw = _raw()
    raw["participants"][0]["window_count"] = 0

    with pytest.raises(ConfigError) as exc:
        parse_phase3_dataset_registry(raw)
    assert "window_count" in exc.value.key  # type: ignore[attr-defined]


def test_phase3_dataset_registry_rejects_probe_mismatch() -> None:
    raw = _raw()
    raw["participants"][0]["accepted_probe_hash"] = "2" * 64

    with pytest.raises(ConfigError) as exc:
        parse_phase3_dataset_registry(raw)
    assert "accepted_probe_hash" in exc.value.key  # type: ignore[attr-defined]


def test_phase3_dataset_registry_rejects_public_raw_data_ref() -> None:
    raw = _raw()
    participant = raw["participants"][0]
    participant["publication_status"] = "published"
    participant["publication_blocker"] = None
    participant["data_ref"] = "file:///private/participant-a.h5"

    with pytest.raises(ConfigError) as exc:
        parse_phase3_dataset_registry(raw)
    assert "data_ref" in exc.value.key  # type: ignore[attr-defined]


def test_phase3_dataset_registry_rejects_unapproved_private_raw_data_ref() -> None:
    raw = _raw()
    raw["run_mode"] = "private_consortium"

    with pytest.raises(ConfigError) as exc:
        parse_phase3_dataset_registry(raw)
    assert "raw_data_path_allowed" in exc.value.key  # type: ignore[attr-defined]


def test_phase3_dataset_registry_rejects_manifest_smoke_hash_mismatch() -> None:
    manifest = default_phase3_consortium_manifest()
    raw = _raw()
    raw["participants"][0]["smoke_report_sha256"] = "f" * 64
    registry = parse_phase3_dataset_registry(raw)

    with pytest.raises(ConfigError) as exc:
        validate_phase3_registry_against_manifest(registry, manifest)
    assert "smoke_report_sha256" in exc.value.key  # type: ignore[attr-defined]
