"""Phase 3 consortium manifest contract (#222)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lensemble.config import (
    CONSORTIUM_MANIFEST_SCHEMA_VERSION,
    default_phase3_consortium_manifest,
    load_consortium_manifest,
    parse_consortium_manifest,
    to_consortium_json,
    validate_coordinator_run_agreement,
    validate_participant_join,
    write_consortium_manifest,
)
from lensemble.errors import ConfigError, SchemaVersionMismatch


def _raw() -> dict:
    return default_phase3_consortium_manifest().model_dump(mode="json")


def test_default_phase3_consortium_manifest_is_valid_for_every_actor() -> None:
    manifest = default_phase3_consortium_manifest()

    assert manifest.schema_version == CONSORTIUM_MANIFEST_SCHEMA_VERSION
    assert manifest.runtime.transport == "network"
    assert manifest.runtime.secure_aggregation_required is True
    assert manifest.runtime.dp_required is True
    assert len(manifest.participants) == 4
    assert validate_coordinator_run_agreement(manifest) == manifest

    for participant in manifest.participants:
        joined = validate_participant_join(
            manifest, participant_id=participant.participant_id
        )
        assert joined == participant


def test_consortium_manifest_round_trips_canonical_json(tmp_path: Path) -> None:
    manifest = default_phase3_consortium_manifest()
    path = write_consortium_manifest(manifest, tmp_path / "phase3.json")

    assert load_consortium_manifest(path) == manifest
    assert json.loads(to_consortium_json(manifest)) == json.loads(path.read_text())


def test_parse_consortium_manifest_gates_future_schema_first() -> None:
    raw = _raw()
    raw["schema_version"] = CONSORTIUM_MANIFEST_SCHEMA_VERSION + 1
    raw["participants"] = "not-even-a-participant-list"

    with pytest.raises(SchemaVersionMismatch):
        parse_consortium_manifest(raw)


def test_consortium_manifest_rejects_duplicate_participant_ids() -> None:
    raw = _raw()
    raw["participants"][1]["participant_id"] = raw["participants"][0]["participant_id"]

    with pytest.raises(ConfigError) as exc:
        parse_consortium_manifest(raw)
    assert exc.value.code.value == "config_invalid"
    assert exc.value.key == "participants.participant_id"  # type: ignore[attr-defined]


def test_consortium_manifest_rejects_probe_mismatch() -> None:
    raw = _raw()
    raw["participants"][0]["accepted_probe_hash"] = "2" * 64

    with pytest.raises(ConfigError) as exc:
        parse_consortium_manifest(raw)
    assert "accepted_probe_hash" in exc.value.key  # type: ignore[attr-defined]


def test_consortium_manifest_rejects_missing_data_declaration() -> None:
    raw = _raw()
    raw["participants"][0]["data"] = None

    with pytest.raises(ConfigError) as exc:
        parse_consortium_manifest(raw)
    assert exc.value.key.endswith(".data")  # type: ignore[attr-defined]


def test_consortium_manifest_rejects_unsupported_dp_policy() -> None:
    raw = _raw()
    raw["dp_policy"]["accountant"] = "prv"

    with pytest.raises(ConfigError) as exc:
        parse_consortium_manifest(raw)
    assert "dp_accountants" in exc.value.key  # type: ignore[attr-defined]


def test_consortium_manifest_rejects_unsupported_network_transport() -> None:
    raw = _raw()
    raw["participants"][0]["capabilities"]["network_transport"] = False

    with pytest.raises(ConfigError) as exc:
        parse_consortium_manifest(raw)
    assert "network_transport" in exc.value.key  # type: ignore[attr-defined]


def test_consortium_manifest_rejects_incompatible_wmcp_version() -> None:
    raw = _raw()
    raw["participants"][0]["action_contract"]["wmcp_version"] = "wmcp-0.0.0"

    with pytest.raises(ConfigError) as exc:
        parse_consortium_manifest(raw)
    assert "wmcp_version" in exc.value.key  # type: ignore[attr-defined]


def test_consortium_manifest_rejects_action_contract_not_accepted() -> None:
    raw = _raw()
    raw["participants"][0]["action_contract"] = copy.deepcopy(
        raw["participants"][0]["action_contract"]
    )
    raw["participants"][0]["action_contract"]["dim"] = 7
    raw["participants"][0]["action_contract"]["low"] = [-1.0] * 7
    raw["participants"][0]["action_contract"]["high"] = [1.0] * 7
    raw["participants"][0]["action_contract"]["units"] = ["unitless"] * 7

    with pytest.raises(ConfigError) as exc:
        parse_consortium_manifest(raw)
    assert "action_contract" in exc.value.key  # type: ignore[attr-defined]


def test_participant_join_rejects_unknown_participant() -> None:
    manifest = default_phase3_consortium_manifest()

    with pytest.raises(ConfigError) as exc:
        validate_participant_join(manifest, participant_id="not-in-the-manifest")
    assert exc.value.key == "participant_id"  # type: ignore[attr-defined]
