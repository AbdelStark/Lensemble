"""Coordinator and participant actors share the same Phase 3 run-agreement gate."""

from __future__ import annotations

import pytest

from lensemble.config import (
    default_phase3_consortium_manifest,
    parse_consortium_manifest,
    validate_coordinator_run_agreement,
    validate_participant_join,
)
from lensemble.errors import ConfigError


def test_coordinator_and_participant_accept_the_same_manifest() -> None:
    manifest = default_phase3_consortium_manifest()

    coordinator_view = validate_coordinator_run_agreement(manifest)
    participant_view = validate_participant_join(
        manifest, participant_id=manifest.participants[0].participant_id
    )

    assert coordinator_view.run_id == manifest.run_id
    assert participant_view.accepted_probe_hash == manifest.public_probe.content_hash


def test_coordinator_and_participant_reject_the_same_probe_mismatch() -> None:
    manifest = default_phase3_consortium_manifest()
    bad_participant = manifest.participants[0].model_copy(
        update={"accepted_probe_hash": "3" * 64}
    )
    bad_manifest = manifest.model_copy(
        update={"participants": (bad_participant, *manifest.participants[1:])}
    )

    with pytest.raises(ConfigError):
        validate_coordinator_run_agreement(bad_manifest)
    with pytest.raises(ConfigError):
        validate_participant_join(
            bad_manifest, participant_id=bad_participant.participant_id
        )


def test_generated_example_payload_is_valid_for_shared_ingress() -> None:
    raw = default_phase3_consortium_manifest().model_dump(mode="json")
    manifest = parse_consortium_manifest(raw)

    assert validate_coordinator_run_agreement(manifest) == manifest
    for participant in manifest.participants:
        assert validate_participant_join(
            manifest, participant_id=participant.participant_id
        )
