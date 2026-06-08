"""The published Phase 3 SO-100 dataset/probe registry is real and placeholder-free (#242).

Validates the checked-in `docs/evidence/phase3_long_run_dataset_registry.json` against its companion
`docs/evidence/phase3_consortium_manifest.json`: every participant is `published` against an immutable
`hf://` silo ref with a recorded non-toy window count, the public probe is pinned identically in both,
and no raw-data path is allowed.
"""

from __future__ import annotations

import json
from pathlib import Path

from lensemble.config import load_consortium_manifest
from lensemble.data import (
    load_phase3_dataset_registry,
    validate_phase3_registry_against_manifest,
)

_REGISTRY = Path("docs/evidence/phase3_long_run_dataset_registry.json")
_MANIFEST = Path("docs/evidence/phase3_consortium_manifest.json")
_SILO_REPO = "abdelstark/lensemble-phase3-so100-silos"


def test_registry_validates_against_consortium_manifest() -> None:
    registry = load_phase3_dataset_registry(_REGISTRY)
    manifest = load_consortium_manifest(_MANIFEST)
    # Raises ConfigError on any disagreement; passing is the assertion.
    validate_phase3_registry_against_manifest(registry, manifest)
    assert registry.consortium_id == manifest.consortium_id
    assert registry.public_probe == manifest.public_probe


def test_every_participant_is_published_with_real_refs() -> None:
    registry = load_phase3_dataset_registry(_REGISTRY)

    assert len(registry.participants) >= 4
    statuses = {p.publication_status for p in registry.participants}
    assert statuses == {"published"}  # zero placeholders

    refs = [p.data_ref for p in registry.participants]
    assert len(set(refs)) == len(refs)  # distinct silos
    for participant in registry.participants:
        assert participant.data_ref.startswith(f"hf://datasets/{_SILO_REPO}/")
        assert participant.publication_blocker is None
        assert participant.raw_data_path_allowed is False
        assert participant.window_count >= 1000  # non-toy training scale
        assert participant.episode_count >= 1
        assert participant.window_steps == 4


def test_probe_pin_and_heldout_split_are_referenced() -> None:
    registry = load_phase3_dataset_registry(_REGISTRY)
    probe_hash = registry.public_probe.content_hash
    assert len(probe_hash) == 64 and all(c in "0123456789abcdef" for c in probe_hash)
    # The disjoint held-out split is named in every participant's held-out policy.
    for participant in registry.participants:
        assert "held-out" in participant.heldout_policy.lower()
    # The manifest pins the same probe hash the launcher reproduces at run time.
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["public_probe"]["content_hash"] == probe_hash
