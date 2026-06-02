"""RunManifest residency guard (RFC-0009 6 / INV-RESIDENCY). Issue #36, T10.

The manifest is a boundary-crossing artifact carrying hashes/seeds/versions/counts only. A raw tensor
reaching the free-form config_resolved must fail closed at the write boundary, never be serialized.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

from lensemble.config import load
from lensemble.config.manifest import build_manifest, write_manifest
from lensemble.errors import ResidencyViolation


def test_raw_tensor_in_config_resolved_is_rejected_at_write(tmp_path: Path) -> None:
    manifest = build_manifest(
        load(),
        run_mode="train_local",
        created_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
    )
    # smuggle a raw tensor into the free-form config tree (frozen model -> model_copy)
    tampered = manifest.model_copy(
        update={"config_resolved": {"weights": torch.zeros(3)}}
    )
    with pytest.raises(ResidencyViolation):
        write_manifest(tampered, tmp_path / "run_manifest.json")
    assert not (
        tmp_path / "run_manifest.json"
    ).exists()  # nothing written (fail-closed)
