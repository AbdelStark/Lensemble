"""The public surface of ``lensemble`` matches conventions 5 / 02-public-api 1 (issue #2)."""

from __future__ import annotations

import re

import lensemble

# The frozen public re-export set (docs/spec/conventions.md 5).
PUBLIC_SURFACE = [
    "LensembleConfig",
    "RunManifest",
    "load",
    "train_local",
    "Coordinator",
    "Participant",
    "RoundState",
    "build_encoder",
    "build_predictor",
    "build_action_head",
    "Objective",
    "evaluate",
    "Planner",
    "frame_drift",
    "procrustes_align",
    "commit_dataset",
    "DatasetCommitment",
    "ContributionLedger",
    "recompute_alignment",
]

# Permissive SemVer (MAJOR.MINOR.PATCH with optional pre-release / build metadata).
_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.\-]+)?$")


def test_version_is_semver() -> None:
    assert isinstance(lensemble.__version__, str)
    assert _SEMVER.match(lensemble.__version__), lensemble.__version__


def test_public_surface_importable() -> None:
    for name in PUBLIC_SURFACE:
        assert hasattr(lensemble, name), f"missing public symbol: {name}"


def test_all_advertises_public_surface() -> None:
    for name in ["__version__", *PUBLIC_SURFACE]:
        assert name in lensemble.__all__, f"{name} absent from __all__"


def test_unknown_attribute_raises() -> None:
    import pytest

    with pytest.raises(AttributeError):
        _ = lensemble.definitely_not_a_public_symbol


def test_all_matches_export_map() -> None:
    # The literal __all__ must not drift from the lazy-export map (_EXPORTS).
    assert set(lensemble.__all__) == {"__version__", *lensemble._EXPORTS}


def test_public_surface_matches_all() -> None:
    assert set(PUBLIC_SURFACE) == set(lensemble.__all__) - {"__version__"}
