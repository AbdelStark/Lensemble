"""Declared data-quality metadata and the WMCP join precondition (RFC-0004 §6 / RFC-0007 §6; #26).

A conforming declaration passes; an invalid ActionSpec, a federation/participant version mismatch, an
embodiment_id mismatch, and an empty dataset are each a hard `ContractViolation`; and the declaration
carries no raw tensor (`INV-RESIDENCY`). The metadata is declared, not verified.
"""

from __future__ import annotations

import dataclasses

import pytest

from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
from lensemble.data import DataQualityMetadata, validate_join_precondition
from lensemble.errors import ContractViolation, LensembleErrorCode


def _spec(
    embodiment_id: str = "so101-arm-7dof", wmcp: str = WMCP_VERSION
) -> ActionSpec:
    return ActionSpec(
        embodiment_id=embodiment_id,
        kind=ActionKind.CONTINUOUS,
        dim=3,
        low=(-1.0, -1.0, -1.0),
        high=(1.0, 1.0, 1.0),
        num_classes=None,
        units=("rad", "rad", "rad"),
        wmcp_version=wmcp,
    )


def _metadata(**over) -> DataQualityMetadata:
    base = dict(
        modality="rgb-video",
        embodiment_id="so101-arm-7dof",
        action_spec=_spec(),
        episode_count=12,
        collection_conditions={"site": "lab-a", "fps": "30"},
    )
    base.update(over)
    return DataQualityMetadata(**base)  # type: ignore[arg-type]


def test_conforming_declaration_passes() -> None:
    assert validate_join_precondition(_metadata()) is None


def test_invalid_action_spec_is_rejected() -> None:
    bad = _metadata(
        action_spec=ActionSpec(
            embodiment_id="so101-arm-7dof",
            kind=ActionKind.CONTINUOUS,
            dim=0,  # invalid
            low=(),
            high=(),
            num_classes=None,
            units=(),
            wmcp_version=WMCP_VERSION,
        )
    )
    with pytest.raises(ContractViolation) as exc:
        validate_join_precondition(bad)
    assert exc.value.code is LensembleErrorCode.WMCP_CONTRACT_VIOLATION


def test_federation_version_mismatch_refused_at_join() -> None:
    # a conforming participant (spec is at WMCP_VERSION) but the federation advertises a different version
    with pytest.raises(ContractViolation) as exc:
        validate_join_precondition(_metadata(), federation_wmcp_version="wmcp-2.0.0")
    assert exc.value.code is LensembleErrorCode.WMCP_CONTRACT_VIOLATION


def test_participant_spec_version_mismatch_refused() -> None:
    bad = _metadata(
        action_spec=_spec(wmcp="wmcp-0.9.0"), embodiment_id="so101-arm-7dof"
    )
    with pytest.raises(ContractViolation):
        validate_join_precondition(bad)


def test_embodiment_id_mismatch_rejected() -> None:
    bad = _metadata(embodiment_id="a-different-arm")  # spec says so101-arm-7dof
    with pytest.raises(ContractViolation):
        validate_join_precondition(bad)


def test_empty_dataset_rejected() -> None:
    with pytest.raises(ContractViolation):
        validate_join_precondition(_metadata(episode_count=0))


def test_metadata_carries_no_raw_tensor() -> None:
    import torch

    meta = _metadata()
    for f in dataclasses.fields(meta):
        value = getattr(meta, f.name)
        assert not isinstance(value, torch.Tensor)
        if isinstance(value, dict):
            assert all(not isinstance(v, torch.Tensor) for v in value.values())


def test_metadata_schema_round_trips_scalar_fields() -> None:
    meta = _metadata()
    as_dict = dataclasses.asdict(meta)
    assert set(as_dict) == {
        "modality",
        "embodiment_id",
        "action_spec",
        "episode_count",
        "collection_conditions",
    }
    assert as_dict["modality"] == "rgb-video"
    assert as_dict["episode_count"] == 12
    assert as_dict["collection_conditions"] == {"site": "lab-a", "fps": "30"}
    # frozen -> hashable and value-equal on reconstruction
    assert _metadata() == meta
