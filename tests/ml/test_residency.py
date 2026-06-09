"""Residency egress guard (RFC-0004 2 / 06-security 3). Issue #23. Security-critical (INV-RESIDENCY)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec, LatentState
from lensemble.data import Episode, EpisodeDataset, Transition, Window
from lensemble.data.residency import EgressRole, guard_egress
from lensemble.errors import LensembleErrorCode, ResidencyViolation


@dataclass
class _PseudoGradient:
    __egress_role__ = EgressRole.PSEUDO_GRADIENT
    delta: torch.Tensor
    l2_norm: float
    dataset_root: str


@dataclass
class _PseudoGradientWithHead:
    __egress_role__ = EgressRole.PSEUDO_GRADIENT
    delta: torch.Tensor
    action_head_params: torch.Tensor  # INV-ACTIONHEAD-LOCAL: must never cross


@dataclass
class _DatasetCommitment:
    __egress_role__ = EgressRole.DATASET_COMMITMENT
    root: bytes
    episode_count: int
    wmcp_version: str


def _latent() -> LatentState:
    return LatentState(
        tokens=torch.zeros(4, 8), num_tokens=4, dim=8, wmcp_version=WMCP_VERSION
    )


def test_raw_observation_tensor_rejected() -> None:
    with pytest.raises(ResidencyViolation) as exc:
        guard_egress(torch.zeros(3, 4, 4))  # a raw observation
    assert exc.value.code == LensembleErrorCode.RESIDENCY_VIOLATION
    assert exc.value.remediation


def test_raw_action_tensor_rejected() -> None:
    with pytest.raises(ResidencyViolation):
        guard_egress({"update": torch.zeros(7)})  # a raw action nested in a message


def test_private_embedding_rejected() -> None:
    with pytest.raises(ResidencyViolation) as exc:
        guard_egress(_latent())  # f_theta(x) — a private embedding
    assert exc.value.tensor_role == "private_embedding"  # type: ignore[attr-defined]


def test_resident_dataset_types_rejected() -> None:
    spec = ActionSpec(
        embodiment_id="emb",
        kind=ActionKind.CONTINUOUS,
        dim=2,
        low=(-1.0, -1.0),
        high=(1.0, 1.0),
        num_classes=None,
        units=("u", "u"),
        wmcp_version=WMCP_VERSION,
    )
    tr = Transition(
        obs_t=torch.zeros(3), action_t=torch.zeros(2), obs_tp1=torch.zeros(3)
    )
    ep = Episode(
        episode_id="e",
        transitions=[tr],
        embodiment_id="emb",
        modality="rgb",
        action_spec=spec,
        collection_meta={},
    )
    for resident in (
        tr,
        ep,
        Window(
            obs=torch.zeros(2, 3),
            actions=torch.zeros(1, 2),
            num_steps=1,
            embodiment_id="emb",
            state=torch.zeros(2, 2),
        ),
        EpisodeDataset([ep]),
    ):
        with pytest.raises(ResidencyViolation):
            guard_egress(resident)


def test_window_state_tensor_is_residency_bound() -> None:
    window = Window(
        obs=torch.zeros(2, 1, 3, 4, 4),
        actions=torch.zeros(1, 2),
        num_steps=1,
        embodiment_id="swipe-dot-2dof",
        state=torch.zeros(2, 2),
    )
    with pytest.raises(ResidencyViolation) as exc:
        guard_egress({"window": window})
    assert exc.value.tensor_role == "raw_window"  # type: ignore[attr-defined]


def test_valid_payload_passes() -> None:
    payload = {
        "update": _PseudoGradient(
            delta=torch.zeros(16), l2_norm=1.5, dataset_root="r_c-abc"
        ),
        "commitment": _DatasetCommitment(
            root=b"\x00" * 32, episode_count=10, wmcp_version=WMCP_VERSION
        ),
        "coordination": {
            "sketch_seed": 7,
            "probe_hash": "deadbeef",
            "global_hash": "cafe",
        },
        "metrics": {"loss/pred": 0.12, "grad_norm": 3.4, "shape": [16]},
    }
    assert guard_egress(payload) is None


def test_action_head_group_on_delta_rejected() -> None:
    # a per-embodiment action head reaching the released delta (INV-ACTIONHEAD-LOCAL)
    with pytest.raises(ResidencyViolation):
        guard_egress(
            _PseudoGradientWithHead(
                delta=torch.zeros(16), action_head_params=torch.zeros(8)
            )
        )


def test_explicit_action_head_marker_rejected() -> None:
    class _Head:
        __egress_role__ = EgressRole.ACTION_HEAD

    with pytest.raises(ResidencyViolation) as exc:
        guard_egress(_Head())
    assert exc.value.tensor_role == "action_head"  # type: ignore[attr-defined]


def test_violation_propagates_not_swallowed() -> None:
    # the guard itself must not catch-and-ignore: a resident payload raises out of guard_egress
    raised = False
    try:
        guard_egress(torch.zeros(2))
    except ResidencyViolation:
        raised = True
    assert raised, "guard_egress must propagate ResidencyViolation (fail-closed)"


def test_unknown_object_is_walked_fail_closed() -> None:
    # An unmarked, unknown carrier is not trusted: the guard walks its __dict__ so a hidden
    # resident tensor is still caught (fail-closed, INV-RESIDENCY). Exercises the __dict__
    # field-iteration and the unknown-object recursion in residency._iter_fields/_inspect.
    class _Smuggler:
        def __init__(self) -> None:
            self.label = "harmless"
            self.hidden = torch.zeros(4)  # a raw tensor tucked behind an unknown type

    with pytest.raises(ResidencyViolation):
        guard_egress(_Smuggler())


def test_unknown_fieldless_object_passes() -> None:
    # An unknown object that carries no inspectable fields (no __dict__, not a dataclass) holds
    # no resident data, so the fail-closed walk finds nothing and the guard returns cleanly.
    class _Opaque:
        __slots__ = ()

    assert guard_egress({"marker": _Opaque()}) is None
