"""``union_action_specs`` — the consortium-agreed action contract across heterogeneous silos (#243).

Sovereign silos partition one embodiment's episodes, so each reports the same dim/units but its own
observed continuous bounds. The accepted manifest action contract must union those bounds so every
participant's local ``ActionSpec`` falls inside it (the launcher's ``preflight`` requires bound equality).
"""

from __future__ import annotations

import pytest

from lensemble.contracts import ActionKind, ActionSpec, union_action_specs

_WMCP = "wmcp-1.0.0"


def _spec(low: tuple[float, ...], high: tuple[float, ...]) -> ActionSpec:
    return ActionSpec(
        embodiment_id="lerobot-6dof",
        kind=ActionKind.CONTINUOUS,
        dim=len(low),
        low=low,
        high=high,
        num_classes=None,
        units=("u",) * len(low),
        wmcp_version=_WMCP,
    )


def test_union_takes_elementwise_min_low_and_max_high() -> None:
    union = union_action_specs(
        [
            _spec((-1.0, 0.0, 5.0), (1.0, 2.0, 9.0)),
            _spec((-2.0, 0.5, 4.0), (0.5, 3.0, 8.0)),
            _spec((-0.5, -1.0, 6.0), (2.0, 1.0, 7.0)),
        ]
    )
    assert union.low == (-2.0, -1.0, 4.0)
    assert union.high == (2.0, 3.0, 9.0)
    assert union.dim == 3 and union.embodiment_id == "lerobot-6dof"


def test_union_of_one_is_identity() -> None:
    spec = _spec((-1.0,), (1.0,))
    assert union_action_specs([spec]) == spec


def test_union_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        union_action_specs([])


def test_union_rejects_dim_or_unit_disagreement() -> None:
    with pytest.raises(ValueError, match="dim"):
        union_action_specs([_spec((-1.0,), (1.0,)), _spec((-1.0, 0.0), (1.0, 2.0))])

    a = _spec((-1.0,), (1.0,))
    b = ActionSpec(
        embodiment_id="other-arm",
        kind=ActionKind.CONTINUOUS,
        dim=1,
        low=(-1.0,),
        high=(1.0,),
        num_classes=None,
        units=("u",),
        wmcp_version=_WMCP,
    )
    with pytest.raises(ValueError, match="embodiment_id"):
        union_action_specs([a, b])


def test_union_of_discrete_specs_requires_identity() -> None:
    discrete = ActionSpec(
        embodiment_id="gripper",
        kind=ActionKind.DISCRETE,
        dim=2,
        low=None,
        high=None,
        num_classes=(3, 4),
        units=("a", "b"),
        wmcp_version=_WMCP,
    )
    assert union_action_specs([discrete, discrete]) == discrete

    other = ActionSpec(
        embodiment_id="gripper",
        kind=ActionKind.DISCRETE,
        dim=2,
        low=None,
        high=None,
        num_classes=(5, 4),
        units=("a", "b"),
        wmcp_version=_WMCP,
    )
    with pytest.raises(ValueError, match="num_classes"):
        union_action_specs([discrete, other])
