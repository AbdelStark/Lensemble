"""Per-embodiment action head h_psi^(c): build, encode, INV-ACTIONHEAD-LOCAL (RFC-0008 4 / RFC-0007 5; #8).

The concrete nn.Module action head the eval harness (#52) requires, filling the orphaned substrate of the
closed issue #8. Covers the continuous MLP head and the discrete per-dim embedding head, the cond_dim seam
(a quadruped dim and an arm dim both produce (B, cond_dim)), the shape-mismatch and config/spec validation
edges, and the local-only state_dict seam. CPU fp32, tiny dims. Placed in tests/ml (CI-gated).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
from lensemble.errors import ConfigError, EvaluationError, LensembleErrorCode
from lensemble.model import build_action_head
from lensemble.model.action_head import ActionHead

_COND = 8


def _cfg(cond_dim: int | None = _COND, *, d: int = _COND) -> SimpleNamespace:
    model: dict[str, object] = {"d": d}
    if cond_dim is not None:
        model["cond_dim"] = cond_dim
    return SimpleNamespace(model=SimpleNamespace(**model))


def _continuous_spec(dim: int = 3) -> ActionSpec:
    return ActionSpec(
        embodiment_id="arm",
        kind=ActionKind.CONTINUOUS,
        dim=dim,
        low=tuple(-1.0 for _ in range(dim)),
        high=tuple(1.0 for _ in range(dim)),
        num_classes=None,
        units=tuple("u" for _ in range(dim)),
        wmcp_version=WMCP_VERSION,
    )


def _discrete_spec(num_classes: tuple[int, ...] = (4, 3)) -> ActionSpec:
    dim = len(num_classes)
    return ActionSpec(
        embodiment_id="grid",
        kind=ActionKind.DISCRETE,
        dim=dim,
        low=None,
        high=None,
        num_classes=num_classes,
        units=tuple("u" for _ in range(dim)),
        wmcp_version=WMCP_VERSION,
    )


# --- continuous head ---


def test_continuous_head_encodes_to_cond_dim() -> None:
    head = build_action_head(_cfg(), _continuous_spec(dim=3))
    assert isinstance(head, ActionHead) and head.cond_dim == _COND
    out = head.encode(torch.randn(5, 3))
    assert tuple(out.shape) == (5, _COND)
    assert out.dtype == torch.float32
    # forward is an alias for encode (composes as a plain nn.Module callable)
    assert torch.equal(head(torch.zeros(2, 3)), head.encode(torch.zeros(2, 3)))


def test_cond_dim_defaults_to_model_d_when_absent() -> None:
    head = build_action_head(_cfg(cond_dim=None, d=_COND), _continuous_spec())
    assert head.cond_dim == _COND  # falls back to cfg.model.d


def test_cond_dim_seam_unifies_distinct_embodiment_dims() -> None:
    # a 7-DoF arm and a 2-DoF gripper both map to the SAME cond_dim (RFC-0007 5)
    arm = build_action_head(_cfg(), _continuous_spec(dim=7))
    grip = build_action_head(_cfg(), _continuous_spec(dim=2))
    assert arm.encode(torch.randn(4, 7)).shape == grip.encode(torch.randn(4, 2)).shape


# --- discrete head ---


def test_discrete_head_sums_per_dim_embeddings() -> None:
    head = build_action_head(_cfg(), _discrete_spec((4, 3)))
    indices = torch.tensor([[0, 0], [3, 2], [1, 1]])
    out = head.encode(indices)
    assert tuple(out.shape) == (3, _COND)
    # the sum is the per-dim embedding contribution (independent of dtype of the input indices)
    out_float_idx = head.encode(indices.to(torch.float32))  # encode casts to int64
    assert torch.allclose(out, out_float_idx)


def test_discrete_head_state_dict_local_is_the_local_seam() -> None:
    head = build_action_head(_cfg(), _discrete_spec((4, 3)))
    local = head.state_dict_local()
    # one embedding table per action dim (INV-ACTIONHEAD-LOCAL: local-only checkpoint)
    assert any("embeddings.0" in k for k in local)
    assert any("embeddings.1" in k for k in local)


# --- encode shape validation ---


def test_encode_rejects_wrong_action_shape() -> None:
    head = build_action_head(_cfg(), _continuous_spec(dim=3))
    with pytest.raises(EvaluationError) as exc:
        head.encode(torch.randn(5, 4))  # last dim != spec.dim
    assert exc.value.code == LensembleErrorCode.EVALUATION_FAILED
    with pytest.raises(EvaluationError):
        head.encode(torch.randn(5))  # not rank-2


# --- build-time config / spec validation ---


def test_build_rejects_missing_model() -> None:
    with pytest.raises(ConfigError):
        build_action_head(SimpleNamespace(), _continuous_spec())


def test_build_rejects_non_positive_cond_dim() -> None:
    with pytest.raises(ConfigError):
        build_action_head(_cfg(cond_dim=0), _continuous_spec())


def test_build_rejects_wmcp_mismatch() -> None:
    spec = ActionSpec(
        embodiment_id="arm",
        kind=ActionKind.CONTINUOUS,
        dim=2,
        low=(-1.0, -1.0),
        high=(1.0, 1.0),
        num_classes=None,
        units=("u", "u"),
        wmcp_version="wmcp-0.0.0",  # not the pinned version
    )
    with pytest.raises(ConfigError) as exc:
        build_action_head(_cfg(), spec)
    assert exc.value.code == LensembleErrorCode.CONFIG_INVALID


def test_build_rejects_non_positive_dim() -> None:
    # bypass ActionSpec's own dataclass (no runtime dim check) via object construction
    spec = _continuous_spec(dim=1)
    bad = ActionSpec(
        embodiment_id=spec.embodiment_id,
        kind=spec.kind,
        dim=0,
        low=(),
        high=(),
        num_classes=None,
        units=(),
        wmcp_version=WMCP_VERSION,
    )
    with pytest.raises(ConfigError):
        build_action_head(_cfg(), bad)


def test_build_rejects_discrete_num_classes_length_mismatch() -> None:
    bad = ActionSpec(
        embodiment_id="grid",
        kind=ActionKind.DISCRETE,
        dim=2,
        low=None,
        high=None,
        num_classes=(4,),  # len != dim
        units=("u", "u"),
        wmcp_version=WMCP_VERSION,
    )
    with pytest.raises(ConfigError):
        build_action_head(_cfg(), bad)


def test_build_rejects_discrete_num_classes_below_two() -> None:
    bad = ActionSpec(
        embodiment_id="grid",
        kind=ActionKind.DISCRETE,
        dim=2,
        low=None,
        high=None,
        num_classes=(4, 1),  # a class count < 2
        units=("u", "u"),
        wmcp_version=WMCP_VERSION,
    )
    with pytest.raises(ConfigError):
        build_action_head(_cfg(), bad)
