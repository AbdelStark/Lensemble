"""WMCP ``LatentState`` conformance (docs/rfcs/RFC-0007 2/4). Issue #6. CPU fallback (fp32)."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from lensemble.contracts import WMCP_VERSION, LatentState, check_latent_state
from lensemble.errors import ContractViolation, LensembleErrorCode


def _state(
    tokens: torch.Tensor, *, num_tokens: int, dim: int, version: str = WMCP_VERSION
) -> LatentState:
    return LatentState(
        tokens=tokens, num_tokens=num_tokens, dim=dim, wmcp_version=version
    )


def test_wmcp_version_is_pinned() -> None:
    assert WMCP_VERSION == "wmcp-1.0.0"


def test_conforming_rank2_and_rank3_return_none() -> None:
    s2 = _state(torch.zeros(4, 8, dtype=torch.float32), num_tokens=4, dim=8)
    assert s2.is_batched is False
    assert check_latent_state(s2) is None
    s3 = _state(torch.zeros(2, 4, 8, dtype=torch.float32), num_tokens=4, dim=8)
    assert s3.is_batched is True
    assert check_latent_state(s3) is None
    # optional expectations satisfied
    assert check_latent_state(s2, expected_dim=8, expected_num_tokens=4) is None


def _assert_violation(state: LatentState, **kwargs: object) -> ContractViolation:
    with pytest.raises(ContractViolation) as exc:
        check_latent_state(state, **kwargs)  # type: ignore[arg-type]
    err = exc.value
    assert err.code == LensembleErrorCode.WMCP_CONTRACT_VIOLATION
    assert isinstance(err.remediation, str) and err.remediation, (
        "remediation must be non-empty"
    )
    return err


def test_wrong_rank_rejected() -> None:
    s = _state(torch.zeros(4, 8, 2, 1, dtype=torch.float32), num_tokens=4, dim=8)
    err = _assert_violation(s)
    assert err.field == "rank"  # type: ignore[attr-defined]


def test_dim_mismatch_rejected() -> None:
    err = _assert_violation(
        _state(torch.zeros(4, 8, dtype=torch.float32), num_tokens=4, dim=16)
    )
    assert err.field == "dim"  # type: ignore[attr-defined]
    assert "16" in err.remediation


def test_num_tokens_mismatch_rejected() -> None:
    err = _assert_violation(
        _state(torch.zeros(4, 8, dtype=torch.float32), num_tokens=7, dim=8)
    )
    assert err.field == "num_tokens"  # type: ignore[attr-defined]
    assert "7" in err.remediation


def test_integer_dtype_rejected() -> None:
    err = _assert_violation(
        _state(torch.zeros(4, 8, dtype=torch.int64), num_tokens=4, dim=8)
    )
    assert err.field == "dtype"  # type: ignore[attr-defined]
    assert "int64" in err.remediation


def test_non_finite_rejected() -> None:
    t = torch.zeros(4, 8, dtype=torch.float32)
    t[0, 0] = float("nan")
    err = _assert_violation(_state(t, num_tokens=4, dim=8))
    assert err.field == "finiteness"  # type: ignore[attr-defined]


def test_wrong_version_rejected() -> None:
    s = _state(
        torch.zeros(4, 8, dtype=torch.float32),
        num_tokens=4,
        dim=8,
        version="wmcp-9.9.9",
    )
    err = _assert_violation(s)
    assert err.field == "wmcp_version"  # type: ignore[attr-defined]
    assert "wmcp-1.0.0" in err.remediation


def test_expected_dim_mismatch_rejected() -> None:
    s = _state(torch.zeros(4, 8, dtype=torch.float32), num_tokens=4, dim=8)
    err = _assert_violation(s, expected_dim=16)
    assert err.field == "expected_dim"  # type: ignore[attr-defined]


def test_check_is_pure_no_mutation() -> None:
    t = torch.randn(4, 8, dtype=torch.float32)
    snapshot = t.clone()
    s = _state(t, num_tokens=4, dim=8)
    check_latent_state(s)
    assert torch.equal(t, snapshot), "check_latent_state must not mutate the tensor"
    # frozen dataclass: fields are immutable
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.dim = 99  # type: ignore[misc]


# --- ActionSpec conformance (RFC-0007 3/4), issue #7 ---

from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from lensemble.contracts import (  # noqa: E402
    ActionHead,
    ActionKind,
    ActionSpec,
    validate_action_spec,
)


def _continuous(dim: int = 3) -> ActionSpec:
    return ActionSpec(
        embodiment_id="so101-arm-7dof",
        kind=ActionKind.CONTINUOUS,
        dim=dim,
        low=tuple(-1.0 for _ in range(dim)),
        high=tuple(1.0 for _ in range(dim)),
        num_classes=None,
        units=tuple("rad" for _ in range(dim)),
        wmcp_version=WMCP_VERSION,
    )


def _discrete(dim: int = 2) -> ActionSpec:
    return ActionSpec(
        embodiment_id="quadruped.gait",
        kind=ActionKind.DISCRETE,
        dim=dim,
        low=None,
        high=None,
        num_classes=tuple(4 for _ in range(dim)),
        units=tuple("idx" for _ in range(dim)),
        wmcp_version=WMCP_VERSION,
    )


def test_valid_specs_validate() -> None:
    assert validate_action_spec(_continuous()) is None
    assert validate_action_spec(_discrete()) is None


def _assert_spec_violation(spec: ActionSpec, field: str) -> None:
    with pytest.raises(ContractViolation) as exc:
        validate_action_spec(spec)
    assert exc.value.code == LensembleErrorCode.WMCP_CONTRACT_VIOLATION
    assert exc.value.remediation
    assert exc.value.field == field  # type: ignore[attr-defined]


def test_invalid_specs_rejected() -> None:
    import dataclasses as dc

    base_c = _continuous(3)
    base_d = _discrete(2)
    _assert_spec_violation(dc.replace(base_c, dim=0, low=(), high=(), units=()), "dim")
    _assert_spec_violation(dc.replace(base_c, units=("rad", "rad")), "units")
    _assert_spec_violation(
        dc.replace(base_c, high=(-1.0, -1.0, -1.0)), "bounds"
    )  # low>=high
    _assert_spec_violation(dc.replace(base_d, num_classes=(4, 1)), "num_classes")  # <2
    _assert_spec_violation(
        dc.replace(base_c, num_classes=(2, 2, 2)), "num_classes"
    )  # continuous w/ classes
    _assert_spec_violation(
        dc.replace(base_d, low=(0.0, 0.0), high=(1.0, 1.0)), "bounds"
    )  # discrete w/ bounds
    _assert_spec_violation(dc.replace(base_c, embodiment_id="Bad ID!"), "embodiment_id")
    _assert_spec_violation(
        dc.replace(base_c, wmcp_version="wmcp-0.0.0"), "wmcp_version"
    )


def test_actionspec_hashable_and_stable() -> None:
    a, b = _continuous(3), _continuous(3)
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1  # usable as a set/dict key


@given(dim=st.integers(min_value=1, max_value=6))
def test_continuous_property_valid(dim: int) -> None:
    spec = ActionSpec(
        embodiment_id="emb-0",
        kind=ActionKind.CONTINUOUS,
        dim=dim,
        low=tuple(float(-i - 1) for i in range(dim)),
        high=tuple(float(i + 1) for i in range(dim)),
        num_classes=None,
        units=tuple("u" for _ in range(dim)),
        wmcp_version=WMCP_VERSION,
    )
    assert validate_action_spec(spec) is None


@given(counts=st.lists(st.integers(min_value=2, max_value=9), min_size=1, max_size=6))
def test_discrete_property_valid(counts: list[int]) -> None:
    dim = len(counts)
    spec = ActionSpec(
        embodiment_id="emb-1",
        kind=ActionKind.DISCRETE,
        dim=dim,
        low=None,
        high=None,
        num_classes=tuple(counts),
        units=tuple("idx" for _ in range(dim)),
        wmcp_version=WMCP_VERSION,
    )
    assert validate_action_spec(spec) is None


# --- ActionHead interface (RFC-0007 5), issue #8 ---


class _RefHead(ActionHead):
    """A minimal conforming head: a fixed linear map (B, spec.dim) -> (B, cond_dim)."""

    def __init__(self, spec: ActionSpec, *, cond_dim: int) -> None:
        super().__init__(spec, cond_dim=cond_dim)  # validates spec before any params
        gen = torch.Generator().manual_seed(0)
        self.weight = torch.randn(spec.dim, cond_dim, generator=gen)

    def encode(self, action: torch.Tensor) -> torch.Tensor:
        return action.to(self.weight.dtype) @ self.weight  # dtype follows compute dtype

    def state_dict_local(self) -> dict[str, torch.Tensor]:
        return {"weight": self.weight}


def test_cond_dim_seam_is_embodiment_independent() -> None:
    # Two embodiments with different spec.dim produce encodings of the SAME cond_dim (RFC-0007 5):
    # this is what lets a 7-DoF arm and a 12-DoF body condition the same shared predictor g_phi.
    cond_dim = 16
    head_7 = _RefHead(_continuous(7), cond_dim=cond_dim)
    head_12 = _RefHead(_continuous(12), cond_dim=cond_dim)
    out_7 = head_7.encode(torch.zeros(4, 7))
    out_12 = head_12.encode(torch.zeros(4, 12))
    assert out_7.shape == (4, cond_dim)
    assert out_12.shape == (4, cond_dim)
    assert head_7.cond_dim == head_12.cond_dim == cond_dim
    assert head_7.spec.dim != head_12.spec.dim  # the input side is free


def test_init_validates_spec_before_allocating() -> None:
    bad = dataclasses.replace(_continuous(3), wmcp_version="wmcp-9.9.9")
    with pytest.raises(ContractViolation) as exc:
        _RefHead(bad, cond_dim=8)  # super().__init__ must validate first (INV-WMCP)
    assert exc.value.code == LensembleErrorCode.WMCP_CONTRACT_VIOLATION


def test_action_head_is_abstract() -> None:
    with pytest.raises(TypeError):
        ActionHead(_continuous(3), cond_dim=8)  # type: ignore[abstract]


def test_local_checkpoint_seam_is_named_state_dict_local() -> None:
    head = _RefHead(_continuous(5), cond_dim=8)
    # the local-only accessor exists under its INV-ACTIONHEAD-LOCAL name...
    assert set(head.state_dict_local()) == {"weight"}
    # ...and NOT under the shared-serializer name a federation/artifact path would pick up.
    assert not hasattr(head, "state_dict")
