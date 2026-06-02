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
