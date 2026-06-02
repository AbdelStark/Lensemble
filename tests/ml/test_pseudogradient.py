"""The PseudoGradient contract — the one private object that crosses the boundary (RFC-0003 3). #38.

delta is the (theta, phi) local-minus-global update over the federated param groups only; an action-head
group raises ResidencyViolation (INV-ACTIONHEAD-LOCAL); structural validation rejects non-fp32 / non-finite
delta and a dataset_root whose length is not 32; the egress guard permits the carrier and only its delta.
"""

from __future__ import annotations

import pytest
import torch

from lensemble.data import guard_egress
from lensemble.errors import LensembleErrorCode, ResidencyViolation
from lensemble.federation import PseudoGradient, build_pseudogradient

_ROOT = b"\x11" * 32


def test_delta_is_local_minus_global_over_encoder_and_predictor() -> None:
    theta_t = torch.zeros(4, 3)
    phi_t = torch.zeros(6)
    theta_c = torch.ones(4, 3)  # a deterministic toy local update
    phi_c = torch.full((6,), 2.0)
    pg = build_pseudogradient(
        {"encoder.w": theta_c - theta_t, "predictor.b": phi_c - phi_t},
        dataset_root=_ROOT,
        round_index=2,
    )
    expected = torch.cat([(theta_c - theta_t).reshape(-1), (phi_c - phi_t).reshape(-1)])
    assert torch.equal(
        pg.delta, expected
    )  # encoder theta first, then predictor phi, exact
    assert pg.delta.numel() == theta_t.numel() + phi_t.numel()  # only federated groups
    assert pg.round_index == 2
    assert pg.l2_norm == pytest.approx(float(expected.norm()))


def test_action_head_group_is_rejected_fail_closed() -> None:
    with pytest.raises(ResidencyViolation) as exc:
        build_pseudogradient(
            {"encoder.w": torch.zeros(4), "action_head.so101.weight": torch.zeros(2)},
            dataset_root=_ROOT,
            round_index=0,
        )
    assert exc.value.code == LensembleErrorCode.RESIDENCY_VIOLATION
    assert exc.value.tensor_role == "action_head"  # type: ignore[attr-defined]


def test_non_federated_group_is_rejected() -> None:
    with pytest.raises(ResidencyViolation):
        build_pseudogradient(
            {"gauge.frame": torch.zeros(4)}, dataset_root=_ROOT, round_index=0
        )


def test_structural_validation_rejects_bad_fields() -> None:
    good = torch.zeros(8, dtype=torch.float32)
    with pytest.raises(ValueError):
        PseudoGradient(
            delta=good.to(torch.float64), l2_norm=0.0, dataset_root=_ROOT, round_index=0
        )
    nan = good.clone()
    nan[0] = float("nan")
    with pytest.raises(ValueError):
        PseudoGradient(delta=nan, l2_norm=0.0, dataset_root=_ROOT, round_index=0)
    with pytest.raises(ValueError):
        PseudoGradient(delta=good, l2_norm=0.0, dataset_root=b"short", round_index=0)
    with pytest.raises(ValueError):
        PseudoGradient(delta=good, l2_norm=0.0, dataset_root=_ROOT, round_index=-1)
    with pytest.raises(ValueError):
        PseudoGradient(
            delta=good, l2_norm=99.0, dataset_root=_ROOT, round_index=0
        )  # l2_norm != ||delta||


def test_egress_guard_permits_pseudogradient() -> None:
    pg = build_pseudogradient(
        {"encoder.w": torch.randn(5), "predictor.w": torch.randn(3)},
        dataset_root=_ROOT,
        round_index=1,
    )
    # the carrier crosses: only `delta` is a tensor, and the egress role permits it
    assert guard_egress(pg) is None
