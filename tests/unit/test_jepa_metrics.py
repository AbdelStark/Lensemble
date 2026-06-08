"""Edge-case unit coverage for the shared residency-safe JEPA metric helpers (#241)."""

from __future__ import annotations

import pytest
import torch

from lensemble.eval.jepa_metrics import (
    JepaWindowMetrics,
    _apply_theta_delta,
    _unflatten_update_delta,
    effective_rank,
    evaluate_jepa_windows,
    mean_frame_drift_deg,
)


class _DummyConfig:
    class model:  # noqa: N801 — mimic the attribute path cfg.model.latent_dim
        latent_dim = 8


def test_evaluate_jepa_windows_returns_none_without_windows() -> None:
    assert (
        evaluate_jepa_windows(
            _DummyConfig(),  # type: ignore[arg-type]
            encoder=None,
            predictor=None,
            action_head=None,
            windows=(),
            max_windows=4,
        )
        is None
    )


def test_jepa_window_metrics_is_frozen() -> None:
    metrics = JepaWindowMetrics(val_pred=1.0, val_sigreg=0.1, effective_rank=3.0)
    with pytest.raises(Exception):
        metrics.val_pred = 2.0  # type: ignore[misc]


def test_effective_rank_is_one_for_rank_one() -> None:
    direction = torch.randn(1, 8)
    rank_one = torch.randn(256, 1) * direction
    assert effective_rank(rank_one) < 1.2


def test_mean_frame_drift_deg_prefers_pairs_then_global() -> None:
    class _Pair:
        def __init__(self, angle: float) -> None:
            self.rotation_angle_deg = angle

    with_pairs = type(
        "R", (), {"pairs": [_Pair(10.0), _Pair(20.0)], "drift_from_global": {}}
    )()
    assert mean_frame_drift_deg(with_pairs) == pytest.approx(15.0)

    global_only = type(
        "R", (), {"pairs": [], "drift_from_global": {"a": 30.0, "b": 50.0}}
    )()
    assert mean_frame_drift_deg(global_only) == pytest.approx(40.0)

    empty = type("R", (), {"pairs": [], "drift_from_global": {}})()
    assert mean_frame_drift_deg(empty) is None


def test_unflatten_update_delta_rejects_wrong_length() -> None:
    theta = {"w": torch.zeros(2, 2)}
    phi = {"b": torch.zeros(2)}
    too_short = torch.zeros(3)
    with pytest.raises(ValueError, match="ended inside"):
        _unflatten_update_delta(theta, phi, too_short)

    too_long = torch.zeros(7)
    with pytest.raises(ValueError, match="trailing values"):
        _unflatten_update_delta(theta, phi, too_long)


def test_apply_theta_delta_handles_integer_buffers() -> None:
    theta = {"f": torch.zeros(2), "i": torch.zeros(2, dtype=torch.int64)}
    delta = {"f": torch.ones(2), "i": torch.tensor([1.4, 1.6])}
    updated = _apply_theta_delta(theta, delta)
    assert torch.equal(updated["f"], torch.ones(2))
    assert updated["i"].dtype == torch.int64
    assert torch.equal(updated["i"], torch.tensor([1, 2]))
