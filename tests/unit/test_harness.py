"""Meta-tests for the shared harness: tolerances and fixture determinism (07 1/6/7). Issue #65."""

from __future__ import annotations

import dataclasses
from typing import Callable

import torch

_EXPECTED_TOLERANCES = {
    "RTOL_LOSS",
    "ATOL_LOSS",
    "SIGREG_NULL_TOL",
    "SIGREG_SIGNAL_FLOOR",
    "RTOL_PROC",
    "ATOL_ORTHO",
    "ANGLE_TOL_DEG",
    "RTOL_DP",
    "RTOL_DP_STD",
    "RTOL_BF16",
    "ATOL_BF16",
    "RTOL_AGG",
}


def test_all_tolerances_exist_and_are_floats(
    tol: object, tolerance_field_names: set[str]
) -> None:
    assert tolerance_field_names == _EXPECTED_TOLERANCES
    for f in dataclasses.fields(tol):  # type: ignore[arg-type]
        assert isinstance(getattr(tol, f.name), float), f.name


def test_tiny_warmstart_is_deterministic(
    make_tiny_warmstart: Callable[..., object],
) -> None:
    a = make_tiny_warmstart(seed=0)
    b = make_tiny_warmstart(seed=0)
    c = make_tiny_warmstart(seed=1)
    sd_a, sd_b, sd_c = a.state_dict(), b.state_dict(), c.state_dict()  # type: ignore[attr-defined]
    assert all(
        torch.equal(sd_a[k], sd_b[k]) for k in sd_a
    )  # byte-identical under same seed
    assert not all(
        torch.equal(sd_a[k], sd_c[k]) for k in sd_a
    )  # different seed differs


def test_synthetic_probe_is_deterministic_and_covers_frame(
    make_synthetic_probe: Callable[..., object],
) -> None:
    p1 = make_synthetic_probe(seed=0)
    p2 = make_synthetic_probe(seed=0)
    assert torch.equal(p1.points, p2.points)  # type: ignore[attr-defined]
    assert p1.k >= p1.d  # type: ignore[attr-defined]  # k >= d landmark coverage
    assert p1.landmark_idx.shape[0] == p1.k  # type: ignore[attr-defined]


def test_fixtures_available(
    tiny_warmstart: object, synthetic_probe: object, toy_env: object, rng: object
) -> None:
    out = tiny_warmstart(torch.randn(3, 8, generator=rng))  # type: ignore[operator]
    assert out.shape == (3, 8)
    toy_env.reset()  # type: ignore[attr-defined]
    assert toy_env.step(torch.zeros(4)).shape == (4,)  # type: ignore[attr-defined]
