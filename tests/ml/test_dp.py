"""Per-participant clip+noise DP mechanism — INV-DP-BOUND (RFC-0012 1 / 07 §2.6). Issue #49.

Clipping projects any delta into the L2 ball of radius C_clip (asserted across magnitudes via hypothesis,
including the boundary and the zero vector); Gaussian noise has empirical std sigma*C_clip, is reproducible
from the seed, and differs across (round, participant) seeds; DPConfig validates its bounds.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lensemble.errors import ConfigError
from lensemble.privacy import DPConfig, add_gaussian_noise, clip_delta, privatize

_DIM = 32
_CLIP = 1.5


def _unit(seed: int, dim: int = _DIM) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)  # tol is an immutable constant
@given(
    scale=st.floats(
        min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
    )
)
def test_clip_enforces_bound(scale: float, tol: object) -> None:
    rtol_dp: float = tol.RTOL_DP  # type: ignore[attr-defined]
    delta = _unit(0) * scale  # a vector of norm ~scale
    clipped, post = clip_delta(delta, _CLIP)
    assert float(clipped.norm()) <= _CLIP * (1 + rtol_dp)  # INV-DP-BOUND
    assert post <= _CLIP * (1 + rtol_dp)
    if scale <= _CLIP:
        assert torch.equal(
            clipped, delta.to(torch.float32)
        )  # below the bound: unchanged


def test_clip_boundary_and_zero_vector() -> None:
    at_bound = _unit(1) * _CLIP  # ||delta|| == C_clip exactly
    clipped, post = clip_delta(at_bound, _CLIP)
    assert post <= _CLIP * (1 + 1e-6)
    zero = torch.zeros(_DIM)
    clipped_zero, post_zero = clip_delta(zero, _CLIP)  # no division by zero
    assert torch.equal(clipped_zero, zero) and post_zero == 0.0


def test_clip_reprojects_fp32_roundoff_after_large_projection() -> None:
    generator = torch.Generator().manual_seed(11)
    delta = torch.randn(524_288, generator=generator)

    clipped, post = clip_delta(delta, 0.5)

    assert float(clipped.norm()) <= 0.5
    assert post <= 0.5


def test_clip_is_bitwise_deterministic() -> None:
    delta = _unit(2) * 10.0
    a, na = clip_delta(delta, _CLIP)
    b, nb = clip_delta(delta, _CLIP)
    assert torch.equal(a, b) and na == nb


def test_gaussian_noise_std_and_reproducibility(tol: object) -> None:
    rtol_std: float = tol.RTOL_DP_STD  # type: ignore[attr-defined]
    sigma = 1.0
    base = torch.zeros(4096)
    draw1 = add_gaussian_noise(base, _CLIP, sigma, torch.Generator().manual_seed(7))
    # empirical std ~ sigma * C_clip
    assert abs(float(draw1.std()) - sigma * _CLIP) <= rtol_std * (sigma * _CLIP)
    # reproducible from the same seed; different across (round, participant) seeds
    draw1b = add_gaussian_noise(base, _CLIP, sigma, torch.Generator().manual_seed(7))
    draw2 = add_gaussian_noise(base, _CLIP, sigma, torch.Generator().manual_seed(8))
    assert torch.equal(draw1, draw1b)
    assert not torch.equal(draw1, draw2)


def test_privatize_clips_then_noises() -> None:
    cfg = DPConfig(clip_norm=_CLIP, noise_multiplier=1.0)
    delta = _unit(3) * 5.0  # above the bound
    private, post = privatize(delta, cfg, torch.Generator().manual_seed(0))
    assert post <= _CLIP * (1 + 1e-6)  # l2_norm is the post-clip, pre-noise norm
    assert private.shape == delta.shape and private.dtype == torch.float32


def test_disabled_is_the_non_private_identity_path() -> None:
    cfg = DPConfig(clip_norm=_CLIP, noise_multiplier=1.0, enabled=False)
    delta = _unit(4) * 5.0
    out, norm = privatize(delta, cfg, torch.Generator().manual_seed(0))
    assert torch.equal(out, delta.to(torch.float32))  # unchanged: no clip, no noise
    assert norm == pytest.approx(float(delta.norm()))


def test_dpconfig_validates_bounds() -> None:
    with pytest.raises(ConfigError):
        DPConfig(clip_norm=0.0, noise_multiplier=1.0)
    with pytest.raises(ConfigError):
        DPConfig(clip_norm=-1.0, noise_multiplier=1.0)
    with pytest.raises(ConfigError):
        DPConfig(clip_norm=_CLIP, noise_multiplier=-0.1)
    assert (
        DPConfig(clip_norm=_CLIP, noise_multiplier=0.0).enabled is True
    )  # sigma=0 allowed
