"""lensemble.privacy.dp — the per-participant clip+noise DP mechanism (RFC-0012 1, INV-DP-BOUND).

Update-level differential privacy on the one object that crosses the participant boundary: clip the
released pseudo-gradient ``Delta_c`` to a fixed L2 bound ``C_clip`` (the per-record sensitivity that makes
the noise calibration sound, ``INV-DP-BOUND``), then add isotropic Gaussian noise
``N(0, (sigma * C_clip)^2 I)``. The clip-then-noise ordering is pinned at the protocol level (RFC-0003 4).

Determinism (RFC-0012 4): clip is a pure, device-agnostic function; noise is deterministic only given its
seeded ``torch.Generator`` (derived from ``(root_seed, round_index, participant_id)`` by the caller and
recorded in the ``RunManifest``). Clip and noise are computed in fp32. ``noise_multiplier = 0`` disables
the noise but is NOT differentially private.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from lensemble.errors import ConfigError, LensembleErrorCode


@dataclass(frozen=True)
class DPConfig:
    """Differential-privacy mechanism config (RFC-0012 1). Validated at construction.

    ``clip_norm`` is ``C_clip`` (the L2 sensitivity bound), ``noise_multiplier`` is ``sigma``. The
    ``(target_epsilon, target_delta)`` are recorded for the accountant (owned elsewhere). ``enabled=False``
    is the non-private honesty path (no clip, no noise).
    """

    clip_norm: float
    noise_multiplier: float
    target_epsilon: float = 8.0
    target_delta: float = 1e-5
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.clip_norm <= 0:
            raise ConfigError(
                f"clip_norm (C_clip) must be > 0, got {self.clip_norm}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="set a positive L2 clip bound C_clip",
            )
        if self.noise_multiplier < 0:
            raise ConfigError(
                f"noise_multiplier (sigma) must be >= 0, got {self.noise_multiplier}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="set sigma >= 0 (0 disables noise but is not DP)",
            )


def clip_delta(delta: Tensor, clip_norm: float) -> tuple[Tensor, float]:
    """Project ``delta`` to the L2 ball of radius ``clip_norm`` (``INV-DP-BOUND``); pure, deterministic.

    Returns ``(clipped_delta, post_clip_norm)`` in fp32. A norm at or below ``clip_norm`` is returned
    unchanged; a larger norm is scaled to exactly ``clip_norm``; the zero vector is handled without a
    division by zero. Postcondition: ``post_clip_norm <= clip_norm`` (within fp32 tolerance).
    """
    d = delta.to(torch.float32)
    norm = float(d.norm())
    if norm > clip_norm:
        clipped = d * (clip_norm / norm)
        return clipped, float(clipped.norm())
    return d, norm


def add_gaussian_noise(
    delta: Tensor,
    clip_norm: float,
    noise_multiplier: float,
    generator: torch.Generator,
) -> Tensor:
    """Add ``N(0, (noise_multiplier * clip_norm)^2 I)`` from a seeded generator; independent per draw.

    Deterministic given ``generator`` (whose ``(root_seed, round_index, participant_id)`` derivation is
    recorded in the ``RunManifest``). Noise is computed in fp32.
    """
    d = delta.to(torch.float32)
    std = noise_multiplier * clip_norm
    noise = torch.randn(d.shape, generator=generator, dtype=torch.float32) * std
    return d + noise


def privatize(
    delta: Tensor, cfg: DPConfig, generator: torch.Generator
) -> tuple[Tensor, float]:
    """Clip then noise (RFC-0012 1): returns ``(private_delta, post_clip_norm)``.

    ``post_clip_norm`` is the post-clip, pre-noise norm (``<= C_clip``) recorded as
    ``PseudoGradient.l2_norm``. With ``cfg.enabled is False`` returns ``(delta, ||delta||)`` unchanged —
    the non-private honesty path (recorded honestly, never silently treated as private).
    """
    if not cfg.enabled:
        d = delta.to(torch.float32)
        return d, float(d.norm())
    clipped, post_clip_norm = clip_delta(delta, cfg.clip_norm)
    private = add_gaussian_noise(
        clipped, cfg.clip_norm, cfg.noise_multiplier, generator
    )
    return private, post_clip_norm
