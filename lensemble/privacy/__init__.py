"""lensemble.privacy — the differential-privacy mechanism and accountant (docs/rfcs/RFC-0012).

The clip+noise mechanism (``INV-DP-BOUND``) privatizes the per-participant round update before it crosses
the boundary; the ``(eps, delta)`` accountant is a separate concern.
"""

from __future__ import annotations

from lensemble.privacy.dp import DPConfig, add_gaussian_noise, clip_delta, privatize

__all__ = ["DPConfig", "clip_delta", "add_gaussian_noise", "privatize"]
