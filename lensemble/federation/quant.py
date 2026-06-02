"""lensemble.federation.quant — optional int8 wire quantization of the pseudo-gradient (RFC-0003 §6).

Symmetric per-tensor int8 quantization of the flat fp32 ``PseudoGradient.delta`` to cut outer-step
bandwidth (per INTELLECT-1's int8 all-reduce). It is **orthogonal to the gauge** and **not credited as
privacy**: the round-trip introduces a bounded, data-independent perturbation, nothing more.

Fixed ordering (RFC-0012 §6): quantization operates on the **already clipped-and-noised** ``Δ_c`` and
**before** secure-aggregation masking. The dequantized sum stays on the deterministic reduction path —
``quantize_int8``/``dequantize_int8`` are pure, integer-then-fixed-scale operations, so the dequantized
``Δ_c`` still passes the outer-step determinism self-check (``INV-AGG-DETERMINISM``); a failure there
raises ``NonDeterministicAggregation`` on the outer step, not here.

The feature is config-gated (``federation.quantize_pseudo_gradient``, default off) and stays optional
until its round-trip error is validated on a real boundary (Stage C / v0.3); a quantized wire object
records the fact in ``PseudoGradient.quantized``.

Round-trip error bound: with a symmetric scale ``s = max|Δ| / 127``, rounding moves each element by at
most ``s/2``, so ``‖dequantize(quantize(Δ)) − Δ‖₂ ≤ sqrt(d)·max|Δ| / 254`` (the per-element half-step,
summed in quadrature). The typical error is far smaller (RMS ≈ ``s/√12``); the bound is the worst case.
"""

from __future__ import annotations

import torch
from torch import Tensor

_INT8_MAX = 127  # symmetric signed range [-127, 127]


def quantize_int8(delta: Tensor) -> tuple[Tensor, float]:
    """Symmetric per-tensor int8 quantization of a flat fp32 ``delta`` -> ``(codes int8, scale fp32)``.

    ``scale = max|delta| / 127``; codes are ``round(delta / scale)`` clamped to ``[-127, 127]``. An
    all-zero (or empty) delta uses ``scale = 1.0`` so dequantization is exact and lossless.
    """
    delta = delta.detach().to(torch.float32)
    max_abs = float(delta.abs().max()) if delta.numel() else 0.0
    scale = max_abs / _INT8_MAX if max_abs > 0.0 else 1.0
    codes = torch.clamp(torch.round(delta / scale), -_INT8_MAX, _INT8_MAX).to(
        torch.int8
    )
    return codes, scale


def dequantize_int8(codes: Tensor, scale: float) -> Tensor:
    """Reconstruct a flat fp32 delta from int8 ``codes`` and the fp32 ``scale``."""
    return codes.to(torch.float32) * scale


def int8_roundtrip_l2_bound(delta: Tensor) -> float:
    """The documented worst-case L2 round-trip error bound ``sqrt(d)·max|Δ| / 254`` for ``delta``."""
    n = delta.numel()
    if n == 0:
        return 0.0
    max_abs = float(delta.detach().abs().max())
    return (n**0.5) * max_abs / (2.0 * _INT8_MAX)


def wire_roundtrip(delta: Tensor) -> Tensor:
    """Apply the int8 wire round-trip ``dequantize_int8(quantize_int8(delta))`` (a lossy, bounded copy).

    Deterministic: a pure function of ``delta`` (no RNG), so it does not perturb ``INV-AGG-DETERMINISM``.
    """
    codes, scale = quantize_int8(delta)
    return dequantize_int8(codes, scale)
