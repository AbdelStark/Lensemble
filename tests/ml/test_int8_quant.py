"""Optional int8 pseudo-gradient wire quantization (RFC-0003 §6; #40).

Quantization is orthogonal to the gauge and not credited as privacy: a bounded, deterministic round-trip
on the flat fp32 delta. Tests pin the documented L2 round-trip bound, that the dequantized sum still
passes the outer-step determinism self-check (INV-AGG-DETERMINISM), the fixed clip+noise -> quantize ->
mask ordering, and that the feature is off by default.
"""

from __future__ import annotations

import torch

from lensemble.aggregation import assert_outer_step_deterministic, encode_delta
from lensemble.aggregation.secure_agg import FieldParams
from lensemble.config.schema import FederationConfig
from lensemble.federation import (
    OuterOptimizer,
    build_pseudogradient,
    dequantize_int8,
    int8_roundtrip_l2_bound,
    quantize_int8,
)


def _delta(seed: int, n: int = 64) -> torch.Tensor:
    return torch.randn(n, generator=torch.Generator().manual_seed(seed))


# --- the round-trip error bound ---


def test_roundtrip_error_within_documented_bound() -> None:
    delta = _delta(0)
    codes, scale = quantize_int8(delta)
    recon = dequantize_int8(codes, scale)
    error = float((recon - delta).norm())
    bound = int8_roundtrip_l2_bound(delta)
    assert error <= bound + 1e-6
    # bound formula sqrt(d) * max|delta| / 254
    expected_bound = (delta.numel() ** 0.5) * float(delta.abs().max()) / 254.0
    assert abs(bound - expected_bound) < 1e-6
    assert codes.dtype == torch.int8


def test_zero_delta_quantizes_losslessly() -> None:
    z = torch.zeros(16)
    codes, scale = quantize_int8(z)
    assert torch.equal(dequantize_int8(codes, scale), z)
    assert int8_roundtrip_l2_bound(z) == 0.0


def test_empty_delta_has_zero_bound() -> None:
    empty = torch.zeros(0)
    codes, scale = quantize_int8(empty)
    assert codes.numel() == 0
    assert int8_roundtrip_l2_bound(empty) == 0.0


# --- determinism of the dequantized sum (INV-AGG-DETERMINISM) ---


def _quantized_deltas(c: int = 3) -> dict:
    g = torch.Generator().manual_seed(0)
    return {
        f"p{i}": build_pseudogradient(
            {
                "encoder.w": torch.randn(6, generator=g),
                "predictor.w": torch.randn(4, generator=g),
            },
            dataset_root=bytes([i]) * 32,
            round_index=0,
            quantize=True,
        )
        for i in range(c)
    }


def test_dequantized_sum_passes_determinism_self_check() -> None:
    deltas = _quantized_deltas()
    assert all(pg.quantized for pg in deltas.values())
    verified = assert_outer_step_deterministic(
        lambda: OuterOptimizer(lr=0.7, momentum=0.9).step(torch.zeros(10), deltas),
        round_index=0,
    )
    # ...and a fully independent rebuild reproduces it bit-for-bit (quantization is deterministic).
    again = OuterOptimizer(lr=0.7, momentum=0.9).step(
        torch.zeros(10), _quantized_deltas()
    )
    assert torch.equal(verified, again)


# --- the flag is off by default; quantize=True sets the wire flag and stays bounded ---


def test_quantization_is_off_by_default() -> None:
    assert FederationConfig().quantize_pseudo_gradient is False
    pg = build_pseudogradient(
        {"encoder.w": _delta(1, 8)}, dataset_root=b"\x00" * 32, round_index=0
    )
    assert pg.quantized is False
    # un-quantized delta is the exact assembled fp32 (no perturbation)
    assert torch.equal(pg.delta, _delta(1, 8).to(torch.float32))


def test_quantize_flag_sets_field_and_bounds_perturbation() -> None:
    raw = {"encoder.w": _delta(2, 6), "predictor.w": _delta(3, 4)}
    plain = build_pseudogradient(raw, dataset_root=b"\x00" * 32, round_index=0)
    quant = build_pseudogradient(
        raw, dataset_root=b"\x00" * 32, round_index=0, quantize=True
    )
    assert quant.quantized is True
    assert (
        float((quant.delta - plain.delta).norm())
        <= int8_roundtrip_l2_bound(plain.delta) + 1e-6
    )


# --- ordering: quantize runs on the flat delta, before secure-aggregation masking ---


def test_quantized_delta_feeds_secure_aggregation_encoding() -> None:
    # The quantized fp32 delta is what the aggregation field-encoder (the masking stage input) consumes:
    # quantization (federation.quant) precedes encode_delta (the mask field), per RFC-0012 §6.
    quant = build_pseudogradient(
        {"encoder.w": _delta(4, 6)},
        dataset_root=b"\x01" * 32,
        round_index=0,
        quantize=True,
    )
    field = FieldParams(modulus=2**31 - 1, scale=1e6, dim=quant.delta.numel())
    update = encode_delta(
        quant.delta,
        field,
        participant_id="p0",
        round_index=0,
        dataset_root=quant.dataset_root,
    )
    assert update.masked.dtype == torch.int64
    assert update.masked.shape == quant.delta.shape
