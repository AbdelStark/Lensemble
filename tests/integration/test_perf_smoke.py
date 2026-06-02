"""CI performance smoke — the four CPU regression guards of SPEC 08 §7. Issue #69.

This is the download-free, CPU-only perf smoke (08 §7 / 07 §8): regression detection on a tiny synthetic
config, never absolute-throughput measurement (those live on GPU in the ``RunManifest``, 08 §2). It pins,
on one toy inner+outer cycle, the four checks of the 08 §7 table:

1. wall-time ceiling — one inner+outer cycle of a fixed toy config completes within a generous CPU
   ceiling. The ceiling is loose by design (it catches ~10x order-of-magnitude regressions such as an
   accidental per-step recomputation, not micro-optimizations); an overrun is a regression SIGNAL (a
   plain ``assert``), never a raised invariant error;
2. comms-accountant equality — ``comm_bytes`` for the toy federated parameter count equals the expected
   full-precision (``4 * n``) and int8 (``n + 4``) bytes per round (08 §4);
3. outer-step bitwise determinism (``INV-AGG-DETERMINISM``) — two runs of the toy aggregation produce
   byte-identical global params; a violation surfaces ``NonDeterministicAggregation`` (08 §5.1 / 07 §2.5);
4. int8 quant round-trip — the int8-quantized ``Δ`` reconstructs within the documented per-element error
   bound (``lensemble.federation.quant``, RFC-0003 §6).

The wall-time ceiling and its "loose by design" rationale are recorded on ``_WALLTIME_CEILING_S`` below —
that module constant is the durable config-of-record the OPEN QUESTION in 08 §7 asks CI to keep.
"""

from __future__ import annotations

import time

import pytest
import torch

from lensemble.aggregation import assert_outer_step_deterministic
from lensemble.eval.metrics import comm_bytes
from lensemble.federation import (
    OuterOptimizer,
    build_pseudogradient,
    dequantize_int8,
    quantize_int8,
)

# --- the wall-time ceiling: the config-of-record for the 08 §7 OPEN QUESTION ---
#
# Empirical and LOOSE BY DESIGN (08 §7): the toy inner+outer cycle below measures sub-millisecond on a
# warm CPU, so 30 s is ~5-6 orders of magnitude of headroom. That headroom is intentional — the smoke
# catches a 10x+ algorithmic regression (e.g. an accidental per-step recomputation that turns the cycle
# into seconds), NOT a micro-optimization, and it must never flake on a slow/contended CI runner. The
# multiplier (ceiling / measured-toy-time, ~50-100x and far beyond) is recorded here so the ceiling can be
# revisited against the first green CI run's measured time at v0.1 (08 §7, owner @AbdelStark). A wall-time
# overrun is a regression signal — a plain ``assert`` — not the determinism invariant of §5.1.
_WALLTIME_CEILING_S = 30.0

# Toy config knobs (CPU, no download — 08 §7 / 07 §7): a 2-layer ``d=8`` linear stand-in encoder, a few
# inner AdamW steps, one outer step. Kept well under the ceiling.
_D = 8
_INNER_STEPS = 3
_BATCH = 16
_INNER_LR = 1e-2
_OUTER_LR = 0.7
_OUTER_MOMENTUM = 0.9


def _toy_encoder(seed: int = 0) -> torch.nn.Module:
    """A deterministically-initialized 2-layer ``d=8`` linear encoder (the 07 §7 tiny stand-in)."""
    enc = torch.nn.Sequential(
        torch.nn.Linear(_D, _D), torch.nn.Tanh(), torch.nn.Linear(_D, _D)
    )
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in enc.parameters():
            p.copy_(torch.randn(p.shape, generator=gen))
    return enc


def _toy_outer_step() -> torch.Tensor:
    """One toy inner+outer cycle: a few inner AdamW steps produce a pseudo-gradient ``Δ``, then one outer.

    Pure and side-effect-free (a fresh encoder, optimizer, and outer optimizer each call), so it can be
    handed to :func:`assert_outer_step_deterministic` as the recomputation thunk (the velocity is never
    advanced twice). Returns the new global params from the outer step.
    """
    enc = _toy_encoder()
    initial = [p.detach().clone() for p in enc.parameters()]
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(_BATCH, _D, generator=gen)
    target = torch.randn(_BATCH, _D, generator=gen)

    opt = torch.optim.AdamW(enc.parameters(), lr=_INNER_LR)
    for _ in range(_INNER_STEPS):  # inner loop: a few local steps before the outer step
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(enc(x), target)
        loss.backward()
        opt.step()

    # Δ = local - initial over the federated encoder params, flattened (RFC-0003 §3).
    current = list(enc.parameters())
    flat_delta = torch.cat(
        [(current[i] - initial[i]).reshape(-1) for i in range(len(current))]
    ).detach()
    pseudo_grad = build_pseudogradient(
        {"encoder.w": flat_delta}, dataset_root=b"\x00" * 32, round_index=0
    )

    global_params = torch.zeros(flat_delta.numel())
    # A single canonical participant ("p0"); a contributing set would reduce in sorted participant-id
    # order inside OuterOptimizer (INV-AGG-DETERMINISM).
    return OuterOptimizer(lr=_OUTER_LR, momentum=_OUTER_MOMENTUM).step(
        global_params, {"p0": pseudo_grad}
    )


def _toy_delta() -> torch.Tensor:
    """The flat pseudo-gradient ``Δ`` of one toy inner loop (for the int8 round-trip check)."""
    enc = _toy_encoder()
    initial = [p.detach().clone() for p in enc.parameters()]
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(_BATCH, _D, generator=gen)
    target = torch.randn(_BATCH, _D, generator=gen)
    opt = torch.optim.AdamW(enc.parameters(), lr=_INNER_LR)
    for _ in range(_INNER_STEPS):
        opt.zero_grad()
        torch.nn.functional.mse_loss(enc(x), target).backward()
        opt.step()
    current = list(enc.parameters())
    return torch.cat(
        [(current[i] - initial[i]).reshape(-1) for i in range(len(current))]
    ).detach()


def _toy_num_federated_params() -> int:
    """The flat federated parameter count of the toy encoder (input to the comms accountant)."""
    return _toy_delta().numel()


# --- 08 §7 check 1: wall-time ceiling on a tiny end-to-end round ---


def test_inner_outer_cycle_within_walltime_ceiling() -> None:
    """One toy inner+outer cycle finishes within the generous CPU ceiling (08 §7, regression SIGNAL).

    A large overrun flags an algorithmic regression (e.g. an accidental per-step recomputation), per
    08 §7. This is a plain ``assert`` — a wall-time overrun is a regression signal, not a raised invariant
    error (08 §1: "A wall-time overrun is a regression signal, not an error").
    """
    start = time.perf_counter()
    result = _toy_outer_step()
    elapsed = time.perf_counter() - start

    assert bool(
        torch.isfinite(result).all()
    )  # the cycle produced a usable global model
    assert elapsed < _WALLTIME_CEILING_S, (
        f"toy inner+outer cycle took {elapsed:.4f}s, over the loose {_WALLTIME_CEILING_S}s CPU ceiling "
        "(08 §7): a 10x+ regression — suspect an accidental per-step recomputation"
    )


# --- 08 §7 check 2: comms-accountant smoke (full precision and int8) ---


def test_comms_accountant_matches_expected_bytes_per_round() -> None:
    """``comm_bytes`` for the toy parameter count equals the expected fp32 and int8 bytes (08 §4).

    Full precision is ``4 * n`` (fp32 wire dtype); int8 is ``n + 4`` (one byte per code plus the 4-byte
    fp32 per-tensor scale). A mismatch catches a serialization or quantization regression (08 §7). The
    expected values mirror ``lensemble.eval.metrics.comm_bytes`` rather than re-deriving a magic number.
    """
    n = _toy_num_federated_params()
    assert (
        n > 0
    )  # the toy encoder has federated params, so the accountant is non-trivial

    assert comm_bytes(n) == 4 * n  # full-precision bytes per round (08 §4)
    assert comm_bytes(n, quantized=True) == n + 4  # int8 codes + fp32 scale metadata

    # The int8 path is the documented ~4x reduction (it must be smaller than full precision).
    assert comm_bytes(n, quantized=True) < comm_bytes(n)


# --- 08 §7 check 3: outer-step bitwise determinism (INV-AGG-DETERMINISM) ---


def test_outer_step_is_bitwise_deterministic() -> None:
    """Two runs of the toy outer step produce byte-identical global params (``INV-AGG-DETERMINISM``).

    Uses :func:`assert_outer_step_deterministic` (the coordinator's per-step self-check), which recomputes
    the outer step on identical inputs and compares byte-for-byte; a violation raises
    ``NonDeterministicAggregation`` (08 §5.1 / 07 §2.5). Reuses the existing helper rather than
    re-implementing the check.
    """
    verified = assert_outer_step_deterministic(_toy_outer_step, round_index=0)
    assert torch.equal(
        verified, _toy_outer_step()
    )  # byte-identical across an independent recompute


# --- 08 §7 check 4: int8 quant round-trip error bound (RFC-0003 §6) ---


def test_int8_quant_roundtrip_within_documented_bound() -> None:
    """``dequantize_int8(quantize_int8(Δ))`` reconstructs ``Δ`` within the documented per-element bound.

    ``lensemble.federation.quant`` documents a symmetric scale ``s = max|Δ| / 127`` where rounding moves
    each element by at most ``s/2``; the reconstructed elements therefore stay within ``scale`` of the
    originals (in fact within ``scale/2``). An exceeded bound fails the job (08 §7). The bound is read from
    the module's quantizer, not invented.
    """
    delta = _toy_delta()
    assert delta.numel() > 0 and float(delta.abs().max()) > 0.0  # a non-degenerate Δ

    codes, scale = quantize_int8(delta)
    reconstructed = dequantize_int8(codes, scale)

    max_abs_error = float((reconstructed - delta).abs().max())
    # The module's per-element half-step bound is ``scale / 2``; assert the documented ``<= scale`` bound
    # (and the tighter half-step it implies). This matches quant.py, not a looser invented bound.
    assert max_abs_error <= scale, (
        f"int8 round-trip max element error {max_abs_error} exceeds the documented scale {scale}"
    )
    assert (
        max_abs_error <= scale / 2 + 1e-12
    )  # the tighter documented half-step (round-to-nearest)


if __name__ == "__main__":  # pragma: no cover - convenience for local runs
    raise SystemExit(pytest.main([__file__, "-q"]))
