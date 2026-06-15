"""Outer-step aggregation determinism — INV-AGG-DETERMINISM (RFC-0003 7 / 07 §2.5). Issue #39.

The Nesterov outer step is a bitwise-reproducible function of (deltas, prior global params): two
in-process runs and a fresh subprocess agree byte-for-byte; the fixed participant-id-sorted reduction is
input-order-independent; a non-bitwise recomputation raises NonDeterministicAggregation; Nesterov is
stable across a varying participant count C.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from lensemble.errors import LensembleErrorCode, NonDeterministicAggregation
from lensemble.federation import (
    OuterOptimizer,
    PseudoGradient,
    assert_bitwise_reproducible,
    build_pseudogradient,
)
from lensemble.federation.outer_optimizer import _content_hash

# A delta-build reproducible in this process and in a subprocess (seeded, fixed shapes/order).
_BUILD = """
import torch, hashlib
from lensemble.federation import OuterOptimizer, build_pseudogradient

def deltas(c):
    g = torch.Generator().manual_seed(0)
    return {
        f"p{i}": build_pseudogradient(
            {"encoder.w": torch.randn(6, generator=g), "predictor.w": torch.randn(4, generator=g)},
            dataset_root=bytes([i]) * 32, round_index=0,
        )
        for i in range(c)
    }

res = OuterOptimizer(lr=0.7, momentum=0.9).step(torch.zeros(10), deltas(3))
arr = res.detach().cpu().numpy()
print(hashlib.sha256(arr.astype(arr.dtype.newbyteorder("<")).tobytes()).hexdigest())
"""


def _deltas(c: int = 3) -> dict:
    g = torch.Generator().manual_seed(0)
    return {
        f"p{i}": build_pseudogradient(
            {
                "encoder.w": torch.randn(6, generator=g),
                "predictor.w": torch.randn(4, generator=g),
            },
            dataset_root=bytes([i]) * 32,
            round_index=0,
        )
        for i in range(c)
    }


def test_outer_step_is_bitwise_reproducible() -> None:
    g0 = torch.zeros(10)
    first = OuterOptimizer(lr=0.7, momentum=0.9).step(g0, _deltas())
    second = OuterOptimizer(lr=0.7, momentum=0.9).step(g0, _deltas())
    assert torch.equal(first, second)
    assert _content_hash(first) == _content_hash(second)

    # ...and a fresh subprocess reproduces the same content hash (no process-dependent state)
    out = subprocess.run(
        [sys.executable, "-c", _BUILD], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == _content_hash(first)


def test_fixed_reduction_order_is_input_order_independent() -> None:
    g0 = torch.zeros(10)
    deltas = _deltas()
    sorted_result = OuterOptimizer(lr=0.7, momentum=0.9).step(g0, deltas)
    shuffled = {
        k: deltas[k] for k in reversed(list(deltas))
    }  # different insertion order
    shuffled_result = OuterOptimizer(lr=0.7, momentum=0.9).step(g0, shuffled)
    assert torch.equal(
        sorted_result, shuffled_result
    )  # participant-id-sorted reduction


def test_nondeterministic_reduction_raises() -> None:
    a = torch.zeros(4)
    b = torch.zeros(4)
    b[0] = 1e-7  # a recomputation that does not reproduce byte-for-byte
    with pytest.raises(NonDeterministicAggregation) as exc:
        assert_bitwise_reproducible(a, b)
    assert exc.value.code == LensembleErrorCode.AGG_NONDETERMINISTIC
    assert exc.value.expected_hash != exc.value.got_hash  # type: ignore[attr-defined]


def test_nesterov_stable_across_varying_participant_count() -> None:
    g0 = torch.zeros(10)
    for c in (2, 3, 4, 8):
        result = OuterOptimizer(lr=0.7, momentum=0.9).step(g0, _deltas(c))
        assert result.shape == g0.shape
        assert bool(
            torch.isfinite(result).all()
        )  # an outer step proceeds with whatever C is present


def test_empty_deltas_rejected() -> None:
    with pytest.raises(NonDeterministicAggregation):
        OuterOptimizer(lr=0.7).step(torch.zeros(10), {})


def test_secure_agg_nondeterministic_field_sum_raises() -> None:
    # the secure-aggregation self-check aborts (never silently averages) on a non-reproducing field sum
    from lensemble.aggregation import assert_field_sum_reproducible

    a = torch.zeros(4, dtype=torch.int64)
    b = a.clone()
    b[0] = 1
    with pytest.raises(NonDeterministicAggregation) as exc:
        assert_field_sum_reproducible(a, b)
    assert exc.value.code == LensembleErrorCode.AGG_NONDETERMINISTIC


# --- the aggregation-layer per-outer-step self-check (#61): the callable the coordinator runs ---


def _outer_compute() -> torch.Tensor:
    # A pure recomputation of one outer step: a fresh optimizer each call (velocity not advanced twice).
    return OuterOptimizer(lr=0.7, momentum=0.9).step(torch.zeros(10), _deltas())


def test_aggregation_self_check_returns_verified_result() -> None:
    from hashlib import sha256

    from safetensors.torch import save

    from lensemble.aggregation import assert_outer_step_deterministic, flat_content_hash

    verified = assert_outer_step_deterministic(_outer_compute, round_index=5)
    assert torch.equal(verified, _outer_compute())
    # Identical safetensors content hash across recomputations (issue acceptance criterion).
    assert (
        sha256(save({"r": verified})).hexdigest()
        == sha256(save({"r": _outer_compute()})).hexdigest()
    )
    # ...and the fresh-subprocess outer step reproduces the same canonical content hash.
    out = subprocess.run(
        [sys.executable, "-c", _BUILD], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == flat_content_hash(verified)


def test_aggregation_self_check_raises_with_round_and_hashes() -> None:
    from lensemble.aggregation import assert_outer_step_deterministic

    calls = {"n": 0}

    def _flaky() -> torch.Tensor:
        calls["n"] += 1
        t = torch.zeros(4)
        if calls["n"] == 2:  # the recomputation does not reproduce byte-for-byte
            t[0] = 1e-9
        return t

    with pytest.raises(NonDeterministicAggregation) as exc:
        assert_outer_step_deterministic(_flaky, round_index=7)
    assert exc.value.code is LensembleErrorCode.AGG_NONDETERMINISTIC
    assert exc.value.round == 7  # type: ignore[attr-defined]
    assert exc.value.expected_hash != exc.value.got_hash  # type: ignore[attr-defined]
    assert exc.value.remediation


def test_aggregation_determinism_self_check_never_swallows() -> None:
    # INV-AGG-DETERMINISM is fail-closed: the self-check module has no try/except that could hide a
    # nondeterministic step — the error always propagates.
    from pathlib import Path

    import lensemble.aggregation.determinism as det

    src = Path(det.__file__).read_text(encoding="utf-8")
    assert "except" not in src and "try:" not in src


# --- Layer-3 Procrustes backstop fed into the SAME outer step stays bitwise-reproducible (#18) ---
#
# The backstop (lensemble.gauge.backstop.procrustes_backstop) realigns each over-threshold participant's
# predictor delta as a PURE LINEAR operation, then re-flattens the (possibly) aligned grouped delta into a
# PseudoGradient and feeds it to the SAME OuterOptimizer.step. RFC-0002 §5: this stays bitwise-deterministic
# (INV-AGG-DETERMINISM) and a degenerate Procrustes (clamp-and-skip) keeps the round alive.

import math  # noqa: E402

_BD = 4  # backstop latent dim d
_BWIDTH = 6  # predictor width
_BN = 64  # probe landmarks (k >> d, well-conditioned)
_BTHRESH = 15.0


def _b_rot(angle_deg: float, d: int = _BD) -> torch.Tensor:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    q = torch.eye(d)
    q[0, 0], q[0, 1], q[1, 0], q[1, 1] = c, -s, s, c
    return q


def _b_grouped_delta(seed: int) -> dict[str, torch.Tensor]:
    """A toy encoder.*/predictor.* grouped delta with the real predictor.* names/shapes."""
    g = torch.Generator().manual_seed(seed)
    return {
        "encoder.norm.weight": torch.randn(_BD, generator=g),
        "encoder.norm.bias": torch.randn(_BD, generator=g),
        "predictor.in_proj.weight": torch.randn(_BWIDTH, _BD, generator=g),
        "predictor.in_proj.bias": torch.randn(_BWIDTH, generator=g),
        "predictor.out_proj.weight": torch.randn(_BD, _BWIDTH, generator=g),
        "predictor.out_proj.bias": torch.randn(_BD, generator=g),
        "predictor.norm.weight": torch.randn(_BWIDTH, generator=g),
        "predictor.norm.bias": torch.randn(_BWIDTH, generator=g),
    }


def _b_e_ref() -> torch.Tensor:
    g = torch.Generator().manual_seed(424242)
    return torch.randn(_BN, _BD, generator=g)


def _b_flatten(grouped: dict) -> PseudoGradient:
    """Flatten an aligned grouped delta into a PseudoGradient via build_pseudogradient (canonical order)."""
    return build_pseudogradient(grouped, dataset_root=b"\x07" * 32, round_index=0)


def _backstop_then_outer_step() -> torch.Tensor:
    """Run procrustes_backstop on two participants (one above, one below τ) then ONE outer step (pure)."""
    from lensemble.gauge import procrustes_backstop

    e_ref = _b_e_ref()
    deltas = {"c0": _b_grouped_delta(31), "c1": _b_grouped_delta(32)}
    embeddings = {
        "c0": e_ref @ _b_rot(40.0),
        "c1": e_ref @ _b_rot(3.0),
    }  # c0 fires, c1 does not
    aligned = procrustes_backstop(
        deltas, embeddings, e_ref, threshold_deg=_BTHRESH, singular_floor=1e-6
    )
    updates = {pid: _b_flatten(aligned[pid]) for pid in aligned}
    # The flat θ⊕φ global params length is the sum of the grouped delta numels (canonical order).
    n = sum(t.numel() for t in deltas["c0"].values())
    return OuterOptimizer(lr=0.7, momentum=0.9).step(torch.zeros(n), updates)


def test_backstop_then_outer_step_is_bitwise_reproducible() -> None:
    from hashlib import sha256

    from safetensors.torch import save

    first = _backstop_then_outer_step()
    second = _backstop_then_outer_step()
    # torch.equal AND an identical safetensors content hash (the issue acceptance criterion).
    assert torch.equal(first, second)
    assert (
        sha256(save({"r": first})).hexdigest()
        == sha256(save({"r": second})).hexdigest()
    )
    assert _content_hash(first) == _content_hash(second)


def test_backstop_degenerate_injection_keeps_round_alive_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from lensemble.gauge import procrustes_backstop

    e_ref = _b_e_ref()
    # A rank-deficient embedding (a zeroed column) makes M = T^T S degenerate after the relaxed retry too,
    # so the backstop SKIPS this participant (keeps the unaligned delta) — the round stays alive.
    deg = e_ref.clone()
    deg[:, _BD - 1] = 0.0
    ref_deg = e_ref.clone()
    ref_deg[:, _BD - 1] = 0.0
    deltas = {"c0": _b_grouped_delta(33)}

    with caplog.at_level(logging.WARNING, logger="lensemble.gauge.backstop"):
        aligned = procrustes_backstop(
            deltas, {"c0": deg}, ref_deg, threshold_deg=_BTHRESH, singular_floor=1e-6
        )

    # The unaligned delta survived; the outer step still commits a finite, reproducible result.
    for name, tensor in deltas["c0"].items():
        assert torch.equal(aligned["c0"][name], tensor)
    n = sum(t.numel() for t in deltas["c0"].values())
    updates = {"c0": _b_flatten(aligned["c0"])}
    r1 = OuterOptimizer(lr=0.7, momentum=0.9).step(torch.zeros(n), updates)
    r2 = OuterOptimizer(lr=0.7, momentum=0.9).step(torch.zeros(n), updates)
    assert bool(torch.isfinite(r1).all())
    assert torch.equal(r1, r2)  # the surviving round is still bitwise-reproducible
    # ...and the clamp-and-skip logged at WARN naming gauge/procrustes_residual.
    assert any(
        rec.levelno == logging.WARNING and "gauge/procrustes_residual" in rec.message
        for rec in caplog.records
    )
