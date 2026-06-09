"""Layer-3 Procrustes re-alignment backstop — the activation-space realization (RFC-0002 §5). Issue #18.

The backstop, immediately before the outer step, conjugates each over-threshold participant's predictor
delta by ``Q_c* = procrustes_align(f_c(P), E_ref)`` as a PURE LINEAR operation, leaving the encoder delta
byte-identical (the maintainer's recorded activation-space decision: a LayerNorm-terminated encoder has no
terminal linear to absorb ``Q`` into). Below threshold the delta is byte-identical; a degenerate (k < d)
embedding triggers the clamp-and-skip path (the unaligned delta survives) with a WARN log; the backstop is
order-independent and dtype-preserving.
"""

from __future__ import annotations

import logging
import math

import pytest
import torch

from lensemble.gauge import (
    procrustes_backstop,
    realign_encoder_frame_delta,
    realign_predictor_delta,
)
from lensemble.gauge.drift import _rotation_angle_deg
from lensemble.gauge.procrustes import procrustes_align

_D = 4  # predictor latent dim d
_WIDTH = 6  # predictor width
_N = 64  # probe landmark count (k >> d, so M = T^T S is well-conditioned)
_THRESHOLD_DEG = 15.0  # gauge.frame_drift_threshold_deg default


def _rot(angle_deg: float, d: int = _D) -> torch.Tensor:
    """A single-plane proper rotation by ``angle_deg`` in the (0, 1) plane of R^d (det = +1)."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    q = torch.eye(d)
    q[0, 0], q[0, 1], q[1, 0], q[1, 1] = c, -s, s, c
    return q


def _toy_predictor_delta(seed: int = 0) -> dict[str, torch.Tensor]:
    """A toy predictor-group delta with the REAL predictor.* param names/shapes (the three rotated + rest)."""
    g = torch.Generator().manual_seed(seed)
    return {
        "predictor.in_proj.weight": torch.randn(_WIDTH, _D, generator=g),
        "predictor.in_proj.bias": torch.randn(_WIDTH, generator=g),
        "predictor.cond_proj.weight": torch.randn(_WIDTH, _D, generator=g),
        "predictor.cond_proj.bias": torch.randn(_WIDTH, generator=g),
        "predictor.out_proj.weight": torch.randn(_D, _WIDTH, generator=g),
        "predictor.out_proj.bias": torch.randn(_D, generator=g),
        "predictor.pos_embed": torch.randn(1, 3, _WIDTH, generator=g),
        "predictor.norm.weight": torch.randn(_WIDTH, generator=g),
        "predictor.norm.bias": torch.randn(_WIDTH, generator=g),
    }


def _toy_grouped_delta(seed: int = 0) -> dict[str, torch.Tensor]:
    """A toy grouped delta carrying BOTH encoder.* and predictor.* params (canonical naming).

    Carries the encoder terminal gauge surface ``encoder.frame_proj.weight`` ``(d, d)`` (#262) alongside the
    non-gauge encoder params (``patch_embed``/``norm``) so the encoder fold + the byte-identical pass-through
    of the rest are both exercised.
    """
    g = torch.Generator().manual_seed(seed + 1000)
    delta = {
        "encoder.patch_embed.weight": torch.randn(_WIDTH, _D, generator=g),
        "encoder.norm.weight": torch.randn(_D, generator=g),
        "encoder.norm.bias": torch.randn(_D, generator=g),
        "encoder.frame_proj.weight": torch.randn(_D, _D, generator=g),
    }
    delta.update(_toy_predictor_delta(seed))
    return delta


def _e_ref() -> torch.Tensor:
    """The reference frame E_ref (n, d), seeded so f_c(P) = E_ref @ Q_c builds a known-drift participant."""
    g = torch.Generator().manual_seed(99)
    return torch.randn(_N, _D, generator=g)


# --- realign_predictor_delta: the weight-expressible conjugation g_phi -> Q g_phi Q^T ---


def test_realign_predictor_delta_rotates_exactly_three_params() -> None:
    q = _rot(30.0)
    delta = _toy_predictor_delta(seed=1)
    out = realign_predictor_delta(delta, q)

    # The three conjugated params match the exact closed forms (RFC-0002 §5).
    assert torch.allclose(
        out["predictor.in_proj.weight"], delta["predictor.in_proj.weight"] @ q.T
    )
    assert torch.allclose(
        out["predictor.out_proj.weight"], q @ delta["predictor.out_proj.weight"]
    )
    assert torch.allclose(
        out["predictor.out_proj.bias"], q @ delta["predictor.out_proj.bias"]
    )

    # The three are ACTUALLY changed (a non-trivial rotation moved them).
    assert not torch.equal(
        out["predictor.in_proj.weight"], delta["predictor.in_proj.weight"]
    )
    assert not torch.equal(
        out["predictor.out_proj.weight"], delta["predictor.out_proj.weight"]
    )
    assert not torch.equal(
        out["predictor.out_proj.bias"], delta["predictor.out_proj.bias"]
    )

    # Everything else in the predictor delta is byte-identical.
    for name in (
        "predictor.in_proj.bias",
        "predictor.cond_proj.weight",
        "predictor.cond_proj.bias",
        "predictor.pos_embed",
        "predictor.norm.weight",
        "predictor.norm.bias",
    ):
        assert torch.equal(out[name], delta[name])


def test_realign_predictor_delta_identity_is_noop() -> None:
    q = torch.eye(_D)
    delta = _toy_predictor_delta(seed=2)
    out = realign_predictor_delta(delta, q)
    for name, tensor in delta.items():
        assert torch.allclose(out[name], tensor)


def test_realign_predictor_delta_does_not_mutate_input() -> None:
    q = _rot(40.0)
    delta = _toy_predictor_delta(seed=3)
    snapshot = {k: v.clone() for k, v in delta.items()}
    realign_predictor_delta(delta, q)
    for name in delta:
        assert torch.equal(delta[name], snapshot[name])  # input untouched


def test_realign_predictor_delta_preserves_dtype() -> None:
    q = _rot(25.0)
    delta = {k: v.to(torch.float64) for k, v in _toy_predictor_delta(seed=4).items()}
    out = realign_predictor_delta(delta, q)
    assert all(t.dtype == torch.float64 for t in out.values())


def test_realign_predictor_delta_rejects_non_square_q() -> None:
    with pytest.raises(ValueError):
        realign_predictor_delta(_toy_predictor_delta(), torch.randn(_D, _D + 1))


# --- realign_encoder_frame_delta: the encoder terminal-frame conjugation Δ <- Q @ Δ (#262) ---


def test_realign_encoder_frame_delta_rotates_only_frame_proj() -> None:
    q = _rot(30.0)
    delta = {
        "encoder.patch_embed.weight": torch.randn(
            _WIDTH, _D, generator=torch.Generator().manual_seed(1)
        ),
        "encoder.norm.weight": torch.randn(_D),
        "encoder.norm.bias": torch.randn(_D),
        "encoder.frame_proj.weight": torch.randn(_D, _D),
    }
    out = realign_encoder_frame_delta(delta, q)
    # The terminal frame surface is conjugated by Δ <- Q @ Δ (rotate the d output rows).
    assert torch.allclose(
        out["encoder.frame_proj.weight"], q @ delta["encoder.frame_proj.weight"]
    )
    assert not torch.equal(
        out["encoder.frame_proj.weight"], delta["encoder.frame_proj.weight"]
    )
    # Every other encoder param is byte-identical (the gauge lives only in the terminal frame).
    for name in (
        "encoder.patch_embed.weight",
        "encoder.norm.weight",
        "encoder.norm.bias",
    ):
        assert torch.equal(out[name], delta[name])


def test_realign_encoder_frame_delta_identity_is_noop() -> None:
    delta = {
        "encoder.frame_proj.weight": torch.randn(_D, _D),
        "encoder.norm.bias": torch.randn(_D),
    }
    out = realign_encoder_frame_delta(delta, torch.eye(_D))
    for name, tensor in delta.items():
        assert torch.allclose(out[name], tensor)


def test_realign_encoder_frame_delta_missing_frame_proj_passes_through() -> None:
    delta = {
        "encoder.norm.weight": torch.randn(_D),
        "encoder.norm.bias": torch.randn(_D),
    }
    out = realign_encoder_frame_delta(delta, _rot(40.0))
    for name, tensor in delta.items():
        assert torch.equal(out[name], tensor)


# --- procrustes_backstop: fires above threshold, byte-identical below, encoder always untouched ---


def test_above_threshold_conjugates_both_gauge_surfaces() -> None:
    e_ref = _e_ref()
    q_c = _rot(30.0)  # 30 deg > 15 deg threshold -> the backstop FIRES
    deltas = {"c0": _toy_grouped_delta(seed=5)}
    embeddings = {"c0": e_ref @ q_c}  # f_c(P) = E_ref @ Q_c (a 30-deg-rotated frame)

    out = procrustes_backstop(
        deltas, embeddings, e_ref, threshold_deg=_THRESHOLD_DEG, singular_floor=1e-6
    )

    aligned = out["c0"]
    original = deltas["c0"]

    # The recovered Q* aligning f_c(P) -> E_ref undoes Q_c, i.e. Q* ≈ Q_c^T (a 30-deg rotation back).
    q_star, _ = procrustes_align(embeddings["c0"], e_ref)
    assert abs(_rotation_angle_deg(q_star) - 30.0) < 1.0

    # The encoder terminal frame surface IS conjugated (Δ <- Q @ Δ), #262.
    assert not torch.equal(
        aligned["encoder.frame_proj.weight"], original["encoder.frame_proj.weight"]
    )
    assert torch.allclose(
        aligned["encoder.frame_proj.weight"],
        q_star @ original["encoder.frame_proj.weight"],
    )

    # The three predictor params ARE conjugated.
    assert not torch.equal(
        aligned["predictor.in_proj.weight"], original["predictor.in_proj.weight"]
    )
    assert not torch.equal(
        aligned["predictor.out_proj.weight"], original["predictor.out_proj.weight"]
    )
    assert not torch.equal(
        aligned["predictor.out_proj.bias"], original["predictor.out_proj.bias"]
    )
    assert torch.allclose(
        aligned["predictor.in_proj.weight"],
        original["predictor.in_proj.weight"] @ q_star.T,
    )
    assert torch.allclose(
        aligned["predictor.out_proj.weight"],
        q_star @ original["predictor.out_proj.weight"],
    )

    # The NON-gauge encoder params (patch_embed/norm) are byte-identical — only the terminal frame folds.
    for name in (
        "encoder.patch_embed.weight",
        "encoder.norm.weight",
        "encoder.norm.bias",
    ):
        assert torch.equal(aligned[name], original[name])

    # The non-rotated predictor params are byte-identical too.
    for name in (
        "predictor.in_proj.bias",
        "predictor.cond_proj.weight",
        "predictor.cond_proj.bias",
        "predictor.pos_embed",
        "predictor.norm.weight",
        "predictor.norm.bias",
    ):
        assert torch.equal(aligned[name], original[name])


def test_below_threshold_is_byte_identical() -> None:
    e_ref = _e_ref()
    q_c = _rot(5.0)  # 5 deg < 15 deg threshold -> the backstop does NOT fire
    deltas = {"c0": _toy_grouped_delta(seed=6)}
    embeddings = {"c0": e_ref @ q_c}

    out = procrustes_backstop(
        deltas, embeddings, e_ref, threshold_deg=_THRESHOLD_DEG, singular_floor=1e-6
    )

    # Un-fired: the WHOLE delta (encoder AND predictor) is byte-identical.
    for name, tensor in deltas["c0"].items():
        assert torch.equal(out["c0"][name], tensor)


def test_non_gauge_encoder_params_byte_identical_mixed_participants() -> None:
    e_ref = _e_ref()
    deltas = {
        "above": _toy_grouped_delta(seed=7),
        "below": _toy_grouped_delta(seed=8),
    }
    embeddings = {
        "above": e_ref @ _rot(40.0),  # fires
        "below": e_ref @ _rot(3.0),  # does not fire
    }
    out = procrustes_backstop(
        deltas, embeddings, e_ref, threshold_deg=_THRESHOLD_DEG, singular_floor=1e-6
    )
    # The NON-gauge encoder params (patch_embed/norm) are byte-identical for BOTH participants — only the
    # terminal frame surface (encoder.frame_proj.weight) and the predictor I/O ever fold.
    for pid in deltas:
        for name in (
            "encoder.patch_embed.weight",
            "encoder.norm.weight",
            "encoder.norm.bias",
        ):
            assert torch.equal(out[pid][name], deltas[pid][name])
    # The above-threshold participant's encoder frame + predictor ARE changed; the below-threshold one's NOT.
    assert not torch.equal(
        out["above"]["encoder.frame_proj.weight"],
        deltas["above"]["encoder.frame_proj.weight"],
    )
    assert not torch.equal(
        out["above"]["predictor.out_proj.weight"],
        deltas["above"]["predictor.out_proj.weight"],
    )
    assert torch.equal(
        out["below"]["encoder.frame_proj.weight"],
        deltas["below"]["encoder.frame_proj.weight"],
    )
    assert torch.equal(
        out["below"]["predictor.out_proj.weight"],
        deltas["below"]["predictor.out_proj.weight"],
    )


def test_backstop_preserves_encoder_rank_where_unaligned_mean_collapses() -> None:
    """#262 acceptance: aggregating mutually-rotated high-rank encoder frames with the backstop ON preserves
    the committed frame's rank, where the unaligned mean collapses it.

    Each participant carries the SAME high-rank terminal-frame signal ``B`` but expressed in a different
    output gauge ``R_c`` (``encoder.frame_proj.weight`` delta = ``R_c @ B``), and its probe frame is
    ``E_ref @ R_c`` so the backstop recovers ``Q_c* = R_c^T``. Conjugating (``Δ <- Q_c* @ Δ``) maps every
    participant's frame delta back to ``B``, so the MEAN is ``B`` (full rank). Without the backstop the mean
    is ``(mean_c R_c) @ B`` — averaging mutually-rotated frames cancels and collapses the rank (RFC-0002 §2.1).
    """
    from lensemble.eval.jepa_metrics import effective_rank

    d = 12
    n = 96  # probe rows (k >> d so M = T^T S is well-conditioned)
    gen = torch.Generator().manual_seed(7)
    e_ref = torch.randn(n, d, generator=gen)
    base = torch.randn(
        d, d, generator=gen
    )  # the shared high-rank terminal-frame signal B

    def _rand_so(seed: int) -> torch.Tensor:
        """A Haar-ish proper rotation in SO(d): QR of a random matrix, sign-fixed to det = +1."""
        a = torch.randn(d, d, generator=torch.Generator().manual_seed(seed))
        q, _ = torch.linalg.qr(a)
        if torch.det(q) < 0:
            q[:, 0] = -q[:, 0]
        return q

    # Six participants, each carrying the SAME high-rank frame signal in a DIFFERENT, widely-spread output
    # gauge (random SO(d) rotations, all far above the 15-deg threshold).
    rotations = {f"c{i}": _rand_so(100 + i) for i in range(6)}
    deltas = {
        pid: {"encoder.frame_proj.weight": r @ base} for pid, r in rotations.items()
    }
    embeddings = {pid: e_ref @ r for pid, r in rotations.items()}

    def _mean_frame(grouped: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        stacked = torch.stack(
            [grouped[pid]["encoder.frame_proj.weight"] for pid in sorted(grouped)]
        )
        return stacked.mean(dim=0)

    unaligned_rank = effective_rank(_mean_frame(deltas))

    aligned = procrustes_backstop(
        deltas, embeddings, e_ref, threshold_deg=_THRESHOLD_DEG, singular_floor=1e-6
    )
    aligned_rank = effective_rank(_mean_frame(aligned))
    base_rank = effective_rank(base)

    # The backstop restores the mean frame to the signal rank; the unaligned mean collapses well below it.
    # Measured on the seeded config: unaligned ~5.58, aligned ~7.12 (== base_rank) — a ~1.5 gap.
    assert aligned_rank > unaligned_rank + 0.8, (
        f"backstop should preserve rank: aligned {aligned_rank:.3f} vs unaligned {unaligned_rank:.3f}"
    )
    assert aligned_rank >= 0.95 * base_rank, (
        f"aligned mean rank {aligned_rank:.3f} should approach the signal rank {base_rank:.3f}"
    )


# --- order-independence + determinism ---


def test_backstop_is_order_independent() -> None:
    e_ref = _e_ref()
    deltas = {
        "c0": _toy_grouped_delta(seed=10),
        "c1": _toy_grouped_delta(seed=11),
        "c2": _toy_grouped_delta(seed=12),
    }
    embeddings = {
        "c0": e_ref @ _rot(30.0),
        "c1": e_ref @ _rot(2.0),
        "c2": e_ref @ _rot(50.0),
    }
    out_forward = procrustes_backstop(
        deltas, embeddings, e_ref, threshold_deg=_THRESHOLD_DEG, singular_floor=1e-6
    )
    permuted = {k: deltas[k] for k in ("c2", "c0", "c1")}  # different dict order
    out_permuted = procrustes_backstop(
        permuted, embeddings, e_ref, threshold_deg=_THRESHOLD_DEG, singular_floor=1e-6
    )
    for pid in deltas:
        for name in deltas[pid]:
            assert torch.equal(out_forward[pid][name], out_permuted[pid][name])


def test_backstop_is_bitwise_deterministic() -> None:
    e_ref = _e_ref()
    deltas = {"c0": _toy_grouped_delta(seed=13)}
    embeddings = {"c0": e_ref @ _rot(35.0)}
    a = procrustes_backstop(
        deltas, embeddings, e_ref, threshold_deg=_THRESHOLD_DEG, singular_floor=1e-6
    )
    b = procrustes_backstop(
        deltas, embeddings, e_ref, threshold_deg=_THRESHOLD_DEG, singular_floor=1e-6
    )
    for name in deltas["c0"]:
        assert torch.equal(a["c0"][name], b["c0"][name])


# --- the degenerate clamp-and-skip path: a k < d embedding survives unaligned with a WARN log ---


def test_degenerate_embedding_skips_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    e_ref = _e_ref()
    # A rank-deficient embedding (a zeroed column) makes M = T^T S degenerate even after the relaxed retry.
    degenerate = e_ref.clone()
    degenerate[:, _D - 1] = 0.0
    e_ref_deg = e_ref.clone()
    e_ref_deg[:, _D - 1] = 0.0
    deltas = {"c0": _toy_grouped_delta(seed=14)}
    embeddings = {"c0": degenerate}

    with caplog.at_level(logging.WARNING, logger="lensemble.gauge.backstop"):
        out = procrustes_backstop(
            deltas,
            embeddings,
            e_ref_deg,
            threshold_deg=_THRESHOLD_DEG,
            singular_floor=1e-6,
        )

    # The participant's UNALIGNED delta survives byte-identical (the backstop was skipped, not aborted).
    for name, tensor in deltas["c0"].items():
        assert torch.equal(out["c0"][name], tensor)
    # ...and a WARN line names gauge/procrustes_residual.
    assert any(
        "gauge/procrustes_residual" in rec.message and rec.levelno == logging.WARNING
        for rec in caplog.records
    )


def test_degenerate_retry_succeeds_when_relaxed_floor_admits() -> None:
    # An embedding whose smallest singular value of M = T^T S sits BETWEEN the relaxed floor and the strict
    # floor: the strict pass raises DegenerateProcrustes, the relaxed retry succeeds, the backstop proceeds.
    e_ref = _e_ref()
    # Shrink the SAME axis in the reference (and so in the rotated embeddings) by 1.5e-3, which puts M's
    # smallest singular value at ~1.7e-4 — below the 1e-3 strict floor we pass, above the relaxed
    # 1e-3 * 1e-3 = 1e-6 floor. f_c(P) = (shrunk E_ref) @ Q_c carries a 30-deg drift.
    e_ref_scaled = e_ref.clone()
    e_ref_scaled[:, _D - 1] *= 1.5e-3
    embeddings = {"c0": e_ref_scaled @ _rot(30.0)}
    deltas = {"c0": _toy_grouped_delta(seed=15)}

    out = procrustes_backstop(
        deltas,
        embeddings,
        e_ref_scaled,
        threshold_deg=_THRESHOLD_DEG,
        singular_floor=1e-3,  # strict floor the first pass trips; relaxed retry (1e-6) admits
    )
    # The retry produced a valid Q*; the predictor delta was conjugated (the round stayed alive).
    assert not torch.equal(
        out["c0"]["predictor.out_proj.weight"],
        deltas["c0"]["predictor.out_proj.weight"],
    )
