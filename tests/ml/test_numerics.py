"""Numerical contract: bf16 forward / fp32 accumulation, device policy, determinism flag. Issue #14.

RFC-0008 7 / conventions 9. bf16 forward (default on CUDA; forced here on CPU to exercise the path)
matches an fp32-only reference within RTOL_BF16/ATOL_BF16; the loss/statistic accumulation stays fp32;
the deterministic flag with a fixed seed reproduces the per-term LossTerms. Runs on the CPU fallback.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from lensemble.model.encoder import Encoder, build_encoder
from lensemble.model.numerics import (
    ACCUMULATION_DTYPE,
    forward_dtype,
    resolve_device,
    set_determinism,
)
from lensemble.model.objective import LossTerms, Objective
from lensemble.model.predictor import Predictor, build_predictor

_D, _N, _COND, _STEPS = 8, 4, 8, 3


def _encoder_cfg() -> SimpleNamespace:
    # patching: (num_frames//tubelet) * (image_size//patch_size)^2 = 1 * 2^2 = 4 = N
    return SimpleNamespace(
        model=SimpleNamespace(
            d=_D,
            num_frames=2,
            image_size=4,
            patch_size=2,
            tubelet=2,
            depth=2,
            num_heads=2,
            in_channels=3,
        )
    )


def _predictor_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            d=_D,
            num_tokens=_N,
            cond_dim=_COND,
            predictor_depth=2,
            predictor_width=16,
            num_heads=4,
        )
    )


def _fixture() -> tuple[Encoder, Predictor, SimpleNamespace, torch.Tensor, Objective]:
    torch.manual_seed(0)
    encoder = build_encoder(_encoder_cfg()).eval()
    predictor = build_predictor(_predictor_cfg()).eval()
    window = SimpleNamespace(
        obs=torch.randn(_STEPS + 1, 2, 3, 4, 4)
    )  # (S+1, T, C, H, W)
    action_embedding = torch.randn(_STEPS, _COND)
    objective = Objective(
        lambda_pred=1.0, lambda_sig=0.5, lambda_anc=0.0, sketch_seed=11, sketch_dim=16
    )
    return encoder, predictor, window, action_embedding, objective


def _set_compute_dtype(
    encoder: Encoder, predictor: Predictor, dtype: torch.dtype
) -> None:
    encoder.compute_dtype = dtype
    predictor.compute_dtype = dtype


def test_device_and_dtype_policy() -> None:
    device = resolve_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cuda", "cpu"}
    # bf16 forward on the CUDA primary; fp32 on the CPU fallback; accumulation always fp32.
    assert forward_dtype(torch.device("cuda")) == torch.bfloat16
    assert forward_dtype(torch.device("cpu")) == torch.float32
    assert ACCUMULATION_DTYPE == torch.float32


def test_bf16_forward_matches_fp32_within_tolerance(tol: object) -> None:
    encoder, predictor, window, action_embedding, objective = _fixture()

    _set_compute_dtype(encoder, predictor, torch.float32)
    with torch.no_grad():
        ref = objective(encoder, predictor, window, action_embedding)

    _set_compute_dtype(
        encoder, predictor, torch.bfloat16
    )  # force the bf16 forward path on CPU
    with torch.no_grad():
        bf16 = objective(encoder, predictor, window, action_embedding)

    rtol: float = tol.RTOL_BF16  # type: ignore[attr-defined]
    atol: float = tol.ATOL_BF16  # type: ignore[attr-defined]
    assert math.isclose(float(ref.pred), float(bf16.pred), rel_tol=rtol, abs_tol=atol)
    assert math.isclose(
        float(ref.sigreg), float(bf16.sigreg), rel_tol=rtol, abs_tol=atol
    )
    # accumulation is fp32 regardless of forward dtype
    for terms in (ref, bf16):
        assert terms.pred.dtype == torch.float32
        assert terms.sigreg.dtype == torch.float32
        assert terms.total.dtype == torch.float32


def test_encoder_accepts_bfloat16_input_with_fp32_master_weights() -> None:
    encoder = build_encoder(_encoder_cfg()).eval()
    clip = torch.randn(_STEPS + 1, 2, 3, 4, 4).to(torch.bfloat16)

    with torch.no_grad():
        encoded = encoder(clip)

    assert tuple(encoded.tokens.shape) == (_STEPS + 1, _N, _D)
    assert next(encoder.parameters()).dtype == torch.float32
    assert encoded.tokens.dtype in {torch.float32, torch.bfloat16}


def test_deterministic_flag_reproduces_loss_terms(tol: object) -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        set_determinism(True)  # gate torch.use_deterministic_algorithms (warn_only)
        encoder, predictor, window, action_embedding, objective = _fixture()
        with torch.no_grad():
            first = objective(encoder, predictor, window, action_embedding)
            second = objective(encoder, predictor, window, action_embedding)
        rtol: float = tol.RTOL_LOSS  # type: ignore[attr-defined]
        atol: float = tol.ATOL_LOSS  # type: ignore[attr-defined]
        # given identical inputs/seed/sketch, the per-term scalars reproduce within fp32 tolerance
        for a, b in (
            (first.pred, second.pred),
            (first.sigreg, second.sigreg),
            (first.total, second.total),
        ):
            assert math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)
    finally:
        set_determinism(previous)  # restore global state for other tests


def test_loss_terms_type() -> None:
    encoder, predictor, window, action_embedding, objective = _fixture()
    terms = objective(encoder, predictor, window, action_embedding)
    assert isinstance(terms, LossTerms)
