"""SIGReg sketch + Epps-Pulley statistic (RFC-0008 6 / 07 2.4, 2.7). Issue #12. CPU fp32."""

from __future__ import annotations

import pytest
import torch

from lensemble.data import guard_egress
from lensemble.errors import ResidencyViolation
from lensemble.model import build_sketch, sigreg_statistic


def test_sigreg_statistic_against_known_samples(tol: object) -> None:
    torch.manual_seed(0)
    sketch = build_sketch(seed=123, d=8, sketch_dim=64)
    normal = torch.randn(8192, 8)
    two_point = torch.sign(
        torch.randn(8192, 8)
    )  # +-1: unit variance, strongly non-normal
    null = sigreg_statistic(normal, sketch)
    signal = sigreg_statistic(two_point, sketch)
    assert float(null) < tol.SIGREG_NULL_TOL, float(null)  # type: ignore[attr-defined]
    assert float(signal) > tol.SIGREG_SIGNAL_FLOOR, float(signal)  # type: ignore[attr-defined]
    assert null.dtype == torch.float32 and null.ndim == 0


def test_statistic_is_reproducible() -> None:
    sketch = build_sketch(seed=1, d=8, sketch_dim=32)
    torch.manual_seed(7)
    z = torch.randn(1024, 8)
    assert torch.equal(sigreg_statistic(z, sketch), sigreg_statistic(z, sketch))


@pytest.mark.parametrize(
    "device", ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]
)
def test_statistic_follows_embedding_device(device: str) -> None:
    """``sigreg_statistic`` device-follows ``embeddings`` even when the sketch is CPU-built.

    ``build_sketch`` always returns a CPU tensor (deterministic seed), but in the GPU inner loop the
    embeddings live on CUDA. The statistic must move the sketch + Epps-Pulley grid onto the embedding
    device rather than raising a cross-device matmul error (the bug the CPU toy never exercised). Runs
    the true cross-device path only when CUDA is present; on CPU it still guards device-following and
    that the move is value-preserving.
    """
    sketch = build_sketch(seed=5, d=8, sketch_dim=32)
    assert sketch.device.type == "cpu"
    emb = torch.randn(256, 8, device=device)
    stat = sigreg_statistic(emb, sketch)
    assert stat.device.type == device
    assert stat.dtype == torch.float32 and stat.ndim == 0
    # the CPU<->device move is value-preserving: the statistic matches the all-CPU computation
    cpu_stat = sigreg_statistic(emb.cpu(), sketch)
    assert torch.allclose(stat.cpu(), cpu_stat, atol=1e-5)


def test_sketch_consistency_across_participants() -> None:
    # two participants given the same s_t derive the identical projection matrix A
    a_p1 = build_sketch(seed=42, d=16, sketch_dim=64)
    a_p2 = build_sketch(seed=42, d=16, sketch_dim=64)
    assert torch.equal(a_p1, a_p2)  # INV-SKETCH-CONSISTENCY
    assert tuple(a_p1.shape) == (16, 64)
    assert torch.allclose(
        a_p1.norm(dim=0), torch.ones(64), atol=1e-5
    )  # unit-norm directions
    # a different seed gives a different sketch
    assert not torch.equal(a_p1, build_sketch(seed=43, d=16, sketch_dim=64))


def test_projection_and_embedding_are_residency_bound() -> None:
    sketch = build_sketch(seed=1, d=8, sketch_dim=4)
    embeddings = torch.randn(32, 8)
    projection = embeddings @ sketch  # raw Az — must never leave the boundary
    raised = 0
    for payload in ({"sigreg_projection": projection}, {"embedding": embeddings}):
        with pytest.raises(ResidencyViolation):
            guard_egress(payload)
        raised += 1
    assert raised == 2  # the guard propagates, never caught-and-ignored
