"""Sketch-seed consistency across participants (RFC-0009 4, INV-SKETCH-CONSISTENCY). Issue #35 (T6)."""

from __future__ import annotations

import torch

from lensemble.config import round_sketch_seed


def _projection_from_seed(
    seed: int, *, d: int = 8, sketch_dim: int = 4
) -> torch.Tensor:
    """Stand-in for the SIGReg sketch matrix A built deterministically from s_t (real A is #12)."""
    gen = torch.Generator().manual_seed(seed % (2**63))
    return torch.randn(sketch_dim, d, generator=gen)


def test_sketch_seed_identical_across_participants() -> None:
    root, t = 1234, 5
    # every participant derives s_t from (root_seed, t) only
    s_a = round_sketch_seed(root, t)
    s_b = round_sketch_seed(root, t)
    assert s_a == s_b
    # ...so the projection matrix A each reconstructs is identical (INV-SKETCH-CONSISTENCY)
    a_a = _projection_from_seed(s_a)
    a_b = _projection_from_seed(s_b)
    assert torch.equal(a_a, a_b)


def test_sketch_seed_varies_by_round() -> None:
    root = 1234
    assert round_sketch_seed(root, 0) != round_sketch_seed(root, 1)
    a0 = _projection_from_seed(round_sketch_seed(root, 0))
    a1 = _projection_from_seed(round_sketch_seed(root, 1))
    assert not torch.equal(a0, a1)  # a fresh sketch each round
