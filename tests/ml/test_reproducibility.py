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


# --- Reproducibility guarantee + override precedence (RFC-0009 6/7). Issue #37 (G4/T7) ---

from datetime import datetime, timezone  # noqa: E402

from lensemble.config import build_manifest, config_hash, load  # noqa: E402

_FIXED = datetime(2026, 6, 2, 12, 0, 0, tzinfo=timezone.utc)


def _tiny_train_local(*overrides: str):
    """A tiny single-site train_local config (few rounds) for the CPU runtime budget (RFC-0009 6)."""
    return load(overrides=["federation.num_rounds=3", *overrides])


def _flatten(tree: dict, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def test_same_seed_same_manifest_hash() -> None:
    """G4: same config + same root_seed -> identical config_hash / component_seeds /
    round_sketch_seeds (RFC-0009 6/7; the seed lineage is deterministic, INV-SKETCH-CONSISTENCY)."""
    first = build_manifest(
        _tiny_train_local(), run_mode="train_local", created_at=_FIXED
    )
    second = build_manifest(
        _tiny_train_local(), run_mode="train_local", created_at=_FIXED
    )
    assert first.config_hash == second.config_hash
    assert first.component_seeds == second.component_seeds
    assert (
        first.round_sketch_seeds == second.round_sketch_seeds
    )  # exact, not approximate


def test_override_changes_exactly_the_targeted_field_and_hash() -> None:
    """T7: a key=value override changes exactly the targeted field, moves config_hash, and the
    manifest records the resolved post-override config (RFC-0009 7 precedence)."""
    from dataclasses import asdict

    base = _tiny_train_local()
    overridden = _tiny_train_local("objective.lambda_anc=0.5")

    base_flat = _flatten(asdict(base))
    over_flat = _flatten(asdict(overridden))
    changed = {k for k in base_flat if base_flat[k] != over_flat[k]}
    assert changed == {"objective.lambda_anc"}  # exactly the targeted leaf
    assert overridden.objective.lambda_anc == 0.5

    assert config_hash(asdict(base)) != config_hash(asdict(overridden))  # hash moves
    manifest = build_manifest(overridden, run_mode="train_local", created_at=_FIXED)
    assert (
        manifest.config_resolved["objective"]["lambda_anc"] == 0.5
    )  # resolved is recorded
