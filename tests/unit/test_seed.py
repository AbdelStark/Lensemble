"""Deterministic seeding scheme (RFC-0009 4). Issue #35 (T5)."""

from __future__ import annotations

import dataclasses

from lensemble.config import (
    SEED_DERIVATION,
    derive,
    load_config,
    round_sketch_seed,
    seed_everything,
)

_MAX = 1 << 63


def test_seed_derivation_id() -> None:
    assert SEED_DERIVATION == "blake3-v1"


def test_derive_is_pure_and_in_range() -> None:
    assert derive(42, "torch") == derive(42, "torch")  # pure
    assert 0 <= derive(42, "torch") < _MAX  # 63-bit non-negative


def test_derive_golden_values() -> None:
    # bit-stable golden vectors (BLAKE3, cross-platform)
    assert derive(42, "torch") == 502105390651072317
    assert derive(0, "python") == 8444696587771374422
    assert round_sketch_seed(7, 3) == 4153186984671930622


def test_derive_distinct_by_label_and_root() -> None:
    assert derive(0, "torch") != derive(0, "numpy")
    assert derive(0, "torch") != derive(1, "torch")


def test_seed_everything_returns_fixed_component_map() -> None:
    cfg = load_config()  # root_seed defaults to 0
    seeds = seed_everything(cfg)
    assert seeds == {
        "python": 8444696587771374422,
        "numpy": 599031662937410987,
        "torch": 7909314452933869089,
        "cuda": 1408643187157566339,
    }


def test_seed_everything_tracks_root_seed() -> None:
    base = load_config()
    cfg = dataclasses.replace(
        base, determinism=dataclasses.replace(base.determinism, root_seed=7)
    )
    seeds = seed_everything(cfg)
    assert seeds["torch"] == derive(7, "torch")
