"""lensemble.config.seed — the deterministic seeding scheme (docs/rfcs/RFC-0009 4).

One root seed derives every component seed and every per-round sketch seed ``s_t``. The derivation is a
pure function of ``(root_seed, label)`` — stable across processes, hosts, and OS, with no dependence on
wall-clock, PID, or RNG state — realizing ``INV-SKETCH-CONSISTENCY``: every participant in round ``t``
reconstructs the identical SIGReg projection matrix ``A`` from ``s_t``. The algorithm id is recorded so a
future change is a versioned migration, not a silent reinterpretation.

``derive`` and ``round_sketch_seed`` are torch-free (BLAKE3 only); ``seed_everything`` imports torch/
numpy lazily so importing this module stays light.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import blake3

if TYPE_CHECKING:
    from lensemble.config.schema import LensembleConfig

SEED_DERIVATION = "blake3-v1"  # recorded in RunManifest.env["seed_derivation"]

_MASK63 = (1 << 63) - 1
_COMPONENTS = ("python", "numpy", "torch", "cuda")


def derive(root_seed: int, label: str) -> int:
    """Deterministic, cross-platform child seed in ``[0, 2**63)``. Pure: no time, PID, or RNG state.

    Collision-resistant and order-independent by ``label``: ``derive(s, "torch")`` is stable regardless
    of when it is called. (BLAKE3 over a domain-separated ``f"{root_seed}:{label}"`` encoding.)
    """
    digest = blake3.blake3(f"{root_seed}:{label}".encode("utf-8")).digest(8)
    return int.from_bytes(digest, "big") & _MASK63


def round_sketch_seed(root_seed: int, round_index: int) -> int:
    """``s_t = derive(root_seed, "sketch:{t}")`` (``INV-SKETCH-CONSISTENCY``, RFC-0002 3).

    A function of ``(root_seed, t)`` only: the coordinator broadcasts ``s_t`` and every participant
    reconstructs the identical projection matrix ``A``.
    """
    return derive(root_seed, f"sketch:{round_index}")


def seed_everything(cfg: "LensembleConfig") -> dict[str, int]:
    """Seed python/numpy/torch/cuda from ``cfg.determinism.root_seed`` and apply determinism flags.

    Returns the component-seed map (the canonical derived seeds, recorded in the ``RunManifest``). The
    CUBLAS workspace env is the caller's responsibility, not applied here.
    """
    import random

    import numpy as np
    import torch

    root = cfg.determinism.root_seed
    seeds = {lib: derive(root, lib) for lib in _COMPONENTS}
    random.seed(seeds["python"])
    np.random.seed(seeds["numpy"] % (2**32))  # numpy legacy seed is uint32
    torch.manual_seed(seeds["torch"])
    if torch.cuda.is_available():  # pragma: no cover - no CUDA in CI
        torch.cuda.manual_seed_all(seeds["cuda"])
    if cfg.determinism.deterministic_inner:
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seeds
