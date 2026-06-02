"""Lensemble: federated, end-to-end JEPA world models.

Entry point for the public Python surface. See ``SPEC.md`` and
``docs/rfcs/RFC-0001-architecture.md`` for the architecture, and
``docs/spec/02-public-api.md`` / ``docs/spec/conventions.md`` (5) for the frozen
public surface re-exported here.

The public names are re-exported **lazily** (PEP 562 ``__getattr__``) so that
``import lensemble`` does not pull heavy optional dependencies (``torch``, ``lance``)
at import time; the owning submodule is imported only when the name is first accessed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

# Public name -> owning submodule (conventions 5 / 02-public-api 1). Frozen at 1.0.
_EXPORTS: dict[str, str] = {
    "LensembleConfig": "lensemble.config",
    "RunManifest": "lensemble.config",
    "load": "lensemble.config",
    "train_local": "lensemble.federation",
    "Coordinator": "lensemble.federation",
    "Participant": "lensemble.federation",
    "RoundState": "lensemble.federation",
    "build_encoder": "lensemble.model",
    "build_predictor": "lensemble.model",
    "build_action_head": "lensemble.model",
    "Objective": "lensemble.model",
    "evaluate": "lensemble.eval",
    "Planner": "lensemble.eval",
    "frame_drift": "lensemble.gauge",
    "procrustes_align": "lensemble.gauge",
    "commit_dataset": "lensemble.provenance",
    "DatasetCommitment": "lensemble.provenance",
    "ContributionLedger": "lensemble.provenance",
    "recompute_alignment": "lensemble.verify",
}

# Literal mirror of `_EXPORTS` (kept in sync by tests/test_public_surface.py).
__all__ = [
    "__version__",
    "LensembleConfig",
    "RunManifest",
    "load",
    "train_local",
    "Coordinator",
    "Participant",
    "RoundState",
    "build_encoder",
    "build_predictor",
    "build_action_head",
    "Objective",
    "evaluate",
    "Planner",
    "frame_drift",
    "procrustes_align",
    "commit_dataset",
    "DatasetCommitment",
    "ContributionLedger",
    "recompute_alignment",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve a public name from its owning submodule (PEP 562)."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'lensemble' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)


if (
    TYPE_CHECKING
):  # static-analysis view of the lazily re-exported surface; no runtime cost
    from lensemble.config import LensembleConfig, RunManifest, load  # noqa: F401
    from lensemble.eval import Planner, evaluate  # noqa: F401
    from lensemble.federation import (  # noqa: F401
        Coordinator,
        Participant,
        RoundState,
        train_local,
    )
    from lensemble.gauge import frame_drift, procrustes_align  # noqa: F401
    from lensemble.model import (
        Objective,
        build_action_head,
        build_encoder,
        build_predictor,
    )  # noqa: F401
    from lensemble.provenance import (
        ContributionLedger,
        DatasetCommitment,
        commit_dataset,
    )  # noqa: F401
    from lensemble.verify import recompute_alignment  # noqa: F401
