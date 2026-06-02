"""lensemble.config.schema — see docs/rfcs/RFC-0009. Stub scaffolded by #2."""
from __future__ import annotations
from typing import Any


class LensembleConfig:
    """Root structured-configuration tree (RFC-0009 2). Implemented by config-schema (#34)."""


def load(path: Any, overrides: list[str] | None = None) -> "LensembleConfig":
    """Load and validate a `LensembleConfig` (RFC-0009 3). Implemented by config-schema (#34)."""
    raise NotImplementedError("lensemble.config.load is implemented by #34 (config-schema)")
