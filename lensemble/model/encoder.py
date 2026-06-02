"""lensemble.model.encoder — see docs/rfcs/RFC-0008. Stub scaffolded by #2."""
from __future__ import annotations
from typing import Any


class Encoder:
    """Video-ViT encoder f_theta emitting a WMCP `LatentState` (RFC-0008). Implemented by #10."""


def build_encoder(cfg: Any) -> "Encoder":
    """Construct the warm-started encoder (RFC-0008). Implemented by #10 (model-encoder-warmstart)."""
    raise NotImplementedError("lensemble.model.build_encoder is implemented by #10")
