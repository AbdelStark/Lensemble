"""lensemble.contracts.latent — the WMCP ``LatentState`` contract (docs/rfcs/RFC-0007 2).

``LatentState`` is the canonical per-clip latent object: the tensor plus the metadata that makes
conformance checkable without inspecting the producing model. The runtime carries no ``torch`` import
(the tensor type is an annotation only), so importing the type stays light; ``check_latent_state``
(``conformance.py``) does the tensor-level validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only; no runtime torch import
    import torch

# The pinned contract version. A federation agrees on exactly one (INV-WMCP, RFC-0007 6).
WMCP_VERSION: str = "wmcp-1.0.0"


@dataclass(frozen=True, slots=True)
class LatentState:
    """A per-clip set of ``N`` latent tokens of dimension ``d`` (conventions 2, 8).

    ``tokens`` is rank-2 ``(N, d)`` for a single clip or rank-3 ``(B, N, d)`` batched. ``num_tokens``
    is ``N`` and ``dim`` is ``d``, both fixed for a given ``wmcp_version`` across the federation.
    The tokens live in the shared, gauge-controlled latent frame; the contract pins the *declared*
    frame via ``wmcp_version`` and defers measurement of actual drift to the gauge diagnostic.
    """

    tokens: "torch.Tensor"  # shape (N, d) for one clip, or (B, N, d) batched
    num_tokens: int  # N: latent tokens emitted per clip by the encoder
    dim: int  # d: latent embedding dimension
    wmcp_version: str  # MUST equal WMCP_VERSION at the gate (INV-WMCP)

    @property
    def is_batched(self) -> bool:
        """``True`` when ``tokens`` is rank-3 ``(B, N, d)``."""
        return self.tokens.ndim == 3
