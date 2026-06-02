"""lensemble.contracts.action_head — the per-embodiment action-head interface (RFC-0007 5).

The ABC pins only the *output* of a head: :meth:`ActionHead.encode` returns a conditioning embedding
whose last dimension is ``cond_dim`` — the federation-fixed conditioning width that the shared
predictor ``g_phi`` consumes. The *input* is free to match the embodiment: ``spec.dim`` differs per
embodiment, ``cond_dim`` does not. This ``cond_dim`` seam is what lets a quadruped (one ``spec.dim``)
and a 7-DoF arm (another) condition the *same* ``g_phi``.

``INV-ACTIONHEAD-LOCAL`` (conventions 7): per-embodiment action heads ``h_psi^(c)`` are never
broadcast, aggregated, or written to a shared artifact. The local-checkpoint accessor is deliberately
named :meth:`ActionHead.state_dict_local` (not ``state_dict``) so no shared serializer picks it up by
convention; the federation/artifact boundary uses that naming seam to exclude head parameters. That
enforcement lives at the boundary (RFC-0007 5), not in this ABC — this issue provides only the seam.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from lensemble.contracts.action import ActionSpec
from lensemble.contracts.conformance import validate_action_spec

if TYPE_CHECKING:
    import torch


class ActionHead(ABC):
    """Abstract per-embodiment conditioning map ``h_psi^(c)`` (RFC-0007 5).

    Attributes:
        spec: the validated :class:`~lensemble.contracts.action.ActionSpec` for this embodiment.
        cond_dim: the federation-fixed conditioning width (from model config); the last dim of
            :meth:`encode`'s output. Identical across embodiments, unlike ``spec.dim``.

    A concrete head (out of scope here; RFC-0008 / model issues) combines this interface with an
    ``nn.Module``. This ABC is pure so the contract layer stays free of model/runtime coupling.
    """

    spec: ActionSpec
    cond_dim: int

    @abstractmethod
    def __init__(self, spec: ActionSpec, *, cond_dim: int) -> None:
        """Validate ``spec`` (``INV-WMCP``) and record the conditioning width.

        Subclasses MUST call ``super().__init__(spec, cond_dim=cond_dim)`` *before* allocating any
        parameters: building a head from an unvalidated spec is the ``INV-WMCP`` violation this
        forecloses, raising :class:`~lensemble.errors.ContractViolation` (``WMCP_CONTRACT_VIOLATION``).
        ``cond_dim`` is federation-fixed (model config); ``spec.dim`` is per-embodiment.
        """
        validate_action_spec(spec)
        self.spec = spec
        self.cond_dim = cond_dim

    @abstractmethod
    def encode(self, action: torch.Tensor) -> torch.Tensor:
        """Map a raw action batch to a conditioning embedding of shape ``(B, cond_dim)``.

        ``action`` is ``(B, spec.dim)`` for a continuous space, or ``(B,)`` / ``(B, spec.dim)`` discrete
        category indices per dim. The output's last dimension is ``cond_dim`` — the shared
        latent-conditioning space the predictor consumes — and its dtype follows the compute dtype. The
        conditioning embedding is a model-internal quantity and crosses no trust boundary.
        """
        ...

    @abstractmethod
    def state_dict_local(self) -> dict[str, torch.Tensor]:
        """Return the head's parameters for LOCAL checkpointing only (``INV-ACTIONHEAD-LOCAL``).

        Deliberately not named ``state_dict``: head parameters are per-embodiment and are never
        broadcast, aggregated, or written to a shared artifact. The name is the seam the
        federation/artifact boundary keys on to exclude them.
        """
        ...
