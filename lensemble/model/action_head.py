"""lensemble.model.action_head — the per-embodiment action head ``h_psi^(c)`` (RFC-0008 4 / RFC-0007 5).

The local map from a raw action batch to the predictor's conditioning embedding ``(B, cond_dim)``: a
continuous head is a small MLP, a discrete head is a sum of per-dimension category embeddings. ``cond_dim``
is the federation-fixed conditioning width the shared predictor ``g_phi`` consumes; ``spec.dim`` is
per-embodiment. The ``cond_dim`` seam is what lets a quadruped (one ``spec.dim``) and a 7-DoF arm (another)
condition the *same* ``g_phi`` (RFC-0007 5). This module is the concrete ``nn.Module`` realization the eval
harness (#52) needs; it fills the orphaned substrate of the closed issue #8.

``INV-ACTIONHEAD-LOCAL`` (conventions 7; RFC-0008 4). The action head is per-embodiment LOCAL state: its
parameters are never broadcast, aggregated, or serialized into a shared artifact. The artifact boundary
(``lensemble.artifacts.checkpoint``) fail-closes on any non-``{encoder, predictor}`` param group; this
head is constructed fresh and lives only in private local state. A deployment loads the participant's
trained local head from its own local checkpoint — that local-load path is out of scope here (the harness
evaluates the shared ``encoder``/``predictor`` with a fresh head).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
from lensemble.contracts.conformance import validate_action_spec
from lensemble.errors import ConfigError, EvaluationError, LensembleErrorCode


class ActionHead(nn.Module):
    """Per-embodiment conditioning map ``h_psi^(c)``: raw actions ``-> (B, cond_dim)`` (RFC-0008 4).

    Continuous (``ActionKind.CONTINUOUS``): an MLP ``Linear(dim, cond_dim) -> GELU -> Linear(cond_dim,
    cond_dim)`` over float actions ``(B, dim)``. Discrete (``ActionKind.DISCRETE``): one
    ``nn.Embedding(num_classes[i], cond_dim)`` per action dimension, summed over integer indices
    ``(B, dim)``. The output last dim is always ``cond_dim`` — the shared latent-conditioning space.

    ``INV-ACTIONHEAD-LOCAL``: this module is LOCAL per-embodiment state, never serialized into a shared
    artifact (the artifact boundary fail-closes on a non-encoder/predictor param group).
    """

    spec: ActionSpec
    cond_dim: int

    def __init__(self, spec: ActionSpec, *, cond_dim: int) -> None:
        super().__init__()
        validate_action_spec(
            spec
        )  # INV-WMCP: never build a head from an unvalidated spec
        self.spec = spec
        self.cond_dim = cond_dim
        if spec.kind is ActionKind.CONTINUOUS:
            self.mlp = nn.Sequential(
                nn.Linear(spec.dim, cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, cond_dim),
            )
        else:  # ActionKind.DISCRETE — validated by validate_action_spec (num_classes present, >= 2)
            assert spec.num_classes is not None  # for the type checker; enforced above
            self.embeddings = nn.ModuleList(
                nn.Embedding(int(n), cond_dim) for n in spec.num_classes
            )

    def encode(self, actions: Tensor) -> Tensor:
        """Map a raw action batch ``(B, spec.dim)`` to a conditioning embedding ``(B, cond_dim)``.

        Validates the action rank/last-dim against ``spec.dim`` (a mismatch is an
        :class:`~lensemble.errors.EvaluationError`, never a silent reshape). For a discrete head the
        actions are integer category indices; for a continuous head they are floats. The conditioning
        embedding is a model-internal quantity and crosses no trust boundary.
        """
        if actions.ndim != 2 or actions.shape[1] != self.spec.dim:
            raise EvaluationError(
                f"action head expects actions of shape (B, {self.spec.dim}), got "
                f"{tuple(actions.shape)}",
                code=LensembleErrorCode.EVALUATION_FAILED,
                remediation="pass a batched action tensor whose last dim equals spec.dim",
            )
        if self.spec.kind is ActionKind.CONTINUOUS:
            return self.mlp(actions.to(torch.float32))
        indices = actions.to(torch.int64)
        out = self.embeddings[0](indices[:, 0])
        for i in range(1, self.spec.dim):
            out = out + self.embeddings[i](indices[:, i])
        return out

    def forward(self, actions: Tensor) -> Tensor:
        """Alias for :meth:`encode` so the head composes as a plain ``nn.Module`` callable."""
        return self.encode(actions)

    def state_dict_local(self) -> dict[str, Tensor]:
        """The head's parameters for LOCAL checkpointing only (``INV-ACTIONHEAD-LOCAL``).

        Deliberately not named ``state_dict``: per-embodiment head parameters are never broadcast,
        aggregated, or written to a shared artifact. The name is the seam the federation/artifact
        boundary keys on to exclude them.
        """
        return dict(self.state_dict())


def build_action_head(cfg: Any, spec: ActionSpec) -> ActionHead:
    """Construct a per-embodiment :class:`ActionHead` from a validated ``ActionSpec`` (RFC-0007/0008; #8).

    Reads ``cond_dim = getattr(cfg.model, "cond_dim", cfg.model.d)`` (the federation-fixed conditioning
    width). Validates ``spec`` (``dim > 0``; for a discrete space ``num_classes`` present with
    ``len == dim`` and each ``>= 2``) and enforces ``spec.wmcp_version == WMCP_VERSION`` (``INV-WMCP``).
    Raises :class:`~lensemble.errors.ConfigError` on an invalid config/spec.

    ``INV-ACTIONHEAD-LOCAL``: the returned head is per-embodiment LOCAL state, never serialized into a
    shared artifact (the artifact boundary fail-closes on a non-encoder/predictor param group).
    """
    model = getattr(cfg, "model", None)
    if model is None:
        raise ConfigError(
            "config has no `model` sub-config",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="provide cfg.model carrying d (and optionally cond_dim)",
        )
    cond_dim = int(getattr(model, "cond_dim", getattr(model, "d")))
    if cond_dim <= 0:
        raise ConfigError(
            f"action-head cond_dim must be > 0, got {cond_dim}",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="set a positive cond_dim (or model.d) for the conditioning width",
        )
    if spec.wmcp_version != WMCP_VERSION:
        raise ConfigError(
            f"ActionSpec wmcp_version {spec.wmcp_version!r} != pinned {WMCP_VERSION!r}",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="declare the embodiment's ActionSpec at the pinned WMCP version (INV-WMCP)",
        )
    if spec.dim <= 0:
        raise ConfigError(
            f"ActionSpec dim must be > 0, got {spec.dim}",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="declare a positive action dimensionality",
        )
    if spec.kind is ActionKind.DISCRETE:
        if spec.num_classes is None or len(spec.num_classes) != spec.dim:
            raise ConfigError(
                "discrete ActionSpec needs num_classes with len == dim",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="declare per-dim num_classes (len == dim) for a discrete space",
            )
        if any(int(n) < 2 for n in spec.num_classes):
            raise ConfigError(
                f"discrete ActionSpec num_classes must each be >= 2, got {spec.num_classes}",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation="a discrete action dimension needs at least two categories",
            )
    return ActionHead(spec, cond_dim=cond_dim)
