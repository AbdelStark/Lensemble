"""lensemble.federation.pseudogradient — the one private object that crosses the boundary (RFC-0003 3).

``PseudoGradient`` is the DiLoCo outer-loop delta ``Δ_c = (θ_c, φ_c) − (θ_t, φ_t)`` — the H-step local
update treated as one gradient — after DP clip+noise, bound to the dataset Merkle root it was computed
under. It is the *only* participant-derived object permitted across a trust boundary, and it crosses only
under secure aggregation + DP.

Invariants enforced here: ``INV-ACTIONHEAD-LOCAL`` — ``delta`` is materialized only over the federated
param groups (encoder ``θ`` + predictor ``φ``); an action-head group reaching it raises
``ResidencyViolation`` (fail-closed, never swallowed). ``INV-COMMIT-BINDING`` — exactly one 32-byte
``dataset_root``. ``INV-RESIDENCY`` — ``delta`` is the only tensor field, and the type carries the
``PseudoGradient`` egress role so the egress guard permits it (and only its ``delta``) to cross. The
DP bound ``‖Δ_c‖ ≤ C_clip`` itself (``INV-DP-BOUND``) is enforced in ``lensemble.privacy.dp``, not here;
``l2_norm`` is the fp32 norm measured AFTER clipping and BEFORE noising.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lensemble.data.residency import EgressRole
from lensemble.errors import LensembleErrorCode, ResidencyViolation

if TYPE_CHECKING:
    from collections.abc import Mapping

_FEDERATED_GROUPS = (
    "encoder",
    "predictor",
)  # the only groups that cross (θ, φ), in this order
_ACTION_HEAD_GROUP = "action_head"
_ROOT_BYTES = 32  # SHA-256 digest length (INV-COMMIT-BINDING)


@dataclass(frozen=True)
class PseudoGradient:
    """The released DiLoCo delta (RFC-0003 3 / 03 6). Frozen; validated at construction.

    Fields: ``delta`` (flat fp32, concat of encoder ``θ`` then predictor ``φ`` param-group deltas in a
    fixed order), ``l2_norm`` (``‖delta‖`` in fp32, post-clip / pre-noise — recorded for the DP-bound
    check), ``dataset_root`` (the 32-byte ``R_c`` this delta binds to, ``INV-COMMIT-BINDING``),
    ``round_index`` (target round ``t``), ``clipped`` (clip projection applied), ``quantized`` (int8 wire
    quantization applied — orthogonal to the gauge).
    """

    # Egress role so the residency guard permits this carrier and ONLY its `delta` tensor to cross.
    __egress_role__ = EgressRole.PSEUDO_GRADIENT

    delta: Tensor
    l2_norm: float
    dataset_root: bytes
    round_index: int
    clipped: bool = False
    quantized: bool = False

    def __post_init__(self) -> None:
        if self.delta.dtype != torch.float32:
            raise ValueError(f"delta must be fp32, got {self.delta.dtype}")
        if not bool(torch.isfinite(self.delta).all()):
            raise ValueError("delta contains non-finite values (NaN/Inf)")
        if len(self.dataset_root) != _ROOT_BYTES:
            raise ValueError(
                f"dataset_root must be {_ROOT_BYTES} bytes (SHA-256), got {len(self.dataset_root)}"
            )
        if self.round_index < 0:
            raise ValueError(f"round_index must be >= 0, got {self.round_index}")
        recomputed = float(self.delta.norm())
        if abs(self.l2_norm - recomputed) > 1e-6 + 1e-5 * recomputed:
            raise ValueError(
                f"l2_norm {self.l2_norm} does not match ||delta|| {recomputed} (post-clip norm)"
            )


def build_pseudogradient(
    param_deltas: Mapping[str, Tensor],
    *,
    dataset_root: bytes,
    round_index: int,
    clipped: bool = False,
    quantized: bool = False,
) -> PseudoGradient:
    """Flatten the federated param-group deltas into a :class:`PseudoGradient` (RFC-0003 3).

    ``param_deltas`` maps a param-group name (``"encoder.*"`` / ``"predictor.*"``) to its delta tensor.
    The flat ``delta`` concatenates the encoder groups then the predictor groups, each sorted by name, in
    a fixed deterministic order. An ``action_head.*`` group (or any non-federated group) raises
    :class:`~lensemble.errors.ResidencyViolation` (``INV-ACTIONHEAD-LOCAL``): per-embodiment heads are
    local and never cross. ``l2_norm`` is the fp32 norm of the assembled ``delta``.
    """
    for name in param_deltas:
        group = name.split(".", 1)[0]
        if group == _ACTION_HEAD_GROUP:
            err = ResidencyViolation(
                f"action-head param group {name!r} may not enter a PseudoGradient (INV-ACTIONHEAD-LOCAL)",
                code=LensembleErrorCode.RESIDENCY_VIOLATION,
                remediation="keep per-embodiment action heads local; checkpoint them with state_dict_local",
            )
            err.tensor_role = "action_head"  # type: ignore[attr-defined]
            raise err
        if group not in _FEDERATED_GROUPS:
            raise ResidencyViolation(
                f"non-federated param group {name!r} may not cross; only encoder/predictor do",
                code=LensembleErrorCode.RESIDENCY_VIOLATION,
                remediation="restrict the released delta to the encoder and predictor param groups",
            )

    ordered = sorted(
        param_deltas, key=lambda k: (_FEDERATED_GROUPS.index(k.split(".", 1)[0]), k)
    )
    if ordered:
        delta = torch.cat(
            [param_deltas[k].reshape(-1).to(torch.float32) for k in ordered]
        )
    else:
        delta = torch.zeros(0, dtype=torch.float32)
    return PseudoGradient(
        delta=delta,
        l2_norm=float(delta.norm()),
        dataset_root=dataset_root,
        round_index=round_index,
        clipped=clipped,
        quantized=quantized,
    )
