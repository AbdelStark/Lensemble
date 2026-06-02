"""lensemble.model.objective — see docs/rfcs/RFC-0008 / RFC-0002. Stub scaffolded by #2."""

from __future__ import annotations

from typing import Any


class Objective:
    """Composite loss lambda_pred*L_pred + lambda_sig*SIGReg + lambda_anc*L_anchor (RFC-0008). #13."""

    def __init__(self, cfg: Any) -> None:
        raise NotImplementedError("lensemble.model.Objective is implemented by #13")
