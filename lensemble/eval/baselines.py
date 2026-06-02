"""lensemble.eval.baselines — the four bracketing baselines + the gap-recovery reducer (RFC-0005 §5).

The headline gap-recovery fraction ``rho`` is only well-defined against four bracketing baselines that
share the same warm-start, public probe, environment suite, seeds, and reporting discipline:

- ``centralized`` — centralized-pooled, the **upper bound** ``S_centralized`` (``train_local`` on pooled
  data, no outer loop, no boundaries);
- ``local-only`` — the **lower bound** ``S_local_only`` (one ``train_local`` per silo);
- ``naive-fedavg`` — the **negative control** (the DiLoCo outer loop with ``lambda_anc = 0`` and no
  Procrustes backstop, exhibiting the collapse the gauge design fixes);
- ``fork-a`` — the **reference / safe degrade** (the encoder frozen at the warm-start, federate ``g_phi``
  only).

The Hydra config groups live under ``configs/baselines/``; this module loads them and reduces the four
``EvalReport.success_rate`` values to ``rho``. The anchored-federation run (the numerator's anchored
configuration) is the recommended rung of the ablation ladder, and running the federated baselines is the
Coordinator's job — this module configures and reduces, it does not run.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from lensemble.config import load_config
from lensemble.errors import EvaluationError, LensembleErrorCode

if TYPE_CHECKING:
    from lensemble.config import LensembleConfig

# The four bracketing baselines (RFC-0005 §5), each a named Hydra config group under configs/baselines/.
BASELINES: tuple[str, ...] = ("centralized", "local-only", "naive-fedavg", "fork-a")

_BASELINE_DIR = Path(__file__).resolve().parents[2] / "configs" / "baselines"


def load_baseline(name: str, *, config_dir: Path | None = None) -> "LensembleConfig":
    """Compose the :class:`~lensemble.config.LensembleConfig` for a named baseline (RFC-0005 §5).

    The baseline config overrides only its distinguishing fields over the shared structured defaults
    (warm-start, probe, seeds, env suite), so the four baselines differ in exactly the gauge knobs that
    define the bracket. Raises :class:`~lensemble.errors.EvaluationError` on an unknown baseline name.
    """
    if name not in BASELINES:
        raise EvaluationError(
            f"unknown baseline {name!r}; expected one of {BASELINES}",
            code=LensembleErrorCode.EVALUATION_FAILED,
            remediation=f"pass a baseline from {BASELINES}",
        )
    return load_config(config_name=name, config_dir=config_dir or _BASELINE_DIR)


def gap_recovery_fraction(
    *,
    success_anchored: float,
    success_local_only: float,
    success_centralized: float,
) -> float:
    """The gap-recovery fraction ``rho`` in ``[0, 1]`` (RFC-0005 §5).

    ``rho = (S_anchored_fed − S_local_only) / (S_centralized_pooled − S_local_only)``: the fraction of the
    centralized-over-local gap that the anchored federation recovers. Clamped to ``[0, 1]``. Raises
    :class:`~lensemble.errors.EvaluationError` if the bracket is degenerate (the centralized upper bound is
    not above the local-only lower bound) or any input is outside ``[0, 1]``.
    """
    for label, value in (
        ("success_anchored", success_anchored),
        ("success_local_only", success_local_only),
        ("success_centralized", success_centralized),
    ):
        if not 0.0 <= value <= 1.0:
            raise EvaluationError(
                f"{label} must be a success rate in [0, 1], got {value}",
                code=LensembleErrorCode.EVALUATION_FAILED,
                remediation="pass EvalReport.success_rate values (fractions in [0, 1])",
            )
    gap = success_centralized - success_local_only
    if gap <= 0.0:
        raise EvaluationError(
            f"degenerate bracket: centralized ({success_centralized}) is not above local-only "
            f"({success_local_only}); rho is undefined",
            code=LensembleErrorCode.EVALUATION_FAILED,
            remediation="rho needs S_centralized > S_local_only; check the bracketing baselines",
        )
    rho = (success_anchored - success_local_only) / gap
    return max(0.0, min(1.0, rho))
