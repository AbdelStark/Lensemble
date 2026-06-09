"""lensemble.eval.metrics — the evaluation metric bodies (RFC-0005 §3-4).

Pure metric functions the eval harness (#52) composes into an ``EvalReport``: planning success and cost
(the headline downstream metric, §3) and the supporting metrics (§4) — effective dimension (the collapse
guard), a representation-probe accuracy, and the communication-byte accountant. Each carries a documented
unit and an in-range contract; an out-of-range value is an :class:`~lensemble.errors.EvaluationError`,
never a silently-logged number.

Effective dimension shares the gauge-diagnostic eigendecomposition discipline (fp32 accumulation,
non-negative conditioning, conventions §9): the participation ratio ``(sum_i sigma_i)^2 / sum_i sigma_i^2``
over the eigenspectrum of ``Cov(f_theta(x))`` is ``1`` for a collapsed (rank-1) representation and ``d``
for an isotropic one, so a collapse re-introduced by averaging across mutually-rotated frames shows up as
``effective_dim -> 1`` (RFC-0002 §2.1). It is reported alongside probe accuracy and MPC success so three
independent signals must agree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lensemble.errors import EvaluationError, LensembleErrorCode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lensemble.eval.mpc import PlanResult

_BYTES_FP32 = 4  # wire dtype: full-precision pseudo-gradient (08 §4)
_BYTES_INT8 = 1  # int8 wire quantization (08 §4)
_INT8_SCALE_BYTES = 4  # the fp32 per-tensor scale shipped alongside the int8 codes


def covariance_eigenvalues(centered: "Tensor") -> "Tensor":
    """Eigenvalues of the sample covariance of a CENTERED ``(n, d)`` matrix, computed STABLY (#264).

    Computing ``eigvalsh(X^T X)`` squares the condition number, which makes the accelerated symmetric-eigen
    solver DIVERGE on a collapsed / ill-conditioned representation (torch raises ``_LinAlgError`` error code
    26 on CUDA) — exactly the regime the collapse metrics must measure. The SVD of the centered matrix is
    backward-stable and yields the same eigenvalues ``sigma_i^2 / (n-1)`` without forming ``X^T X``; a CPU
    fallback covers the rare accelerated-SVD non-convergence. Returns the covariance eigenvalues.
    """
    n = max(1, centered.shape[0] - 1)
    try:
        sv = torch.linalg.svdvals(centered)
    except Exception:  # noqa: BLE001 — accelerated SVD non-convergence → robust CPU solver
        sv = torch.linalg.svdvals(centered.detach().cpu())
    return (sv.to(torch.float32) ** 2) / n


def _fail(message: str, remediation: str) -> EvaluationError:
    return EvaluationError(
        message, code=LensembleErrorCode.EVALUATION_FAILED, remediation=remediation
    )


# --- the downstream metric: planning success and cost (RFC-0005 §3) ---


def success_rate(successes: "Sequence[bool]") -> float:
    """Fraction of held-out episodes that reached the goal (RFC-0005 §3). Unit: fraction in ``[0, 1]``.

    Raises :class:`~lensemble.errors.EvaluationError` on an empty episode set (an undefined rate).
    """
    n = len(successes)
    if n == 0:
        raise _fail(
            "success_rate over zero episodes is undefined",
            "evaluate at least one held-out episode",
        )
    return sum(1 for s in successes if s) / n


def planning_cost(plan_results: "Sequence[PlanResult]") -> tuple[int, float]:
    """Planning cost from the per-action ``PlanResult``\\ s: ``(planning_samples, time_per_action_ms)``.

    ``planning_samples`` is the planner samples drawn per action (``num_samples * num_iters``);
    ``time_per_action_ms`` is the mean wall-clock planning time per executed action in milliseconds
    (RFC-0005 §3-4). Raises on an empty set.
    """
    if not plan_results:
        raise _fail(
            "planning_cost over zero planning calls is undefined",
            "plan at least one action before reporting planning cost",
        )
    samples_per_action = plan_results[0].num_samples * plan_results[0].num_iters
    mean_seconds = sum(p.wall_time_s for p in plan_results) / len(plan_results)
    return samples_per_action, mean_seconds * 1000.0


# --- the collapse guard: effective dimension (RFC-0005 §4) ---


def effective_dim(embeddings: Tensor) -> float:
    """Participation ratio of the embedding covariance eigenspectrum — the collapse guard (RFC-0005 §4).

    ``embeddings`` is ``(num_vectors, d)`` (a ``(B, N, d)`` batch is flattened to ``(B*N, d)``). Returns
    ``(sum_i lambda_i)^2 / sum_i lambda_i^2`` over the non-negative eigenvalues ``lambda_i`` of the fp32
    sample covariance — in ``[1, d]``: ``~1`` for a rank-1 (collapsed) representation, ``d`` for an
    isotropic one. fp32 accumulation; eigenvalues are clamped non-negative (conditioning). Raises
    :class:`~lensemble.errors.EvaluationError` on fewer than two vectors or a degenerate (zero-variance)
    covariance, never a meaningless dimension.
    """
    x = embeddings.detach().reshape(embeddings.shape[0], -1).to(torch.float32)
    if x.shape[0] < 2:
        raise _fail(
            f"effective_dim needs >= 2 embedding vectors, got {x.shape[0]}",
            "pass at least two embeddings so the covariance is defined",
        )
    centered = x - x.mean(dim=0, keepdim=True)
    eigvals = covariance_eigenvalues(centered).clamp_min(0.0)
    sum1 = float(eigvals.sum())
    sum2 = float(eigvals.square().sum())
    if sum1 <= 0.0 or sum2 <= 0.0:
        raise _fail(
            "degenerate (zero-variance) covariance; effective_dim is undefined",
            "a collapsed-to-a-point representation has no participation ratio; investigate the encoder",
        )
    return (sum1 * sum1) / sum2


# --- supporting: representation-probe accuracy (RFC-0005 §4) ---


def linear_probe_accuracy(
    train_x: Tensor,
    train_y: Tensor,
    test_x: Tensor,
    test_y: Tensor,
    *,
    ridge: float = 1e-3,
) -> float:
    """Held-out accuracy of a closed-form linear probe on the frozen embeddings (RFC-0005 §4, supporting).

    Fits a ridge one-hot least-squares classifier on ``(train_x, train_y)`` and reports argmax accuracy on
    ``(test_x, test_y)``. Unit: fraction in ``[0, 1]``. Deterministic (no SGD). Raises on shape/label
    inconsistency.
    """
    tx = train_x.detach().reshape(train_x.shape[0], -1).to(torch.float32)
    ex = test_x.detach().reshape(test_x.shape[0], -1).to(torch.float32)
    ty = train_y.detach().to(torch.int64)
    ey = test_y.detach().to(torch.int64)
    if tx.shape[0] != ty.shape[0] or ex.shape[0] != ey.shape[0]:
        raise _fail(
            "probe x/y length mismatch",
            "pass aligned (embeddings, labels) for the train and test splits",
        )
    num_classes = int(torch.cat([ty, ey]).max()) + 1
    onehot = torch.zeros(tx.shape[0], num_classes)
    onehot[torch.arange(tx.shape[0]), ty] = 1.0
    gram = tx.transpose(-2, -1) @ tx + ridge * torch.eye(tx.shape[1])
    weights = torch.linalg.solve(gram, tx.transpose(-2, -1) @ onehot)
    pred = (ex @ weights).argmax(dim=-1)
    return float((pred == ey).to(torch.float32).mean())


# --- the communication-byte accountant (RFC-0005 §4 / 08 §4) ---


def comm_bytes(num_federated_params: int, *, quantized: bool = False) -> int:
    """Serialized per-round pseudo-gradient bytes (``fed/comm_bytes``, 08 §4). Unit: bytes.

    Full precision is ``4 * num_federated_params`` (fp32 wire dtype); int8 quantization is
    ``1 * num_federated_params`` plus a 4-byte per-tensor scale. Raises on a negative parameter count.
    """
    if num_federated_params < 0:
        raise _fail(
            f"num_federated_params must be >= 0, got {num_federated_params}",
            "pass the encoder+predictor federated parameter count",
        )
    if quantized:
        return num_federated_params * _BYTES_INT8 + _INT8_SCALE_BYTES
    return num_federated_params * _BYTES_FP32


def quant_ratio(num_federated_params: int) -> float:
    """The fp32-to-int8 communication reduction (``fed/quant_ratio``, ~4x; 08 §4). Unit: ratio."""
    return comm_bytes(num_federated_params, quantized=False) / max(
        1, comm_bytes(num_federated_params, quantized=True)
    )
