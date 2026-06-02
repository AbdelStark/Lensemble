"""lensemble.model.numerics — the model subsystem's dtype/device/determinism contract (RFC-0008 7).

The contract (conventions 9, RFC-0008 7):

- **Forward dtype**: bf16 forward by default on the CUDA primary; the CPU fallback computes in fp32 (CPU
  bf16 is imprecise/slow and the CI configs run fp32). Master weights are always fp32.
- **Accumulation**: loss and statistic accumulation is fp32 (:data:`ACCUMULATION_DTYPE`; fp64 where
  configured) — see ``Objective.__call__`` and ``sigreg_statistic``.
- **Device**: CUDA primary, CPU fallback for the small CI configs (tests pass on CPU).
- **Determinism (inner)**: best-effort and seed-pinned; full determinism is gated by the config flag
  via :func:`set_determinism` (``torch.use_deterministic_algorithms``). The inner forward/backward is
  NOT required to be bitwise-deterministic — that requirement (``INV-AGG-DETERMINISM``) is on the
  aggregation/outer-step path, owned elsewhere. What this module guarantees is narrower and sufficient:
  per-term loss scalars reproduce within fp32 tolerance given identical inputs, seed, and sketch.

The ``build_*`` constructors call :func:`apply_numerics` to place a module on the resolved device and
set its ``compute_dtype``; ``Encoder.forward``/``Predictor.forward`` run their compute under
:func:`autocast_forward`, a no-op when the compute dtype is fp32 (so the CPU path is unchanged).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import nn

# Loss / statistic accumulation dtype (conventions 9; fp64 where a config opts in for aggregation).
ACCUMULATION_DTYPE = torch.float32


def resolve_device(prefer_cuda: bool = True) -> torch.device:
    """The compute device: CUDA primary, CPU fallback (conventions 9). Tests run on the CPU fallback."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def forward_dtype(device: torch.device) -> torch.dtype:
    """The forward compute dtype: bf16 on the CUDA primary, fp32 on the CPU fallback (RFC-0008 7).

    bf16 forward is the default where it pays off (CUDA); the CPU fallback computes in fp32 because CPU
    bf16 is imprecise and slow and the CI configs are tiny. Accumulation is fp32 regardless.
    """
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def apply_numerics(module: nn.Module, device: torch.device) -> None:
    """Place ``module`` on ``device`` and record its forward ``compute_dtype`` (called by ``build_*``)."""
    module.to(device)
    module.compute_dtype = forward_dtype(device)  # type: ignore[assignment]


@contextmanager
def autocast_forward(
    device: torch.device, dtype: torch.dtype = torch.float32
) -> Iterator[None]:
    """Run a forward under bf16 autocast when ``dtype`` is not fp32; a no-op for fp32 (CPU fallback).

    Autocast lowers matmul/conv to ``dtype`` while master weights stay fp32 — the bf16-forward /
    fp32-master-weights contract. Disabled for fp32 so the CPU path is bit-for-bit the pre-existing one.
    """
    enabled = dtype != torch.float32
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
        yield


def set_determinism(enabled: bool, *, warn_only: bool = True) -> None:
    """Gate ``torch.use_deterministic_algorithms`` from the config flag (best-effort inner determinism).

    ``warn_only`` keeps an op without a deterministic implementation from raising on the CPU fallback;
    the inner loop is best-effort, not bitwise (the bitwise guarantee is the outer-step path's).
    """
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
