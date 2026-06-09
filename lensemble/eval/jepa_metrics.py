"""Residency-safe per-round JEPA learning metrics.

The federated HF Jobs launchers (the claim-MVP ``train_federated_lewm.py`` and the Phase 3 consortium
``train_phase3_consortium.py``) both report the same four scalars per closed round so their reports are
directly comparable:

- ``val_pred`` — held-out next-latent prediction MSE (does the world model *learn* to predict?).
- ``val_sigreg`` — the SIGReg anti-collapse statistic on held-out embeddings.
- ``effective_rank`` — ``exp(entropy of the embedding-covariance eigenspectrum)``; ``~1-3`` means the
  representation has collapsed, a healthy run keeps a large fraction of ``d`` (RFC-0005 §4).
- ``frame_drift_deg`` — the inter-participant latent gauge rotation on the public probe (RFC-0002).

Every function consumes committed checkpoints, the public probe, and *pseudo-gradient deltas* only —
never raw participant trajectories — so the scalars are safe to emit into a published report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from lensemble.artifacts import load_checkpoint
from lensemble.config import LensembleConfig
from lensemble.contracts import LatentState
from lensemble.data.episode import Window
from lensemble.data.probe import PublicProbe
from lensemble.gauge import frame_drift
from lensemble.model import (
    build_encoder,
    build_predictor,
    build_sketch,
    sigreg_statistic,
)
from lensemble.model.numerics import module_input_tensor

__all__ = [
    "JepaWindowMetrics",
    "effective_rank",
    "evaluate_jepa_windows",
    "frame_drift_deg_from_updates",
    "load_checkpoint_groups",
    "load_round_models",
    "mean_frame_drift_deg",
]


@dataclass(frozen=True)
class JepaWindowMetrics:
    """Held-out scalar metrics for one committed global model."""

    val_pred: float
    val_sigreg: float
    effective_rank: float


def effective_rank(embeddings: torch.Tensor) -> float:
    """``exp(entropy of the covariance eigenspectrum)`` — the collapse read-out (RFC-0005 §4).

    The covariance eigenvalues are computed via the stable SVD path (:func:`covariance_eigenvalues`): a
    collapsed representation's ``X^T X`` is so ill-conditioned that ``eigvalsh`` diverges on CUDA (#264),
    and the collapsed regime is exactly what this metric must read out.
    """
    from lensemble.eval.metrics import covariance_eigenvalues

    x = embeddings.reshape(-1, embeddings.shape[-1]).to(torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    ev = covariance_eigenvalues(x).clamp_min(1e-12)
    p = ev / ev.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def load_checkpoint_groups(
    checkpoint_dir: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split a committed checkpoint into ``encoder.*`` (θ) and ``predictor.*`` (φ) state dicts."""

    weights, _header = load_checkpoint(checkpoint_dir)
    theta = {
        k.removeprefix("encoder."): v
        for k, v in weights.items()
        if k.startswith("encoder.")
    }
    phi = {
        k.removeprefix("predictor."): v
        for k, v in weights.items()
        if k.startswith("predictor.")
    }
    return theta, phi


def load_round_models(cfg: LensembleConfig, checkpoint_dir: Path) -> tuple[Any, Any]:
    """Rebuild and load the encoder/predictor from a committed round checkpoint (eval mode)."""

    theta, phi = load_checkpoint_groups(checkpoint_dir)
    encoder = build_encoder(cfg).eval()
    predictor = build_predictor(cfg).eval()
    encoder.load_state_dict(theta, strict=True)
    predictor.load_state_dict(phi, strict=True)
    return encoder, predictor


def evaluate_jepa_windows(
    cfg: LensembleConfig,
    *,
    encoder: Any,
    predictor: Any,
    action_head: Any,
    windows: Sequence[Window],
    max_windows: int,
) -> JepaWindowMetrics | None:
    """Compute ``val_pred``/``val_sigreg``/``effective_rank`` over up to ``max_windows`` held-out windows.

    Returns ``None`` when no usable window is available (so the caller can record absent metrics rather
    than a misleading zero).
    """

    sketch = build_sketch(0, int(cfg.model.latent_dim), 64)
    pred_losses: list[float] = []
    sigreg_losses: list[float] = []
    embeddings: list[torch.Tensor] = []
    remaining = max(0, int(max_windows))
    with torch.no_grad():
        for window in windows:
            if remaining <= 0:
                break
            encoded = encoder(window.obs)
            tokens = encoded.tokens
            input_latent = LatentState(
                tokens=tokens[:-1],
                num_tokens=encoded.num_tokens,
                dim=encoded.dim,
                wmcp_version=encoded.wmcp_version,
            )
            target = tokens[1:]
            action_embedding = action_head.encode(window.actions)
            pred_tokens = predictor(input_latent, action_embedding).tokens
            pred_losses.append(float((pred_tokens - target).pow(2).mean()))
            sigreg_losses.append(
                float(sigreg_statistic(tokens.reshape(-1, tokens.shape[-1]), sketch))
            )
            embeddings.append(tokens.reshape(-1, tokens.shape[-1]).cpu())
            remaining -= 1
    if not pred_losses or not embeddings:
        return None
    return JepaWindowMetrics(
        val_pred=sum(pred_losses) / len(pred_losses),
        val_sigreg=sum(sigreg_losses) / len(sigreg_losses),
        effective_rank=effective_rank(torch.cat(embeddings, dim=0)),
    )


def _probe_embedding(encoder: Any, probe: PublicProbe) -> torch.Tensor:
    encoded = encoder(module_input_tensor(encoder, probe.points))
    return encoded.tokens.reshape(-1, encoded.tokens.shape[-1]).detach().cpu()


def _unflatten_update_delta(
    theta_template: Mapping[str, torch.Tensor],
    phi_template: Mapping[str, torch.Tensor],
    flat: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    theta: dict[str, torch.Tensor] = {}
    phi: dict[str, torch.Tensor] = {}
    offset = 0
    for group, template, out in (
        ("encoder", theta_template, theta),
        ("predictor", phi_template, phi),
    ):
        for name in sorted(template):
            ref = template[name]
            end = offset + ref.numel()
            if end > flat.numel():
                raise ValueError(
                    f"released delta ended inside {group}.{name}; expected {end} values, "
                    f"got {flat.numel()}"
                )
            out[name] = flat[offset:end].reshape(ref.shape)
            offset = end
    if offset != flat.numel():
        raise ValueError(
            f"released delta has {flat.numel() - offset} trailing values after "
            "encoder/predictor groups"
        )
    return theta, phi


def _apply_theta_delta(
    theta: Mapping[str, torch.Tensor], delta: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    updated: dict[str, torch.Tensor] = {}
    for name, base in theta.items():
        change = delta[name].to(device=base.device)
        if torch.is_floating_point(base):
            updated[name] = base + change.to(dtype=base.dtype)
        else:
            updated[name] = (
                (base.to(torch.float32) + change).round().to(dtype=base.dtype)
            )
    return updated


def mean_frame_drift_deg(report: Any) -> float | None:
    """Mean inter-participant rotation angle (deg), falling back to drift-from-global."""

    angles = [pair.rotation_angle_deg for pair in report.pairs]
    if not angles:
        angles = list(report.drift_from_global.values())
    if not angles:
        return None
    return sum(float(angle) for angle in angles) / len(angles)


def frame_drift_deg_from_updates(
    cfg: LensembleConfig,
    *,
    prior_checkpoint_dir: Path,
    final_encoder: Any,
    probe: PublicProbe,
    updates: Mapping[str, Any],
    prior_round: int,
) -> float | None:
    """Inter-participant latent frame-drift (deg) on the public probe for one closed round (RFC-0002).

    Reconstructs each participant's local encoder by applying its pseudo-gradient θ-delta to the prior
    committed global θ, embeds the public probe, and measures the optimal-rotation angle against the
    aggregated global. Uses only committed weights + released deltas — no raw trajectories.
    """

    if not updates:
        return None
    prior_theta, prior_phi = load_checkpoint_groups(prior_checkpoint_dir)
    embeddings = {"global": _probe_embedding(final_encoder, probe)}
    for participant_id, update in sorted(updates.items()):
        if int(update.round_index) != prior_round:
            raise ValueError(
                f"participant {participant_id!r} update is for round {update.round_index}, "
                f"but this drift measurement expects round {prior_round}"
            )
        theta_delta, _phi_delta = _unflatten_update_delta(
            prior_theta, prior_phi, update.delta
        )
        local_encoder = build_encoder(cfg).eval()
        local_encoder.load_state_dict(
            _apply_theta_delta(prior_theta, theta_delta), strict=True
        )
        embeddings[participant_id] = _probe_embedding(local_encoder, probe)
    report = frame_drift(
        embeddings,
        round_index=prior_round,
        probe=probe,
        expected_probe_hash=probe.content_hash.hex(),
        # A strong anchor (the converged regime) can pin participants onto a near-identical frame; that
        # coinciding-frame case is ~0° drift, not a degenerate-SVD error that should abort the metric.
        degenerate_safe=True,
    )
    return mean_frame_drift_deg(report)
