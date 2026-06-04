"""lensemble.model.objective — the composite SIGReg-JEPA + frame-anchor loss (RFC-0008 5).

The objective is the three weighted terms
``L = lambda_pred * E||g_phi(f(x_t),a_t) - sg[f(x_{t+1})]||^2 + lambda_sig * SIGReg_A(f(x)) + lambda_anc * L_anchor``.
:meth:`Objective.__call__` returns a frozen :class:`LossTerms` carrying each per-term scalar (so the
metric stream can emit ``loss/pred``, ``loss/sigreg``, ``loss/anchor``) plus the weighted ``total`` that
``.backward()`` is called on. The per-term scalars are aggregate metrics, not raw embeddings, so emitting
them does not engage ``INV-RESIDENCY``.

Per-round sketch-seed contract. The objective is constructed per round so the SIGReg sketch ``A`` is
derived from the broadcast seed ``s_t`` and is identical across participants (``INV-SKETCH-CONSISTENCY``,
RFC-0002 3). The anchor term is *injected* as an :class:`AnchorTerm` callable rather than imported, so
``model`` never imports ``gauge`` (the module DAG stays acyclic, RFC-0001 3).

Gauge symmetry (RFC-0002 2/4). Under ``f -> Qf, g -> QgQ^T`` for ``Q in O(d)`` the prediction term is
invariant (``||Qr|| = ||r||``) and SIGReg is invariant *when its sketch co-rotates* (``A -> QA``, since the
sketch is a frame of latent-space directions: ``SIGReg_{QA}(Qf) = SIGReg_A(f)`` exactly). Only the anchor
term breaks the symmetry, by design. ``lambda_anc == 0.0`` with ``anchor=None`` is the bare LeJEPA
objective used by the gauge-invariance test and by Fork A (encoder frozen).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from lensemble.contracts import LatentState, check_latent_state
from lensemble.model.sigreg import build_sketch, sigreg_statistic


class _EncoderLike(Protocol):
    """Anything that maps observations to a ``LatentState`` (the encoder ``f_theta``).

    Structural, not the concrete :class:`~lensemble.model.encoder.Encoder`, so the gauge-invariance
    test can pass a rotated encoder ``Qf`` (RFC-0002 4) and Fork A can pass a frozen one.
    """

    def __call__(self, obs: object) -> LatentState: ...


class _PredictorLike(Protocol):
    """Anything exposing the stop-gradiented prediction residual (the predictor ``g_phi``).

    Structural so the gauge test can pass the conjugated predictor ``QgQ^T``.
    """

    def prediction_residual(
        self,
        latent: LatentState,
        action_embedding: Tensor,
        next_latent: LatentState,
    ) -> Tensor: ...


@dataclass(frozen=True)
class LossTerms:
    """Per-term loss scalars (RFC-0015 metric names in parentheses). Each is an fp32 0-dim tensor.

    ``pred`` (loss/pred), ``sigreg`` (loss/sigreg), ``anchor`` (loss/anchor), and ``total`` — the
    weighted sum ``.backward()`` is called on (``total`` requires grad).
    """

    pred: Tensor
    sigreg: Tensor
    anchor: Tensor
    total: Tensor


class AnchorTerm(Protocol):
    """The injected, unweighted frame-anchor term (``gauge.FrameAnchor.loss``); fp32 0-dim.

    Injected by the caller rather than imported, to keep ``model`` from importing ``gauge``. It may
    raise ``GaugeError('FrameDriftExceeded')`` when the landmark anchor is under-determined (``k < d``).
    """

    def __call__(self, encoder: _EncoderLike, /) -> Tensor: ...


class Objective:
    """The three-term SIGReg-JEPA + frame-anchor loss (RFC-0008 5).

    Constructed per round from the broadcast sketch seed ``s_t`` (``INV-SKETCH-CONSISTENCY``) and the
    injected anchor. ``lambda_anc == 0.0`` with ``anchor=None`` is the bare LeJEPA objective.
    """

    def __init__(
        self,
        *,
        lambda_pred: float,
        lambda_sig: float,
        lambda_anc: float,
        sketch_seed: int,
        sketch_dim: int = 64,
        ep_knots: int = 17,
        anchor: AnchorTerm | None = None,
        sketch: Tensor | None = None,
    ) -> None:
        """Construct the per-round objective.

        ``sketch`` defaults to ``build_sketch(sketch_seed, d, sketch_dim)`` built lazily at first call,
        once ``d`` is known from the encoder. An explicit ``sketch`` overrides the seed-built one; it is
        used by the gauge-invariance test, which co-rotates the SIGReg frame (``A -> QA``) under the
        ``O(d)`` gauge transform. ``ep_knots`` is the Epps-Pulley integration grid (RFC-0008 6).
        """
        self.lambda_pred = float(lambda_pred)
        self.lambda_sig = float(lambda_sig)
        self.lambda_anc = float(lambda_anc)
        self.sketch_seed = int(sketch_seed)
        self.sketch_dim = int(sketch_dim)
        self.ep_knots = int(ep_knots)
        self.anchor = anchor
        self._sketch = sketch

    def __call__(
        self,
        encoder: _EncoderLike,
        predictor: _PredictorLike,
        window: object,
        action_embedding: Tensor,
    ) -> LossTerms:
        """Encode, condition, predict, and evaluate the three terms; return the weighted total.

        ``window.obs`` is the ``(num_steps + 1, *modality)`` frame stack (03 5); it is encoded as a
        batch so that frames ``[:-1]`` are the inputs ``f(x_t)`` and ``[1:]`` the targets ``f(x_{t+1})``.
        ``action_embedding`` is ``(num_steps, cond_dim)``. The target ``f(x_{t+1})`` is stop-gradiented
        inside :meth:`Predictor.prediction_residual`.

        Raises ``ContractViolation`` (``INV-WMCP``) if the encoder output is non-conformant, and
        surfaces ``GaugeError`` from the injected anchor when the landmark anchor is under-determined.
        """
        obs = getattr(window, "obs")
        encoded = encoder(obs)
        check_latent_state(encoded)  # INV-WMCP — hard reject on a non-conformant latent
        tokens = encoded.tokens
        d = tokens.shape[-1]

        sketch = self._sketch
        if sketch is None:
            sketch = build_sketch(self.sketch_seed, d, self.sketch_dim)
            self._sketch = sketch
        elif sketch.shape[0] != d:
            raise ValueError(
                f"sketch row dim {sketch.shape[0]} != encoder latent dim {d}"
            )

        input_latent = LatentState(
            tokens=tokens[:-1],
            num_tokens=encoded.num_tokens,
            dim=encoded.dim,
            wmcp_version=encoded.wmcp_version,
        )
        target_latent = LatentState(
            tokens=tokens[1:],
            num_tokens=encoded.num_tokens,
            dim=encoded.dim,
            wmcp_version=encoded.wmcp_version,
        )

        # Prediction term: residual carries the stop-gradient on f(x_{t+1}) (Predictor.prediction_residual).
        residual = predictor.prediction_residual(
            input_latent, action_embedding, target_latent
        )
        pred = residual.pow(2).mean().to(torch.float32)

        # SIGReg over all embeddings flattened to (M, d); the shared sketch A is fixed for the round.
        sigreg = sigreg_statistic(
            tokens.reshape(-1, d), sketch, ep_knots=self.ep_knots
        ).to(torch.float32)

        if self.anchor is not None:
            anchor = self.anchor(encoder).to(torch.float32)
        else:
            # On the encoder's device so the weighted total stays single-device (CUDA inner loop);
            # `pred` and `sigreg` already follow `tokens.device`.
            anchor = torch.zeros((), dtype=torch.float32, device=tokens.device)

        total = (
            self.lambda_pred * pred
            + self.lambda_sig * sigreg
            + self.lambda_anc * anchor
        )
        return LossTerms(pred=pred, sigreg=sigreg, anchor=anchor, total=total)
