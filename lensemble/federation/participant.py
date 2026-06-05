"""lensemble.federation.participant — the sovereign-data ``Participant`` local round (RFC-0013 §1/§3).

The ``Participant`` holds data that never leaves its trust boundary (``INV-RESIDENCY``). One outer round
it runs ``H`` inner AdamW steps on its local windows, forms the DiLoCo delta ``Δ_c = (θ_c,φ_c) −
(θ_t,φ_t)`` over the federated encoder/predictor groups only (``INV-ACTIONHEAD-LOCAL``), DP-privatizes it
(clip-then-noise, the LOCKED ordering of RFC-0003 §4), binds it to its dataset Merkle root ``R_c``
(``INV-COMMIT-BINDING``), and returns the one object permitted across the boundary — a
:class:`~lensemble.federation.pseudogradient.PseudoGradient`.

Preconditions checked before any compute (``local_round``):

- ``INV-PROBE-PIN`` — ``global_state.probe_hash`` equals the participant's pinned-probe content hash, else
  :class:`~lensemble.errors.ProbeError` (``PROBE_INVALID``) and the round is refused.
- ``INV-WARMSTART-T0`` — at ``round_index == 0`` the fetched encoder's content hash equals the pinned
  warm-start hash, else :class:`~lensemble.errors.GaugeError`. (03 §7 states ``CheckpointIntegrityError``
  here; the #43 acceptance criterion pins ``GaugeError`` — the criterion governs, see ``_check_warmstart``.)
- ``INV-SKETCH-CONSISTENCY`` — ``round_seed == global_state.sketch_seed``; a participant whose derived ``A``
  would disagree is caught *before* release with :class:`~lensemble.errors.GaugeError`.

Postconditions on the released ``PseudoGradient`` (RFC-0013 §1):
``l2_norm == ‖released delta‖`` (the honest released norm; the DP bound ``INV-DP-BOUND`` is enforced on the
*post-clip* norm inside ``lensemble.privacy.dp.privatize``, so ``l2_norm`` MAY exceed ``C_clip`` after
noising), ``clipped is True``, the flat delta covers ONLY the ``(θ, φ)`` groups, and exactly one 32-byte
``dataset_root`` binds it. No raw observation/action/embedding crosses — only ``delta`` does, gated by the
``EgressRole.PSEUDO_GRADIENT`` carrier (``INV-RESIDENCY``).

#22 DATA-LAYER BOUNDARY. RFC-0013's ``Participant`` signature carries no data parameter. The participant's
pinned probe, local windows, and dataset root are resolved through protected hooks backed by ``cfg.data``:
``_local_dataset`` loads the configured source, ``_local_windows`` yields windows, and ``_dataset_root``
commits the loaded episodes to ``R_c``. Tests may still override the hooks to inject tiny fixtures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lensemble.config.seed import derive, round_sketch_seed
from lensemble.data.probe import PublicProbe, load_probe
from lensemble.errors import (
    GaugeError,
    LensembleErrorCode,
    ProbeError,
    RoundError,
)
from lensemble.federation.pseudogradient import PseudoGradient, build_pseudogradient
from lensemble.federation.state import GlobalState
from lensemble.gauge.anchor import FrameAnchor
from lensemble.model.action_head import ActionHead, build_action_head
from lensemble.model.encoder import (
    Encoder,
    build_encoder,
    encoder_content_hash,
    snapshot_reference,
)
from lensemble.model.numerics import module_input_tensor
from lensemble.model.objective import AnchorTerm, Objective
from lensemble.model.predictor import Predictor, build_predictor
from lensemble.model.sigreg import build_sketch
from lensemble.privacy.dp import DPConfig, privatize

if TYPE_CHECKING:
    from lensemble.config.schema import LensembleConfig
    from lensemble.contracts import ActionSpec
    from lensemble.data.dataset import EpisodeDataset
    from lensemble.data.episode import Window
    from lensemble.federation.transport import Transport

# Inner-loop AdamW learning rate when the config carries no override. The inner loop only needs to RUN
# and produce a real Δ (it is not convergence-tested); a small, stable default keeps the toy step finite.
_INNER_LR = 1e-3
# The fp32 tolerance the post-clip DP bound is asserted within (INV-DP-BOUND); matches dp.clip_delta's.
_DP_BOUND_TOL = 1e-6


@dataclass(frozen=True)
class RunResult:
    """Result of a single-site local training run (02-public-api 1.2; the Stage-A ``train_local`` output).

    Carries the committed checkpoint directory, its hash-verified ``content_hash`` (``INV-CHECKPOINT-HASH``,
    the value to commit), the deterministic eval-style ``RunManifest`` hash binding the run's exact inputs,
    and the final inner-loop loss. Scalars/hashes/paths only — no raw tensor crosses this boundary
    (``INV-RESIDENCY``).
    """

    checkpoint_dir: (
        Path  # the committed artifact directory (weights.safetensors + header.json)
    )
    checkpoint_hash: (
        str  # the artifact content_hash (64 lowercase hex, INV-CHECKPOINT-HASH)
    )
    manifest_hash: (
        str  # SHA-256 over the train_local-mode RunManifest (created_at excluded)
    )
    final_loss: float  # the last inner-AdamW-step objective total (a convergence diagnostic, not a gate)


class Participant:
    """Holds sovereign data; runs the inner loop; emits a privatized, bound ``PseudoGradient`` (RFC-0013 §1).

    Trusts neither coordinator nor peers with raw data (``INV-RESIDENCY``). One ``Participant`` per
    federation member; constructed with the signature fixed by conventions §5 / RFC-0013 §1.
    """

    def __init__(
        self,
        config: "LensembleConfig",
        *,
        participant_id: str,
        transport: "Transport",
    ) -> None:
        self.config = config
        self.participant_id = participant_id
        self.transport = transport

    # --- #22 data-layer boundary hooks (overridden by tests; resolved from cfg.data by default) ---

    def _pinned_probe(self) -> PublicProbe:
        """The participant's pinned public probe ``P`` (the #22 data-layer boundary).

        Default: load it from ``cfg.data.probe_path`` (``lensemble.data.probe.load_probe``). A run with no
        pinned probe path is refused with a clear :class:`~lensemble.errors.RoundError` (an anchored
        federated round requires a pinned probe, RFC-0004). Tests override this to inject a tiny probe.
        """
        probe_path = getattr(self.config.data, "probe_path", None)
        if probe_path is None:
            raise RoundError(
                "no pinned probe configured (cfg.data.probe_path is None); the #22 data layer resolves "
                "the participant's probe — override _pinned_probe or set a probe path",
                code=LensembleErrorCode.ROUND_FAILED,
                remediation="set cfg.data.probe_path to the pinned public probe, or override _pinned_probe",
            )
        return load_probe(probe_path)

    def _local_dataset(self) -> "EpisodeDataset":
        """Resolve the participant-local dataset from ``cfg.data`` through the #22 adapter registry."""
        from lensemble.data.adapters import load_episodes

        source = getattr(self.config.data, "data_source", None)
        if source is None:
            raise RoundError(
                "no local data source configured (cfg.data.data_source is None); the #22 data layer yields "
                "the participant's windows/commitment/spec — set cfg.data.data_source or override the "
                "participant data hooks",
                code=LensembleErrorCode.ROUND_FAILED,
                remediation="set cfg.data.data_source to a local episode store, or override participant data hooks",
            )
        return load_episodes(source, fmt=self.config.data.format)

    def _local_windows(self) -> Sequence["Window"]:
        """The participant's local training windows (the #22 data-layer boundary).

        Default (the #22-backed resolution, #167): when ``cfg.data.data_source`` is set the windows are
        materialized through the #22 data adapter — ``load_episodes(cfg.data.data_source, fmt=cfg.data.format)``
        then ``list(ds.windows(cfg.data.window_steps))``. With no configured source it fails closed citing
        #22 (the participant-override path tests exercise). The windows are RAW, residency-bound, and never
        cross a boundary — only the released ``delta`` does (``INV-RESIDENCY``).
        """
        dataset = self._local_dataset()
        return list(dataset.windows(int(self.config.data.window_steps)))

    def _local_windows_for_horizon(self, horizon: int) -> Sequence["Window"]:
        """Resolve only the local windows the inner loop can consume.

        ``_inner_loop`` indexes ``windows[step % len(windows)]`` for exactly
        ``horizon`` steps, so materializing more than ``horizon`` windows cannot
        affect training. For configured data sources, stream just that bounded
        prefix to keep large Phase 2 HDF5 silos from copying thousands of image
        windows into memory during a participant round. Override-only tests and
        custom participants with no configured source keep the legacy
        ``_local_windows`` path.
        """
        source = getattr(self.config.data, "data_source", None)
        if source is None:
            return self._local_windows()
        limit = max(1, int(horizon))
        dataset = self._local_dataset()
        windows: list[Window] = []
        for window in dataset.windows(int(self.config.data.window_steps)):
            windows.append(window)
            if len(windows) >= limit:
                break
        return windows

    def _dataset_root(self) -> bytes:
        """The participant's local dataset Merkle root ``R_c`` (32 bytes, ``INV-COMMIT-BINDING``).

        Default: commit the same local dataset resolved from ``cfg.data.data_source`` via
        :func:`lensemble.provenance.commit_dataset`; exactly one ``R_c`` binds the released delta
        (``INV-COMMIT-BINDING``). With no configured source this fails closed through ``_local_dataset``.
        """
        from lensemble.provenance import commit_dataset

        commitment = commit_dataset(self._local_dataset())
        return bytes.fromhex(commitment.merkle_root)

    def _action_spec(self) -> "ActionSpec":
        """The per-embodiment ``ActionSpec`` the local action head is built from (RFC-0007/0008; #22).

        Default (the #22-backed resolution, #167): when ``cfg.data.data_source`` is set, the spec is the
        ``action_spec`` declared on the loaded episodes (``load_episodes(...).episodes[0].action_spec``) —
        every episode in a participant store shares one embodiment. With no configured source this fails
        closed citing #22. Tests override it (or override ``_local_windows`` to also carry a spec). The
        action head is per-embodiment LOCAL state and is NEVER federated (``INV-ACTIONHEAD-LOCAL``).
        """
        source = getattr(self.config.data, "data_source", None)
        episodes = self._local_dataset().episodes
        if not episodes:
            raise RoundError(
                f"data source {source!r} declares no episodes, so no embodiment ActionSpec can be resolved",
                code=LensembleErrorCode.ROUND_FAILED,
                remediation="point cfg.data.data_source at a non-empty local episode store",
            )
        return episodes[0].action_spec

    # --- the federated round (RFC-0013 §1) ---

    def join(self, coordinator_endpoint: str) -> GlobalState:
        """Register with the coordinator and recover the latest committed ``GlobalState`` (RFC-0013 §3).

        Registers ``participant_id`` at ``coordinator_endpoint``, recovers the committed state, then
        VALIDATES it by fetching its θ/φ refs through ``transport.fetch_params`` (which hash-verifies the
        weights, ``INV-CHECKPOINT-HASH``). A tampered θ/φ artifact raises
        :class:`~lensemble.errors.CheckpointIntegrityError` from the fetch and ``join`` does not return.

        RECOVERY CONTRACT (RFC-0013 §3). A long-absent rejoiner adopts the recovered ``GlobalState`` as the
        SOLE source of truth and discards any stale local state — it does not carry forward an old round's
        ``(θ, φ)`` or round index, so the rejoiner is byte-identical to a fresh joiner from this point.
        When the recovered ``GlobalState.round_index == 0`` this ALSO revalidates ``INV-WARMSTART-T0``: the
        fetched encoder's content hash must equal the pinned warm-start (``_warmstart_hash``), else the
        gauge is not closed at ``t=0`` and :class:`~lensemble.errors.GaugeError` is raised here — the same
        check :meth:`local_round` runs, hoisted to the recovery path so a round-0 rejoiner fails fast.
        """
        self.transport.register(self.participant_id, coordinator_endpoint)
        gs = self.transport.recover_global_state(participant_id=self.participant_id)
        # Validate the recovered refs resolve and hash-verify before trusting the state (rejoiner path).
        theta_weights = self.transport.fetch_params(gs.theta_ref)
        self.transport.fetch_params(gs.phi_ref)
        # INV-WARMSTART-T0 on the recovery path: a round-0 rejoiner revalidates the warm-start gauge from
        # the recovered θ before adopting the state (RFC-0013 §3). Build a fresh encoder, load the fetched
        # θ_0, and run the SAME _check_warmstart local_round runs (a drift → GaugeError).
        if gs.round_index == 0:
            encoder = build_encoder(self.config)
            encoder.load_state_dict(theta_weights, strict=True)
            self._check_warmstart(gs, encoder)
        return gs

    def local_round(self, global_state: GlobalState, round_seed: int) -> PseudoGradient:
        """Run ``H`` inner AdamW steps, form ``Δ_c``, clip+noise, bind to ``R_c``, return it (RFC-0013 §1).

        See the module docstring for the full pre/postcondition contract (``INV-PROBE-PIN``,
        ``INV-WARMSTART-T0``, ``INV-SKETCH-CONSISTENCY``, ``INV-DP-BOUND``, ``INV-ACTIONHEAD-LOCAL``,
        ``INV-COMMIT-BINDING``, ``INV-RESIDENCY``).
        """
        cfg = self.config
        probe = self._pinned_probe()
        self._check_probe_pin(global_state, probe)

        # Fetch (θ_t, φ_t); fetch_params hash-verifies internally (CheckpointIntegrityError on tamper).
        theta_weights = self.transport.fetch_params(global_state.theta_ref)
        phi_weights = self.transport.fetch_params(global_state.phi_ref)

        encoder = build_encoder(cfg)
        predictor = build_predictor(cfg)
        encoder.load_state_dict(theta_weights, strict=True)
        predictor.load_state_dict(phi_weights, strict=True)

        # INV-WARMSTART-T0: at t==0 the loaded encoder must be hash-identical to the pinned warm-start.
        self._check_warmstart(global_state, encoder)

        # INV-SKETCH-CONSISTENCY: the round seed must equal the broadcast sketch seed.
        self._check_sketch_consistency(global_state, round_seed)

        # Per-embodiment LOCAL action head (INV-ACTIONHEAD-LOCAL): never federated, never released.
        action_head = build_action_head(cfg, self._action_spec())

        objective = self._build_objective(cfg, round_seed, encoder, probe)

        # Snapshot the global (θ_t, φ_t) BEFORE the inner loop so Δ = local − global is exact.
        theta_global = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
        phi_global = {k: v.detach().clone() for k, v in predictor.state_dict().items()}

        self._run_inner_loop(cfg, encoder, predictor, action_head, objective)

        # Δ over ONLY the encoder/predictor groups (state_dict keys prefixed encoder./predictor.).
        param_deltas: dict[str, Tensor] = {}
        for name, local in encoder.state_dict().items():
            param_deltas[f"encoder.{name}"] = local.detach() - theta_global[name]
        for name, local in predictor.state_dict().items():
            param_deltas[f"predictor.{name}"] = local.detach() - phi_global[name]
        # The action head is deliberately excluded (INV-ACTIONHEAD-LOCAL); never added to param_deltas.

        dataset_root = self._dataset_root()
        quantize = bool(getattr(cfg.federation, "quantize_pseudo_gradient", False))

        # build_pseudogradient flattens in the canonical (θ then φ, sorted) order AND fail-closes on any
        # non-federated / action-head group (ResidencyViolation, INV-ACTIONHEAD-LOCAL). We take its flat
        # delta as the DP input, then re-wrap the privatized delta.
        canonical = build_pseudogradient(
            param_deltas,
            dataset_root=dataset_root,
            round_index=global_state.round_index,
            clipped=False,
        )
        flat_delta = canonical.delta

        # DP clip-then-noise (the LOCKED ordering): the noise generator is seeded deterministically by
        # (root_seed, round_index, participant_id) and recorded in the RunManifest (RFC-0012 §4).
        dp_cfg = DPConfig(
            clip_norm=cfg.privacy.clip_norm,
            noise_multiplier=cfg.privacy.noise_multiplier,
            target_epsilon=cfg.privacy.epsilon,
            target_delta=cfg.privacy.delta,
            enabled=cfg.privacy.enabled,
        )
        generator = torch.Generator()
        generator.manual_seed(
            derive(
                cfg.determinism.root_seed,
                f"dp:{global_state.round_index}:{self.participant_id}",
            )
            % (2**63)
        )
        private, post_clip = privatize(flat_delta, dp_cfg, generator)

        # INV-DP-BOUND: the post-clip (pre-noise) norm is the bounded sensitivity (<= C_clip when DP is on).
        if dp_cfg.enabled:
            assert post_clip <= cfg.privacy.clip_norm + _DP_BOUND_TOL, (
                f"INV-DP-BOUND violated: post-clip norm {post_clip} > C_clip "
                f"{cfg.privacy.clip_norm}"
            )

        if quantize:
            from lensemble.federation.quant import wire_roundtrip

            private = wire_roundtrip(private)

        # The released delta is clipped-AND-noised; l2_norm is its HONEST norm (may exceed C_clip after
        # noising — correct; the DP bound is on `post_clip`, not on the released norm).
        return PseudoGradient(
            delta=private,
            l2_norm=float(private.norm()),
            dataset_root=dataset_root,
            round_index=global_state.round_index,
            clipped=True,
            quantized=quantize,
        )

    # --- precondition checks ---

    def _check_probe_pin(self, global_state: GlobalState, probe: PublicProbe) -> None:
        """INV-PROBE-PIN: refuse a round whose ``probe_hash`` differs from the pinned probe's hash."""
        if global_state.probe_hash != probe.content_hash:
            err = ProbeError(
                "GlobalState.probe_hash does not match the pinned probe content hash; refusing the round "
                "(a probe change is a re-anchoring event, RFC-0004 §3.1)",
                code=LensembleErrorCode.PROBE_INVALID,
                remediation="re-pin to the federation's probe or refuse the round (INV-PROBE-PIN)",
            )
            err.expected_hash = probe.content_hash  # type: ignore[attr-defined]
            err.got_hash = global_state.probe_hash  # type: ignore[attr-defined]
            raise err

    def _check_warmstart(self, global_state: GlobalState, encoder: Encoder) -> None:
        """INV-WARMSTART-T0: at ``round_index == 0`` the loaded encoder must equal the pinned warm-start.

        SPEC 03 §7 names ``CheckpointIntegrityError`` for a round-0 encoder-hash mismatch, but the #43
        acceptance criterion pins :class:`~lensemble.errors.GaugeError` here (the gauge is *closed* at
        ``t=0`` only when every participant's round-0 encoder is byte-identical; a drift IS a gauge
        failure). The issue governs: we raise ``GaugeError`` with the ``FRAME_DRIFT_EXCEEDED`` code.
        """
        if global_state.round_index != 0:
            return
        actual = encoder_content_hash(encoder)
        expected = self._warmstart_hash()
        if expected is not None and actual != expected:
            err = GaugeError(
                "round-0 encoder hash does not match the pinned warm-start; the latent gauge is not "
                "closed at t=0 (INV-WARMSTART-T0)",
                code=LensembleErrorCode.FRAME_DRIFT_EXCEEDED,
                remediation="load the pinned V-JEPA warm-start so every round-0 encoder is byte-identical",
            )
            err.expected_hash = expected  # type: ignore[attr-defined]
            err.got_hash = actual  # type: ignore[attr-defined]
            raise err

    def _warmstart_hash(self) -> str | None:
        """The pinned warm-start content hash (the #22/#10 boundary); ``None`` disables the t=0 check.

        Default: ``None`` (no pinned hash wired here — a real run resolves it from the warm-start release
        manifest). Tests override it to pin the round-0 encoder hash and exercise ``INV-WARMSTART-T0``.
        """
        return None

    def _check_sketch_consistency(
        self, global_state: GlobalState, round_seed: int
    ) -> None:
        """INV-SKETCH-CONSISTENCY: ``round_seed`` must equal the broadcast ``sketch_seed``.

        A participant whose derived projection ``A`` would disagree is caught HERE, before release, rather
        than surfacing as a non-deterministic aggregation at the outer step. Raises
        :class:`~lensemble.errors.GaugeError` (a local frame/consistency failure).
        """
        if round_seed != global_state.sketch_seed:
            err = GaugeError(
                f"round_seed {round_seed} != GlobalState.sketch_seed {global_state.sketch_seed}; the "
                "derived SIGReg sketch A would disagree across participants (INV-SKETCH-CONSISTENCY)",
                code=LensembleErrorCode.GAUGE_FAILED,
                remediation="run with the broadcast sketch_seed s_t so every participant derives the same A",
            )
            err.round_seed = round_seed  # type: ignore[attr-defined]
            err.sketch_seed = global_state.sketch_seed  # type: ignore[attr-defined]
            raise err

    # --- objective + inner loop ---

    def _build_objective(
        self,
        cfg: "LensembleConfig",
        round_seed: int,
        encoder: Encoder,
        probe: PublicProbe,
    ) -> Objective:
        """Build the per-round :class:`~lensemble.model.objective.Objective` from ``cfg.objective``.

        The SIGReg sketch ``A`` is derived from ``round_seed`` (``INV-SKETCH-CONSISTENCY``). When
        ``lambda_anc > 0`` the frame anchor is constructed from the pinned probe and the round-0 reference
        snapshot ``f_ref`` (``snapshot_reference(encoder)`` — at ``t=0`` the loaded encoder IS the
        warm-start, so its snapshot is the canonical ``f_ref``; ``INV-PROBE-PIN``/``INV-WARMSTART-T0``).
        With ``lambda_anc == 0`` the bare LeJEPA objective runs with ``anchor=None``.
        """
        o = cfg.objective
        anchor: AnchorTerm | None = None
        if float(o.lambda_anc) > 0.0:
            f_ref = snapshot_reference(encoder)
            landmarks = module_input_tensor(f_ref, probe.points[probe.landmark_idx])
            ref_embeddings = f_ref(landmarks).tokens.detach()
            anchor_obj = FrameAnchor(
                probe,
                ref_embeddings,
                variant=o.anchor_variant,
                probe_hash=probe.content_hash.hex(),
            )
            anchor = anchor_obj.loss
        # Derive A once here to assert it is constructible at this (seed, d); the Objective re-derives it
        # lazily at first call from the same seed, so the two agree (INV-SKETCH-CONSISTENCY). The encoder
        # latent dim d is the model's latent_dim (the V-JEPA hidden width).
        _ = build_sketch(round_seed, encoder.d, int(o.sigreg_sketch_dim))
        return Objective(
            lambda_pred=float(o.lambda_pred),
            lambda_sig=float(o.lambda_sig),
            lambda_anc=float(o.lambda_anc),
            sketch_seed=round_seed,
            sketch_dim=int(o.sigreg_sketch_dim),
            ep_knots=int(o.sigreg_knots),
            anchor=anchor,
            target_stop_gradient=bool(o.target_stop_gradient),
        )

    def _run_inner_loop(
        self,
        cfg: "LensembleConfig",
        encoder: Encoder,
        predictor: Predictor,
        action_head: ActionHead,
        objective: Objective,
    ) -> float:
        """Run ``H = cfg.federation.inner_horizon`` inner AdamW steps over θ/φ + the local head.

        Resolves the participant's local windows through the #22 hook, then delegates to the shared
        :func:`_inner_loop` (the SAME loop ``train_local`` runs, so the Stage-A single-site path and the
        federated round share one inner-loop body). Returns the final-step objective total (a diagnostic).
        The loop only needs to RUN and produce a real Δ — it is not convergence-tested; ``H`` and the data
        are kept tiny in tests. Optimizes the encoder, predictor, AND the local action head (the head is
        trained locally; only θ/φ are later released).
        """
        lr = float(getattr(cfg.federation, "inner_lr", _INNER_LR))
        horizon = int(cfg.federation.inner_horizon)
        windows = self._local_windows_for_horizon(horizon)
        return _inner_loop(
            encoder, predictor, action_head, objective, windows, horizon=horizon, lr=lr
        )


def _inner_loop(
    encoder: Encoder,
    predictor: Predictor,
    action_head: ActionHead,
    objective: Objective,
    windows: Sequence["Window"],
    *,
    horizon: int,
    lr: float,
) -> float:
    """The shared ``H``-step inner AdamW loop over θ/φ + the local head (RFC-0013 §1 / RFC-0001 Stage A).

    The single inner-loop body used by BOTH :meth:`Participant._run_inner_loop` (the federated round) and
    :func:`train_local` (the single-site Stage-A path), so the two never drift. Each step:
    ``loss = objective(encoder, predictor, window, action_head.encode(window.actions)).total``; ``backward``;
    ``step``. Cycles the windows when ``horizon`` exceeds their count. Fails closed on no windows
    (``RoundError``) — the loop needs at least one local :class:`~lensemble.data.episode.Window` to produce a
    real Δ. Returns the LAST step's objective total (a convergence diagnostic, not a gate; the toy loop is
    not convergence-tested). Optimizes the encoder, predictor, AND the local action head — only θ/φ are
    ever released downstream (``INV-ACTIONHEAD-LOCAL``).
    """
    if not windows:
        raise RoundError(
            "no local windows to run the inner loop on",
            code=LensembleErrorCode.ROUND_FAILED,
            remediation="provide at least one local Window (the #22 loader yields them)",
        )
    params = (
        list(encoder.parameters())
        + list(predictor.parameters())
        + list(action_head.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=lr)
    encoder.train()
    predictor.train()
    action_head.train()
    # Place each residency-bound window on the encoder's compute device so the forward runs single-device
    # (CUDA inner loop): the encoder/predictor/head are built on `resolve_device()` while the loader yields
    # CPU windows, and `autocast_forward` keys autocast on the input device — a CPU clip through a CUDA
    # encoder otherwise crashes (`Input type (BFloat16) and bias type (float)`). The move is in-process
    # (no boundary crossing, INV-RESIDENCY) and a no-op on the CPU fallback.
    device = next(encoder.parameters()).device
    final_loss = 0.0
    for step in range(horizon):
        raw = windows[step % len(windows)]
        window = replace(raw, obs=raw.obs.to(device), actions=raw.actions.to(device))
        optimizer.zero_grad(set_to_none=True)
        action_embedding = action_head.encode(window.actions)
        loss = objective(encoder, predictor, window, action_embedding).total
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return final_loss


def train_local(
    config: "LensembleConfig", *, run_dir: "Path | None" = None
) -> RunResult:
    """Single-site Stage-A training: warm-start → inner loop → commit checkpoint → manifest (RFC-0001).

    The single-site convenience path of [02-public-api §1.2]: it builds the encoder/predictor/action-head
    from ``config`` (the #166/#168 ``LensembleConfig``→architecture bridge), resolves local windows through
    the SAME #22 data layer the federated ``Participant`` uses (``cfg.data.data_source`` →
    ``load_episodes`` → ``windows(cfg.data.window_steps)``), runs ``cfg.federation.inner_horizon`` inner
    AdamW steps minimizing the composite SIGReg-JEPA objective via the shared :func:`_inner_loop`, then
    hash-commits the trained ``(θ, φ)`` with :func:`~lensemble.artifacts.checkpoint.save_checkpoint`
    (``INV-CHECKPOINT-HASH``) and binds the run to a deterministic ``train_local``-mode ``RunManifest``.

    Single-site == round 0: the objective's SIGReg sketch seed is ``round_sketch_seed(root_seed, 0)``
    (``INV-SKETCH-CONSISTENCY``). The anchor: with ``cfg.objective.lambda_anc > 0`` it is built from the
    pinned probe (``cfg.data.probe_path``) exactly as the participant builds it (round-0 reference snapshot
    ``f_ref``); with ``lambda_anc == 0`` the bare LeJEPA objective runs (``anchor=None``).

    Returns a :class:`RunResult` (checkpoint dir + ``content_hash`` + manifest hash + final loss). Only
    the shared ``encoder``/``predictor`` weights enter the artifact (``INV-ACTIONHEAD-LOCAL``: the artifact
    boundary fail-closes on a non-encoder/predictor group; the local head is never serialized here). No raw
    observation/action crosses a boundary — the windows are RAW and local (``INV-RESIDENCY``).

    Raises :class:`~lensemble.errors.RoundError` when no local windows resolve (no ``data_source`` and the
    store yields nothing), and :class:`~lensemble.errors.ConfigError` on an invalid ``cfg.model``/spec.
    """
    import hashlib
    import tempfile

    from lensemble.artifacts.checkpoint import (
        model_arch_from_config,
        save_checkpoint,
    )
    from lensemble.config.manifest import build_manifest, config_hash

    cfg = config
    encoder = build_encoder(cfg)
    predictor = build_predictor(cfg)

    # Resolve the action spec + local windows through the #22 data layer (a single-participant Participant
    # reuses the SAME hooks the federated path uses; the default hooks read cfg.data — #167).
    from lensemble.federation.transport import InProcessTransport

    site = Participant(cfg, participant_id="local", transport=InProcessTransport())
    action_head = build_action_head(cfg, site._action_spec())
    horizon = int(cfg.federation.inner_horizon)
    windows = site._local_windows_for_horizon(horizon)

    # Single-site == round 0: the broadcast sketch seed s_0 (INV-SKETCH-CONSISTENCY).
    sketch_seed = round_sketch_seed(cfg.determinism.root_seed, 0)

    # Build the objective. With lambda_anc > 0 the anchor is built from the pinned probe exactly as the
    # participant does; with lambda_anc == 0 the bare LeJEPA objective runs (anchor=None).
    o = cfg.objective
    anchor: AnchorTerm | None = None
    if float(o.lambda_anc) > 0.0:
        probe = site._pinned_probe()
        f_ref = snapshot_reference(encoder)
        landmarks = module_input_tensor(f_ref, probe.points[probe.landmark_idx])
        ref_embeddings = f_ref(landmarks).tokens.detach()
        anchor = FrameAnchor(
            probe,
            ref_embeddings,
            variant=o.anchor_variant,
            probe_hash=probe.content_hash.hex(),
        ).loss
    objective = Objective(
        lambda_pred=float(o.lambda_pred),
        lambda_sig=float(o.lambda_sig),
        lambda_anc=float(o.lambda_anc),
        sketch_seed=sketch_seed,
        sketch_dim=int(o.sigreg_sketch_dim),
        ep_knots=int(o.sigreg_knots),
        anchor=anchor,
        target_stop_gradient=bool(o.target_stop_gradient),
    )

    lr = float(getattr(cfg.federation, "inner_lr", _INNER_LR))
    final_loss = _inner_loop(
        encoder, predictor, action_head, objective, windows, horizon=horizon, lr=lr
    )

    # Commit the trained (θ, φ) to a hash-verified artifact (INV-CHECKPOINT-HASH). Only the shared groups
    # enter the artifact — the action head is local-only (INV-ACTIONHEAD-LOCAL; the boundary fail-closes).
    run_dir = (
        Path(run_dir)
        if run_dir is not None
        else Path(tempfile.mkdtemp(prefix="lensemble-train-local-"))
    )
    ckpt_dir = run_dir / "checkpoint"
    weights: dict[str, Tensor] = {}
    for name, tensor in encoder.state_dict().items():
        weights[f"encoder.{name}"] = tensor
    for name, tensor in predictor.state_dict().items():
        weights[f"predictor.{name}"] = tensor
    checkpoint_hash = save_checkpoint(
        ckpt_dir,
        weights,
        wmcp_version=cfg.model.wmcp_version,
        round_index=0,  # single-site Stage A is round 0
        config_hash=config_hash(_config_asdict(cfg)),
        parent_hash=None,
        model_arch=model_arch_from_config(cfg),  # self-describing checkpoint (#171)
    )

    manifest = build_manifest(cfg, run_mode="train_local")
    manifest_hash = hashlib.sha256(
        manifest.model_dump_json(exclude={"created_at"}).encode()
    ).hexdigest()

    return RunResult(
        checkpoint_dir=ckpt_dir,
        checkpoint_hash=checkpoint_hash,
        manifest_hash=manifest_hash,
        final_loss=final_loss,
    )


def _config_asdict(cfg: "LensembleConfig") -> dict[str, object]:
    """``dataclasses.asdict`` of the resolved config (the canonical config_hash input, RFC-0009 7)."""
    from dataclasses import asdict

    return asdict(cfg)
