"""lensemble.federation.coordinator — the outer-round orchestrator (RFC-0013 §1/§4/§6).

The ``Coordinator`` is the runtime that executes the DiLoCo outer loop of
[RFC-0003](../../docs/rfcs/RFC-0003-federated-protocol.md): it owns the canonical global model
``(θ_t, φ_t)``, drives one outer round through the :class:`~lensemble.federation.round.RoundState`
machine, runs the Nesterov outer step over *only* the encoder/predictor param groups, and hash-commits
each round. One ``Coordinator`` per federation runs a single sequential round loop
(RFC-0013 §6: round ``t+1`` does not open until round ``t`` reaches ``CLOSED`` or ``ABORTED``).

Per-round lifecycle (RFC-0013 §1/§2), all on the single coordinator thread in the canonical order:

- **OPEN** — pin ``(θ_t, φ_t)``, derive ``s_t = round_sketch_seed(root_seed, t)``, build the round
  :class:`~lensemble.federation.state.GlobalState` (refs to the committed θ/φ, ``sketch_seed=s_t``), and
  ``transport.broadcast_round_open`` it (the ``RoundOpen`` payload, RFC-0013 §5). The broadcast state
  references ONLY θ/φ — never an action head (``INV-ACTIONHEAD-LOCAL``).
- **COLLECTING** — ``transport.collect_updates(t)``. The round quorum is
  ``K = max(cfg.federation.fault_tolerance_min_participants, cfg.federation.secure_agg_threshold)``
  (RFC-0013 §3): below ``K`` survivors the secure-aggregation reveal cannot be unblinded, so a
  contributing count below ``K`` aborts the round with
  :class:`~lensemble.errors.FaultToleranceExceeded` (the round → ``ABORTED``, the global hash AND the
  round index unchanged — no partial commit, no advance, so the SAME round ``t`` may be re-attempted once
  enough updates are staged). The *present* set is whatever ``collect_updates(t)`` returns: the
  in-process transport models the post-``collect_timeout_s`` present set as the collected set (the
  wall-clock drop is the network seam #45), and a delta for a PAST round is never back-applied — a
  dropped participant reconciles by contributing at the NEXT round.
- **AGGREGATING** — the determinism self-check (``INV-AGG-DETERMINISM``, RFC-0013 §4): the reduction
  ``(1/C)·Σ_c Δ_c`` is re-run under the canonical participant-id-sorted order and compared bitwise via
  :func:`~lensemble.aggregation.determinism.assert_outer_step_deterministic` (a FRESH optimizer per call
  inside the thunk → pure). A mismatch raises :class:`~lensemble.errors.NonDeterministicAggregation`
  (security-critical, never swallowed) and the round → ``ABORTED``. Arrival order does not matter — the
  reduction is over the total order on ``participant_id``.
- **ALIGNING** — frame drift is measured on the probe when per-participant embeddings are available, AND
  the Layer-3 Procrustes backstop (RFC-0002 §5, #18) is applied when BOTH per-participant embeddings and a
  reference frame ``E_ref`` are wired (the ``_probe_embeddings``/``_reference_embeddings`` #18/#22 seam):
  each participant whose drift exceeds ``cfg.gauge.frame_drift_threshold_deg`` has its PREDICTOR delta
  conjugated by ``Q_c* = procrustes_align(f_c(P), E_ref)`` as a PURE LINEAR operation
  (``g_phi -> Q g_phi Q^T``, applied to the released delta). The ENCODER delta is left UNCHANGED — the
  activation-space realization of the recorded #18 decision: the encoder ends in a ``LayerNorm`` with no
  terminal ``(d, d)`` linear to fold ``Q`` into, so the encoder-frame component is bounded by the Layer-2
  anchor (RFC-0002 §4) and lives in activation space, NOT in the committed weights (which is why
  ``recompute_alignment`` (#62) measures residual encoder drift rather than verifying a weight-fold — the
  #18/#62 verifiability tradeoff). The backstop is a deterministic pre-step feeding the SAME outer reduction
  (``INV-AGG-DETERMINISM``). With nothing wired (the default) ALIGNING is a byte-identical MEASURED
  PASS-THROUGH and does not mutate θ/φ. With the masking secure-aggregation backend the coordinator sees
  only the summed update (no per-participant deltas), so the backstop is a Stage-B / simulated-backend
  operation; the hooks are the seam.
- **COMMITTING** — the PERSISTENT :class:`~lensemble.federation.outer.OuterOptimizer.step` folds the
  averaged delta into the global params → ``θ_{t+1}⊕φ_{t+1}`` (covers ONLY θ/φ; the deltas are
  ``PseudoGradient`` s that by construction carry no action head, ``INV-ACTIONHEAD-LOCAL``). The flat
  vector is UN-flattened via the param manifest into an ``encoder.*``/``predictor.*`` state_dict and
  hash-committed with :func:`~lensemble.artifacts.checkpoint.save_checkpoint` (round ``t+1``,
  ``parent_hash`` = the current global hash, ``INV-CHECKPOINT-HASH``). A
  :class:`~lensemble.provenance.ledger.ContributionRecord` is appended to the
  :class:`~lensemble.provenance.ledger.ContributionLedger` recording the contributing participants, their
  dataset roots, and the resulting ``global_model_hash``. ``driver.commit`` advances the canonical hash
  (→ ``CLOSED``).
- **CLOSED → next** — ``driver.open_next`` opens round ``t+1`` unless this was the last requested round.

The averaging denominator is the actual contributing count ``C_t`` (recorded in the ``ContributionRecord``;
:class:`~lensemble.federation.outer.OuterOptimizer.average_deltas` divides by ``len(deltas)``), so the
outer step is reproducible from recorded inputs.

#22/#04 BOUNDARY (probe pin). When ``cfg.data.probe_path`` is set the pinned probe is loaded and hashed
into ``GlobalState.probe_hash``; otherwise a fixed 32-byte placeholder is used (a participant pinning a
real probe would refuse such a round, ``INV-PROBE-PIN``). The real probe resolution lands with #22/#04.
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import torch

from lensemble.aggregation.determinism import assert_outer_step_deterministic
from lensemble.config.manifest import config_hash
from lensemble.config.seed import round_sketch_seed
from lensemble.data.probe import load_probe
from lensemble.errors import (
    FaultToleranceExceeded,
    LensembleErrorCode,
    NonDeterministicAggregation,
)
from lensemble.federation.outer import OuterOptimizer
from lensemble.federation.pseudogradient import PseudoGradient, build_pseudogradient
from lensemble.federation.round import RoundDriver, RoundState
from lensemble.federation.state import GlobalState, ParamRef
from lensemble.federation.transport import weights_content_hash
from lensemble.gauge.backstop import procrustes_backstop
from lensemble.gauge.drift import FrameDriftReport, frame_drift
from lensemble.model.encoder import build_encoder
from lensemble.model.predictor import build_predictor
from lensemble.provenance.ledger import ContributionLedger, ContributionRecord

if TYPE_CHECKING:
    from torch import Tensor

    from lensemble.config.schema import LensembleConfig
    from lensemble.federation.transport import Transport

# The federated param groups, in the order build_pseudogradient flattens them (encoder θ, then predictor
# φ). A PseudoGradient.delta concatenates the encoder groups (sorted) then the predictor groups (sorted),
# so the flat global params must follow the SAME order to align element-wise (INV-AGG-DETERMINISM input).
_ENCODER_GROUP = "encoder"
_PREDICTOR_GROUP = "predictor"

# The #22/#04 probe placeholder: a fixed 32-byte hash used when no probe is pinned (cfg.data.probe_path is
# None). A participant pinning a real probe refuses a round whose probe_hash differs (INV-PROBE-PIN); the
# real probe resolution is the #22/#04 boundary.
_PROBE_PLACEHOLDER = b"\x00" * 32


class _ParamSlot(NamedTuple):
    """One entry of the flat↔grouped param manifest: a contiguous span of the flat θ⊕φ vector.

    ``group`` is ``"encoder"``/``"predictor"``; ``name`` is the bare state_dict key within that group;
    ``[start, start+numel)`` is its slice of the flat vector; ``shape`` un-flattens that slice back to the
    tensor's original shape. The manifest is built once and reused every round (the shapes are fixed).
    """

    group: str
    name: str
    start: int
    numel: int
    shape: torch.Size


class Coordinator:
    """Orchestrates the DiLoCo outer rounds, holds the canonical global model, runs the outer optimizer.

    Untrusted w.r.t. raw data; a single sequential round loop (RFC-0013 §1/§6). Constructed with the
    signature fixed by conventions §5 / RFC-0013 §1: ``Coordinator(config, *, transport)``.
    """

    def __init__(self, config: "LensembleConfig", *, transport: "Transport") -> None:
        self.config = config
        self.transport = transport

        cfg = config
        # Build the initial global model on CPU (the tiny-config / warm-start path; #43's participants
        # rebuild the same architecture from cfg and load the fetched weights).
        torch.manual_seed(cfg.determinism.root_seed)
        encoder = build_encoder(cfg)
        predictor = build_predictor(cfg)
        theta_weights = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
        phi_weights = {k: v.detach().clone() for k, v in predictor.state_dict().items()}

        # The canonical flat global params θ⊕φ AND the manifest to un-flatten the post-step vector back to
        # an encoder.*/predictor.* state_dict. Both follow build_pseudogradient's canonical order exactly
        # (encoder group sorted, then predictor group sorted), so the flat params align element-wise with
        # each PseudoGradient.delta — the precondition for the deterministic reduction (INV-AGG-DETERMINISM).
        self._param_manifest: tuple[_ParamSlot, ...]
        self._global_params: Tensor
        self._param_manifest, self._global_params = _flatten_groups(
            theta_weights, phi_weights
        )

        # The PERSISTENT outer optimizer (carries the Nesterov velocity across rounds, RFC-0003 §7). The
        # AGGREGATING determinism self-check reconstructs a FRESH optimizer per call so its velocity is not
        # advanced twice (assert_outer_step_deterministic invokes the thunk twice).
        self._optimizer = OuterOptimizer(
            lr=cfg.federation.outer_lr,
            momentum=cfg.federation.outer_nesterov_momentum,
        )

        # The append-only contribution ledger + the artifacts dir (per-coordinator temp dir; the real run
        # resolves these from cfg). Round artifacts are committed under artifacts_dir/round-XXXX.
        self._artifacts_dir = Path(tempfile.mkdtemp(prefix="lensemble-coordinator-"))
        self._ledger = ContributionLedger(self._artifacts_dir / "ledger.jsonl", [])
        self._config_hash = config_hash(asdict(cfg))
        self._probe_hash = self._resolve_probe_hash(cfg)

        # The frame-drift report measured at ALIGNING each round (None until the first measured round). The
        # Layer-3 Procrustes backstop (#18) is applied alongside it when the #18/#22 hooks are wired; this is
        # the diagnostic record, not the fold-in (the fold-in lands in _align_updates).
        self._last_drift: FrameDriftReport | None = None

        # The present count of the most recent COLLECTING (for run()'s below-K FaultToleranceExceeded; the
        # quorum is recomputed from cfg). Set each round at COLLECTING.
        self._last_contributing: int = 0

        # Commit the round-0 artifact to mint the initial global hash, then build + broadcast GlobalState_0.
        initial_hash = self._commit_checkpoint(
            theta_weights, phi_weights, round_index=0, parent_hash=None
        )
        self._driver = RoundDriver(global_hash=initial_hash, round_index=0)
        self._theta_weights = theta_weights
        self._phi_weights = phi_weights
        self._global_state = self._open_round(round_index=0, global_hash=initial_hash)

    # --- public surface (conventions §5 / RFC-0013 §1) ---

    def run(self, num_rounds: int) -> None:
        """Drive ``num_rounds`` outer rounds through the ``RoundState`` machine (RFC-0013 §1/§2).

        Each round is ``OPEN → COLLECTING → AGGREGATING → ALIGNING → COMMITTING → CLOSED`` or
        short-circuits to ``ABORTED``. On ``CLOSED`` the committed global hash advances and a
        :class:`~lensemble.provenance.ledger.ContributionRecord` is appended.

        RETRY SEMANTICS (the choice). ``run`` is the FAIL-FAST driver and ``try_round`` is the
        EXPLICIT-RETRY driver; both share the SAME single round body (:meth:`_run_one_round`), so the
        sequential-loop invariant (round ``t+1`` does not open until round ``t`` is ``CLOSED``, §6) is
        coherent across both. ``run`` SURFACES a below-quorum round as a raised
        :class:`~lensemble.errors.FaultToleranceExceeded` (code ``FAULT_TOLERANCE_EXCEEDED``, carrying
        ``contributing``/``quorum``); ``try_round`` does NOT raise on below-K, returning
        ``RoundState.ABORTED`` instead so staging-and-re-attempting the SAME round ``t`` is the supported
        elastic path (the round index/hash are unchanged on abort). A non-reproducible reduction always
        raises :class:`~lensemble.errors.NonDeterministicAggregation` (security-critical, never swallowed) —
        an abort drives the round to ``ABORTED`` with the global hash unchanged (no partial commit).

        END STATE. ``run`` opens the next round only BETWEEN rounds, so after the last requested round the
        driver rests in ``CLOSED`` (the next round is not opened speculatively); ``try_round`` opens round
        ``t+1`` immediately after a commit so the next ``try_round`` attempts ``t+1`` (its end state after a
        commit is ``OPEN`` on the new round).
        """
        for i in range(num_rounds):
            t = self._driver.round_index
            state = self._run_one_round(t)
            if state is RoundState.ABORTED:
                # run() is fail-fast: surface the below-quorum round as a raised FaultToleranceExceeded
                # (the round is ABORTED, the global hash + round index unchanged). A non-reproducible
                # reduction already propagated out of _run_one_round (never swallowed), so an ABORTED here
                # is the below-K case.
                raise self._build_quorum_error(t, self._last_contributing)
            # Open the next round only BETWEEN rounds (RFC-0013 §6: the loop is sequential; round t+1 does
            # not open until round t is CLOSED). The last requested round is left CLOSED (not re-opened).
            if i < num_rounds - 1:
                self._driver.open_next()
                self._global_state = self._open_round(
                    round_index=self._driver.round_index,
                    global_hash=self._driver.global_hash,
                )

    def try_round(self) -> RoundState:
        """Attempt the CURRENT round once; return the resulting :class:`RoundState` (RFC-0013 §1/§3).

        The explicit-retry entry point. Drives the current round ``t`` through
        ``COLLECTING → AGGREGATING → ALIGNING → COMMITTING → CLOSED`` over the PRESENT contributing set and,
        on success, advances the canonical hash and opens round ``t+1`` (so the next ``try_round`` attempts
        ``t+1``). If the present count is below the quorum
        ``K = max(fault_tolerance_min_participants, secure_agg_threshold)`` the round goes to ``ABORTED``
        and this returns ``RoundState.ABORTED`` WITHOUT raising and WITHOUT advancing the round index or the
        global hash — the caller may stage more updates and re-attempt the SAME round ``t``. (Contrast
        :meth:`run`, which surfaces the below-K case as a raised
        :class:`~lensemble.errors.FaultToleranceExceeded`.) A non-reproducible reduction still raises
        :class:`~lensemble.errors.NonDeterministicAggregation` (security-critical, never swallowed).

        Elastic completion is over the present ``C_t`` (the absent participants are simply not in the
        ``ContributionRecord``); the in-process present set models the ``collect_timeout_s`` drop (the
        wall-clock timeout is the #45 seam) and a delta for a PAST round is never back-applied.
        """
        t = self._driver.round_index
        state = self._run_one_round(t)
        # Open the next round only when this one CLOSED (RFC-0013 §6: the loop is sequential; round t+1 does
        # not open until round t reaches CLOSED). On ABORTED the round index/hash stay put for a re-attempt.
        if state is RoundState.CLOSED:
            self._driver.open_next()
            self._global_state = self._open_round(
                round_index=self._driver.round_index,
                global_hash=self._driver.global_hash,
            )
        return state

    def round_state(self) -> RoundState:
        """The current :class:`~lensemble.federation.round.RoundState` (observability / test hook)."""
        return self._driver.state

    def global_state(self) -> GlobalState:
        """The current canonical broadcast :class:`~lensemble.federation.state.GlobalState`."""
        return self._global_state

    # --- introspection hooks (test / observability; not part of the RFC-0013 §1 minimal surface) ---

    def global_state_hash(self) -> str:
        """The canonical committed global-model content hash (advances only on a successful commit)."""
        return self._driver.global_hash

    def global_params(self) -> "Tensor":
        """The current canonical flat global params ``θ_t⊕φ_t`` (encoder θ then predictor φ, canonical)."""
        return self._global_params

    def ledger_records(self) -> tuple[ContributionRecord, ...]:
        """The contribution-ledger records appended so far (RFC-0014 §7)."""
        return self._ledger.records

    def frame_drift_report(self) -> FrameDriftReport | None:
        """The last frame-drift report measured at ``ALIGNING`` (``None`` before the first measured round)."""
        return self._last_drift

    # --- the round loop (RFC-0013 §1/§2) ---

    def _run_one_round(self, t: int) -> RoundState:
        """Drive one round ``t`` through ``COLLECTING → AGGREGATING → ALIGNING → COMMITTING → CLOSED``.

        Returns the resulting :class:`RoundState` (``CLOSED`` on a successful commit, ``ABORTED`` below
        quorum). A below-K round does NOT raise here (the global hash + round index stay put so the SAME
        round may be re-attempted); :meth:`run` translates a returned ``ABORTED`` into a raised
        :class:`~lensemble.errors.FaultToleranceExceeded`. A non-reproducible reduction is still raised
        (security-critical, never swallowed).
        """
        # A prior below-K attempt at THIS round left the driver ABORTED (round index/hash unchanged); a
        # re-attempt re-opens the SAME round t. Reset OPEN so COLLECTING is a legal transition again (the
        # round index is not advanced — open_next, the only incrementer, is not called on an abort).
        if self._driver.state is RoundState.ABORTED:
            self._driver.state = RoundState.OPEN

        # 1. COLLECTING — fix the present set; abort below the quorum K (the global hash + round index stay
        # unchanged so the SAME round t can be re-attempted once enough updates are staged). K is the HIGHER
        # of the fault-tolerance floor and the secure-aggregation reveal threshold t_agg (RFC-0013 §3):
        # below t_agg the masking sum cannot be unblinded, so the higher of the two gates the round.
        self._driver.to(RoundState.COLLECTING)
        updates = dict(self.transport.collect_updates(t))
        self._last_contributing = len(updates)
        quorum = self._quorum()
        if len(updates) < quorum:
            # Below K: drive the round to ABORTED WITHOUT raising and return the state. The global hash +
            # round index are untouched (RoundDriver only advances the hash on commit), so the SAME round t
            # can be re-attempted. run() turns this returned ABORTED into a raised FaultToleranceExceeded;
            # try_round() returns it for the explicit-retry path. (We set ABORTED directly rather than
            # call driver.abort(), which always raises — the non-raising abort is intentional here.)
            self._driver.state = RoundState.ABORTED
            return RoundState.ABORTED

        # The per-participant probe embeddings + the reference frame E_ref for ALIGNING (both None by
        # default — the measured pass-through). The Layer-3 Procrustes backstop (#18) fires only when BOTH
        # are wired (the #18/#22 seam); the backstop runs in ALIGNING but the deltas it produces must be the
        # ones the AGGREGATING self-check and COMMITTING step both see, so embeddings are resolved here.
        embeddings = self._probe_embeddings(t)
        e_ref = self._reference_embeddings(t)

        # 2. AGGREGATING — the determinism self-check (INV-AGG-DETERMINISM). assert_outer_step_deterministic
        # re-runs the PURE thunk twice (a FRESH optimizer each call, so its velocity is not advanced) and
        # compares the two reductions bitwise; a mismatch raises NonDeterministicAggregation and aborts. The
        # thunk reduces the SAME (possibly backstop-aligned) deltas COMMITTING commits — the backstop is a
        # pure linear pre-step, so the self-check verifies exactly the committed reduction.
        self._driver.to(RoundState.AGGREGATING)
        prior_params = self._global_params
        lr = self.config.federation.outer_lr
        momentum = self.config.federation.outer_nesterov_momentum
        try:
            assert_outer_step_deterministic(
                lambda: OuterOptimizer(lr=lr, momentum=momentum).step(
                    prior_params, self._align_updates(updates, embeddings, e_ref)
                ),
                round_index=t,
            )
        except NonDeterministicAggregation as exc:
            # Security-critical: the reduction was not bitwise-reproducible. Drive the round to ABORTED
            # (the global hash is left unchanged — no partial commit) and re-raise; NEVER swallowed.
            self._driver.abort(exc)  # → ABORTED, re-raises
            raise  # defensive: abort always raises (this line is unreachable, keeps types total)

        # 3. ALIGNING — measure frame drift on the probe AND apply the Layer-3 Procrustes backstop (#18) when
        # per-participant embeddings + a reference E_ref are wired (the #18/#22 seam). The backstop conjugates
        # each over-threshold participant's PREDICTOR delta by Q_c* as a PURE LINEAR operation (RFC-0002 §5);
        # the encoder delta is left UNCHANGED (the activation-space decision — a LayerNorm-terminated encoder
        # has no terminal linear to fold Q into). With nothing wired this is the measured pass-through it was:
        # the aligned updates are byte-identical to `updates`, so every existing coordinator test stays green.
        self._driver.to(RoundState.ALIGNING)
        self._measure_drift(t, embeddings)
        aligned_updates = self._align_updates(updates, embeddings, e_ref)

        # 4. COMMITTING — the PERSISTENT outer step folds the averaged (aligned) delta into the global params
        # (only θ/φ); un-flatten via the manifest; hash-commit; append the ContributionRecord; advance hash.
        self._driver.to(RoundState.COMMITTING)
        new_params = self._optimizer.step(prior_params, aligned_updates)
        theta_weights, phi_weights = _unflatten_groups(self._param_manifest, new_params)
        new_hash = self._commit_checkpoint(
            theta_weights,
            phi_weights,
            round_index=t + 1,
            parent_hash=self._driver.global_hash,
        )
        self._append_contribution(t, updates, new_hash)
        self._driver.commit(new_hash)  # → CLOSED, advances the canonical global hash

        # Update the canonical state for round t+1 (the broadcast for t+1 happens at the next OPEN).
        self._global_params = new_params
        self._theta_weights = theta_weights
        self._phi_weights = phi_weights
        return RoundState.CLOSED

    def _quorum(self) -> int:
        """The round quorum ``K = max(fault_tolerance_min_participants, secure_agg_threshold)`` (§3).

        The HIGHER of the fault-tolerance floor and the secure-aggregation reveal threshold ``t_agg``:
        below ``t_agg`` survivors the masking sum cannot be unblinded (RFC-0011), so even if the
        fault-tolerance floor is met the round still cannot complete — the higher threshold gates.
        """
        fed = self.config.federation
        return max(fed.fault_tolerance_min_participants, fed.secure_agg_threshold)

    def _build_quorum_error(self, t: int, contributing: int) -> FaultToleranceExceeded:
        """The below-quorum :class:`~lensemble.errors.FaultToleranceExceeded` carrying ``contributing``/``quorum``."""
        quorum = self._quorum()
        err = FaultToleranceExceeded(
            f"round {t} has {contributing} contributing participant(s), below the quorum K={quorum} "
            f"(= max(fault_tolerance_min_participants, secure_agg_threshold)); discarding the round (the "
            f"global hash and round index are unchanged)",
            code=LensembleErrorCode.FAULT_TOLERANCE_EXCEEDED,
            remediation="stage more updates for the same round and re-attempt, or lower the quorum knobs",
        )
        err.contributing = contributing  # type: ignore[attr-defined]
        err.quorum = quorum  # type: ignore[attr-defined]
        return err

    def _open_round(self, *, round_index: int, global_hash: str) -> GlobalState:
        """OPEN: pin (θ_t, φ_t), derive s_t, build + broadcast the round GlobalState (RFC-0013 §1/§5)."""
        sketch_seed = round_sketch_seed(self.config.determinism.root_seed, round_index)
        theta_hash = weights_content_hash(self._theta_weights)
        phi_hash = weights_content_hash(self._phi_weights)
        # ParamRef.content_hash is minted AS weights_content_hash(group_weights) — the exact canonical hash
        # InProcessTransport.fetch_params recomputes — so a participant fetching θ_t/φ_t round-trips and
        # hash-verifies (INV-CHECKPOINT-HASH). The locator carries the committed-artifact round.
        theta_ref = ParamRef(
            content_hash=theta_hash,
            locator=f"artifact://round-{round_index:05d}/encoder",
        )
        phi_ref = ParamRef(
            content_hash=phi_hash,
            locator=f"artifact://round-{round_index:05d}/predictor",
        )
        gs = GlobalState(
            theta_ref=theta_ref,
            phi_ref=phi_ref,
            round_index=round_index,
            sketch_seed=sketch_seed,
            probe_hash=self._probe_hash,
            wmcp_version=self.config.model.wmcp_version,
        )
        # Seed the transport fetch store so a participant can fetch θ_t/φ_t (commit publishes the committed
        # GlobalState AND stores each group under its ref's content hash, consistent with fetch_params).
        self.transport.broadcast_round_open(gs)
        _seed_fetch_store(self.transport, gs, self._theta_weights, self._phi_weights)
        return gs

    # --- helpers ---

    def _commit_checkpoint(
        self,
        theta_weights: dict[str, "Tensor"],
        phi_weights: dict[str, "Tensor"],
        *,
        round_index: int,
        parent_hash: str | None,
    ) -> str:
        """Hash-commit (θ, φ) to a per-round artifact dir; return its content hash (``INV-CHECKPOINT-HASH``).

        ``save_checkpoint`` rejects any non-{encoder,predictor} tensor before writing
        (``INV-ACTIONHEAD-LOCAL``), so the committed artifact carries ONLY the federated groups.
        """
        weights = {f"{_ENCODER_GROUP}.{k}": v for k, v in theta_weights.items()}
        weights.update({f"{_PREDICTOR_GROUP}.{k}": v for k, v in phi_weights.items()})
        # Import locally so the module import graph stays light and the checkpoint dep points inward.
        from lensemble.artifacts.checkpoint import (
            model_arch_from_config,
            save_checkpoint,
        )

        # The committed checkpoint is self-describing (#171): record the encoder architecture so
        # recompute_alignment (#62) can reconstruct f_theta. Header metadata only — never hashed.
        return save_checkpoint(
            self._artifacts_dir / f"round-{round_index:05d}",
            weights,
            wmcp_version=self.config.model.wmcp_version,
            round_index=round_index,
            config_hash=self._config_hash,
            parent_hash=parent_hash,
            model_arch=model_arch_from_config(self.config),
        )

    def _append_contribution(
        self, t: int, updates: dict[str, "PseudoGradient"], new_hash: str
    ) -> None:
        """Append the round's :class:`ContributionRecord` (participants sorted; their dataset roots).

        ``prev_record_hash`` is left unset: :meth:`ContributionLedger.append` chains the record to the
        ledger tail's content hash internally (the hash-chain link, RFC-0014 §7), so the first record
        chains to ``None`` and each subsequent one to its predecessor.
        """
        participants = tuple(sorted(updates))
        dataset_roots = {pid: updates[pid].dataset_root.hex() for pid in participants}
        record = ContributionRecord(
            round_index=t,
            participants=participants,
            dataset_roots=dataset_roots,
            global_model_hash=new_hash,
        )
        self._ledger.append(record)

    def _measure_drift(self, t: int, embeddings: "dict[str, Tensor] | None") -> None:
        """ALIGNING: measure the frame-drift report on the probe IF per-participant embeddings are available.

        The frame-drift diagnostic (the headline figure, RFC-0002 §9) is measured here. The Procrustes
        backstop fold-in itself is applied by :meth:`_align_updates`; this only records the report. With no
        per-participant embeddings wired (#18/#22 boundary) there is nothing to measure and the report stays
        unset (the measured pass-through).
        """
        if embeddings is None or len(embeddings) < 2:
            return
        self._last_drift = frame_drift(
            embeddings,
            round_index=t,
            probe=self._probe(),
            expected_probe_hash=self._probe_hash.hex(),
        )

    def _align_updates(
        self,
        updates: "dict[str, PseudoGradient]",
        embeddings: "dict[str, Tensor] | None",
        e_ref: "Tensor | None",
    ) -> "dict[str, PseudoGradient]":
        """The Layer-3 Procrustes backstop (#18): realign the over-threshold deltas before the outer step.

        When per-participant ``embeddings`` AND a reference ``e_ref`` are wired (the #18/#22 seam),
        un-flattens each contributing ``PseudoGradient.delta`` into its grouped ``encoder.*``/``predictor.*``
        form via the param manifest, runs :func:`~lensemble.gauge.backstop.procrustes_backstop` (threshold =
        ``cfg.gauge.frame_drift_threshold_deg``, floor = ``cfg.gauge.procrustes_singular_floor``), and
        re-flattens the aligned grouped deltas back into ``PseudoGradient`` s (the canonical
        ``build_pseudogradient`` order, element-wise aligned with the global params). A participant without a
        wired embedding is passed through unchanged.

        With nothing wired (the default — ``embeddings``/``e_ref`` are ``None``) this is the IDENTITY: the
        returned mapping IS ``updates`` (the same objects), so ALIGNING stays the byte-identical pass-through
        and every existing coordinator test commits the same hash. The backstop is a pure linear operation on
        the released deltas, so the result feeds the SAME ``OuterOptimizer.step`` and a re-run with identical
        inputs commits the identical hash (``INV-AGG-DETERMINISM``).

        With the masking secure-aggregation backend the coordinator sees ONLY the masked sum (no
        per-participant deltas), so the backstop is a Stage-B / simulated-backend operation; ``embeddings`` /
        ``e_ref`` are the seam that makes it observable.
        """
        if embeddings is None or e_ref is None:
            return updates  # identity pass-through (the default; no backstop wired)

        # Un-flatten each contributing delta into its grouped encoder.*/predictor.* form (the backstop input).
        grouped: dict[str, dict[str, Tensor]] = {}
        backstop_ids: list[str] = []
        for pid in updates:
            if pid not in embeddings:
                continue  # a participant without a wired embedding is passed through unchanged
            theta, phi = _unflatten_groups(self._param_manifest, updates[pid].delta)
            grouped[pid] = {f"{_ENCODER_GROUP}.{k}": v for k, v in theta.items()}
            grouped[pid].update({f"{_PREDICTOR_GROUP}.{k}": v for k, v in phi.items()})
            backstop_ids.append(pid)
        if not grouped:
            return (
                updates  # no participant had a wired embedding — pass through unchanged
            )

        aligned_grouped = procrustes_backstop(
            grouped,
            {pid: embeddings[pid] for pid in grouped},
            e_ref,
            threshold_deg=self.config.gauge.frame_drift_threshold_deg,
            singular_floor=self.config.gauge.procrustes_singular_floor,
        )

        # Re-flatten the aligned grouped deltas into PseudoGradients in the SAME canonical order, preserving
        # each PseudoGradient's binding metadata (dataset_root / round_index / clipped). Participants without a
        # wired embedding keep their original PseudoGradient.
        aligned: dict[str, PseudoGradient] = dict(updates)
        for pid in backstop_ids:
            original = updates[pid]
            aligned[pid] = build_pseudogradient(
                aligned_grouped[pid],
                dataset_root=original.dataset_root,
                round_index=original.round_index,
                clipped=original.clipped,
            )
        return aligned

    def _probe_embeddings(
        self,
        t: int,  # noqa: ARG002 — t is the #18 boundary hook signature (unused here)
    ) -> "dict[str, Tensor] | None":
        """Per-participant probe embeddings for ALIGNING (the #18/#22 boundary; ``None`` here).

        The Layer-3 Procrustes fold-in (#18) consumes these; with no embeddings wired here the ALIGNING
        state is a measured pass-through. Subclasses / #18 override this to supply ``f_c(P)`` per
        participant (and the reserved ``"global"`` key for the aggregated model).
        """
        return None

    def _reference_embeddings(
        self,
        t: int,  # noqa: ARG002 — t is the #18 boundary hook signature (unused here)
    ) -> "Tensor | None":
        """The reference frame ``E_ref`` ``(n, d)`` the Layer-3 backstop aligns each participant to (#18).

        Returns ``None`` by default, so the backstop is un-wired and ALIGNING is a byte-identical
        pass-through. The Layer-3 Procrustes backstop fires only when BOTH this and :meth:`_probe_embeddings`
        return non-``None`` (the #18/#22 seam): each over-threshold participant's predictor delta is
        conjugated by ``Q_c* = procrustes_align(f_c(P), E_ref)`` before the outer step (RFC-0002 §5).
        Subclasses / #18 override this to supply the reference frame (for example the round-0 ``E_ref``).
        """
        return None

    def _probe(self) -> object | None:
        """The pinned public probe for drift measurement (the #22/#04 boundary; ``None`` here)."""
        return None

    def _resolve_probe_hash(self, cfg: "LensembleConfig") -> bytes:
        """The 32-byte ``probe_hash`` for the broadcast ``GlobalState`` (``INV-PROBE-PIN``; #22/#04).

        When ``cfg.data.probe_path`` is set the pinned probe is loaded and its content hash used; otherwise
        a fixed 32-byte placeholder is used (a participant pinning a real probe would refuse such a round —
        the real probe resolution lands with #22/#04).
        """
        probe_path = getattr(cfg.data, "probe_path", None)
        if probe_path is None:
            return _PROBE_PLACEHOLDER
        return load_probe(Path(probe_path)).content_hash


def _flatten_groups(
    theta_weights: dict[str, "Tensor"],
    phi_weights: dict[str, "Tensor"],
) -> tuple[tuple[_ParamSlot, ...], "Tensor"]:
    """Flatten θ⊕φ into the canonical order build_pseudogradient uses, returning (manifest, flat params).

    The order is: every ``encoder.<name>`` sorted by full key, then every ``predictor.<name>`` sorted by
    full key (``build_pseudogradient`` keys the groups ``encoder.*``/``predictor.*`` and sorts by
    ``(group_index, full_key)``). Building the manifest from the SAME ordering guarantees the flat global
    params align element-wise with each ``PseudoGradient.delta`` — the precondition for the deterministic
    reduction and the correct un-flatten of the post-step vector (``INV-AGG-DETERMINISM``).
    """
    slots: list[_ParamSlot] = []
    chunks: list[Tensor] = []
    start = 0
    for group, weights in (
        (_ENCODER_GROUP, theta_weights),
        (_PREDICTOR_GROUP, phi_weights),
    ):
        for name in sorted(weights):  # sorted within the group (the canonical order)
            tensor = weights[name]
            numel = tensor.numel()
            slots.append(_ParamSlot(group, name, start, numel, tensor.shape))
            chunks.append(tensor.detach().reshape(-1).to(torch.float32))
            start += numel
    flat = torch.cat(chunks) if chunks else torch.zeros(0, dtype=torch.float32)
    return tuple(slots), flat


def _unflatten_groups(
    manifest: tuple[_ParamSlot, ...], flat: "Tensor"
) -> tuple[dict[str, "Tensor"], dict[str, "Tensor"]]:
    """Un-flatten the post-step flat θ⊕φ vector back into encoder / predictor state_dicts via the manifest.

    Inverse of :func:`_flatten_groups`: each slot's contiguous span is reshaped to its stored shape and
    routed to its group. The bare state_dict keys are restored (no ``encoder.``/``predictor.`` prefix —
    those are re-applied at the checkpoint boundary), so the result loads back into a fresh encoder /
    predictor with ``strict=True``.
    """
    theta: dict[str, Tensor] = {}
    phi: dict[str, Tensor] = {}
    for slot in manifest:
        span = flat[slot.start : slot.start + slot.numel].reshape(slot.shape)
        if slot.group == _ENCODER_GROUP:
            theta[slot.name] = span
        else:
            phi[slot.name] = span
    return theta, phi


def _seed_fetch_store(
    transport: "Transport",
    gs: GlobalState,
    theta_weights: dict[str, "Tensor"],
    phi_weights: dict[str, "Tensor"],
) -> None:
    """Store θ/φ in the transport's fetch store under their refs' hashes so ``fetch_params`` round-trips.

    Uses ``InProcessTransport.commit`` (the seam keyed by ``theta_ref.content_hash`` /
    ``phi_ref.content_hash``) when available — consistent with how ``fetch_params`` recomputes
    ``weights_content_hash`` — so a participant fetching θ_t/φ_t resolves and hash-verifies
    (``INV-CHECKPOINT-HASH``). A network transport (#45) resolves refs from its own artifact store; this
    seeding is the single-process path and is a no-op when the transport has no ``commit`` seam.
    """
    commit = getattr(transport, "commit", None)
    if commit is not None:
        commit(gs, theta_weights=theta_weights, phi_weights=phi_weights)
