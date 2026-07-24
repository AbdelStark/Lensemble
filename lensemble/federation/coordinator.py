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
  :func:`~lensemble.aggregation.determinism.assert_outer_step_deterministic`. Its pure thunk previews the
  persistent optimizer twice from the current Nesterov velocity, so round ``t >= 1`` verifies the exact
  stateful computation that ``COMMITTING`` will apply without advancing momentum during verification. A
  mismatch raises :class:`~lensemble.errors.NonDeterministicAggregation` (security-critical, never
  swallowed) and the round → ``ABORTED``. Arrival order does not matter — the reduction is over the total
  order on ``participant_id``.
- **ALIGNING** — frame drift is measured when per-participant embeddings are available. The optional
  coordinator-side Layer-3 backstop is an explicitly raw-plaintext research harness: with a pinned probe,
  privacy and quantization disabled, and the simulated backend, it reconstructs each local full weight,
  applies ``Q_c*``, and re-differences exactly once before the determinism preview. It fails closed on
  release-transformed updates or a sum-only backend (``INV-ALIGN-BEFORE-RELEASE``). The target secure path
  performs participant-specific alignment before clip/noise/quantization/masking; after reveal this state
  can run only aggregate/committed-model diagnostics.
- **COMMITTING** — the PERSISTENT :class:`~lensemble.federation.outer_optimizer.OuterOptimizer.step` folds the
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
:class:`~lensemble.federation.outer_optimizer.OuterOptimizer.average_deltas` divides by ``len(deltas)``), so the
outer step is reproducible within the live run from the carried optimizer velocity and recorded inputs.

#22/#04 BOUNDARY (probe pin). When ``cfg.data.probe_path`` is set the pinned probe is loaded and hashed
into ``GlobalState.probe_hash``; otherwise a fixed 32-byte placeholder is used (a participant pinning a
real probe would refuse such a round, ``INV-PROBE-PIN``). The real probe resolution lands with #22/#04.
"""

from __future__ import annotations

import shutil
import tempfile
import weakref
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import torch

from lensemble.aggregation.determinism import assert_outer_step_deterministic
from lensemble.config.manifest import config_hash
from lensemble.config.seed import round_sketch_seed
from lensemble.data.probe import load_probe
from lensemble.errors import (
    ConfigError,
    FaultToleranceExceeded,
    LensembleError,
    LensembleErrorCode,
    NonDeterministicAggregation,
    RoundError,
)
from lensemble.federation.outer_optimizer import (
    OuterOptimizer,
    assert_bitwise_reproducible,
)
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

    def __init__(
        self,
        config: "LensembleConfig",
        *,
        transport: "Transport",
        artifacts_dir: "Path | None" = None,
        enable_backstop: bool = False,
        warm_start: "dict[str, Tensor] | None" = None,
    ) -> None:
        self.config = config
        self.transport = transport
        # #262: the coordinator-side Layer-3 Procrustes backstop is an explicitly PLAINTEXT research
        # harness. It needs each raw, individually keyed local-minus-global delta so it can reconstruct
        # full local weights, align them, and re-difference before the outer step. A real participant-side
        # privacy/quantization/masking boundary must instead apply alignment BEFORE those release
        # transforms (INV-ALIGN-BEFORE-RELEASE); a sum-only coordinator cannot recover this information.
        # Default OFF — the base coordinator stays the byte-identical measured pass-through.
        self._enable_backstop = bool(enable_backstop)
        self._validate_coordinator_backstop_mode(config)
        self._round_updates: dict[str, PseudoGradient] = {}

        cfg = config
        # Build the initial global model on CPU (the tiny-config / warm-start path; #43's participants
        # rebuild the same architecture from cfg and load the fetched weights).
        torch.manual_seed(cfg.determinism.root_seed)
        encoder = build_encoder(cfg)
        predictor = build_predictor(cfg)
        # WARM-START (#260 wiring; the MVP 2-phase Fork-A path): load a committed checkpoint's
        # encoder.*/predictor.* weights into the round-0 global θ_0/φ_0 BEFORE snapshotting, so every
        # participant fetches the warm-started global (INV-WARMSTART-T0 holds by construction — one
        # broadcast global). Combined with cfg.model.encoder_frozen (Fork A) this freezes the converged
        # gauge-aligned encoder and federates ONLY the predictor — giving the predictor a stationary,
        # shared latent target so its DiLoCo-averaged updates co-adapt coherently. load_state_dict copies
        # onto each param's device, so a CPU-loaded checkpoint warm-starts a CUDA-built model.
        if warm_start is not None:
            enc_sd = {
                k[len("encoder.") :]: v
                for k, v in warm_start.items()
                if k.startswith("encoder.")
            }
            phi_sd = {
                k[len("predictor.") :]: v
                for k, v in warm_start.items()
                if k.startswith("predictor.")
            }
            if enc_sd:
                encoder.load_state_dict(enc_sd, strict=True)
            if phi_sd:
                predictor.load_state_dict(phi_sd, strict=True)
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

        # The PERSISTENT outer optimizer carries Nesterov velocity across rounds (RFC-0003 §7).
        # AGGREGATING calls its pure preview_step twice, verifying the exact stateful computation without
        # advancing velocity; COMMITTING then calls step once to commit that same computation.
        self._optimizer = OuterOptimizer(
            lr=cfg.federation.outer_lr,
            momentum=cfg.federation.outer_nesterov_momentum,
        )

        # The append-only contribution ledger + the artifacts dir. A caller may pass an explicit
        # ``artifacts_dir`` — a persistent run-dir it OWNS, where the committed checkpoints live (the real
        # run). When it is None the coordinator creates a throwaway ``tempfile.mkdtemp`` it OWNS and cleans
        # up: a ``weakref.finalize`` removes it when the coordinator is GC'd, and :meth:`close` / the context
        # manager remove it eagerly — so a constructed-and-dropped coordinator leaks no temp dir (#178).
        # Round artifacts are committed under ``artifacts_dir/round-XXXX``.
        if artifacts_dir is None:
            self._artifacts_dir = Path(
                tempfile.mkdtemp(prefix="lensemble-coordinator-")
            )
            self._owns_artifacts_dir = True
            self._artifacts_finalizer: weakref.finalize | None = weakref.finalize(
                self, _rmtree_quiet, self._artifacts_dir
            )
        else:
            self._artifacts_dir = Path(artifacts_dir)
            self._artifacts_dir.mkdir(parents=True, exist_ok=True)
            self._owns_artifacts_dir = (
                False  # caller-owned; it persists with the committed checkpoints
            )
            self._artifacts_finalizer = None
        self._ledger = ContributionLedger(self._artifacts_dir / "ledger.jsonl", [])
        self._config_hash = config_hash(asdict(cfg))
        self._probe_hash = self._resolve_probe_hash(cfg)

        # #262: resolve the pinned probe + the round-0 reference frame E_ref = f_ref(P) ONCE, when the live
        # backstop is enabled and a real probe is pinned. f_ref is the round-0 encoder snapshot (the SAME
        # round-0 frame the participants' anchor targets derive from, INV-PROBE-PIN/INV-WARMSTART-T0), so
        # E_ref and each per-round f_c(P) live on one consistent reference. Both are measured the SAME way —
        # encoder(probe landmarks).tokens reshaped to (k*N, d) — so the Procrustes alignment is meaningful.
        self._backstop_probe: object | None = None
        self._backstop_e_ref: Tensor | None = None
        if self._enable_backstop:
            self._setup_backstop(cfg, encoder)

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

    # --- lifecycle: temp-artifacts cleanup (#178) ---

    def close(self) -> None:
        """Remove the coordinator's OWNED temp artifacts dir (the ``tempfile.mkdtemp`` it created).

        Idempotent and safe: a caller-provided ``artifacts_dir`` is never touched (it persists with its
        committed checkpoints). For an auto-created temp dir this removes it eagerly and cancels the
        GC finalizer (#178). Use the coordinator as a context manager (``with Coordinator(...) as c:``) to
        clean up automatically, or pass an explicit ``artifacts_dir`` for a persistent run.
        """
        if self._owns_artifacts_dir:
            _rmtree_quiet(self._artifacts_dir)
            if self._artifacts_finalizer is not None:
                self._artifacts_finalizer.detach()  # the dir is already gone; don't re-run on GC
                self._artifacts_finalizer = None

    def __enter__(self) -> "Coordinator":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

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
        try:
            collected = dict(self.transport.collect_updates(t))
        except Exception:
            # Network ingress validation/binding failures can surface as either typed Lensemble errors or
            # the wire validator's native exception. Preserve that exact error while terminating the
            # in-flight round; otherwise a retry is wedged in COLLECTING and its next transition is illegal.
            self._driver.state = RoundState.ABORTED
            raise
        try:
            updates = self._validate_updates(t, collected)
        except RoundError as exc:
            # A malformed/stale update is an in-flight round failure, not a Python broadcasting rule.
            # Abort before quorum/aggregation so the canonical hash and optimizer state stay untouched.
            self._driver.abort(exc)
            raise  # defensive: abort always raises
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
        # Stash the round's updates first so the #262 live backstop can reconstruct each participant's
        # encoder (θ_t + Δ_enc) to measure f_c(P) — embeddings are computed ONCE here and reused by both the
        # determinism self-check thunk and the real reduction, so INV-AGG-DETERMINISM holds.
        self._round_updates = updates
        embeddings = self._probe_embeddings(t)
        e_ref = self._reference_embeddings(t)

        # 2. AGGREGATING — materialize the plaintext research-harness alignment ONCE, then run the
        # determinism self-check (INV-AGG-DETERMINISM) and the eventual commit from that exact mapping.
        # A participant-side production path supplies already-aligned releases here, so this call is the
        # identity. Recomputing a participant-specific alignment inside the preview thunk would verify two
        # previews but not necessarily the third mapping consumed by COMMITTING.
        # assert_outer_step_deterministic calls the PERSISTENT optimizer's PURE preview twice: both use its
        # current carried velocity without advancing it, so round 1+ verifies the exact Nesterov computation
        # COMMITTING will apply.
        self._driver.to(RoundState.AGGREGATING)
        prior_params = self._global_params
        try:
            aligned_updates = self._align_updates(updates, embeddings, e_ref)
            assert_outer_step_deterministic(
                lambda: self._optimizer.preview_step(prior_params, aligned_updates),
                round_index=t,
            )
        except (ConfigError, NonDeterministicAggregation) as exc:
            # Invalid information-flow ordering or a non-reproducible reduction is fail-closed. Drive the
            # round to ABORTED (the global hash is left unchanged — no partial commit) and re-raise.
            self._driver.abort(exc)  # → ABORTED, re-raises
            raise  # defensive: abort always raises (this line is unreachable, keeps types total)

        # 3. ALIGNING — measure frame drift. In the coordinator-side plaintext research harness the aligned
        # mapping was materialized immediately above so the determinism preview and commit share one exact
        # input. On the target secure path, participant-specific alignment already happened before release;
        # this coordinator state is therefore diagnostic/verification only.
        self._driver.to(RoundState.ALIGNING)
        self._measure_drift(t, embeddings)

        # 4. COMMITTING — the PERSISTENT outer step folds the averaged (aligned) delta into the global params
        # (only θ/φ); un-flatten via the manifest; hash-commit; append the ContributionRecord; advance hash.
        self._driver.to(RoundState.COMMITTING)
        # Prepare the exact candidate without advancing Nesterov velocity. Checkpoint/ledger writes are
        # fallible; committing optimizer state before either succeeds makes a failed round alter the next
        # retry even though its canonical model hash stayed unchanged.
        new_params = self._optimizer.preview_step(prior_params, aligned_updates)
        theta_weights, phi_weights = _unflatten_groups(self._param_manifest, new_params)
        try:
            new_hash = self._commit_checkpoint(
                theta_weights,
                phi_weights,
                round_index=t + 1,
                parent_hash=self._driver.global_hash,
            )
            self._append_contribution(t, updates, new_hash)
        except LensembleError as exc:
            self._driver.abort(exc)
            raise  # defensive: abort always raises
        except Exception:
            # Preserve the original I/O/runtime exception while still making the retry state explicit.
            # No optimizer/global-model mutation has happened: only the pure preview ran.
            self._driver.state = RoundState.ABORTED
            raise

        # All fallible commit surfaces succeeded. Commit the previously previewed velocity now and assert
        # that the stateful step reproduces the artifact candidate before advancing the canonical hash.
        committed_params = self._optimizer.step(prior_params, aligned_updates)
        assert_bitwise_reproducible(new_params, committed_params)
        self._driver.commit(new_hash)  # → CLOSED, advances the canonical global hash

        # Update the canonical state for round t+1 (the broadcast for t+1 happens at the next OPEN).
        self._global_params = committed_params
        self._theta_weights = theta_weights
        self._phi_weights = phi_weights
        return RoundState.CLOSED

    def _validate_updates(
        self, t: int, updates: "dict[str, PseudoGradient]"
    ) -> "dict[str, PseudoGradient]":
        """Validate and snapshot exact flat parameter vectors before quorum/aggregation.

        A length-one delta otherwise broadcasts across every global parameter under PyTorch addition;
        a stale update can likewise be stored under a different transport round. Both are malformed
        round inputs, never valid optimizer semantics. Returned carriers own their storage and are
        normalized to the canonical global device so wire-decoded CPU updates work with a CUDA coordinator.
        """
        expected_numel = self._global_params.numel()
        expected_device = self._global_params.device
        validated: dict[str, PseudoGradient] = {}
        for participant_id, update in updates.items():
            if not isinstance(participant_id, str) or not participant_id:
                raise RoundError(
                    f"round {t} contains an update with an invalid participant id "
                    f"{participant_id!r}",
                    code=LensembleErrorCode.ROUND_FAILED,
                    remediation="key every update by one non-empty registered participant id",
                )
            if not isinstance(update, PseudoGradient):
                raise RoundError(
                    f"participant {participant_id!r} supplied {type(update).__name__}, "
                    "not a PseudoGradient",
                    code=LensembleErrorCode.ROUND_FAILED,
                    remediation="submit a validated released PseudoGradient carrier",
                )
            if update.round_index != t:
                raise RoundError(
                    f"participant {participant_id!r} update targets round "
                    f"{update.round_index}, but the coordinator is collecting round {t}",
                    code=LensembleErrorCode.ROUND_FAILED,
                    remediation="discard stale/future updates and recompute from the current RoundOpen",
                )
            if update.delta.ndim != 1 or update.delta.numel() != expected_numel:
                raise RoundError(
                    f"participant {participant_id!r} delta shape "
                    f"{tuple(update.delta.shape)} does not match the canonical flat parameter "
                    f"length {expected_numel}",
                    code=LensembleErrorCode.ROUND_FAILED,
                    remediation="flatten encoder then predictor deltas using the current parameter manifest",
                )
            # Reconstruct once against the original metadata to catch an in-place mutation of a nominally
            # frozen carrier (Tensor storage is mutable), then take a coordinator-owned device-local copy.
            try:
                PseudoGradient(
                    delta=update.delta,
                    l2_norm=update.l2_norm,
                    dataset_root=update.dataset_root,
                    round_index=update.round_index,
                    clipped=update.clipped,
                    quantized=update.quantized,
                )
            except ValueError as exc:
                raise RoundError(
                    f"participant {participant_id!r} update metadata is inconsistent: {exc}",
                    code=LensembleErrorCode.ROUND_FAILED,
                    remediation="rebuild the PseudoGradient from the released delta before submission",
                ) from exc
            delta = (
                update.delta.detach()
                .to(device=expected_device, dtype=torch.float32)
                .contiguous()
                .clone()
            )
            validated[participant_id] = PseudoGradient(
                delta=delta,
                l2_norm=float(delta.norm()),
                dataset_root=update.dataset_root,
                round_index=update.round_index,
                clipped=update.clipped,
                quantized=update.quantized,
            )
        return validated

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
        # The drift report is a DIAGNOSTIC, never a gate: a strong anchor can pin two participants onto a
        # near-identical frame whose inter-pair Procrustes M is rank-deficient. That is the GOOD anchored
        # case (drift → 0), so the diagnostic records it as 0° (degenerate_safe) rather than aborting.
        self._last_drift = frame_drift(
            embeddings,
            round_index=t,
            probe=self._probe(),
            expected_probe_hash=self._probe_hash.hex(),
            degenerate_safe=True,
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

        The coordinator passes its current grouped global weights so the backstop reconstructs each local
        gauge-bearing weight, aligns the full weight, and re-differences from that same global baseline.
        With nothing wired (the default — ``embeddings``/``e_ref`` are ``None``), this is the IDENTITY: the
        returned mapping IS ``updates``. The result feeds the SAME ``OuterOptimizer.step`` and a re-run with
        identical inputs commits the identical hash (``INV-AGG-DETERMINISM``).

        This is a raw-plaintext research harness. It cannot follow clipping/noise, quantization, or a
        sum-only secure-aggregation boundary. The target path applies the same full-weight transform and
        re-difference participant-side before all release transforms (``INV-ALIGN-BEFORE-RELEASE``).
        """
        if embeddings is None or e_ref is None:
            return updates  # identity pass-through (the default; no backstop wired)

        # Un-flatten each contributing delta into grouped encoder.*/predictor.* form.
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

        release_transformed = sorted(
            pid
            for pid in backstop_ids
            if updates[pid].clipped or updates[pid].quantized
        )
        if release_transformed:
            raise ConfigError(
                "coordinator-side Procrustes alignment received release-transformed "
                f"updates from {release_transformed}; participant-specific alignment "
                "cannot run after clipping/noise or quantization "
                "(INV-ALIGN-BEFORE-RELEASE)",
                code=LensembleErrorCode.CONFIG_INVALID,
                remediation=(
                    "apply the full-weight gauge transform and re-difference on each "
                    "participant before clipping, noise, quantization, and masking; "
                    "otherwise use a trusted/MPC boundary that can perform that order"
                ),
            )

        global_grouped = {
            f"{_ENCODER_GROUP}.{name}": tensor
            for name, tensor in self._theta_weights.items()
        }
        global_grouped.update(
            {
                f"{_PREDICTOR_GROUP}.{name}": tensor
                for name, tensor in self._phi_weights.items()
            }
        )
        aligned_grouped = procrustes_backstop(
            grouped,
            {pid: embeddings[pid] for pid in grouped},
            e_ref,
            global_weights=global_grouped,
            threshold_deg=self.config.gauge.frame_drift_threshold_deg,
            singular_floor=self.config.gauge.procrustes_singular_floor,
        )

        # Re-flatten the aligned grouped deltas into PseudoGradients in the SAME canonical order. Only raw
        # plaintext updates may reach this research harness, so release-transform metadata remains false.
        # Dataset binding and round binding are preserved. Participants without a wired embedding keep
        # their original PseudoGradient.
        aligned: dict[str, PseudoGradient] = dict(updates)
        for pid in backstop_ids:
            original = updates[pid]
            aligned[pid] = build_pseudogradient(
                aligned_grouped[pid],
                dataset_root=original.dataset_root,
                round_index=original.round_index,
                clipped=False,
            )
        return aligned

    def _validate_coordinator_backstop_mode(self, cfg: "LensembleConfig") -> None:
        """Fail closed when the plaintext backstop is combined with a release boundary.

        The guard is active only when a probe is pinned and the backstop can actually fire. Keeping
        ``enable_backstop=True`` inert when no probe exists preserves the diagnostic pass-through contract.
        """
        if not self._enable_backstop or getattr(cfg.data, "probe_path", None) is None:
            return

        incompatible: list[str] = []
        if cfg.privacy.enabled:
            incompatible.append("privacy.enabled=True")
        if cfg.federation.quantize_pseudo_gradient:
            incompatible.append("federation.quantize_pseudo_gradient=True")
        if cfg.federation.aggregation_backend != "simulated":
            incompatible.append(
                f"federation.aggregation_backend={cfg.federation.aggregation_backend!r}"
            )
        if not incompatible:
            return

        raise ConfigError(
            "coordinator-side Procrustes alignment requires raw, individually "
            "visible updates in the plaintext simulated harness; incompatible "
            f"settings: {', '.join(incompatible)} "
            "(INV-ALIGN-BEFORE-RELEASE)",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation=(
                "disable the coordinator backstop, or run the explicit plaintext "
                "research harness with privacy and quantization disabled; a secure "
                "deployment must align and re-difference participant-side before "
                "clipping, noise, quantization, and masking, or use a trusted/MPC boundary"
            ),
        )

    def _setup_backstop(self, cfg: "LensembleConfig", encoder: object) -> None:
        """#262: load the pinned probe + compute the round-0 reference frame ``E_ref = f_ref(P)``.

        Resolves the probe from ``cfg.data.probe_path`` (when set) and forwards the round-0 encoder snapshot
        ``f_ref`` on the probe landmarks to ``E_ref`` ``(k*N, d)`` — the shared frame each over-threshold
        participant is aligned onto. A run with the backstop enabled but no pinned probe simply leaves the
        backstop un-wired (the measured pass-through): there is no shared frame to align to.
        """
        from lensemble.data.probe import load_probe
        from lensemble.model.encoder import snapshot_reference
        from lensemble.model.numerics import module_input_tensor

        probe_path = getattr(cfg.data, "probe_path", None)
        if probe_path is None:
            return
        probe = load_probe(Path(probe_path))
        f_ref = snapshot_reference(encoder)  # type: ignore[arg-type]
        landmarks = module_input_tensor(f_ref, probe.points[probe.landmark_idx])
        with torch.no_grad():
            tokens = f_ref(landmarks).tokens.to(torch.float32)
        self._backstop_probe = probe
        self._backstop_e_ref = tokens.reshape(-1, tokens.shape[-1])

    def _probe_embeddings(
        self,
        t: int,  # noqa: ARG002 — t is the #18 boundary hook signature (unused here)
    ) -> "dict[str, Tensor] | None":
        """Per-participant probe embeddings ``f_c(P)`` for ALIGNING (the #18/#22/#262 boundary).

        With the plaintext backstop harness wired (#262), reconstructs each contributing participant's
        encoder from the current global ``θ_t`` plus its raw encoder delta (``θ_c = θ_t + Δ_enc``) and forwards it on the
        pinned probe landmarks to ``f_c(P)`` ``(k*N, d)`` — the trained frame the drift diagnostic measures
        and the Procrustes backstop aligns. Computed ONCE per round (reused by the determinism self-check and
        the committed reduction). Reconstruction uses the SAME canonical un-flatten as the outer step, so the
        measured frame is exactly the one whose delta is aggregated. With the backstop un-wired (the default)
        returns ``None`` and ALIGNING is the byte-identical measured pass-through.
        """
        from lensemble.model.encoder import build_encoder
        from lensemble.model.numerics import module_input_tensor

        probe = self._backstop_probe
        if not self._enable_backstop or probe is None or not self._round_updates:
            return None
        landmarks = probe.points[probe.landmark_idx]  # type: ignore[attr-defined]
        embeddings: dict[str, Tensor] = {}
        for pid, pg in self._round_updates.items():
            theta_delta, _phi = _unflatten_groups(self._param_manifest, pg.delta)
            recon = {
                name: self._theta_weights[name] + theta_delta[name]
                for name in self._theta_weights
            }
            enc = build_encoder(self.config)
            enc.load_state_dict(recon, strict=True)
            enc.eval()
            with torch.no_grad():
                tokens = enc(module_input_tensor(enc, landmarks)).tokens.to(
                    torch.float32
                )
            embeddings[pid] = tokens.reshape(-1, tokens.shape[-1])
        return embeddings

    def _reference_embeddings(
        self,
        t: int,  # noqa: ARG002 — t is the #18 boundary hook signature (unused here)
    ) -> "Tensor | None":
        """The reference frame ``E_ref`` ``(k*N, d)`` the Layer-3 backstop aligns each participant to (#262).

        Returns the round-0 ``E_ref = f_ref(P)`` computed at construction when the live backstop is wired,
        else ``None`` (the byte-identical pass-through). The Layer-3 Procrustes backstop fires only when BOTH
        this and :meth:`_probe_embeddings` return non-``None``: each over-threshold participant's encoder
        terminal frame + predictor I/O are reconstructed from global + delta, aligned by the row-space
        ``Q_c* = procrustes_align(f_c(P), E_ref)``, and re-differenced before the outer step (RFC-0002 §5).
        """
        if not self._enable_backstop:
            return None
        return self._backstop_e_ref

    def _probe(self) -> object | None:
        """The pinned public probe for the drift diagnostic (the live backstop's probe, or ``None``)."""
        return self._backstop_probe

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


def _rmtree_quiet(path: Path) -> None:
    """Remove ``path`` recursively, ignoring errors (the coordinator's owned temp dir; #178).

    Module-level (not a bound method) so :func:`weakref.finalize` can hold it without keeping the
    coordinator alive — the finalizer fires when the coordinator is GC'd, removing the throwaway
    ``tempfile.mkdtemp`` artifacts dir even if the caller never calls :meth:`Coordinator.close`.
    """
    shutil.rmtree(path, ignore_errors=True)


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
