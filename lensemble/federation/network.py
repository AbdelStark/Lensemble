"""lensemble.federation.network — the networked control-plane transport (RFC-0013 §5, Stage C).

This module is the Stage-C realization of the runtime's control plane. It is layered:

1. :class:`MessageChannel` — the LOW-LEVEL RFC-0013 §5 wire interface: ``send`` / ``recv`` (None on
   timeout) / ``broadcast`` / ``peers``. It carries the four
   :mod:`~lensemble.federation.messages` ``ControlMessage`` s and nothing else.
   :class:`LoopbackChannel` is the in-process, in-memory realization (per-peer FIFO inboxes) — the
   testable stand-in for the real socket transport (the gRPC-vs-HTTP wire choice is an Open Question
   bound at Stage C; the state machine is transport-agnostic, RFC-0013 §Open Questions).
2. :class:`NetworkedTransport` — implements the OPERATION-ORIENTED ``Transport`` Protocol
   (:class:`lensemble.federation.transport.Transport` — the SAME seam ``Coordinator`` #42 and
   ``Participant`` #43 consume) OVER a :class:`MessageChannel`. The runtime is unchanged: a
   ``NetworkedTransport`` drops into ``Coordinator(cfg, transport=...)`` exactly where an
   ``InProcessTransport`` would, and realizes each operation by exchanging the four §5 messages.

The bridge (how the op-oriented ``Transport`` maps onto the four messages):

==============================  ==========================================================
``Transport`` operation         realized over the channel as
==============================  ==========================================================
``broadcast_round_open(gs)``    emit a ``RoundOpen`` (refs/seed/probe hash) via ``broadcast``;
                                seed the local hash→weights store so ``fetch_params`` resolves
``submit_update(...)``          ``send`` an ``Update`` (via ``from_pseudogradient``) to the coord
``collect_updates(t)``          drain the coordinator inbox via ``recv`` until ``None``; ingress-
                                validate + bind-check every ``Update``/``Commitment``; rebuild the
                                ``PseudoGradient`` s for round ``t``
``recover_global_state(...)``   return the last broadcast ``GlobalState``
``fetch_params(ref)``           resolve ``ref`` from the local artifact store, HASH-VERIFIED
``register(...)``               record the participant endpoint (control-plane bookkeeping)
==============================  ==========================================================

Residency over the wire (``INV-RESIDENCY``). Every outbound participant payload is routed through
:func:`lensemble.federation.messages.from_pseudogradient`, which residency-guards the carrier so a raw
tensor/observation can never be placed in a message (fail-closed ``ResidencyViolation``). The four
messages carry only hashes, coordination scalars, and the released ``delta`` — the artifact bytes are
fetched OUT-OF-BAND by locator, never inlined in a message (03 §7).

Ingress validation (RFC-0013 §5/§7). ``collect_updates`` re-validates EVERY received message at ingress
(``parse_control_message`` — a malformed/too-new payload raises the typed error and the update is NOT
counted, so the round state does not advance) and checks each ``Update``'s ``Δ_c`` is bound to a valid
committed ``R_c`` via :func:`lensemble.provenance.commit.verify_binding`
(:class:`~lensemble.errors.CommitmentMismatch` on mismatch, ``INV-COMMIT-BINDING``, never swallowed).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from lensemble.errors import (
    CommitmentMismatch,
    LensembleErrorCode,
)
from lensemble.federation.messages import (
    Commitment,
    RoundClose,
    RoundOpen,
    Update,
    from_pseudogradient,
    parse_control_message,
    to_delta_tensor,
)
from lensemble.federation.pseudogradient import PseudoGradient
from lensemble.federation.state import GlobalState, ParamRef
from lensemble.federation.transport import weights_content_hash
from lensemble.provenance.commit import verify_binding
from lensemble.provenance.merkle import CommitmentScheme

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

# The control messages share one structural shape on the wire: a `model_dump()` -> dict that the ingress
# validator re-parses. Any object exposing that is acceptable to a channel (a real socket transport
# serializes the dict to bytes); LoopbackChannel passes the message object through directly.
ControlMessageLike = RoundOpen | Commitment | Update | RoundClose


@runtime_checkable
class MessageChannel(Protocol):
    """The low-level RFC-0013 §5 control-plane wire interface (``send``/``recv``/``broadcast``/``peers``).

    Carries the four :mod:`~lensemble.federation.messages` ``ControlMessage`` s between named peers and
    nothing else. ``recv`` returns ``None`` on timeout (the §5 contract — the coordinator never blocks
    indefinitely on a participant). A real implementation is a socket / gRPC / HTTP transport (the wire
    choice is the Stage-C Open Question); :class:`LoopbackChannel` is the in-process realization.
    """

    def send(self, peer_id: str, message: ControlMessageLike) -> None:
        """Enqueue ``message`` onto ``peer_id``'s inbox (point-to-point)."""
        ...

    def recv(self, *, timeout_s: float) -> ControlMessageLike | None:
        """Pop the next message from THIS node's inbox, or ``None`` if none arrives within ``timeout_s``."""
        ...

    def broadcast(self, message: ControlMessageLike) -> None:
        """Enqueue ``message`` onto every peer's inbox (not the sender's own)."""
        ...

    def peers(self) -> "Sequence[str]":
        """The peer ids this node can reach (excluding itself)."""
        ...


class LoopbackChannel:
    """An in-memory :class:`MessageChannel` for one node of a connected set (RFC-0013 §5, Stage B/test).

    Per-peer FIFO inboxes shared across a connected set of channels built together by
    :meth:`connected_pair` / :meth:`connected_mesh`. ``send``/``broadcast`` append to the recipient(s)'
    inbox; ``recv`` pops THIS node's inbox, returning ``None`` once it is empty (the simulated-budget
    timeout — there is no real wall clock, so an empty inbox is the §5 timeout). This is the testable
    stand-in for the Stage-C socket transport; the real network transport is the production seam that
    swaps in behind the same :class:`MessageChannel` Protocol.
    """

    def __init__(
        self, node_id: str, inboxes: dict[str, deque[ControlMessageLike]]
    ) -> None:
        # `inboxes` is SHARED across every channel in the connected set: node_id -> its FIFO inbox. A send
        # to peer X appends to inboxes[X]; a recv on this node pops inboxes[node_id]. The shared mapping is
        # what makes the set "connected" in one process.
        self._node_id = node_id
        self._inboxes = inboxes

    @property
    def node_id(self) -> str:
        """This channel's node id (the inbox it ``recv`` s from)."""
        return self._node_id

    def send(self, peer_id: str, message: ControlMessageLike) -> None:
        """Enqueue ``message`` onto ``peer_id``'s inbox (RFC-0013 §5). Unknown peer → ``KeyError``."""
        if peer_id not in self._inboxes:
            raise KeyError(
                f"unknown peer {peer_id!r}; not in the connected set {sorted(self._inboxes)}"
            )
        self._inboxes[peer_id].append(message)

    def recv(self, *, timeout_s: float) -> ControlMessageLike | None:  # noqa: ARG002 — budget seam
        """Pop the next message from THIS node's inbox, or ``None`` if empty (the §5 timeout contract).

        ``timeout_s`` is the simulated budget: with no real wall clock an empty inbox IS the timeout, so
        an empty inbox returns ``None`` regardless of the budget (the seam a real socket transport fills
        with a blocking poll). A non-empty inbox returns its head FIFO.
        """
        inbox = self._inboxes[self._node_id]
        if not inbox:
            return None
        return inbox.popleft()

    def broadcast(self, message: ControlMessageLike) -> None:
        """Enqueue ``message`` onto every peer's inbox EXCEPT this node's own (RFC-0013 §5)."""
        for peer_id, inbox in self._inboxes.items():
            if peer_id != self._node_id:
                inbox.append(message)

    def peers(self) -> "Sequence[str]":
        """The peer ids this node can reach (the connected set minus itself, RFC-0013 §5)."""
        return tuple(pid for pid in self._inboxes if pid != self._node_id)

    @classmethod
    def connected_pair(cls, a_id: str, b_id: str) -> dict[str, LoopbackChannel]:
        """Two channels sharing one inbox set, so ``a`` and ``b`` can exchange messages in one process."""
        return cls.connected_mesh(a_id, b_id)

    @classmethod
    def connected_mesh(cls, *node_ids: str) -> dict[str, LoopbackChannel]:
        """A fully connected set of channels over ``node_ids`` sharing one per-peer inbox mapping.

        Returns ``{node_id: LoopbackChannel}``. Every channel sees the SAME ``inboxes`` dict, so a send
        from any node reaches the recipient's inbox and a ``recv`` on that node drains it — the in-process
        model of a connected network (Stage B harness / Stage-C two-node setup driven in one test process).
        """
        if len(set(node_ids)) != len(node_ids):
            raise ValueError(f"node ids must be unique, got {node_ids}")
        inboxes: dict[str, deque[ControlMessageLike]] = {
            nid: deque() for nid in node_ids
        }
        return {nid: cls(nid, inboxes) for nid in node_ids}


class NetworkedTransport:
    """The networked :class:`~lensemble.federation.transport.Transport` over a :class:`MessageChannel`.

    Implements the OPERATION-ORIENTED ``Transport`` Protocol (``register`` / ``recover_global_state`` /
    ``fetch_params`` / ``submit_update`` / ``broadcast_round_open`` / ``collect_updates``) by exchanging
    the four RFC-0013 §5 ``ControlMessage`` s over ``channel``. It is interchangeable with
    ``InProcessTransport`` in ``Coordinator(cfg, transport=...)`` and ``Participant(...)``: the runtime is
    unchanged; only the transport (and thus the wire layer) differs. See the module docstring for the
    operation→message bridge and the ingress-validation / binding-check contract.

    The transport is the coordinator-side node by default (``coordinator_id`` is this node's id on the
    channel when it drives the coordinator), but the SAME object also serves the participant-side
    ``submit_update`` (it ``send`` s to the coordinator peer). One ``NetworkedTransport`` per node; the
    coordinator's instance owns the artifact store and the per-participant committed-root map.
    """

    def __init__(self, *, channel: MessageChannel, coordinator_id: str) -> None:
        self._channel = channel
        self._coordinator_id = coordinator_id
        # The control-plane bookkeeping the in-process transport keeps in memory; here it is local to this
        # node and refreshed from the wire (recover_global_state returns the last broadcast GlobalState).
        self._registered: dict[str, str] = {}
        self._committed: GlobalState | None = None
        # The hash→weights artifact store backing fetch_params (seeded at broadcast_round_open; a real run
        # resolves a ref from the networked artifact store by locator — INV-CHECKPOINT-HASH on fetch).
        self._weights: dict[str, dict[str, Tensor]] = {}
        # The per-(round, participant) committed dataset root R_c, set by an ingress Commitment (or the
        # participant-side commit_root helper). An Update's Δ_c is bound against this (INV-COMMIT-BINDING).
        self._committed_roots: dict[tuple[int, str], bytes] = {}
        # The commitment scheme R_c is checked under (Phase-1 sha256 / 32-byte; CommitmentScheme default).
        self._scheme = CommitmentScheme()

    # --- participant-facing operations (RFC-0013 §5; consumed by #43) ---

    def register(self, participant_id: str, endpoint: str) -> None:
        """Register a participant under ``participant_id`` at ``endpoint`` (control-plane bookkeeping)."""
        self._registered[participant_id] = endpoint

    def recover_global_state(self, *, participant_id: str) -> GlobalState:  # noqa: ARG002 — id is the seam
        """Return the last broadcast :class:`GlobalState` (the rejoiner-recovery path, RFC-0013 §3).

        Over the wire the recovered state is the most recent ``RoundOpen`` the coordinator broadcast,
        cached here when :meth:`broadcast_round_open` ran. No committed state yet (no round opened) is a
        :class:`~lensemble.errors.CheckpointIntegrityError` — the same fail-closed contract as
        ``InProcessTransport.recover_global_state``.
        """
        if self._committed is None:
            from lensemble.errors import CheckpointIntegrityError

            raise CheckpointIntegrityError(
                "no committed GlobalState to recover; the coordinator has not broadcast a round",
                code=LensembleErrorCode.CHECKPOINT_INTEGRITY,
                remediation="open round 0 (broadcast_round_open) before a participant joins",
            )
        return self._committed

    def fetch_params(self, ref: ParamRef) -> dict[str, "Tensor"]:
        """Resolve ``ref`` to its weights, HASH-VERIFIED against ``ref.content_hash`` (``INV-CHECKPOINT-HASH``).

        The artifact is fetched OUT-OF-BAND by locator (the four §5 messages carry only hashes, never the
        weight bytes — 03 §7); here the local store backs that fetch. The recomputed
        :func:`~lensemble.federation.transport.weights_content_hash` is re-verified against
        ``ref.content_hash`` so a tampered/missing artifact fails closed with
        :class:`~lensemble.errors.CheckpointIntegrityError` (no tensors returned), exactly as
        ``InProcessTransport.fetch_params`` does.
        """
        from lensemble.errors import CheckpointIntegrityError

        stored = self._weights.get(ref.content_hash)
        if stored is None:
            raise CheckpointIntegrityError(
                f"no weights stored under content hash {ref.content_hash} (locator {ref.locator!r})",
                code=LensembleErrorCode.CHECKPOINT_INTEGRITY,
                remediation="commit the artifact under its content hash before fetching it",
            )
        actual = weights_content_hash(stored)
        if actual != ref.content_hash:
            err = CheckpointIntegrityError(
                f"fetched-weights content hash mismatch: expected {ref.content_hash}, got {actual}",
                code=LensembleErrorCode.CHECKPOINT_INTEGRITY,
                remediation="the artifact is corrupt or tampered; do not load it (INV-CHECKPOINT-HASH)",
            )
            err.expected_hash = ref.content_hash  # type: ignore[attr-defined]
            err.got_hash = actual  # type: ignore[attr-defined]
            raise err
        return dict(stored)

    def submit_update(
        self,
        *,
        participant_id: str,
        round_index: int,
        update: PseudoGradient,
    ) -> None:
        """``send`` the participant's privatized, bound ``PseudoGradient`` as an ``Update`` (RFC-0013 §5).

        Routes the delta through :func:`~lensemble.federation.messages.from_pseudogradient`, which
        residency-guards the carrier (``INV-RESIDENCY``: a non-``PseudoGradient`` raw payload fails closed)
        and serializes the released ``Δ_c`` as JSON-native floats. The ``Update`` is sent point-to-point to
        the coordinator peer (``participant → aggregator``, §5).
        """
        message = from_pseudogradient(update, participant_id=participant_id)
        # `round_index` is carried by the Update (== update.round_index); kept in the signature for the
        # op-oriented Transport contract. Send to the coordinator (the aggregator endpoint).
        _ = round_index
        self._channel.send(self._coordinator_id, message)

    # --- coordinator-facing operations (RFC-0013 §5; consumed by #42) ---

    def broadcast_round_open(self, global_state: GlobalState) -> None:
        """Emit a ``RoundOpen`` over ``broadcast`` and cache the committed state (RFC-0013 §5).

        Builds the ``RoundOpen`` from ``global_state`` (refs as hash+locator, sketch seed, probe hash) and
        broadcasts it to every participant peer; caches ``global_state`` so ``recover_global_state`` returns
        it. The θ/φ weights are seeded into the fetch store by :meth:`commit` (the coordinator calls it via
        the ``_seed_fetch_store`` seam right after this), so ``fetch_params`` resolves θ_t/φ_t out-of-band.
        """
        self._committed = global_state
        message = RoundOpen(
            theta_ref_hash=global_state.theta_ref.content_hash,
            theta_ref_locator=global_state.theta_ref.locator,
            phi_ref_hash=global_state.phi_ref.content_hash,
            phi_ref_locator=global_state.phi_ref.locator,
            round_index=global_state.round_index,
            sketch_seed=global_state.sketch_seed,
            probe_hash=global_state.probe_hash.hex(),
            landmark_hashes=(),
            inner_horizon=0,
        )
        self._channel.broadcast(message)

    def collect_updates(self, round_index: int) -> "Mapping[str, PseudoGradient]":
        """Drain the coordinator inbox; ingress-validate + bind-check; return round-``t`` updates (RFC-0013 §5).

        Pops every queued message via ``recv`` until ``None`` (the §5 timeout). EACH message is
        re-validated at ingress (``parse_control_message`` — a malformed/too-new payload raises the typed
        error and the update is NOT counted, so the round state does not advance). A ``Commitment`` records
        the participant's committed ``R_c`` for this round. An ``Update`` for ``round_index`` is bound
        against that committed ``R_c`` via :func:`~lensemble.provenance.commit.verify_binding`
        (:class:`~lensemble.errors.CommitmentMismatch` on mismatch — ``INV-COMMIT-BINDING``, never
        swallowed), then reconstructed into a :class:`PseudoGradient` (which re-validates finiteness and the
        32-byte ``R_c`` length). Returns ``{participant_id: PseudoGradient}`` for the matching round.
        """
        collected: dict[str, PseudoGradient] = {}
        while True:
            raw_message = self._channel.recv(timeout_s=0.0)
            if raw_message is None:
                break  # §5 timeout: the inbox is drained, the present set is fixed
            # Ingress re-validation: a real socket transport receives bytes; here we re-parse the message's
            # dict so a hostile/malformed payload is rejected at the SAME ingress gate (never trusted raw).
            message = parse_control_message(raw_message.model_dump())
            if isinstance(message, Commitment):
                # A Commitment records the participant's committed R_c for the round (the binding target).
                self._committed_roots[(message.round_index, message.participant_id)] = (
                    bytes.fromhex(message.dataset_root)
                )
            elif isinstance(message, Update):
                if message.round_index != round_index:
                    continue  # an Update for a different round is not part of THIS round's present set
                pg = self._ingest_update(message)
                collected[message.participant_id] = pg
            # RoundOpen/RoundClose on the coordinator inbox are control echoes; ignored here.
        return collected

    # --- seeding seams (used by the coordinator's _seed_fetch_store; mirror InProcessTransport.commit) ---

    def commit(
        self,
        global_state: GlobalState,
        *,
        theta_weights: "Mapping[str, Tensor]",
        phi_weights: "Mapping[str, Tensor]",
    ) -> None:
        """Cache ``global_state`` and store θ/φ under their refs' hashes so ``fetch_params`` round-trips.

        The same seam ``InProcessTransport.commit`` exposes (the coordinator's ``_seed_fetch_store`` calls
        it via ``getattr(transport, "commit", None)`` right after ``broadcast_round_open``). The weights are
        keyed by ``theta_ref.content_hash`` / ``phi_ref.content_hash`` — the exact hash ``fetch_params``
        recomputes — so a participant fetching θ_t/φ_t resolves and hash-verifies (``INV-CHECKPOINT-HASH``).
        """
        self._committed = global_state
        self._weights[global_state.theta_ref.content_hash] = dict(theta_weights)
        self._weights[global_state.phi_ref.content_hash] = dict(phi_weights)

    def commit_root(
        self, *, participant_id: str, round_index: int, root: bytes
    ) -> None:
        """Record a participant's committed ``R_c`` for ``round_index`` (the ``Commitment`` seam).

        Equivalent to the participant sending a ``Commitment`` then the coordinator ingesting it: the
        binding target an ``Update``'s ``Δ_c`` is checked against (``INV-COMMIT-BINDING``). Tests/the
        participant node use this to register ``R_c`` before submitting an ``Update`` (the in-process
        convenience for the two-step Commitment→Update handshake).
        """
        self._committed_roots[(round_index, participant_id)] = root

    # --- helpers ---

    def _ingest_update(self, message: Update) -> PseudoGradient:
        """Bind-check an ingress ``Update`` and reconstruct its ``PseudoGradient`` (residency-guarded).

        The ``Δ_c`` ``dataset_root`` is bound against the participant's committed ``R_c`` via
        :func:`~lensemble.provenance.commit.verify_binding` (:class:`~lensemble.errors.CommitmentMismatch`
        on mismatch, ``INV-COMMIT-BINDING``, never swallowed; an uncommitted participant has no valid
        ``R_c``, so its update is rejected the same way). The reconstructed
        :class:`PseudoGradient` re-validates finiteness + the 32-byte root, so a malformed delta fails
        closed at the carrier boundary too.
        """
        declared_root = bytes.fromhex(message.dataset_root)
        committed_root = self._committed_roots.get(
            (message.round_index, message.participant_id)
        )
        if committed_root is None:
            # No Commitment ever bound this participant for the round: there is no valid R_c to bind to, so
            # the unattributed delta is rejected fail-closed (INV-COMMIT-BINDING; never swallowed).
            raise CommitmentMismatch(
                f"participant {message.participant_id!r} submitted an Update for round "
                f"{message.round_index} with no committed R_c; rejecting (INV-COMMIT-BINDING)",
                code=LensembleErrorCode.COMMITMENT_MISMATCH,
                remediation="send a Commitment binding R_c before the Update; never swallow",
            )
        # verify_binding raises CommitmentMismatch on a root mismatch / MerkleVerificationError on a
        # malformed root — both typed ingress errors, never swallowed.
        verify_binding(committed_root, declared_root, self._scheme)
        delta = to_delta_tensor(message)
        return PseudoGradient(
            delta=delta,
            l2_norm=float(message.l2_norm),
            dataset_root=declared_root,
            round_index=message.round_index,
            clipped=True,
        )
