"""lensemble.federation.transport — the participant↔coordinator message seam (RFC-0013 §1/§5).

``Transport`` is the abstract control-plane the runtime classes talk through: the ``Participant`` (#43)
calls ``register`` / ``recover_global_state`` / ``fetch_params`` / ``submit_update``, and the
``Coordinator`` (#42) calls ``broadcast_round_open`` / ``collect_updates``. Defining it once here lets
#42 and #45 reuse the *same* seam (the network transport of RFC-0013 §5 implements this Protocol).

``fetch_params`` is the integrity boundary: it resolves a :class:`~lensemble.federation.state.ParamRef`
to its weights and **re-verifies** the recomputed content hash against ``ref.content_hash``
(``INV-CHECKPOINT-HASH``); a mismatch is a :class:`~lensemble.errors.CheckpointIntegrityError` and no
tensors are returned. The canonical hash is the same SHA-256-over-sorted-safetensors-bytes that
``lensemble.model.encoder.encoder_content_hash`` uses, so a weight set seeded under that hash round-trips.

``InProcessTransport`` is the in-memory realization used by single-process Stage-A/B runs and the test
suite: a registered-participants set, the latest committed ``GlobalState``, a ``content_hash -> weights``
store backing ``fetch_params``, and a per-round ``dict[participant_id, PseudoGradient]`` backing
``submit_update`` / ``collect_updates``. ``commit`` is the seeding helper that publishes a committed state
together with its θ/φ weights (storing each under its ref's content hash).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from safetensors.torch import save as st_save

from lensemble.errors import CheckpointIntegrityError, LensembleErrorCode

if TYPE_CHECKING:
    from torch import Tensor

    from lensemble.federation.pseudogradient import PseudoGradient
    from lensemble.federation.state import GlobalState, ParamRef


def weights_content_hash(weights: Mapping[str, "Tensor"]) -> str:
    """SHA-256 (64 lowercase hex) over the canonical safetensors bytes of a weight ``state_dict``.

    The same canonical serialization ``lensemble.model.encoder.encoder_content_hash`` uses (sorted keys,
    detached CPU-contiguous tensors), so a ``ParamRef.content_hash`` minted from an encoder/predictor
    state-dict round-trips through :meth:`Transport.fetch_params`. Realizes ``INV-CHECKPOINT-HASH``.
    """
    contiguous = {k: v.detach().cpu().contiguous() for k, v in weights.items()}
    return hashlib.sha256(st_save(contiguous)).hexdigest()


@runtime_checkable
class Transport(Protocol):
    """The participant↔coordinator control-plane seam (RFC-0013 §1/§5).

    The participant-facing methods (``register``, ``recover_global_state``, ``fetch_params``,
    ``submit_update``) are exercised by #43; the coordinator-facing methods (``broadcast_round_open``,
    ``collect_updates``) are exercised by the ``Coordinator`` (#42). A network implementation (RFC-0013 §5)
    is a real trust boundary and routes every payload through the residency guard; the in-process
    implementation is single-process and trusted.
    """

    def register(self, participant_id: str, endpoint: str) -> None:
        """Register a participant with the coordinator under ``participant_id`` at ``endpoint``."""
        ...

    def recover_global_state(self, *, participant_id: str) -> "GlobalState":
        """Return the latest committed ``GlobalState`` (the rejoiner-recovery path, RFC-0013 §3)."""
        ...

    def fetch_params(self, ref: "ParamRef") -> dict[str, "Tensor"]:
        """Resolve ``ref`` to its weights, HASH-VERIFIED against ``ref.content_hash``.

        Raises :class:`~lensemble.errors.CheckpointIntegrityError` on a recomputed-hash mismatch
        (tamper/corruption); no tensors are returned in that case (``INV-CHECKPOINT-HASH``).
        """
        ...

    def submit_update(
        self,
        *,
        participant_id: str,
        round_index: int,
        update: "PseudoGradient",
    ) -> None:
        """Submit a participant's privatized, bound ``PseudoGradient`` for ``round_index`` (#43)."""
        ...

    def broadcast_round_open(self, global_state: "GlobalState") -> None:
        """Publish a new round's ``GlobalState`` as the committed state (coordinator side, #42)."""
        ...

    def collect_updates(self, round_index: int) -> Mapping[str, "PseudoGradient"]:
        """Return the submitted updates for ``round_index`` keyed by participant id (coordinator, #42)."""
        ...


class InProcessTransport:
    """An in-memory :class:`Transport` for single-process runs and tests (RFC-0013 §1).

    Holds the registered-participants set, the latest committed ``GlobalState``, a
    ``content_hash -> weights`` store backing ``fetch_params``, and a per-round
    ``dict[participant_id, PseudoGradient]`` backing ``submit_update`` / ``collect_updates``. Single
    process and trusted: there is no real trust boundary to cross, so no residency guard runs here.
    """

    def __init__(self) -> None:
        self._registered: dict[str, str] = {}
        self._committed: "GlobalState | None" = None
        self._weights: dict[str, dict[str, "Tensor"]] = {}
        self._updates: dict[int, dict[str, "PseudoGradient"]] = {}

    def register(self, participant_id: str, endpoint: str) -> None:
        self._registered[participant_id] = endpoint

    def recover_global_state(self, *, participant_id: str) -> "GlobalState":
        if self._committed is None:
            raise CheckpointIntegrityError(
                "no committed GlobalState to recover; the coordinator has not broadcast a round",
                code=LensembleErrorCode.CHECKPOINT_INTEGRITY,
                remediation="open round 0 (broadcast_round_open / commit) before a participant joins",
            )
        return self._committed

    def fetch_params(self, ref: "ParamRef") -> dict[str, "Tensor"]:
        stored = self._weights.get(ref.content_hash)
        if stored is None:
            raise CheckpointIntegrityError(
                f"no weights stored under content hash {ref.content_hash} (locator {ref.locator!r})",
                code=LensembleErrorCode.CHECKPOINT_INTEGRITY,
                remediation="commit the artifact under its content hash before fetching it",
            )
        # Re-verify the recomputed hash against the ref so a tampered store fails closed (INV-CHECKPOINT-HASH).
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
        update: "PseudoGradient",
    ) -> None:
        self._updates.setdefault(round_index, {})[participant_id] = update

    def broadcast_round_open(self, global_state: "GlobalState") -> None:
        self._committed = global_state

    def collect_updates(self, round_index: int) -> Mapping[str, "PseudoGradient"]:
        return dict(self._updates.get(round_index, {}))

    # --- test/seed helper (not part of the Transport Protocol) ---
    def commit(
        self,
        global_state: "GlobalState",
        *,
        theta_weights: Mapping[str, "Tensor"],
        phi_weights: Mapping[str, "Tensor"],
    ) -> None:
        """Publish ``global_state`` as committed and store its θ/φ weights under their refs' hashes.

        The seam tests and the single-process coordinator use to populate the ``fetch_params`` store: the
        weights are keyed by ``theta_ref.content_hash`` / ``phi_ref.content_hash`` so a subsequent
        ``fetch_params`` resolves and hash-verifies them. A direct write to the store (bypassing the
        ref's recomputed hash) is the seam a tamper test exploits.
        """
        self._committed = global_state
        self._weights[global_state.theta_ref.content_hash] = dict(theta_weights)
        self._weights[global_state.phi_ref.content_hash] = dict(phi_weights)

    def corrupt_stored_weights(
        self, content_hash: str, weights: Mapping[str, "Tensor"]
    ) -> None:
        """Overwrite the store under ``content_hash`` with mismatching ``weights`` (tamper, tests only).

        Leaves the ``ParamRef.content_hash`` unchanged while the stored bytes change, so the next
        ``fetch_params`` recomputes a different hash and fails closed (``CheckpointIntegrityError``).
        """
        self._weights[content_hash] = dict(weights)
