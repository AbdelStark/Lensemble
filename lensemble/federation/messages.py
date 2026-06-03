"""lensemble.federation.messages — the four boundary-crossing ControlMessages (RFC-0013 §5 / RFC-0003 §8).

These are the *wire layer* beneath the operation-oriented ``Transport`` Protocol
(:mod:`lensemble.federation.transport`). The runtime classes (``Coordinator`` #42, ``Participant`` #43)
talk through that op-oriented ``Transport`` (``register`` / ``fetch_params`` / ``submit_update`` /
``collect_updates`` / ...); the *networked* realization of that Protocol
(:class:`~lensemble.federation.network.NetworkedTransport`) carries each operation as one or more of the
four ``ControlMessage`` s defined here, exchanged over a low-level
:class:`~lensemble.federation.network.MessageChannel`. The four messages reproduce the RFC-0013 §5 table
exactly, each bound to the ``RoundState`` transition it drives:

==============  =====================  ==============================  ===========================
Message         Direction              Drives transition               Protection
==============  =====================  ==============================  ===========================
``RoundOpen``   coord → participant    enters ``OPEN`` (broadcast)     integrity (hash)
``Commitment``  participant → coord    counts toward the quorum        binding (``INV-COMMIT-BINDING``)
``Update``      participant → agg.     counts toward ``AGGREGATING``   DP clip+noise + secure-agg mask
``RoundClose``  coord → all            marks ``CLOSED``                integrity (``INV-CHECKPOINT-HASH``)
==============  =====================  ==============================  ===========================

Conventions (conventions §8): every message is a frozen pydantic v2 model with ``extra="forbid"`` and an
integer ``schema_version`` (gated by :func:`parse_control_message` FIRST, fail-closed, before any field
validation — mirroring :func:`lensemble.provenance.commit.parse_dataset_commitment`). A discriminated
``kind`` literal tags each message so a single raw dict round-trips to the right model.

``INV-RESIDENCY`` (the load-bearing property of this module). **No** message carries a raw observation,
raw action, or private embedding ``f_theta(x)``. ``RoundOpen`` / ``RoundClose`` carry only content hashes
and coordination scalars; ``Commitment`` carries only the 32-byte dataset Merkle root ``R_c`` as hex; the
``Update`` carries the *released* masked ``Δ_c`` as a JSON-native finite list of floats — exactly the
:class:`~lensemble.federation.pseudogradient.PseudoGradient.delta` that crosses, nothing else. The
``Update`` constructors (:func:`from_pseudogradient` / :func:`to_delta_tensor`) route the carrier through
:func:`lensemble.data.residency.guard_egress` so a non-``PseudoGradient``-shaped raw-tensor payload fails
closed with :class:`~lensemble.errors.ResidencyViolation` (never swallowed).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated, Literal, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

from lensemble.data.residency import guard_egress
from lensemble.errors import LensembleErrorCode, SchemaVersionMismatch
from lensemble.federation.pseudogradient import PseudoGradient

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from torch import Tensor

# The wire schema version for the four control messages (conventions §8). Bumped on any change to a
# message's field set / semantics; gated FIRST by parse_control_message (a too-new version fails closed).
CONTROL_MESSAGE_SCHEMA_VERSION = 1

_HASH_HEX_LEN = 64  # SHA-256 as lowercase hex (INV-CHECKPOINT-HASH / INV-PROBE-PIN / INV-COMMIT-BINDING)


def _is_hex(value: str, *, length: int) -> bool:
    """True iff ``value`` is exactly ``length`` lowercase hexadecimal characters."""
    return len(value) == length and all(c in "0123456789abcdef" for c in value)


class _ControlMessageBase(BaseModel):
    """Frozen pydantic v2 base for every control message: ``extra="forbid"`` + integer schema version.

    The ``kind`` literal (set on each subclass) is the discriminator :func:`parse_control_message` keys on
    so one raw dict round-trips to the right model. Frozen + extra-forbid is the conventions §8 contract
    for a boundary-crossing payload: an unexpected field is a malformed message, rejected at ingress.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=CONTROL_MESSAGE_SCHEMA_VERSION, ge=1)


class RoundOpen(_ControlMessageBase):
    """coord → participant; enters ``OPEN`` (broadcast). Carries integrity hashes only (RFC-0013 §5).

    Mirrors :class:`~lensemble.federation.state.GlobalState` on the wire: ``theta_ref`` / ``phi_ref`` are
    carried as a (content-hash, locator) pair (the global ``(θ_t, φ_t)`` are referenced by hash, never
    inlined — 03 §7); ``probe_hash`` / ``landmark_hashes`` are the pinned-probe content hashes
    (``INV-PROBE-PIN``); ``sketch_seed`` is ``s_t`` (``INV-SKETCH-CONSISTENCY``); ``inner_horizon`` is the
    ``H`` local-step count. No tensor, no raw data crosses (``INV-RESIDENCY``).
    """

    kind: Literal["round_open"] = "round_open"
    theta_ref_hash: str  # θ_t content hash, 64-char lowercase hex (INV-CHECKPOINT-HASH)
    theta_ref_locator: str  # θ_t fetch locator (RFC-0010)
    phi_ref_hash: str  # φ_t content hash, 64-char lowercase hex
    phi_ref_locator: str  # φ_t fetch locator
    round_index: int = Field(ge=0)
    sketch_seed: int  # s_t (INV-SKETCH-CONSISTENCY)
    probe_hash: str  # 32-byte pinned-probe content hash as 64-char hex (INV-PROBE-PIN)
    landmark_hashes: tuple[str, ...]  # per-landmark content hashes (integrity)
    inner_horizon: int = Field(ge=0)  # H, the local inner-step count

    @field_validator("theta_ref_hash", "phi_ref_hash", "probe_hash")
    @classmethod
    def _hash_is_hex64(cls, v: str) -> str:
        if not _is_hex(v, length=_HASH_HEX_LEN):
            raise ValueError(
                f"hash must be {_HASH_HEX_LEN}-char lowercase hex, got {v!r}"
            )
        return v

    @field_validator("theta_ref_locator", "phi_ref_locator")
    @classmethod
    def _locator_nonempty(cls, v: str) -> str:
        if not v:
            raise ValueError("ref locator must be a non-empty fetch locator")
        return v

    @field_validator("landmark_hashes")
    @classmethod
    def _landmarks_are_hex64(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for h in v:
            if not _is_hex(h, length=_HASH_HEX_LEN):
                raise ValueError(
                    f"landmark hash must be {_HASH_HEX_LEN}-char hex, got {h!r}"
                )
        return v


class Commitment(_ControlMessageBase):
    """participant → coord; counts toward the ``OPEN→COLLECTING`` quorum (RFC-0013 §5).

    Carries the participant's dataset Merkle root ``R_c`` (``INV-COMMIT-BINDING``, RFC-0014) as 64-char
    lowercase hex — the binding a later ``Update``'s ``dataset_root`` is checked against at ingress
    (:func:`lensemble.provenance.commit.verify_binding`). No dataset, no raw episode crosses; only the
    32-byte root (``INV-RESIDENCY``).
    """

    kind: Literal["commitment"] = "commitment"
    participant_id: str = Field(min_length=1)
    round_index: int = Field(ge=0)
    dataset_root: str  # R_c as 64-char lowercase hex (INV-COMMIT-BINDING)

    @field_validator("dataset_root")
    @classmethod
    def _root_is_hex64(cls, v: str) -> str:
        if not _is_hex(v, length=_HASH_HEX_LEN):
            raise ValueError(
                f"dataset_root must be {_HASH_HEX_LEN}-char lowercase hex, got {v!r}"
            )
        return v


class Update(_ControlMessageBase):
    """participant → aggregator; counts toward ``COLLECTING→AGGREGATING`` (RFC-0013 §5).

    Carries the *released* masked ``Δ_c`` — the
    :class:`~lensemble.federation.pseudogradient.PseudoGradient.delta` after DP clip+noise (and secure-agg
    masking, owned by RFC-0011) — as a JSON-native finite ``tuple[float, ...]``, bound to the participant's
    ``dataset_root`` ``R_c``. This is the ONLY participant-derived object permitted across the boundary; it
    is NEVER a raw observation/action/embedding (``INV-RESIDENCY``), and the ``delta`` is validated finite
    (a NaN/Inf is a malformed update). Build it via :func:`from_pseudogradient` (which residency-guards the
    carrier) and recover the tensor via :func:`to_delta_tensor`.
    """

    kind: Literal["update"] = "update"
    participant_id: str = Field(min_length=1)
    round_index: int = Field(ge=0)
    dataset_root: (
        str  # R_c the delta binds to, 64-char lowercase hex (INV-COMMIT-BINDING)
    )
    delta: tuple[
        float, ...
    ]  # the masked Δ_c as JSON-native finite floats (never a tensor)
    l2_norm: float = Field(ge=0.0)  # ‖delta‖ recorded for the DP-bound check

    @field_validator("dataset_root")
    @classmethod
    def _root_is_hex64(cls, v: str) -> str:
        if not _is_hex(v, length=_HASH_HEX_LEN):
            raise ValueError(
                f"dataset_root must be {_HASH_HEX_LEN}-char lowercase hex, got {v!r}"
            )
        return v

    @field_validator("delta")
    @classmethod
    def _delta_is_finite(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if any(not math.isfinite(x) for x in v):
            raise ValueError("Update.delta contains non-finite values (NaN/Inf)")
        return v

    @field_validator("l2_norm")
    @classmethod
    def _l2_norm_is_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Update.l2_norm must be finite")
        return v


class RoundClose(_ControlMessageBase):
    """coord → all; marks ``CLOSED`` (RFC-0013 §5).

    Carries the committed ``(θ_{t+1}, φ_{t+1})`` global-model content hash (``INV-CHECKPOINT-HASH``) as
    64-char lowercase hex — an integrity hash, no weights, no raw data (``INV-RESIDENCY``).
    """

    kind: Literal["round_close"] = "round_close"
    round_index: int = Field(ge=0)
    global_model_hash: str  # (θ_{t+1}, φ_{t+1}) content hash, 64-char lowercase hex

    @field_validator("global_model_hash")
    @classmethod
    def _hash_is_hex64(cls, v: str) -> str:
        if not _is_hex(v, length=_HASH_HEX_LEN):
            raise ValueError(
                f"global_model_hash must be {_HASH_HEX_LEN}-char lowercase hex, got {v!r}"
            )
        return v


# The discriminated union over the four messages: pydantic selects the model by the `kind` literal so a
# single raw dict validates to exactly one message type (or fails). Annotated with the discriminator field.
ControlMessage = Annotated[
    Union[RoundOpen, Commitment, Update, RoundClose],
    Field(discriminator="kind"),
]


class _ControlMessageEnvelope(BaseModel):
    """Internal adapter holding the discriminated union, used by :func:`parse_control_message`."""

    model_config = ConfigDict(extra="forbid")

    message: ControlMessage


def parse_control_message(
    raw: "Mapping[str, Any]",
) -> RoundOpen | Commitment | Update | RoundClose:
    """Validate a raw dict into one of the four control messages, gating ``schema_version`` FIRST.

    Fail-closed loader (conventions §8, mirroring
    :func:`lensemble.provenance.commit.parse_dataset_commitment`): a missing / non-integer / too-new
    ``schema_version`` raises :class:`~lensemble.errors.SchemaVersionMismatch` BEFORE any field
    validation. A well-versioned but malformed payload (unknown ``kind``, missing/extra field, non-finite
    ``delta``) raises the pydantic :class:`~pydantic.ValidationError`. Both are typed ingress errors that
    the networked transport never swallows (RFC-0013 §7: a malformed message → reject, do not advance).
    """
    version = raw.get("schema_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version > CONTROL_MESSAGE_SCHEMA_VERSION
    ):
        raise SchemaVersionMismatch(
            f"ControlMessage schema_version {version!r} exceeds reader max "
            f"{CONTROL_MESSAGE_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation="upgrade lensemble to read this message, or re-emit at the supported schema",
        )
    # The discriminated union validates `kind` and the per-message fields (extra="forbid"); a malformed
    # payload raises pydantic ValidationError, propagated unchanged to the ingress caller.
    return _ControlMessageEnvelope.model_validate({"message": dict(raw)}).message


def from_pseudogradient(pg: PseudoGradient, *, participant_id: str) -> Update:
    """Build an :class:`Update` from a :class:`PseudoGradient`, residency-guarded (``INV-RESIDENCY``).

    Routes ``pg`` through :func:`lensemble.data.residency.guard_egress` FIRST so only a real
    ``PseudoGradient`` carrier (its ``delta`` the sole permitted tensor) can become an ``Update``; a
    non-``PseudoGradient``-shaped raw-tensor payload fails closed with
    :class:`~lensemble.errors.ResidencyViolation` (never swallowed). The released ``delta`` is serialized as
    a JSON-native finite ``tuple[float, ...]`` (no tensor crosses the wire), bound to ``pg.dataset_root``.
    """
    # Fail-closed: only a vetted PseudoGradient carrier passes the guard. A bare-tensor / unknown payload
    # raises ResidencyViolation here, before any field is read onto the wire.
    guard_egress(pg, boundary="participant->coordinator")
    if not isinstance(pg, PseudoGradient):
        # Defensive: guard_egress permits any object exposing __egress_role__ == PSEUDO_GRADIENT, but the
        # Update wire form is defined ONLY for the real carrier. Reject anything else fail-closed.
        raise _residency_violation()
    delta = pg.delta.detach().cpu().to(torch.float32)
    return Update(
        participant_id=participant_id,
        round_index=pg.round_index,
        dataset_root=pg.dataset_root.hex(),
        delta=tuple(float(x) for x in delta.tolist()),
        l2_norm=float(pg.l2_norm),
    )


def to_delta_tensor(update: Update) -> "Tensor":
    """Recover the flat fp32 ``Δ_c`` tensor from an :class:`Update` (the inbound wire→tensor seam).

    The inverse of :func:`from_pseudogradient`'s serialization. The result is a 1-D fp32 tensor; the
    caller (the networked transport's ``collect_updates``) re-wraps it in a
    :class:`~lensemble.federation.pseudogradient.PseudoGradient` (which re-validates finiteness and the
    ``R_c`` length, ``INV-COMMIT-BINDING``).
    """
    return torch.tensor(update.delta, dtype=torch.float32)


def _residency_violation() -> "Any":
    """Build the fail-closed :class:`~lensemble.errors.ResidencyViolation` for a non-carrier payload."""
    from lensemble.errors import ResidencyViolation

    err = ResidencyViolation(
        "only a PseudoGradient may become an Update; a raw payload may not cross the boundary "
        "(INV-RESIDENCY)",
        code=LensembleErrorCode.RESIDENCY_VIOLATION,
        remediation="release a privatized PseudoGradient.delta; never place a raw tensor in a message",
    )
    err.tensor_role = "raw_tensor_or_embedding"  # type: ignore[attr-defined]
    return err
