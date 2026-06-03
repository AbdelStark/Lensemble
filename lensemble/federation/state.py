"""lensemble.federation.state — the broadcast round state ``GlobalState`` + ``ParamRef`` (03 §7).

``GlobalState`` is the canonical global-model reference plus the per-round public parameters every
participant needs to run a *comparable* local round; the ``Coordinator`` (#42) constructs it and broadcasts
it in the ``RoundOpen`` message (`RFC-0003 §8`). A ``ParamRef`` is a content hash plus a fetch locator for
a safetensors weight artifact (RFC-0010) — the global ``(θ_t, φ_t)`` are referenced, never broadcast inline.

Invariants pinned here (the *shape* invariants; the runtime invariants are enforced where they are used):

- ``INV-CHECKPOINT-HASH`` — a ``ParamRef.content_hash`` is a 64-char lowercase SHA-256 hex of the
  canonical safetensors artifact; the fetch path re-verifies it (``lensemble.federation.transport``).
- ``INV-SKETCH-CONSISTENCY`` — ``sketch_seed`` (``s_t``) is the single seed every participant derives the
  identical SIGReg projection ``A`` from (``lensemble.model.sigreg.build_sketch``).
- ``INV-PROBE-PIN`` — ``probe_hash`` is the 32-byte content hash of the pinned public probe; a
  participant refuses a round whose ``probe_hash`` differs from its pinned probe (``ProbeError``).
- ``INV-WARMSTART-T0`` — at ``round_index == 0`` ``theta_ref`` resolves to weights hash-identical to the
  pinned warm-start; enforced at the participant (``GaugeError`` per the #43 acceptance criterion).
"""

from __future__ import annotations

from dataclasses import dataclass

_HASH_HEX_LEN = 64  # SHA-256 as lowercase hex (INV-CHECKPOINT-HASH)
_PROBE_HASH_BYTES = 32  # SHA-256 digest length (INV-PROBE-PIN)


def _is_hex64(value: str) -> bool:
    """True iff ``value`` is exactly 64 lowercase hexadecimal characters (a SHA-256 hex digest)."""
    if len(value) != _HASH_HEX_LEN:
        return False
    return all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class ParamRef:
    """A reference to a shared weight artifact: a content hash + a fetch locator (03 §7 / RFC-0010).

    ``content_hash`` is the 64-char lowercase SHA-256 hex of the canonical safetensors artifact
    (``INV-CHECKPOINT-HASH``); ``locator`` is the fetch handle a ``Transport`` resolves to the weights.
    The global ``(θ_t, φ_t)`` are referenced by hash, never broadcast inline (03 §7).
    """

    content_hash: (
        str  # SHA-256 hex (64) of the safetensors artifact (INV-CHECKPOINT-HASH)
    )
    locator: str  # fetch locator for the artifact (RFC-0010)

    def __post_init__(self) -> None:
        if not _is_hex64(self.content_hash):
            raise ValueError(
                f"ParamRef.content_hash must be 64 lowercase hex chars (SHA-256), got "
                f"{self.content_hash!r}"
            )
        if not self.locator:
            raise ValueError("ParamRef.locator must be a non-empty fetch locator")


@dataclass(frozen=True)
class GlobalState:
    """The broadcast per-round global state (03 §7 / RFC-0013 §1). Frozen; validated at construction.

    Carries the round-$t$ encoder/predictor artifact references (``theta_ref``/``phi_ref``), the round
    index ``t``, the SIGReg ``sketch_seed`` ``s_t`` (``INV-SKETCH-CONSISTENCY``), the 32-byte pinned probe
    ``probe_hash`` (``INV-PROBE-PIN``), and the ``wmcp_version`` contract pinned for the round.

    Constructed by the ``Coordinator`` (#42) and consumed by ``Participant.local_round`` (#43). The
    ``round_index == 0`` case additionally pins ``INV-WARMSTART-T0`` (enforced at the participant).
    """

    theta_ref: ParamRef
    phi_ref: ParamRef
    round_index: int
    sketch_seed: int
    probe_hash: bytes  # 32-byte SHA-256 of the pinned public probe content
    wmcp_version: str

    def __post_init__(self) -> None:
        if self.round_index < 0:
            raise ValueError(
                f"GlobalState.round_index must be >= 0, got {self.round_index}"
            )
        if len(self.probe_hash) != _PROBE_HASH_BYTES:
            raise ValueError(
                f"GlobalState.probe_hash must be {_PROBE_HASH_BYTES} bytes (SHA-256), got "
                f"{len(self.probe_hash)}"
            )
        if not self.wmcp_version:
            raise ValueError(
                "GlobalState.wmcp_version must be a non-empty contract tag"
            )
