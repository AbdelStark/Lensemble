"""lensemble.aggregation.tee — TEE-attested secure aggregation (RFC-0011 §5/§6, Backend B).

The second config-selectable secure-aggregation backend, behind the same ``SecureAggregator`` interface
as the pairwise-mask backend (#47) and the in-process simulated backend (#46). Where masking hides each
``Δ_c`` behind cancelling pseudo-random masks, this backend terminates the participant channel *inside an
attested enclave*: participants send **plaintext** (DP-privatized) ``Δ_c``, the enclave computes
``Σ_c Δ_c``, returns only the sum, and proves via remote attestation that it ran the pinned, auditable
aggregation code and retains nothing.

Trust assumption (RFC-0011 §5, distinct from masking's). The masking backend assumes an honest-but-curious
aggregator that learns at most the sum, with a residual collusion bound (an aggregator colluding with
``C-1`` participants can isolate the remaining update). The TEE backend instead assumes
**hardware-attestation trust**: a vendor root of trust signs an attestation quote over the enclave's
measured identity, and the participant refuses to send unless that measurement matches the pinned
aggregator ``code_hash``. This substitutes one residual-trust assumption for the other — neither strictly
stronger nor weaker — and is exposed to side-channel / microarchitectural attacks against the enclave, so
it is a pragmatic proxy, not a cryptographic guarantee (RFC-0011 §Drawbacks).

Simulation scope (v0.2). This ships against a *software-simulated* enclave: the real enclave provisioning,
the production attestation channel, and the transport are RFC-0013's (#45). Here the enclave measurement
is a domain-separated hash of the pinned ``code_hash`` and the quote is an HMAC over
``(enclave_measurement, code_hash)`` keyed by the vendor root — faithful in structure (a participant that
cannot reproduce the keyed signature is rejected) without standing in for hardware attestation.

Security posture: :meth:`TEEAggregator.aggregate` returns ONLY the fp32 sum and never materializes,
stores, or returns an individual ``Δ_c`` (``INV-RESIDENCY``); the reduction reuses the simulated backend's
fixed-order integer-field summation (``INV-AGG-DETERMINISM``). The only value crossing the enclave egress
is the reduced sum: :meth:`TEEAggregator.egress` is the single egress checkpoint — it returns the vetted
reduced sum and routes any individual ``Δ_c`` (a ``MaskedUpdate`` carrying a per-participant tensor, or a
bare resident tensor) through the residency guard, which is fail-closed with
:class:`~lensemble.errors.ResidencyViolation` — never caught-and-ignored (07 §2.7).
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from torch import Tensor

from lensemble.aggregation.secure_agg import (
    FieldParams,
    MaskedUpdate,
    _lift_to_signed,
    _sum_mod,
    assert_field_sum_reproducible,
)
from lensemble.data.residency import guard_egress
from lensemble.errors import LensembleErrorCode, SecureAggregationError

if TYPE_CHECKING:
    from collections.abc import Mapping

# Domain-separation tags keep the enclave-measurement hash and the attestation HMAC from colliding with
# any other keyed/unkeyed digest in the system (the masking backend's seed KDF, the provenance roots).
_MEASUREMENT_DOMAIN = b"lensemble/tee/measurement/v1\x00"
_QUOTE_DOMAIN = b"lensemble/tee/quote/v1\x00"
# The egress boundary label the TEE aggregator stamps on a residency violation (RFC-0011 §7).
_ENCLAVE_BOUNDARY = "enclave->coordinator"


@runtime_checkable
class SecureAggregator(Protocol):
    """The secure-sum backend contract both Backend A (masking) and Backend B (TEE) satisfy (RFC-0011 §1/§6).

    Structural: any object exposing ``aggregate(updates, *, field, round_index, threshold, recovery) ->
    Tensor`` is a ``SecureAggregator``. The backend is a config choice (``federation.aggregation_backend``,
    RFC-0011 §6); the round driver depends only on this interface so the masking, TEE, and simulated
    backends are interchangeable. ``runtime_checkable`` enables the structural ``isinstance`` test, which
    (per :mod:`typing`) checks method *presence*, not the signature.
    """

    def aggregate(
        self,
        updates: "Mapping[str, MaskedUpdate]",
        *,
        field: FieldParams,
        round_index: int,
        threshold: int,
        recovery: object | None = ...,
    ) -> Tensor:
        """Reveal the fp32 plaintext ``Σ_c Δ_c`` over the surviving set; never an individual ``Δ_c``."""
        ...


def enclave_measurement_for(code_hash: str) -> bytes:
    """The simulated enclave measurement (MRENCLAVE-equivalent) for an enclave running ``code_hash``.

    A domain-separated SHA-256 of the pinned aggregator ``code_hash`` (hex). In a real TEE this is the
    hardware-measured identity of the loaded code; here it is a deterministic stand-in so a measurement can
    be checked against the pinned ``code_hash`` (RFC-0011 §5). Not security-bearing on its own — the
    vendor-root quote is what binds it (see :func:`sign_quote`).
    """
    return hashlib.sha256(_MEASUREMENT_DOMAIN + bytes.fromhex(code_hash)).digest()


def sign_quote(
    enclave_measurement: bytes, code_hash: str, *, vendor_root: bytes
) -> bytes:
    """The simulated attestation quote: an HMAC over ``(enclave_measurement, code_hash)`` keyed by the vendor root.

    Stands in for a hardware-signed remote-attestation quote: only a holder of ``vendor_root`` can produce
    a quote that :func:`verify_attestation` accepts, so a forged measurement (or one signed under an
    attacker's root) is rejected. A software HMAC is the expected fidelity at v0.2 (RFC-0011 §Testing).
    """
    return hmac.new(
        vendor_root,
        _QUOTE_DOMAIN + enclave_measurement + bytes.fromhex(code_hash),
        hashlib.sha256,
    ).digest()


@dataclass(frozen=True)
class TEEAttestation:
    """A remote-attestation report from the simulated aggregation enclave (RFC-0011 §5).

    Fields (exactly as RFC-0011 §5 declares them):

    - ``enclave_measurement``: the measured enclave identity (MRENCLAVE-equivalent) — a hash of the loaded
      aggregation code.
    - ``quote``: the signed attestation quote, verified against the vendor attestation root.
    - ``code_hash``: the hex SHA-256 of the pinned aggregator code the enclave runs (recorded in the
      ``RunManifest``).

    The participant verifies this report against the pinned ``code_hash`` and the vendor root **before**
    opening the channel (:func:`verify_attestation`); a failed verification means the participant refuses to
    send. The trust assumption is hardware-attestation trust (a vendor root, with side-channel exposure),
    distinct from the masking backend's collusion-bounded honest-but-curious assumption (RFC-0011 §5).
    Construction validates shapes/hex so a malformed report cannot masquerade as a valid one.
    """

    enclave_measurement: (
        bytes  # MRENCLAVE-equivalent: hash of the loaded aggregation code
    )
    quote: bytes  # signed attestation quote, verified against the vendor root
    code_hash: str  # hex SHA-256 of the pinned aggregation code (RunManifest)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.enclave_measurement, (bytes, bytearray))
            or not self.enclave_measurement
        ):
            raise _attestation_error(
                "TEEAttestation.enclave_measurement must be non-empty bytes (the measured enclave identity)"
            )
        if not isinstance(self.quote, (bytes, bytearray)) or not self.quote:
            raise _attestation_error(
                "TEEAttestation.quote must be a non-empty signed attestation quote"
            )
        if not isinstance(self.code_hash, str):
            raise _attestation_error("TEEAttestation.code_hash must be a hex string")
        try:
            bytes.fromhex(self.code_hash)
        except ValueError as exc:
            raise _attestation_error(
                f"TEEAttestation.code_hash is not valid hex: {self.code_hash!r}"
            ) from exc


def _attestation_error(message: str) -> SecureAggregationError:
    """Build a ``SecureAggregationError`` carrying ``cause="attestation_failed"`` (RFC-0011 §5/§8).

    The participant treats this as fail-closed: it refuses to send when attestation does not verify.
    """
    err = SecureAggregationError(
        message,
        code=LensembleErrorCode.SECURE_AGG_FAILED,
        remediation="refuse to send to an unattested enclave; require the enclave to run the pinned "
        "aggregator code (matching code_hash) and present a quote signed by the vendor root",
    )
    err.cause = "attestation_failed"  # type: ignore[attr-defined]
    return err


def verify_attestation(
    attestation: TEEAttestation, *, pinned_code_hash: str, vendor_root: bytes
) -> None:
    """Participant-side attestation check, run BEFORE opening the channel (RFC-0011 §5). Fail-closed.

    Verifies, in order, that (a) the enclave runs the pinned aggregator code — the attestation's
    ``code_hash`` and its ``enclave_measurement`` both match the pinned ``code_hash`` — and (b) the
    ``quote`` is a valid signature over ``(enclave_measurement, code_hash)`` keyed by ``vendor_root`` (the
    simulated vendor attestation root). On ANY failure raises
    :class:`~lensemble.errors.SecureAggregationError` with ``cause="attestation_failed"`` and the
    participant must not proceed (it refuses to send). Returns ``None`` (a no-op) when the attestation
    verifies.

    This substitutes hardware-attestation trust (the vendor root of trust, with side-channel exposure) for
    the masking backend's collusion-bounded honest-but-curious assumption (RFC-0011 §5/§Drawbacks).
    """
    # (a) The enclave must run the pinned aggregator code: its self-declared code_hash and its measured
    # identity must both match the code pinned in the RunManifest. hmac.compare_digest is constant-time.
    expected_measurement = enclave_measurement_for(pinned_code_hash)
    if not hmac.compare_digest(
        attestation.code_hash.encode("ascii"), pinned_code_hash.encode("ascii")
    ):
        raise _attestation_error(
            f"attested code_hash {attestation.code_hash!r} != pinned {pinned_code_hash!r}; the enclave is "
            "not running the pinned aggregator code"
        )
    if not hmac.compare_digest(attestation.enclave_measurement, expected_measurement):
        raise _attestation_error(
            "enclave_measurement does not match the pinned code_hash; the enclave is not running the "
            "pinned aggregator code"
        )

    # (b) The quote must verify against the vendor root (a holder of vendor_root signed this measurement).
    expected_quote = sign_quote(
        attestation.enclave_measurement, attestation.code_hash, vendor_root=vendor_root
    )
    if not hmac.compare_digest(attestation.quote, expected_quote):
        raise _attestation_error(
            "attestation quote did not verify against the vendor attestation root (forged or stale quote)"
        )


class TEEAggregator:
    """TEE-attested secure-sum aggregator (RFC-0011 §5/§6, Backend B; software-simulated enclave, v0.2).

    Constructed with the enclave's :class:`TEEAttestation` (or a bare hex ``code_hash``) so the aggregator
    advertises the measured identity a participant verifies before sending. :meth:`aggregate` runs *inside
    the enclave boundary*: it reconstructs the fp32 plaintext ``Σ_c Δ_c`` over the surviving set and returns
    **only** the sum — it never materializes or returns an individual ``Δ_c`` (``INV-RESIDENCY``). Below
    ``threshold`` survivors it raises :class:`~lensemble.errors.SecureAggregationError` and returns no
    partial sum, exactly like the masking and simulated backends.

    Trust assumption (RFC-0011 §5): hardware-attestation trust (the vendor root of trust, with side-channel
    exposure) — distinct from, and neither strictly stronger nor weaker than, the masking backend's
    collusion-bounded honest-but-curious assumption. The enclave egress is policed by :meth:`egress`: the
    only value permitted to cross is the reduced sum; any individual delta is fail-closed.
    """

    def __init__(self, attestation: TEEAttestation | str) -> None:
        if isinstance(attestation, str):
            # A bare code_hash: advertise a measurement for it (a real enclave would carry a full quote).
            measurement = enclave_measurement_for(attestation)
            attestation = TEEAttestation(
                enclave_measurement=measurement,
                quote=b"\x00",  # placeholder; a code_hash-only construction is not vendor-attestable
                code_hash=attestation,
            )
        self._attestation = attestation
        # Identities of the reduced sums this enclave has produced — the only tensors permitted to egress
        # (every other tensor, including an individual Δ_c, is refused by the residency guard).
        self._vetted_sums: set[int] = set()

    @property
    def attestation(self) -> TEEAttestation:
        """The enclave's attestation report (what a participant verifies before sending)."""
        return self._attestation

    def aggregate(
        self,
        updates: "Mapping[str, MaskedUpdate]",
        *,
        field: FieldParams,
        round_index: int,
        threshold: int,
        recovery: object | None = None,
    ) -> Tensor:
        """Reconstruct the fp32 plaintext ``Σ_c Δ_c`` inside the enclave boundary (RFC-0011 §1/§5).

        Sums the participants' encoded plaintext deltas in fixed participant-id order over the associative
        integer field (``_sum_mod``), re-derives the sum once more for the determinism self-check
        (``INV-AGG-DETERMINISM``), recentres with ``_lift_to_signed``, and divides by ``field.scale``. Below
        ``threshold`` survivors raises :class:`~lensemble.errors.SecureAggregationError`
        (``present``/``threshold``/``round``/``cause``) and returns no partial sum. The returned sum is the
        only value that crosses the enclave egress (:meth:`egress`); an individual ``Δ_c`` never leaves the
        boundary (``INV-RESIDENCY``). ``recovery`` is unused — the TEE backend has no per-pair seed material.
        """
        present = sorted(updates)
        if len(present) < threshold:
            err = SecureAggregationError(
                f"only {len(present)} survivors < threshold {threshold}; refusing a partial sum",
                code=LensembleErrorCode.SECURE_AGG_FAILED,
                remediation="wait for the secure-aggregation threshold of survivors, or abort the round",
            )
            err.round = round_index  # type: ignore[attr-defined]
            err.present = len(present)  # type: ignore[attr-defined]
            err.threshold = threshold  # type: ignore[attr-defined]
            err.cause = "below_threshold"  # type: ignore[attr-defined]
            raise err

        # Inside the enclave: sum the plaintext deltas in fixed coordinate order (associative integer field).
        ordered = [updates[pid].masked for pid in present]
        total = _sum_mod(ordered, field.modulus)
        assert_field_sum_reproducible(
            total, _sum_mod(ordered, field.modulus)
        )  # determinism self-check
        signed = _lift_to_signed(total, field.modulus)
        revealed = signed.float() / field.scale
        # Mark this reduced sum as the one value vetted to cross the enclave boundary (INV-RESIDENCY).
        self._vetted_sums.add(id(revealed))
        return self.egress(revealed)

    def egress(self, value: object) -> Tensor:
        """The single enclave-egress checkpoint: only the reduced ``Σ_c Δ_c`` may leave (``INV-RESIDENCY``).

        Returns ``value`` unchanged iff it is a reduced sum this enclave produced (the one value RFC-0011 §7
        lets cross the aggregation boundary). Anything else — an individual ``Δ_c`` (a
        :class:`~lensemble.aggregation.secure_agg.MaskedUpdate` carrying a per-participant tensor), a bare
        per-participant tensor, or any other resident artifact — is routed through
        :func:`~lensemble.data.residency.guard_egress`, which is fail-closed with
        :class:`~lensemble.errors.ResidencyViolation` and never caught-and-ignored (07 §2.7, RFC-0011 §7/§8).
        """
        if isinstance(value, Tensor) and id(value) in self._vetted_sums:
            return value
        # Not a vetted sum: refuse via the residency guard. A MaskedUpdate (unknown dataclass with a
        # per-participant tensor field) and a bare tensor both fail closed -> ResidencyViolation.
        guard_egress(value, boundary=_ENCLAVE_BOUNDARY)
        # Unreachable for a resident payload (the guard raises); a clean payload that is somehow not the
        # vetted sum is still not permitted to masquerade as one, so refuse explicitly (fail-closed).
        raise _residency_refusal()


def _residency_refusal() -> SecureAggregationError:
    """Defensive fail-closed refusal for a non-resident, non-sum egress (should be unreachable)."""
    return SecureAggregationError(  # pragma: no cover - defensive; guard_egress raises first
        "only the reduced Σ_c Δ_c may leave the enclave boundary",
        code=LensembleErrorCode.SECURE_AGG_FAILED,
        remediation="route only the aggregated sum through the enclave egress (INV-RESIDENCY)",
    )
