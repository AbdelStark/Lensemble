"""TEE-attested secure-aggregation backend (RFC-0011 §5/§6, 07 §2.7). Issue #48.

A second, config-selectable secure-aggregation backend behind the same ``SecureAggregator`` interface as
the masking backend (#47) and the simulated backend (#46). The simulated enclave computes the fp32
plaintext ``Σ_c Δ_c`` inside the enclave boundary and returns ONLY the sum; the participant verifies the
enclave's ``TEEAttestation`` against the pinned ``code_hash`` and the vendor root BEFORE sending, refusing
to send on a failed verification (``cause="attestation_failed"``). The enclave never materializes or
returns an individual ``Δ_c`` (``INV-RESIDENCY``); an attempt to egress one trips the residency guard.
"""

from __future__ import annotations

import pytest
import torch

from lensemble.aggregation import (
    FieldParams,
    PairwiseMaskAggregator,
    SecureAggregator,
    SimulatedSecureAggregator,
    TEEAggregator,
    TEEAttestation,
    encode_delta,
    verify_attestation,
)
from lensemble.aggregation.tee import enclave_measurement_for, sign_quote
from lensemble.config import load_config
from lensemble.errors import (
    ConfigError,
    LensembleErrorCode,
    ResidencyViolation,
    SecureAggregationError,
)

# Reuse the FieldParams/MaskedUpdate/encode_delta fixtures from the simulated-backend test (#46).
_DIM = 8
_FIELD = FieldParams(modulus=2**32, scale=2.0**16, dim=_DIM)
_ROOT = b"\x00" * 32
# The pinned aggregator code identity (hex SHA-256), as it would live in the RunManifest.
_CODE_HASH = "ab" * 32
_VENDOR_ROOT = b"vendor-attestation-root-key-0001"


def _deltas(c: int = 4) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {f"p{i}": torch.randn(_DIM, generator=g) for i in range(c)}


def _updates(deltas: dict[str, torch.Tensor]) -> dict:
    return {
        pid: encode_delta(
            d, _FIELD, participant_id=pid, round_index=0, dataset_root=_ROOT
        )
        for pid, d in deltas.items()
    }


def _good_attestation(code_hash: str = _CODE_HASH) -> TEEAttestation:
    """A well-formed attestation for an enclave running the pinned ``code_hash``."""
    measurement = enclave_measurement_for(code_hash)
    return TEEAttestation(
        enclave_measurement=measurement,
        quote=sign_quote(measurement, code_hash, vendor_root=_VENDOR_ROOT),
        code_hash=code_hash,
    )


# --- protocol interchangeability (RFC-0011 §6) ---


def test_both_backends_satisfy_secure_aggregator_protocol() -> None:
    # Structural (runtime-checkable Protocol) interchangeability behind one interface.
    assert isinstance(PairwiseMaskAggregator({}), SecureAggregator)
    assert isinstance(TEEAggregator(_good_attestation()), SecureAggregator)
    assert isinstance(SimulatedSecureAggregator(), SecureAggregator)


def test_tee_reveals_same_sum_as_simulated_backend() -> None:
    deltas = _deltas(4)
    updates = _updates(deltas)
    plaintext = torch.stack(list(deltas.values())).sum(dim=0)
    tee = TEEAggregator(_good_attestation()).aggregate(
        updates, field=_FIELD, round_index=0, threshold=3
    )
    simulated = SimulatedSecureAggregator().aggregate(
        updates, field=_FIELD, round_index=0, threshold=3
    )
    assert torch.equal(tee, simulated)  # the TEE path yields the correct sum
    assert torch.allclose(tee, plaintext, atol=len(deltas) / _FIELD.scale)
    assert tee.dtype == torch.float32


# --- attestation verification (fail-closed on the participant side, RFC-0011 §5) ---


def test_valid_attestation_passes_then_aggregate_yields_correct_sum() -> None:
    attestation = _good_attestation()
    # The participant-side check runs BEFORE opening the channel and does not raise.
    assert (
        verify_attestation(
            attestation, pinned_code_hash=_CODE_HASH, vendor_root=_VENDOR_ROOT
        )
        is None
    )
    deltas = _deltas(3)
    revealed = TEEAggregator(attestation).aggregate(
        _updates(deltas), field=_FIELD, round_index=0, threshold=2
    )
    plaintext = torch.stack(list(deltas.values())).sum(dim=0)
    assert torch.allclose(revealed, plaintext, atol=len(deltas) / _FIELD.scale)


def test_measurement_mismatch_rejected_participant_refuses_to_send() -> None:
    # The enclave advertises a measurement for a DIFFERENT code than the pinned code_hash.
    wrong = _good_attestation(code_hash="cd" * 32)
    with pytest.raises(SecureAggregationError) as exc:
        verify_attestation(wrong, pinned_code_hash=_CODE_HASH, vendor_root=_VENDOR_ROOT)
    assert exc.value.code == LensembleErrorCode.SECURE_AGG_FAILED
    assert exc.value.cause == "attestation_failed"  # type: ignore[attr-defined]
    assert exc.value.remediation, "remediation must be non-empty"


def test_quote_forged_against_wrong_vendor_root_rejected() -> None:
    measurement = enclave_measurement_for(_CODE_HASH)
    forged = TEEAttestation(
        enclave_measurement=measurement,
        quote=sign_quote(
            measurement, _CODE_HASH, vendor_root=b"attacker-root-not-vendor-0000000"
        ),
        code_hash=_CODE_HASH,
    )
    with pytest.raises(SecureAggregationError) as exc:
        verify_attestation(
            forged, pinned_code_hash=_CODE_HASH, vendor_root=_VENDOR_ROOT
        )
    assert exc.value.cause == "attestation_failed"  # type: ignore[attr-defined]


def test_attestation_code_hash_field_must_match_pinned() -> None:
    # A measurement that matches the pinned code_hash but a self-declared code_hash field that does not.
    measurement = enclave_measurement_for(_CODE_HASH)
    inconsistent = TEEAttestation(
        enclave_measurement=measurement,
        quote=sign_quote(measurement, "ef" * 32, vendor_root=_VENDOR_ROOT),
        code_hash="ef" * 32,
    )
    with pytest.raises(SecureAggregationError) as exc:
        verify_attestation(
            inconsistent, pinned_code_hash=_CODE_HASH, vendor_root=_VENDOR_ROOT
        )
    assert exc.value.cause == "attestation_failed"  # type: ignore[attr-defined]


# --- TEEAttestation construction validation ---


def test_attestation_rejects_non_hex_code_hash() -> None:
    with pytest.raises(SecureAggregationError):
        TEEAttestation(
            enclave_measurement=b"\x00" * 32, quote=b"\x00" * 32, code_hash="not-hex!!"
        )


def test_attestation_rejects_empty_measurement_or_quote() -> None:
    with pytest.raises(SecureAggregationError):
        TEEAttestation(
            enclave_measurement=b"", quote=b"\x00" * 32, code_hash=_CODE_HASH
        )
    with pytest.raises(SecureAggregationError):
        TEEAttestation(
            enclave_measurement=b"\x00" * 32, quote=b"", code_hash=_CODE_HASH
        )


def test_attestation_rejects_non_bytes_fields() -> None:
    with pytest.raises(SecureAggregationError):
        TEEAttestation(
            enclave_measurement="not-bytes",  # type: ignore[arg-type]
            quote=b"\x00" * 32,
            code_hash=_CODE_HASH,
        )


def test_attestation_rejects_non_str_code_hash() -> None:
    with pytest.raises(SecureAggregationError):
        TEEAttestation(
            enclave_measurement=b"\x00" * 32,
            quote=b"\x00" * 32,
            code_hash=b"\xab" * 32,  # type: ignore[arg-type]
        )


def test_measurement_mismatch_with_matching_code_hash_rejected() -> None:
    # The self-declared code_hash matches the pinned one, but the measured identity is for other code:
    # a real enclave running tampered code would report a measurement != enclave_measurement_for(code).
    tampered = TEEAttestation(
        enclave_measurement=b"\x11" * 32,  # not enclave_measurement_for(_CODE_HASH)
        quote=sign_quote(b"\x11" * 32, _CODE_HASH, vendor_root=_VENDOR_ROOT),
        code_hash=_CODE_HASH,
    )
    with pytest.raises(SecureAggregationError) as exc:
        verify_attestation(
            tampered, pinned_code_hash=_CODE_HASH, vendor_root=_VENDOR_ROOT
        )
    assert exc.value.cause == "attestation_failed"  # type: ignore[attr-defined]


# --- below-threshold: fail closed, no partial sum (like the simulated/masking backends) ---


def test_below_threshold_refuses_partial_sum() -> None:
    updates = _updates(_deltas(2))
    with pytest.raises(SecureAggregationError) as exc:
        TEEAggregator(_good_attestation()).aggregate(
            updates, field=_FIELD, round_index=4, threshold=3
        )
    assert exc.value.code == LensembleErrorCode.SECURE_AGG_FAILED
    assert exc.value.present == 2  # type: ignore[attr-defined]
    assert exc.value.threshold == 3  # type: ignore[attr-defined]
    assert exc.value.round == 4  # type: ignore[attr-defined]
    assert exc.value.cause == "below_threshold"  # type: ignore[attr-defined]


# --- residency egress (INV-RESIDENCY, 07 §2.7) ---


def test_enclave_returns_only_the_reduced_sum() -> None:
    # The only value crossing the enclave egress is the reduced sum (shape (dim,)), never a per-participant
    # stack: structurally the aggregator cannot return an individual delta.
    deltas = _deltas(4)
    revealed = TEEAggregator(_good_attestation()).aggregate(
        _updates(deltas), field=_FIELD, round_index=0, threshold=3
    )
    assert revealed.shape == (_DIM,)


def test_emitting_individual_delta_through_enclave_egress_raises() -> None:
    # Routing an individual MaskedUpdate (carrying a per-participant tensor) through the enclave egress
    # guard is fail-closed: ResidencyViolation, never swallowed (INV-RESIDENCY).
    agg = TEEAggregator(_good_attestation())
    one_update = next(iter(_updates(_deltas(2)).values()))
    with pytest.raises(ResidencyViolation) as exc:
        agg.egress(one_update)
    assert exc.value.code == LensembleErrorCode.RESIDENCY_VIOLATION


def test_egress_permits_the_reduced_sum() -> None:
    # The reduced fp32 sum is the one permitted egress value (so the guard is not vacuously rejecting).
    agg = TEEAggregator(_good_attestation())
    revealed = agg.aggregate(
        _updates(_deltas(3)), field=_FIELD, round_index=0, threshold=2
    )
    assert agg.egress(revealed) is revealed


def test_egress_refuses_a_bare_per_participant_tensor() -> None:
    # A raw per-participant masked tensor (not the vetted sum) is refused by the guard (INV-RESIDENCY).
    agg = TEEAggregator(_good_attestation())
    one_update = next(iter(_updates(_deltas(2)).values()))
    with pytest.raises(ResidencyViolation):
        agg.egress(one_update.masked)


def test_egress_refuses_a_clean_non_sum_value() -> None:
    # Even a clean (non-resident) value that is not a reduced sum may not masquerade as the egress payload:
    # only the aggregated Σ_c Δ_c crosses the enclave boundary (fail-closed).
    agg = TEEAggregator(_good_attestation())
    with pytest.raises(SecureAggregationError):
        agg.egress({"norm": 1.0, "count": 3})


# --- aggregator construction (RFC-0011 §5/§6) ---


def test_aggregator_from_bare_code_hash_advertises_measurement() -> None:
    # A code_hash-only construction advertises the matching enclave measurement (the quote is a placeholder).
    agg = TEEAggregator(_CODE_HASH)
    assert agg.attestation.code_hash == _CODE_HASH
    assert agg.attestation.enclave_measurement == enclave_measurement_for(_CODE_HASH)
    revealed = agg.aggregate(
        _updates(_deltas(3)), field=_FIELD, round_index=0, threshold=2
    )
    assert revealed.shape == (_DIM,)


# --- config selection (RFC-0011 §6) ---


def test_config_resolves_tee_backend() -> None:
    cfg = load_config(overrides=["federation.aggregation_backend=tee"])
    assert cfg.federation.aggregation_backend == "tee"


def test_config_default_backend_is_masking() -> None:
    assert load_config().federation.aggregation_backend == "masking"


def test_config_rejects_unknown_backend() -> None:
    with pytest.raises(ConfigError):
        load_config(overrides=["federation.aggregation_backend=enclave"])
