"""lensemble.aggregation.masking — Bonawitz-style pairwise additive masking (RFC-0011 §2/§4, Backend A).

The default secure-aggregation backend. Each ordered participant pair ``(c, c')`` shares a per-round seed
``seed_{c,c'}`` from which both deterministically expand the same pseudo-random integer mask
``m_{c,c'}``; ``c`` **adds** it for every ``c' > c`` and **subtracts** it for every ``c' < c``, so the
masks cancel exactly in the sum. Each participant also adds a private **self-mask** ``b_c``. Over the
associative integer field of :mod:`lensemble.aggregation.secure_agg` the revealed sum is therefore
order-independent (``INV-AGG-DETERMINISM``) and each ``masked_c`` is computationally hiding — it carries
no recoverable ``Δ_c`` alone (``INV-RESIDENCY``). Threshold (Shamir) secret sharing of the seeds makes the
round dropout-robust.

Scope boundary (RFC-0011 §2/§4): the DH key-agreement, the public-key routing, and the share-distribution
**transport** are the control plane (RFC-0013 / #45); this module *consumes* the seeds/shares and
implements the masking math, the reconstruction, and the dropout recovery. The DH here is a toy
prime-field model (``pow(pk, sk, p)``) standing in for the production X25519 the transport will carry; its
job is only to make the simulation's reconstruction faithful (symmetric ``KA``), not to be the deployed
key exchange.

The double-masking rule (RFC-0011 §4) reconstructs **exactly one** seed per participant — the self-mask
seed ``r_c`` for a survivor (never its pairwise seeds), the DH private key ``sk_d`` for a dropout (never
its self-mask) — so no observed ``masked_c`` is ever de-masked to an isolated ``encode(Δ_c)``. Below the
threshold of survivors the aggregator returns **no partial sum** and raises ``SecureAggregationError``.

Security note: the inherited collusion bound (an aggregator colluding with ``C-1`` participants can
isolate the remaining update) is the standard single-server secure-aggregation limit; DP (RFC-0012) is
what bounds the residual leakage. This is a v0.3 (Stage C) backend; the in-process driver,
``FieldParams``/``lift_signed`` primitives, and the no-wrap modulus sizing are reused from
``secure_agg`` (#46).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lensemble.aggregation.secure_agg import (
    FieldParams,
    MaskedUpdate,
    _lift_to_signed,
    encode_delta,
)
from lensemble.errors import LensembleErrorCode, SecureAggregationError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Toy Diffie-Hellman group for the simulation (production X25519 is the transport's, #45). A 127-bit
# Mersenne prime keeps ``pow`` instant; security in the simulation is structural, not from this size.
_DH_PRIME = (1 << 127) - 1
_DH_GENERATOR = 5
# Shamir secret sharing field for the seeds. Prime > any 64-bit seed; secrets are reduced into it.
_SHAMIR_PRIME = (1 << 127) - 1
_TORCH_SEED_MASK = (1 << 63) - 1  # torch.Generator.manual_seed takes an int64


# --- key agreement + seed derivation (the simulation's view of the control-plane setup) ---


def dh_keypair(secret: int) -> tuple[int, int]:
    """A toy DH keypair ``(sk, pk)`` from a secret scalar: ``pk = g^sk mod p`` (production X25519 is #45)."""
    sk = secret % (_DH_PRIME - 1) or 1
    return sk, pow(_DH_GENERATOR, sk, _DH_PRIME)


def key_agreement(sk: int, peer_pk: int) -> int:
    """Symmetric shared secret ``KA(sk, pk') = pk'^sk mod p`` (``= KA(sk', pk)``)."""
    return pow(peer_pk, sk, _DH_PRIME)


def _kdf_seed(*parts: bytes) -> int:
    """Domain-separated SHA-256 of the parts -> a 64-bit seed for mask expansion."""
    digest = hashlib.sha256(
        b"lensemble/secagg/seed/v1\x00" + b"\x00".join(parts)
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _round_bytes(round_index: int) -> bytes:
    return int(round_index).to_bytes(8, "big")


def pairwise_seed(shared_secret: int, round_index: int) -> int:
    """The per-round pairwise seed ``KDF(KA(.) ‖ round_index)`` both parties expand the mask from."""
    return _kdf_seed(shared_secret.to_bytes(16, "big"), _round_bytes(round_index))


def self_seed(secret: int, round_index: int) -> int:
    """The per-round self-mask seed ``KDF(r_c ‖ round_index)``."""
    return _kdf_seed(
        b"self", int(secret).to_bytes(16, "big"), _round_bytes(round_index)
    )


def expand_mask(seed: int, dim: int, modulus: int) -> Tensor:
    """Deterministically expand ``seed`` to a uniform integer mask in ``[0, modulus)^dim`` (the CSPRNG)."""
    gen = torch.Generator().manual_seed(seed & _TORCH_SEED_MASK)
    return torch.randint(0, modulus, (dim,), generator=gen, dtype=torch.int64)


# --- Shamir threshold secret sharing of the seeds (over a prime field) ---


def shamir_split(secret: int, *, num_shares: int, threshold: int) -> list[bytes]:
    """Split ``secret`` into ``num_shares`` Shamir shares; any ``threshold`` reconstruct it.

    A degree ``threshold-1`` polynomial over GF(``_SHAMIR_PRIME``) with the secret as the constant term,
    evaluated at ``x = 1..num_shares``. Each share is ``x (2 bytes big-endian) ‖ y (16 bytes)``. The
    coefficients are derived deterministically from the secret (no global RNG) so a participant's setup is
    reproducible.
    """
    if not 1 <= threshold <= num_shares:
        raise ValueError(
            f"need 1 <= threshold ({threshold}) <= num_shares ({num_shares})"
        )
    secret %= _SHAMIR_PRIME
    coeffs = [secret]
    for j in range(1, threshold):
        h = hashlib.sha256(b"shamir" + secret.to_bytes(16, "big") + bytes([j])).digest()
        coeffs.append(int.from_bytes(h, "big") % _SHAMIR_PRIME)
    shares: list[bytes] = []
    for x in range(1, num_shares + 1):
        y = 0
        for coeff in reversed(coeffs):  # Horner evaluation at x
            y = (y * x + coeff) % _SHAMIR_PRIME
        shares.append(x.to_bytes(2, "big") + y.to_bytes(16, "big"))
    return shares


def shamir_reconstruct(shares: "Sequence[bytes]") -> int:
    """Reconstruct the secret from ``>= threshold`` Shamir shares via Lagrange interpolation at ``x=0``."""
    points = [
        (int.from_bytes(s[:2], "big"), int.from_bytes(s[2:], "big")) for s in shares
    ]
    if len({x for x, _ in points}) != len(points):
        raise ValueError("duplicate share x-coordinates")
    secret = 0
    for i, (xi, yi) in enumerate(points):
        num, den = 1, 1
        for j, (xj, _) in enumerate(points):
            if i == j:
                continue
            num = (num * (-xj)) % _SHAMIR_PRIME
            den = (den * (xi - xj)) % _SHAMIR_PRIME
        term = (yi * num * pow(den, -1, _SHAMIR_PRIME)) % _SHAMIR_PRIME
        secret = (secret + term) % _SHAMIR_PRIME
    return secret


# --- the participant-side masked-update construction (RFC-0011 §2) ---


def build_masked_update(
    delta: Tensor,
    *,
    participant_id: str,
    round_index: int,
    field: FieldParams,
    secret_key: int,
    self_mask_secret: int,
    peer_public_keys: "Mapping[str, int]",
    dataset_root: bytes,
) -> MaskedUpdate:
    """Construct participant ``c``'s ``MaskedUpdate`` (RFC-0011 §2): ``encode(Δ_c) + b_c + Σ±m_{c,c'}``.

    ``peer_public_keys`` maps every *other* participant id to its public key; the add-for-``c'>c`` /
    subtract-for-``c'<c`` sign rule is the lexicographic order on participant id. The encoded delta and
    masks live in ``[0, modulus)`` and the sum is taken mod ``modulus``; the result carries no recoverable
    ``Δ_c`` alone.
    """
    encoded = encode_delta(
        delta,
        field,
        participant_id=participant_id,
        round_index=round_index,
        dataset_root=dataset_root,
    ).masked.to(torch.int64)
    acc = encoded.clone()
    acc = (
        acc
        + expand_mask(
            self_seed(self_mask_secret, round_index), field.dim, field.modulus
        )
    ) % field.modulus
    for peer_id, peer_pk in peer_public_keys.items():
        shared = key_agreement(secret_key, peer_pk)
        mask = expand_mask(pairwise_seed(shared, round_index), field.dim, field.modulus)
        if peer_id > participant_id:  # add for c' > c
            acc = (acc + mask) % field.modulus
        else:  # subtract for c' < c
            acc = (acc - mask) % field.modulus
    return MaskedUpdate(
        participant_id=participant_id,
        round_index=round_index,
        masked=acc,
        dataset_root=dataset_root,
    )


# --- dropout-recovery data + the aggregator (RFC-0011 §4) ---


@dataclass(frozen=True)
class DropoutRecovery:
    """Shamir shares collected during the round, used only at reconstruction (RFC-0011 §4).

    The double-masking rule holds **exactly one** seed kind per participant: ``self_mask_shares`` for a
    survivor (its self-mask seed ``r_c``), ``pairwise_shares`` for a dropout (its DH private key ``sk_d``).
    A participant must never appear in both. ``threshold`` is ``t_agg`` — the minimum shares (and so
    survivors) needed to reconstruct any seed.
    """

    threshold: int
    self_mask_shares: "Mapping[str, Sequence[bytes]]"  # survivor c -> shares of r_c
    pairwise_shares: "Mapping[str, Sequence[bytes]]"  # dropout d -> shares of sk_d


class PairwiseMaskAggregator:
    """Bonawitz-style pairwise-mask secure aggregator (RFC-0011 §2/§4, Backend A — the default).

    Constructed with the round's routed **public** keys (the control plane's, RFC-0013). ``aggregate``
    reconstructs only the survivors' self-masks and the dropouts' pairwise masks (the double-masking rule)
    and returns **only** the plaintext ``Σ_c Δ_c`` — never an individual ``Δ_c`` (``INV-RESIDENCY``).
    """

    def __init__(self, public_keys: "Mapping[str, int]") -> None:
        self._public_keys = dict(public_keys)

    def aggregate(
        self,
        updates: "Mapping[str, MaskedUpdate]",
        *,
        field: FieldParams,
        round_index: int,
        threshold: int,
        recovery: DropoutRecovery,
    ) -> Tensor:
        """Reconstruct the fp32 plaintext ``Σ_{c in S} Δ_c`` over the surviving set ``S`` (RFC-0011 §2/§4).

        Below ``threshold`` survivors raises :class:`~lensemble.errors.SecureAggregationError`
        (``present``/``threshold``/``round``/``cause``) and returns no partial sum. Reconstruction sums the
        masked updates in fixed participant-id order, removes the survivors' self-masks ``Σ b_c`` and the
        uncancelled pairwise terms each survivor added against a dropout, then ``lift_signed`` recentres and
        divides by ``scale``. The integer field makes the sum order-independent (``INV-AGG-DETERMINISM``).
        """
        present = sorted(updates)
        if len(present) < threshold:
            err = SecureAggregationError(
                f"only {len(present)} survivors < threshold {threshold}; cannot reconstruct masks, "
                "refusing a partial sum",
                code=LensembleErrorCode.SECURE_AGG_FAILED,
                remediation="wait for the secure-aggregation threshold of survivors, or abort the round",
            )
            err.round = round_index  # type: ignore[attr-defined]
            err.present = len(present)  # type: ignore[attr-defined]
            err.threshold = threshold  # type: ignore[attr-defined]
            err.cause = "below_threshold"  # type: ignore[attr-defined]
            raise err

        # Sum the masked updates in fixed coordinate order (associative integer field).
        total = torch.zeros(field.dim, dtype=torch.int64)
        for pid in present:
            total = (total + updates[pid].masked.to(torch.int64)) % field.modulus

        # Remove the survivors' self-masks (reconstruct r_c from t shares -> b_c). Never their pairwise seeds.
        for c in present:
            r_c = shamir_reconstruct(list(recovery.self_mask_shares[c])[:threshold])
            b_c = expand_mask(self_seed(r_c, round_index), field.dim, field.modulus)
            total = (total - b_c) % field.modulus

        # Remove the uncancelled pairwise terms each survivor added against a dropout (reconstruct sk_d).
        for d, shares in recovery.pairwise_shares.items():
            sk_d = shamir_reconstruct(list(shares)[:threshold])
            for c in present:
                shared = key_agreement(sk_d, self._public_keys[c])  # KA(sk_d, pk_c)
                mask = expand_mask(
                    pairwise_seed(shared, round_index), field.dim, field.modulus
                )
                # survivor c added +m for (d > c) and -m for (d < c); undo that residual term.
                if d > c:
                    total = (total - mask) % field.modulus
                else:
                    total = (total + mask) % field.modulus

        signed = _lift_to_signed(total, field.modulus)
        return signed.to(torch.float32) / field.scale
