"""lensemble.verify.recompute — Phase-1 free public recomputation of frame alignment (RFC-0006 §4).

The one Phase-2 verification mechanism that ships in Phase 1 because it costs nothing: a deterministic,
side-effect-free recomputation of the committed model's frame alignment to the reference frame, from the
*public* probe plus the *committed* weights — so anyone can recompute and check it without a ZK proof
(RFC-0006 §4). It re-runs the same closed-form orthogonal Procrustes alignment
``Q* = V Uᵀ`` from the SVD ``E_refᵀ f_θ(P) = U Σ Vᵀ`` (RFC-0002 §4/§5) on the fp32/fp64 path.

What it MEASURES (and what it does NOT, the #18 caveat). :func:`recompute_alignment` measures the
committed model ``f_θ``'s frame alignment to the round-0 reference frame ``f_ref`` on the pinned probe:
it recovers ``Q*``, its rotation angle, and the Procrustes residual, and emits a reproducible
:class:`~lensemble.gauge.drift.FrameDriftReport`. It does **NOT** verify that the Layer-3 re-alignment
backstop was *applied* to the committed weights, because — per the recorded #18 decision — that backstop
rotates the contribution in **activation space** at aggregation, not as a fold into the committed
weights. So a committed checkpoint carries no weight-level evidence of the backstop; what is publicly
recomputable and checkable here is the *measured alignment* of the committed model to the reference, the
honest free guarantee (RFC-0006 §4 / RFC-0002 §5).

Determinism (``INV-AGG-DETERMINISM``): the recomputation is bitwise-deterministic given (committed
weights, pinned probe, reference targets). The ``procrustes_q_hash`` — a SHA-256 over the canonical
little-endian bytes of ``Q*`` at its fp32/fp64 work dtype — is the cross-process / cross-platform
verifiability key: two independent recomputations agree on it bit-for-bit (RFC-0006 §4 testing strategy,
doubling as the RFC-0015 diagnostic-reproducibility property).

Preconditions, fail-closed (never swallowed): the checkpoint content hash is verified on load
(``INV-CHECKPOINT-HASH`` → :class:`~lensemble.errors.CheckpointIntegrityError`); a too-new checkpoint
schema raises :class:`~lensemble.errors.SchemaVersionMismatch`; the probe content hash is recomputed and
checked against its pinned ``content_hash`` (``INV-PROBE-PIN`` → :class:`~lensemble.errors.ProbeError`);
a non-self-describing checkpoint (``header.model_arch is None``) fails closed via ``Encoder.from_header``;
and a near-degenerate probe embedding raises :class:`~lensemble.errors.DegenerateProcrustes` rather than
yield a silent garbage ``Q*`` (RFC-0006 §4).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from lensemble.artifacts.checkpoint import load_checkpoint
from lensemble.data.probe import load_probe, probe_content_hash
from lensemble.errors import LensembleErrorCode, ProbeError, SchemaVersionMismatch
from lensemble.gauge.drift import (
    FRAME_DRIFT_SCHEMA_VERSION,
    FrameDriftReport,
    _rotation_angle_deg,
    frame_drift,
)
from lensemble.gauge.procrustes import procrustes_align
from lensemble.model.encoder import Encoder

if TYPE_CHECKING:
    from lensemble.data.probe import PublicProbe

# The on-disk record schema version for the alignment claim / recomputation records (RFC-0006 §4).
ALIGNMENT_SCHEMA_VERSION = 1

# The reserved participant id under which the committed model's recovered alignment to the reference frame
# is reported in the FrameDriftReport (the "drift_from_global" entry), paired against the reference frame.
_COMMITTED_KEY = "committed"
_ENCODER_PREFIX = "encoder."

# fp32/fp64 comparison tolerance for matching a recomputed residual / rotation angle to an expected claim
# (RFC-0006 §4: "within the fp32/fp64 tolerance"). The procrustes_q_hash is matched EXACTLY (bitwise).
_MATCH_ATOL = 1e-4


class AlignmentClaim(BaseModel):
    """The coordinator's published frame-alignment claim for a round (RFC-0006 §4 schema).

    Records the committed model's recovered alignment to the round-0 reference frame on the pinned probe:
    the canonical ``procrustes_q_hash`` of ``Q*`` (the cross-process verifiability key), the Frobenius
    ``procrustes_residual`` ``‖f_θ(P)·Q* − E_ref‖_F`` (RFC-0002 §5), the recovered ``rotation_angle_deg``,
    and the pinned ``probe_hash`` (``INV-PROBE-PIN``). Frozen; unknown fields rejected; ``schema_version``
    gated first by :func:`parse_alignment_claim`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = ALIGNMENT_SCHEMA_VERSION
    round_index: int
    procrustes_q_hash: (
        str  # SHA-256 (hex) of canonical bytes of Q* (the claimed O(d) alignment)
    )
    procrustes_residual: float  # ‖f_θ(P)·Q* − E_ref‖_F (RFC-0002 §5)
    rotation_angle_deg: float  # recovered principal rotation angle of Q* on the probe
    probe_hash: str  # the pinned probe these embeddings came from (INV-PROBE-PIN), hex


class AlignmentRecomputation(BaseModel):
    """A locally recomputed frame-alignment record, optionally checked against a claim (RFC-0006 §4).

    ``recomputed`` is the :class:`AlignmentClaim` computed from public inputs alone. When an ``expected``
    claim was supplied, ``matches_expected`` is ``True`` iff the ``procrustes_q_hash`` matches EXACTLY
    (the bitwise determinism property) **and** the residual / rotation angle agree within the fp32/fp64
    tolerance; ``max_abs_residual_delta`` records ``|recomputed.residual − expected.residual|``. Both are
    ``None`` when no ``expected`` claim was given. Frozen; ``schema_version`` gated first by
    :func:`parse_alignment_recomputation`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = ALIGNMENT_SCHEMA_VERSION
    round_index: int
    probe_hash: str  # must equal the pinned commitment, else ProbeError refused the run
    recomputed: AlignmentClaim  # locally recomputed from public inputs alone
    matches_expected: bool | None = None  # None if `expected` was not supplied
    max_abs_residual_delta: float | None = (
        None  # |recomputed.residual − expected.residual|
    )


def procrustes_q_hash(q_star: Tensor) -> str:
    """SHA-256 (64 hex) over the canonical little-endian bytes of ``Q*`` (the verifiability key).

    Platform-stable, mirroring :func:`lensemble.aggregation.determinism.flat_content_hash`: ``Q*`` is
    detached, moved to CPU, made contiguous, and viewed little-endian at its stored fp32/fp64 work dtype,
    so the hash is byte-identical on a big- or little-endian host and across processes (RFC-0006 §4).
    """
    array = q_star.detach().cpu().contiguous().numpy()
    little_endian = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return hashlib.sha256(little_endian.tobytes()).hexdigest()


def _load_committed_encoder(committed_weights: Path) -> tuple[Encoder, int]:
    """Load + hash-verify the committed checkpoint and rebuild ``f_θ`` from its self-describing header.

    Propagates :class:`~lensemble.errors.CheckpointIntegrityError` (tamper) /
    :class:`~lensemble.errors.SchemaVersionMismatch` (too-new) from ``load_checkpoint``, and the clear
    :class:`~lensemble.errors.ArtifactError` from ``Encoder.from_header`` when the checkpoint is not
    self-describing (``header.model_arch is None``) — none is caught here. Returns ``(eval-mode encoder,
    round_index)``.
    """
    weights, header = load_checkpoint(Path(committed_weights))
    encoder = Encoder.from_header(header).eval()
    encoder_state = {
        name[len(_ENCODER_PREFIX) :]: tensor
        for name, tensor in weights.items()
        if name.startswith(_ENCODER_PREFIX)
    }
    encoder.load_state_dict(encoder_state, strict=True)
    return encoder, header.round_index


def _verify_probe_pin(probe: "PublicProbe") -> str:
    """INV-PROBE-PIN: recompute the probe content hash and refuse on a mismatch with its stored pin.

    A probe whose recomputed ``probe_content_hash(points, landmark_idx)`` differs from its stored
    ``content_hash`` is rejected with :class:`~lensemble.errors.ProbeError` (``PROBE_INVALID``) — the
    recomputation never runs against a probe that does not match its own pin. Returns the verified hex hash.
    """
    recomputed = probe_content_hash(probe.points, probe.landmark_idx)
    if recomputed != probe.content_hash:
        raise ProbeError(
            "probe content hash does not match its pinned content_hash; refusing to recompute alignment "
            "against an unpinned probe (a probe change is a re-anchoring event, RFC-0004 §3.1)",
            code=LensembleErrorCode.PROBE_INVALID,
            remediation="re-pin the probe to its committed content hash (INV-PROBE-PIN)",
        )
    return recomputed.hex()


@torch.no_grad()
def _embed_landmarks(encoder: Encoder, probe: "PublicProbe") -> Tensor:
    """Compute ``f_θ(landmarks)`` of shape ``(k, N, d)`` from the committed encoder, then flatten to (k·N, d)."""
    landmarks = probe.points[probe.landmark_idx]
    tokens = encoder(landmarks).tokens  # (k, N, d)
    return tokens.reshape(-1, tokens.shape[-1])  # (k·N, d)


def recompute_alignment(committed_weights: Path, probe: Path) -> FrameDriftReport:
    """Publicly recompute the committed model's frame alignment to the reference (RFC-0006 §4; SPEC 02 §1.8).

    The frozen ``1.0`` signature ``recompute_alignment(committed_weights: Path, probe: Path) ->
    FrameDriftReport``. Deterministic given (committed weights, pinned probe): reconstructs ``f_θ`` from
    the self-describing checkpoint header (#171), recomputes ``f_θ(P)`` and the closed-form Procrustes
    alignment ``Q*`` to the probe's pinned reference targets ``E_ref = f_ref(P)``, and returns a
    :class:`~lensemble.gauge.drift.FrameDriftReport` whose ``drift_from_global['committed']`` is the
    recovered rotation angle of the committed model to the reference frame.

    The #18 caveat: this MEASURES the committed model's alignment to the reference frame; it does NOT
    verify the Layer-3 re-alignment backstop was applied, because that backstop rotates in activation
    space at aggregation (the recorded #18 decision), not as a fold into the committed weights.

    Fail-closed preconditions (never swallowed): ``CheckpointIntegrityError`` (committed-weights tamper) /
    ``SchemaVersionMismatch`` (too-new checkpoint) from the load; ``ArtifactError`` from
    ``Encoder.from_header`` on a non-self-describing checkpoint; ``ProbeError`` (``PROBE_INVALID``) on a
    probe whose recomputed content hash differs from its pin (``INV-PROBE-PIN``); and
    ``DegenerateProcrustes`` (``PROCRUSTES_DEGENERATE``) on a rank-deficient probe embedding — never a
    silent garbage ``Q*`` (RFC-0006 §4).
    """
    encoder, round_index = _load_committed_encoder(Path(committed_weights))
    public_probe = load_probe(Path(probe))
    probe_hash = _verify_probe_pin(public_probe)

    f_theta_flat = _embed_landmarks(encoder, public_probe)  # (k·N, d)
    e_ref = public_probe.landmark_targets  # (k, N, d) = f_ref(landmarks), pinned
    e_ref_flat = e_ref.reshape(-1, e_ref.shape[-1])  # (k·N, d)

    # Reuse `frame_drift` so `drift_from_global['committed']` is the recovered rotation angle to the
    # reference frame and the INV-PROBE-PIN / DegenerateProcrustes paths are exercised by the same
    # primitive. The reserved "global" key carries the reference targets E_ref; the committed model is the
    # one participant. The header round_index and the verified probe hash populate the report.
    return frame_drift(
        {_COMMITTED_KEY: f_theta_flat, "global": e_ref_flat},
        round_index=round_index,
        probe=public_probe,
        expected_probe_hash=probe_hash,
    )


def recompute_alignment_claim(
    committed_weights: Path,
    probe: Path,
    *,
    expected: AlignmentClaim | None = None,
) -> AlignmentRecomputation:
    """Recompute the alignment as an :class:`AlignmentRecomputation` record, optionally checking a claim.

    Runs the same deterministic recomputation as :func:`recompute_alignment` but fills the richer on-disk
    record of RFC-0006 §4: the recovered ``Q*``'s canonical ``procrustes_q_hash`` (the verifiability key),
    the Procrustes ``residual``, the ``rotation_angle_deg``, and the pinned ``probe_hash``. When
    ``expected`` is supplied, ``matches_expected`` is ``True`` iff the ``procrustes_q_hash`` matches
    EXACTLY **and** the residual / rotation angle agree within the fp32/fp64 tolerance, and
    ``max_abs_residual_delta`` records ``|recomputed.residual − expected.residual|`` (RFC-0006 §4). Shares
    every fail-closed precondition of :func:`recompute_alignment`.
    """
    encoder, round_index = _load_committed_encoder(Path(committed_weights))
    public_probe = load_probe(Path(probe))
    probe_hash = _verify_probe_pin(public_probe)

    f_theta_flat = _embed_landmarks(encoder, public_probe)  # (k·N, d)
    e_ref = public_probe.landmark_targets
    e_ref_flat = e_ref.reshape(-1, e_ref.shape[-1])  # (k·N, d)

    # `procrustes_align(source, target)` finds Q* mapping source onto target; the committed model's
    # embedding is the source, the pinned reference targets the target — DegenerateProcrustes propagates.
    q_star, residual = procrustes_align(f_theta_flat, e_ref_flat)
    recomputed = AlignmentClaim(
        round_index=round_index,
        procrustes_q_hash=procrustes_q_hash(q_star),
        procrustes_residual=residual,
        rotation_angle_deg=_rotation_angle_deg(q_star),
        probe_hash=probe_hash,
    )

    matches_expected: bool | None = None
    max_abs_residual_delta: float | None = None
    if expected is not None:
        max_abs_residual_delta = abs(
            recomputed.procrustes_residual - expected.procrustes_residual
        )
        matches_expected = (
            recomputed.procrustes_q_hash == expected.procrustes_q_hash
            and max_abs_residual_delta <= _MATCH_ATOL
            and abs(recomputed.rotation_angle_deg - expected.rotation_angle_deg)
            <= _MATCH_ATOL
        )

    return AlignmentRecomputation(
        round_index=round_index,
        probe_hash=probe_hash,
        recomputed=recomputed,
        matches_expected=matches_expected,
        max_abs_residual_delta=max_abs_residual_delta,
    )


def _gate_schema_version(raw: dict[str, object], record: str) -> int:
    """Gate an on-disk record's ``schema_version`` FIRST (mirrors ``parse_eval_report``).

    A non-integer or a version exceeding :data:`ALIGNMENT_SCHEMA_VERSION` raises
    :class:`~lensemble.errors.SchemaVersionMismatch` (``SCHEMA_VERSION_MISMATCH``) before any field
    validation — an unknown/too-new claim record is rejected, not silently coerced (RFC-0006 §4).
    """
    version = raw.get("schema_version")
    if not isinstance(version, int) or version > ALIGNMENT_SCHEMA_VERSION:
        raise SchemaVersionMismatch(
            f"{record} schema_version {version!r} exceeds reader max {ALIGNMENT_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation=f"read with a build supporting schema_version <= {ALIGNMENT_SCHEMA_VERSION}",
        )
    return version


def parse_alignment_claim(raw: dict[str, object]) -> AlignmentClaim:
    """Parse a raw dict back to an :class:`AlignmentClaim`; gate ``schema_version`` first (RFC-0006 §4)."""
    _gate_schema_version(raw, "alignment-claim")
    return AlignmentClaim.model_validate(raw)


def parse_alignment_recomputation(raw: dict[str, object]) -> AlignmentRecomputation:
    """Parse a raw dict back to an :class:`AlignmentRecomputation`; gate ``schema_version`` first."""
    _gate_schema_version(raw, "alignment-recomputation")
    return AlignmentRecomputation.model_validate(raw)


__all__ = [
    "ALIGNMENT_SCHEMA_VERSION",
    "FRAME_DRIFT_SCHEMA_VERSION",
    "AlignmentClaim",
    "AlignmentRecomputation",
    "procrustes_q_hash",
    "recompute_alignment",
    "recompute_alignment_claim",
    "parse_alignment_claim",
    "parse_alignment_recomputation",
]
