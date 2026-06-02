"""lensemble.verify.stark — the Phase-2 prover/verifier seam (RFC-0006 §7 roadmap). NOT built in Phase 1.

This module reserves the entry-point signatures for the Phase-2 verifiable layer — the Stwo Circle-STARK
proof of correct outer-step aggregation — so that adding the real prover in **Stage D (post-v1.0)** is
additive and needs no protocol or format rework. Every entry point here is a stub that raises Python's
built-in :class:`NotImplementedError` (a deliberate non-``LensembleError`` case: the operation is
*absent*, not *failed*) with a remediation pointing at the free Phase-1 public-recomputation path
(``lensemble verify recompute``, RFC-0006 §4) and the Stage-D roadmap (RFC-0006 §7).

These names are intentionally **outside** the frozen 1.0 public surface (02 §1) — they are reached as
``lensemble.verify.stark.*``, not re-exported from ``lensemble.verify`` — so the real prover can land in
Stage D without a SemVer break. The exact Phase-2 prover/verifier signature and circuit are an Open
Question (02 §6 / RFC-0006 Open Questions); this module reserves the seam, not the final shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

# The remediation surfaced by every stub and by `lensemble verify prove`. Names the phase, the free
# Phase-1 path (`recompute`), and the roadmap so the message alone tells a user what to do instead.
STAGE_D_REMEDIATION = (
    "the verifiable-contribution prover is a Phase-2 capability (Stage D, post-v1.0) and is not built in "
    "this release; use `lensemble verify recompute` for the free Phase-1 public recomputation of the "
    "frame alignment (RFC-0006 §4). Roadmap: RFC-0006 §7."
)


def prove_outer_step(
    prior_global: "Path", committed_global: "Path", *, round_index: int
) -> bytes:
    """Phase-2 STARK proof that ``committed_global`` is the correct outer step from ``prior_global``.

    The prover over the outer-step circuit (the deterministic Nesterov aggregation,
    ``INV-AGG-DETERMINISM``) — Stage D. Raises :class:`NotImplementedError` in Phase 1.
    """
    raise NotImplementedError(STAGE_D_REMEDIATION)


def verify_outer_step_proof(
    proof: bytes, prior_global: "Path", committed_global: "Path", *, round_index: int
) -> bool:
    """Phase-2 verification of a :func:`prove_outer_step` proof against the public inputs — Stage D.

    Raises :class:`NotImplementedError` in Phase 1.
    """
    raise NotImplementedError(STAGE_D_REMEDIATION)


def prove_round(cfg: Any) -> bytes:
    """Config-driven Phase-2 proof of a round's outer-step correctness — the ``verify prove`` CLI seam.

    Resolves the round's committed artifacts from ``cfg`` and calls the prover — Stage D. Raises
    :class:`NotImplementedError` in Phase 1; the CLI maps the raise to a non-zero exit (02 §4).
    """
    raise NotImplementedError(STAGE_D_REMEDIATION)
