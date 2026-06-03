"""lensemble.verify — Phase-2 verifiable layer + public recomputation (docs/rfcs/RFC-0006)."""

from __future__ import annotations

from .recompute import (
    AlignmentClaim,
    AlignmentRecomputation,
    parse_alignment_claim,
    parse_alignment_recomputation,
    procrustes_q_hash,
    recompute_alignment,
    recompute_alignment_claim,
)

__all__ = [
    "AlignmentClaim",
    "AlignmentRecomputation",
    "parse_alignment_claim",
    "parse_alignment_recomputation",
    "procrustes_q_hash",
    "recompute_alignment",
    "recompute_alignment_claim",
]
