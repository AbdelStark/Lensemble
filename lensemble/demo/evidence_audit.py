"""lensemble.demo.evidence_audit — claim audit for real-mode demo evidence (#323, gate G7).

Validates a ``demo-evidence/1`` export from a ``real-lewm-tworooms`` run against the Tapestry-like
claim boundary of docs/roadmap/TAPESTRY_LEWM.md: hash-bound checkpoint/export/revision references,
the required claim-boundary and non-claim phrases, honest privacy status, health-flag presence,
and residency (no raw participant data, adapter tensors, or tokens anywhere in the export).

``audit_real_lewm_evidence`` returns the list of violations (empty == clean); callers that need
fail-closed behavior use ``require_clean_evidence``. The audit REJECTS evidence that overclaims:
production/browser-training/paper-scale phrasings outside a negation are violations too.
"""

from __future__ import annotations

import json
import re
from typing import Any

from lensemble.demo.federated import (
    EVIDENCE_SCHEMA,
    LEWM_UPDATE_SCHEMA,
    REAL_LEWM_MODE,
    FederatedDemoError,
)

__all__ = ["audit_real_lewm_evidence", "require_clean_evidence", "REQUIRED_PHRASES"]

# the claim boundary the card/evidence must carry, and the claims it must NOT make
REQUIRED_PHRASES = (
    "Tapestry-like",
    "bounded",
    "not",  # the non-claim text must actually negate something
)
_REQUIRED_NONCLAIM_FRAGMENTS = (
    "from-scratch",
    "production browser training",
    "paper-scale",
)
# overclaim patterns: these phrases may only appear inside a negation ("not", "no", "never")
_OVERCLAIM_PATTERNS = (
    re.compile(r"benchmark (win|parity) (over|with)", re.IGNORECASE),
    re.compile(r"cryptographic proof of honest computation(?! )", re.IGNORECASE),
)

_HASH64 = re.compile(r"^[0-9a-f]{64}$")
_HASH40 = re.compile(r"^[0-9a-f]{40}$")

_FORBIDDEN_EVIDENCE_SUBSTRINGS = (
    '"delta"',
    '"adapterState"',
    '"pixels"',
    '"frames"',
    "ptok-",
)


def _is_hash64(value: object) -> bool:
    return isinstance(value, str) and bool(_HASH64.match(value))


def audit_real_lewm_evidence(evidence: dict[str, Any]) -> list[str]:
    """Return every claim-boundary/binding/residency violation in the evidence export."""
    violations: list[str] = []

    if evidence.get("schema") != EVIDENCE_SCHEMA:
        violations.append(f"schema must be {EVIDENCE_SCHEMA}")
    if evidence.get("runMode") != REAL_LEWM_MODE:
        violations.append(f"runMode must be {REAL_LEWM_MODE}")

    # --- claim boundary / non-claims ---
    non_claim = str(evidence.get("nonClaimText", ""))
    for phrase in REQUIRED_PHRASES:
        if phrase not in non_claim:
            violations.append(f"nonClaimText is missing the required phrase {phrase!r}")
    for fragment in _REQUIRED_NONCLAIM_FRAGMENTS:
        if fragment not in non_claim:
            violations.append(f"nonClaimText must negate {fragment!r}")

    # --- checkpoint/export binding ---
    binding = evidence.get("lewmBinding") or {}
    checkpoint = binding.get("checkpoint") or {}
    if not isinstance(checkpoint.get("revision"), str) or not _HASH40.match(
        str(checkpoint.get("revision", ""))
    ):
        violations.append("lewmBinding.checkpoint.revision must be a pinned 40-hex revision")
    if not _is_hash64(checkpoint.get("weightsSha256")):
        violations.append("lewmBinding.checkpoint.weightsSha256 must be a 64-hex hash")
    graph_hashes = binding.get("exportGraphHashes") or {}
    if not graph_hashes or not all(_is_hash64(v) for v in graph_hashes.values()):
        violations.append("lewmBinding.exportGraphHashes must be non-empty 64-hex hashes")
    if not binding.get("adapterSpec"):
        violations.append("lewmBinding.adapterSpec must describe the trainable subset")

    # --- privacy honesty ---
    privacy = evidence.get("privacy") or {}
    for key in ("secureAggregation", "differentialPrivacy"):
        status = str(privacy.get(key, ""))
        if "absent" not in status and "simulated" not in status and "enabled" not in status:
            violations.append(
                f"privacy.{key} must state exactly what is absent/simulated/enabled"
            )

    # --- hash-bound revisions and updates ---
    revisions = evidence.get("modelRevisions") or []
    update_hashes = evidence.get("updateHashes") or []
    if not all(_is_hash64(h) for h in update_hashes):
        violations.append("updateHashes must all be 64-hex hashes")
    seen_revisions = {"initial"}
    for revision in revisions:
        rid = str(revision.get("modelRevisionId", ""))
        if not _is_hash64(revision.get("sha256")):
            violations.append(f"model revision {rid} is missing its 64-hex sha256")
        parent = str(revision.get("parentModelRevisionId", ""))
        if parent not in seen_revisions:
            violations.append(f"model revision {rid} has unbound parent {parent!r}")
        seen_revisions.add(rid)
        sources = revision.get("sourceUpdateHashes") or []
        if not sources or not all(h in update_hashes for h in sources):
            violations.append(f"model revision {rid} source updates are not in updateHashes")
        base = (revision.get("baseCheckpoint") or {}).get("revision")
        if base != checkpoint.get("revision"):
            violations.append(f"model revision {rid} is not bound to the pinned checkpoint")

    # --- metrics honesty: every closed round carries real summaries and health flags ---
    for metric in evidence.get("roundMetrics") or []:
        round_index = metric.get("round")
        if "healthFlags" not in metric or "healthy" not in metric:
            violations.append(f"round {round_index} metrics are missing health flags")
        for key in ("predLossLastMean", "sigregStatisticMean", "effectiveRankMean"):
            if not isinstance(metric.get(key), (int, float)):
                violations.append(f"round {round_index} metrics are missing {key}")

    # --- participant update metadata stays bounded ---
    for participant in evidence.get("participants") or []:
        for round_key, metadata in (participant.get("updateMetadata") or {}).items():
            if metadata.get("schema") != LEWM_UPDATE_SCHEMA:
                violations.append(
                    f"participant {participant.get('id')} round {round_key} update is not "
                    f"{LEWM_UPDATE_SCHEMA}"
                )
            if not _is_hash64(metadata.get("hash")):
                violations.append(
                    f"participant {participant.get('id')} round {round_key} update hash invalid"
                )

    # --- residency: nothing raw, no tensors, no tokens ---
    encoded = json.dumps(evidence)
    for needle in _FORBIDDEN_EVIDENCE_SUBSTRINGS:
        if needle in encoded:
            violations.append(f"evidence contains forbidden content {needle!r}")

    # --- overclaim scan over the whole export ---
    for pattern in _OVERCLAIM_PATTERNS:
        match = pattern.search(encoded)
        if match:
            start = max(0, match.start() - 60)
            context = encoded[start : match.end()].lower()
            if "not " not in context and "no " not in context and "never" not in context:
                violations.append(f"unnegated overclaim {match.group(0)!r}")

    return violations


def require_clean_evidence(evidence: dict[str, Any]) -> None:
    violations = audit_real_lewm_evidence(evidence)
    if violations:
        raise FederatedDemoError(
            "claim_audit_failed",
            "; ".join(violations[:8]),
            status=422,
        )
