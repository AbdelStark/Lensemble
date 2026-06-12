"""Browser demo support surfaces for Lensemble."""

from lensemble.demo.evidence_audit import (
    audit_real_lewm_evidence,
    require_clean_evidence,
)
from lensemble.demo.federated import (
    CLAIM_BOUNDARY,
    LEWM_CLAIM_BOUNDARY,
    FederatedDemoError,
    FederatedDemoService,
)

__all__ = [
    "CLAIM_BOUNDARY",
    "LEWM_CLAIM_BOUNDARY",
    "FederatedDemoError",
    "FederatedDemoService",
    "audit_real_lewm_evidence",
    "require_clean_evidence",
]
