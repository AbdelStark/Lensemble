"""lensemble.data — the per-participant data layer (docs/rfcs/RFC-0004).

Episodes/transitions/windows are residency-bound (``INV-RESIDENCY``); only pseudo-gradients leave a
boundary. The on-disk backend adapters (``lance``/``hdf5``/``lerobot``) land here behind the
``EpisodeDataset.fmt`` selector and the ``register_adapter`` extension point (#22); the egress guard
lands with #23. Importing this package self-registers the three built-in adapters.
"""

from __future__ import annotations

from lensemble.data.adapters import load_episodes, register_adapter, save_episodes
from lensemble.data.dataset import EpisodeDataset
from lensemble.data.episode import Episode, Transition, Window
from lensemble.data.phase2 import (
    PHASE2_DATASET_SMOKE_SCHEMA_VERSION,
    Phase2ActionSpecEvidence,
    Phase2DatasetSmokeReport,
    Phase2SiloSmokeEvidence,
    build_phase2_dataset_smoke_report,
)
from lensemble.data.phase3 import (
    PHASE3_DATASET_REGISTRY_SCHEMA_VERSION,
    Phase3DatasetParticipantDeclaration,
    Phase3DatasetProbeRegistry,
    Phase3ProbeGovernance,
    load_phase3_dataset_registry,
    parse_phase3_dataset_registry,
    phase3_registry_from_consortium_manifest,
    to_phase3_dataset_registry_json,
    validate_coordinator_registry_preflight,
    validate_participant_registry_preflight,
    validate_phase3_dataset_registry,
    validate_phase3_registry_against_manifest,
    write_phase3_dataset_registry,
)
from lensemble.data.quality import DataQualityMetadata, validate_join_precondition
from lensemble.data.residency import EgressRole, guard_egress

__all__ = [
    "Transition",
    "Episode",
    "Window",
    "EpisodeDataset",
    "save_episodes",
    "load_episodes",
    "register_adapter",
    "guard_egress",
    "EgressRole",
    "DataQualityMetadata",
    "validate_join_precondition",
    "PHASE2_DATASET_SMOKE_SCHEMA_VERSION",
    "Phase2ActionSpecEvidence",
    "Phase2SiloSmokeEvidence",
    "Phase2DatasetSmokeReport",
    "build_phase2_dataset_smoke_report",
    "PHASE3_DATASET_REGISTRY_SCHEMA_VERSION",
    "Phase3ProbeGovernance",
    "Phase3DatasetParticipantDeclaration",
    "Phase3DatasetProbeRegistry",
    "phase3_registry_from_consortium_manifest",
    "parse_phase3_dataset_registry",
    "load_phase3_dataset_registry",
    "write_phase3_dataset_registry",
    "to_phase3_dataset_registry_json",
    "validate_phase3_dataset_registry",
    "validate_phase3_registry_against_manifest",
    "validate_participant_registry_preflight",
    "validate_coordinator_registry_preflight",
]
