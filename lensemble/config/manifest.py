"""lensemble.config.manifest — the RunManifest and the canonical config_hash (RFC-0009 6/7).

One machine-readable reproducibility record per run: it fingerprints the exact inputs (resolved config,
seed lineage, source state, environment, dependency versions, probe/dataset commitments) so a third
party can reproduce — or audit — the run. Written to the run directory *before* training starts, so a
crashed run still leaves a record.

``config_hash`` (RFC-0009 7) is SHA-256 over a *canonical* byte serialization of the resolved config:
(1) resolve to a plain dict, (2) drop the fixed non-semantic allowlist (output sinks, which do not change
the computation; a new field defaults to *included* so omissions fail safe), (3)
``json.dumps(sort_keys=True, separators=(",",":"), ensure_ascii=True)``, (4) SHA-256 hex. The algorithm
id ``sha256-canon-v1`` is recorded in ``env["config_hash_algo"]`` so a later migration to a
STARK-friendly hash is a versioned, forward-compatible change.

Residency (``INV-RESIDENCY``). The manifest is a boundary-crossing artifact: it carries hashes, seeds,
versions, counts — never a raw observation/action/private embedding. :func:`write_manifest` routes the
free-form ``config_resolved`` through the redaction guard (RFC-0015), which fails closed on a tensor-like
leaf; ``extra="forbid"`` blocks any field outside the schema (and so any accidental secret).
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lensemble.config.schema import LensembleConfig
from lensemble.config.seed import SEED_DERIVATION, derive, round_sketch_seed
from lensemble.errors import ConfigError, LensembleErrorCode, SchemaVersionMismatch
from lensemble.observability import redact

MANIFEST_SCHEMA_VERSION = 1
CONFIG_HASH_ALGO = "sha256-canon-v1"

# Fixed, documented non-semantic exclusion allowlist (RFC-0009 7 step 2): output sinks do not change the
# computation. Dot-paths into the resolved tree. A NEW field defaults to *included* (fail-safe).
_NON_SEMANTIC_FIELDS: tuple[str, ...] = (
    "observability.log_path",
    "observability.metrics_path",
)

# Pinned dependencies whose versions are fingerprinted (conventions 11); absent ones are skipped.
_FINGERPRINT_DEPS: tuple[str, ...] = (
    "torch",
    "numpy",
    "safetensors",
    "pydantic",
    "omegaconf",
    "hydra-core",
    "blake3",
    "lance",
    "h5py",
)
# The seed-lineage components (conventions 9): one root seed derives each.
_SEED_COMPONENTS: tuple[str, ...] = ("python", "numpy", "torch", "cuda")


class RunManifest(BaseModel):
    """Per-run reproducibility record (RFC-0009 6). Frozen; unknown fields are rejected (no secrets)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=MANIFEST_SCHEMA_VERSION)
    config_hash: str  # SHA-256 over the canonical resolved-config bytes (7)
    config_resolved: dict[
        str, Any
    ]  # the fully-resolved config tree (for re-instantiation)
    root_seed: int
    component_seeds: dict[
        str, int
    ]  # {python,numpy,torch,cuda} = derive(root_seed, lib)
    round_sketch_seeds: dict[int, int]  # t -> s_t = round_sketch_seed(root_seed, t)
    git_sha: str  # repo commit; "+dirty" suffix if the tree is not clean
    env: dict[
        str, str
    ]  # python/torch/CUDA/OS/seed_derivation/determinism + config_hash_algo
    dependency_versions: dict[str, str]
    probe_hash: str | None = None  # pinned probe content hash (INV-PROBE-PIN)
    dataset_roots: dict[str, str] = Field(
        default_factory=dict
    )  # participant -> R_c (RFC-0014)
    wmcp_version: str  # latent-contract version in force (INV-WMCP)
    run_mode: str  # train_local | coordinator | participant | eval
    created_at: datetime


def _json_native(tree: dict[str, Any]) -> dict[str, Any]:
    """Round-trip through JSON so the tree contains only JSON-native types (tuples -> lists)."""
    return json.loads(json.dumps(tree))


def _drop_non_semantic(tree: dict[str, Any]) -> dict[str, Any]:
    pruned = copy.deepcopy(tree)
    for dotted in _NON_SEMANTIC_FIELDS:
        parts = dotted.split(".")
        node: Any = pruned
        for key in parts[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(parts[-1], None)
    return pruned


def config_hash(config_resolved: dict[str, Any]) -> str:
    """SHA-256 over the canonical bytes of the resolved config (RFC-0009 7; algo ``sha256-canon-v1``).

    Drops the fixed non-semantic allowlist, then ``json.dumps(sort_keys, compact, ascii)`` -> SHA-256 hex.
    Identical across hosts/Python builds for the same semantic config.
    """
    pruned = _drop_non_semantic(_json_native(config_resolved))
    canonical = json.dumps(
        pruned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git_sha() -> str:
    """``git rev-parse HEAD`` with a ``+dirty`` suffix when the tree is not clean; ``unknown`` if no git."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{head}+dirty" if status else head
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for dep in _FINGERPRINT_DEPS:
        try:
            versions[dep] = metadata.version(dep)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _env(cfg: LensembleConfig) -> dict[str, str]:
    import torch

    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda) if torch.version.cuda else "none",  # type: ignore[attr-defined]
        "cuda_available": str(torch.cuda.is_available()),
        "os": platform.platform(),
        "seed_derivation": SEED_DERIVATION,
        "deterministic_aggregation": str(cfg.determinism.deterministic_aggregation),
        "deterministic_inner": str(cfg.determinism.deterministic_inner),
        "config_hash_algo": CONFIG_HASH_ALGO,
    }


def build_manifest(
    cfg: LensembleConfig,
    *,
    run_mode: str | None = None,
    probe_hash: str | None = None,
    dataset_roots: dict[str, str] | None = None,
    created_at: datetime | None = None,
) -> RunManifest:
    """Generate a :class:`RunManifest` from a resolved config (never hand-authored; RFC-0009 6).

    Records the full seed lineage, source/env/dependency fingerprint, and the canonical ``config_hash``.
    ``run_mode`` defaults to ``cfg.run_mode``; ``created_at`` defaults to the current UTC time.
    """
    resolved = _json_native(_config_to_dict(cfg))
    root_seed = cfg.determinism.root_seed
    return RunManifest(
        config_hash=config_hash(resolved),
        config_resolved=resolved,
        root_seed=root_seed,
        component_seeds={lib: derive(root_seed, lib) for lib in _SEED_COMPONENTS},
        round_sketch_seeds={
            t: round_sketch_seed(root_seed, t) for t in range(cfg.federation.num_rounds)
        },
        git_sha=_git_sha(),
        env=_env(cfg),
        dependency_versions=_dependency_versions(),
        probe_hash=probe_hash,
        dataset_roots=dict(dataset_roots or {}),
        wmcp_version=cfg.model.wmcp_version,
        run_mode=run_mode or cfg.run_mode,
        created_at=created_at or datetime.now(timezone.utc),
    )


def _config_to_dict(cfg: LensembleConfig) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(cfg)


def to_json(manifest: RunManifest) -> str:
    """Canonical JSON for a manifest (sorted keys, compact, ASCII) — stable across hosts."""
    return json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def write_manifest(manifest: RunManifest, path: Path) -> Path:
    """Vet residency, then write the manifest as canonical JSON; returns the written path.

    The free-form ``config_resolved`` is routed through the redaction guard (``INV-RESIDENCY``): a
    tensor-like leaf raises :class:`~lensemble.errors.ResidencyViolation`, fail-closed, before any write.
    """
    redact(manifest.config_resolved, field="config_resolved")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(manifest), encoding="utf-8")
    return path


def load_manifest(path: Path) -> RunManifest:
    """Load and verify a manifest.

    Raises :class:`~lensemble.errors.SchemaVersionMismatch` if ``schema_version`` exceeds this reader's
    max (never a best-effort parse), and :class:`~lensemble.errors.ConfigError` if ``config_resolved``
    does not re-hash to the stored ``config_hash`` (the config is not what the manifest claims).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    version = raw.get("schema_version")
    if not isinstance(version, int) or version > MANIFEST_SCHEMA_VERSION:
        raise SchemaVersionMismatch(
            f"manifest schema_version {version!r} exceeds reader max {MANIFEST_SCHEMA_VERSION}",
            code=LensembleErrorCode.SCHEMA_VERSION_MISMATCH,
            remediation=f"read with a build supporting schema_version <= {version!r}, or migrate",
        )
    manifest = RunManifest.model_validate(raw)  # extra="forbid" rejects unknown fields
    if config_hash(manifest.config_resolved) != manifest.config_hash:
        raise ConfigError(
            "config_resolved does not reproduce the stored config_hash",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="the manifest's config tree was altered; do not trust its config_hash",
        )
    return manifest
