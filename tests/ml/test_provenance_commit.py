"""DatasetCommitment, commit_dataset, and the Δ_c↔R_c binding (RFC-0014 §4, §6; #29).

Security-critical surface (`INV-COMMIT-BINDING`): `lensemble.provenance.commit` is held to 100% coverage
by the CI gate (07 §8), so every branch — the deterministic commit, the empty/conflicting-dataset
failures, the fail-closed binding, and the schema-version gate — is exercised here. Exact byte equality,
no numerical tolerance. The build helper is module-level so the determinism test re-imports it by file
path in a subprocess (a cross-process `R_c` check). Tests live under `tests/ml/` so the CI gate runs them
(the issue's illustrative `tests/provenance/` path is collected by no gate).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
from lensemble.data.dataset import EpisodeDataset
from lensemble.data.episode import Episode, Transition
from lensemble.errors import (
    CommitmentMismatch,
    MerkleVerificationError,
    ProvenanceError,
    SchemaVersionMismatch,
)
from lensemble.federation.pseudogradient import PseudoGradient
from lensemble.provenance.commit import (
    DatasetCommitment,
    commit_dataset,
    parse_dataset_commitment,
    verify_binding,
)
from lensemble.provenance.merkle import (
    DIGEST_SIZE,
    CommitmentScheme,
    HashDomain,
    _h,
    episode_leaf_hash,
    merkle_root,
)

_SCHEME = CommitmentScheme()


def _spec(
    embodiment_id: str = "so101-arm-7dof", wmcp: str = WMCP_VERSION
) -> ActionSpec:
    return ActionSpec(
        embodiment_id=embodiment_id,
        kind=ActionKind.CONTINUOUS,
        dim=3,
        low=(-1.0, -1.0, -1.0),
        high=(1.0, 1.0, 1.0),
        num_classes=None,
        units=("rad", "rad", "rad"),
        wmcp_version=wmcp,
    )


def _episode(seed: int, embodiment_id: str = "so101-arm-7dof") -> Episode:
    gen = torch.Generator().manual_seed(seed)
    transitions = [
        Transition(
            obs_t=torch.randint(0, 256, (3, 2, 4, 4), dtype=torch.uint8, generator=gen),
            action_t=torch.randn(3, generator=gen),
            obs_tp1=torch.randint(
                0, 256, (3, 2, 4, 4), dtype=torch.uint8, generator=gen
            ),
        )
        for _ in range(2)
    ]
    return Episode(
        episode_id=f"ep-{seed}",
        transitions=transitions,
        embodiment_id=embodiment_id,
        modality="rgb-video",
        action_spec=_spec(embodiment_id),
        collection_meta={"site": "lab-a"},
    )


def build_dataset(seeds: tuple[int, ...] = (1, 2, 3)) -> EpisodeDataset:
    """A deterministic in-memory dataset; reproducible byte-for-byte from the seeds alone."""
    return EpisodeDataset([_episode(s) for s in seeds])


def _expected_root_hex(seeds: tuple[int, ...]) -> str:
    leaves = [_h(HashDomain.LEAF, episode_leaf_hash(_episode(s))) for s in seeds]
    return merkle_root(leaves).hex()


# --- commit_dataset: shape, fields, and the independent-root cross-check ---


def test_commit_dataset_fields() -> None:
    c = commit_dataset(build_dataset((1, 2, 3)))
    assert isinstance(c, DatasetCommitment)
    assert c.schema_version == 1
    assert c.episode_count == 3
    assert c.hash_algorithm == "sha256"
    assert c.wmcp_version == WMCP_VERSION
    assert c.embodiment_ids == ("so101-arm-7dof",)
    assert len(c.merkle_root) == 64 and c.merkle_root == c.merkle_root.lower()
    assert c.merkle_root == _expected_root_hex((1, 2, 3))


def test_commit_dataset_multi_embodiment_ids_sorted_unique() -> None:
    ds = EpisodeDataset(
        [_episode(1, "arm-b"), _episode(2, "arm-a"), _episode(3, "arm-a")]
    )
    c = commit_dataset(ds)
    assert c.embodiment_ids == ("arm-a", "arm-b")


# --- determinism: in-process, permutation-invariant, cross-process ---


def test_commit_root_is_deterministic_and_order_independent() -> None:
    a = commit_dataset(build_dataset((1, 2, 3))).merkle_root
    b = commit_dataset(build_dataset((3, 1, 2))).merkle_root  # permuted enumeration
    assert a == b == _expected_root_hex((1, 2, 3))


_SUBPROC = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("commit_builder", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.stdout.write(mod.commit_dataset(mod.build_dataset()).merkle_root)
"""


def test_commit_root_is_stable_across_processes() -> None:
    in_process = commit_dataset(build_dataset()).merkle_root
    out = subprocess.run(
        [sys.executable, "-c", _SUBPROC, str(Path(__file__).resolve())],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "9"},
        check=True,
    )
    assert out.stdout.strip() == in_process


# --- empty / conflicting dataset failures (explicit, not coerced) ---


def test_empty_dataset_raises_provenance_error() -> None:
    with pytest.raises(ProvenanceError):
        commit_dataset(EpisodeDataset([]))


def test_conflicting_wmcp_versions_raise_provenance_error() -> None:
    e1 = _episode(1)
    e2 = Episode(
        episode_id="ep-x",
        transitions=e1.transitions,
        embodiment_id=e1.embodiment_id,
        modality=e1.modality,
        action_spec=_spec(wmcp="wmcp-9.9.9"),  # a different contract version
        collection_meta=e1.collection_meta,
    )
    with pytest.raises(ProvenanceError):
        commit_dataset(EpisodeDataset([e1, e2]))


# --- the binding (INV-COMMIT-BINDING), fail-closed ---


def test_verify_binding_matches_returns_none() -> None:
    root = bytes.fromhex(commit_dataset(build_dataset()).merkle_root)
    assert verify_binding(root, root, _SCHEME) is None


def test_verify_binding_mismatch_raises_commitment_mismatch() -> None:
    committed = bytes.fromhex(commit_dataset(build_dataset((1, 2))).merkle_root)
    other = bytes.fromhex(commit_dataset(build_dataset((3, 4))).merkle_root)
    with pytest.raises(CommitmentMismatch):
        verify_binding(committed, other, _SCHEME)


def test_verify_binding_short_root_raises_merkle_verification_error() -> None:
    committed = bytes.fromhex(commit_dataset(build_dataset()).merkle_root)
    with pytest.raises(MerkleVerificationError):
        verify_binding(committed, b"too-short", _SCHEME)


def test_binding_error_is_security_critical_never_swallowed() -> None:
    # CommitmentMismatch derives from ProvenanceError -> LensembleError and carries code + remediation;
    # the runtime is required not to catch it. Pin the type/contract here.
    committed = bytes.fromhex(commit_dataset(build_dataset((1,))).merkle_root)
    with pytest.raises(CommitmentMismatch) as exc:
        verify_binding(committed, bytes(DIGEST_SIZE), _SCHEME)
    assert isinstance(exc.value, ProvenanceError)
    assert exc.value.code.value == "commitment_mismatch"
    assert exc.value.remediation


def test_binding_against_pseudogradient_dataset_root() -> None:
    # End-to-end hex/bytes boundary: a PseudoGradient binds to R_c via its 32-byte dataset_root.
    c = commit_dataset(build_dataset())
    root = bytes.fromhex(c.merkle_root)
    pg = PseudoGradient(
        delta=torch.zeros(4, dtype=torch.float32),
        l2_norm=0.0,
        dataset_root=root,
        round_index=0,
    )
    assert verify_binding(root, pg.dataset_root, _SCHEME) is None
    bad = PseudoGradient(
        delta=torch.zeros(4, dtype=torch.float32),
        l2_norm=0.0,
        dataset_root=bytes(DIGEST_SIZE),  # all-zero root, not the committed one
        round_index=0,
    )
    with pytest.raises(CommitmentMismatch):
        verify_binding(root, bad.dataset_root, _SCHEME)


# --- DatasetCommitment validation and schema-version gate ---


def _valid_commitment_dict() -> dict:
    c = commit_dataset(build_dataset())
    return json.loads(c.model_dump_json())


def test_commitment_rejects_non_hex64_root() -> None:
    raw = _valid_commitment_dict()
    raw["merkle_root"] = "DEADBEEF"  # too short and uppercase
    with pytest.raises(ValidationError):
        DatasetCommitment.model_validate(raw)


def test_commitment_rejects_zero_episode_count() -> None:
    raw = _valid_commitment_dict()
    raw["episode_count"] = 0
    with pytest.raises(ValidationError):
        DatasetCommitment.model_validate(raw)


def test_commitment_forbids_extra_fields_and_is_frozen() -> None:
    raw = _valid_commitment_dict()
    raw["surprise"] = 1
    with pytest.raises(ValidationError):
        DatasetCommitment.model_validate(raw)
    c = commit_dataset(build_dataset())
    with pytest.raises(ValidationError):
        c.merkle_root = "0" * 64  # type: ignore[misc]  # frozen


def test_parse_round_trips_and_rejects_future_schema() -> None:
    raw = _valid_commitment_dict()
    assert parse_dataset_commitment(json.dumps(raw)).merkle_root == raw["merkle_root"]
    assert parse_dataset_commitment(raw).episode_count == raw["episode_count"]
    future = {**raw, "schema_version": 99}
    with pytest.raises(SchemaVersionMismatch):
        parse_dataset_commitment(future)
    noninteger = {**raw, "schema_version": "1"}
    with pytest.raises(SchemaVersionMismatch):
        parse_dataset_commitment(noninteger)


# --- the read-only episodes accessor commit_dataset reads ---


def test_episode_dataset_exposes_episodes_readonly() -> None:
    ds = build_dataset((1, 2))
    assert len(ds.episodes) == 2
    assert all(isinstance(e, Episode) for e in ds.episodes)
    assert ds.episodes == ds.episodes  # stable tuple
