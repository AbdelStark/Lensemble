"""Canonical episode hashing (RFC-0014 §1-2; #27).

Exact-byte tests, no numerical tolerance: domain separation, cross-process / cross-PYTHONHASHSEED
determinism, the no-pickle / no-raw-data residency contract (``INV-RESIDENCY``), and the digest size.

The episode builder is module-level so the determinism test can re-import it by file path in a
subprocess and confirm the canonical bytes are stable across processes (and across hash-salt seeds).
Provenance tests live under ``tests/ml/`` (matching ``test_provenance_commit.py`` / ``test_hashing.py``)
so the CI gate (07 §8.4 runs ``tests/ml``) actually executes them; the issue's ``tests/provenance/``
path is illustrative and would be collected by no CI gate.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
from lensemble.data.episode import Episode, Transition
from lensemble.provenance.merkle import (
    _EPISODE_CANON_TAG,
    DIGEST_SIZE,
    HashDomain,
    _h,
    canonical_episode_bytes,
    episode_leaf_hash,
)

_SEED = 7
_ALLOWED_META_KEYS = {
    "action_spec_digest",
    "embodiment_id",
    "episode_length",
    "modality",
}


def _action_spec() -> ActionSpec:
    return ActionSpec(
        embodiment_id="so101-arm-7dof",
        kind=ActionKind.CONTINUOUS,
        dim=3,
        low=(-1.0, -1.0, -1.0),
        high=(1.0, 1.0, 1.0),
        num_classes=None,
        units=("rad", "rad", "rad"),
        wmcp_version=WMCP_VERSION,
    )


def build_episode(seed: int = _SEED, num_transitions: int = 3) -> Episode:
    """A deterministic tiny episode: uint8 ``(C,T,H,W)`` observations and float actions, seeded RNG.

    Reproducible byte-for-byte from ``seed`` alone, so a subprocess can rebuild it and confirm the leaf
    hash is process-independent.
    """
    gen = torch.Generator().manual_seed(seed)
    transitions = []
    for _ in range(num_transitions):
        obs_t = torch.randint(0, 256, (3, 2, 4, 4), dtype=torch.uint8, generator=gen)
        action_t = torch.randn(3, generator=gen)
        obs_tp1 = torch.randint(0, 256, (3, 2, 4, 4), dtype=torch.uint8, generator=gen)
        transitions.append(Transition(obs_t=obs_t, action_t=action_t, obs_tp1=obs_tp1))
    return Episode(
        episode_id="ep-0",
        transitions=transitions,
        embodiment_id="so101-arm-7dof",
        modality="rgb-video",
        action_spec=_action_spec(),
        collection_meta={"site": "lab-a"},
    )


# --- digest size and in-process determinism ---


def test_episode_leaf_hash_is_digest_size() -> None:
    leaf = episode_leaf_hash(build_episode())
    assert isinstance(leaf, bytes)
    assert len(leaf) == DIGEST_SIZE == 32


def test_canonical_bytes_are_deterministic_in_process() -> None:
    a = canonical_episode_bytes(build_episode())
    b = canonical_episode_bytes(build_episode())
    assert a == b  # exact byte equality, no tolerance
    assert episode_leaf_hash(build_episode()) == episode_leaf_hash(build_episode())


# --- domain separation (RFC-0014 §1): same payload, distinct digests per domain ---


def test_domain_separation_no_cross_domain_collision() -> None:
    payload = b"the-same-preimage-bytes"
    digests = {
        d: _h(d, payload)
        for d in (
            HashDomain.EPISODE,
            HashDomain.LEAF,
            HashDomain.NODE,
            HashDomain.ROOT,
        )
    }
    assert all(len(v) == DIGEST_SIZE for v in digests.values())
    assert len(set(digests.values())) == 4  # all four are pairwise distinct


def test_episode_domain_differs_from_raw_sha256() -> None:
    import hashlib

    body = canonical_episode_bytes(build_episode())
    # The leaf hash is domain-tagged, so it must not equal a naive SHA-256 of the same body.
    assert episode_leaf_hash(build_episode()) != hashlib.sha256(body).digest()


# --- cross-process / cross-hash-salt determinism (the RFC-0014 §2 RISK conformance check) ---

_SUBPROC = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("eh_builder", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.stdout.write(mod.episode_leaf_hash(mod.build_episode()).hex())
"""


def _subprocess_leaf_hex(hashseed: str) -> str:
    import os

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    out = subprocess.run(
        [sys.executable, "-c", _SUBPROC, str(Path(__file__).resolve())],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return out.stdout.strip()


def test_leaf_hash_is_stable_across_processes_and_hash_salts() -> None:
    in_process = episode_leaf_hash(build_episode()).hex()
    # Two child processes with different PYTHONHASHSEED: a salted str/hash leak would diverge here.
    assert _subprocess_leaf_hex("0") == in_process
    assert _subprocess_leaf_hex("12345") == in_process


# --- INV-RESIDENCY: no pickle, metadata carries no raw observation ---


def _read_meta(canonical: bytes) -> dict:
    import struct

    assert canonical.startswith(_EPISODE_CANON_TAG)
    cursor = len(_EPISODE_CANON_TAG)
    (meta_len,) = struct.unpack_from("<Q", canonical, cursor)
    cursor += 8
    return json.loads(canonical[cursor : cursor + meta_len].decode("utf-8"))


def test_metadata_is_restricted_to_declared_fields() -> None:
    meta = _read_meta(canonical_episode_bytes(build_episode()))
    assert set(meta) == _ALLOWED_META_KEYS
    assert meta["modality"] == "rgb-video"
    assert meta["embodiment_id"] == "so101-arm-7dof"
    assert meta["episode_length"] == 3
    assert (
        isinstance(meta["action_spec_digest"], str)
        and len(meta["action_spec_digest"]) == 64
    )


def test_canonical_bytes_contain_no_pickle() -> None:
    body = canonical_episode_bytes(build_episode())
    # A pickle stream (protocol >= 2) opens with the PROTO opcode 0x80; ours opens with the ASCII tag.
    assert not body.startswith(b"\x80")
    assert body.startswith(_EPISODE_CANON_TAG)
    with pytest.raises(Exception):  # not a pickle payload
        pickle.loads(body)


def test_module_source_uses_no_pickle() -> None:
    import lensemble.provenance.merkle as m

    src = Path(m.__file__).read_text(encoding="utf-8")
    assert "import pickle" not in src
    assert "pickle." not in src


# --- a re-quantized / different episode is a different leaf (RFC-0014 §2 RISK) ---


def test_requantized_observation_changes_the_leaf() -> None:
    base = build_episode()
    requantized = Episode(
        episode_id=base.episode_id,
        transitions=[
            Transition(
                obs_t=tr.obs_t.to(torch.float16),  # same values, different stored dtype
                action_t=tr.action_t,
                obs_tp1=tr.obs_tp1.to(torch.float16),
            )
            for tr in base.transitions
        ],
        embodiment_id=base.embodiment_id,
        modality=base.modality,
        action_spec=base.action_spec,
        collection_meta=base.collection_meta,
    )
    assert episode_leaf_hash(base) != episode_leaf_hash(requantized)


def test_different_episodes_have_different_leaves() -> None:
    assert episode_leaf_hash(build_episode(seed=1)) != episode_leaf_hash(
        build_episode(seed=2)
    )


def test_bfloat16_tensors_hash_deterministically() -> None:
    # bf16 has no numpy dtype; its bytes go through the int16-view path. Exercise it and confirm the
    # encoding is deterministic and dtype-distinct from an fp32 episode with the same values.
    base = build_episode(num_transitions=1)
    bf16 = Episode(
        episode_id=base.episode_id,
        transitions=[
            Transition(
                obs_t=tr.obs_t.to(torch.bfloat16),
                action_t=tr.action_t.to(torch.bfloat16),
                obs_tp1=tr.obs_tp1.to(torch.bfloat16),
            )
            for tr in base.transitions
        ],
        embodiment_id=base.embodiment_id,
        modality=base.modality,
        action_spec=base.action_spec,
        collection_meta=base.collection_meta,
    )
    assert episode_leaf_hash(bf16) == episode_leaf_hash(bf16)
    assert len(episode_leaf_hash(bf16)) == DIGEST_SIZE
    fp32 = Episode(
        episode_id=base.episode_id,
        transitions=[
            Transition(
                obs_t=tr.obs_t.to(torch.float32),
                action_t=tr.action_t.to(torch.float32),
                obs_tp1=tr.obs_tp1.to(torch.float32),
            )
            for tr in base.transitions
        ],
        embodiment_id=base.embodiment_id,
        modality=base.modality,
        action_spec=base.action_spec,
        collection_meta=base.collection_meta,
    )
    assert episode_leaf_hash(bf16) != episode_leaf_hash(fp32)
