"""Provenance commit stub contract (RFC-0014). Security-critical surface (INV-COMMIT-BINDING).

`commit_dataset` is scaffolded but unimplemented before #29. This pins the explicit-failure
contract (prefer a loud `NotImplementedError` over a silent wrong root) so the security-critical
module stays at full coverage as the CI gate (07 §8) holds it to 100%.
"""

from __future__ import annotations

import pytest

from lensemble.provenance.commit import DatasetCommitment, commit_dataset


def test_commit_dataset_fails_explicitly_until_implemented() -> None:
    with pytest.raises(NotImplementedError, match="#29"):
        commit_dataset(object())


def test_dataset_commitment_type_is_exported() -> None:
    # The result type is named now so downstream contracts can reference it before the body lands.
    assert isinstance(DatasetCommitment(), DatasetCommitment)
