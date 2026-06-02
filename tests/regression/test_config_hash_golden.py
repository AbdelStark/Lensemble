"""Golden vector pinning the canonical config_hash of the default config (RFC-0009 7). Issue #37.

A regression guard: if the canonical-JSON encoding (RFC-0009 7) or a config default changes, this fails
so the change is intentional and reviewed — config_hash is the comparison key every silo uses, so a
silent drift would split the federation. Update the golden deliberately, never to make CI pass.
"""

from __future__ import annotations

from dataclasses import asdict

from lensemble.config import config_hash, load

# SHA-256 over the canonical bytes of the default LensembleConfig (algo sha256-canon-v1).
# Re-pinned for #54: ModelConfig gained `encoder_frozen: bool = False` (Fork A baseline), an intentional,
# reviewed schema addition that shifts the default config's canonical encoding.
_GOLDEN_DEFAULT_CONFIG_HASH = (
    "d2ad8a4fb7fd0c594c196a36dc84ad8e0a6a9ababd53b97965c2609eb11e99c0"
)


def test_default_config_hash_is_pinned() -> None:
    actual = config_hash(asdict(load()))
    assert actual == _GOLDEN_DEFAULT_CONFIG_HASH, (
        "config_hash of the default config drifted; canonicalization (RFC-0009 7) or a "
        "config default changed. Update the golden only if the change is intentional."
    )
