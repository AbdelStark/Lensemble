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
# Re-pinned for #44: FederationConfig gained `secure_agg_threshold: int = 2` and
# `collect_timeout_s: float = 30.0` (the elasticity quorum/timeout knobs, RFC-0013 §3), an intentional,
# reviewed schema addition that shifts the default config's canonical encoding.
# Re-pinned for #48: FederationConfig gained `aggregation_backend: Literal[...] = "masking"` (the
# secure-aggregation backend selector, RFC-0011 §6 — masking #47 / tee #48 / simulated #46), an
# intentional, reviewed schema addition that shifts the default config's canonical encoding.
# Re-pinned for #166: ModelConfig gained the ViT-shape bridge fields (num_frames/tubelet/image_size/
# patch_size/depth/num_heads/in_channels/mlp_ratio) so build_encoder/build_predictor are callable from a
# load_config() config — another intentional, reviewed schema addition.
_GOLDEN_DEFAULT_CONFIG_HASH = (
    "9fce731672c45ab4082ab67649d0ccb8af79a36e2c58f0103648ff68ff9685be"
)


def test_default_config_hash_is_pinned() -> None:
    actual = config_hash(asdict(load()))
    assert actual == _GOLDEN_DEFAULT_CONFIG_HASH, (
        "config_hash of the default config drifted; canonicalization (RFC-0009 7) or a "
        "config default changed. Update the golden only if the change is intentional."
    )
