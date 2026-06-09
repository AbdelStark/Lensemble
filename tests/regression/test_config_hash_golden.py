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
# Re-pinned for #167: DataConfig gained `data_source: str | None = None` (the #22 local-episode source the
# participant's default `_local_windows` resolves via `load_episodes`) and `window_steps: int = 1` (the
# training-window horizon `num_steps`), an intentional, reviewed schema addition that shifts the default
# config's canonical encoding.
# Re-pinned for #191: ObjectiveConfig gained `target_stop_gradient: bool = True`, preserving the existing
# default path while enabling claim-grade LeWorldModel base mode with `False`.
# Re-pinned for #259: FederationConfig gained `inner_lr: float = 1e-3` (the participant inner-loop AdamW
# step size, exposed as a launcher knob for the centralized-like frequent-sync regime), an intentional,
# reviewed schema addition that shifts the default config's canonical encoding.
_GOLDEN_DEFAULT_CONFIG_HASH = (
    "a60d96c4f563541fa437d0ea48f3bf005d64c9a7da1267704b0213bb9875912d"
)


def test_default_config_hash_is_pinned() -> None:
    actual = config_hash(asdict(load()))
    assert actual == _GOLDEN_DEFAULT_CONFIG_HASH, (
        "config_hash of the default config drifted; canonicalization (RFC-0009 7) or a "
        "config default changed. Update the golden only if the change is intentional."
    )
