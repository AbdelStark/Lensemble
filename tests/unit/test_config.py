"""Structured config tree + boundary validation (RFC-0009 2/3). Issue #34."""

from __future__ import annotations

import dataclasses

import pytest

from lensemble.config import LensembleConfig, load_config, validate_config
from lensemble.errors import ConfigError, LensembleErrorCode

# --- T1: load, freeze, resolve, override ---


def test_default_loads_and_resolves() -> None:
    cfg = load_config()
    assert isinstance(cfg, LensembleConfig)
    assert cfg.model.latent_dim == 1024
    assert cfg.federation.participant_count == 4
    assert cfg.run_mode == "train_local"
    assert cfg.objective.anchor_variant == "landmark"
    assert cfg.objective.target_stop_gradient is True


def test_frozen_instance_rejects_mutation() -> None:
    cfg = load_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.run_mode = "eval"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.model.latent_dim = 1  # type: ignore[misc]


def test_overrides_apply_with_precedence() -> None:
    cfg = load_config(
        overrides=[
            "objective.lambda_anc=0.5",
            "objective.target_stop_gradient=false",
            "federation.participant_count=8",
        ]
    )
    assert cfg.objective.lambda_anc == 0.5
    assert cfg.objective.target_stop_gradient is False
    assert cfg.federation.participant_count == 8


def test_unknown_override_key_rejected() -> None:
    with pytest.raises(ConfigError):
        load_config(overrides=["nonexistent.key=1"])


def test_type_violation_rejected() -> None:
    with pytest.raises(ConfigError):
        load_config(overrides=["federation.participant_count=not_an_int"])


# --- T2: every cross-field validation rule raises ConfigError ---


def _cfg(**top: object) -> LensembleConfig:
    return dataclasses.replace(load_config(), **top)


def _expect(cfg: LensembleConfig) -> ConfigError:
    with pytest.raises(ConfigError) as exc:
        validate_config(cfg)
    assert exc.value.code == LensembleErrorCode.CONFIG_INVALID
    assert exc.value.remediation, "remediation must be non-empty"
    return exc.value


def test_rule_latent_dim_consistency() -> None:
    base = load_config()
    _expect(
        dataclasses.replace(base, model=dataclasses.replace(base.model, latent_dim=999))
    )


def test_rule_landmark_coverage() -> None:
    base = load_config()
    _expect(
        dataclasses.replace(
            base, gauge=dataclasses.replace(base.gauge, anchor_landmark_count=10)
        )
    )


def test_rule_fault_tolerance_floor() -> None:
    base = load_config()
    _expect(
        dataclasses.replace(
            base,
            federation=dataclasses.replace(
                base.federation, fault_tolerance_min_participants=99
            ),
        )
    )


def test_rule_secure_agg_threshold_above_participant_count() -> None:
    # K = max(min_participants, secure_agg_threshold); t_agg must be in (0, C] (#44, RFC-0013 §3).
    base = load_config()
    err = _expect(
        dataclasses.replace(
            base,
            federation=dataclasses.replace(base.federation, secure_agg_threshold=99),
        )
    )
    assert err.key == "federation.secure_agg_threshold"  # type: ignore[attr-defined]


def test_rule_secure_agg_threshold_non_positive() -> None:
    base = load_config()
    _expect(
        dataclasses.replace(
            base,
            federation=dataclasses.replace(base.federation, secure_agg_threshold=0),
        )
    )


def test_rule_collect_timeout_non_positive() -> None:
    # The COLLECTING wall-time budget must be strictly positive (#44, RFC-0013 §3).
    base = load_config()
    err = _expect(
        dataclasses.replace(
            base,
            federation=dataclasses.replace(base.federation, collect_timeout_s=0.0),
        )
    )
    assert err.key == "federation.collect_timeout_s"  # type: ignore[attr-defined]


def test_rule_window_steps_non_positive() -> None:
    # The training-window horizon must be a positive step count (#167; mirrors EpisodeDataset.windows).
    base = load_config()
    err = _expect(
        dataclasses.replace(base, data=dataclasses.replace(base.data, window_steps=0))
    )
    assert err.key == "data.window_steps"  # type: ignore[attr-defined]


def test_rule_eval_planning_samples_non_positive() -> None:
    base = load_config()
    err = _expect(
        dataclasses.replace(
            base, eval=dataclasses.replace(base.eval, planning_samples=0)
        )
    )
    assert err.key == "eval.planning_samples"  # type: ignore[attr-defined]


def test_rule_eval_horizon_non_positive() -> None:
    base = load_config()
    err = _expect(
        dataclasses.replace(base, eval=dataclasses.replace(base.eval, horizon=0))
    )
    assert err.key == "eval.horizon"  # type: ignore[attr-defined]


def test_data_source_defaults_none_and_window_steps_one() -> None:
    # The #167 toy-pipeline knobs: no configured source by default; horizon 1.
    cfg = load_config()
    assert cfg.data.data_source is None
    assert cfg.data.window_steps == 1


def test_data_source_override_resolves() -> None:
    cfg = load_config(
        overrides=["data.data_source=/tmp/ds.lance", "data.window_steps=2"]
    )
    assert cfg.data.data_source == "/tmp/ds.lance"
    assert cfg.data.window_steps == 2


def test_rule_dp_budget() -> None:
    base = load_config()
    _expect(
        dataclasses.replace(base, privacy=dataclasses.replace(base.privacy, delta=2.0))
    )


def test_rule_agg_determinism_for_federated() -> None:
    base = load_config()
    bad = dataclasses.replace(
        base,
        run_mode="coordinator",
        determinism=dataclasses.replace(
            base.determinism, deterministic_aggregation=False
        ),
        data=dataclasses.replace(base.data, probe_path="probe.safetensors"),
    )
    _expect(bad)


def test_rule_residency_on_network_transport() -> None:
    base = load_config()
    bad = dataclasses.replace(
        base,
        federation=dataclasses.replace(base.federation, transport="network"),
        data=dataclasses.replace(base.data, residency_enforced=False),
    )
    _expect(bad)


def test_rule_variant_b_singular_floor() -> None:
    base = load_config()
    bad = dataclasses.replace(
        base,
        objective=dataclasses.replace(base.objective, anchor_variant="rotational"),
        gauge=dataclasses.replace(base.gauge, procrustes_singular_floor=0.0),
    )
    _expect(bad)


def test_rule_probe_presence_for_anchored_runs() -> None:
    base = load_config()
    # coordinator run with an active anchor (lambda_anc>0) but no probe pinned
    bad = dataclasses.replace(
        base, run_mode="coordinator"
    )  # probe_path defaults to None
    err = _expect(bad)
    assert "probe" in err.remediation.lower()


# --- T3: the ViT-shape bridge (#166) — load_config() drives build_encoder/build_predictor ---


def test_model_config_exposes_vit_shape_fields() -> None:
    # The bridge fields build_encoder/build_predictor read must exist on the typed ModelConfig
    # (the gap #166 closes — previously only `latent_dim`/`num_tokens` existed and the builders
    # crashed with AttributeError for any load_config() config).
    m = load_config().model
    for f in (
        "num_frames",
        "tubelet",
        "image_size",
        "patch_size",
        "depth",
        "num_heads",
        "in_channels",
        "mlp_ratio",
    ):
        assert hasattr(m, f), f"ModelConfig is missing ViT-shape field {f!r}"


def test_default_vit_shape_is_self_consistent() -> None:
    # num_tokens == (num_frames // tubelet) * (image_size // patch_size) ** 2, and the patching divides.
    m = load_config().model
    assert m.num_frames % m.tubelet == 0
    assert m.image_size % m.patch_size == 0
    derived = (m.num_frames // m.tubelet) * (m.image_size // m.patch_size) ** 2
    assert derived == m.num_tokens
    assert (
        m.latent_dim % m.num_heads == 0
    )  # heads divide the hidden dim (build_encoder check)


def test_build_encoder_predictor_from_load_config() -> None:
    # The acceptance property: build_encoder/build_predictor succeed (no AttributeError) and the
    # constructed modules' shape attrs match the config. Use a tiny consistent override so the unit
    # test stays cheap (the default 1024-dim/24-depth shape is exercised by the dry checks above and
    # the ml suites); the override keeps the SAME coherence rule num_tokens == derived.
    from lensemble.model import build_encoder, build_predictor

    cfg = load_config(
        overrides=[
            "model.encoder=vjepa2-vit-l",  # keep ENCODER_DIM consistency (latent_dim==1024)
            "model.num_tokens=4",  # == the derived token count below (the #166 coherence rule)
            "model.num_frames=2",
            "model.tubelet=2",
            "model.image_size=8",
            "model.patch_size=4",
            "model.depth=1",
            "model.predictor_depth=1",
            "model.num_heads=8",
        ]
    )
    # (2//2) * (8//4)**2 = 1 * 4 = 4 tokens; latent_dim stays 1024 (ENCODER_DIM["vjepa2-vit-l"]).
    assert cfg.model.num_tokens == 4
    encoder = build_encoder(cfg)
    predictor = build_predictor(cfg)
    assert encoder.d == cfg.model.latent_dim
    assert encoder.num_tokens == cfg.model.num_tokens
    assert predictor.d == cfg.model.latent_dim
    assert predictor.num_tokens == cfg.model.num_tokens

    # build_action_head must also resolve cond_dim from a real ModelConfig (which has no cond_dim
    # field) via the latent_dim fallback (#166 — previously it read the nonexistent cfg.model.d).
    from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
    from lensemble.model import build_action_head

    spec = ActionSpec(
        embodiment_id="toy",
        kind=ActionKind.CONTINUOUS,
        dim=2,
        low=(-1.0, -1.0),
        high=(1.0, 1.0),
        num_classes=None,
        units=("u", "u"),
        wmcp_version=WMCP_VERSION,
    )
    head = build_action_head(cfg, spec)
    assert head.cond_dim == cfg.model.latent_dim  # cond_dim falls back to latent_dim


def test_dynamic_env_small_shape_preset_is_valid() -> None:
    cfg = load_config(
        overrides=[
            "model.encoder=scratch",
            "model.latent_dim=128",
            "model.num_tokens=9",
            "model.num_frames=1",
            "model.tubelet=1",
            "model.image_size=48",
            "model.patch_size=16",
            "model.depth=4",
            "model.num_heads=4",
            "model.predictor_depth=2",
            "model.predictor_width=128",
            "objective.sigreg_sketch_dim=64",
            "data.format=synthetic-dynamic",
            "data.data_source=synthetic-dynamic://swipe-dot?seed=0&n_episodes=8&steps=64&image_size=48",
            "data.window_steps=2",
            "eval.env_id=kinematic://swipe-dot",
        ]
    )
    assert cfg.model.encoder == "scratch"
    assert cfg.model.latent_dim == 128
    assert cfg.model.num_tokens == 9
    assert cfg.model.num_tokens == (1 // 1) * (48 // 16) ** 2
    assert cfg.model.latent_dim % cfg.model.num_heads == 0
    assert cfg.objective.sigreg_sketch_dim <= cfg.model.latent_dim
    assert cfg.gauge.anchor_landmark_count >= cfg.model.latent_dim
    assert cfg.data.format == "synthetic-dynamic"


def test_rule_vit_shape_inconsistency_rejected() -> None:
    # An inconsistent ViT shape (num_tokens not equal to the derived token count) is a ConfigError.
    base = load_config()
    err = _expect(
        dataclasses.replace(
            base,
            model=dataclasses.replace(base.model, num_tokens=base.model.num_tokens + 1),
        )
    )
    assert err.key == "model.num_tokens"  # type: ignore[attr-defined]


def test_rule_num_heads_must_divide_latent_dim() -> None:
    # num_heads must divide latent_dim (mirrors build_encoder's runtime check).
    base = load_config()
    err = _expect(
        dataclasses.replace(base, model=dataclasses.replace(base.model, num_heads=7))
    )
    assert err.key == "model.num_heads"  # type: ignore[attr-defined]
