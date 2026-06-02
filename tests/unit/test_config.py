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


def test_frozen_instance_rejects_mutation() -> None:
    cfg = load_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.run_mode = "eval"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.model.latent_dim = 1  # type: ignore[misc]


def test_overrides_apply_with_precedence() -> None:
    cfg = load_config(
        overrides=["objective.lambda_anc=0.5", "federation.participant_count=8"]
    )
    assert cfg.objective.lambda_anc == 0.5
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
