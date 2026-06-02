"""Eval baseline configs + the gap-recovery reducer (RFC-0005 §5; #54).

Each of the four bracketing baseline config groups resolves to a valid LensembleConfig that shares the
warm-start / probe / seed fields and carries the expected distinguishing knobs (lambda_anc = 0 for naive,
a frozen encoder for Fork A); the gap-recovery reducer returns the analytic rho, clamps to [0, 1], and
rejects a degenerate bracket. Placed in tests/ml (CI-gated).
"""

from __future__ import annotations

import pytest

from lensemble.config import LensembleConfig
from lensemble.errors import EvaluationError
from lensemble.eval import BASELINES, gap_recovery_fraction, load_baseline

# --- config composition: the four bracketing baselines (RFC-0005 §5) ---


def test_all_baselines_resolve_to_valid_configs() -> None:
    configs = {name: load_baseline(name) for name in BASELINES}
    assert set(configs) == {"centralized", "local-only", "naive-fedavg", "fork-a"}
    assert all(isinstance(c, LensembleConfig) for c in configs.values())


def test_baselines_share_warmstart_probe_and_seed() -> None:
    configs = [load_baseline(name) for name in BASELINES]
    warm_starts = {c.model.warm_start_release for c in configs}
    seeds = {c.determinism.root_seed for c in configs}
    encoders = {c.model.encoder for c in configs}
    probes = {c.data.probe_path for c in configs}
    assert len(warm_starts) == 1  # shared warm-start
    assert len(seeds) == 1  # shared seed
    assert len(encoders) == 1  # shared encoder family / probe basis
    assert len(probes) == 1 and None not in probes  # shared pinned public probe


def test_centralized_and_local_only_are_train_local() -> None:
    assert load_baseline("centralized").run_mode == "train_local"
    assert load_baseline("local-only").run_mode == "train_local"


def test_naive_fedavg_turns_the_anchor_off() -> None:
    cfg = load_baseline("naive-fedavg")
    assert cfg.run_mode == "coordinator"
    assert cfg.objective.lambda_anc == 0.0  # the negative control: anchor off


def test_fork_a_freezes_the_encoder() -> None:
    cfg = load_baseline("fork-a")
    assert cfg.run_mode == "coordinator"
    assert cfg.model.encoder_frozen is True  # encoder frozen, federate g_phi only


def test_unknown_baseline_raises() -> None:
    with pytest.raises(EvaluationError):
        load_baseline("bogus")


# --- the gap-recovery fraction reducer ---


def test_gap_recovery_fraction_is_analytic_and_clamped() -> None:
    rho = gap_recovery_fraction(
        success_anchored=0.8, success_local_only=0.4, success_centralized=0.9
    )
    assert rho == pytest.approx((0.8 - 0.4) / (0.9 - 0.4))  # 0.8
    # below the lower bound clamps to 0; above the upper bound clamps to 1
    assert (
        gap_recovery_fraction(
            success_anchored=0.3, success_local_only=0.4, success_centralized=0.9
        )
        == 0.0
    )
    assert (
        gap_recovery_fraction(
            success_anchored=1.0, success_local_only=0.4, success_centralized=0.9
        )
        == 1.0
    )


def test_gap_recovery_fraction_rejects_degenerate_bracket() -> None:
    with pytest.raises(EvaluationError):
        gap_recovery_fraction(
            success_anchored=0.5, success_local_only=0.9, success_centralized=0.8
        )  # centralized not above local-only


def test_gap_recovery_fraction_rejects_out_of_range() -> None:
    with pytest.raises(EvaluationError):
        gap_recovery_fraction(
            success_anchored=1.5, success_local_only=0.4, success_centralized=0.9
        )
