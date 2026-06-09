"""The ablation ladder — the paper's core experiment, as a CPU regression guard (#55).

RFC-0005 §6 (the ablation ladder) realizes the RFC-0002 §4 gauge fix *additively*: each rung adds exactly
one mechanism (naive FedAvg -> + shared sketch A -> + Procrustes backstop -> + frame-anchor loss ->
+ function-space distillation), and the runner reports all three metric families (frame drift §2, MPC
success §3, effective dim §4) at each rung. This is the small-config CPU version that
[07-testing-strategy §3](docs/spec/07-testing-strategy.md) names: the five rungs RUN on tiny synthetic
per-silo data and the load-bearing qualitative ordering (naive worst on drift; anchored flat,
RFC-0005 §6) is asserted as the regression guard for the central experiment.

The drift signal is NON-VACUOUS because the silos hold genuinely DIFFERENT data (different per-episode
seeds): under the naive rung (no gauge control) the per-silo latent frames diverge, while under the
anchored rung the Variant-A landmark anchor pins each frame back onto the round-0 reference. IDENTICAL
silo data would show no drift regardless of rung and make the experiment vacuous.

Placed in tests/ml: the §8 CI gate scans tests/{unit,property,integration,ml,e2e,regression}. The issue
named ``tests/eval/`` which the CI gate does NOT scan, so the test lives in tests/ml (where the other
model-bearing federation runtime tests — test_coordinator/test_participant — also live).

Shapes (the encoder/window contract, mirrors tests/e2e/test_toy_pipeline + tests/ml/test_coordinator): a
clip is ``(_T, _C, _H, _W)``; a ``Window.obs`` is ``(window_steps + 1, _T, _C, _H, _W)``; the probe
carries ``k = _D >= d`` landmark clips so the landmark anchor pins all O(d) gauge dofs. Dims/rounds/silos
are tiny but sufficient for a non-vacuous drift signal, and the whole run is CPU-fast, downloads nothing.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pytest
import torch

from lensemble.config import LensembleConfig
from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
from lensemble.data.episode import Window
from lensemble.eval.ablation import LADDER_RUNGS, RungReport, lambda_anc_sweep
from lensemble.federation.ablation import run_ablation_ladder
from lensemble.federation.simulation import (
    SiloData,
    SimulationResult,
    run_federated_simulation,
)

# --- the tiny CPU model config (the proven tests/ml shape) ---

_D = 8
_NUM_TOKENS = 4  # (2//2)*(4//2)**2 = 4
_T, _C, _H, _W = 2, 3, 4, 4
_ACTION_DIM = 2
_WINDOW_STEPS = 1


@dataclass(frozen=True)
class _ModelConfig:
    encoder: str = "vjepa2-vit-l"
    warm_start_release: str = "vjepa2-2.0"
    latent_dim: int = _D
    num_tokens: int = _NUM_TOKENS
    predictor_depth: int = 1
    predictor_width: int = _D
    wmcp_version: str = WMCP_VERSION
    encoder_frozen: bool = False
    d: int = _D
    in_channels: int = _C
    num_frames: int = _T
    image_size: int = _H
    patch_size: int = 2
    tubelet: int = 2
    depth: int = 1
    num_heads: int = 2
    cond_dim: int = _D


def _spec() -> ActionSpec:
    return ActionSpec(
        embodiment_id="toy",
        kind=ActionKind.CONTINUOUS,
        dim=_ACTION_DIM,
        low=(-1.0, -1.0),
        high=(1.0, 1.0),
        num_classes=None,
        units=("u", "u"),
        wmcp_version=WMCP_VERSION,
    )


def _silo_windows(seed: int, n: int = 4) -> list[Window]:
    """``n`` tiny windows from a silo-specific seed (DIFFERENT data per silo → genuine frame drift)."""
    gen = torch.Generator().manual_seed(seed)
    windows: list[Window] = []
    for _ in range(n):
        windows.append(
            Window(
                obs=torch.randn(_WINDOW_STEPS + 1, _T, _C, _H, _W, generator=gen),
                actions=torch.randn(_WINDOW_STEPS, _ACTION_DIM, generator=gen),
                num_steps=_WINDOW_STEPS,
                embodiment_id="toy",
            )
        )
    return windows


def _silos(num_silos: int = 2) -> list[SiloData]:
    """Per-silo data with DIFFERENT seeds (so the naive frames genuinely diverge), shared spec + root."""
    return [
        SiloData(
            participant_id=f"silo-{i}",
            windows=_silo_windows(seed=1000 + 31 * i),
            action_spec=_spec(),
            dataset_root=bytes([i + 1]) * 32,
        )
        for i in range(num_silos)
    ]


def _base_cfg() -> LensembleConfig:
    """A coordinator-mode config with the tiny model, DP off (honest delta), a quorum of 2 silos."""
    base = LensembleConfig()
    federation = dataclasses.replace(
        base.federation,
        inner_horizon=30,  # enough inner DiLoCo steps that the frames genuinely diverge (and the anchor pins)
        participant_count=2,
        fault_tolerance_min_participants=2,
        secure_agg_threshold=2,
        outer_lr=0.1,  # a GENTLE outer step so the multi-round average is stable on the toy budget
        outer_nesterov_momentum=0.0,
    )
    privacy = dataclasses.replace(base.privacy, enabled=False)
    # lambda_anc=0 on the base: the bare-harness default (no probe pinned). The ladder runner sets the
    # per-rung lambda_anc itself (and pins a real probe for the anchored rungs).
    objective = dataclasses.replace(base.objective, lambda_anc=0.0)
    # The built-in deterministic toy eval world (#167) so the per-rung MPC success_rate is the KNOWN 0.5.
    eval_cfg = dataclasses.replace(base.eval, env_id="synthetic://toy")
    return dataclasses.replace(
        base,
        model=_ModelConfig(),  # type: ignore[arg-type]
        federation=federation,
        privacy=privacy,
        objective=objective,
        eval=eval_cfg,
        run_mode="coordinator",
        determinism=dataclasses.replace(
            base.determinism, deterministic_aggregation=True
        ),
    )


# --- the harness runs a live multi-round federated simulation ---


def test_simulation_runs_multi_round_and_reports_finite_metrics() -> None:
    cfg = _base_cfg()
    result = run_federated_simulation(_silos(), cfg=cfg, num_rounds=2)
    assert isinstance(result, SimulationResult)
    assert len(result.per_round) == 2
    for rec in result.per_round:
        assert rec.frame_drift_angle_deg >= 0.0
        assert torch.isfinite(torch.tensor(rec.frame_drift_angle_deg))
        assert rec.effective_dim > 0.0
        assert 0.0 <= rec.success_rate <= 1.0
    # The committed global hash advanced over the rounds (a real federated run, not a no-op).
    assert result.final_global_hash != result.initial_global_hash


# --- the ladder: all five rungs RUN on CPU and yield a finite RungReport each ---


def test_all_five_rungs_run_and_report() -> None:
    cfg = _base_cfg()
    reports = run_ablation_ladder(cfg, _silos(), num_rounds=2)
    assert set(reports) == {name for name, _ in LADDER_RUNGS}
    assert len(reports) == 5
    for name, report in reports.items():
        assert isinstance(report, RungReport), name
        assert report.frame_drift_angle_deg >= 0.0, name
        assert report.frame_drift_residual >= 0.0, name
        assert 0.0 <= report.success_rate <= 1.0, name
        assert report.effective_dim > 0.0, name


def test_ladder_rungs_are_the_five_additive_mechanisms_in_order() -> None:
    names = [name for name, _ in LADDER_RUNGS]
    assert names == [
        "naive-fedavg",
        "shared-sketch",
        "procrustes-backstop",
        "frame-anchor",
        "distillation",
    ]


# --- the load-bearing qualitative ordering (RFC-0005 §6): naive worst on drift; anchored flat ---


def test_naive_drifts_more_than_anchored_by_a_margin() -> None:
    # The central-experiment regression guard. The silos hold genuinely DIFFERENT data, so under the naive
    # rung (lambda_anc=0, backstop off) the per-silo frames diverge; under the anchored rung the Variant-A
    # landmark anchor pins each frame onto the round-0 reference. We assert the LOAD-BEARING claim — naive
    # drift > anchored drift, by a clear margin — plus that the anchored rung's drift is small ("flat"),
    # which is the honest, non-flaky statement of "naive worst on drift; anchored flat" on a toy CPU budget
    # (a strict 5-rung monotonic ordering is not reliable at this scale; this relaxed-but-meaningful
    # ordering is). Everything is seeded so the claim is deterministic across runs.
    cfg = _base_cfg()
    reports = run_ablation_ladder(cfg, _silos(), num_rounds=3)
    naive = reports["naive-fedavg"]
    anchored = reports["frame-anchor"]

    # THE EXACT CLAIM (RFC-0005 §6): the naive rung's mean inter-silo frame drift materially EXCEEDS the
    # anchored rung's, by a clear margin. With the seeded toy config above the measured values are stable:
    # naive ~25.5 deg, anchored ~7.3 deg (a ~18 deg gap), so the >= 8 deg margin is a wide, non-flaky floor.
    assert naive.frame_drift_angle_deg > anchored.frame_drift_angle_deg + 8.0, (
        f"naive drift {naive.frame_drift_angle_deg:.3f} deg should exceed anchored "
        f"{anchored.frame_drift_angle_deg:.3f} deg by a clear margin (RFC-0005 §6)"
    )
    # The anchored frame is held "flat": its mean inter-silo drift is small in absolute terms (~7.3 deg,
    # bounded well under the naive ~25.5 deg) — the Variant-A landmark anchor pinning the frame.
    assert anchored.frame_drift_angle_deg < 12.0, (
        f"the anchored rung should hold the frame flat, got "
        f"{anchored.frame_drift_angle_deg:.3f} deg"
    )


# --- the lambda_anc sweep (RFC-0002 §7): each value -> a distinct valid LensembleConfig ---


def test_lambda_anc_sweep_resolves_distinct_valid_configs() -> None:
    cfg = _base_cfg()
    # An anchored sweep config requires a pinned probe (validate_config); the sweep is over the knob only,
    # so we use a non-anchored base (train_local + no probe) to keep each resolved config valid as a config.
    base = dataclasses.replace(
        cfg,
        run_mode="train_local",
        objective=dataclasses.replace(cfg.objective, lambda_anc=0.0),
    )
    values = [0.0, 0.5, 1.0, 2.0]
    sweep = lambda_anc_sweep(base, values)
    assert set(sweep) == set(values)
    for v in values:
        resolved = sweep[v]
        assert isinstance(resolved, LensembleConfig)
        assert resolved.objective.lambda_anc == v
    # The configs are DISTINCT objects with distinct knobs (a real config-group override per value).
    assert len({id(c) for c in sweep.values()}) == len(values)
    assert sweep[0.0].objective.lambda_anc != sweep[2.0].objective.lambda_anc


def test_tuned_anchor_lowers_round_n_drift_vs_weak_001() -> None:
    """#261: the tuned per-round frame leash (lambda_anc=1.0) holds round-N frame drift far below 0.01.

    The #249 runs used ``lambda_anc=0.01`` — 100x below the ``ObjectiveConfig`` schema default of 1.0 —
    too weak to hold each participant near the shared broadcast global through the ``H`` inner steps, so
    their released deltas rotated apart before aggregation. This is the falsifiable #261 claim on a seeded
    toy CPU budget: at the tuned strength the inter-participant frame drift measured on the SAME pinned
    probe at the final round is materially lower than at 0.01, while the committed global frame's effective
    dimension is preserved (no collapse, no saturation). The anchor target is the SAME shared round-0
    reference for both sweep points (one composed probe), so the only variable is the leash STRENGTH.
    """
    from lensemble.eval.ablation import RungSpec, cleanup_rung, compose_rung

    base = _base_cfg()
    # Compose an anchored rung once to pin a real k>=d landmark probe (round-0 f_ref targets); both sweep
    # points reuse its probe_path so the anchor reference is identical and only lambda_anc differs.
    composed = compose_rung(
        base,
        RungSpec(
            "frame-anchor",
            lambda_sig=0.1,
            lambda_anc=1.0,
            backstop=False,
            distill=False,
        ),
    )
    try:
        cfg_tuned = composed.cfg  # lambda_anc = 1.0 (the schema strength)
        cfg_weak = dataclasses.replace(
            cfg_tuned,
            objective=dataclasses.replace(cfg_tuned.objective, lambda_anc=0.01),
        )
        silos = _silos(num_silos=3)
        weak = run_federated_simulation(
            silos, cfg=cfg_weak, num_rounds=3, backstop=False
        )
        tuned = run_federated_simulation(
            silos, cfg=cfg_tuned, num_rounds=3, backstop=False
        )
        # Determinism: a re-run reproduces the round-N drift bit-for-bit (INV-AGG-DETERMINISM upstream).
        tuned_again = run_federated_simulation(
            silos, cfg=cfg_tuned, num_rounds=3, backstop=False
        )
    finally:
        cleanup_rung(composed)

    weak_drift = weak.per_round[-1].frame_drift_angle_deg
    tuned_drift = tuned.per_round[-1].frame_drift_angle_deg
    # THE #261 CLAIM: round-N inter-participant frame drift is materially lower at the tuned strength.
    # Measured on the seeded toy config: weak (0.01) ~15.3 deg, tuned (1.0) ~5.7 deg — a ~9.6 deg gap, so
    # the >= 6 deg margin is a wide, non-flaky floor.
    assert tuned_drift < weak_drift - 6.0, (
        f"tuned lambda_anc=1.0 drift {tuned_drift:.3f} deg should be materially below the weak "
        f"lambda_anc=0.01 drift {weak_drift:.3f} deg (#261)"
    )
    # Rank preserved: the committed global frame's effective dimension is NOT collapsed under the tuned
    # anchor (toy d=8; measured ~5.1) and stays within a hair of the weak baseline (no saturation killing
    # the frame's spread).
    tuned_eff = tuned.per_round[-1].effective_dim
    weak_eff = weak.per_round[-1].effective_dim
    assert tuned_eff > 2.0, (
        f"tuned anchor collapsed the frame (eff_dim {tuned_eff:.3f})"
    )
    assert tuned_eff >= 0.85 * weak_eff, (
        f"tuned anchor saturated the frame: eff_dim {tuned_eff:.3f} fell well below the weak "
        f"baseline {weak_eff:.3f}"
    )
    assert tuned_again.per_round[-1].frame_drift_angle_deg == tuned_drift


def test_rung_report_is_frozen() -> None:
    report = RungReport(
        frame_drift_residual=0.1,
        frame_drift_angle_deg=1.0,
        success_rate=0.5,
        effective_dim=2.0,
    )
    with pytest.raises((TypeError, ValueError, AttributeError)):
        report.success_rate = 0.9  # type: ignore[misc]


# --- harness edge cases (validation, single-silo zero drift, degenerate-pair safety) ---


def test_harness_rejects_empty_silos_and_nonpositive_rounds() -> None:
    cfg = _base_cfg()
    with pytest.raises(ValueError, match="at least one silo"):
        run_federated_simulation([], cfg=cfg, num_rounds=1)
    with pytest.raises(ValueError, match="num_rounds"):
        run_federated_simulation(_silos(num_silos=1), cfg=cfg, num_rounds=0)


def test_single_silo_run_reports_zero_inter_silo_drift() -> None:
    # One silo => no inter-silo PAIR => the mean drift is 0.0 by definition (the <2-silo path). The quorum
    # must match a single contributor for the round to CLOSE.
    base = _base_cfg()
    cfg = dataclasses.replace(
        base,
        federation=dataclasses.replace(
            base.federation,
            participant_count=1,
            fault_tolerance_min_participants=1,
            secure_agg_threshold=1,
        ),
    )
    result = run_federated_simulation(_silos(num_silos=1), cfg=cfg, num_rounds=1)
    assert result.per_round[0].frame_drift_angle_deg == 0.0
    assert result.per_round[0].frame_drift_residual == 0.0


def test_inter_silo_drift_is_degenerate_safe_for_collapsed_frames() -> None:
    # When the silos' probe frames live in a LOW-DIMENSIONAL subspace (a collapsed frame — the failure the
    # anchor guards against), the inter-pair Procrustes M = T^T S has a near-zero singular value and
    # raises DegenerateProcrustes; the harness must treat that pair as ~0 deg drift rather than abort. Two
    # frames spanning only a 2-dim subspace of R^d (d > 2) trigger it.
    from lensemble.federation.simulation import _inter_silo_drift

    basis = torch.randn(2, _D)  # both frames live in this 2-dim subspace of R^d
    a = torch.randn(16, 2) @ basis
    b = torch.randn(16, 2) @ basis
    angle, residual = _inter_silo_drift({"a": a, "b": b})
    assert angle == 0.0
    assert residual == 0.0


def test_inter_silo_drift_zero_for_coinciding_full_rank_frames() -> None:
    # Two identical FULL-RANK frames align cleanly (Q ~ I): the recovered angle is ~0 deg (no degeneracy).
    # The tolerance is a fraction of a degree, not bitwise zero: the closed-form Procrustes recovers Q=I
    # only to the SVD's numerical precision, which is platform-dependent (a different BLAS/LAPACK can
    # leave ~0.05 deg of spurious rotation on a symmetric-PSD M); ~0.05 deg is "no drift" for this guard.
    from lensemble.federation.simulation import _inter_silo_drift

    frame = torch.randn(16, _D)
    angle, _residual = _inter_silo_drift({"a": frame, "b": frame.clone()})
    assert angle < 0.5  # near-zero to SVD precision (cross-platform), not bitwise zero


def test_success_rate_falls_back_to_zero_for_unresolvable_env() -> None:
    # With an eval env_id that cannot resolve (no toy world, no stable-worldmodel), the success metric is a
    # graceful 0.0 rather than an abort — the ladder's load-bearing signal is the frame drift (§2/§6).
    base = _base_cfg()
    cfg = dataclasses.replace(
        base, eval=dataclasses.replace(base.eval, env_id="nonexistent://world")
    )
    result = run_federated_simulation(_silos(), cfg=cfg, num_rounds=1)
    assert result.per_round[0].success_rate == 0.0
