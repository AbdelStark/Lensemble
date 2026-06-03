"""The RFC-0005 §7 sweeps — non-IID severity, C/H, and scale — over the §6 ladder (#56).

These sweeps drive the [RFC-0005 §7](docs/rfcs/RFC-0005-evaluation.md#7-non-iid-severity--scale-sweeps)
robustness axes (Claim 4: the recipe holds across heterogeneity and scale) OVER the §6 ablation-ladder
rungs, REUSING #55's harness (`run_federated_simulation`) and runner (`run_ablation_ladder`). The sweeps
split across the RFC-0001 §3 module band (eval L6 may not import federation L7): the COMPOSE side — the
synthetic non-IID partition, the seeded pair-sampling — lives in `lensemble.eval.sweeps`; the DRIVERS that
call the harness live one band up in `lensemble.federation.sweeps`.

The non-IID severity axis is SYNTHETIC: the real `stable-worldmodel` factors-of-variation are DEFERRED
(vendoring is maintainer-gated, #96), so the partition shifts each silo's synthetic toy distribution by a
per-silo mean offset scaled by the severity. A `factor` other than `"synthetic"` fail-closes with a clear
`EvaluationError` — the documented seam for the real factors-of-variation path (#96).

Placed in tests/ml: the §8 CI gate scans tests/{unit,property,integration,ml,e2e,regression}. The issue
named `tests/eval/` which the CI gate does NOT scan, so (like `tests/ml/test_ablation_ladder.py`) the test
lives in tests/ml. Everything is seeded, CPU-only, downloads nothing, with tiny dims/rounds/silos so the
suite stays fast; the sweeps assert the LOAD-BEARING trend (with a margin), never a flaky exact curve.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TypedDict

import pytest
import torch

from lensemble.config import LensembleConfig
from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
from lensemble.errors import EvaluationError
from lensemble.eval.ablation import RungReport
from lensemble.eval.sweeps import (
    SiloPartition,
    partition_synthetic_noniid,
    sample_drift_pairs,
)
from lensemble.federation.sweeps import (
    non_iid_severity_sweep,
    participant_horizon_sweep,
    scale_sweep,
)

# --- the tiny CPU model config (mirrors tests/ml/test_ablation_ladder shape) ---

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


def _base_cfg(
    *, participant_count: int = 2, inner_horizon: int = 30
) -> LensembleConfig:
    """A coordinator-mode config with the tiny model, DP off, a quorum matching the silo count."""
    base = LensembleConfig()
    federation = dataclasses.replace(
        base.federation,
        inner_horizon=inner_horizon,
        participant_count=participant_count,
        fault_tolerance_min_participants=participant_count,
        secure_agg_threshold=participant_count,
        outer_lr=0.1,
        outer_nesterov_momentum=0.0,
    )
    privacy = dataclasses.replace(base.privacy, enabled=False)
    objective = dataclasses.replace(base.objective, lambda_anc=0.0)
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


class _PartitionKwargs(TypedDict):
    """The clip-shape kwargs the synthetic partition needs (a precise type so ``**`` keeps ``factor: str``)."""

    num_windows: int
    window_steps: int
    num_frames: int
    in_channels: int
    image_size: int
    action_dim: int


def _partition_kwargs() -> _PartitionKwargs:
    """The clip-shape kwargs the synthetic partition needs to match the tiny model config."""
    return _PartitionKwargs(
        num_windows=4,
        window_steps=_WINDOW_STEPS,
        num_frames=_T,
        in_channels=_C,
        image_size=_H,
        action_dim=_ACTION_DIM,
    )


# --- the synthetic non-IID partition (the compose side, eval L6) ---


def test_partition_returns_one_silo_partition_per_silo_with_right_shapes() -> None:
    silos = partition_synthetic_noniid(
        2, severity=1.0, seed=7, action_spec=_spec(), **_partition_kwargs()
    )
    assert len(silos) == 2
    for s in silos:
        assert isinstance(s, SiloPartition)
        assert len(s.windows) == 4
        for w in s.windows:
            assert w.obs.shape == (_WINDOW_STEPS + 1, _T, _C, _H, _W)
            assert w.actions.shape == (_WINDOW_STEPS, _ACTION_DIM)
        # Distinct residency roots per silo (INV-COMMIT-BINDING) and distinct ids.
    assert silos[0].participant_id != silos[1].participant_id
    assert silos[0].dataset_root != silos[1].dataset_root


def test_partition_is_deterministic_for_a_fixed_seed() -> None:
    a = partition_synthetic_noniid(
        2, severity=1.0, seed=7, action_spec=_spec(), **_partition_kwargs()
    )
    b = partition_synthetic_noniid(
        2, severity=1.0, seed=7, action_spec=_spec(), **_partition_kwargs()
    )
    for sa, sb in zip(a, b, strict=True):
        for wa, wb in zip(sa.windows, sb.windows, strict=True):
            assert torch.equal(wa.obs, wb.obs)


def test_partition_severity_zero_makes_all_silos_share_the_distribution() -> None:
    # severity 0 => near-IID: every silo draws the SAME synthetic distribution (identical per-silo data),
    # so the inter-silo mean offset is zero. severity 1 => each silo's mean is shifted by its own factor.
    near_iid = partition_synthetic_noniid(
        3, severity=0.0, seed=3, action_spec=_spec(), **_partition_kwargs()
    )
    means = [torch.stack([w.obs for w in s.windows]).mean() for s in near_iid]
    # All silo means coincide at severity 0 (same draw); they diverge at severity 1.
    spread_iid = max(means) - min(means)
    strong = partition_synthetic_noniid(
        3, severity=1.0, seed=3, action_spec=_spec(), **_partition_kwargs()
    )
    means_strong = [torch.stack([w.obs for w in s.windows]).mean() for s in strong]
    spread_strong = max(means_strong) - min(means_strong)
    assert spread_iid < 1e-6
    assert spread_strong > spread_iid + 0.5


def test_partition_non_synthetic_factor_fails_closed_for_the_real_factor_seam() -> None:
    # The #96 seam: the real stable-worldmodel factors-of-variation are deferred (maintainer-gated
    # vendoring). A non-"synthetic" factor fail-closes with a clear EvaluationError rather than silently
    # falling back to the synthetic partition.
    with pytest.raises(EvaluationError, match="stable-worldmodel"):
        partition_synthetic_noniid(
            2,
            severity=1.0,
            seed=1,
            action_spec=_spec(),
            factor="embodiment",
            **_partition_kwargs(),
        )


def test_partition_rejects_bad_silo_count_and_severity() -> None:
    # Defensive validation: a non-positive silo count and an out-of-[0,1] severity are caller errors.
    with pytest.raises(ValueError, match="num_silos"):
        partition_synthetic_noniid(
            0, severity=1.0, seed=1, action_spec=_spec(), **_partition_kwargs()
        )
    with pytest.raises(ValueError, match="severity"):
        partition_synthetic_noniid(
            2, severity=1.5, seed=1, action_spec=_spec(), **_partition_kwargs()
        )


# --- the non-IID severity sweep (the driver, federation L7) ---


def test_non_iid_severity_sweep_runs_and_reports_finite_drift() -> None:
    cfg = _base_cfg()
    result = non_iid_severity_sweep(
        cfg, severities=[0.0, 1.0], num_silos=2, num_rounds=2, seed=11
    )
    assert set(result) == {0.0, 1.0}
    for reports in result.values():
        for report in reports.values():
            assert isinstance(report, RungReport)
            assert torch.isfinite(torch.tensor(report.frame_drift_angle_deg))
            assert report.frame_drift_angle_deg >= 0.0


def test_higher_severity_drifts_more_for_naive_and_anchor_stays_lower() -> None:
    # THE LOAD-BEARING CLAIM (RFC-0005 §7 / Claim 4): a stronger non-IID partition yields MORE inter-silo
    # frame drift for the NAIVE rung (the per-silo distribution shift pushes the unconstrained frames
    # further apart), while the ANCHORED rung stays lower than naive at high severity (the Variant-A anchor
    # pins each frame onto the round-0 reference regardless of the data shift). We assert the directional
    # claim with a margin — not a flaky exact curve — with everything seeded so it is deterministic.
    cfg = _base_cfg()
    result = non_iid_severity_sweep(
        cfg, severities=[0.0, 1.0], num_silos=2, num_rounds=3, seed=11
    )
    naive_iid = result[0.0]["naive-fedavg"].frame_drift_angle_deg
    naive_strong = result[1.0]["naive-fedavg"].frame_drift_angle_deg
    anchored_strong = result[1.0]["frame-anchor"].frame_drift_angle_deg

    # Claim 1: stronger non-IID => more naive drift (by a clear margin).
    assert naive_strong > naive_iid + 3.0, (
        f"naive drift should grow with non-IID severity: iid={naive_iid:.3f} deg, "
        f"strong={naive_strong:.3f} deg (RFC-0005 §7)"
    )
    # Claim 2: at high severity the anchored rung stays well below naive (the anchor holds the frame).
    assert anchored_strong < naive_strong - 5.0, (
        f"at strong non-IID the anchored rung ({anchored_strong:.3f} deg) should stay well below "
        f"naive ({naive_strong:.3f} deg) (RFC-0005 §7)"
    )


def test_non_iid_severity_sweep_propagates_the_real_factor_seam() -> None:
    cfg = _base_cfg()
    with pytest.raises(EvaluationError, match="stable-worldmodel"):
        non_iid_severity_sweep(
            cfg,
            severities=[1.0],
            num_silos=2,
            num_rounds=1,
            seed=1,
            factor="embodiment",
        )


# --- the participant-count C and inner-horizon H sweep ---


def test_participant_horizon_sweep_runs_across_points_with_finite_drift() -> None:
    cfg = _base_cfg()
    result = participant_horizon_sweep(
        cfg, counts=[2], horizons=[10, 40], num_rounds=2, seed=5
    )
    assert set(result) == {(2, 10), (2, 40)}
    for reports in result.values():
        for report in reports.values():
            assert torch.isfinite(torch.tensor(report.frame_drift_angle_deg))
            assert report.frame_drift_angle_deg >= 0.0


def test_longer_inner_horizon_drifts_more_for_naive() -> None:
    # RFC-0002 §2.1 / RFC-0005 §7: a longer inner horizon H rotates the per-silo frames further apart
    # before the outer step, so the NAIVE rung's inter-silo drift grows with H. We assert the directional
    # claim with a margin (seeded, deterministic), comparing a short vs a long H at fixed C.
    cfg = _base_cfg()
    result = participant_horizon_sweep(
        cfg, counts=[2], horizons=[8, 48], num_rounds=3, seed=5
    )
    short_h = result[(2, 8)]["naive-fedavg"].frame_drift_angle_deg
    long_h = result[(2, 48)]["naive-fedavg"].frame_drift_angle_deg
    assert long_h > short_h + 3.0, (
        f"a longer inner horizon should rotate frames further apart: H=8 -> {short_h:.3f} deg, "
        f"H=48 -> {long_h:.3f} deg (RFC-0002 §2.1)"
    )


# --- the scale step ---


def test_scale_sweep_runs_at_multiple_latent_dims_with_finite_metrics() -> None:
    # The recipe holds as the encoder grows: at each latent_dim the anchored rung stays low (the anchor
    # pins the frame regardless of scale). Tiny dims (8 -> 16) keep it CPU-fast; each is a coherent ViT
    # shape (num_heads=2 divides both 8 and 16; num_tokens is independent of latent_dim).
    cfg = _base_cfg()
    result = scale_sweep(cfg, latent_dims=[8, 16], num_rounds=2, seed=9)
    assert set(result) == {8, 16}
    for dim, reports in result.items():
        for report in reports.values():
            assert torch.isfinite(torch.tensor(report.frame_drift_angle_deg)), dim
            assert report.frame_drift_angle_deg >= 0.0, dim
        # The recipe holds: the anchored rung stays below the naive rung at every scale.
        anchored = reports["frame-anchor"].frame_drift_angle_deg
        naive = reports["naive-fedavg"].frame_drift_angle_deg
        assert anchored < naive, f"anchor should hold at latent_dim={dim}"


# --- seeded pair-sampling for the O(C^2) drift diagnostic at large C ---


def test_sample_drift_pairs_is_deterministic_and_bounded() -> None:
    ids = [f"silo-{i}" for i in range(8)]  # 8C2 = 28 pairs; we sample a bounded subset
    a = sample_drift_pairs(ids, max_pairs=5, seed=42)
    b = sample_drift_pairs(ids, max_pairs=5, seed=42)
    assert a == b  # same seed => same pairs
    assert len(a) == 5  # bounded by max_pairs
    for x, y in a:
        assert x in ids and y in ids and x != y  # valid, distinct, unordered-within
    # No duplicate pairs (each unordered pair at most once).
    norm = {tuple(sorted(p)) for p in a}
    assert len(norm) == len(a)


def test_sample_drift_pairs_different_seed_gives_different_set() -> None:
    ids = [f"silo-{i}" for i in range(8)]
    a = sample_drift_pairs(ids, max_pairs=5, seed=42)
    c = sample_drift_pairs(ids, max_pairs=5, seed=43)
    assert a != c


def test_sample_drift_pairs_caps_at_total_pairs_when_max_exceeds() -> None:
    ids = ["a", "b", "c"]  # 3C2 = 3 total pairs
    pairs = sample_drift_pairs(ids, max_pairs=100, seed=1)
    assert len(pairs) == 3  # cannot exceed the C-choose-2 total
    assert {tuple(sorted(p)) for p in pairs} == {("a", "b"), ("a", "c"), ("b", "c")}


def test_sample_drift_pairs_empty_when_no_pairs_or_zero_budget() -> None:
    # No pairs to draw (a single id => zero unordered pairs) or a zero budget => an empty sample.
    assert sample_drift_pairs(["solo"], max_pairs=5, seed=1) == []
    assert sample_drift_pairs(["a", "b", "c"], max_pairs=0, seed=1) == []


def test_sampled_pairs_are_recordable_in_a_run_manifest() -> None:
    # The sampled pair set must be RECORDED so the drift figure stays reproducible (RFC-0005 §7/§8). The
    # RunManifest is the reproducibility sink; the sampled pairs serialize to a manifest-native dict value
    # (a list of [a, b] lists) without tripping the residency redaction guard (no tensors, just ids).
    from lensemble.config.manifest import build_manifest
    from lensemble.observability import redact

    ids = [f"silo-{i}" for i in range(8)]
    pairs = sample_drift_pairs(ids, max_pairs=4, seed=7)
    recorded = {"sampled_drift_pairs": [list(p) for p in pairs]}
    # The manifest's residency guard must accept the id-only record (it fails closed on tensors).
    redact(recorded, field="sampled_drift_pairs")
    manifest = build_manifest(_base_cfg())
    assert manifest.config_hash  # the manifest builds (the record rides alongside it)


# --- disk hygiene: the sweeps (many runs) leak NO temp dirs ---


def test_sweeps_leak_no_temp_dirs() -> None:
    # #55's harness rmtree's each Coordinator's tempfile.mkdtemp artifacts dir on exit; the sweeps build
    # MANY coordinators, so a leak would fill the disk. We assert the OS temp dir gains no lensemble-* dir
    # across a full (small) non-IID severity sweep.
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.gettempdir())

    def _lensemble_dirs() -> set[str]:
        return {p.name for p in tmp.glob("lensemble-*") if p.is_dir()}

    before = _lensemble_dirs()
    cfg = _base_cfg()
    non_iid_severity_sweep(
        cfg, severities=[0.0, 1.0], num_silos=2, num_rounds=1, seed=2
    )
    after = _lensemble_dirs()
    assert after == before, f"leaked temp dirs: {sorted(after - before)}"
