"""The public surface of ``lensemble`` matches conventions 5 / 02-public-api 1 (issue #2)."""

from __future__ import annotations

import inspect
import re
from dataclasses import fields
from pathlib import Path

import pytest
import torch

import lensemble
from lensemble.data import load_episodes

# The current pre-1.0 top-level re-export set (docs/spec/02-public-api.md 1).
PUBLIC_SURFACE = [
    "LensembleConfig",
    "RunManifest",
    "load",
    "train_local",
    "Coordinator",
    "Participant",
    "RoundState",
    "build_encoder",
    "build_predictor",
    "build_action_head",
    "Objective",
    "evaluate",
    "Planner",
    "frame_drift",
    "procrustes_align",
    "commit_dataset",
    "DatasetCommitment",
    "ContributionLedger",
    "recompute_alignment",
]

# Permissive SemVer (MAJOR.MINOR.PATCH with optional pre-release / build metadata).
_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.\-]+)?$")
_API_DOC = Path(__file__).resolve().parents[2] / "docs" / "spec" / "02-public-api.md"

_TINY_CPU_OVERRIDES = [
    "model.encoder=scratch",
    "model.latent_dim=8",
    "model.num_tokens=4",
    "model.predictor_depth=1",
    "model.predictor_width=8",
    "model.num_frames=1",
    "model.tubelet=1",
    "model.image_size=4",
    "model.patch_size=2",
    "model.depth=1",
    "model.num_heads=2",
    "objective.lambda_anc=0.0",
    "objective.sigreg_sketch_dim=8",
    "gauge.anchor_landmark_count=8",
    "federation.inner_horizon=1",
    "data.format=synthetic-dynamic",
    (
        "data.data_source=synthetic-dynamic://swipe-dot"
        "?seed=0&n_episodes=1&steps=2&image_size=4"
    ),
    "eval.env_id=synthetic://toy",
    "eval.planning_samples=8",
    "eval.horizon=2",
]


def test_version_is_semver() -> None:
    assert isinstance(lensemble.__version__, str)
    assert _SEMVER.match(lensemble.__version__), lensemble.__version__


def test_public_surface_importable() -> None:
    for name in PUBLIC_SURFACE:
        assert hasattr(lensemble, name), f"missing public symbol: {name}"


def test_all_advertises_public_surface() -> None:
    for name in ["__version__", *PUBLIC_SURFACE]:
        assert name in lensemble.__all__, f"{name} absent from __all__"


def test_unknown_attribute_raises() -> None:
    import pytest

    with pytest.raises(AttributeError):
        _ = lensemble.definitely_not_a_public_symbol


def test_all_matches_export_map() -> None:
    # The literal __all__ must not drift from the lazy-export map (_EXPORTS).
    assert set(lensemble.__all__) == {"__version__", *lensemble._EXPORTS}


def test_public_surface_matches_all() -> None:
    assert set(PUBLIC_SURFACE) == set(lensemble.__all__) - {"__version__"}


def test_module_tagline_is_pre_one_point_zero_research_scope() -> None:
    assert lensemble.__doc__ is not None
    assert lensemble.__doc__.startswith(
        "Lensemble: research toolkit for federated JEPA-style world-model experiments."
    )
    assert "frozen public surface" not in lensemble.__doc__


def test_documented_public_signatures_match_runtime() -> None:
    signatures = {
        "load": tuple(inspect.signature(lensemble.load).parameters),
        "train_local": tuple(inspect.signature(lensemble.train_local).parameters),
        "Coordinator": tuple(inspect.signature(lensemble.Coordinator).parameters),
        "Participant": tuple(inspect.signature(lensemble.Participant).parameters),
        "Objective": tuple(inspect.signature(lensemble.Objective).parameters),
        "evaluate": tuple(inspect.signature(lensemble.evaluate).parameters),
        "Planner": tuple(inspect.signature(lensemble.Planner).parameters),
        "frame_drift": tuple(inspect.signature(lensemble.frame_drift).parameters),
        "procrustes_align": tuple(
            inspect.signature(lensemble.procrustes_align).parameters
        ),
        "recompute_alignment": tuple(
            inspect.signature(lensemble.recompute_alignment).parameters
        ),
    }
    assert signatures == {
        "load": ("config_name", "overrides", "config_dir"),
        "train_local": ("config", "run_dir"),
        "Coordinator": (
            "config",
            "transport",
            "artifacts_dir",
            "enable_backstop",
            "warm_start",
        ),
        "Participant": ("config", "participant_id", "transport"),
        "Objective": (
            "lambda_pred",
            "lambda_sig",
            "lambda_anc",
            "sketch_seed",
            "sketch_dim",
            "ep_knots",
            "anchor",
            "sketch",
            "target_stop_gradient",
        ),
        "evaluate": (
            "checkpoint",
            "env_id",
            "cfg",
            "num_episodes",
            "planner_iters",
        ),
        "Planner": (
            "family",
            "horizon",
            "num_samples",
            "action_dim",
            "seed",
            "num_iters",
            "elite_frac",
            "init_std",
            "temperature",
        ),
        "frame_drift": (
            "embeddings",
            "round_index",
            "probe",
            "expected_probe_hash",
            "degenerate_safe",
        ),
        "procrustes_align": ("source", "target", "singular_floor"),
        "recompute_alignment": ("committed_weights", "probe"),
    }

    assert (
        inspect.signature(lensemble.Coordinator).parameters["transport"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        inspect.signature(lensemble.Participant).parameters["participant_id"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        inspect.signature(lensemble.Objective).parameters["lambda_pred"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


def test_public_api_doc_excludes_stale_planned_signatures() -> None:
    text = _API_DOC.read_text(encoding="utf-8")
    stale_fragments = (
        "lensemble.config.load(Path(",
        "Participant(cfg, dataset=",
        "Coordinator(cfg, participants=",
        "obj = Objective(cfg)",
        "result.manifest.config_hash",
        "result.checkpoint.content_hash",
        "commitment.root",
        'Path("ckpts/round_200.safetensors")',
    )
    for fragment in stale_fragments:
        assert fragment not in text

    assert "not frozen" in text
    assert "run-manifest/skeleton-0" in text
    assert "Manifest-only stub" in text


def test_documented_cpu_quickstart_contract(tmp_path: Path) -> None:
    cfg = lensemble.load(overrides=list(_TINY_CPU_OVERRIDES))
    trained = lensemble.train_local(cfg, run_dir=tmp_path)

    assert tuple(field.name for field in fields(trained)) == (
        "checkpoint_dir",
        "checkpoint_hash",
        "manifest_hash",
        "final_loss",
    )
    assert (trained.checkpoint_dir / "weights.safetensors").is_file()
    assert (trained.checkpoint_dir / "header.json").is_file()
    assert re.fullmatch(r"[0-9a-f]{64}", trained.checkpoint_hash)
    assert re.fullmatch(r"[0-9a-f]{64}", trained.manifest_hash)
    assert torch.isfinite(torch.tensor(trained.final_loss))

    report = lensemble.evaluate(
        trained.checkpoint_dir,
        "synthetic://toy",
        cfg=cfg,
        num_episodes=2,
        planner_iters=1,
    )
    assert report.checkpoint_hash == trained.checkpoint_hash
    assert 0.0 <= report.success_rate <= 1.0
    assert report.planning_samples == 8
    assert report.effective_dim > 0.0
    assert report.probe_accuracy is None
    assert report.state_probe_r2 is None


def test_documented_gauge_example_uses_pair_fields() -> None:
    points = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    rotated = points @ rotation

    q_star, residual = lensemble.procrustes_align(points, rotated)
    assert torch.allclose(points @ q_star, rotated, atol=1e-6)
    assert residual == pytest.approx(0.0, abs=1e-6)

    report = lensemble.frame_drift({"site-a": points, "site-b": rotated}, round_index=0)
    assert report.probe_hash == ""
    assert len(report.pairs) == 1
    assert report.pairs[0].rotation_angle_deg == pytest.approx(90.0)
    assert report.pairs[0].procrustes_residual == pytest.approx(0.0, abs=1e-6)
    assert not hasattr(report, "mean_rotation_angle")


def test_documented_provenance_fields_and_ledger_constructor(tmp_path: Path) -> None:
    dataset = load_episodes(
        ("synthetic-dynamic://swipe-dot?seed=0&n_episodes=1&steps=2&image_size=4"),
        fmt="synthetic-dynamic",
    )
    commitment = lensemble.commit_dataset(dataset)
    assert re.fullmatch(r"[0-9a-f]{64}", commitment.merkle_root)
    assert commitment.episode_count == 1
    assert not hasattr(commitment, "root")

    ledger = lensemble.ContributionLedger(tmp_path / "ledger.jsonl", records=())
    assert ledger.records == ()
    assert ledger.verify_chain() is True
    assert tuple(inspect.signature(lensemble.ContributionLedger.open).parameters) == (
        "path",
    )
