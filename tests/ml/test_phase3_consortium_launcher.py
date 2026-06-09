"""CPU smoke test for the Phase 3 consortium HF Jobs launcher (#241).

Mirrors ``tests/e2e/test_toy_pipeline.py``: it generates tiny per-participant + held-out toy datasets (the
proven CPU V-JEPA shape: ``latent_dim=8``, ``num_tokens=4``), loads
``deploy/hfjobs/train_phase3_consortium.py`` via ``importlib`` and drives its ``main([...])`` entry point.

Two paths are covered:

1. ``--dry-run`` pins the public-probe hash, builds + validates the manifest and dataset/probe registry,
   preflights every participant agent, and writes ``phase3_consortium_dry_run.json`` WITHOUT running any
   federated round (no ``coordinator-artifacts`` round dirs are created).
2. A real tiny ``--num-rounds 2`` run drives the full ``Phase3CoordinatorService`` +
   ``Phase3ParticipantAgent`` runtime and emits REAL per-round JEPA metrics. The run is deterministic
   (two identical runs into different dirs agree on every per-round metric tuple) and residency-safe (the
   serialized report carries no raw-data keys).

Four participants with quorum 3 (``--min-trainers 3``, ``--secure-agg-threshold 3``) match the Phase 3
long-run smoke shape. The model is kept tiny so the whole run is CPU-fast and downloads nothing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
from lensemble.data import Episode, EpisodeDataset, Transition, save_episodes

# --- the tiny consistent CPU shape (mirrors the toy pipeline) ---
_T, _C, _H, _W = 2, 3, 4, 4
_ACTION_DIM = 2
_WINDOW_STEPS = 1
_LATENT_DIM = 8


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


def _episode(episode_id: str, n_transitions: int, seed: int) -> Episode:
    gen = torch.Generator().manual_seed(seed)
    transitions: list[Transition] = []
    for _ in range(n_transitions):
        transitions.append(
            Transition(
                obs_t=torch.randn(_T, _C, _H, _W, generator=gen),
                action_t=torch.randn(_ACTION_DIM, generator=gen),
                obs_tp1=torch.randn(_T, _C, _H, _W, generator=gen),
            )
        )
    return Episode(
        episode_id=episode_id,
        transitions=transitions,
        embodiment_id="toy",
        modality="rgb-video",
        action_spec=_spec(),
        collection_meta={"site": "phase3-consortium-test"},
    )


def _write_dataset(path: Path, *, seeds: tuple[int, ...], fmt: str = "hdf5") -> Path:
    dataset = EpisodeDataset(
        [_episode(f"ep-{i}", 3, seed=seed) for i, seed in enumerate(seeds)],
        fmt=fmt,  # type: ignore[arg-type]
    )
    save_episodes(dataset, path, fmt=fmt)  # type: ignore[arg-type]
    return path


def _load_launcher() -> Any:
    path = (
        Path(__file__).resolve().parents[2]
        / "deploy"
        / "hfjobs"
        / "train_phase3_consortium.py"
    )
    spec = importlib.util.spec_from_file_location("lensemble_phase3_consortium", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_silos(tmp_path: Path) -> tuple[list[Path], Path]:
    silos = [
        _write_dataset(tmp_path / f"silo{i}.h5", seeds=(10 * (i + 1), 10 * (i + 1) + 1))
        for i in range(4)
    ]
    heldout = _write_dataset(tmp_path / "heldout.h5", seeds=(900, 901))
    return silos, heldout


def _tiny_argv(silos: list[Path], heldout: Path, out_dir: Path) -> list[str]:
    argv: list[str] = []
    for silo in silos:
        argv += ["--data-source", str(silo)]
    argv += [
        "--heldout-source",
        str(heldout),
        "--data-format",
        "hdf5",
        "--out-dir",
        str(out_dir),
        "--num-rounds",
        "2",
        "--inner-horizon",
        "1",
        "--window-steps",
        str(_WINDOW_STEPS),
        "--latent-dim",
        str(_LATENT_DIM),
        "--depth",
        "1",
        "--predictor-depth",
        "1",
        "--num-heads",
        "2",
        "--image-size",
        str(_H),
        "--patch-size",
        "2",
        "--num-frames",
        str(_T),
        "--tubelet",
        "2",
        "--lambda-anc",
        "0.0",
        "--secure-agg-threshold",
        "3",
        "--min-trainers",
        "3",
        "--metric-windows",
        "2",
    ]
    return argv


def test_dry_run_pins_probe_and_validates_without_compute(tmp_path: Path) -> None:
    module = _load_launcher()
    silos, heldout = _write_silos(tmp_path)
    out_dir = tmp_path / "dry-run"
    argv = _tiny_argv(silos, heldout, out_dir) + ["--dry-run"]

    payload = module.main(argv)

    assert payload["dry_run"] is True
    assert len(payload["participant_ids"]) == 4
    probe_hash = payload["public_probe_hash"]
    assert len(probe_hash) == 64
    assert all(c in "0123456789abcdef" for c in probe_hash)
    assert set(payload["window_counts"]) == set(payload["participant_ids"])
    assert all(count >= 1 for count in payload["window_counts"].values())

    # The dry-run wrote both governance artifacts and the dry-run JSON.
    assert Path(payload["manifest_path"]).exists()
    assert Path(payload["registry_path"]).exists()
    dry_run_json = out_dir / "phase3_consortium_dry_run.json"
    assert dry_run_json.exists()
    on_disk = json.loads(dry_run_json.read_text(encoding="utf-8"))
    assert on_disk["public_probe_hash"] == probe_hash
    assert any(
        check.startswith("public_probe_hash_pinned:") for check in on_disk["checks"]
    )

    # No federated round ran: no coordinator-artifacts round dirs exist.
    artifacts = out_dir / "coordinator-artifacts"
    round_dirs = list(artifacts.glob("round-*")) if artifacts.exists() else []
    assert round_dirs == []


def _metric_tuples(report_path: Path) -> list[tuple[float, float, float, float]]:
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    tuples: list[tuple[float, float, float, float]] = []
    for row in raw["rounds"]:
        assert row["val_pred"] is not None
        assert row["val_sigreg"] is not None
        assert row["effective_rank"] is not None
        assert row["frame_drift_deg"] is not None
        tuples.append(
            (
                float(row["val_pred"]),
                float(row["val_sigreg"]),
                float(row["effective_rank"]),
                float(row["frame_drift_deg"]),
            )
        )
    return tuples


def test_real_run_emits_real_per_round_metrics_deterministically(
    tmp_path: Path,
) -> None:
    module = _load_launcher()
    silos, heldout = _write_silos(tmp_path)

    out_a = tmp_path / "run-a"
    summary_a = module.main(_tiny_argv(silos, heldout, out_a))
    assert summary_a["dry_run"] is False
    assert summary_a["closed_rounds"] == 2
    assert summary_a["completed_target"] is True
    assert summary_a["pushed"] is False

    report_a = out_a / "phase3_long_run_smoke_report.json"
    assert report_a.exists()
    tuples_a = _metric_tuples(report_a)
    assert len(tuples_a) == 2

    # Determinism: a second identical run into a different dir reproduces every metric tuple.
    out_b = tmp_path / "run-b"
    module.main(_tiny_argv(silos, heldout, out_b))
    report_b = out_b / "phase3_long_run_smoke_report.json"
    tuples_b = _metric_tuples(report_b)
    assert tuples_a == tuples_b

    # The coordinator committed real per-round checkpoints.
    rounds = list((out_a / "coordinator-artifacts").glob("round-*"))
    assert len(rounds) >= 2

    # Residency: the serialized report carries no raw-data JSON keys. The bare substrings are checked as
    # quoted keys so legitimate values like "hf_jobs_release" (which contains "obs") do not false-positive.
    raw_report = json.loads(report_a.read_text(encoding="utf-8"))

    def _keys(node: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(node, dict):
            for key, value in node.items():
                found.add(key)
                found |= _keys(value)
        elif isinstance(node, list):
            for item in node:
                found |= _keys(item)
        return found

    report_keys = _keys(raw_report)
    for forbidden in ("obs", "observation", "trajectory"):
        assert forbidden not in report_keys


def test_outer_step_and_anchor_knobs_thread_into_coordinator_config(
    tmp_path: Path,
) -> None:
    """#263/#261: --outer-lr/--outer-momentum/--anchor-variant/--lambda-anc reach the coordinator config.

    The DiLoCo outer step (``OuterOptimizer(lr, momentum)``) is built from ``cfg.federation.outer_lr`` /
    ``outer_nesterov_momentum`` and the frame anchor from ``cfg.objective.lambda_anc`` / ``anchor_variant``,
    so threading the launcher knobs into those config fields is what makes them load-bearing for the run.
    The defaults encode the M1 decisions: the tuned anchor strength (1.0, not the #249 0.01) and a
    conservative outer step (smaller lr, zero Nesterov momentum) that does not amplify a partial aggregate.
    """
    module = _load_launcher()
    args = module._args(
        [
            "--data-source",
            "lerobot-h5:///tmp/x.h5",
            "--outer-lr",
            "0.3",
            "--outer-momentum",
            "0.1",
            "--anchor-variant",
            "rotational",
            "--lambda-anc",
            "0.7",
        ]
    )
    cfg = module._coordinator_cfg(
        args, probe_path=tmp_path / "probe.safetensors", participant_count=4
    )
    assert cfg.federation.outer_lr == 0.3
    assert cfg.federation.outer_nesterov_momentum == 0.1
    assert cfg.objective.anchor_variant == "rotational"
    assert cfg.objective.lambda_anc == 0.7

    defaults = module._args(["--data-source", "lerobot-h5:///tmp/x.h5"])
    assert defaults.lambda_anc == 1.0  # tuned strength, not the #249 0.01
    assert defaults.outer_lr == 0.5  # conservative real-run outer step
    assert defaults.outer_momentum == 0.0  # zero Nesterov momentum
    assert defaults.anchor_variant == "landmark"


def test_encoder_backend_threads_into_model_config() -> None:
    """RFC-0017 dynamic-env jobs can request a from-scratch encoder without mutating legacy defaults."""
    module = _load_launcher()

    defaults = module._model_cfg(
        module._args(["--data-source", "synthetic-dynamic://swipe-dot?seed=1"])
    )
    assert defaults.encoder == "vjepa2-vit-l"

    scratch = module._model_cfg(
        module._args(
            [
                "--data-source",
                "synthetic-dynamic://swipe-dot?seed=1",
                "--encoder",
                "scratch",
            ]
        )
    )
    assert scratch.encoder == "scratch"


def test_warm_start_fork_a_freezes_encoder_and_federates_predictor(
    tmp_path: Path,
) -> None:
    """2-phase Fork-A (#259 MVP): Phase 2 warm-starts from a committed checkpoint and, with
    ``--encoder-frozen``, holds the encoder byte-identical while federating only the predictor.
    """
    from lensemble.eval.jepa_metrics import load_checkpoint_groups

    module = _load_launcher()
    silos, heldout = _write_silos(tmp_path)

    # Phase 1: a tiny federated run → committed round-0002 checkpoint. DP off so the frozen-encoder
    # assertion below is exact (with DP on, Gaussian noise is added to every released delta — including a
    # frozen encoder's zero delta — so the committed encoder would move by noise, not by training).
    out1 = tmp_path / "phase1"
    module.main(_tiny_argv(silos, heldout, out1) + ["--no-privacy"])
    warm_ckpt = out1 / "coordinator-artifacts" / "round-00002"
    assert (warm_ckpt / "weights.safetensors").exists()

    # Phase 2: warm-start from Phase 1 + freeze the encoder (Fork A); DP off.
    out2 = tmp_path / "phase2"
    summary = module.main(
        _tiny_argv(silos, heldout, out2)
        + ["--warm-start", str(warm_ckpt), "--encoder-frozen", "--no-privacy"]
    )
    assert summary["closed_rounds"] == 2

    manifest = json.loads(
        (out2 / "phase3_consortium_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model"]["base_checkpoint_ref"] == str(warm_ckpt)

    # Fork A: the committed encoder is byte-identical to the warm-start (frozen); the predictor moved.
    ws_theta, ws_phi = load_checkpoint_groups(warm_ckpt)
    fin_theta, fin_phi = load_checkpoint_groups(
        out2 / "coordinator-artifacts" / "round-00002"
    )
    for k in ws_theta:
        assert torch.equal(ws_theta[k], fin_theta[k]), f"frozen encoder param {k} moved"
    assert any(not torch.equal(ws_phi[k], fin_phi[k]) for k in ws_phi), (
        "the federated predictor should have changed"
    )


def test_custom_outer_step_run_is_deterministic(tmp_path: Path) -> None:
    """#263 CPU smoke: a run with custom --outer-lr/--outer-momentum closes and is bitwise-reproducible.

    The knobs flow into the persistent ``OuterOptimizer`` the coordinator steps each round; a second
    identical run into a fresh dir reproduces every per-round metric tuple (``INV-AGG-DETERMINISM``).
    """
    module = _load_launcher()
    silos, heldout = _write_silos(tmp_path)
    extra = ["--outer-lr", "0.3", "--outer-momentum", "0.1"]

    out_a = tmp_path / "outer-a"
    summary_a = module.main(_tiny_argv(silos, heldout, out_a) + extra)
    assert summary_a["closed_rounds"] == 2
    tuples_a = _metric_tuples(out_a / "phase3_long_run_smoke_report.json")

    out_b = tmp_path / "outer-b"
    module.main(_tiny_argv(silos, heldout, out_b) + extra)
    tuples_b = _metric_tuples(out_b / "phase3_long_run_smoke_report.json")
    assert tuples_a == tuples_b
