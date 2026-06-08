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
