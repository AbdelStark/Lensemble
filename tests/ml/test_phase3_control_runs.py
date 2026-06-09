"""Phase 3 matched-control runs (#244): the two control knobs the consortium launcher exposes.

Two controls let a reviewer quantify what Phase 3 federation actually buys, both measured against the
SAME residency-safe per-round JEPA metrics the real run emits:

1. **Fork A — ``--encoder-frozen`` (RFC-0002).** Freeze ``f_theta`` at the warm start and federate only
   ``g_phi``. The launcher flag flips ``cfg.model.encoder_frozen``; the library inner loop
   (:func:`lensemble.federation.participant._inner_loop`) optimizes only ``requires_grad`` params, so a
   frozen encoder contributes a zero encoder-delta while the predictor still trains. The first test pins
   this at the library inner-loop level (encoder weights bit-unchanged, predictor weights moved); a second
   test drives a tiny real ``--encoder-frozen`` run end-to-end through the launcher.

2. **``--local-only`` (the no-aggregation baseline).** Train every participant in ISOLATION on only its
   own silo with NO coordinator aggregation, then report per-participant held-out metrics + the
   inter-participant latent frame-drift — the divergence federated aggregation is designed to close. The
   third test drives a tiny ``--local-only`` run and asserts it (a) writes ``phase3_local_only_report.json``
   with per-participant metrics + a ``frame_drift_deg``, (b) publishes one representative checkpoint for
   downstream eval without a coordinator ledger, and (c) is residency-safe (no raw-data keys in the report).

The toy CPU V-JEPA shape mirrors ``tests/ml/test_phase3_consortium_launcher.py`` and the toy pipeline
(``latent_dim=8``, ``num_tokens=4``) so the whole module is CPU-fast and downloads nothing.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lensemble.config import LensembleConfig
from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
from lensemble.data import Episode, EpisodeDataset, Transition, save_episodes
from lensemble.data.episode import Window
from lensemble.federation.participant import _inner_loop
from lensemble.model import build_encoder, build_predictor
from lensemble.model.action_head import build_action_head
from lensemble.model.objective import Objective

# --- the tiny consistent CPU shape (mirrors the toy pipeline + the consortium launcher smoke) ---
_T, _C, _H, _W = 2, 3, 4, 4
_ACTION_DIM = 2
_WINDOW_STEPS = 1
_LATENT_DIM = 8
_NUM_TOKENS = (
    4  # (num_frames//tubelet) * (image_size//patch_size)**2 = (2//2)*(4//2)**2 = 4
)
_NUM_STEPS = 2  # window obs is (num_steps + 1) clips


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


# --------------------------------------------------------------------------------------------------- #
# 1. Fork A: a frozen encoder produces a zero encoder-delta while the predictor still trains.
# --------------------------------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _ForkAModelConfig:
    """A tiny ModelConfig carrying both the real fields and the V-JEPA shape fields ``build_*`` reads."""

    encoder: str = "vjepa2-vit-l"
    warm_start_release: str = "vjepa2-2.0"
    latent_dim: int = _LATENT_DIM
    num_tokens: int = _NUM_TOKENS
    predictor_depth: int = 1
    predictor_width: int = _LATENT_DIM
    wmcp_version: str = WMCP_VERSION
    encoder_frozen: bool = False
    d: int = _LATENT_DIM
    in_channels: int = _C
    num_frames: int = _T
    image_size: int = _H
    patch_size: int = 2
    tubelet: int = 2
    depth: int = 1
    num_heads: int = 2
    cond_dim: int = _LATENT_DIM


def _fork_a_cfg(*, encoder_frozen: bool) -> LensembleConfig:
    base = LensembleConfig()
    model = _ForkAModelConfig(encoder_frozen=encoder_frozen)
    return dataclasses.replace(base, model=model, run_mode="participant")  # type: ignore[arg-type]


def _windows(seed: int = 1, count: int = 2) -> list[Window]:
    gen = torch.Generator().manual_seed(seed)
    out: list[Window] = []
    for _ in range(count):
        obs = torch.randn(_NUM_STEPS + 1, _T, _C, _H, _W, generator=gen)
        actions = torch.randn(_NUM_STEPS, _ACTION_DIM, generator=gen)
        out.append(
            Window(obs=obs, actions=actions, num_steps=_NUM_STEPS, embodiment_id="toy")
        )
    return out


def _bare_objective() -> Objective:
    # The bare LeJEPA objective (anchor=None) — enough to drive a real Δ on the predictor without needing
    # a probe/frame anchor. lambda_anc == 0.0 means no FrameAnchor is constructed.
    return Objective(
        lambda_pred=1.0,
        lambda_sig=0.1,
        lambda_anc=0.0,
        sketch_seed=7,
        sketch_dim=int(_LATENT_DIM),
        anchor=None,
        target_stop_gradient=False,
    )


def test_fork_a_frozen_encoder_yields_zero_encoder_delta() -> None:
    # Fork A (RFC-0002): with model.encoder_frozen=True, the shared inner loop must leave EVERY encoder
    # weight bit-identical to the warm start (a zero encoder-delta) while still training the predictor.
    cfg = _fork_a_cfg(encoder_frozen=True)
    encoder = build_encoder(cfg)
    predictor = build_predictor(cfg)
    action_head = build_action_head(cfg, _spec())

    # build_encoder freezes every encoder param when encoder_frozen=True (INV-ACTIONHEAD-LOCAL / Fork A).
    assert all(not p.requires_grad for p in encoder.parameters())
    assert any(p.requires_grad for p in predictor.parameters())

    enc_before = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    pred_before = {k: v.detach().clone() for k, v in predictor.state_dict().items()}

    _inner_loop(
        encoder,
        predictor,
        action_head,
        _bare_objective(),
        _windows(),
        horizon=3,
        lr=1e-2,
    )

    # The encoder weights are BIT-unchanged: a frozen f_theta contributes a zero encoder-delta.
    enc_after = encoder.state_dict()
    for name, before in enc_before.items():
        delta = (enc_after[name].detach() - before).abs()
        assert float(delta.max()) < 1e-6, (
            f"frozen encoder weight {name} moved by {float(delta.max())}"
        )

    # The predictor weights DID move (g_phi is still federated/trained): a nonzero predictor-delta exists.
    pred_after = predictor.state_dict()
    moved = max(
        float((pred_after[name].detach() - before).abs().max())
        for name, before in pred_before.items()
    )
    assert moved > 1e-6, (
        "predictor weights did not move; the inner loop trained nothing"
    )


def test_fork_a_unfrozen_encoder_does_train_the_encoder() -> None:
    # The control's control: with encoder_frozen=False the SAME inner loop DOES move encoder weights, so
    # the frozen assertion above is meaningful (it is not a no-op for an untrained loop).
    cfg = _fork_a_cfg(encoder_frozen=False)
    encoder = build_encoder(cfg)
    predictor = build_predictor(cfg)
    action_head = build_action_head(cfg, _spec())

    assert any(p.requires_grad for p in encoder.parameters())

    enc_before = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    _inner_loop(
        encoder,
        predictor,
        action_head,
        _bare_objective(),
        _windows(),
        horizon=3,
        lr=1e-2,
    )
    enc_after = encoder.state_dict()
    moved = max(
        float((enc_after[name].detach() - before).abs().max())
        for name, before in enc_before.items()
    )
    assert moved > 1e-6, (
        "an unfrozen encoder must train; the inner loop moved no encoder weight"
    )


# --------------------------------------------------------------------------------------------------- #
# 2 + 3. Launcher smoke: --encoder-frozen and --local-only drive the real launcher end-to-end.
# --------------------------------------------------------------------------------------------------- #


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
        collection_meta={"site": "phase3-control-runs-test"},
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


def _write_silos(tmp_path: Path, *, count: int) -> tuple[list[Path], Path]:
    silos = [
        _write_dataset(tmp_path / f"silo{i}.h5", seeds=(10 * (i + 1), 10 * (i + 1) + 1))
        for i in range(count)
    ]
    heldout = _write_dataset(tmp_path / "heldout.h5", seeds=(900, 901))
    return silos, heldout


def _tiny_argv(
    silos: list[Path], heldout: Path, out_dir: Path, *, min_trainers: int
) -> list[str]:
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
        str(min_trainers),
        "--min-trainers",
        str(min_trainers),
        "--metric-windows",
        "2",
    ]
    return argv


def _all_keys(node: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            found.add(key)
            found |= _all_keys(value)
    elif isinstance(node, list):
        for item in node:
            found |= _all_keys(item)
    return found


def test_encoder_frozen_real_run_completes_with_metrics(tmp_path: Path) -> None:
    # Fork A end-to-end: a tiny real run with --encoder-frozen completes and produces a report whose
    # rounds carry metrics (determinism is not required here — only that it runs and reports).
    module = _load_launcher()
    silos, heldout = _write_silos(tmp_path, count=4)
    out_dir = tmp_path / "frozen-run"
    argv = _tiny_argv(silos, heldout, out_dir, min_trainers=3) + ["--encoder-frozen"]

    summary = module.main(argv)
    assert summary["dry_run"] is False
    assert summary["closed_rounds"] == 2
    assert summary["completed_target"] is True

    report_path = out_dir / "phase3_long_run_smoke_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(report["rounds"]) == 2
    for row in report["rounds"]:
        assert row["val_pred"] is not None
        assert row["val_sigreg"] is not None
        assert row["effective_rank"] is not None
        assert row["frame_drift_deg"] is not None


def test_local_only_run_reports_per_participant_without_aggregation(
    tmp_path: Path,
) -> None:
    # The no-aggregation baseline: a tiny --local-only run over 2 toy silos completes, writes
    # phase3_local_only_report.json with per-participant metrics + a frame_drift_deg, and runs NO
    # coordinator aggregation ledger. It still writes one representative checkpoint for downstream eval.
    module = _load_launcher()
    silos, heldout = _write_silos(tmp_path, count=2)
    out_dir = tmp_path / "local-only"
    argv = _tiny_argv(silos, heldout, out_dir, min_trainers=2) + ["--local-only"]

    payload = module.main(argv)

    assert payload["mode"] == "local-only"
    assert len(payload["per_participant"]) == 2
    seen_ids = {row["participant_id"] for row in payload["per_participant"]}
    assert seen_ids == {"silo-0", "silo-1"}
    for row in payload["per_participant"]:
        assert row["val_pred"] is not None
        assert row["val_sigreg"] is not None
        assert row["effective_rank"] is not None
    assert payload["frame_drift_deg"] is not None
    assert float(payload["frame_drift_deg"]) >= 0.0
    assert payload["pushed"] is False

    report_path = out_dir / "phase3_local_only_report.json"
    assert report_path.exists()
    on_disk = json.loads(report_path.read_text(encoding="utf-8"))
    assert on_disk["mode"] == "local-only"
    assert on_disk["claim_boundary"]

    # NO aggregation ran: the coordinator ledger is absent, but a representative single-site checkpoint
    # is published under the shared checkpoint layout so the local-only control can be evaluated.
    artifacts = out_dir / "coordinator-artifacts"
    round_dirs = list(artifacts.glob("round-*")) if artifacts.exists() else []
    checkpoint_path = Path(payload["representative_checkpoint_path"])
    assert round_dirs == [checkpoint_path]
    assert (checkpoint_path / "header.json").exists()
    assert (checkpoint_path / "weights.safetensors").exists()
    assert not (artifacts / "ledger.jsonl").exists()

    # Residency: the serialized report carries no raw-data JSON keys.
    report_keys = _all_keys(on_disk)
    for forbidden in ("obs", "observation", "trajectory", "actions", "windows"):
        assert forbidden not in report_keys
