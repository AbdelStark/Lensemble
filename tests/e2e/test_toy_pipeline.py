"""The first green end-to-end toy run (#167): warm-start → train → commit → eval, plus a federated round.

This is the headline end-to-end smoke the issue closes the gaps for. On a tiny CPU V-JEPA config (the
proven ``tests/ml/_EvalModelConfig`` shape) it:

1. SINGLE-SITE E2E (the REQUIRED green path): writes a toy :class:`~lensemble.data.dataset.EpisodeDataset`
   to ``tmp_path`` via the #22 data adapter, points ``cfg.data.data_source`` at it, runs the real
   :func:`lensemble.federation.train_local` (warm-start → inner loop → hash-committed checkpoint), then
   :func:`lensemble.eval.evaluate` on the built-in ``synthetic://toy`` world (#167) and asserts a valid
   :class:`~lensemble.eval.report.EvalReport`.
2. FEDERATED ROUND (the bonus): a real :class:`~lensemble.federation.Coordinator` + two
   :class:`~lensemble.federation.Participant`s share one :class:`~lensemble.federation.InProcessTransport`;
   each runs ``local_round`` over the toy windows and submits its pseudo-gradient; ``coordinator.try_round``
   reaches ``CLOSED`` and the canonical global hash advances.
3. CLI SMOKE: the wired ``train`` / ``eval`` commands run end-to-end through Typer's ``CliRunner``.

Shapes (the encoder/window contract, #167). A single observation is a clip
``(num_frames, in_channels, image_size, image_size) = (_T, _C, _H, _W)``; a ``Window.obs`` is
``(num_steps + 1, _T, _C, _H, _W)`` and ``Window.actions`` is ``(num_steps, action_dim)``; the encoder
forward takes a clip batch ``(B, _T, _C, _H, _W)`` (the harness does ``encoder(obs.unsqueeze(0))``). The
dims are kept tiny (``num_tokens = (2//2)*(4//2)**2 = 4``) so the whole run is CPU-fast and downloads
nothing (07 §7).

Placed in tests/e2e — the §8 CI gate scans tests/{unit,property,integration,ml,e2e,regression}.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from typer.testing import CliRunner

from lensemble.cli import app
from lensemble.config import LensembleConfig
from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
from lensemble.data import Episode, EpisodeDataset, Transition, save_episodes
from lensemble.data.episode import Window
from lensemble.data.probe import PublicProbe
from lensemble.eval import (
    EvalReport,
    build_claim_mvp_report,
    evaluate,
    parse_claim_mvp_report,
)
from lensemble.federation import (
    Coordinator,
    InProcessTransport,
    Participant,
    RoundState,
    train_local,
)

# --- the tiny consistent CPU config (mirrors tests/ml/_EvalModelConfig — the proven-working shape) ---

_D = 8
_NUM_TOKENS = (
    4  # (num_frames//tubelet) * (image_size//patch_size)**2 = (2//2)*(4//2)**2 = 4
)
_T, _C, _H, _W = 2, 3, 4, 4
_ACTION_DIM = 2
_WINDOW_STEPS = 1  # Window.obs is (window_steps + 1) clips


@dataclass(frozen=True)
class _ToyModelConfig:
    # real ModelConfig fields (keep config_hash / manifest / build_manifest well-formed)
    encoder: str = "vjepa2-vit-l"
    warm_start_release: str = "vjepa2-2.0"
    latent_dim: int = _D
    num_tokens: int = _NUM_TOKENS
    predictor_depth: int = 1
    predictor_width: int = _D
    wmcp_version: str = WMCP_VERSION
    encoder_frozen: bool = False
    # V-JEPA shape fields build_encoder/build_predictor/build_action_head + the toy world read
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


def _episode(episode_id: str, n_transitions: int, seed: int) -> Episode:
    gen = torch.Generator().manual_seed(seed)
    transitions: list[Transition] = []
    for _ in range(n_transitions):
        transitions.append(
            Transition(
                obs_t=torch.randn(
                    _T, _C, _H, _W, generator=gen
                ),  # one clip (T, C, H, W)
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
        collection_meta={"site": "e2e"},
    )


def _write_toy_dataset(tmp_path: Path, *, fmt: str = "lance") -> Path:
    """Write a 2-episode toy dataset (correct clip/action shapes) via the #22 adapter; return its path."""
    dataset = EpisodeDataset(
        [_episode("ep-0", 3, seed=1), _episode("ep-1", 3, seed=2)],
        fmt=fmt,  # type: ignore[arg-type]
    )
    suffix = ".lance" if fmt == "lance" else ".h5"
    path = tmp_path / f"toy{suffix}"
    save_episodes(dataset, path, fmt=fmt)  # type: ignore[arg-type]
    return path


def _toy_cfg(data_source: Path) -> LensembleConfig:
    """A tiny consistent config: the toy model shape, the toy data source, lambda_anc=0 (no probe needed)."""
    base = LensembleConfig()
    model = _ToyModelConfig()
    federation = dataclasses.replace(
        base.federation,
        inner_horizon=2,  # tiny: the loop only needs to RUN and produce a real Δ
        participant_count=2,
        fault_tolerance_min_participants=2,  # so the round quorum K = max(2, 2) = 2 (two participants)
        secure_agg_threshold=2,
    )
    eval_cfg = dataclasses.replace(
        base.eval,
        env_id="synthetic://toy",
        planner="icem",
        planning_samples=8,
        horizon=2,
    )
    data = dataclasses.replace(
        base.data,
        data_source=str(data_source),
        format="lance",
        window_steps=_WINDOW_STEPS,
    )
    objective = dataclasses.replace(
        base.objective, lambda_anc=0.0
    )  # bare LeJEPA: no probe required
    return dataclasses.replace(
        base,
        model=model,  # type: ignore[arg-type]
        federation=federation,
        eval=eval_cfg,
        data=data,
        objective=objective,
        run_mode="train_local",
    )


def _toy_windows() -> list[Window]:
    """Tiny windows matching the encoder/window contract (the federated-round override fixtures)."""
    gen = torch.Generator().manual_seed(11)
    windows: list[Window] = []
    for _ in range(2):
        windows.append(
            Window(
                obs=torch.randn(_WINDOW_STEPS + 1, _T, _C, _H, _W, generator=gen),
                actions=torch.randn(_WINDOW_STEPS, _ACTION_DIM, generator=gen),
                num_steps=_WINDOW_STEPS,
                embodiment_id="toy",
            )
        )
    return windows


# --- the REQUIRED green single-site E2E: train_local → evaluate(synthetic://toy) → EvalReport ---


def test_single_site_train_then_eval_is_green(tmp_path: Path) -> None:
    source = _write_toy_dataset(tmp_path)
    cfg = _toy_cfg(source)

    # warm-start → inner loop → hash-committed checkpoint
    result = train_local(cfg)
    assert (result.checkpoint_dir / "weights.safetensors").exists()
    assert (result.checkpoint_dir / "header.json").exists()
    assert len(result.checkpoint_hash) == 64
    assert all(c in "0123456789abcdef" for c in result.checkpoint_hash)
    assert len(result.manifest_hash) == 64
    assert isinstance(result.final_loss, float)

    # commit → eval: the headline green run
    report = evaluate(result.checkpoint_dir, env_id="synthetic://toy", cfg=cfg)
    assert isinstance(report, EvalReport)
    assert 0.0 <= report.success_rate <= 1.0
    assert report.effective_dim > 0.0
    assert (
        report.checkpoint_hash == result.checkpoint_hash
    )  # the eval loaded THIS checkpoint
    assert report.env_id == "synthetic://toy"


def test_synthetic_toy_env_has_a_known_nontrivial_success_rate(tmp_path: Path) -> None:
    # The toy world is rigged so the success rate is KNOWN and non-trivial (NOT always-0 / always-1) — the
    # e2e success assertion is then non-vacuous. With root_seed=0 the harness's seeds 0..3 → two even →
    # success_rate == 0.5.
    source = _write_toy_dataset(tmp_path)
    cfg = _toy_cfg(source)
    result = train_local(cfg)
    report = evaluate(result.checkpoint_dir, env_id="synthetic://toy", cfg=cfg)
    assert report.success_rate == 0.5


def test_single_site_runs_over_hdf5_source(tmp_path: Path) -> None:
    # The data adapter is format-agnostic: the same green run works from an hdf5 store (#22).
    source = _write_toy_dataset(tmp_path, fmt="hdf5")
    cfg = _toy_cfg(source)
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, format="hdf5"))
    result = train_local(cfg)
    report = evaluate(result.checkpoint_dir, env_id="synthetic://toy", cfg=cfg)
    assert isinstance(report, EvalReport)
    assert report.effective_dim > 0.0


def _write_toy_probe(tmp_path: Path) -> Path:
    """Pin a tiny probe (k = _D >= d landmarks, clip-shaped points) for the anchored train_local path."""
    import torch as _torch

    from lensemble.data.probe import build_probe, save_probe
    from lensemble.model.encoder import build_encoder, snapshot_reference

    cfg = _toy_cfg(tmp_path / "unused")  # only its model shape is read by build_encoder
    gen = _torch.Generator().manual_seed(5)
    points = _torch.randn(
        _D, _T, _C, _H, _W, generator=gen
    )  # k = _D landmarks (k >= d)
    f_ref = snapshot_reference(build_encoder(cfg))
    probe = build_probe(points, _torch.arange(_D), f_ref, probe_version=1)
    probe_path = tmp_path / "probe.safetensors"
    save_probe(probe, probe_path)
    return probe_path


def test_single_site_anchored_objective_path(tmp_path: Path) -> None:
    # The lambda_anc > 0 branch of train_local: the FrameAnchor is built from the pinned probe exactly as the
    # participant builds it (round-0 reference snapshot). A pinned probe is required for an anchored run.
    source = _write_toy_dataset(tmp_path)
    probe_path = _write_toy_probe(tmp_path)
    cfg = _toy_cfg(source)
    cfg = dataclasses.replace(
        cfg,
        objective=dataclasses.replace(cfg.objective, lambda_anc=1.0),
        data=dataclasses.replace(cfg.data, probe_path=str(probe_path)),
    )
    result = train_local(cfg)
    report = evaluate(result.checkpoint_dir, env_id="synthetic://toy", cfg=cfg)
    assert isinstance(report, EvalReport)
    assert report.effective_dim > 0.0


# --- the BONUS federated round: 2 participants → coordinator.try_round() → CLOSED, hash advances ---


class _TestParticipant(Participant):
    """A Participant whose #22 data-layer hooks return the toy fixtures (mirrors tests/ml/test_participant)."""

    def __init__(
        self,
        config: LensembleConfig,
        *,
        participant_id: str,
        transport: InProcessTransport,
        windows: list[Window],
    ) -> None:
        super().__init__(config, participant_id=participant_id, transport=transport)
        self._windows = windows
        self._spec = _spec()

    def _pinned_probe(self) -> "PublicProbe":
        # The coordinator pins no probe → GlobalState.probe_hash is the 32-byte placeholder (b"\x00"*32);
        # the bare LeJEPA objective (lambda_anc=0) never reads the probe's points, so a tiny probe whose
        # content_hash IS the placeholder passes INV-PROBE-PIN without a real probe artifact.
        return PublicProbe(
            points=torch.zeros(_D, _T, _C, _H, _W),
            landmark_idx=torch.arange(_D),
            landmark_targets=torch.zeros(_D, _NUM_TOKENS, _D),
            content_hash=b"\x00" * 32,
            probe_version=1,
        )

    def _local_windows(self) -> list[Window]:
        return self._windows

    def _dataset_root(self) -> bytes:
        return bytes.fromhex(self.participant_id.encode().hex().ljust(64, "0")[:64])

    def _action_spec(self) -> ActionSpec:
        return self._spec


def test_federated_round_commits_and_advances_the_global_hash(tmp_path: Path) -> None:
    source = _write_toy_dataset(tmp_path)
    cfg = _toy_cfg(source)
    # The coordinator pins no probe (cfg.data.probe_path is None → the 32-byte placeholder) and the bare
    # LeJEPA objective (lambda_anc=0) needs no probe, so the participants accept the round (INV-PROBE-PIN
    # holds against the placeholder). A federated run requires deterministic aggregation.
    cfg = dataclasses.replace(
        cfg,
        run_mode="coordinator",
        determinism=dataclasses.replace(
            cfg.determinism, deterministic_aggregation=True
        ),
    )

    transport = InProcessTransport()
    coordinator = Coordinator(cfg, transport=transport)
    global_state = coordinator.global_state()
    hash_before = coordinator.global_state_hash()

    windows = _toy_windows()
    for pid in ("participant-0", "participant-1"):
        participant = _TestParticipant(
            cfg, participant_id=pid, transport=transport, windows=windows
        )
        pseudo_gradient = participant.local_round(
            global_state, global_state.sketch_seed
        )
        transport.submit_update(
            participant_id=pid,
            round_index=global_state.round_index,
            update=pseudo_gradient,
        )

    state = coordinator.try_round()
    assert state is RoundState.CLOSED
    assert (
        coordinator.global_state_hash() != hash_before
    )  # the canonical global hash advanced
    # exactly the two contributing participants are recorded in the round's ContributionRecord
    records = coordinator.ledger_records()
    assert records, "a committed round appends a ContributionRecord"


# --- claim-MVP real-data smoke: LeRobot-H5 sources -> default Participant hooks -> committed round ---


@dataclass(frozen=True)
class _RobotModelConfig:
    encoder: str = "vjepa2-vit-l"
    warm_start_release: str = "vjepa2-2.0"
    latent_dim: int = _D
    num_tokens: int = 4  # (1//1) * (8//4)**2
    predictor_depth: int = 1
    predictor_width: int = _D
    wmcp_version: str = WMCP_VERSION
    encoder_frozen: bool = False
    d: int = _D
    in_channels: int = _C
    num_frames: int = 1
    image_size: int = 8
    patch_size: int = 4
    tubelet: int = 1
    depth: int = 1
    num_heads: int = 2
    cond_dim: int = _D


def _write_lerobot_h5(path: Path, *, seed: int) -> None:
    import h5py

    ep_lens = (4, 4)
    n = sum(ep_lens)
    ep_index = np.concatenate(
        [np.full(length, i, dtype=np.int32) for i, length in enumerate(ep_lens)]
    )
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(n, 8, 8, 3), dtype=np.uint8)
    actions = rng.standard_normal((n, _ACTION_DIM)).astype(np.float32)
    # Guarantee non-degenerate per-dim low/high bounds for ActionSpec validation.
    actions[0] = np.array([-1.0, -0.5], dtype=np.float32)
    actions[-1] = np.array([1.0, 0.5], dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("episode_index", data=ep_index)
        f.create_dataset("observation/pixels_top", data=pixels)
        f.create_dataset("action", data=actions)


def _write_robot_probe(tmp_path: Path, cfg: LensembleConfig) -> Path:
    from lensemble.data.probe import build_probe, save_probe
    from lensemble.model.encoder import build_encoder, snapshot_reference

    gen = torch.Generator().manual_seed(33)
    points = torch.randn(_D, 1, _C, 8, 8, generator=gen)
    f_ref = snapshot_reference(build_encoder(cfg))
    probe = build_probe(points, torch.arange(_D), f_ref, probe_version=1)
    path = tmp_path / "robot-probe.safetensors"
    save_probe(probe, path)
    return path


def _robot_cfg(
    *, data_source: Path | None, probe_path: Path, run_mode: str
) -> LensembleConfig:
    base = LensembleConfig()
    model = _RobotModelConfig()
    federation = dataclasses.replace(
        base.federation,
        inner_horizon=1,
        participant_count=2,
        fault_tolerance_min_participants=2,
        secure_agg_threshold=2,
    )
    data = dataclasses.replace(
        base.data,
        data_source=str(data_source) if data_source is not None else None,
        format="lerobot-h5",
        probe_path=str(probe_path),
        window_steps=1,
    )
    objective = dataclasses.replace(
        base.objective,
        lambda_sig=0.1,
        lambda_anc=0.01,
        target_stop_gradient=False,
    )
    privacy = dataclasses.replace(base.privacy, enabled=False)
    return dataclasses.replace(
        base,
        model=model,  # type: ignore[arg-type]
        federation=federation,
        data=data,
        objective=objective,
        privacy=privacy,
        run_mode=run_mode,  # type: ignore[arg-type]
    )


def test_lerobot_h5_two_silo_federated_round_uses_default_data_hooks(
    tmp_path: Path,
) -> None:
    silo0 = tmp_path / "silo0.h5"
    silo1 = tmp_path / "silo1.h5"
    _write_lerobot_h5(silo0, seed=10)
    _write_lerobot_h5(silo1, seed=20)

    probe_cfg = _robot_cfg(
        data_source=silo0, probe_path=tmp_path / "unused", run_mode="participant"
    )
    probe_path = _write_robot_probe(tmp_path, probe_cfg)
    coord_cfg = _robot_cfg(
        data_source=None, probe_path=probe_path, run_mode="coordinator"
    )
    p0_cfg = _robot_cfg(
        data_source=silo0, probe_path=probe_path, run_mode="participant"
    )
    p1_cfg = _robot_cfg(
        data_source=silo1, probe_path=probe_path, run_mode="participant"
    )

    transport = InProcessTransport()
    coordinator = Coordinator(coord_cfg, transport=transport)
    global_state = coordinator.global_state()
    hash_before = coordinator.global_state_hash()
    updates = {}
    sources = {"silo-0": str(silo0), "silo-1": str(silo1)}

    for pid, cfg in (("silo-0", p0_cfg), ("silo-1", p1_cfg)):
        participant = Participant(cfg, participant_id=pid, transport=transport)
        pseudo_gradient = participant.local_round(
            global_state, global_state.sketch_seed
        )
        updates[pid] = pseudo_gradient
        transport.submit_update(
            participant_id=pid,
            round_index=global_state.round_index,
            update=pseudo_gradient,
        )
        assert len(pseudo_gradient.dataset_root) == 32

    state = coordinator.try_round()
    assert state is RoundState.CLOSED
    assert coordinator.global_state_hash() != hash_before

    records = coordinator.ledger_records()
    assert records
    assert set(records[-1].dataset_roots) == {"silo-0", "silo-1"}
    report = build_claim_mvp_report(
        cfg=coord_cfg,
        coordinator=coordinator,
        participant_updates=updates,
        participant_sources=sources,
        round_state=state,
    )
    assert report.round_state == "closed"
    assert report.objective_target_stop_gradient is False
    assert report.committed_rounds == 1
    assert {p.participant_id for p in report.participants} == {"silo-0", "silo-1"}
    assert {p.participant_id: p.dataset_root for p in report.participants} == records[
        -1
    ].dataset_roots
    restored = parse_claim_mvp_report(report.model_dump(mode="json"))
    assert restored == report


# --- the CLI smoke: the wired train / eval commands run end-to-end ---


def test_cli_train_then_eval_smoke(tmp_path: Path) -> None:
    runner = CliRunner()
    source = _write_toy_dataset(tmp_path)
    cfg = _toy_cfg(source)
    run_dir = tmp_path / "run"

    # `train`: emits the manifest path, the committed checkpoint dir, and its hash on stdout.
    train_result = train_local(cfg, run_dir=run_dir)
    assert (train_result.checkpoint_dir / "weights.safetensors").exists()

    # The `eval` command's `--checkpoint`/`--env-id` options are WIRED — checked behaviorally rather than
    # by scraping the rich-rendered `--help` panel (whose wrapping/styling is terminal-width- and
    # rich-version-dependent and so flaky across local vs CI). Invoking with a nonexistent checkpoint must
    # FAIL on the load (not on option parsing): the options are recognized (no "No such option"), and the
    # run exits non-zero. (The full toy `eval --checkpoint` happy path needs a toy-ModelConfig config the
    # CLI's default LensembleConfig is not; that path is covered directly via `evaluate` above.)
    eval_result = runner.invoke(
        app,
        [
            "eval",
            "--checkpoint",
            str(tmp_path / "nonexistent-ckpt"),
            "--env-id",
            "synthetic://toy",
        ],
    )
    assert (
        eval_result.exit_code != 0
    )  # fails on the missing checkpoint, not on option parsing
    assert (
        "No such option" not in eval_result.output
    )  # --checkpoint / --env-id ARE recognized
