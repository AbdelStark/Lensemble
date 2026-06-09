"""Eval harness end-to-end on a toy env + the EvalReport contract (RFC-0005 §3 / 03 §13.1; #52).

A tiny CPU encoder/predictor is hash-verified-checkpointed, a deterministic stub ``EvalWorld`` (a
"stub world model" with closed-form dynamics and a rigged, KNOWN ``succeeded()``) is registered, and
``evaluate`` is wired end-to-end: it returns an ``EvalReport`` with the known ``success_rate``, correct
field types, a populated ``checkpoint_hash`` / ``run_manifest_hash``, and ``effective_dim > 0``. The
security-relevant edges are covered too — a tampered checkpoint raises ``CheckpointIntegrityError``, a
too-new ``schema_version`` raises ``SchemaVersionMismatch`` (both the report-parse path and the
checkpoint path), an unresolvable ``env_id`` raises ``EvaluationError``, the report round-trips, an
out-of-range field raises ``EvaluationError``, and no raw tensor reaches the report sink
(``INV-RESIDENCY``). Placed in tests/ml (the §8 CI gate collects tests/ml, not tests/eval).
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import Tensor

from lensemble.config import LensembleConfig
from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
from lensemble.errors import (
    CheckpointIntegrityError,
    EvaluationError,
    LensembleErrorCode,
    SchemaVersionMismatch,
)
from lensemble.eval import (
    EVAL_REPORT_SCHEMA_VERSION,
    EvalReport,
    evaluate,
    parse_eval_report,
    register_env,
    resolve_env,
)
from lensemble.model import build_encoder, build_predictor

# --- a tiny CPU model config carrying BOTH the real ModelConfig fields (so build_manifest /
# config_hash see a well-formed tree) AND the V-JEPA shape fields build_encoder/predictor read. This
# mirrors the SimpleNamespace model the CLI hand-builds; here it is a frozen dataclass so
# dataclasses.asdict(cfg) (used by build_manifest) recurses it. ---

_D = 8
_NUM_TOKENS = (
    4  # (num_frames//tubelet) * (image_size//patch_size)**2 = (2//2)*(4//2)**2 = 4
)
_T, _C, _H, _W = 2, 3, 4, 4


@dataclass(frozen=True)
class _EvalModelConfig:
    # real ModelConfig fields (keep config_hash well-formed)
    encoder: str = "vjepa2-vit-l"
    warm_start_release: str = "vjepa2-2.0"
    latent_dim: int = _D
    num_tokens: int = _NUM_TOKENS
    predictor_depth: int = 1
    predictor_width: int = _D
    wmcp_version: str = WMCP_VERSION
    encoder_frozen: bool = False
    # V-JEPA shape fields build_encoder/build_predictor/build_action_head read
    d: int = _D
    in_channels: int = _C
    num_frames: int = _T
    image_size: int = _H
    patch_size: int = 2
    tubelet: int = 2
    depth: int = 1
    num_heads: int = 2
    cond_dim: int = _D


def _cfg(**model_overrides: object) -> LensembleConfig:
    base = LensembleConfig()
    eval_cfg = dataclasses.replace(
        base.eval, env_id="test://toy", planner="icem", planning_samples=8, horizon=2
    )
    model = _EvalModelConfig(**model_overrides)  # type: ignore[arg-type]
    return dataclasses.replace(base, model=model, eval=eval_cfg, run_mode="eval")


def _save_model_checkpoint(cfg: LensembleConfig, ckpt_dir: Path) -> str:
    """Build a fresh encoder+predictor and save their weights as a hash-verified checkpoint."""
    from lensemble.artifacts import save_checkpoint

    torch.manual_seed(0)
    encoder = build_encoder(cfg)
    predictor = build_predictor(cfg)
    weights: dict[str, Tensor] = {}
    for name, tensor in encoder.state_dict().items():
        weights[f"encoder.{name}"] = tensor
    for name, tensor in predictor.state_dict().items():
        weights[f"predictor.{name}"] = tensor
    return save_checkpoint(
        ckpt_dir,
        weights,
        wmcp_version=WMCP_VERSION,
        round_index=0,
        config_hash="b" * 64,
        parent_hash=None,
    )


# --- a deterministic stub EvalWorld (a "stub world model") with a rigged success outcome ---


class _ToyWorld:
    """Closed-form CPU eval-world with a KNOWN success outcome (RFC-0005 Testing Strategy).

    ``succeeded()`` is rigged by ``success_pattern``: ``"always"`` -> 1.0 success rate, ``"alternate"``
    -> 0.5 over an even episode count. Observations are tiny ``(T, C, H, W)`` clips so the encoder runs
    on the CPU fallback; dynamics are a simple decay-plus-action so the planner has something to drive.
    """

    success_pattern = "always"  # class attr the factory flips per test

    def __init__(self, cfg: LensembleConfig) -> None:
        self.action_spec = ActionSpec(
            embodiment_id="toy",
            kind=ActionKind.CONTINUOUS,
            dim=2,
            low=(-1.0, -1.0),
            high=(1.0, 1.0),
            num_classes=None,
            units=("u", "u"),
            wmcp_version=WMCP_VERSION,
        )
        self._seed = 0
        self._steps = 0
        self._reset_count = 0

    def reset(self, seed: int) -> Tensor:
        self._seed = seed
        self._steps = 0
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(_T, _C, _H, _W, generator=gen)

    def goal(self) -> Tensor:
        gen = torch.Generator().manual_seed(7919)  # fixed goal clip
        return torch.randn(_T, _C, _H, _W, generator=gen)

    def step(self, action: Tensor) -> Tensor:
        self._steps += 1
        gen = torch.Generator().manual_seed(self._seed + self._steps)
        return torch.randn(_T, _C, _H, _W, generator=gen)

    def succeeded(self) -> bool:
        if type(self).success_pattern == "always":
            return True
        # "alternate": deterministic by the reset seed parity -> 0.5 over even seeds
        return self._seed % 2 == 0


def _register_toy(success_pattern: str = "always") -> None:
    def factory(cfg: LensembleConfig) -> _ToyWorld:
        world = _ToyWorld(cfg)
        type(world).success_pattern = success_pattern
        return world

    register_env("test://toy", factory)


# --- acceptance: success-rate end-to-end ---


def test_evaluate_returns_evalreport_with_known_success_rate(tmp_path: Path) -> None:
    _register_toy("always")
    cfg = _cfg()
    h = _save_model_checkpoint(cfg, tmp_path / "ckpt")

    report = evaluate(tmp_path / "ckpt", "test://toy", cfg=cfg)

    assert isinstance(report, EvalReport)
    assert report.success_rate == 1.0  # rigged: always-True
    assert report.planner == cfg.eval.planner == "icem"
    assert report.planning_samples == cfg.eval.planning_samples
    assert report.effective_dim > 0.0
    assert (
        isinstance(report.time_per_action_ms, float) and report.time_per_action_ms >= 0
    )
    assert report.probe_accuracy is None  # no probe wired here
    assert report.checkpoint_hash == h and len(report.checkpoint_hash) == 64
    assert report.run_manifest_hash and len(report.run_manifest_hash) == 64
    assert report.env_id == "test://toy"
    assert report.schema_version == EVAL_REPORT_SCHEMA_VERSION


def test_evaluate_records_the_alternating_success_rate(tmp_path: Path) -> None:
    _register_toy("alternate")
    cfg = _cfg()
    _save_model_checkpoint(cfg, tmp_path / "ckpt")
    report = evaluate(tmp_path / "ckpt", "test://toy", cfg=cfg)
    assert report.success_rate == 0.5  # half the deterministic seeds are even


def test_evaluate_populates_state_probe_r2_when_world_exposes_state(
    tmp_path: Path,
) -> None:
    cfg = _cfg()
    cfg = dataclasses.replace(
        cfg,
        eval=dataclasses.replace(
            cfg.eval,
            env_id="kinematic://swipe-dot",
            planning_samples=4,
            horizon=2,
        ),
    )
    _save_model_checkpoint(cfg, tmp_path / "ckpt")
    report = evaluate(
        tmp_path / "ckpt",
        "kinematic://swipe-dot",
        cfg=cfg,
        planner_iters=1,
    )
    assert report.state_probe_r2 is not None
    assert math.isfinite(report.state_probe_r2)
    assert report.state_probe_r2 <= 1.0
    assert report.probe_accuracy is None


# --- acceptance: a tampered checkpoint is rejected before any model loads ---


def test_evaluate_rejects_a_tampered_checkpoint(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    _register_toy("always")
    cfg = _cfg()
    _save_model_checkpoint(cfg, tmp_path / "ckpt")
    # overwrite the weight payload with different (valid safetensors) bytes -> hash mismatch
    save_file(
        {"encoder.norm.weight": torch.ones(_D)},
        str(tmp_path / "ckpt" / "weights.safetensors"),
    )
    with pytest.raises(CheckpointIntegrityError):
        evaluate(tmp_path / "ckpt", "test://toy", cfg=cfg)


# --- acceptance: a too-new schema_version is rejected (report-parse and checkpoint paths) ---


def test_parse_eval_report_rejects_too_new_schema() -> None:
    raw = {
        "schema_version": EVAL_REPORT_SCHEMA_VERSION + 1,
        "checkpoint_hash": "a" * 64,
        "env_id": "test://toy",
        "planner": "icem",
        "success_rate": 1.0,
        "planning_samples": 8,
        "time_per_action_ms": 1.0,
        "effective_dim": 2.0,
        "probe_accuracy": None,
        "state_probe_r2": None,
        "run_manifest_hash": "c" * 64,
    }
    with pytest.raises(SchemaVersionMismatch):
        parse_eval_report(raw)


def test_evaluate_rejects_a_too_new_checkpoint_schema(tmp_path: Path) -> None:
    _register_toy("always")
    cfg = _cfg()
    _save_model_checkpoint(cfg, tmp_path / "ckpt")
    header_path = tmp_path / "ckpt" / "header.json"
    raw = json.loads(header_path.read_text())
    raw["schema_version"] = 999
    header_path.write_text(json.dumps(raw))
    with pytest.raises(SchemaVersionMismatch):
        evaluate(tmp_path / "ckpt", "test://toy", cfg=cfg)


# --- acceptance: an unresolvable env_id raises EvaluationError ---


def test_unknown_env_id_raises(tmp_path: Path) -> None:
    cfg = _cfg()
    with pytest.raises(EvaluationError):
        resolve_env("totally-unknown-env", cfg=cfg)


def test_stable_worldmodel_env_id_raises_when_lib_absent(tmp_path: Path) -> None:
    cfg = _cfg()
    pytest.importorskip  # noqa: B018 - documents the intent; the assert below requires absence
    try:
        import stable_worldmodel  # type: ignore  # noqa: F401
    except ImportError:
        with pytest.raises(EvaluationError) as exc:
            resolve_env("stable-worldmodel://pusht", cfg=cfg)
        assert exc.value.code == LensembleErrorCode.EVALUATION_FAILED
        assert "stable-worldmodel" in exc.value.remediation
    else:  # pragma: no cover - the suite runs without the (unvendored) library, #96
        pytest.skip(
            "stable-worldmodel is vendored; the ImportError path is unreachable"
        )


# --- acceptance: the EvalReport round-trips and enforces ranges ---


def test_eval_report_roundtrips() -> None:
    report = EvalReport(
        schema_version=EVAL_REPORT_SCHEMA_VERSION,
        checkpoint_hash="a" * 64,
        env_id="test://toy",
        planner="cem",
        success_rate=0.75,
        planning_samples=16,
        time_per_action_ms=12.5,
        effective_dim=3.2,
        probe_accuracy=0.9,
        run_manifest_hash="d" * 64,
    )
    assert parse_eval_report(report.model_dump()) == report


def test_eval_report_rejects_out_of_range_success_rate() -> None:
    with pytest.raises(EvaluationError):
        EvalReport(
            schema_version=EVAL_REPORT_SCHEMA_VERSION,
            checkpoint_hash="a" * 64,
            env_id="e",
            planner="cem",
            success_rate=1.5,  # out of [0, 1]
            planning_samples=1,
            time_per_action_ms=1.0,
            effective_dim=1.0,
            probe_accuracy=None,
            run_manifest_hash="d" * 64,
        )


def test_eval_report_rejects_non_positive_effective_dim() -> None:
    with pytest.raises(EvaluationError):
        EvalReport(
            schema_version=EVAL_REPORT_SCHEMA_VERSION,
            checkpoint_hash="a" * 64,
            env_id="e",
            planner="cem",
            success_rate=0.5,
            planning_samples=1,
            time_per_action_ms=1.0,
            effective_dim=0.0,  # must be > 0
            probe_accuracy=None,
            run_manifest_hash="d" * 64,
        )


def test_eval_report_rejects_out_of_range_probe_accuracy() -> None:
    with pytest.raises(EvaluationError):
        EvalReport(
            schema_version=EVAL_REPORT_SCHEMA_VERSION,
            checkpoint_hash="a" * 64,
            env_id="e",
            planner="cem",
            success_rate=0.5,
            planning_samples=1,
            time_per_action_ms=1.0,
            effective_dim=1.0,
            probe_accuracy=1.5,  # out of [0, 1]
            run_manifest_hash="d" * 64,
        )


def test_eval_report_rejects_invalid_state_probe_r2() -> None:
    with pytest.raises(EvaluationError):
        EvalReport(
            schema_version=EVAL_REPORT_SCHEMA_VERSION,
            checkpoint_hash="a" * 64,
            env_id="e",
            planner="cem",
            success_rate=0.5,
            planning_samples=1,
            time_per_action_ms=1.0,
            effective_dim=1.0,
            probe_accuracy=None,
            state_probe_r2=1.1,
            run_manifest_hash="d" * 64,
        )


# --- residency: only scalar metrics / hashes / counts reach the report sink (INV-RESIDENCY) ---


def test_report_carries_no_raw_tensor(tmp_path: Path) -> None:
    _register_toy("always")
    cfg = _cfg()
    _save_model_checkpoint(cfg, tmp_path / "ckpt")
    report = evaluate(tmp_path / "ckpt", "test://toy", cfg=cfg)
    dumped = report.model_dump()
    for key, value in dumped.items():
        assert isinstance(value, (str, int, float, type(None))), (
            f"EvalReport.{key} is {type(value)}; the report sink must carry only scalars/hashes/counts"
        )
    # structurally: the JSON serialization carries no tensor/ndarray
    encoded = report.model_dump_json()
    assert "tensor" not in encoded.lower()
