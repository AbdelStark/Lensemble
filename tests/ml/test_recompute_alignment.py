"""Public recomputation of frame alignment — the one free Phase-2 mechanism (RFC-0006 §4). Issue #62.

A tiny CPU encoder is built on a consistent config, snapshotted as ``f_ref``, and a probe is built whose
``landmark_targets = f_ref(landmarks)`` are the pinned reference targets ``E_ref``. The encoder (+ a
predictor, so the artifact carries both shared groups) is committed via ``save_checkpoint`` WITH a
``ModelArchDescriptor`` so the checkpoint is self-describing (#171) — the prerequisite ``from_header``
needs. Then ``recompute_alignment`` reconstructs ``f_θ`` and recomputes the closed-form Procrustes
alignment from the committed weights + the pinned probe alone (RFC-0006 §4).

Coverage: the HONEST case (committed encoder IS ``f_ref`` → recovered rotation angle ~0) returns a
``FrameDriftReport`` and ``matches_expected=True``; a PERTURBED claim → ``matches_expected=False``; two
independent recomputations agree bitwise on ``procrustes_q_hash`` (the cross-process verifiability
property); fail-closed edges — a probe whose recomputed content hash differs raises ``ProbeError``, a
tampered checkpoint raises ``CheckpointIntegrityError``, a rank-deficient probe embedding raises
``DegenerateProcrustes`` (never a silent garbage ``Q*``), a non-self-describing checkpoint raises the
clear ``from_header`` ``ArtifactError``, and a too-new record schema raises ``SchemaVersionMismatch``; and
the CLI exits 0 on an honest run and non-zero on a mismatched ``--expected`` claim.

The #18 caveat (documented on ``recompute_alignment``): this MEASURES the committed model's alignment to
the reference frame; it does NOT verify the Layer-3 backstop was applied (that backstop rotates in
ACTIVATION space, the recorded #18 decision, not as a fold into the committed weights).

Placed in tests/ml: the §8 CI gate collects tests/ml, NOT the tests/verify directory the issue named
(which CI does not scan), so the acceptance tests live here to actually run in the gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from typer.testing import CliRunner

from lensemble.artifacts import save_checkpoint
from lensemble.artifacts.checkpoint import model_arch_from_config
from lensemble.cli import app
from lensemble.contracts import WMCP_VERSION
from lensemble.data.probe import build_probe, save_probe
from lensemble.errors import (
    ArtifactError,
    CheckpointIntegrityError,
    DegenerateProcrustes,
    ProbeError,
    SchemaVersionMismatch,
)
from lensemble.gauge.drift import FrameDriftReport
from lensemble.model import build_encoder, build_predictor
from lensemble.model.encoder import snapshot_reference
from lensemble.verify import (
    AlignmentClaim,
    AlignmentRecomputation,
    parse_alignment_claim,
    parse_alignment_recomputation,
    procrustes_q_hash,
    recompute_alignment,
    recompute_alignment_claim,
)
from lensemble.verify.recompute import ALIGNMENT_SCHEMA_VERSION

_runner = CliRunner()

# A tiny consistent ViT config: d=8 with num_tokens = (4//2)*(8//2)**2 = 32. The 32 tokens make the
# flattened probe embedding (k·N, d) genuinely full rank in d: the real Encoder ends in a LayerNorm, which
# zero-means each token row, so a per-token embedding lives in a (d-1)-dim subspace — too few tokens give a
# rank-(d-1) matrix and a degenerate Procrustes even in the honest f_θ == f_ref case. With 32 tokens the
# across-token variation restores full rank, so the honest case recovers Q* = I (angle ~0). (Verified.)
_D = 8
_T, _C, _H, _W = 4, 3, 8, 8
_PATCH, _TUBELET = 2, 2
_NUM_TOKENS = (_T // _TUBELET) * (_H // _PATCH) ** 2  # = 2 * 16 = 32


@dataclass(frozen=True)
class _ModelCfg:
    # real ModelConfig fields (keep config_hash / model_arch_from_config well-formed)
    encoder: str = "vjepa2-vit-l"
    warm_start_release: str = "vjepa2-2.0"
    latent_dim: int = _D
    num_tokens: int = _NUM_TOKENS
    predictor_depth: int = 1
    predictor_width: int = _D
    wmcp_version: str = WMCP_VERSION
    encoder_frozen: bool = False
    # V-JEPA shape fields build_encoder / build_predictor / model_arch_from_config read
    d: int = _D
    in_channels: int = _C
    num_frames: int = _T
    image_size: int = _H
    patch_size: int = _PATCH
    tubelet: int = _TUBELET
    depth: int = 1
    num_heads: int = 2
    cond_dim: int = _D


def _cfg(**model_overrides: object) -> SimpleNamespace:
    return SimpleNamespace(model=_ModelCfg(**model_overrides))  # type: ignore[arg-type]


def _points(p: int = 16, *, t: int = _T, h: int = _H, w: int = _W) -> Tensor:
    gen = torch.Generator().manual_seed(0)
    return torch.randn(p, t, _C, h, w, generator=gen)


def _commit_encoder(
    ckpt_dir: Path, cfg: SimpleNamespace, *, self_describing: bool = True
) -> tuple[str, object]:
    """Build a fresh encoder (+ predictor), snapshot f_ref, and commit the encoder weights.

    Returns ``(content_hash, f_ref)``. With ``self_describing`` the checkpoint carries a
    ``ModelArchDescriptor`` (#171) so ``Encoder.from_header`` can reconstruct it; without it the header's
    ``model_arch`` is ``None`` (the legacy / non-self-describing case).
    """
    torch.manual_seed(0)
    encoder = build_encoder(cfg)
    predictor = build_predictor(cfg)
    f_ref = snapshot_reference(encoder)  # the committed encoder IS f_ref (honest case)
    weights: dict[str, Tensor] = {}
    for name, tensor in encoder.state_dict().items():
        weights[f"encoder.{name}"] = tensor
    for name, tensor in predictor.state_dict().items():
        weights[f"predictor.{name}"] = tensor
    content_hash = save_checkpoint(
        ckpt_dir,
        weights,
        wmcp_version=WMCP_VERSION,
        round_index=7,
        config_hash="b" * 64,
        parent_hash=None,
        model_arch=model_arch_from_config(cfg) if self_describing else None,
    )
    return content_hash, f_ref


def _build_probe_for(
    cfg: SimpleNamespace, f_ref: object, *, k: int = _D, num_points: int = 16
) -> object:
    """Build a probe whose landmark_targets = f_ref(landmarks) are the pinned E_ref (k landmarks)."""
    points = _points(num_points)
    return build_probe(points, torch.arange(k), f_ref)  # type: ignore[arg-type]


def _honest_setup(tmp_path: Path, **model_overrides: object) -> tuple[Path, Path]:
    """Commit a self-describing checkpoint + save the matching probe; return (ckpt_dir, probe_path)."""
    cfg = _cfg(**model_overrides)
    ckpt_dir = tmp_path / "ckpt"
    _, f_ref = _commit_encoder(ckpt_dir, cfg)
    probe = _build_probe_for(cfg, f_ref)
    probe_path = tmp_path / "probe.safetensors"
    save_probe(probe, probe_path)  # type: ignore[arg-type]
    return ckpt_dir, probe_path


# --- acceptance: honest recomputation returns a FrameDriftReport with ~0 angle to the reference ---


def test_recompute_alignment_returns_report_with_zero_angle(
    tmp_path: Path, tol: object
) -> None:
    angle_tol: float = tol.ANGLE_TOL_DEG  # type: ignore[attr-defined]
    ckpt_dir, probe_path = _honest_setup(tmp_path)

    report = recompute_alignment(ckpt_dir, probe_path)

    assert isinstance(report, FrameDriftReport)
    assert report.round_index == 7  # from the committed header
    # the committed encoder IS f_ref, so the committed model's frame is the reference frame: angle ~ 0
    assert report.drift_from_global["committed"] < angle_tol


def test_recompute_alignment_claim_matches_an_honest_claim(tmp_path: Path) -> None:
    ckpt_dir, probe_path = _honest_setup(tmp_path)

    # the honest claim is exactly what an honest recomputation produces
    honest = recompute_alignment_claim(ckpt_dir, probe_path)
    assert isinstance(honest, AlignmentRecomputation)
    assert honest.matches_expected is None  # no expected supplied
    expected_claim = honest.recomputed

    checked = recompute_alignment_claim(ckpt_dir, probe_path, expected=expected_claim)
    assert checked.matches_expected is True
    assert checked.max_abs_residual_delta is not None
    assert checked.max_abs_residual_delta <= 1e-6
    assert checked.recomputed.round_index == 7
    assert checked.probe_hash == expected_claim.probe_hash


# --- acceptance: a perturbed expected claim (wrong q_hash) reports matches_expected=False ---


def test_perturbed_claim_does_not_match(tmp_path: Path) -> None:
    ckpt_dir, probe_path = _honest_setup(tmp_path)
    honest = recompute_alignment_claim(ckpt_dir, probe_path).recomputed
    perturbed = honest.model_copy(update={"procrustes_q_hash": "f" * 64})

    checked = recompute_alignment_claim(ckpt_dir, probe_path, expected=perturbed)
    assert checked.matches_expected is False
    # the recomputed record is still the honest one (the claim is what is rejected, not the recomputation)
    assert checked.recomputed.procrustes_q_hash == honest.procrustes_q_hash


def test_perturbed_residual_claim_does_not_match(tmp_path: Path) -> None:
    ckpt_dir, probe_path = _honest_setup(tmp_path)
    honest = recompute_alignment_claim(ckpt_dir, probe_path).recomputed
    # same q_hash but a residual far outside the fp32 tolerance -> mismatch
    perturbed = honest.model_copy(
        update={"procrustes_residual": honest.procrustes_residual + 1.0}
    )
    checked = recompute_alignment_claim(ckpt_dir, probe_path, expected=perturbed)
    assert checked.matches_expected is False
    assert checked.max_abs_residual_delta is not None
    assert checked.max_abs_residual_delta >= 1.0 - 1e-6


# --- acceptance: two independent recomputations agree bitwise on procrustes_q_hash ---


def test_procrustes_q_hash_is_bitwise_reproducible(tmp_path: Path) -> None:
    ckpt_dir, probe_path = _honest_setup(tmp_path)
    first = recompute_alignment_claim(ckpt_dir, probe_path).recomputed
    second = recompute_alignment_claim(ckpt_dir, probe_path).recomputed
    assert (
        first.procrustes_q_hash == second.procrustes_q_hash
    )  # cross-process verifiability key
    assert first == second  # the whole claim is deterministic
    # and the full FrameDriftReport is deterministic too
    assert recompute_alignment(ckpt_dir, probe_path) == recompute_alignment(
        ckpt_dir, probe_path
    )


def test_procrustes_q_hash_is_a_64_hex_sha256() -> None:
    q = torch.eye(_D, dtype=torch.float32)
    h = procrustes_q_hash(q)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)
    assert procrustes_q_hash(q) == h  # deterministic


# --- fail-closed: probe whose recomputed content hash differs raises ProbeError ---


def test_probe_pin_mismatch_raises(tmp_path: Path) -> None:
    cfg = _cfg()
    ckpt_dir = tmp_path / "ckpt"
    _, f_ref = _commit_encoder(ckpt_dir, cfg)
    probe = _build_probe_for(cfg, f_ref)
    probe_path = tmp_path / "probe.safetensors"
    save_probe(probe, probe_path)  # type: ignore[arg-type]
    # corrupt the stored content_hash metadata so it no longer matches the recomputed points/idx hash
    tensors: dict[str, Tensor] = {}
    with safe_open(str(probe_path), framework="pt") as f:  # type: ignore[no-untyped-call]
        meta = dict(f.metadata() or {})
        for key in ("points", "landmark_idx", "landmark_targets"):
            tensors[key] = f.get_tensor(key)
    meta["content_hash"] = "00" * 32  # a pin that does not match the recomputed hash
    save_file(tensors, str(probe_path), metadata=meta)

    with pytest.raises(ProbeError):
        recompute_alignment(ckpt_dir, probe_path)
    with pytest.raises(ProbeError):
        recompute_alignment_claim(ckpt_dir, probe_path)


# --- fail-closed: a tampered checkpoint raises CheckpointIntegrityError ---


def test_tampered_checkpoint_raises(tmp_path: Path) -> None:
    ckpt_dir, probe_path = _honest_setup(tmp_path)
    # overwrite the weight payload with different (valid safetensors) bytes -> content-hash mismatch
    save_file(
        {"encoder.norm.weight": torch.ones(_D)},
        str(ckpt_dir / "weights.safetensors"),
    )
    with pytest.raises(CheckpointIntegrityError):
        recompute_alignment(ckpt_dir, probe_path)


# --- fail-closed: a non-self-describing checkpoint raises the clear from_header ArtifactError ---


def test_non_self_describing_checkpoint_raises(tmp_path: Path) -> None:
    cfg = _cfg()
    ckpt_dir = tmp_path / "ckpt"
    _, f_ref = _commit_encoder(ckpt_dir, cfg, self_describing=False)
    probe = _build_probe_for(cfg, f_ref)
    probe_path = tmp_path / "probe.safetensors"
    save_probe(probe, probe_path)  # type: ignore[arg-type]
    with pytest.raises(ArtifactError) as exc:
        recompute_alignment(ckpt_dir, probe_path)
    assert exc.value.remediation  # points at re-committing with a ModelArchDescriptor


# --- fail-closed: a rank-deficient probe embedding raises DegenerateProcrustes (no silent garbage Q*) ---


def test_rank_deficient_probe_raises_degenerate(tmp_path: Path) -> None:
    # A small-token config (num_tokens = (2//2)*(4//2)**2 = 4) with a single landmark gives a flattened
    # embedding of just k·N = 1*4 = 4 < d = 8 rows -> M = E_refᵀ f_θ is rank-deficient and the closed-form
    # Procrustes raises DegenerateProcrustes rather than return a silent garbage Q* (RFC-0006 §4).
    cfg = _cfg(num_frames=2, image_size=4, num_tokens=4)
    ckpt_dir = tmp_path / "ckpt"
    _, f_ref = _commit_encoder(ckpt_dir, cfg)
    probe = build_probe(
        _points(16, t=2, h=4, w=4),
        torch.tensor([0]),  # k=1 landmark
        f_ref,  # type: ignore[arg-type]
    )
    probe_path = tmp_path / "probe.safetensors"
    save_probe(probe, probe_path)  # type: ignore[arg-type]
    with pytest.raises(DegenerateProcrustes):
        recompute_alignment(ckpt_dir, probe_path)
    with pytest.raises(DegenerateProcrustes):
        recompute_alignment_claim(ckpt_dir, probe_path)


# --- the on-disk records: schema gating + round-trip ---


def test_claim_round_trips_and_gates_schema() -> None:
    claim = AlignmentClaim(
        round_index=3,
        procrustes_q_hash="a" * 64,
        procrustes_residual=0.1,
        rotation_angle_deg=2.5,
        probe_hash="b" * 64,
    )
    assert parse_alignment_claim(claim.model_dump()) == claim
    with pytest.raises(SchemaVersionMismatch):
        parse_alignment_claim(
            {**claim.model_dump(), "schema_version": ALIGNMENT_SCHEMA_VERSION + 1}
        )


def test_recomputation_round_trips_and_gates_schema(tmp_path: Path) -> None:
    ckpt_dir, probe_path = _honest_setup(tmp_path)
    record = recompute_alignment_claim(ckpt_dir, probe_path)
    assert parse_alignment_recomputation(record.model_dump()) == record
    with pytest.raises(SchemaVersionMismatch):
        parse_alignment_recomputation(
            {**record.model_dump(), "schema_version": ALIGNMENT_SCHEMA_VERSION + 1}
        )


# --- CLI: exits 0 on an honest run, non-zero on a mismatched --expected claim ---


def test_cli_recompute_honest_exits_zero(tmp_path: Path) -> None:
    ckpt_dir, probe_path = _honest_setup(tmp_path)
    res = _runner.invoke(
        app,
        [
            "verify",
            "recompute",
            "--checkpoint",
            str(ckpt_dir),
            "--probe",
            str(probe_path),
        ],
    )
    assert res.exit_code == 0, res.output
    # the report JSON is on stdout
    payload = json.loads(res.stdout.strip().splitlines()[0])
    assert (
        "drift_from_global" in payload and "committed" in payload["drift_from_global"]
    )


def test_cli_recompute_matching_expected_exits_zero(tmp_path: Path) -> None:
    ckpt_dir, probe_path = _honest_setup(tmp_path)
    honest = recompute_alignment_claim(ckpt_dir, probe_path).recomputed
    claim_path = tmp_path / "claim.json"
    claim_path.write_text(honest.model_dump_json(), encoding="utf-8")
    res = _runner.invoke(
        app,
        [
            "verify",
            "recompute",
            "--checkpoint",
            str(ckpt_dir),
            "--probe",
            str(probe_path),
            "--expected",
            str(claim_path),
        ],
    )
    assert res.exit_code == 0, res.output
    record = parse_alignment_recomputation(
        json.loads(res.stdout.strip().splitlines()[0])
    )
    assert record.matches_expected is True


def test_cli_recompute_mismatched_expected_exits_nonzero(tmp_path: Path) -> None:
    ckpt_dir, probe_path = _honest_setup(tmp_path)
    honest = recompute_alignment_claim(ckpt_dir, probe_path).recomputed
    perturbed = honest.model_copy(update={"procrustes_q_hash": "f" * 64})
    claim_path = tmp_path / "bad_claim.json"
    claim_path.write_text(perturbed.model_dump_json(), encoding="utf-8")
    res = _runner.invoke(
        app,
        [
            "verify",
            "recompute",
            "--checkpoint",
            str(ckpt_dir),
            "--probe",
            str(probe_path),
            "--expected",
            str(claim_path),
        ],
    )
    assert res.exit_code != 0
    # the record JSON (with matches_expected=False) was still emitted to stdout before the non-zero exit
    record = parse_alignment_recomputation(
        json.loads(res.stdout.strip().splitlines()[0])
    )
    assert record.matches_expected is False


def test_cli_recompute_missing_expected_file_exits_nonzero(tmp_path: Path) -> None:
    ckpt_dir, probe_path = _honest_setup(tmp_path)
    res = _runner.invoke(
        app,
        [
            "verify",
            "recompute",
            "--checkpoint",
            str(ckpt_dir),
            "--probe",
            str(probe_path),
            "--expected",
            str(tmp_path / "does_not_exist.json"),
        ],
    )
    assert res.exit_code != 0


def test_cli_recompute_tampered_checkpoint_exits_nonzero(tmp_path: Path) -> None:
    ckpt_dir, probe_path = _honest_setup(tmp_path)
    save_file(
        {"encoder.norm.weight": torch.ones(_D)},
        str(ckpt_dir / "weights.safetensors"),
    )
    res = _runner.invoke(
        app,
        [
            "verify",
            "recompute",
            "--checkpoint",
            str(ckpt_dir),
            "--probe",
            str(probe_path),
        ],
    )
    assert res.exit_code == 1
    assert "checkpoint_integrity" in res.output
