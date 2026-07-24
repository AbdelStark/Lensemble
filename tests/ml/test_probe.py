"""Public probe set, landmark targets, hash pinning (RFC-0004 3). Issue #24. INV-PROBE-PIN."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typer.testing import CliRunner

from lensemble.cli import app
from lensemble.data.probe import (
    PUBLIC_PROBE_HASH_ALGORITHM,
    PUBLIC_PROBE_HASH_VERSION,
    PublicProbe,
    build_probe,
    load_probe,
    probe_content_hash,
    probe_source_hash,
    save_probe,
    verify_probe_pin,
)
from lensemble.errors import LensembleErrorCode, ProbeError
from lensemble.model.encoder import build_encoder, snapshot_reference

_runner = CliRunner()


def _f_ref():
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            d=4,
            num_frames=2,
            image_size=4,
            patch_size=2,
            tubelet=2,
            depth=2,
            num_heads=2,
        )
    )
    enc = build_encoder(cfg)  # d=4, num_tokens = (2//2)*(4//2)**2 = 4
    return enc, snapshot_reference(enc)


def _points(p: int = 6) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(p, 2, 3, 4, 4)


def test_verify_accepts_pinned_hash() -> None:
    _, f_ref = _f_ref()
    probe = build_probe(_points(), torch.arange(4), f_ref)  # k=4, d=4
    assert verify_probe_pin(probe, probe.content_hash) is None
    assert tuple(probe.landmark_targets.shape) == (4, 4, 4)  # (k, N, d)


@pytest.mark.parametrize(
    "device", ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]
)
def test_probe_build_and_forward_are_device_consistent(device: str) -> None:
    """The probe build + a later encoder forward on ``probe.points`` are device-consistent (#182).

    ``_shared_probe`` builds the probe on ``resolve_device()`` so the ``f_ref`` forward (in
    ``build_probe``) and the ``probe_embeddings`` forward run on the encoder's device (CUDA in the inner
    loop). Here the probe is built on ``device`` and an encoder forward is run on its points: it must not
    raise a device/dtype mismatch (the GPU bug), and the content hash stays device-invariant
    (``probe_content_hash`` canonicalizes via ``.cpu()``). On CPU this guards the contract; with CUDA
    present it exercises the real path.
    """
    enc, f_ref = _f_ref()
    enc, f_ref = enc.to(device), f_ref.to(device)
    points = _points().to(device)
    probe = build_probe(points, torch.arange(4), f_ref)
    landmarks = probe.points[probe.landmark_idx]
    tokens = enc(landmarks).tokens  # the probe_embeddings forward — must run on-device
    assert tokens.device.type == device
    assert probe.content_hash == probe_content_hash(
        points.cpu(), torch.arange(4), probe.landmark_targets.cpu()
    )


def test_build_probe_accepts_bfloat16_points_with_float32_reference() -> None:
    _, f_ref = _f_ref()
    points = _points().to(torch.bfloat16)

    probe = build_probe(points, torch.arange(4), f_ref)

    assert probe.points.dtype == torch.bfloat16
    assert probe.landmark_targets.dtype == torch.float32
    assert probe.content_hash == probe_content_hash(
        points, torch.arange(4), probe.landmark_targets
    )


def test_build_probe_canonicalizes_landmark_indices_to_int64() -> None:
    _, f_ref = _f_ref()

    probe = build_probe(_points(), torch.arange(4, dtype=torch.int32), f_ref)

    assert probe.landmark_idx.dtype == torch.int64
    assert probe.landmark_idx.device == probe.points.device

    with pytest.raises(ProbeError, match="rank-1 integer"):
        build_probe(_points(), torch.arange(4, dtype=torch.float32), f_ref)
    with pytest.raises(ProbeError, match="nonempty"):
        build_probe(_points()[:0], torch.zeros(0, dtype=torch.int64), f_ref)


def test_target_change_changes_full_hash_but_not_source_fingerprint() -> None:
    _, f_ref = _f_ref()
    probe = build_probe(_points(), torch.arange(4), f_ref)
    changed_targets = probe.landmark_targets.clone()
    changed_targets[0, 0, 0] += 1.0

    assert probe_source_hash(probe.points, probe.landmark_idx) == probe_source_hash(
        probe.points, probe.landmark_idx
    )
    assert probe.content_hash != probe_content_hash(
        probe.points, probe.landmark_idx, changed_targets
    )


_HASH_DEVICES = [
    "cpu",
    *(["cuda"] if torch.cuda.is_available() else []),
    *(["mps"] if torch.backends.mps.is_available() else []),
]


@pytest.mark.parametrize("device", _HASH_DEVICES)
def test_full_probe_hash_is_device_invariant(device: str) -> None:
    points = _points()
    landmark_idx = torch.arange(4)
    targets = torch.randn(4, 4, 4, generator=torch.Generator().manual_seed(17))
    expected = probe_content_hash(points, landmark_idx, targets)

    assert (
        probe_content_hash(
            points.to(device), landmark_idx.to(device), targets.to(device)
        )
        == expected
    )


def test_full_probe_hash_is_layout_invariant() -> None:
    points = _points()
    landmark_idx = torch.arange(4)
    targets = torch.randn(4, 4, 4, generator=torch.Generator().manual_seed(17))
    noncontiguous_points = points.transpose(-1, -2).contiguous().transpose(-1, -2)
    noncontiguous_idx = torch.stack((landmark_idx, landmark_idx), dim=1)[:, 0]
    noncontiguous_targets = targets.transpose(-1, -2).contiguous().transpose(-1, -2)
    assert not noncontiguous_points.is_contiguous()
    assert not noncontiguous_idx.is_contiguous()
    assert not noncontiguous_targets.is_contiguous()

    assert probe_content_hash(
        noncontiguous_points, noncontiguous_idx, noncontiguous_targets
    ) == probe_content_hash(points, landmark_idx, targets)


def _saved_probe(tmp_path: Path) -> tuple[PublicProbe, Path]:
    _, f_ref = _f_ref()
    probe = build_probe(_points(), torch.arange(4), f_ref)
    path = tmp_path / "probe.safetensors"
    save_probe(probe, path)
    return probe, path


def _probe_payload(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as handle:  # type: ignore[no-untyped-call]
        metadata = dict(handle.metadata() or {})
        for key in ("points", "landmark_idx", "landmark_targets"):
            tensors[key] = handle.get_tensor(key)
    return tensors, metadata


def test_load_probe_rejects_tampered_landmark_targets(tmp_path: Path) -> None:
    _, path = _saved_probe(tmp_path)
    tensors, metadata = _probe_payload(path)
    tensors["landmark_targets"] = tensors["landmark_targets"].clone()
    tensors["landmark_targets"][0, 0, 0] += 1.0
    save_file(tensors, str(path), metadata=metadata)

    with pytest.raises(ProbeError, match="stored content_hash"):
        load_probe(path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("content_hash", "00" * 32),
        ("content_hash_algorithm", "sha256-points-only"),
        ("content_hash_version", "1"),
    ],
)
def test_load_probe_rejects_tampered_hash_metadata(
    tmp_path: Path, key: str, value: str
) -> None:
    _, path = _saved_probe(tmp_path)
    tensors, metadata = _probe_payload(path)
    metadata[key] = value
    save_file(tensors, str(path), metadata=metadata)

    with pytest.raises(ProbeError):
        load_probe(path)


def test_load_probe_rejects_legacy_points_only_metadata(tmp_path: Path) -> None:
    probe, path = _saved_probe(tmp_path)
    tensors, _ = _probe_payload(path)
    save_file(
        tensors,
        str(path),
        metadata={
            "content_hash": probe_source_hash(probe.points, probe.landmark_idx).hex(),
            "probe_version": str(probe.probe_version),
        },
    )

    with pytest.raises(ProbeError, match="legacy.*points-only"):
        load_probe(path)


def test_save_probe_rejects_stale_hash_after_target_change(tmp_path: Path) -> None:
    probe, _ = _saved_probe(tmp_path)
    changed = PublicProbe(
        points=probe.points,
        landmark_idx=probe.landmark_idx,
        landmark_targets=probe.landmark_targets + 1.0,
        content_hash=probe.content_hash,
        probe_version=probe.probe_version,
    )

    with pytest.raises(ProbeError, match="stored content_hash"):
        save_probe(changed, tmp_path / "changed.safetensors")


def test_saved_probe_records_current_hash_contract(tmp_path: Path) -> None:
    _, path = _saved_probe(tmp_path)
    _, metadata = _probe_payload(path)

    assert metadata["content_hash_algorithm"] == PUBLIC_PROBE_HASH_ALGORITHM
    assert metadata["content_hash_version"] == str(PUBLIC_PROBE_HASH_VERSION)


def test_verify_rejects_mismatched_hash() -> None:
    _, f_ref = _f_ref()
    probe = build_probe(_points(), torch.arange(4), f_ref)
    with pytest.raises(ProbeError) as exc:
        verify_probe_pin(probe, b"\x00" * 32)
    assert exc.value.code == LensembleErrorCode.PROBE_INVALID
    assert exc.value.expected_hash == b"\x00" * 32  # type: ignore[attr-defined]
    assert exc.value.got_hash == probe.content_hash  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("points", "landmark_idx", "targets", "message"),
    [
        (
            torch.empty(0, 2),
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, 1, 4),
            "nonempty",
        ),
        (
            torch.zeros(4, 2),
            torch.arange(4, dtype=torch.int32),
            torch.zeros(4, 1, 4),
            "torch.int64",
        ),
        (
            torch.zeros(4, 2),
            torch.arange(4),
            torch.zeros(4, 4),
            "tensor ranks",
        ),
    ],
)
def test_verify_rejects_malformed_probe_tensor_contract(
    points: torch.Tensor,
    landmark_idx: torch.Tensor,
    targets: torch.Tensor,
    message: str,
) -> None:
    probe = PublicProbe(
        points=points,
        landmark_idx=landmark_idx,
        landmark_targets=targets,
        content_hash=b"\x00" * 32,
        probe_version=1,
    )

    with pytest.raises(ProbeError, match=message):
        verify_probe_pin(probe, probe.content_hash)


def test_under_coverage_k_lt_d_rejected() -> None:
    _, f_ref = _f_ref()
    probe = build_probe(_points(), torch.arange(2), f_ref)  # k=2 < d=4
    with pytest.raises(ProbeError) as exc:
        verify_probe_pin(probe, probe.content_hash)
    assert exc.value.num_landmarks == 2 and exc.value.d == 4  # type: ignore[attr-defined]


def test_targets_derive_only_from_f_ref() -> None:
    enc, f_ref = _f_ref()
    points = _points()
    probe = build_probe(points, torch.arange(4), f_ref)
    targets0 = probe.landmark_targets.clone()
    # a later "training step" mutates the live encoder; f_ref (and the probe) must be unaffected
    with torch.no_grad():
        enc.pos_embed.add_(1.0)
    later = snapshot_reference(enc)(points[torch.arange(4)]).tokens
    assert torch.equal(probe.landmark_targets, targets0)  # unchanged (INV-PROBE-PIN)
    assert not torch.allclose(
        later, targets0
    )  # the current encoder would give different targets


def test_cli_build_pin_verify(tmp_path: Path) -> None:
    points_path = tmp_path / "points.safetensors"
    save_file({"points": _points()}, str(points_path))
    out = tmp_path / "probe.safetensors"
    res = _runner.invoke(
        app,
        [
            "probe",
            "build",
            "--points",
            str(points_path),
            "--out",
            str(out),
            "--d",
            "4",
            "--num-frames",
            "2",
            "--image-size",
            "4",
            "--patch-size",
            "2",
            "--tubelet",
            "2",
            "--num-heads",
            "2",
            "--k",
            "4",
        ],
    )
    assert res.exit_code == 0, res.output
    assert '"d": 4' in res.output and out.exists()

    pin = _runner.invoke(app, ["probe", "pin", str(out)])
    assert pin.exit_code == 0
    content_hash_hex = pin.output.strip().splitlines()[0]

    ok = _runner.invoke(app, ["probe", "verify", str(out), "--hash", content_hash_hex])
    assert ok.exit_code == 0 and "ok" in ok.output

    bad = _runner.invoke(app, ["probe", "verify", str(out), "--hash", "00" * 32])
    assert bad.exit_code == 1 and "probe_invalid" in bad.output

    for invalid_hash in ("not-hex", "00" * 31):
        invalid = _runner.invoke(
            app, ["probe", "verify", str(out), "--hash", invalid_hash]
        )
        assert invalid.exit_code == 1
        assert "invalid --hash" in invalid.output or "exactly 64" in invalid.output
