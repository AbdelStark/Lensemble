"""Probe versioning / re-anchoring (RFC-0004 §3.1; #25).

A probe content change (or a warm-start change) is a deliberate, recorded, federation-wide re-anchoring
event: it bumps `probe_version`, recomputes `content_hash`, and recomputes landmark targets against the
current `f_ref` (`INV-PROBE-PIN`). A mid-run probe-hash mismatch is rejected; the RunManifest records the
probe version so a run is reproducible against the exact probe it used.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lensemble.config import load
from lensemble.config.manifest import build_manifest
from lensemble.data.probe import build_probe, reanchor_probe, verify_probe_pin
from lensemble.errors import LensembleErrorCode, ProbeError
from lensemble.model.encoder import build_encoder, snapshot_reference

_CFG = SimpleNamespace(
    model=SimpleNamespace(
        d=4, num_frames=2, image_size=4, patch_size=2, tubelet=2, depth=2, num_heads=2
    )
)


def _f_ref(seed: int):
    torch.manual_seed(seed)
    return snapshot_reference(build_encoder(_CFG))


def _points(p: int = 6, seed: int = 0) -> torch.Tensor:
    return torch.randn(p, 2, 3, 4, 4, generator=torch.Generator().manual_seed(seed))


def test_content_change_bumps_version_and_hash() -> None:
    f_ref = _f_ref(1)
    v1 = build_probe(_points(seed=0), torch.arange(4), f_ref, probe_version=1)
    v2 = reanchor_probe(v1, f_ref, points=_points(seed=99))  # new content
    assert v2.probe_version == v1.probe_version + 1 == 2
    assert v2.content_hash != v1.content_hash


def test_reanchor_recomputes_targets_against_current_f_ref() -> None:
    points = _points(seed=0)
    f_ref_a, f_ref_b = _f_ref(1), _f_ref(2)  # distinct warm-starts
    v1 = build_probe(points, torch.arange(4), f_ref_a, probe_version=1)
    v2 = reanchor_probe(v1, f_ref_b)  # same content, new f_ref
    assert v2.probe_version == 2
    assert v2.content_hash == v1.content_hash  # points/landmark_idx unchanged
    # the targets are recomputed against the new f_ref: they differ from v1 and match f_ref_b
    assert not torch.equal(v2.landmark_targets, v1.landmark_targets)
    expected = f_ref_b(points[torch.arange(4)]).tokens.detach()
    assert torch.allclose(v2.landmark_targets, expected)


def test_roundopen_with_mismatched_probe_hash_rejected() -> None:
    f_ref = _f_ref(1)
    probe = build_probe(_points(), torch.arange(4), f_ref)
    # the pinned hash accepts itself; a different broadcast hash is rejected (INV-PROBE-PIN)
    assert verify_probe_pin(probe, probe.content_hash) is None
    with pytest.raises(ProbeError) as exc:
        verify_probe_pin(probe, b"\x01" * 32)
    assert exc.value.code == LensembleErrorCode.PROBE_INVALID


def test_manifest_records_probe_version() -> None:
    f_ref = _f_ref(1)
    v1 = build_probe(_points(), torch.arange(4), f_ref, probe_version=1)
    v2 = reanchor_probe(v1, f_ref, points=_points(seed=7))
    cfg = load()
    manifest = build_manifest(
        cfg, probe_hash=v2.content_hash.hex(), probe_version=v2.probe_version
    )
    assert manifest.probe_version == v2.probe_version == 2
    assert manifest.probe_hash == v2.content_hash.hex()
    # the field round-trips through the manifest JSON
    from lensemble.config.manifest import RunManifest

    again = RunManifest.model_validate_json(manifest.model_dump_json())
    assert again.probe_version == 2
