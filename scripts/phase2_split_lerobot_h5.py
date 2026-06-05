#!/usr/bin/env python3
"""Split one LeRobot-H5 file into deterministic Phase 2 participant silos.

The split policy is episode-level modulo assignment: source episode k goes to
``k % num_silos``. Frames are never duplicated across output silos, episode
order is preserved inside each silo, and ``episode_index`` is remapped to local
0-based ids. The script copies row-aligned HDF5 datasets in chunks so large
camera stacks do not need to be materialized in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

PHASE2_SPLIT_MANIFEST_SCHEMA_VERSION = 1
_EPISODE_INDEX = "episode_index"
_ACTION = "action"


@dataclass(frozen=True)
class _Segment:
    source_episode: int
    source_start: int
    source_end: int

    @property
    def frame_count(self) -> int:
        return self.source_end - self.source_start


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def _episode_segments(ep_index: np.ndarray) -> list[_Segment]:
    if ep_index.ndim != 1:
        raise ValueError(f"episode_index must be 1-D, got shape {ep_index.shape}")
    if len(ep_index) == 0:
        raise ValueError("episode_index is empty")
    bounds = np.flatnonzero(np.diff(ep_index)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(ep_index)]])
    return [
        _Segment(source_episode=i, source_start=int(start), source_end=int(end))
        for i, (start, end) in enumerate(zip(starts.tolist(), ends.tolist()))
    ]


def _assigned_segments(
    segments: list[_Segment], num_silos: int
) -> list[list[_Segment]]:
    assigned: list[list[_Segment]] = [[] for _ in range(num_silos)]
    for segment in segments:
        assigned[segment.source_episode % num_silos].append(segment)
    return assigned


def _dataset_kwargs(source: h5py.Dataset, shape: tuple[int, ...]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if source.compression is not None:
        kwargs["compression"] = source.compression
        if source.compression_opts is not None:
            kwargs["compression_opts"] = source.compression_opts
    if source.shuffle:
        kwargs["shuffle"] = source.shuffle
    if source.fletcher32:
        kwargs["fletcher32"] = source.fletcher32
    if source.chunks is not None and shape:
        first = min(max(1, int(source.chunks[0])), max(1, shape[0]))
        kwargs["chunks"] = (first, *source.chunks[1:])
    return kwargs


def _create_row_dataset(
    src: h5py.Dataset,
    dst_group: h5py.Group,
    name: str,
    *,
    shape: tuple[int, ...],
) -> h5py.Dataset:
    return dst_group.create_dataset(
        name,
        shape=shape,
        dtype=src.dtype,
        **_dataset_kwargs(src, shape),
    )


def _copy_row_aligned_dataset(
    src: h5py.Dataset,
    dst_group: h5py.Group,
    name: str,
    *,
    segments: list[_Segment],
    frame_count: int,
) -> None:
    dst = _create_row_dataset(
        src,
        dst_group,
        name,
        shape=(frame_count, *src.shape[1:]),
    )
    cursor = 0
    for segment in segments:
        length = segment.frame_count
        dst[cursor : cursor + length] = src[segment.source_start : segment.source_end]
        cursor += length
    _copy_attrs(src.attrs, dst.attrs)


def _write_episode_index(
    src: h5py.Dataset,
    dst_group: h5py.Group,
    *,
    segments: list[_Segment],
    frame_count: int,
) -> None:
    dst = _create_row_dataset(src, dst_group, _EPISODE_INDEX, shape=(frame_count,))
    cursor = 0
    for local_episode, segment in enumerate(segments):
        length = segment.frame_count
        dst[cursor : cursor + length] = np.full(length, local_episode, dtype=src.dtype)
        cursor += length
    _copy_attrs(src.attrs, dst.attrs)


def _copy_node(
    src_node: h5py.Group | h5py.Dataset,
    dst_group: h5py.Group,
    name: str,
    *,
    total_frames: int,
    segments: list[_Segment],
    frame_count: int,
) -> None:
    if isinstance(src_node, h5py.Group):
        child_group = dst_group.create_group(name)
        _copy_attrs(src_node.attrs, child_group.attrs)
        for child_name, child in src_node.items():
            _copy_node(
                child,
                child_group,
                str(child_name),
                total_frames=total_frames,
                segments=segments,
                frame_count=frame_count,
            )
        return

    if name == _EPISODE_INDEX:
        _write_episode_index(
            src_node,
            dst_group,
            segments=segments,
            frame_count=frame_count,
        )
    elif src_node.ndim >= 1 and int(src_node.shape[0]) == total_frames:
        _copy_row_aligned_dataset(
            src_node,
            dst_group,
            name,
            segments=segments,
            frame_count=frame_count,
        )
    else:
        dst = dst_group.create_dataset(name, data=src_node[()])
        _copy_attrs(src_node.attrs, dst.attrs)


def split_lerobot_h5(
    source: Path,
    output_dir: Path,
    *,
    prefix: str = "phase2-silo",
    num_silos: int = 2,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Split ``source`` into ``num_silos`` LeRobot-H5 participant files."""
    if num_silos < 2:
        raise ValueError(f"num_silos must be >= 2, got {num_silos}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source = source.expanduser().resolve()
    manifest_path = manifest_path or output_dir / "phase2_silo_manifest.json"

    with h5py.File(source, "r") as src:
        for required in (_EPISODE_INDEX, _ACTION):
            if required not in src:
                raise ValueError(
                    f"LeRobot-H5 source missing required dataset {required!r}"
                )
        ep_index = np.asarray(src[_EPISODE_INDEX][:])
        total_frames = int(ep_index.shape[0])
        segments = _episode_segments(ep_index)
        assigned = _assigned_segments(segments, num_silos)
        if any(not silo_segments for silo_segments in assigned):
            raise ValueError(
                f"episode-mod split would produce an empty silo: "
                f"{[len(s) for s in assigned]}"
            )

        silo_manifests: list[dict[str, Any]] = []
        for silo_idx, silo_segments in enumerate(assigned):
            frame_count = sum(segment.frame_count for segment in silo_segments)
            output_path = output_dir / f"{prefix}{silo_idx}.h5"
            with h5py.File(output_path, "w") as dst:
                _copy_attrs(src.attrs, dst.attrs)
                for name, child in src.items():
                    _copy_node(
                        child,
                        dst,
                        str(name),
                        total_frames=total_frames,
                        segments=silo_segments,
                        frame_count=frame_count,
                    )
            silo_manifests.append(
                {
                    "silo_index": silo_idx,
                    "path": str(output_path),
                    "sha256": _sha256_file(output_path),
                    "source_episode_indices": [
                        segment.source_episode for segment in silo_segments
                    ],
                    "episode_count": len(silo_segments),
                    "frame_count": frame_count,
                }
            )

    manifest = {
        "schema_version": PHASE2_SPLIT_MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source),
        "source_sha256": _sha256_file(source),
        "source_size_bytes": source.stat().st_size,
        "policy": "episode_modulo",
        "num_silos": num_silos,
        "total_episodes": len(segments),
        "total_frames": total_frames,
        "silos": silo_manifests,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="phase2-silo")
    parser.add_argument("--num-silos", type=int, default=2)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _args()
    manifest = split_lerobot_h5(
        args.input,
        args.output_dir,
        prefix=str(args.prefix),
        num_silos=int(args.num_silos),
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
