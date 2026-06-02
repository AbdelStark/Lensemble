"""lensemble.cli — the Typer CLI app (docs/rfcs/RFC-0001; CLI surface 02-public-api 4).

This module hosts the ``lensemble`` CLI. The full command set (``train``, ``federate``, ``eval``,
``doctor``, ...) and the shared ``--config`` / ``RunManifest`` plumbing land with cli-skeleton (#5);
this file currently provides the ``probe`` command group (#24). Commands are additive — #5 extends
``app`` without disturbing ``probe``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import typer
from safetensors import safe_open

if TYPE_CHECKING:
    from torch import Tensor

from lensemble.data.probe import (
    build_probe,
    load_probe,
    probe_content_hash,
    probe_record,
    save_probe,
    verify_probe_pin,
)
from lensemble.errors import LensembleError

app = typer.Typer(
    add_completion=False, help="Lensemble: federated, end-to-end JEPA world models."
)
probe_app = typer.Typer(
    add_completion=False, help="Manage the shared public probe set P (RFC-0004 3)."
)
app.add_typer(probe_app, name="probe")


def _load_points(path: Path) -> "Tensor":
    with safe_open(str(path), framework="pt") as f:  # type: ignore[no-untyped-call]
        if "points" not in f.keys():
            raise typer.BadParameter(f"{path} has no 'points' tensor")
        return f.get_tensor("points")


@probe_app.command("build")
def probe_build(
    points: Path = typer.Option(
        ..., help="safetensors file with a 'points' tensor (P, T, C, H, W)"
    ),
    out: Path = typer.Option(..., help="output probe path"),
    d: int = typer.Option(..., help="latent dimension d"),
    num_frames: int = typer.Option(...),
    image_size: int = typer.Option(...),
    patch_size: int = typer.Option(...),
    tubelet: int = typer.Option(...),
    depth: int = typer.Option(2),
    num_heads: int = typer.Option(4),
    in_channels: int = typer.Option(3),
    k: int = typer.Option(..., help="number of landmarks (must be >= d)"),
    warmstart: Path | None = typer.Option(
        None, help="pinned V-JEPA 2 warm-start safetensors"
    ),
    expected_hash: str | None = typer.Option(None, help="expected warm-start SHA-256"),
    probe_version: int = typer.Option(1),
) -> None:
    """Build a PublicProbe: derive landmark targets from a pinned f_ref and pin the content hash."""
    import torch

    from lensemble.model.encoder import (
        build_encoder,
        load_warmstart,
        snapshot_reference,
    )

    pts = _load_points(points)
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            d=d,
            num_frames=num_frames,
            image_size=image_size,
            patch_size=patch_size,
            tubelet=tubelet,
            depth=depth,
            num_heads=num_heads,
            in_channels=in_channels,
        )
    )
    encoder = build_encoder(cfg)
    if warmstart is not None:
        if expected_hash is None:
            raise typer.BadParameter("--expected-hash is required with --warmstart")
        load_warmstart(encoder, warmstart, expected_hash=expected_hash)
    f_ref = snapshot_reference(encoder)
    probe = build_probe(pts, torch.arange(k), f_ref, probe_version=probe_version)
    save_probe(probe, out)
    typer.echo(probe_record(probe))


@probe_app.command("pin")
def probe_pin(
    probe_path: Path = typer.Argument(..., help="probe file to (re)pin"),
) -> None:
    """Recompute and report the probe content hash (the pinned hash for RoundOpen)."""
    probe = load_probe(probe_path)
    recomputed = probe_content_hash(probe.points, probe.landmark_idx)
    typer.echo(recomputed.hex())
    if recomputed != probe.content_hash:
        typer.echo(
            "warning: stored content_hash differs from recomputed; re-saving pin",
            err=True,
        )
        save_probe(
            type(probe)(
                probe.points,
                probe.landmark_idx,
                probe.landmark_targets,
                recomputed,
                probe.probe_version,
            ),
            probe_path,
        )


@probe_app.command("verify")
def probe_verify(
    probe_path: Path = typer.Argument(..., help="probe file to verify"),
    against: str = typer.Option(
        ..., "--hash", help="expected probe content hash (hex)"
    ),
) -> None:
    """Verify a held probe against a pinned hash (INV-PROBE-PIN). Exit 1 on mismatch/under-coverage."""
    probe = load_probe(probe_path)
    try:
        verify_probe_pin(probe, bytes.fromhex(against))
    except LensembleError as err:
        typer.echo(f"{err.code.value}: {err}", err=True)
        raise typer.Exit(code=1) from err
    typer.echo("ok")


def main() -> None:  # pragma: no cover - entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())  # type: ignore[func-returns-value]
