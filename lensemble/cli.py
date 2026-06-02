"""lensemble.cli — the Typer CLI app (docs/rfcs/RFC-0001; CLI surface 02-public-api 4).

Hosts the ``lensemble`` CLI: the command tree of 02-public-api 4 (``train``, ``federate
coordinator|participant``, ``eval``, ``probe build|pin|verify``, ``commit dataset``, ``drift``,
``verify recompute|prove``, ``doctor``), the shared ``--config`` + Hydra ``key=value`` override
loading into a ``LensembleConfig``, per-command ``RunManifest`` emission, and the stdout/exit-code
contract. Issue #5.

Contract (02-public-api 4): machine-readable output (the manifest path, hashes, report JSON) goes to
**stdout**; human-readable progress/notes go to **stderr**. Exit codes: ``0`` success; ``1`` for any
:class:`~lensemble.errors.LensembleError` (the message carries ``.code`` and ``.remediation``); ``2``
a Typer/usage error; ``130`` interrupted (04-error-model 7.1).

The domain commands here are skeletons: they load+validate config at the boundary and emit a manifest;
the actual train/federate/eval/provenance behavior lands with each owning subsystem. The canonical
``RunManifest`` schema and serialization are owned by #36 — until it lands, :func:`_emit_manifest`
writes a minimal manifest file and the command prints its path; #36 routes this through
``RunManifest`` proper without changing the CLI surface.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import typer
from safetensors import safe_open

from lensemble import __version__
from lensemble.config import SEED_DERIVATION, LensembleConfig, load
from lensemble.data.probe import (
    build_probe,
    load_probe,
    probe_content_hash,
    probe_record,
    save_probe,
    verify_probe_pin,
)
from lensemble.errors import ConfigError, LensembleError, LensembleErrorCode
from lensemble.verify import stark

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import Tensor

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Lensemble: federated, end-to-end JEPA world models.",
)

# Known encoder identities a warm-start release can name (02-public-api 1.1 / config schema).
_KNOWN_ENCODERS = frozenset({"vjepa2-vit-l", "vjepa2-vit-h", "vjepa2-vit-g"})


# --- supervisory boundary: map a LensembleError to exit 1, logging code + remediation (04 7 rule 1) ---
@contextmanager
def _supervise() -> "Iterator[None]":
    """The one permitted broad-ish catch (04 7 rule 1): a ``LensembleError`` exits 1 with its code and
    remediation on stderr. ``typer.Exit`` (already an exit-code carrier) passes straight through."""
    try:
        yield
    except LensembleError as err:
        typer.echo(f"{err.code.value}: {err}", err=True)
        if err.remediation:
            typer.echo(f"remediation: {err.remediation}", err=True)
        raise typer.Exit(code=1) from err


def _compose(config: Path | None, overrides: list[str] | None) -> LensembleConfig:
    """Compose a ``LensembleConfig`` from ``--config`` + ``key=value`` overrides (validate at boundary).

    An unknown override key or invalid value raises :class:`ConfigError` (``CONFIG_INVALID``) before any
    work begins. A given-but-missing ``--config`` path is itself a ``CONFIG_INVALID`` failure.
    """
    ovr = list(overrides or [])
    if config is None:
        return load(overrides=ovr)
    if not config.exists():
        raise ConfigError(
            f"config file not found: {config}",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="pass an existing --config path, or omit --config to use the defaults",
        )
    return load(config_name=config.stem, overrides=ovr, config_dir=config.parent)


def _emit_manifest(cfg: LensembleConfig, *, command: str, run_dir: Path) -> Path:
    """Write a minimal RunManifest to ``run_dir`` and return its path (machine output: 02 4).

    The canonical schema/serialization is #36's; this records the command, the resolved config, the
    seed-derivation tag, and the version — enough to reproduce the invocation — and is replaced in place
    by ``RunManifest`` when #36 lands.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "run-manifest/skeleton-0",
        "command": command,
        "lensemble_version": __version__,
        "run_mode": cfg.run_mode,
        "seed_derivation": SEED_DERIVATION,
        "root_seed": cfg.determinism.root_seed,
        "config": asdict(cfg),
    }
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _stub_command(
    command: str,
    config: Path | None,
    overrides: list[str] | None,
    run_dir: Path,
    *,
    owner: str,
) -> None:
    """Shared skeleton body: compose config, emit a manifest, print its path; note the owning subsystem."""
    with _supervise():
        cfg = _compose(config, overrides)
        path = _emit_manifest(cfg, command=command, run_dir=run_dir)
        typer.echo(str(path))  # machine-readable: manifest path -> stdout
        typer.echo(
            f"{command}: config validated, manifest written; behavior owned by {owner}.",
            err=True,  # human-readable -> stderr
        )


# Shared option/argument shapes (kept identical across the domain commands).
_CONFIG_OPT = typer.Option(
    None, "--config", help="base config .yaml (omit for built-in defaults)"
)
_RUNDIR_OPT = typer.Option(
    Path("runs"), "--run-dir", help="run directory the RunManifest is written to"
)
_OVERRIDES_ARG = typer.Argument(
    None, help="Hydra dot-path overrides, e.g. federation.participant_count=8"
)


@app.command("train")
def train(
    config: Path | None = _CONFIG_OPT,
    run_dir: Path = _RUNDIR_OPT,
    overrides: list[str] | None = _OVERRIDES_ARG,
) -> None:
    """Train a single-node JEPA world model end-to-end (the centralized baseline)."""
    _stub_command("train", config, overrides, run_dir, owner="model.train_local")


federate_app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="Run a federated training round."
)
app.add_typer(federate_app, name="federate")


@federate_app.command("coordinator")
def federate_coordinator(
    listen: str = typer.Option(
        "in-process", "--listen", help="address to listen on (transport per RFC-0013)"
    ),
    config: Path | None = _CONFIG_OPT,
    run_dir: Path = _RUNDIR_OPT,
    overrides: list[str] | None = _OVERRIDES_ARG,
) -> None:
    """Run the aggregation coordinator for a federated run (RFC-0003 / RFC-0013)."""
    _stub_command(
        "federate coordinator",
        config,
        overrides,
        run_dir,
        owner="federation.Coordinator",
    )


@federate_app.command("participant")
def federate_participant(
    coordinator: str = typer.Option(
        "in-process",
        "--coordinator",
        help="coordinator address (transport per RFC-0013)",
    ),
    config: Path | None = _CONFIG_OPT,
    run_dir: Path = _RUNDIR_OPT,
    overrides: list[str] | None = _OVERRIDES_ARG,
) -> None:
    """Run a federated participant: local inner loop, released pseudo-gradient (RFC-0003)."""
    _stub_command(
        "federate participant",
        config,
        overrides,
        run_dir,
        owner="federation.Participant",
    )


@app.command("eval")
def eval_(
    config: Path | None = _CONFIG_OPT,
    run_dir: Path = _RUNDIR_OPT,
    overrides: list[str] | None = _OVERRIDES_ARG,
) -> None:
    """Evaluate a checkpoint by latent-MPC rollout on a held-out env (RFC-0005)."""
    _stub_command("eval", config, overrides, run_dir, owner="eval.evaluate")


commit_app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="Dataset provenance commitments."
)
app.add_typer(commit_app, name="commit")


@commit_app.command("dataset")
def commit_dataset(
    config: Path | None = _CONFIG_OPT,
    run_dir: Path = _RUNDIR_OPT,
    overrides: list[str] | None = _OVERRIDES_ARG,
) -> None:
    """Compute a dataset commitment (Merkle root R_c + WMCP metadata, RFC-0014)."""
    _stub_command(
        "commit dataset", config, overrides, run_dir, owner="provenance.commit_dataset"
    )


@app.command("drift")
def drift(
    config: Path | None = _CONFIG_OPT,
    run_dir: Path = _RUNDIR_OPT,
    overrides: list[str] | None = _OVERRIDES_ARG,
) -> None:
    """Measure latent frame-drift against the probe anchor (RFC-0002)."""
    _stub_command("drift", config, overrides, run_dir, owner="gauge.frame_drift")


verify_app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="Verifiable-contribution layer."
)
app.add_typer(verify_app, name="verify")


@verify_app.command("recompute")
def verify_recompute(
    config: Path | None = _CONFIG_OPT,
    run_dir: Path = _RUNDIR_OPT,
    overrides: list[str] | None = _OVERRIDES_ARG,
) -> None:
    """Recompute and check a round's aggregation/alignment from released artifacts (RFC-0006)."""
    _stub_command(
        "verify recompute",
        config,
        overrides,
        run_dir,
        owner="verify.recompute_alignment",
    )


@verify_app.command("prove")
def verify_prove(
    config: Path | None = _CONFIG_OPT,
    overrides: list[str] | None = _OVERRIDES_ARG,
) -> None:
    """Produce a succinct proof of correct contribution (Phase 2; not available in Phase 1)."""
    with _supervise():
        cfg = _compose(config, overrides)  # still validate config at the boundary
    try:
        stark.prove_round(
            cfg
        )  # the Phase-2 seam; raises NotImplementedError in Phase 1
    except NotImplementedError as exc:
        typer.echo(f"verify prove: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("doctor")
def doctor(
    config: Path | None = _CONFIG_OPT,
    overrides: list[str] | None = _OVERRIDES_ARG,
) -> None:
    """Self-check the environment and config: Python/torch versions, warm-start, determinism."""
    with _supervise():
        cfg = _compose(config, overrides)
        checks = _doctor_checks(cfg)
        ok = all(c["ok"] for c in checks)
        typer.echo(json.dumps({"ok": ok, "checks": checks}, sort_keys=True))  # stdout
        for c in checks:
            mark = "ok" if c["ok"] else "FAIL"
            typer.echo(f"[{mark}] {c['name']}: {c['detail']}", err=True)
        if not ok:
            raise typer.Exit(code=1)


def _doctor_checks(cfg: LensembleConfig) -> list[dict[str, object]]:
    """The ordered doctor checks (02 4): each is ``{name, ok, detail}``; any ``ok=False`` fails doctor."""
    import torch

    checks: list[dict[str, object]] = []

    py_ok = sys.version_info >= (3, 11)
    checks.append(
        {
            "name": "python",
            "ok": py_ok,
            "detail": f"{sys.version_info.major}.{sys.version_info.minor} (need >= 3.11)",
        }
    )

    torch_parts = torch.__version__.split("+")[0].split(".")
    torch_ver = (int(torch_parts[0]), int(torch_parts[1]))
    torch_ok = (2, 4) <= torch_ver < (3, 0)
    checks.append(
        {
            "name": "torch",
            "ok": torch_ok,
            "detail": f"{torch.__version__} (need >= 2.4, < 3)",
        }
    )

    ws = cfg.model.warm_start_release.strip()
    ws_ok = bool(ws) and cfg.model.encoder in _KNOWN_ENCODERS
    checks.append(
        {
            "name": "warm_start",
            "ok": ws_ok,
            "detail": f"release={cfg.model.warm_start_release!r} encoder={cfg.model.encoder!r}",
        }
    )

    det_ok = cfg.determinism.deterministic_aggregation is True
    checks.append(
        {
            "name": "determinism",
            "ok": det_ok,
            "detail": "deterministic_aggregation must be on (INV-AGG-DETERMINISM)",
        }
    )

    # A cheap, in-process aggregation-determinism self-check: a fixed-order reduction at the configured
    # dtype is bitwise-reproducible. The full two-process bitwise gate is #68 / 07 §8.5.
    dtype = (
        torch.float64 if cfg.determinism.aggregation_dtype == "fp64" else torch.float32
    )
    parts = [torch.tensor([1.0, 2.0, 4.0], dtype=dtype) for _ in range(8)]
    first = torch.zeros(3, dtype=dtype)
    second = torch.zeros(3, dtype=dtype)
    for p in parts:
        first = first + p
    for p in parts:
        second = second + p
    agg_ok = bool(torch.equal(first, second))
    checks.append(
        {
            "name": "aggregation_determinism",
            "ok": agg_ok,
            "detail": f"fixed-order reduction bitwise-stable at {cfg.determinism.aggregation_dtype}",
        }
    )
    return checks


# --- probe command group (#24): manage the shared public probe set P (RFC-0004 3) ---
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
    try:
        app()
    except KeyboardInterrupt:  # 130 interrupted (02-public-api 4 exit-code contract)
        typer.echo("interrupted", err=True)
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
