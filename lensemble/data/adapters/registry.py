"""lensemble.data.adapters.registry — the ``fmt``/URI-scheme dispatcher for the data backends (RFC-0004 §1).

This is the single extension point named in
[02 — Public API §5.2](../../../docs/spec/02-public-api.md#52-registering-a-new-data-adapter): a
module-level registry keyed by ``fmt`` (``"lance" | "hdf5" | "lerobot"`` and any user key). The three
built-in adapters self-register at import; a new adapter plugs in through :func:`register_adapter` and is
selected the same way the built-ins are — by ``EpisodeDataset.fmt`` or, for the read-only LeRobot view,
the ``lerobot://`` URI scheme.

Residency (``INV-RESIDENCY``, 02 §5.2): every adapter materializes RAW, local episodes *inside* the
trust boundary. The on-disk ``lance``/``hdf5`` files are local participant artifacts; no adapter exposes
an egress / serialize-outbound path (a boundary-crossing payload is inspected only by
``lensemble.data.residency.guard_egress``). The read-only ``lerobot://`` view never participates in
commitment or egress by construction (RFC-0004 §1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:  # avoid a runtime import cycle through dataset.py at module import
    from lensemble.data.dataset import EpisodeDataset, Format

# An adapter loader resolves a source (path or URI) to a read-back EpisodeDataset; a saver (optional —
# a read-only adapter like lerobot has none) writes the in-memory dataset to disk.
Loader = Callable[["str | Path"], "EpisodeDataset"]
Saver = Callable[["EpisodeDataset", Path], None]


@dataclass(frozen=True)
class _Adapter:
    """A registered backend: its loader and (optional) saver. ``saver is None`` ⇒ read-only."""

    fmt: str
    loader: Loader
    saver: Optional[Saver]


# Module-level registry the dispatcher consults (02 §5.2). Built-ins register at import (see __init__).
_REGISTRY: dict[str, _Adapter] = {}

# Path suffix -> fmt, for `load_episodes(path)` without an explicit fmt (the lerobot view has no suffix).
_SUFFIX_TO_FMT: dict[str, str] = {
    ".lance": "lance",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
}

_LEROBOT_SCHEME = "lerobot://"
# LeRobot-layout single-file HDF5 (episode_index + observation/pixels* + action). Its path ends in `.h5`,
# which the suffix table maps to the lensemble `hdf5` store, so it MUST be selected by an explicit scheme
# (or fmt=) rather than the suffix.
_LEROBOT_H5_SCHEME = "lerobot-h5://"


def register_adapter(
    fmt: str,
    *,
    loader: Loader,
    saver: Saver | None = None,
) -> None:
    """Register (or replace) a data adapter under ``fmt`` (the 02 §5.2 extension point).

    ``loader`` resolves a source to a read-back :class:`~lensemble.data.dataset.EpisodeDataset`; ``saver``
    writes one to disk and is omitted for a read-only adapter (e.g. ``lerobot``), in which case
    :func:`save_episodes` raises for that ``fmt``. Re-registering an ``fmt`` replaces it (so a test can
    rebind a deterministic stub). A registered adapter MUST honor residency (``INV-RESIDENCY``): it
    materializes local raw episodes only and exposes no egress path.
    """
    _REGISTRY[fmt] = _Adapter(fmt=fmt, loader=loader, saver=saver)


def _require(fmt: str) -> _Adapter:
    adapter = _REGISTRY.get(fmt)
    if adapter is None:
        raise ValueError(
            f"unknown data-adapter fmt {fmt!r}; "
            f"registered: {sorted(_REGISTRY)} (register one with register_adapter)"
        )
    return adapter


def save_episodes(dataset: "EpisodeDataset", path: Path, *, fmt: "Format") -> None:
    """Write ``dataset``'s in-memory episodes to ``path`` via the backend keyed by ``fmt`` (RFC-0004 §1).

    Raises :class:`ValueError` on an unknown ``fmt`` or on a read-only adapter (``lerobot`` has no saver —
    the ``lerobot://`` view is read-only by construction). The written file is a local participant
    artifact (``INV-RESIDENCY``); it is never an egress payload.
    """
    adapter = _require(fmt)
    if adapter.saver is None:
        raise ValueError(
            f"adapter {fmt!r} is read-only and cannot save; "
            "the lerobot:// view is read-only by construction (RFC-0004 §1) — "
            "write through the lance or hdf5 backend instead"
        )
    adapter.saver(dataset, Path(path))


def _resolve_fmt(source: "str | Path", fmt: "Format | None") -> str:
    """Resolve the backend key from an explicit ``fmt``, the ``lerobot://`` scheme, or the path suffix."""
    if fmt is not None:
        return fmt
    text = str(source)
    if text.startswith(_LEROBOT_H5_SCHEME):
        return "lerobot-h5"
    if text.startswith(_LEROBOT_SCHEME):
        return "lerobot"
    suffix = Path(text).suffix.lower()
    resolved = _SUFFIX_TO_FMT.get(suffix)
    if resolved is None:
        raise ValueError(
            f"cannot infer a data-adapter fmt from {text!r}: unknown suffix {suffix!r}; "
            f"pass fmt= explicitly (known suffixes: {sorted(_SUFFIX_TO_FMT)})"
        )
    return resolved


def load_episodes(
    source: "str | Path", *, fmt: "Format | None" = None
) -> "EpisodeDataset":
    """Resolve ``source`` to a read-back :class:`~lensemble.data.dataset.EpisodeDataset` (RFC-0004 §1).

    Backend selection: an explicit ``fmt``; OR a ``lerobot://<repo_id>`` URI → the read-only lerobot
    adapter; OR inference from the path suffix (``.lance`` → ``lance``; ``.h5``/``.hdf5`` → ``hdf5``).
    The returned dataset carries the same ``fmt``. Raises :class:`ValueError` on an unresolvable source
    or unknown ``fmt``. The materialized episodes are RAW and local (``INV-RESIDENCY``).
    """
    resolved = _resolve_fmt(source, fmt)
    adapter = _require(resolved)
    return adapter.loader(source)
