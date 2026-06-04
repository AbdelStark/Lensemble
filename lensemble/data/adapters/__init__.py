"""lensemble.data.adapters — the on-disk storage backends behind ``EpisodeDataset.fmt`` (RFC-0004 §1).

Three backends resolve through one ``fmt``/URI-scheme dispatcher (:mod:`registry`): the ``lance``
reference store (default; append-friendly, indexed window reads), the portable single-file ``hdf5``
store, and the read-only ``lerobot://<repo_id>`` adapter (an optional extra, conformance-checked on
load). A new adapter plugs in through :func:`register_adapter` — the extension point documented in
[02 — Public API §5.2](../../../docs/spec/02-public-api.md#52-registering-a-new-data-adapter).

Residency (``INV-RESIDENCY``, 02 §5.2): every adapter materializes RAW, local episodes inside the trust
boundary; the on-disk ``lance``/``hdf5`` files are local participant artifacts and no adapter exposes an
egress / serialize-outbound path. The read-only ``lerobot://`` view never participates in commitment or
egress (RFC-0004 §1).

The built-in adapters self-register at import below, so a bare ``import lensemble.data`` wires the
``fmt`` selector. ``lance``/``h5py`` are pinned runtime deps for the two on-disk backends; ``lerobot``
is an optional extra (imported lazily inside its adapter).
"""

from __future__ import annotations

from lensemble.data.adapters.hdf5_backend import load_hdf5, save_hdf5
from lensemble.data.adapters.lance_backend import load_lance, save_lance
from lensemble.data.adapters.lerobot_adapter import (
    _validate_episode_conformance,
    load_lerobot,
)
from lensemble.data.adapters.lerobot_h5_backend import load_lerobot_h5
from lensemble.data.adapters.registry import (
    load_episodes,
    register_adapter,
    save_episodes,
)

# Built-in adapters (02 §5.2). The lerobot view is read-only → no saver (save_episodes raises for it).
register_adapter("lance", loader=load_lance, saver=save_lance)
register_adapter("hdf5", loader=load_hdf5, saver=save_hdf5)
register_adapter("lerobot", loader=load_lerobot, saver=None)
# Read-only LeRobot-layout single-file HDF5 (real robot datasets as a first-class data_source).
register_adapter("lerobot-h5", loader=load_lerobot_h5, saver=None)

__all__ = [
    "save_episodes",
    "load_episodes",
    "register_adapter",
    "load_lerobot_h5",
    "_validate_episode_conformance",
]
