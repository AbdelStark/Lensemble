"""lensemble.eval.sweeps — the RFC-0005 §7 sweep COMPOSE side (the synthetic non-IID partition; #56).

The eval-side (band L6) pieces the §7 robustness sweeps need that carry NO ``federation`` dependency
(RFC-0001 §3: ``eval`` may not import ``federation``): the synthetic non-IID per-silo partition
(:func:`partition_synthetic_noniid`) and the seeded ``O(C^2)`` drift-pair sampler
(:func:`sample_drift_pairs`). The DRIVERS that run these through the live federated-simulation harness
live one band up in :mod:`lensemble.federation.sweeps`, which depends DOWN onto this module — mirroring the
:mod:`lensemble.eval.ablation` (compose) vs :mod:`lensemble.federation.ablation` (drive) split that #55
established.

**The non-IID severity axis is SYNTHETIC.** The real ``stable-worldmodel`` factors-of-variation give
*controlled, reproducible* heterogeneity (RFC-0005 §7) but the suite is NOT vendored yet (maintainer-gated,
#96), so here the partition shifts each silo's synthetic toy distribution by a per-silo mean offset scaled
by the severity ``s in [0, 1]``: ``s = 0`` draws the SAME distribution for every silo (near-IID; the
inter-silo offset is zero), ``s = 1`` shifts silo ``c``'s mean by ``c * unit`` (strongly non-IID; the
per-silo frames diverge under the naive rung). A ``factor`` other than ``"synthetic"`` is the DOCUMENTED
SEAM for the real factors-of-variation path: it fail-closes with a clear :class:`~lensemble.errors.EvaluationError`
("real factors-of-variation need the vendored stable-worldmodel suite, #96") rather than silently degrading
to the synthetic partition (mirrors :func:`lensemble.eval.world.resolve_env`'s ``stable-worldmodel://``
fail-closed branch).

The ``O(C^2)`` pairwise-drift diagnostic is enumerated by default (the paper's central figure, RFC-0005
Alternatives Considered); at large ``C`` :func:`sample_drift_pairs` deterministically samples a BOUNDED set
of participant pairs instead, and the sampled set is RECORDED (it belongs in the
:class:`~lensemble.config.manifest.RunManifest`) so the figure stays reproducible (RFC-0005 §8: pair
sampling at large ``C`` is seeded and recorded).
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from lensemble.errors import EvaluationError, LensembleErrorCode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lensemble.contracts import ActionSpec
    from lensemble.data.episode import Window

__all__ = [
    "SiloPartition",
    "partition_synthetic_noniid",
    "sample_drift_pairs",
    "SYNTHETIC_FACTOR",
]

# The only supported non-IID partition factor until stable-worldmodel is vendored (#96). Any other value
# is the real factors-of-variation seam and fail-closes (see partition_synthetic_noniid).
SYNTHETIC_FACTOR = "synthetic"


@dataclass(frozen=True)
class SiloPartition:
    """One silo's residency-bound synthetic data + binding metadata (the layer-neutral silo record).

    The COMPOSE side returns this band-L6 dataclass (it names only ``data``/``contracts`` types, L3/L2)
    rather than the band-L7 ``federation.simulation.SiloData``, so the eval band stays below federation
    (RFC-0001 §3). The federation driver maps each :class:`SiloPartition` to a ``SiloData`` before running
    the harness. ``windows`` are the silo's RAW local windows (they never cross the transport,
    ``INV-RESIDENCY``); ``dataset_root`` is the 32-byte Merkle root ``R_c`` the released delta binds to
    (``INV-COMMIT-BINDING``).
    """

    participant_id: str
    windows: tuple["Window", ...]
    action_spec: "ActionSpec"
    dataset_root: bytes


def partition_synthetic_noniid(
    num_silos: int,
    *,
    severity: float,
    seed: int,
    action_spec: "ActionSpec",
    num_windows: int = 4,
    window_steps: int = 1,
    num_frames: int = 2,
    in_channels: int = 3,
    image_size: int = 4,
    action_dim: int = 2,
    factor: str = SYNTHETIC_FACTOR,
) -> list[SiloPartition]:
    """Partition synthetic per-silo toy data by a synthetic factor scaled by ``severity`` (RFC-0005 §7).

    Produces ``num_silos`` :class:`SiloPartition` records whose synthetic distributions are SHIFTED apart
    by the non-IID ``severity`` ``s in [0, 1]``:

    * ``s = 0`` (near-IID): every silo draws the SAME synthetic distribution — identical per-silo windows
      from one shared draw, so the inter-silo mean offset is exactly zero and the frames do not diverge
      from heterogeneity.
    * ``s = 1`` (strongly non-IID): silo ``c``'s observation mean is shifted by ``c * _UNIT_OFFSET`` (a
      per-silo factor index scaled by the severity), so the per-silo distributions are pulled apart and the
      naive frames diverge.

    Interpolating: silo ``c``'s mean offset is ``s * c * _UNIT_OFFSET``. Each silo also gets a distinct
    participant id and a distinct 32-byte ``dataset_root`` (``INV-COMMIT-BINDING``). The draw is FULLY
    DETERMINISTIC given ``(num_silos, severity, seed, shape...)`` so a swept point is reproducible
    (conventions §9). The clip shapes match the encoder config (a ``Window.obs`` is
    ``(window_steps + 1, num_frames, in_channels, image_size, image_size)``; ``actions`` is
    ``(window_steps, action_dim)``).

    The ``factor`` argument is the DEFERRED real-factors seam (#96): only ``"synthetic"`` is supported;
    any other value (e.g. ``"embodiment"`` / a real factor-of-variation name) raises
    :class:`~lensemble.errors.EvaluationError` — the real factors-of-variation partition needs the vendored
    ``stable-worldmodel`` suite (maintainer-gated). Fail-closed: it never silently falls back to the
    synthetic partition.
    """
    if factor != SYNTHETIC_FACTOR:
        raise EvaluationError(
            f"non-IID factor {factor!r} is not supported: real factors-of-variation need the "
            "vendored stable-worldmodel suite (#96)",
            code=LensembleErrorCode.EVALUATION_FAILED,
            remediation="use factor='synthetic', or vendor stable-worldmodel and wire its "
            "factors-of-variation partition (issue #96)",
        )
    if num_silos < 1:
        raise ValueError(f"num_silos must be >= 1, got {num_silos}")
    if not (0.0 <= severity <= 1.0):
        raise ValueError(f"severity must be in [0, 1], got {severity}")

    from lensemble.data.episode import Window

    silos: list[SiloPartition] = []
    for c in range(num_silos):
        # Severity 0 => every silo draws from the SAME generator stream (identical data, near-IID). Severity
        # > 0 => a per-silo seed so the silos hold genuinely DIFFERENT windows (a non-vacuous drift signal),
        # plus a per-silo MEAN offset scaled by severity so the distributions are pulled apart.
        silo_seed = seed if severity == 0.0 else seed + 1009 * (c + 1)
        offset = severity * c * _UNIT_OFFSET
        gen = torch.Generator().manual_seed(silo_seed)
        windows: list[Window] = []
        for _ in range(num_windows):
            obs = (
                torch.randn(
                    window_steps + 1,
                    num_frames,
                    in_channels,
                    image_size,
                    image_size,
                    generator=gen,
                )
                + offset
            )
            actions = torch.randn(window_steps, action_dim, generator=gen)
            windows.append(
                Window(
                    obs=obs,
                    actions=actions,
                    num_steps=window_steps,
                    embodiment_id=action_spec.embodiment_id,
                )
            )
        silos.append(
            SiloPartition(
                participant_id=f"silo-{c}",
                windows=tuple(windows),
                action_spec=action_spec,
                dataset_root=bytes([(c + 1) % 256]) * 32,
            )
        )
    return silos


# The per-silo-index mean-offset unit at full severity. Large enough (relative to the unit-variance toy
# draw) that the per-silo distribution shift is the DOMINANT inter-silo signal — so the naive frame drift
# grows visibly with severity within the tiny inner-loop budget (RFC-0005 §7) — while staying CPU-trivial.
_UNIT_OFFSET = 2.0


def sample_drift_pairs(
    participant_ids: "Sequence[str]",
    *,
    max_pairs: int,
    seed: int,
) -> list[tuple[str, str]]:
    """Deterministically sample a BOUNDED set of participant pairs for the ``O(C^2)`` drift figure (§7).

    Pairwise drift is ``O(C^2)`` per round; full enumeration is the default for the paper's central figure,
    and at large ``C`` this is the documented degrade (RFC-0005 Alternatives Considered): a seeded,
    reproducible sample of at most ``max_pairs`` distinct unordered participant pairs. The sampled set is
    a deterministic function of ``(participant_ids, max_pairs, seed)`` so the figure stays reproducible, and
    it must be RECORDED in the :class:`~lensemble.config.manifest.RunManifest` (RFC-0005 §8: pair sampling
    at large ``C`` is seeded AND recorded) — the returned pairs serialize to a manifest-native list of
    ``[a, b]`` lists (ids only; no tensor, so the residency redaction guard accepts them).

    Caps at the ``C-choose-2`` total when ``max_pairs`` exceeds it (it cannot return more pairs than exist).
    Each returned pair is ordered with ``a < b`` by id, and no unordered pair appears twice. A different
    ``seed`` yields a different sample (the diagnostic samples a fresh subset per recorded run).
    """
    ids = list(participant_ids)
    all_pairs = [
        (a, b) for a, b in itertools.combinations(sorted(set(ids)), 2)
    ]  # sorted+deduped => deterministic enumeration order; each unordered pair once, a < b
    k = min(int(max_pairs), len(all_pairs))
    if k <= 0:
        return []
    rng = random.Random(seed)
    return sorted(rng.sample(all_pairs, k))
