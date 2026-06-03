"""lensemble.data.adapters.lerobot_adapter — the read-only ``lerobot://<repo_id>`` view (RFC-0004 §1).

A ``lerobot://<repo_id>`` source resolves a LeRobot-Hub dataset to a READ-ONLY
:class:`~lensemble.data.dataset.EpisodeDataset` view — "train directly on LeRobot-Hub robot datasets
without re-ingest" ([conventions §11](../../../docs/spec/conventions.md#11-external-dependencies)). The
``lerobot`` library is an OPTIONAL extra: it is imported lazily, and an absent library raises a clear
error naming the remediation. The view is read-only by construction, so it never participates in a
dataset commitment or an egress (RFC-0004 §1) — this adapter registers no saver.

**On-load conformance** is the testable core (RFC-0007 §4): every resolved episode is validated against
the :class:`~lensemble.data.episode.Episode` schema and the WMCP ``ActionSpec`` before it is exposed —
a mismatched action space (action tensor trailing dim != ``ActionSpec.dim``, an embodiment-id
disagreement, or an invalid kind/``num_classes``) or a latent-incompatible modality raises
:class:`~lensemble.errors.ContractViolation` (code ``WMCP_CONTRACT_VIOLATION``). The check
:func:`_validate_episode_conformance` is factored so it is unit-testable WITHOUT ``lerobot`` installed
(pass a constructed ``Episode`` + an ``ActionSpec`` directly).

Residency (``INV-RESIDENCY``): the resolved episodes are RAW and materialized locally inside the trust
boundary; the read-only view exposes no egress / serialize-outbound path.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from lensemble.contracts import validate_action_spec
from lensemble.errors import ContractViolation, LensembleErrorCode

if TYPE_CHECKING:
    from lensemble.contracts import ActionSpec
    from lensemble.data.dataset import EpisodeDataset
    from lensemble.data.episode import Episode

_SCHEME = "lerobot://"

# Modalities the WMCP latent contract can encode (RFC-0007). A LeRobot record outside this set is
# latent-incompatible and rejected on load (a hard reject, not a silent coercion).
_LATENT_COMPATIBLE_MODALITIES = frozenset({"rgb-video", "rgb-image", "depth-video"})


def _conformance_fail(message: str, remediation: str) -> ContractViolation:
    err = ContractViolation(
        message,
        code=LensembleErrorCode.WMCP_CONTRACT_VIOLATION,
        remediation=remediation,
    )
    err.field = "action_spec"  # type: ignore[attr-defined]
    return err


def _validate_episode_conformance(episode: "Episode", spec: "ActionSpec") -> None:
    """Validate a resolved episode against the ``Episode`` schema + the WMCP ``ActionSpec`` (RFC-0007 §4).

    Checks, in order: the ``ActionSpec`` is itself valid (:func:`validate_action_spec`); the episode's
    declared ``embodiment_id`` matches the spec's; the modality is latent-compatible; and every
    transition's ``action_t`` trailing dim equals ``ActionSpec.dim`` (a discrete spec also requires each
    action index in ``[0, num_classes[j])``). Raises :class:`~lensemble.errors.ContractViolation`
    (``WMCP_CONTRACT_VIOLATION``) on the first failing clause — a hard reject on the load path, never a
    coercion; no-op return on success. Pure: no I/O, no mutation, no ``lerobot`` import (so it is
    unit-testable without the optional extra).
    """
    # 1. The spec must itself be a valid WMCP ActionSpec (raises ContractViolation on any rule).
    validate_action_spec(spec)

    # 2. The episode's embodiment must be the one the spec describes (else the action head is wrong).
    if episode.embodiment_id != spec.embodiment_id:
        raise _conformance_fail(
            f"episode {episode.episode_id!r} declares embodiment_id "
            f"{episode.embodiment_id!r} but its ActionSpec describes {spec.embodiment_id!r}",
            "resolve the lerobot repo against the ActionSpec for its embodiment",
        )

    # 3. The modality must be one the WMCP latent contract can encode (RFC-0007).
    if episode.modality not in _LATENT_COMPATIBLE_MODALITIES:
        raise _conformance_fail(
            f"episode {episode.episode_id!r} modality {episode.modality!r} is "
            f"latent-incompatible; expected one of {sorted(_LATENT_COMPATIBLE_MODALITIES)}",
            "convert the LeRobot record to a latent-compatible modality before training",
        )

    # 4. Every transition's action vector must match the ActionSpec dimensionality (and discrete range).
    for idx, t in enumerate(episode.transitions):
        if t.action_t.ndim == 0 or int(t.action_t.shape[-1]) != spec.dim:
            got = tuple(int(s) for s in t.action_t.shape)
            raise _conformance_fail(
                f"episode {episode.episode_id!r} transition {idx}: action trailing dim "
                f"!= ActionSpec.dim ({spec.dim}); got action shape {got}",
                "the LeRobot action space does not match the declared ActionSpec.dim",
            )
        if spec.num_classes is not None:
            flat = t.action_t.reshape(-1, spec.dim)
            for j, n in enumerate(spec.num_classes):
                col = flat[:, j]
                if bool((col < 0).any()) or bool((col >= n).any()):
                    raise _conformance_fail(
                        f"episode {episode.episode_id!r} transition {idx}: discrete action "
                        f"index out of range for dim {j} (num_classes={n})",
                        "a discrete action index must lie in [0, num_classes[j])",
                    )


def load_lerobot(source: str | Path) -> "EpisodeDataset":
    """Resolve a ``lerobot://<repo_id>`` source to a read-only ``EpisodeDataset`` (RFC-0004 §1).

    Imports ``lerobot`` lazily; on :class:`ImportError` raises a clear
    :class:`~lensemble.errors.ContractViolation` naming the optional ``lerobot`` extra. Every resolved
    episode is validated by :func:`_validate_episode_conformance` before the view is returned (a hard
    reject on a mismatched action space or latent-incompatible modality). The view is read-only:
    ``exportable`` stays ``False`` and no saver is registered.
    """
    repo_id = str(source)
    if repo_id.startswith(_SCHEME):
        repo_id = repo_id[len(_SCHEME) :]

    if not _lerobot_available():
        raise ContractViolation(
            f"cannot resolve lerobot://{repo_id!r}: the optional 'lerobot' library is not importable",
            code=LensembleErrorCode.WMCP_CONTRACT_VIOLATION,
            remediation="install the optional 'lerobot' extra (pip install lensemble[lerobot]) "
            "to resolve LeRobot-Hub datasets",
        )

    # Reached only with the optional lerobot library installed, which the CI suite deliberately does
    # not vendor (the network/Hub path is the single documented coverage gap for the adapter, #22).
    return _resolve_lerobot_view(repo_id)  # pragma: no cover


def _lerobot_available() -> bool:
    """Whether the optional ``lerobot`` library is importable (factored so the absent path is testable)."""
    try:
        import lerobot  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True  # pragma: no cover - the CI suite runs without the optional lerobot extra (#22)


def _resolve_lerobot_view(repo_id: str) -> "EpisodeDataset":  # pragma: no cover
    """Build the read-only ``EpisodeDataset`` view from a resolved LeRobot ``repo_id`` (RFC-0004 §1).

    Reached only with the optional ``lerobot`` library installed (not vendored in CI, #22). Every
    resolved episode is conformance-checked before the view is exposed; the view is read-only
    (``exportable`` stays ``False``).
    """
    from lensemble.data.dataset import EpisodeDataset

    episodes = _resolve_lerobot_episodes(repo_id)
    for episode in episodes:
        _validate_episode_conformance(episode, episode.action_spec)
    return EpisodeDataset(episodes, path=None, fmt="lerobot", exportable=False)


def _resolve_lerobot_episodes(repo_id: str) -> "list[Episode]":  # pragma: no cover
    """Materialize a LeRobot-Hub ``repo_id`` into local ``Episode``s (the network/Hub path).

    Reached only with the optional ``lerobot`` library installed; the CI suite does not vendor it, so
    this body is the single documented coverage gap for the adapter (#22). The episodes it returns are
    RAW and local (``INV-RESIDENCY``); the read-only view never re-exports them.
    """
    raise ContractViolation(
        f"lerobot repo {repo_id!r} resolution is not wired to a LeRobot reader yet",
        code=LensembleErrorCode.WMCP_CONTRACT_VIOLATION,
        remediation="wire the lerobot dataset reader (issue #22 follow-up)",
    )
