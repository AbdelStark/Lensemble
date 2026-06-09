"""Deterministic RFC-0017 swipe-dot data source (RFC-0004 §1, RFC-0017).

``synthetic-dynamic://swipe-dot`` resolves to local, read-only ``EpisodeDataset`` objects whose raw
observations, actions, and true ``(x,y)`` state labels are residency-bound. The adapter has no saver and
returns ``exportable=False``; only scalar probe metrics derived locally from the labels may cross a trust
boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
from lensemble.contracts.conformance import validate_action_spec
from lensemble.data.adapters.lerobot_adapter import _validate_episode_conformance
from lensemble.data.episode import Episode, Transition
from lensemble.errors import ContractViolation, LensembleErrorCode

if TYPE_CHECKING:
    from lensemble.data.dataset import EpisodeDataset

_SCHEME = "synthetic-dynamic"
_ENV_NAME = "swipe-dot"
_ACTION_DIM = 2
_DEFAULT_SEED = 0
_DEFAULT_EPISODES = 8
_DEFAULT_STEPS = 64
_DEFAULT_IMAGE_SIZE = 48
_DEFAULT_STEP_SCALE = 0.12
_DEFAULT_SIGMA = 0.055


def _fail(message: str, remediation: str) -> ContractViolation:
    return ContractViolation(
        message,
        code=LensembleErrorCode.WMCP_CONTRACT_VIOLATION,
        remediation=remediation,
    )


def _single_int(params: dict[str, list[str]], key: str, default: int) -> int:
    raw = params.get(key, [str(default)])
    if len(raw) != 1:
        raise _fail(
            f"synthetic-dynamic URI parameter {key!r} must appear once",
            "pass a single integer query value",
        )
    try:
        value = int(raw[0])
    except ValueError as exc:
        raise _fail(
            f"synthetic-dynamic URI parameter {key!r} must be an integer, got {raw[0]!r}",
            "pass an integer query value",
        ) from exc
    return value


def _single_float(params: dict[str, list[str]], key: str, default: float) -> float:
    raw = params.get(key, [str(default)])
    if len(raw) != 1:
        raise _fail(
            f"synthetic-dynamic URI parameter {key!r} must appear once",
            "pass a single numeric query value",
        )
    try:
        value = float(raw[0])
    except ValueError as exc:
        raise _fail(
            f"synthetic-dynamic URI parameter {key!r} must be numeric, got {raw[0]!r}",
            "pass a numeric query value",
        ) from exc
    return value


def _parse_source(source: str | Path) -> tuple[int, int, int, int, float, float]:
    parsed = urlparse(str(source))
    if parsed.scheme != _SCHEME or parsed.netloc != _ENV_NAME:
        raise _fail(
            f"unsupported synthetic-dynamic source {source!r}",
            "use synthetic-dynamic://swipe-dot?seed=&n_episodes=&steps=&image_size=",
        )
    params = parse_qs(parsed.query, keep_blank_values=False)
    seed = _single_int(params, "seed", _DEFAULT_SEED)
    n_episodes = _single_int(params, "n_episodes", _DEFAULT_EPISODES)
    steps = _single_int(params, "steps", _DEFAULT_STEPS)
    image_size = _single_int(params, "image_size", _DEFAULT_IMAGE_SIZE)
    step_scale = _single_float(params, "step_scale", _DEFAULT_STEP_SCALE)
    sigma = _single_float(params, "sigma", _DEFAULT_SIGMA)
    if (
        n_episodes <= 0
        or steps <= 0
        or image_size <= 0
        or step_scale <= 0
        or sigma <= 0
    ):
        raise _fail(
            "synthetic-dynamic parameters must be positive",
            "set n_episodes, steps, image_size, step_scale, and sigma to positive values",
        )
    return seed, n_episodes, steps, image_size, step_scale, sigma


def _action_spec() -> ActionSpec:
    spec = ActionSpec(
        embodiment_id="swipe-dot-2dof",
        kind=ActionKind.CONTINUOUS,
        dim=_ACTION_DIM,
        low=(-1.0, -1.0),
        high=(1.0, 1.0),
        num_classes=None,
        units=("u", "u"),
        wmcp_version=WMCP_VERSION,
    )
    validate_action_spec(spec)
    return spec


def _render_state(
    state: torch.Tensor, *, image_size: int, sigma: float
) -> torch.Tensor:
    coords = torch.linspace(0.0, 1.0, image_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    dx = xx - state[0].to(torch.float32)
    dy = yy - state[1].to(torch.float32)
    blob = torch.exp(-0.5 * (dx.square() + dy.square()) / (sigma * sigma))
    blob = blob.clamp(0.0, 1.0)
    frame = torch.stack(
        (
            blob,
            (0.35 + 0.65 * blob).clamp(0.0, 1.0),
            (1.0 - 0.55 * blob).clamp(0.0, 1.0),
        ),
        dim=0,
    )
    return frame.unsqueeze(0)  # (1, 3, H, W), single-frame rgb-video clip


def _generate_episodes(
    *,
    seed: int,
    n_episodes: int,
    steps: int,
    image_size: int,
    step_scale: float,
    sigma: float,
) -> list[Episode]:
    np_rng = np.random.default_rng(seed)
    torch_gen = torch.Generator().manual_seed(seed)
    spec = _action_spec()
    episodes: list[Episode] = []
    for ep_idx in range(n_episodes):
        init = torch.from_numpy(np_rng.uniform(0.15, 0.85, size=2).astype("float32"))
        states = [init]
        actions: list[torch.Tensor] = []
        for _ in range(steps):
            action = torch.empty(_ACTION_DIM, dtype=torch.float32).uniform_(
                -1.0, 1.0, generator=torch_gen
            )
            actions.append(action)
            states.append((states[-1] + step_scale * action).clamp(0.0, 1.0))
        clips = [
            _render_state(state, image_size=image_size, sigma=sigma) for state in states
        ]
        transitions = [
            Transition(
                obs_t=clips[i],
                action_t=actions[i],
                obs_tp1=clips[i + 1],
                state_t=states[i],
                state_tp1=states[i + 1],
            )
            for i in range(steps)
        ]
        episode = Episode(
            episode_id=f"swipe-dot-seed{seed}-ep{ep_idx}",
            transitions=transitions,
            embodiment_id=spec.embodiment_id,
            modality="rgb-video",
            action_spec=spec,
            collection_meta={
                "source": "synthetic-dynamic",
                "env_id": "swipe-dot",
                "seed": str(seed),
                "episode_index": str(ep_idx),
                "steps": str(steps),
                "image_size": str(image_size),
                "step_scale": f"{step_scale:.8g}",
                "sigma": f"{sigma:.8g}",
                "state_channel": "resident-xy",
            },
        )
        _validate_episode_conformance(episode, episode.action_spec)
        episodes.append(episode)
    return episodes


def load_synthetic_dynamic(source: str | Path) -> "EpisodeDataset":
    """Resolve ``synthetic-dynamic://swipe-dot`` to a read-only resident episode dataset."""
    seed, n_episodes, steps, image_size, step_scale, sigma = _parse_source(source)
    episodes = _generate_episodes(
        seed=seed,
        n_episodes=n_episodes,
        steps=steps,
        image_size=image_size,
        step_scale=step_scale,
        sigma=sigma,
    )
    from lensemble.data.dataset import EpisodeDataset

    return EpisodeDataset(
        episodes, path=None, fmt="synthetic-dynamic", exportable=False
    )
