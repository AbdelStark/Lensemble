"""lensemble.eval.harness — latent-MPC evaluation on a held-out env -> ``EvalReport`` (RFC-0005 §3; #52).

:func:`evaluate` is the public entry point ([02-public-api §1.5], RFC-0005 §3): it loads a hash-verified
checkpoint (``INV-CHECKPOINT-HASH``), reconstructs the frozen ``encoder``/``predictor``, resolves the eval
env from ``env_id``, wires the trained world model into the :class:`~lensemble.eval.mpc.Planner` as a
batched latent ``dynamics``, runs a small fixed set of seed-pinned latent-MPC episodes, and assembles an
:class:`~lensemble.eval.report.EvalReport` of scalar metrics, hashes, and counts.

Residency (``INV-RESIDENCY``). The report carries ONLY scalars / hashes / counts — never a raw
observation/action/latent tensor; this is structurally true because :class:`EvalReport` has no tensor
field. Read-only: the loaded checkpoint is never mutated or re-committed.

Determinism. Best-effort, seed-pinned (conventions 9): the episode seeds derive deterministically from
``cfg.determinism.root_seed`` and the planner draws from one seeded generator per episode, so a run is
reproducible on the same device class (not required to be bitwise-identical across hardware — the
aggregation path is the only bitwise surface, ``INV-AGG-DETERMINISM``).

``INV-ACTIONHEAD-LOCAL``. The shared checkpoint carries only ``encoder``/``predictor`` weights; the action
head is constructed fresh and lives in local state. A deployment loads the participant's *trained* local
head from its own local checkpoint — that local-load path is out of scope here.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from lensemble.artifacts import load_checkpoint
from lensemble.config import build_manifest
from lensemble.contracts import LatentState
from lensemble.errors import EvaluationError, LensembleErrorCode
from lensemble.eval.metrics import effective_dim, success_rate
from lensemble.eval.mpc import Planner
from lensemble.eval.report import EVAL_REPORT_SCHEMA_VERSION, EvalReport
from lensemble.eval.world import resolve_env
from lensemble.model import build_action_head, build_encoder, build_predictor

if TYPE_CHECKING:
    from torch import Tensor

    from lensemble.config import LensembleConfig

# A small, fast, fixed episode count for the CPU eval (RFC-0005 Testing Strategy: tiny synthetic).
_NUM_EPISODES = 4


def _split_weights(weights: dict[str, "Tensor"], prefix: str) -> dict[str, "Tensor"]:
    """Select the ``prefix.*`` weights and strip the prefix (``encoder.`` / ``predictor.``)."""
    cut = len(prefix)
    return {name[cut:]: t for name, t in weights.items() if name.startswith(prefix)}


def evaluate(
    checkpoint: Path,
    env_id: str,
    *,
    cfg: "LensembleConfig",
    num_episodes: int = _NUM_EPISODES,
    planner_iters: int = 4,
) -> EvalReport:
    """Run latent-MPC evaluation on a held-out env and return an :class:`EvalReport` (RFC-0005 §3).

    Preconditions: ``checkpoint`` is a hash-verified artifact (``INV-CHECKPOINT-HASH``) whose weights are
    keyed ``encoder.*`` / ``predictor.*``; ``env_id`` resolves to a registered or ``stable-worldmodel://``
    environment. ``cfg.eval`` fixes the planner family / horizon / sample count and
    ``cfg.determinism.root_seed`` pins the episode seeds. ``cfg.model`` must expose the encoder/predictor
    build fields ``build_encoder`` / ``build_predictor`` read (the same V-JEPA shape namespace the CLI
    supplies): the typed ``ModelConfig``->architecture bridge is not in the tree yet (the CLI hand-builds
    it), so pass a config whose ``model`` carries those fields.

    Postconditions: returns an :class:`EvalReport` carrying ``success_rate`` (held-out MPC success
    fraction), ``planning_samples``, ``time_per_action_ms`` (mean planning wall-cost per action),
    ``effective_dim`` (the collapse guard over the per-episode latents), ``probe_accuracy`` (``None`` — no
    probe is wired here), ``checkpoint_hash`` (the artifact ``content_hash``), and ``run_manifest_hash``
    (a deterministic hash over the eval-mode ``RunManifest``, excluding the non-semantic ``created_at``).
    The report carries no tensor (``INV-RESIDENCY``).

    Raises: :class:`~lensemble.errors.CheckpointIntegrityError` on a tampered checkpoint and
    :class:`~lensemble.errors.SchemaVersionMismatch` on a too-new artifact (both propagate from
    ``load_checkpoint``, never caught); :class:`~lensemble.errors.EvaluationError` on an unresolvable
    ``env_id`` or a diverging plan; :class:`~lensemble.errors.ConfigError` on an invalid ``cfg.model`` /
    ``ActionSpec``. Determinism is best-effort and seed-pinned (conventions 9).
    """
    if num_episodes < 2:
        raise EvaluationError(
            f"num_episodes must be at least 2, got {num_episodes}",
            code=LensembleErrorCode.EVALUATION_FAILED,
            remediation="evaluate at least two held-out episodes so effective_dim is defined",
        )
    if planner_iters <= 0:
        raise EvaluationError(
            f"planner_iters must be positive, got {planner_iters}",
            code=LensembleErrorCode.EVALUATION_FAILED,
            remediation="run at least one planner refinement iteration",
        )

    # 1. Hash-verified load (propagates CheckpointIntegrityError / SchemaVersionMismatch).
    weights, header = load_checkpoint(Path(checkpoint))

    # 2. Reconstruct the frozen encoder/predictor; load the prefix-split weights strictly.
    encoder = build_encoder(cfg).eval()
    predictor = build_predictor(cfg).eval()
    encoder.load_state_dict(_split_weights(weights, "encoder."), strict=True)
    predictor.load_state_dict(_split_weights(weights, "predictor."), strict=True)

    # 3. Resolve the env; build a fresh LOCAL action head (INV-ACTIONHEAD-LOCAL).
    world = resolve_env(env_id, cfg=cfg)
    action_head = build_action_head(cfg, world.action_spec).eval()

    # 4. The batched latent dynamics: flattened (num_samples, N*d) <-> (num_samples, N, d) LatentState.
    n_tokens, d = encoder.num_tokens, encoder.d

    def dynamics(latents: "Tensor", actions: "Tensor") -> "Tensor":
        tokens = latents.reshape(latents.shape[0], n_tokens, d)
        state = LatentState(
            tokens=tokens,
            num_tokens=n_tokens,
            dim=d,
            wmcp_version=encoder.wmcp_version,
        )
        cond = action_head.encode(actions)
        nxt = predictor.forward(state, cond)
        return nxt.tokens.reshape(latents.shape[0], n_tokens * d)

    # 5. Seed-pinned episodes: derive the seeds deterministically from root_seed.
    root_seed = int(cfg.determinism.root_seed)
    seeds = [root_seed + i for i in range(num_episodes)]
    action_dim = world.action_spec.dim

    successes: list[bool] = []
    episode_latents: list[Tensor] = []
    per_action_seconds: list[float] = []

    with torch.no_grad():
        goal_clip = world.goal()
        initial_clips = [world.reset(seed) for seed in seeds]
        encoded = encoder(torch.stack([goal_clip, *initial_clips], dim=0)).tokens
        encoded = encoded.reshape(len(seeds) + 1, -1)
        zg = encoded[:1]
        for idx, seed in enumerate(seeds):
            z0 = encoded[idx + 1 : idx + 2]
            episode_latents.append(z0.reshape(-1))

            planner = Planner(
                family=cfg.eval.planner,
                horizon=cfg.eval.horizon,
                num_samples=cfg.eval.planning_samples,
                action_dim=action_dim,
                seed=seed,
                num_iters=planner_iters,
            )
            start = time.perf_counter()
            plan = planner.plan(dynamics, z0, zg)
            per_action_seconds.append((time.perf_counter() - start) / cfg.eval.horizon)

            world.reset(seed)
            for t in range(plan.actions.shape[0]):
                world.step(plan.actions[t].detach().cpu())
            successes.append(bool(world.succeeded()))

    # 6. Metrics (consume lensemble.eval.metrics); probe is unwired here -> None.
    rate = success_rate(successes)
    eff_dim = effective_dim(torch.stack(episode_latents))
    time_per_action_ms = (sum(per_action_seconds) / len(per_action_seconds)) * 1000.0

    # 7. Bind the report to a deterministic eval-mode RunManifest hash (created_at is non-semantic).
    manifest = build_manifest(cfg, run_mode="eval")
    run_manifest_hash = hashlib.sha256(
        manifest.model_dump_json(exclude={"created_at"}).encode()
    ).hexdigest()

    # 8. Assemble (the model_validator enforces the 03 §13.1 ranges).
    return EvalReport(
        schema_version=EVAL_REPORT_SCHEMA_VERSION,
        checkpoint_hash=header.content_hash,
        env_id=env_id,
        planner=cfg.eval.planner,
        success_rate=rate,
        planning_samples=int(cfg.eval.planning_samples),
        time_per_action_ms=time_per_action_ms,
        effective_dim=eff_dim,
        probe_accuracy=None,
        run_manifest_hash=run_manifest_hash,
    )
