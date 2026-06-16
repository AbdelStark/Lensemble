#!/usr/bin/env python3
"""Spike #335 harness: federated training of the world model itself, end to end.

Runs real small-scale CPU experiments through the SAME stack the GPU launchers use
(``lensemble.federation.Coordinator`` / ``Participant`` DiLoCo path, ``lensemble.model.Objective``,
``lensemble.eval.state_probe_r2``) on the in-repo dynamic swipe-dot env. The goal is decision
evidence, not a published checkpoint: does federated full-model training reach local-only parity on
the binding ground-truth probe, does it hold magnitude (not just rank), and which knobs move it.

Deterministic and CPU-only. Writes results/spike_results.json and results/browser_feasibility.json.

  uv run python docs/spikes/0001-federated-world-model-training/run_spike.py
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from lensemble.config.schema import (
    DataConfig,
    FederationConfig,
    GaugeConfig,
    LensembleConfig,
    ModelConfig,
    ObjectiveConfig,
    PrivacyConfig,
)
from lensemble.data import load_episodes
from lensemble.data.probe import PublicProbe, probe_content_hash, save_probe
from lensemble.eval import state_probe_r2
from lensemble.eval.jepa_metrics import effective_rank
from lensemble.federation import (
    Coordinator,
    InProcessTransport,
    Participant,
    RoundState,
)
from lensemble.gauge import FrameAnchor
from lensemble.model import build_encoder
from lensemble.model.action_head import build_action_head
from lensemble.model.objective import Objective
from lensemble.model.predictor import build_predictor

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# --- model sized like the proven CPU gate (latent128/depth4) so the local-only baseline reliably
# --- clears state_probe_r2 >= 0.5 and the federated-vs-local comparison is trustworthy -----------
LATENT_DIM = 128
DEPTH = 4
PRED_DEPTH = 3
NUM_TOKENS = 9
ANCHOR_LANDMARKS = 128
INNER_HORIZON = 100
ROUNDS = 3
SILOS = 3
HELDOUT_SEED = 99

# Non-IID silos: distinct seeds (different trajectories / state coverage) with SHARED dynamics
# (same step_scale). This keeps a single global predictor able to fit all silos, so the pooled
# central run is a valid no-aggregation-penalty upper bound, and isolates the federation/gauge
# effect from a dynamics-mismatch confound.
SILO_SOURCES = [
    "synthetic-dynamic://swipe-dot?seed=11&n_episodes=40&steps=16&image_size=48&step_scale=0.7&sigma=0.06",
    "synthetic-dynamic://swipe-dot?seed=22&n_episodes=40&steps=16&image_size=48&step_scale=0.7&sigma=0.06",
    "synthetic-dynamic://swipe-dot?seed=33&n_episodes=40&steps=16&image_size=48&step_scale=0.7&sigma=0.06",
]
SEEDS = [0, 1, 2]
HELDOUT_SOURCE = (
    f"synthetic-dynamic://swipe-dot?seed={HELDOUT_SEED}&n_episodes=8&steps=16"
    "&image_size=48&step_scale=0.7&sigma=0.06"
)
WINDOW_STEPS = 8


def make_cfg(
    *,
    source: str | None,
    run_mode: str,
    probe_path: Path | None = None,
    lambda_sig: float = 0.3,
    lambda_anc: float = 0.05,
    target_stop_gradient: bool = False,
    encoder_frozen: bool = False,
    participant_count: int = SILOS,
    inner_horizon: int = INNER_HORIZON,
    num_rounds: int = ROUNDS,
    outer_lr: float = 1.0,
) -> LensembleConfig:
    base = LensembleConfig()
    return dataclasses.replace(
        base,
        model=ModelConfig(
            encoder="scratch",
            latent_dim=LATENT_DIM,
            num_tokens=NUM_TOKENS,
            predictor_depth=PRED_DEPTH,
            predictor_width=LATENT_DIM,
            num_frames=1,
            tubelet=1,
            image_size=48,
            patch_size=16,
            depth=DEPTH,
            num_heads=4,
            in_channels=3,
            mlp_ratio=2.0,
            encoder_frozen=encoder_frozen,
            wmcp_version=base.model.wmcp_version,
        ),
        objective=ObjectiveConfig(
            lambda_pred=1.0,
            lambda_sig=lambda_sig,
            lambda_anc=lambda_anc,
            target_stop_gradient=target_stop_gradient,
            sigreg_sketch_dim=32,
            sigreg_knots=9,
        ),
        gauge=GaugeConfig(
            # high threshold: the backstop MEASURES drift every round but never aborts the spike run
            frame_drift_threshold_deg=175.0,
            anchor_landmark_count=ANCHOR_LANDMARKS,
        ),
        federation=FederationConfig(
            participant_count=participant_count,
            inner_horizon=inner_horizon,
            inner_lr=1e-3,
            num_rounds=num_rounds,
            outer_lr=outer_lr,
            outer_nesterov_momentum=0.0,
            fault_tolerance_min_participants=participant_count,
            secure_agg_threshold=min(2, participant_count),
            collect_timeout_s=5.0,
            aggregation_backend="simulated",
        ),
        privacy=PrivacyConfig(
            enabled=False, clip_norm=10.0, noise_multiplier=0.0, epsilon=8.0, delta=1e-5
        ),
        data=DataConfig(
            format="synthetic-dynamic",
            probe_path=None if probe_path is None else str(probe_path),
            data_source=source,
            window_steps=WINDOW_STEPS,
        ),
        determinism=dataclasses.replace(base.determinism, root_seed=123),
        run_mode=run_mode,  # type: ignore[arg-type]
    )


# --- metrics --------------------------------------------------------------------------------------


def heldout_windows() -> list:
    return list(load_episodes(HELDOUT_SOURCE).windows(WINDOW_STEPS))


def encode_latents(encoder: nn.Module, windows: list) -> tuple[Tensor, Tensor]:
    latents, states = [], []
    encoder.eval()
    with torch.no_grad():
        for w in windows:
            assert w.state is not None
            latents.append(encoder(w.obs).tokens.float())
            states.append(w.state.float())
    encoder.train()
    return torch.cat(latents, dim=0), torch.cat(states, dim=0)


def magnitude_metrics(latents: Tensor) -> dict[str, float]:
    """Absolute held-out magnitude metrics (the #259 blind spot effective_rank cannot see).

    latent_std_mean: mean over dims of per-dim std (centered) on held-out latents.
    latent_rms: sqrt(mean(latent^2)), the absolute scale.
    A magnitude-collapsed encoder has latent_std_mean -> 0 while effective_rank can stay high.
    """
    flat = latents.reshape(-1, latents.shape[-1])
    per_dim_std = flat.std(dim=0, unbiased=False)
    return {
        "latent_std_mean": float(per_dim_std.mean()),
        "latent_rms": float(flat.pow(2).mean().sqrt()),
    }


def grounded_metrics(encoder: nn.Module, windows: list) -> dict[str, float]:
    x, y = encode_latents(encoder, windows)
    split = int(0.7 * x.shape[0])
    r2 = state_probe_r2(x[:split], y[:split], x[split:], y[split:])
    rank = float(effective_rank(x.reshape(-1, x.shape[-1])))
    out = {"state_probe_r2": float(r2), "effective_rank": rank}
    out.update(magnitude_metrics(x))
    return out


def write_probe(path: Path, cfg: LensembleConfig, source: str) -> None:
    ds = load_episodes(source)
    train_windows = list(ds.windows(WINDOW_STEPS))
    points = torch.cat(
        [w.obs[:1] for w in train_windows[: ANCHOR_LANDMARKS * 2]], dim=0
    )
    landmark_idx = torch.arange(ANCHOR_LANDMARKS)
    encoder = build_encoder(cfg)
    with torch.no_grad():
        targets = encoder(points[landmark_idx]).tokens.detach()
    save_probe(
        PublicProbe(
            points=points,
            landmark_idx=landmark_idx,
            landmark_targets=targets,
            content_hash=probe_content_hash(points, landmark_idx),
            probe_version=1,
        ),
        path,
    )


def build_anchor(encoder: nn.Module, source: str):
    ds = load_episodes(source)
    train_windows = list(ds.windows(WINDOW_STEPS))
    points = torch.cat(
        [w.obs[:1] for w in train_windows[: ANCHOR_LANDMARKS * 2]], dim=0
    )
    landmark_idx = torch.arange(ANCHOR_LANDMARKS)
    with torch.no_grad():
        targets = encoder(points[landmark_idx]).tokens.detach()
    probe = PublicProbe(
        points=points,
        landmark_idx=landmark_idx,
        landmark_targets=targets,
        content_hash=probe_content_hash(points, landmark_idx),
        probe_version=1,
    )
    return FrameAnchor(
        probe, targets, variant="landmark", probe_hash=probe.content_hash.hex()
    ).loss


# --- central training (local-only baseline + pooled upper bound) ----------------------------------


def train_central(
    *,
    sources: list[str],
    steps: int,
    lambda_sig: float,
    lambda_anc: float,
    target_stop_gradient: bool,
    anchored: bool,
    shuffle: bool = False,
    seed: int = 0,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    torch.manual_seed(seed)
    windows: list = []
    for s in sources:
        windows.extend(list(load_episodes(s).windows(WINDOW_STEPS)))
    if shuffle:
        # mix the silos so the pooled upper bound actually sees all of them every few steps
        order = torch.randperm(len(windows), generator=torch.Generator().manual_seed(5))
        windows = [windows[i] for i in order.tolist()]
    cfg = make_cfg(source=sources[0], run_mode="participant", lambda_sig=lambda_sig)
    ds0 = load_episodes(sources[0])
    encoder = build_encoder(cfg)
    predictor = build_predictor(cfg)
    action_head = build_action_head(cfg, ds0.episodes[0].action_spec)
    anchor = build_anchor(encoder, sources[0]) if anchored else None
    objective = Objective(
        lambda_pred=1.0,
        lambda_sig=lambda_sig,
        lambda_anc=lambda_anc if anchored else 0.0,
        sketch_seed=0,
        sketch_dim=32,
        ep_knots=9,
        anchor=anchor,
        target_stop_gradient=target_stop_gradient,
    )
    opt = torch.optim.AdamW(
        list(encoder.parameters())
        + list(predictor.parameters())
        + list(action_head.parameters()),
        lr=1e-3,
    )
    for step in range(steps):
        w = windows[step % len(windows)]
        loss = objective(encoder, predictor, w, action_head.encode(w.actions))
        opt.zero_grad()
        loss.total.backward()
        opt.step()
    return encoder, predictor, action_head


# --- federated training through the real Coordinator/Participant DiLoCo path -----------------------


def run_federated(
    *,
    tmp: Path,
    label: str,
    lambda_sig: float,
    lambda_anc: float,
    target_stop_gradient: bool,
    encoder_frozen: bool,
    num_rounds: int,
    outer_lr: float,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    probe_path = tmp / f"probe-{label}-{seed}.safetensors"
    coord_cfg = make_cfg(
        source=SILO_SOURCES[0],
        run_mode="coordinator",
        probe_path=probe_path,
        lambda_sig=lambda_sig,
        lambda_anc=lambda_anc,
        target_stop_gradient=target_stop_gradient,
        encoder_frozen=encoder_frozen,
        num_rounds=num_rounds,
        outer_lr=outer_lr,
    )
    write_probe(probe_path, coord_cfg, SILO_SOURCES[0])
    transport = InProcessTransport()
    drift_last = None
    with Coordinator(
        coord_cfg,
        transport=transport,
        artifacts_dir=tmp / f"coord-{label}-{seed}",
        enable_backstop=True,
    ) as coordinator:
        for _round in range(num_rounds):
            global_state = coordinator.global_state()
            for idx, source in enumerate(SILO_SOURCES):
                p_cfg = dataclasses.replace(
                    coord_cfg,
                    data=dataclasses.replace(coord_cfg.data, data_source=source),
                    run_mode="participant",
                )
                participant = Participant(
                    p_cfg, participant_id=f"p{idx}", transport=transport
                )
                update = participant.local_round(
                    global_state, round_seed=global_state.sketch_seed
                )
                transport.submit_update(
                    participant_id=f"p{idx}",
                    round_index=global_state.round_index,
                    update=update,
                )
            state = coordinator.try_round()
            assert state is RoundState.CLOSED, state
        report = coordinator.frame_drift_report()
        if report is not None:
            drift_last = max(p.rotation_angle_deg for p in report.pairs)
        final = coordinator.global_state()
        aggregate = build_encoder(coord_cfg)
        aggregate.load_state_dict(transport.fetch_params(final.theta_ref), strict=True)
    metrics = grounded_metrics(aggregate, heldout_windows())
    metrics["max_frame_drift_deg"] = None if drift_last is None else float(drift_last)
    return metrics


# --- controls -------------------------------------------------------------------------------------


def random_encoder_metrics() -> dict:
    cfg = make_cfg(source=SILO_SOURCES[0], run_mode="participant")
    torch.manual_seed(1)
    enc = build_encoder(cfg)
    return grounded_metrics(enc, heldout_windows())


def collapsed_encoder_metrics() -> dict:
    """A deliberately magnitude-collapsed encoder: rank stays high, magnitude ~0, probe ~0.

    Demonstrates the #259 blind spot directly with real numbers.
    """
    windows = heldout_windows()
    cfg = make_cfg(source=SILO_SOURCES[0], run_mode="participant")
    enc = build_encoder(cfg)
    x, y = encode_latents(enc, windows)
    collapsed = 1e-6 * torch.randn(x.shape, generator=torch.Generator().manual_seed(23))
    split = int(0.7 * collapsed.shape[0])
    r2 = state_probe_r2(collapsed[:split], y[:split], collapsed[split:], y[split:])
    rank = float(effective_rank(collapsed.reshape(-1, collapsed.shape[-1])))
    out = {
        "state_probe_r2": float(r2),
        "effective_rank": rank,
        "max_frame_drift_deg": None,
    }
    out.update(magnitude_metrics(collapsed))
    return out


# --- main ----------------------------------------------------------------------------------------


def _dist(values: list[float]) -> dict[str, float]:
    t = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)),
        "min": float(t.min()),
        "max": float(t.max()),
        "n": len(values),
    }


def _aggregate(per_seed: list[dict]) -> dict:
    keys = [k for k, v in per_seed[0].items() if isinstance(v, (int, float))]
    return {
        k: _dist([float(r[k]) for r in per_seed if r.get(k) is not None]) for k in keys
    }


def main() -> None:
    import shutil

    RESULTS.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS / "_artifacts"
    shutil.rmtree(
        tmp, ignore_errors=True
    )  # the coordinator requires a fresh artifacts dir per run
    tmp.mkdir(parents=True, exist_ok=True)
    started = time.time()
    local_steps = INNER_HORIZON * ROUNDS
    pooled_steps = SILOS * INNER_HORIZON * ROUNDS

    # configs as callables of (seed) -> metrics dict
    def local_only(seed: int) -> dict:
        enc, _p, _a = train_central(
            sources=[SILO_SOURCES[0]],
            steps=local_steps,
            lambda_sig=0.3,
            lambda_anc=0.05,
            target_stop_gradient=False,
            anchored=True,
            seed=seed,
        )
        return grounded_metrics(enc, heldout_windows())

    def central_pooled(seed: int) -> dict:
        enc, _p, _a = train_central(
            sources=SILO_SOURCES,
            steps=pooled_steps,
            lambda_sig=0.3,
            lambda_anc=0.05,
            target_stop_gradient=False,
            anchored=True,
            shuffle=True,
            seed=seed,
        )
        return grounded_metrics(enc, heldout_windows())

    fed = lambda **kw: lambda seed: run_federated(tmp=tmp, seed=seed, **kw)  # noqa: E731

    configs: dict[str, object] = {
        "local_only_central": local_only,
        "central_pooled_upperbound": central_pooled,
        "fed_forkB_claimgrade": fed(
            label="forkB",
            lambda_sig=0.3,
            lambda_anc=0.05,
            target_stop_gradient=False,
            encoder_frozen=False,
            num_rounds=ROUNDS,
            outer_lr=1.0,
        ),
        "fed_forkB_stopgrad": fed(
            label="forkB_sg",
            lambda_sig=0.3,
            lambda_anc=0.05,
            target_stop_gradient=True,
            encoder_frozen=False,
            num_rounds=ROUNDS,
            outer_lr=1.0,
        ),
        "fed_forkB_anchor_strong": fed(
            label="forkB_anc",
            lambda_sig=0.3,
            lambda_anc=0.5,
            target_stop_gradient=False,
            encoder_frozen=False,
            num_rounds=ROUNDS,
            outer_lr=1.0,
        ),
        "fed_forkB_low_outer_lr": fed(
            label="forkB_lr",
            lambda_sig=0.3,
            lambda_anc=0.05,
            target_stop_gradient=False,
            encoder_frozen=False,
            num_rounds=ROUNDS,
            outer_lr=0.3,
        ),
        "fed_forkA_frozen_scratch": fed(
            label="forkA",
            lambda_sig=0.3,
            lambda_anc=0.05,
            target_stop_gradient=False,
            encoder_frozen=True,
            num_rounds=ROUNDS,
            outer_lr=1.0,
        ),
    }

    per_seed_runs: dict[str, list[dict]] = {name: [] for name in configs}
    for name, fn in configs.items():
        for seed in SEEDS:
            per_seed_runs[name].append(fn(seed))  # type: ignore[operator]

    agg = {name: _aggregate(rows) for name, rows in per_seed_runs.items()}

    # derived per-seed: margin over local-only and the RFC-0005 gap-recovery fraction rho,
    # matched seed-by-seed so the comparison is paired
    local_r2 = [r["state_probe_r2"] for r in per_seed_runs["local_only_central"]]
    central_r2 = [
        r["state_probe_r2"] for r in per_seed_runs["central_pooled_upperbound"]
    ]
    local_std = [r["latent_std_mean"] for r in per_seed_runs["local_only_central"]]
    derived: dict[str, dict] = {}
    for name, rows in per_seed_runs.items():
        if not name.startswith("fed_"):
            continue
        margins, rhos, passes, mag_ratio = [], [], [], []
        for i, r in enumerate(rows):
            margins.append(r["state_probe_r2"] - local_r2[i])
            denom = central_r2[i] - local_r2[i]
            rhos.append(
                (r["state_probe_r2"] - local_r2[i]) / denom if denom > 1e-6 else None
            )
            passes.append(r["state_probe_r2"] - local_r2[i] >= 0.05)
            mag_ratio.append(
                r["latent_std_mean"] / local_std[i] if local_std[i] > 0 else None
            )
        derived[name] = {
            "state_probe_r2": _dist([r["state_probe_r2"] for r in rows]),
            "margin_over_local_only": _dist(margins),
            "seeds_passing_margin_0_05": f"{sum(passes)}/{len(passes)}",
            "gap_recovery_rho": _dist([x for x in rhos if x is not None])
            if any(x is not None for x in rhos)
            else None,
            "latent_std_mean": _dist([r["latent_std_mean"] for r in rows]),
            "latent_std_ratio_to_local": _dist([x for x in mag_ratio if x is not None])
            if any(x is not None for x in mag_ratio)
            else None,
            "max_frame_drift_deg": _dist(
                [
                    r["max_frame_drift_deg"]
                    for r in rows
                    if r.get("max_frame_drift_deg") is not None
                ]
            )
            if any(r.get("max_frame_drift_deg") is not None for r in rows)
            else None,
        }

    summary = {
        "schema": "spike-335-federated-wm/2",
        "env": "synthetic-dynamic://swipe-dot (in-repo), CPU",
        "seeds": SEEDS,
        "model": {
            "latent_dim": LATENT_DIM,
            "depth": DEPTH,
            "predictor_depth": PRED_DEPTH,
            "num_tokens": NUM_TOKENS,
            "image_size": 48,
        },
        "federation": {
            "silos": SILOS,
            "inner_horizon": INNER_HORIZON,
            "rounds": ROUNDS,
            "dp": "off",
            "non_iid": "distinct seeds, shared dynamics (step_scale=0.7)",
        },
        "binding_gate": "RFC-0017 state_probe_r2 >= 0.5 and >= 0.05 margin over local-only",
        "controls": {
            "random_encoder": random_encoder_metrics(),
            "collapsed_encoder": collapsed_encoder_metrics(),
        },
        "aggregated_metrics": agg,
        "federated_vs_local": derived,
        "per_seed_runs": per_seed_runs,
        "elapsed_s": round(time.time() - started, 1),
    }
    (RESULTS / "spike_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "local_only_r2": agg["local_only_central"]["state_probe_r2"],
                "central_pooled_r2": agg["central_pooled_upperbound"]["state_probe_r2"],
                "federated_vs_local": {
                    k: {
                        "r2_mean": v["state_probe_r2"]["mean"],
                        "margin_mean": v["margin_over_local_only"]["mean"],
                        "passing": v["seeds_passing_margin_0_05"],
                        "std_ratio_to_local_mean": None
                        if v["latent_std_ratio_to_local"] is None
                        else v["latent_std_ratio_to_local"]["mean"],
                        "drift_mean": None
                        if v["max_frame_drift_deg"] is None
                        else v["max_frame_drift_deg"]["mean"],
                    }
                    for k, v in derived.items()
                },
            },
            indent=2,
        )
    )
    print(f"elapsed {summary['elapsed_s']}s -> {RESULTS / 'spike_results.json'}")


if __name__ == "__main__":
    main()
