"""lensemble.config.schema — the structured configuration tree and boundary validation (RFC-0009 2/3).

``LensembleConfig`` is a frozen, fully-typed dataclass composed from Hydra structured-config groups,
validated at load by ``validate_config`` — one of the four boundary-validation points (conventions 6).
Validation never coerces: an out-of-range value is a :class:`~lensemble.errors.ConfigError`
(``CONFIG_INVALID``), not a clamp.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, get_args, get_origin, get_type_hints

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from lensemble.errors import ConfigError, LensembleErrorCode

# Expected encoder latent (ViT hidden) dimension per warm-start variant. These are the standard ViT
# widths; confirm against the pinned V-JEPA 2 release (RFC-0008).
ENCODER_DIM: dict[str, int] = {
    "vjepa2-vit-l": 1024,
    "vjepa2-vit-h": 1280,
    "vjepa2-vit-g": 1408,
}


@dataclass(frozen=True)
class ModelConfig:
    encoder: Literal["vjepa2-vit-l", "vjepa2-vit-h", "vjepa2-vit-g"] = "vjepa2-vit-l"
    warm_start_release: str = "vjepa2-2.0"
    latent_dim: int = 1024  # d
    num_tokens: int = 256  # N
    predictor_depth: int = 12
    predictor_width: int = 1024
    wmcp_version: str = "wmcp-1.0.0"  # gated at federation join (INV-WMCP)
    encoder_frozen: bool = False  # Fork A (RFC-0002): freeze the encoder at warm-start, federate g_phi only


@dataclass(frozen=True)
class ObjectiveConfig:
    lambda_pred: float = 1.0
    lambda_sig: float = 0.1
    lambda_anc: float = 1.0  # the central gauge knob (RFC-0002 7)
    sigreg_sketch_dim: int = 64
    sigreg_knots: int = 17
    anchor_variant: Literal["landmark", "rotational"] = "landmark"


@dataclass(frozen=True)
class GaugeConfig:
    frame_drift_threshold_deg: float = 15.0
    procrustes_singular_floor: float = 1e-6
    anchor_landmark_count: int = 2048  # k >= d


@dataclass(frozen=True)
class FederationConfig:
    participant_count: int = 4  # C
    inner_horizon: int = 50  # H
    num_rounds: int = 100
    outer_lr: float = 0.7
    outer_nesterov_momentum: float = 0.9
    quantize_pseudo_gradient: bool = False
    fault_tolerance_min_participants: int = 3
    # t_agg — the minimum survivors the secure-aggregation reveal needs (RFC-0011); the round quorum is
    # K = max(fault_tolerance_min_participants, secure_agg_threshold) (RFC-0013 §3). Below t_agg the
    # masking sum cannot be unblinded, so a round may not complete.
    secure_agg_threshold: int = 2
    # The per-round COLLECTING wall-time budget (seconds) after which a non-arriving participant is dropped
    # for the round (RFC-0013 §3); loose by design — a liveness/quality knob, not a correctness gate. The
    # in-process transport models the post-timeout present set as the collected set (the network seam #45
    # enforces the wall clock).
    collect_timeout_s: float = 30.0
    transport: Literal["in_process", "network"] = (
        "in_process"  # network => a real trust boundary
    )


@dataclass(frozen=True)
class PrivacyConfig:
    enabled: bool = True
    clip_norm: float = 1.0  # C_clip (INV-DP-BOUND)
    noise_multiplier: float = 1.0  # sigma
    epsilon: float = 8.0
    delta: float = 1e-5
    accountant: Literal["rdp", "prv"] = "rdp"


@dataclass(frozen=True)
class DataConfig:
    format: Literal["lance", "hdf5", "lerobot"] = "lance"
    residency_enforced: bool = (
        True  # INV-RESIDENCY (fail-closed; never disabled in Stage C)
    )
    probe_path: str | None = None
    embodiment_id: str = "default"


@dataclass(frozen=True)
class EvalConfig:
    env_id: str = "stable-worldmodel://pusht"
    planner: Literal["cem", "icem", "mppi"] = "icem"
    planning_samples: int = 512
    horizon: int = 16


@dataclass(frozen=True)
class ObservabilityConfig:
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_path: str = "run.log.jsonl"
    metrics_path: str = "metrics.jsonl"
    tensorboard: bool = False
    wandb: bool = False


@dataclass(frozen=True)
class DeterminismConfig:
    root_seed: int = 0
    deterministic_inner: bool = False
    deterministic_aggregation: bool = (
        True  # INV-AGG-DETERMINISM (always on for federated runs)
    )
    aggregation_dtype: Literal["fp32", "fp64"] = "fp32"


@dataclass(frozen=True)
class LensembleConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    gauge: GaugeConfig = field(default_factory=GaugeConfig)
    federation: FederationConfig = field(default_factory=FederationConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    determinism: DeterminismConfig = field(default_factory=DeterminismConfig)
    run_mode: Literal["train_local", "coordinator", "participant", "eval"] = (
        "train_local"
    )


def _register() -> None:
    """Register every group + the root under canonical names (RFC-0009 2). Confined to this module.

    Nodes are registered as plain dicts (``asdict`` of the dataclass defaults), not the dataclasses
    themselves, because the installed OmegaConf rejects ``typing.Literal`` annotations in structured
    configs. The typed schema and validation live in :func:`load_config` / :func:`validate_config`.
    """
    cs = ConfigStore.instance()
    cs.store(name="default", node=asdict(LensembleConfig()))
    for group, node in (
        ("model", ModelConfig),
        ("objective", ObjectiveConfig),
        ("gauge", GaugeConfig),
        ("federation", FederationConfig),
        ("privacy", PrivacyConfig),
        ("data", DataConfig),
        ("eval", EvalConfig),
        ("observability", ObservabilityConfig),
        ("determinism", DeterminismConfig),
    ):
        cs.store(group=group, name="default", node=asdict(node()))


_register()


def _fail(key: str, value: object, expected: str, remediation: str) -> ConfigError:
    err = ConfigError(
        f"invalid config: {key}={value!r} ({expected})",
        code=LensembleErrorCode.CONFIG_INVALID,
        remediation=remediation,
    )
    err.key = key  # type: ignore[attr-defined]
    err.value = value  # type: ignore[attr-defined]
    err.expected = expected  # type: ignore[attr-defined]
    return err


_GROUPS = (
    "model",
    "objective",
    "gauge",
    "federation",
    "privacy",
    "data",
    "eval",
    "observability",
    "determinism",
)


def _validate_literals(cfg: LensembleConfig) -> None:
    """Enforce ``Literal`` field membership (OmegaConf does not type-check ``Literal``)."""
    nodes: list[tuple[str, object]] = [("", cfg)]
    for g in _GROUPS:
        nodes.append((g, getattr(cfg, g)))
    for prefix, node in nodes:
        hints = get_type_hints(type(node))
        for name, ann in hints.items():
            if get_origin(ann) is Literal:
                allowed = get_args(ann)
                val = getattr(node, name)
                if val not in allowed:
                    key = f"{prefix}.{name}" if prefix else name
                    raise _fail(
                        key, val, f"one of {list(allowed)}", "choose an allowed value"
                    )


def _coerce(key: str, ann: object, val: object) -> object:
    """Validate (and minimally coerce) a primitive value against its field annotation.

    OmegaConf over a plain dict does not type-check primitives, so per-field types are enforced here:
    an out-of-type value is a :class:`~lensemble.errors.ConfigError`, never a silent coercion (except
    the standard ``int -> float`` widening).
    """
    origin = get_origin(ann)
    if origin is Literal:
        return val  # membership is checked by _validate_literals
    if ann is bool:
        if not isinstance(val, bool):
            raise _fail(key, val, "a bool", "set a true/false value")
        return val
    if ann is int:
        if isinstance(val, bool) or not isinstance(val, int):
            raise _fail(key, val, "an int", "set an integer value")
        return val
    if ann is float:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise _fail(key, val, "a float", "set a numeric value")
        return float(val)
    if ann is str:
        if not isinstance(val, str):
            raise _fail(key, val, "a str", "set a string value")
        return val
    args = get_args(ann)
    if origin is None and args == () and ann is type(None):  # pragma: no cover
        return val
    if args and type(None) in args:  # Optional[T] (e.g. str | None)
        if val is None:
            return None
        (inner,) = [a for a in args if a is not type(None)]
        return _coerce(key, inner, val)
    return val  # pragma: no cover - defensive


def _build(container: dict) -> LensembleConfig:
    """Reconstruct the frozen dataclass tree from a resolved container, type-checking each field."""
    groups = {
        "model": ModelConfig,
        "objective": ObjectiveConfig,
        "gauge": GaugeConfig,
        "federation": FederationConfig,
        "privacy": PrivacyConfig,
        "data": DataConfig,
        "eval": EvalConfig,
        "observability": ObservabilityConfig,
        "determinism": DeterminismConfig,
    }
    kwargs: dict[str, object] = {}
    for name, cls in groups.items():
        hints = get_type_hints(cls)
        data = container[name]
        validated = {f: _coerce(f"{name}.{f}", hints[f], data[f]) for f in hints}
        kwargs[name] = cls(**validated)
    run_mode = _coerce(
        "run_mode", get_type_hints(LensembleConfig)["run_mode"], container["run_mode"]
    )
    return LensembleConfig(run_mode=run_mode, **kwargs)  # type: ignore[arg-type]


def validate_config(cfg: LensembleConfig) -> None:
    """Enforce the RFC-0009 3 cross-field rules at the configuration boundary.

    Raises :class:`~lensemble.errors.ConfigError` (``CONFIG_INVALID``) carrying ``key``/``value``/
    ``expected`` and a non-empty ``remediation`` on the first violation. Never coerces.
    """
    _validate_literals(cfg)
    m, o, g, fed = cfg.model, cfg.objective, cfg.gauge, cfg.federation
    p, d, det = cfg.privacy, cfg.data, cfg.determinism

    # Latent dimension positive and consistent with the warm-start release's emitted dimension.
    if m.latent_dim <= 0 or m.num_tokens <= 0:
        raise _fail(
            "model.latent_dim",
            m.latent_dim,
            "latent_dim > 0 and num_tokens > 0",
            "set positive latent_dim and num_tokens",
        )
    expected_dim = ENCODER_DIM.get(m.encoder)
    if expected_dim is not None and m.latent_dim != expected_dim:
        raise _fail(
            "model.latent_dim",
            m.latent_dim,
            f"== {expected_dim} for encoder {m.encoder}",
            "pin a matching V-JEPA 2 release or set latent_dim to the release's emitted dimension",
        )
    if o.sigreg_sketch_dim > m.latent_dim:
        raise _fail(
            "objective.sigreg_sketch_dim",
            o.sigreg_sketch_dim,
            f"<= latent_dim ({m.latent_dim})",
            "the SIGReg sketch dimension may not exceed d",
        )

    # Landmark coverage: k >= d.
    if g.anchor_landmark_count < m.latent_dim:
        raise _fail(
            "gauge.anchor_landmark_count",
            g.anchor_landmark_count,
            f">= latent_dim ({m.latent_dim})",
            "the anchor needs k>=d landmarks to pin the frame; raise anchor_landmark_count",
        )

    # Fault-tolerance floor: 0 < min <= C.
    if not (0 < fed.fault_tolerance_min_participants <= fed.participant_count):
        raise _fail(
            "federation.fault_tolerance_min_participants",
            fed.fault_tolerance_min_participants,
            f"in (0, participant_count={fed.participant_count}]",
            "min participants must be in (0, C]",
        )

    # Secure-aggregation reveal threshold t_agg: 0 < t_agg <= C (RFC-0011/RFC-0013 §3). The round quorum
    # K = max(min_participants, t_agg) is then in (0, C], so a federation of C members can always reach it.
    if not (0 < fed.secure_agg_threshold <= fed.participant_count):
        raise _fail(
            "federation.secure_agg_threshold",
            fed.secure_agg_threshold,
            f"in (0, participant_count={fed.participant_count}]",
            "the secure-aggregation reveal threshold t_agg must be in (0, C]",
        )

    # COLLECTING wall-time budget strictly positive (a non-positive timeout would drop everyone, RFC-0013 §3).
    if not (fed.collect_timeout_s > 0):
        raise _fail(
            "federation.collect_timeout_s",
            fed.collect_timeout_s,
            "> 0",
            "the per-round COLLECTING timeout must be a positive wall-time budget",
        )

    # DP budget well-formed.
    if p.enabled and not (
        p.clip_norm > 0
        and p.noise_multiplier >= 0
        and 0 < p.delta < 1
        and p.epsilon > 0
    ):
        raise _fail(
            "privacy",
            (p.clip_norm, p.noise_multiplier, p.epsilon, p.delta),
            "clip_norm>0, noise_multiplier>=0, 0<delta<1, epsilon>0",
            "DP budget malformed; see RFC-0012",
        )

    # Aggregation determinism for federated runs (INV-AGG-DETERMINISM).
    if (
        cfg.run_mode in {"coordinator", "participant"}
        and not det.deterministic_aggregation
    ):
        raise _fail(
            "determinism.deterministic_aggregation",
            det.deterministic_aggregation,
            "True for federated runs",
            "federated runs require deterministic_aggregation=true (INV-AGG-DETERMINISM)",
        )

    # Residency across a network boundary (INV-RESIDENCY).
    if fed.transport == "network" and not d.residency_enforced:
        raise _fail(
            "data.residency_enforced",
            d.residency_enforced,
            "True when transport == 'network'",
            "residency enforcement may not be disabled across a real trust boundary (INV-RESIDENCY)",
        )

    # Variant/SVD coherence (Variant B needs a singular-value floor).
    if o.anchor_variant == "rotational" and not (g.procrustes_singular_floor > 0):
        raise _fail(
            "gauge.procrustes_singular_floor",
            g.procrustes_singular_floor,
            "> 0 for Variant B",
            "Variant B needs a singular-value floor to guard the SVD (RFC-0002)",
        )

    # Probe presence for anchored federated/eval runs.
    if (
        cfg.run_mode in {"coordinator", "participant", "eval"}
        and o.lambda_anc > 0
        and d.probe_path is None
    ):
        raise _fail(
            "data.probe_path",
            d.probe_path,
            "a pinned probe path for anchored runs",
            "anchored federation requires a pinned public probe (RFC-0004)",
        )


def load_config(
    config_name: str = "default",
    overrides: list[str] | None = None,
    *,
    config_dir: Path | None = None,
) -> LensembleConfig:
    """Compose a frozen ``LensembleConfig`` from Hydra groups + ``key=value`` overrides (RFC-0009 3).

    Precedence (lowest to highest): structured-config defaults < the named config file (``config_name``)
    < group selections < ``key=value`` overrides. ``struct`` mode rejects unknown keys. Raises
    :class:`~lensemble.errors.ConfigError` on a composition failure, a type/range violation, or a
    cross-field inconsistency (see :func:`validate_config`).
    """
    overrides = list(overrides or [])
    try:
        # OmegaConf.create over the dataclass defaults infers and type-checks primitive types and, in
        # struct mode, rejects unknown keys. (We do not use OmegaConf.structured/Hydra compose here:
        # this OmegaConf version does not support typing.Literal in structured configs; Literal
        # membership is validated by validate_config. ConfigStore registration is kept for the record.)
        base = OmegaConf.create(asdict(LensembleConfig()))
        OmegaConf.set_struct(base, True)
        if config_dir is not None:
            file_path = Path(config_dir) / f"{config_name}.yaml"
            if file_path.exists():
                base = OmegaConf.merge(base, OmegaConf.load(str(file_path)))
        merged = (
            OmegaConf.merge(base, OmegaConf.from_dotlist(overrides))
            if overrides
            else base
        )
        container = OmegaConf.to_container(merged, resolve=True)
    except (
        Exception
    ) as exc:  # OmegaConf composition or primitive-type error -> typed ConfigError
        raise ConfigError(
            f"config composition failed: {exc}",
            code=LensembleErrorCode.CONFIG_INVALID,
            remediation="check the override keys/types against the LensembleConfig schema (struct mode is on)",
        ) from exc
    obj = _build(container)  # type: ignore[arg-type]
    validate_config(obj)
    return obj


# conventions 5 names this `load`; RFC-0009 3 names it `load_config`. Keep both.
load = load_config
