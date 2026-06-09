# RFC-0017 - Dynamic Env & Ungameable Ground-Truth Metrics

| | |
|---|---|
| **RFC** | 0017 |
| **Title** | Dynamic Env & Ungameable Ground-Truth Metrics |
| **Slug** | dynamic-env-ungameable-metrics |
| **Status** | Draft |
| **Track** | Standards |
| **Authors** | @AbdelStark |
| **Created** | 2026-06-09 |
| **Target milestone** | Dynamic-env MVP, issue [#273](https://github.com/AbdelStark/Lensemble/issues/273) |
| **Area** | data, eval |
| **Requires** | [RFC-0004](RFC-0004-data-provenance.md), [RFC-0005](RFC-0005-evaluation.md), [RFC-0009](RFC-0009-configuration-reproducibility.md) |
| **Informs** | [RFC-0002](RFC-0002-gauge-and-aggregation.md), [RFC-0010](RFC-0010-artifact-checkpoint-format.md), [RFC-0015](RFC-0015-observability-diagnostics.md) |

## Summary

This RFC defines the lightweight fully-observable dynamic control environment used by the dynamic-env
MVP: a two-dimensional kinematic agent rendered as a Gaussian dot. The environment state is the true
position `p=(x,y) in [0,1]^2`; a continuous 2-DOF action `a in [-1,1]^2` updates it by
`p' = clamp(p + k * a, 0, 1)`. Observations are single-frame `rgb-video` clips of shape
`(1, 3, H, W)` with float32 values in `[0,1]`.

The MVP usefulness claim is bound to an external ground-truth target, not to the encoder's own latent
space. The single binding gate is held-out `state_probe_r2`, a linear regression from resident latents
to the true `(x,y)` state. Closed-loop latent-MPC success is reported against the environment's true
`succeeded()` predicate but is non-binding because the planner still optimizes the gameable latent
goal-energy.

## Motivation

The prior SO-100 MVP trained from scratch on near-static robot video. It converged in the gauge sense but
collapsed on held-out data: the encoder produced a near-constant latent with per-element variance around
`7.5e-6`. That failure also reproduced centrally, so it is a data/task failure rather than a federation
bug. On near-static video, "predict no change" is close to optimal for a JEPA prediction objective.

The existing collapse guards do not catch this failure. `effective_rank` and `effective_dim` normalize
the eigenspectrum by its sum, so they are scale-invariant: a magnitude-collapsed latent can look
geometrically healthy while being useless for a downstream external target. The dynamic environment makes
constant latents wrong, because held-out state changes are visible and action-conditioned.

## Environment Contract

The canonical env id is `kinematic://swipe-dot` for evaluation and
`synthetic-dynamic://swipe-dot?...` for deterministic local data generation.

- State: `p=(x,y)`, float32, resident, in `[0,1]^2`.
- Action: continuous `ActionSpec`, `embodiment_id="swipe-dot-2dof"`, `dim=2`, `low=(-1,-1)`,
  `high=(1,1)`, `units=("u","u")`.
- Dynamics: `p' = clamp(p + k * a, 0, 1)`, with `k` recorded by the source metadata.
- Observation: rendered Gaussian dot, `(1, 3, H, W)` float32 `[0,1]`, modality `rgb-video`.
- Success: true-state distance from `p` to a fixed goal below a tolerance; never reset-seed parity.
- State accessor: eval worlds that expose ground truth implement `state() -> Tensor` returning true
  `(x,y)` inside the trust boundary.

## Tiny Shape

The validated CPU probe shape is the intentionally small from-scratch ViT setting:

| Field | Value |
|---|---:|
| `encoder` | `scratch` |
| `latent_dim` | 128 |
| `image_size` | 48 |
| `patch_size` | 16 |
| `num_frames` | 1 |
| `tubelet` | 1 |
| `depth` | 4 |
| `num_heads` | 4 |

The token arithmetic is fixed by RFC-0009 config validation:

```text
num_tokens = (num_frames // tubelet) * (image_size // patch_size)^2
           = (1 // 1) * (48 // 16)^2
           = 1 * 3^2
           = 9
```

`num_heads` divides `latent_dim` (`128 % 4 == 0`). `sigreg_sketch_dim` must be no larger than
`latent_dim`, and `anchor_landmark_count` must remain at least `latent_dim`.

## Determinism

The data source is deterministic from a URI seed. It must seed a dedicated `torch.Generator` and a
dedicated NumPy `default_rng`, and it must not touch global RNG state. This keeps the synthetic episodes
byte-identical across loads without perturbing encoder initialization or any federated run RNG stream.

The source URI records all parameters needed for reproduction:

```text
synthetic-dynamic://swipe-dot?seed=0&n_episodes=8&steps=64&image_size=48
```

## Residency

Observations, actions, and true `(x,y)` labels are raw resident data under `INV-RESIDENCY`. The
`Transition` and `Window` contracts may carry the optional state tensors, but those tensors never cross a
participant boundary. The data source is read-only and returns `exportable=False`. Boundary payloads may
contain only scalar metrics such as held-out R2, hashes, counts, and redacted observability fields.

## Metrics Hierarchy

The single binding usefulness metric is held-out `state_probe_r2`: a closed-form linear probe from the
encoder latent to true `(x,y)`, using mean-pooled tokens, per-feature standardization, and a fixed
train/held-out split. The MVP gate is a pinned `state_probe_r2 >= 0.5` and a pinned absolute margin over
random-encoder, naive-FedAvg, and local-only controls, including the DP-on run.

Closed-loop latent-MPC `success_rate` is reported against the true environment `succeeded()` predicate
and compared to a random-action chance baseline. It is not a binding gate: the planner optimizes actions
to reduce distance to the encoder's own goal latent, so the objective is still gameable by a degenerate
latent representation.

`skill_vs_identity`, latent goal-energy, `effective_rank`, and `effective_dim` are supporting signals
only. `skill_vs_identity` and latent goal-energy target the encoder's own latent. `effective_rank` and
`effective_dim` are scale-invariant and therefore blind to magnitude collapse.

## Unvalidated Regimes

Two risks remain explicit until the dynamic-env evidence bundle closes them:

- The CPU proof ran with the anchor off (`lambda_anc=0`), while the federated MVP reuses the anchored
  gauge with `lambda_anc>0` and a pinned probe. The anchored regime is new and must pass a CPU gate before
  the GPU run is claimed.
- The single-site toy proof can overfit a tiny window count. The final claim requires many seeded
  non-IID participants and a disjoint held-out split.

## Honest Boundaries

This is a lightweight synthetic control environment, not the SO-100 robot task. It sidesteps the
unvendored `stable-worldmodel` blocker because the env is vendored in-process, but it does not claim
closed-loop physical robot success. In-browser training is out of scope: the SIGReg Epps-Pulley
characteristic-function integral has no off-the-shelf JavaScript autograd implementation. The stretch
target is in-browser inference plus environment simulation only.

## Acceptance

- `kinematic://swipe-dot` resolves to an action-sensitive eval world with true `state()` and
  true-state `succeeded()`.
- `synthetic-dynamic://swipe-dot?...` resolves to a deterministic read-only local dataset with resident
  `(x,y)` labels.
- The held-out `state_probe_r2` report is schema-versioned and binds every claim to checkpoint and
  control hashes.
- Docs, roadmap, model card, and evidence bundle carry the metric hierarchy and honest boundaries above.
