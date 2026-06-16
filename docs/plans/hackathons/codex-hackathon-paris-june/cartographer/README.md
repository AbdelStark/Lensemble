# Codex Hackathon — Paris (June) — Cartographer plan

Comprehensive implementation plan for **Cartographer** ([#339](https://github.com/AbdelStark/Lensemble/issues/339)): a federated JEPA world-model demo whose downstream payoff is an interactive WebGPU map of the model's latent manifold and its planning.

> **Backbone (Act 1):** a live federated **adapter-continuation** round on the frozen LeWorldModel TwoRooms checkpoint — the room trains a tiny adapter together, no data shared.
> **Cherry (Act 2):** Cartographer — fly through the latent manifold, watch latent MPC plan, toggle healthy↔collapsed and pre↔post-federation.

## Read in order

| Doc | Purpose |
|---|---|
| [`00-overview.md`](00-overview.md) | Pitch, scope, the 10 locked decisions (D1–D10), high-level architecture, timeline, project DoD |
| [`01-architecture.md`](01-architecture.md) | Implementation reference: every component, verified APIs with file:line, what's `[BUILD]` |
| [`02-data-contracts.md`](02-data-contracts.md) | Frozen JSON schemas: `cartographer-manifold/1` (viewer) and `lewm-manifold/1` (evidence) |
| [`03-workstreams-and-issues.md`](03-workstreams-and-issues.md) | The 9 child issues, dependency graph, ordering, test/gate matrix |
| [`04-demo-runsheet.md`](04-demo-runsheet.md) | 90-second demo script, live Act-1 option, fallback ladder, pre-flight checklist |
| [`05-risks-and-claim-discipline.md`](05-risks-and-claim-discipline.md) | Risk register + the binding claim-discipline checklist and approved/forbidden phrasings |

## Sibling ideas (not chosen)
- [#337 Latent Genie](https://github.com/AbdelStark/Lensemble/issues/337) — playable latent world (blocked by: no pixel decoder).
- [#338 Surprise-meter](https://github.com/AbdelStark/Lensemble/issues/338) — live prediction-error meter (blocked by: CLS-latent → scalar, no per-patch head).
- Why Cartographer won: see `00-overview.md` §2 — lowest live-failure risk (pre-baked), no architecture gap, still a screensaver-grade shareable visual.

## Source of truth
The plan is grounded in a feasibility spike + four code-recon passes (see `#339` comments and `00`/`01`). Headline facts: model is **CLS-latent, 192-d, 3-frame window, no decoder**; inference is **~6 ms/step on CPU** (>80 fps); federated improvement **+12.3% committed / +16.8% seed-mean** (already certified in `docs/evidence/`); real healthy **effective rank ≈ 9.86 / 192**.
