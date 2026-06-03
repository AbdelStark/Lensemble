# `third_party/` — vendored, separately-licensed subtrees

This tree holds ecosystem code that Lensemble vendors at a recorded upstream commit SHA so it can be
**modified in place** to ship the proof of concept fast. The policy — layout, patch-series discipline,
license handling, the import-boundary shim, and the upstream-sync procedure — is ratified in
[RFC-0016 — Deployment, Vendoring & Topology §2](../docs/rfcs/RFC-0016-deployment-vendoring-topology.md).
Each vendored project records its source URL, vendored SHA + date, license SPDX + in-tree LICENSE path,
local-modification log, and sync procedure in its own `UPSTREAM.md`:
[`stable_worldmodel/UPSTREAM.md`](stable_worldmodel/UPSTREAM.md) and
[`stable_pretraining/UPSTREAM.md`](stable_pretraining/UPSTREAM.md). These subtrees are **separately
licensed** (MIT) and live **outside** the `lensemble` import DAG of
[RFC-0001 §3](../docs/rfcs/RFC-0001-architecture.md#3-dependency-layering-no-cycles): **no `third_party`
symbol is re-exported from `lensemble.__init__`** — Lensemble reaches them only through a thin internal
shim, confining upstream API churn to one module.

> **Status:** the directories here are *scaffolding*. Real source lands only when the maintainer confirms
> each candidate SHA against upstream and clones the pristine snapshot in (the maintainer-gated
> real-vendor step of [#96](https://github.com/AbdelStark/Lensemble/issues/96)). See each `UPSTREAM.md`.
