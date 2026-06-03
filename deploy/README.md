# `deploy/` — infrastructure-as-code (topology backbone)

Infrastructure-as-code for the Lensemble deployment topology, ratified in
[RFC-0016 — Deployment, Vendoring & Topology §4](../docs/rfcs/RFC-0016-deployment-vendoring-topology.md)
("one config source, three substrates, one transport"). `LensembleConfig`
([RFC-0009 §2](../docs/rfcs/RFC-0009-configuration-reproducibility.md#2-structured-configuration-tree))
is the single source of truth; every substrate here is a *renderer* over it.

> **Status:** these are **stubs** for the topology backbone. They make RFC-0016's acceptance
> (`docker compose --profile cpu up`, `helm template`, `kustomize build`) *structurally* unblocked, but
> are not yet runnable deployments. The real wiring (a working image, the networked control-plane
> transport, and config-driven rendering) lands with #45's networked transport and the real vendoring
> (#96).

| Path | Substrate (RFC-0016 §4) | Status |
|---|---|---|
| [`compose.yaml`](compose.yaml) | Layer 1 — Docker Compose, local multi-node (coordinator + participant, `cpu` profile) | topology stub |
| [`helm/Chart.yaml`](helm/Chart.yaml) | Layer 2 — Kubernetes (Helm chart), Stage C+ | minimal valid stub |
| [`kustomize/base/kustomization.yaml`](kustomize/base/kustomization.yaml) | Layer 2 — Kubernetes (Kustomize base + overlays), Stage C+ | minimal valid stub |
