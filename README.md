# Lensemble

Federated JEPA world models for sovereign robotics data.

[![CI](https://img.shields.io/github/actions/workflow/status/AbdelStark/Lensemble/ci.yml?branch=main&label=ci&style=for-the-badge)](https://github.com/AbdelStark/Lensemble/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/AbdelStark/Lensemble/docs.yml?branch=main&label=docs&style=for-the-badge)](https://github.com/AbdelStark/Lensemble/actions/workflows/docs.yml)
[![Determinism](https://img.shields.io/github/actions/workflow/status/AbdelStark/Lensemble/determinism.yml?branch=main&label=determinism&style=for-the-badge)](https://github.com/AbdelStark/Lensemble/actions/workflows/determinism.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-3C7A57?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/status-research%20alpha-7A4EAB?style=for-the-badge)](#status)

Lensemble is a research implementation of federated, action-conditioned JEPA world models. It is built for the hard case: each participant keeps raw trajectory data local, trains against the same protocol, and only releases update artifacts for aggregation.

The central problem is the JEPA latent gauge. In self-supervised representation learning, two participants can learn useful but rotated latent frames. Naive FedAvg then averages incompatible coordinates. Lensemble implements the frame-anchor, aggregation, privacy, provenance, and evaluation machinery needed to study that failure mode directly.

## What is in this repo

- A typed Python package and `lensemble` CLI.
- End-to-end JEPA model components, SIGReg/anchor objectives, and federated round machinery.
- Secure aggregation and differential-privacy plumbing for the release path.
- Hash-bound checkpoints, manifests, dataset provenance, and evidence bundles.
- CPU-oriented tests for contracts, determinism, residency, aggregation, DP, and dynamic-env evaluation.
- A browser federated demo for QR joins, WebSocket orchestration, bounded tiny browser updates, aggregation, inference, and evidence export.

## Status

| Area | Current state |
|---|---|
| SO-100 federation | Gauge-only result. Anchored federation controls the latent frame where naive FedAvg fails. It does not prove downstream robotics usefulness. |
| Dynamic env | Educational systems demo. Federated scratch reaches `state_probe_r2=0.8885337114`, but local-only reaches `0.8838405609`; the `0.0046931505` margin misses the required `0.05`. |
| Browser demo | Implemented as local/public-demo orchestration with WebSocket primary transport, REST polling fallback, bounded tiny browser update vectors, aggregation, inference UI, and evidence export. This is not production browser training. |
| Proof layer | Artifact and provenance contracts exist. There is no cryptographic proof of honest participant computation yet. |
| Clinical, safety, or deployment claim | None. This is a research codebase. |

## Quickstart

```bash
git clone https://github.com/AbdelStark/Lensemble.git
cd Lensemble

uv venv .venv --python 3.11
uv pip install "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[dev,docs]"
```

Check the CLI:

```bash
uv run lensemble --help
```

Run the browser federated demo:

```bash
uv run lensemble demo federated --port 8765
```

Open the printed URL, usually:

```text
http://127.0.0.1:8765/web/federated-demo/
```

For tunnel or LAN rehearsal, bind the coordinator and provide the external base
URL used in QR codes and WSS URLs:

```bash
uv run lensemble demo federated \
  --host 0.0.0.0 \
  --public-base-url https://YOUR-TUNNEL.trycloudflare.com/web/federated-demo \
  --public-demo \
  --deployment-target cloudflare-tunnel
```

## Architecture

```text
participant data stays local
  -> local JEPA training
  -> update artifact, hash, metadata
  -> secure aggregation and DP accounting
  -> frame-anchored global update
  -> checkpoint, manifest, evidence bundle
  -> evaluation against explicit baselines
```

The design is specified in [SPEC.md](SPEC.md), with normative sections in [docs/spec](docs/spec/) and decision records in [docs/rfcs](docs/rfcs/).

Start here:

- [RFC-0002: latent gauge and frame-anchored aggregation](docs/rfcs/RFC-0002-gauge-and-aggregation.md)
- [RFC-0005: evaluation protocol](docs/rfcs/RFC-0005-evaluation.md)
- [RFC-0017: dynamic-env metric gate](docs/rfcs/RFC-0017-dynamic-env-ungameable-metrics.md)
- [Browser federated demo roadmap](docs/roadmap/BROWSER_FEDERATED_DEMO.md)
- [Dynamic-env evidence roadmap](docs/roadmap/DYNAMIC_ENV.md)

## Evidence

The project treats results as artifact-bound. The important public evidence surfaces are checked in:

- [Phase 3 evidence bundle](docs/evidence/phase3_evidence_bundle.json)
- [Phase 3 model card](docs/evidence/phase3_model_card.md)
- [Dynamic-env roadmap and acceptance matrix](docs/roadmap/DYNAMIC_ENV.md)
- [Browser federated demo docs](docs/roadmap/BROWSER_FEDERATED_DEMO.md)

The short read: Lensemble has credible systems and gauge-control evidence. It does not yet have a claim-grade result that federated training materially beats local-only on the binding dynamic-env metric.

SO-100 is not a downstream-useful world model: held-out latent magnitude collapse (`~7.5e-6`; `thoughts/collapse_fix_probe.py`), the central ceiling probe (`thoughts/central_ceiling_probe.py`), `skill_vs_identity is gameable`, and `effective_rank is scale-invariant`.

## Development

Useful local gates:

```bash
uv run ruff check .
uv run pyright
uv run pytest tests/unit tests/property tests/integration tests/ml tests/e2e tests/regression
uv run python scripts/check_docs_links.py docs SPEC.md README.md
uv run python -m mkdocs build --strict
git diff --check
```

The blocking CI gates run on CPU and do not download private datasets, checkpoints, or probes. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full merge rule.

## Repository map

```text
lensemble/              Python package and CLI
tests/                  unit, property, integration, ML, e2e, regression
docs/spec/              normative architecture and public API docs
docs/rfcs/              design records and research contracts
docs/roadmap/           implementation and evidence state
docs/evidence/          checked-in reports, bundles, and model cards
web/federated-demo/     browser federation demo
deploy/hfjobs/          Hugging Face Jobs launchers
scripts/                release, evidence, docs, and benchmark producers
```

## Non-claims

Lensemble is not a production federation stack, not a browser-training product, not a cryptographic proof system, and not evidence of closed-loop physical robot success. Raw participant trajectories are not released and should not cross trust boundaries.

## License

Code is [Apache-2.0](LICENSE). Documentation is [CC-BY-4.0](LICENSE-docs). Released data artifacts use [CDLA-Permissive-2.0](LICENSE-data).
