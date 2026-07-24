# Lensemble

Research stack for reproducible federated training and evaluation of action-conditioned JEPA world models.

[![CI](https://img.shields.io/github/actions/workflow/status/AbdelStark/Lensemble/ci.yml?branch=main&label=ci&style=for-the-badge)](https://github.com/AbdelStark/Lensemble/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/AbdelStark/Lensemble/docs.yml?branch=main&label=docs&style=for-the-badge)](https://github.com/AbdelStark/Lensemble/actions/workflows/docs.yml)
[![Determinism](https://img.shields.io/github/actions/workflow/status/AbdelStark/Lensemble/determinism.yml?branch=main&label=determinism&style=for-the-badge)](https://github.com/AbdelStark/Lensemble/actions/workflows/determinism.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-3C7A57?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/status-research%20alpha-7A4EAB?style=for-the-badge)](#status)

Lensemble is a research implementation of federated, action-conditioned JEPA world models. It is built for the hard case: each participant keeps raw trajectory data local, trains against the same protocol, and only releases update artifacts for aggregation.

The central problem is the JEPA latent gauge. In self-supervised representation learning, two participants can learn useful but rotated latent frames. Naive FedAvg then averages incompatible coordinates. Lensemble implements the frame-anchor, aggregation, privacy, provenance, and evaluation machinery needed to study that failure mode directly.

The repository can execute joint encoder-and-predictor training through its
`Coordinator` and `Participant` federation stack. That is an implemented
research capability, not evidence that the current recipe has achieved a
simultaneously useful, magnitude-stable, and gauge-stable federated world model.
The checked-in full-model results predate correction of an outer-step sign
defect and must be rerun before they support new scientific claims. That work
remains tracked in [issue #335](https://github.com/AbdelStark/Lensemble/issues/335).

## What is in this repo

- A typed Python package and `lensemble` CLI.
- A full encoder-and-predictor execution path, SIGReg/anchor objectives, and federated round machinery.
- Tested secure-aggregation and differential-privacy building blocks, with the
  missing live integration documented explicitly.
- Hash-bound checkpoints, manifests, dataset provenance, and evidence bundles.
- CPU-oriented tests for contracts, determinism, residency, aggregation, DP, and dynamic-env evaluation.
- A browser federated demo for QR joins, WebSocket orchestration, bounded tiny browser updates, aggregation, inference, and evidence export.

## Status

| Area | Current state |
|---|---|
| SO-100 federation | Historical, pre-sign-correction gauge-only result. Anchored federation controls the latent frame where naive FedAvg fails. It does not prove downstream robotics usefulness and requires a corrected rerun. |
| Dynamic env | Historical, pre-sign-correction educational systems result. Federated scratch reaches `state_probe_r2=0.8885337114`, but local-only reaches `0.8838405609`; the `0.0046931505` margin misses the required `0.05`. A corrected rerun is required. |
| Browser demo (orchestration) | Local and public-demo orchestration with WebSocket primary transport, REST polling fallback, bounded update vectors, aggregation, inference UI, and evidence export. Not production browser training. |
| Browser demo (`real-lewm-tworooms`) | Runs the pinned LeWorldModel TwoRooms checkpoint in the browser via hash-checked ONNX and trains a bounded 12,512 parameter (0.069%) residual adapter on the frozen predictor output. Only the adapter trains and federates; the world model stays frozen. The before/after probe is system-composed (real adapter deltas go through the real server validation and aggregation path, then the probe scores the server-produced revision), collapse-checked on held-out pairs, and seed-robust across 5 seeds (mean +16.8%, worst +5.4%). Single local coordinator, mean of clipped deltas, no robust aggregation or DP in this path. Not full-model or production browser training. |
| Privacy path | Clip/noise and aggregation backend primitives are implemented, but the live coordinator still commits from visible participant deltas. Its secure sum is a post-commit cross-check, accounting is not cumulative, and deterministic noise is reconstructible from the shared run seed. There is no current end-to-end secure-aggregation or differential-privacy claim. |
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

Validate the core federation loop:

```bash
uv run pytest \
  tests/e2e/test_toy_pipeline.py::test_federated_round_commits_and_advances_the_global_hash \
  -q
```

This CPU test runs two real `Participant` instances through the in-process
`Coordinator`, closes a deterministic aggregation round, and verifies that the
committed global hash advances. It is the small CI smoke for the core training
machinery, not benchmark evidence.

Inspect the CLI after the core smoke passes:

```bash
uv run lensemble --help
```

## Optional browser demo

The browser experience is an educational adapter-continuation demo, not the
primary full-model training path. It keeps the LeWorldModel checkpoint frozen
and federates only a bounded residual adapter.

Start it locally:

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
  -> participant-side frame alignment and re-difference (target protocol)
  -> clip, secret noise, optional quantization, encode/mask
  -> secure sum (target) or explicitly reported plaintext research path (current)
  -> deterministic outer update
  -> checkpoint, manifest, evidence bundle
  -> evaluation against explicit baselines
```

The secure-aggregation and privacy-accounting modules are research building
blocks around this loop; they are not yet the optimizer's end-to-end release
path. See [RFC-0011](docs/rfcs/RFC-0011-secure-aggregation.md) and
[RFC-0012](docs/rfcs/RFC-0012-differential-privacy.md) for the exact blockers.

The design is specified in [SPEC.md](SPEC.md), with normative sections in [docs/spec](docs/spec/) and decision records in [docs/rfcs](docs/rfcs/).

Start here:

- [RFC-0002: latent gauge and frame-anchored aggregation](docs/rfcs/RFC-0002-gauge-and-aggregation.md)
- [RFC-0005: evaluation protocol](docs/rfcs/RFC-0005-evaluation.md)
- [RFC-0017: dynamic-env metric gate](docs/rfcs/RFC-0017-dynamic-env-ungameable-metrics.md)
- [Dynamic-env evidence roadmap](docs/roadmap/DYNAMIC_ENV.md)

## Evidence

The project treats results as artifact-bound. The important public evidence surfaces are checked in:

- [Phase 3 evidence bundle](docs/evidence/phase3_evidence_bundle.json)
- [Phase 3 model card](docs/evidence/phase3_model_card.md)
- [Tapestry-like LeWM demo card](docs/evidence/lewm_tworooms_demo_card.md)
- [LeWM TwoRooms system-composed probe](docs/evidence/lewm_tworooms_system_probe.json)
- [LeWM TwoRooms probe seed sweep](docs/evidence/lewm_tworooms_probe_seedsweep.json)
- [Dynamic-env roadmap and acceptance matrix](docs/roadmap/DYNAMIC_ENV.md)
- [Browser federated demo docs](docs/roadmap/BROWSER_FEDERATED_DEMO.md)

The short read: Lensemble has systems and historical gauge-control evidence,
but its full-model results predate correction of the outer-step sign defect.
They must be rerun before supporting new claims. The project does not yet have
a claim-grade result that is simultaneously useful, magnitude-stable, and
gauge-stable under federation; [issue #335](https://github.com/AbdelStark/Lensemble/issues/335)
tracks that research loop.

The LeWM TwoRooms browser demo has a credible result for a narrow claim: federated adaptation of a bounded adapter on a frozen checkpoint. The headline probe number is produced by the shipped system, not offline math. Real adapter deltas pass the server validation and aggregation path, and the probe scores the server-produced revision. The held-out gain (+12.3% MSE on the committed seed) is checked to be bias-correction, not latent collapse, and holds across 5 seeds (worst +5.4%). It is not full-model federated training, runs through a single local coordinator, and does not wire secure aggregation or DP in that path.

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

## Project policies

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow,
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations,
[SECURITY.md](SECURITY.md) for private vulnerability reporting, and
[CITATION.cff](CITATION.cff) for the machine-readable software citation.

## License

Code is [Apache-2.0](LICENSE). Documentation is [CC-BY-4.0](LICENSE-docs). Released data artifacts use [CDLA-Permissive-2.0](LICENSE-data).
