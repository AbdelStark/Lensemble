# Lensemble

Research stack for reproducible federated training and evaluation of action-conditioned JEPA world models.

Lensemble implements the model, participant, coordinator, aggregation, privacy,
artifact, and evaluation machinery needed to study federated JEPA world models
while participant trajectories remain local.

## Research status

The code can execute joint encoder-and-predictor training through the federation
stack. That software capability is not an achieved useful and gauge-stable
scientific result. The checked-in full-model evidence predates correction of an
outer-step sign defect and must be rerun before it supports new claims. The
corrected research loop remains open in
[#335](https://github.com/AbdelStark/Lensemble/issues/335).

The optional browser experience is narrower: it performs federated continuation
of a bounded residual adapter on a frozen LeWorldModel checkpoint. It is not
full-model browser training.

## Install

From a clean checkout:

```bash
uv venv .venv --python 3.11
uv pip install "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[dev,docs]"
```

## Validate the core loop

```bash
uv run pytest \
  tests/e2e/test_toy_pipeline.py::test_federated_round_commits_and_advances_the_global_hash \
  -q
```

This small CPU test runs two real participants through the in-process
coordinator, closes a deterministic aggregation round, and verifies that the
committed global hash advances. It is a software smoke test, not benchmark
evidence.

## Read by task

- Understand the system: [architecture](spec/01-architecture.md) and
  [federated protocol](rfcs/RFC-0003-federated-protocol.md).
- Understand the central research problem:
  [latent gauge and aggregation](rfcs/RFC-0002-gauge-and-aggregation.md).
- Inspect evaluation contracts:
  [evaluation protocol](rfcs/RFC-0005-evaluation.md) and
  [dynamic-env metrics](rfcs/RFC-0017-dynamic-env-ungameable-metrics.md).
- Inspect the historical, pre-outer-step-sign-correction CPU investigation
  (the checked values require a corrected rerun and are not current-code
  reproduction evidence):
  [full-model federation spike](spikes/0001-federated-world-model-training/README.md).
- Inspect public claim surfaces:
  [historical Phase 3 model card](evidence/phase3_model_card.md) and
  [frozen-checkpoint adapter demo card](evidence/lewm_tworooms_demo_card.md).
- Use the Python package: [API reference](reference.md).
- Navigate the complete contract corpus:
  [specification overview](spec/00-overview.md) or
  [repository index](https://github.com/AbdelStark/Lensemble/blob/main/SPEC.md).

## Optional browser demo

```bash
uv run lensemble demo federated --port 8765
```

Follow the printed local URL. Read the
[demo card](evidence/lewm_tworooms_demo_card.md) before presenting its result.

## Claim boundary

Lensemble is a research codebase. It is not a production federation service, a
cryptographic proof system, a full-model browser trainer, or evidence of
closed-loop physical robot success.
