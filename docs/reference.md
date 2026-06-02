# API Reference

The 1.0-frozen public surface ([Public API](spec/02-public-api.md)). `lensemble.__version__` is the
package SemVer string. Generated from the docstrings of the symbols re-exported from `lensemble`.

## Configuration

::: lensemble.config
    options:
      members:
        - LensembleConfig
        - RunManifest
        - load

## Model

::: lensemble.model
    options:
      members:
        - build_encoder
        - build_predictor
        - build_action_head
        - Objective

## Gauge

::: lensemble.gauge
    options:
      members:
        - frame_drift
        - procrustes_align

## Federation & runtime

::: lensemble.federation
    options:
      members:
        - Coordinator
        - Participant
        - RoundState
        - train_local

## Evaluation

::: lensemble.eval
    options:
      members:
        - evaluate
        - Planner

## Provenance

::: lensemble.provenance
    options:
      members:
        - commit_dataset
        - DatasetCommitment
        - ContributionLedger

## Verification

::: lensemble.verify
    options:
      members:
        - recompute_alignment
