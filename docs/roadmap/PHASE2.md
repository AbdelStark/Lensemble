# Phase 2 Empirical Evidence Roadmap

Phase 2 turns the claim-MVP into a scaled evidence bundle. It is empirical by
default: larger participant silos, GPU-backed HF Jobs, downstream evaluation,
baselines, curves, and a model-card/report artifact. The cryptographic proof
layer remains governed by RFC-0006 and is not implicitly complete when this
roadmap closes.

Tracker: [#200](https://github.com/AbdelStark/Lensemble/issues/200)

## Baseline

The claim-MVP established the narrow end-to-end path:

- two LeRobot-H5 participant silos;
- claim-grade live-target LeWorldModel objective
  (`objective.target_stop_gradient=false`);
- one closed federated round;
- published dataset/checkpoint/report artifacts;
- final HF Job
  [`6a229653e52fdd2a02ed9125`](https://huggingface.co/jobs/abdelstark/6a229653e52fdd2a02ed9125);
- final global hash
  `cf1c99a7e94ca610daa3bfc00c99d9ee68e9e34a302a96d848508e88edf4c0d5`.

Phase 2 starts from that working substrate and raises the evidence bar.

## Workstreams

| Issue | Workstream | Exit gate |
|---|---|---|
| [#201](https://github.com/AbdelStark/Lensemble/issues/201) | Participant-silo dataset contract and refs | Publish or mount at least two non-toy silo refs, or record the exact blocker. |
| [#202](https://github.com/AbdelStark/Lensemble/issues/202) | GPU-backed multi-round HF Jobs | Complete at least one pinned GPU HF Job that publishes checkpoint/report artifacts. |
| [#206](https://github.com/AbdelStark/Lensemble/issues/206) | Downstream planning/eval report | Generate an EvalReport-style artifact from a Phase 2 checkpoint. |
| [#205](https://github.com/AbdelStark/Lensemble/issues/205) | Baselines, ablations, and curves | Produce generated curve/table artifacts tied to run/config/checkpoint hashes. |
| [#204](https://github.com/AbdelStark/Lensemble/issues/204) | Evidence bundle and model card | Publish one model-card/report bundle with artifact refs and claim boundaries. |
| [#203](https://github.com/AbdelStark/Lensemble/issues/203) | README and roadmap docs | Keep public docs aligned with the tracker and known limitations. |

The machine-readable matrix backing these rows is rendered by:

```bash
uv run --extra dev python scripts/phase2_matrix.py --format markdown
uv run --extra dev python scripts/phase2_matrix.py --format json
```

## Minimum Evidence Contract

Before #200 closes, the final report must include:

- dataset repositories or immutable dataset refs;
- participant ids, dataset roots, action specs, frame skip/windowing, and held-out split policy;
- pinned git SHA and HF Job command;
- checkpoint repo and final model hash;
- per-round or curve-ready metrics for prediction loss, SIGReg, effective rank, and frame drift;
- downstream eval report when an environment is available;
- baseline/ablation table, or explicit blockers for missing controls;
- model-card text that distinguishes engineering evidence from paper-scale performance claims.

## Starter HF Jobs Shape

Use the federated launcher with a pinned commit and mounted silos. Prefer a
dry-run before any expensive run:

```bash
hf jobs uv run --flavor h200 --timeout 2h --secrets HF_TOKEN \
  --with 'lensemble @ git+https://github.com/AbdelStark/Lensemble.git@<SHA>' \
  -v hf://datasets/<org>/<phase2-silo-a>:/data/a \
  -v hf://datasets/<org>/<phase2-silo-b>:/data/b \
  -d https://raw.githubusercontent.com/AbdelStark/Lensemble/<SHA>/deploy/hfjobs/train_federated_lewm.py \
  --data-source lerobot-h5:///data/a/<silo-a>.h5 \
  --data-source lerobot-h5:///data/b/<silo-b>.h5 \
  --out-dir /tmp/lensemble-phase2 \
  --image-size 224 --patch-size 14 --latent-dim 192 \
  --depth 12 --predictor-depth 6 --num-heads 3 \
  --probe-points 1024 --inner-horizon 4 --window-steps 4 \
  --num-rounds 8 --metric-windows 256 \
  --push --out-repo <org>/lensemble-phase2-checkpoint
```

Adjust model scale and HF flavor only after a smaller dry-run verifies mounts,
window counts, probe generation, and report publication.

## Claim Boundary

Closing Phase 2 should support a stronger engineering statement: Lensemble can
train and evaluate a federated end-to-end JEPA-style world model on larger
participant-local robot silos with published artifacts. It should not claim
paper-scale LeWorldModel performance, broad robotics generalization, or
cryptographic contribution proofs unless those are separately evidenced.
