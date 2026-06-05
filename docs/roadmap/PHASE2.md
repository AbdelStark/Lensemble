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

Dataset refs must pass the participant-silo smoke gate before a GPU run starts:

```bash
uv run --extra dev python scripts/phase2_split_lerobot_h5.py \
  --input /data/source/svla_so100_pickplace.h5 \
  --output-dir /tmp/lensemble-phase2-silos \
  --prefix phase2-so100-silo \
  --num-silos 2

uv run --extra dev python scripts/phase2_dataset_smoke.py \
  --data-source lerobot-h5:///tmp/lensemble-phase2-silos/phase2-so100-silo0.h5 \
  --data-source lerobot-h5:///tmp/lensemble-phase2-silos/phase2-so100-silo1.h5 \
  --participant-id phase2-a \
  --participant-id phase2-b \
  --window-steps 4 \
  --output phase2_dataset_smoke.json
```

The split policy is deterministic episode-level modulo assignment: source
episode `k` goes to silo `k % num_silos`, frames are never duplicated, and each
output HDF5 remaps `episode_index` to local 0-based ids. The split manifest
records the source hash, output hashes, selected source episode ids, frame
counts, and file paths. The smoke JSON report records participant ids, adapter
format, episode/window counts, dataset Merkle roots, action specs, and
first-window tensor shapes. It does not serialize raw observations, raw actions,
or private embeddings.

## Published Phase 2 Data Refs

The first Phase 2 dataset refs are published at
[`abdelstark/lensemble-phase2-so100-silos`](https://huggingface.co/datasets/abdelstark/lensemble-phase2-so100-silos)
revision `97336927606fea6fbfda308bb7cee6e7b48999fa`:

| Participant | File ref | Episodes | Windows (`window_steps=4`) | Dataset root |
|---|---|---:|---:|---|
| `phase2-so100-a` | `hf://datasets/abdelstark/lensemble-phase2-so100-silos/phase2-so100-silo0.h5` | 25 | 3149 | `df4dceed9ee55b95f2827f8b02ec3aa6b86a02421052eb84cfd96b41d7947c0a` |
| `phase2-so100-b` | `hf://datasets/abdelstark/lensemble-phase2-so100-silos/phase2-so100-silo1.h5` | 25 | 3210 | `ce6a42bab6edbdefd47f53f4cfc306cb4ed3db84d9f8ac8f7fcb2adc103c7b52` |

Source/provenance:

- derived from `abdelstark/so100-pickplace-lewm-ready/svla_so100_pickplace.h5`
  at revision `c210cb2f37b42954d31a17027e142c4cbdc7f7f8`;
- upstream HDF5 attrs identify `lerobot/svla_so100_pickplace` at revision
  `3d6d687a25cdf1565cdf24550814f72d999a861d`;
- upstream Hub metadata is public, ungated, and tagged `license:apache-2.0`;
- source SHA-256:
  `9dbeba303311d61a0129a6dcf3d0196524e7d8f58bb823e05dde0101546535ed`;
- silo file SHA-256 values are recorded in
  `phase2_silo_manifest.json` in the dataset repo.

Data contract:

- accepted format: `lerobot-h5`;
- camera/windowing: the current adapter reads `observation/pixels_top`,
  decodes uint8 frames to `[0,1]` float clips of shape `(1, 3, 224, 224)`,
  and produces windows with observation shape `(5, 1, 3, 224, 224)` for
  `window_steps=4`;
- action spec: continuous `lerobot-6dof`, action shape `(4, 6)`;
- declared held-out split policy for #206: the final local episode in each
  silo is reserved for held-out evaluation (`source_episode=48` for
  `phase2-so100-a`, `source_episode=49` for `phase2-so100-b`); train/eval
  reports must record whether they honor or intentionally override this split.

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
  -v hf://datasets/abdelstark/lensemble-phase2-so100-silos:/data/phase2 \
  -d https://raw.githubusercontent.com/AbdelStark/Lensemble/<SHA>/deploy/hfjobs/train_federated_lewm.py \
  --data-source lerobot-h5:///data/phase2/phase2-so100-silo0.h5 \
  --data-source lerobot-h5:///data/phase2/phase2-so100-silo1.h5 \
  --participant-id phase2-so100-a \
  --participant-id phase2-so100-b \
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
