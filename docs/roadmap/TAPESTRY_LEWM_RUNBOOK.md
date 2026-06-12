# Tapestry-like LeWM TwoRooms — Rehearsal Gate and Researcher Runbook

The end-to-end gate and stage runbook for the Tapestry-like real-LeWM federated demo
([#324](https://github.com/AbdelStark/Lensemble/issues/324), epic
[#314](https://github.com/AbdelStark/Lensemble/issues/314); contract in
[TAPESTRY_LEWM.md](TAPESTRY_LEWM.md), public evidence surface in the
[demo card](../evidence/lewm_tworooms_demo_card.md)).

## Automated gates (run before any showing)

```bash
# 0. one-time: artifacts the browser loads (downloads the pinned checkpoint; ~74 MB of graphs)
uv run python scripts/lewm_tworooms_ingest.py
uv run --with onnx --with onnxscript --with onnxruntime python scripts/lewm_tworooms_export.py

# 1. orchestration/evidence rehearsal: 2-participant smoke + 4-participant dropout/reconnect
#    + stale-round rejection + claim audit; add a longer configurable gate when needed
uv run python scripts/lewm_demo_rehearsal.py
uv run python scripts/lewm_demo_rehearsal.py --long-rounds 50

# 2. the real-math gates (need the expert dataset tworoom.h5 and node)
uv run --with onnxruntime --with hdf5plugin python scripts/lewm_tworooms_realdata_check.py --h5 <tworoom.h5>
uv run --with onnxruntime --with hdf5plugin python scripts/lewm_adapter_overfit_check.py --h5 <tworoom.h5>
uv run --with onnxruntime --with hdf5plugin python scripts/lewm_probe_check.py --h5 <tworoom.h5>

# 3. the standing repo gates
uv run pytest tests/ml/test_lewm_tworooms.py tests/ml/test_lewm_export.py \
  tests/ml/test_lewm_tworooms_browser.py tests/ml/test_lewm_adapter.py \
  tests/ml/test_lewm_federation.py tests/ml/test_lewm_demo_integration.py \
  tests/ml/test_lewm_probe.py tests/ml/test_lewm_evidence_audit.py \
  tests/ml/test_federated_demo_app.py
uv run python scripts/check_docs_links.py docs SPEC.md README.md
uv run python -m mkdocs build --strict
git diff --check
```

Gate rule: if `lewm_probe_check.py` reports a verdict other than `improved`, the negative result
is recorded in `docs/evidence/lewm_tworooms_probe_check.json` and **blocks public positive
claims** — show the systems story, state the negative result, do not soften it.

## Two-browser local rehearsal (the #324 acceptance path)

1. `uv run lensemble demo federated --port 8765` — startup must print
   `real_lewm_mode=available checkpoint_revision=77adaae0bc31`. If it prints `unavailable`, run
   step 0 above; the real mode fails closed without the export.
2. Open the printed host URL. Create a run with learner path **real-lewm-tworooms** (the
   simulator mode refuses it), quorum 2, rounds 2–5 to start.
3. Join from two browser tabs/devices via the QR/join URL — one default (**auto**), one set to
   **manual** for the debug view.
4. The auto tab proceeds without any clicking: runtime load (hash-verified), resident rollout
   collection, adapter training, bounded delta submission. The manual tab shows the same work
   behind an explicit button. Progress reflects real optimizer steps; there are no scripted
   sleeps anywhere in the path.
5. Watch the host dashboard over WebSocket (no refreshing): participant slots, round progress,
   per-participant adapter metrics (pred loss first→last, SIGReg, effective rank), round metric
   cards with health flags, and `lewmrev-*` revisions.
6. After the final round: run the **before/after validation probe** from the host dashboard. It
   scores the final global adapter against the parent checkpoint on a fixed seeded validation
   set in the host browser and prints `improved`/`flat`/`worse` honestly.
7. Export evidence JSON. It must carry `runMode: real-lewm-tworooms`, the checkpoint/export
   binding, per-round delta hashes, health flags, and the Tapestry-like non-claim text; the
   claim audit (`lensemble.demo.evidence_audit`) accepts exactly this shape.
8. Open the TwoRooms lab (`#/tworooms`) to show checkpoint-backed rollout/planning inference
   directly.

## Four-phone stage runbook

Same as the two-browser path, plus the [BROWSER_FEDERATED_DEMO.md](BROWSER_FEDERATED_DEMO.md)
public-path checklist (Cloudflare tunnel / LAN, `--public-base-url`, QR joins, screens awake).
Stage-specific notes:

- Quorum 3 of 4 so one phone can die on stage without ending the run; rehearse the dropout once
  (the run continues, the event timeline shows `participant.dropped`).
- Rehearse one reconnect: refresh a phone mid-round; its slot resumes (`connection.opened`,
  `reconnectCount` increments) without a new join.
- WebGPU phones run rounds in seconds; WASM fallback is slower but real — say so rather than
  shrinking the work budget on stage.
- Keep rate limits at their default (disabled) unless the event network requires them.

## What to say (and what not to claim)

Narration that stays inside the claim boundary:

> Each phone keeps its TwoRooms rollouts local, runs real checkpoint-backed LeWorldModel
> inference through hash-verified exported graphs, trains a small bounded adapter around the
> frozen checkpoint, and submits only a clipped adapter delta with metric summaries. The
> coordinator validates every delta against the pinned checkpoint and export hashes, averages
> them deterministically, and publishes hash-bound global revisions. This is a Tapestry-like
> federated **adaptation** demo around a real LeWorldModel checkpoint — sovereign participants,
> bounded artifacts, claim-bounded evidence.

What the metrics prove, and no more:

- prediction loss decreasing per participant = the bounded subset really trains in the browser;
- the +12% before/after probe (committed evidence) = aggregated revisions beat the parent
  checkpoint on a fixed held-out validation probe, because the frozen predictor's residual is
  systematically biased — a demo-scale result, **not** a benchmark;
- SIGReg/effective-rank/latent-std flags = the anti-collapse diagnostics are live, not décor.

Use "Tapestry-like" explicitly when describing the architecture — it names the concept set
(sovereign participants, shared protocol, bounded artifacts, coordinator aggregation,
observability, claim-bounded evidence), not an affiliation.

Never claim: full from-scratch LeWM browser pretraining, production browser training,
paper-scale TwoRooms/PushT parity, cryptographic honest-computation proof, physical robotics
success, or that any raw frame/action/latent/tensor/weight left a browser. The
[demo card](../evidence/lewm_tworooms_demo_card.md) carries the full non-claim list — read it
before presenting.

Remaining future work (say it when asked): secure aggregation and DP for the demo path (the
main stack's RFC-0011/0012 are not wired into it), richer planning metrics in the federated
view, the PushT checkpoint, and WebGPU training of larger subsets.

## Failure paths

| Symptom | Action |
|---|---|
| `real_lewm_mode=unavailable` at startup | regenerate the export (step 0); real-mode run creation fails closed until then |
| Participant shows `real-lewm runtime unavailable: …` | that browser lacks WASM/WebGPU or cannot fetch the graphs; the round fails visibly by design — fix the network or use another device; never switch the run to the surrogate path mid-demo |
| Probe says `flat`/`worse` | report it; the run's evidence stays valid as a systems demo, positive adaptation claims are blocked |
| Health flags on a round card | read them out (collapse warning / flat loss); they exist to be seen |
| Quorum lost | the run fails with an explicit event; create a fresh run (no persistent state) |
