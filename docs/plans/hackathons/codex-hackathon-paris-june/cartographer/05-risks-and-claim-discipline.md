# Cartographer — Risks & Claim Discipline

## 1. Risk register

| # | Risk | Likelihood | Impact | Mitigation | Owner issue |
|---|---|---|---|---|---|
| R1 | Instrumented MPC doesn't reach goals / candidates look like noise | Med | High (kills the "watch it plan" beat) | Tune horizon/samples on harvested start/goal pairs; if planning is weak, fall back to showing the **chosen** rollout only (still compelling); unit-test on toy dynamics first | CART-2 |
| R2 | LeWM 3-frame dynamics closure is wrong (single-step vs temporal mismatch) | Med | High | Validate the ring-buffer dynamics against `model.rollout` on a known trajectory before wiring into the planner; assert latent agreement | CART-2 |
| R3 | Python adapter port (mulberry32 init) diverges from JS | Med | Med | Default to the **node bridge** (`lewm_system_round.mjs op=probe`) already used by `system_probe.py`; Python port is stretch with ≤1e-3 parity gate | CART-3 |
| R4 | PCA to 3D on a ~10-eff-rank cloud looks like a blob | Low-Med | Med | Report `varianceExplained`; if 3D is muddy, color by episode/time and add gentle jitter; UMAP is a labelled stretch | CART-4 |
| R5 | `manifold.json` too big / viewer janky | Low | Med | Enforce caps (4000 pts, 24 candidates/iter); LOG downsampling; WebGL handles this trivially | CART-7/8 |
| R6 | WebGPU absent on projector machine | Med | High | Use Three.js **WebGLRenderer by default**; WebGPU is opt-in; clip fallback (rung D) | CART-8 |
| R7 | Bake fails on the day | Low | High | Committed pre-event fallback `manifold.json` (rung C) | CART-9 |
| R8 | Over-claim slips onto screen/X | Med | High (credibility) | The claim checklist below; nonClaims rendered from the artifact; evidence test asserts negations | all |
| R9 | Harvest normalization mismatch (uint8/ImageNet) → garbage latents | Med | High | Verify `encode_frames` input expectation against `lewm_tworooms.py`; sanity-check eff_rank ≈ 9.86 and latent_std ≈ 0.90 against the certified system-probe diagnostics | CART-1 |
| R10 | Time overrun (solo, one day) | Med | Med | Strict critical path (CART-1→4→7→9); CART-8 parallel on mock; v0 fallback baked Wed eve | — |
| R11 | Touching transformers version breaks checkpoint parity | Low | High | Do not change the env; pinned reconstruction is parity-tested; bake in the existing `uv` env | CART-1/3 |

## 2. Claim-discipline checklist (AGENTS.md §Claim Discipline is binding)

Every public surface — viewer text, `manifold.json.nonClaims`, evidence JSON, X post, issue copy — must satisfy:

- [ ] **Federation is described as** "federated adapter continuation on a frozen checkpoint" — **never** "federated world-model training" or "the room trained the world model." The room trained a **12,512-param (0.069%) adapter**; the backbone is frozen.
- [ ] The federated improvement number (+12.3% committed / +16.8% seed-mean) is **sourced** to `lewm_tworooms_system_probe.json` / `lewm_tworooms_probe_seedsweep.json`, not re-derived.
- [ ] The collapsed cloud is labelled **"synthetic illustration"** wherever shown; never implied to be a trained model.
- [ ] The 3D view is labelled a **PCA projection** with variance-explained shown; no claim that latent distances are exact.
- [ ] **No** "latent vs pixel compute" ratio is stated as a benchmark (there is no pixel-rollout baseline). "Plans in latent space on a laptop at ~6 ms/step" is fine (measured); "100× cheaper than pixels" is **forbidden**.
- [ ] **No** secure-aggregation, differential-privacy, cryptographic-proof, paper-scale-performance, or closed-loop-robot claims.
- [ ] Planning latency/success are presented as **measured on CPU via the exported graphs**, with trial counts.
- [ ] Every on-screen number traces to `docs/evidence/lewm_tworooms_manifold.json`.

## 3. Approved one-liners (safe to say on stage / X)

- "A JEPA world model that plans in a 192-dimensional latent space — on a laptop."
- "A room of people improved its predictions by training a tiny adapter on their own private data; nobody shared a frame."
- "Held-out prediction error dropped ~12% after federated adapter continuation — system-composed, seed-robust."
- "Toggle the anti-collapse structure off and planning dies — keeping that structure is the hard part."
- "Everything on screen is backed by a generated evidence file."

## 4. Forbidden phrasings (do not say)

- ✗ "We federated-trained a world model." → it's adapter continuation on a frozen model.
- ✗ "100× cheaper than Sora/Genie / than pixel world models." → no such benchmark was run.
- ✗ "This is a collapsed model we trained." → it's synthetic.
- ✗ "Private and secure by cryptographic proof / differential privacy." → not wired in this path.
- ✗ "Beats local-only / paper-scale results." → not claimed by this project.

## 5. Definition of done — claims
- [ ] Checklist §2 fully ticked.
- [ ] `tests/ml/test_lewm_manifold.py` asserts the mandatory `nonClaims` negations.
- [ ] A second person (or a re-read pass) reviews the X post against §3/§4 before posting.
