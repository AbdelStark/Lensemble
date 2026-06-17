# Cartographer — Technical Architecture

This document is the implementation reference. Every external API below was verified by reading the source during planning; file:line citations are included so an implementer (or Codex) can jump straight to the call site. Where something **does not exist yet**, it is marked **[BUILD]**.

Conventions (AGENTS.md / CONTRIBUTING.md):
- Python package code → `lensemble/`; evidence-producer scripts → `scripts/`; evidence JSON → `docs/evidence/`; web demos → `web/<name>/` (self-contained); vendored JS → `web/<name>/vendor/`; tests → `tests/{unit,ml,integration,...}/`.
- Tensors serialize via `safetensors`, never `pickle`/`torch.load`. New code: `ruff` + `pyright` clean. Public-surface schema changes get a `CHANGELOG.md` entry under `## [Unreleased]`.
- Use `uv run …`. Narrowest test gate first.

---

## 0′. Corrections baked in (verified against source 2026-06-17)

The 16-agent ground-truth pass found these in the first draft; they are fixed below.

| # | Was | Verified truth | Source |
|---|---|---|---|
| K1 | "torch `encode_frames` may expect already-normalized input — verify" | **Torch `encode_frames` does NOT normalize.** ImageNet z-score lives only in the ONNX `EncoderGraph`. The harvest must apply `(px/255 → HWC→CHW → (·−mean)/std)` in Python (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]). Measured: L2(raw vs normed) = **30.28** — skipping it gives off-distribution latents and eff_rank/std sanity fails. | `lewm_tworooms.py:557-563`; `lewm_export.py:73-76` |
| K2 | (unstated) | **Torch `encode_actions` does NOT z-score either.** Action z-score is baked only in the ONNX `ActionGraph`; the harvest/MPC must tile the 2-DOF z-score (mean=[0.00309341,−0.05298233], std=[0.86747581,0.86776555]) before `encode_actions`. Stats: `docs/evidence/lewm_tworooms_action_stats.json`. | `lewm_tworooms.py:565-567`; `lewm_export.py:108-110` |
| K3 | point cloud and pre/post toggle are one population | **Two different latent populations.** CART-1 harvest = **encoder CLS** latents (the manifold shape). The adapter operates on **predictor outputs** `z_hat` (= `pairs.x`, frozen-predictor-output). So the pre/post clouds must be **predictor-output pairs**, not the encoder-CLS cloud. Keep them separate in `manifold.json` (point cloud vs pre/post). | `lewm_probe.mjs:51-56` |
| K4 | `adapterFromInitAndOffset(offset)` | Object arg: `adapterFromInitAndOffset({inputDim:192, hiddenDim:32, initSeed:42, offset})`; `adapterForward(adapter,x,n)→{y,h}` batched. `createAdapter` default `seed=1`; init is Gaussian × 1/√inputDim (fan-in), **not** Xavier. | `lewm_adapter.mjs:151,208,231` |
| K5 | `run_system_composed_probe(participants, …)` returns the offset | Keyword-only (`seed` required); **does not return the offset** — read `service.model_revision(run_id, final_id)["adapterState"]` (len 12512), never write it into a `demo-evidence/1` bundle (substring forbidden). | `system_probe.py:119,214` |
| K6 | `build_probe_split(h5_path, *, …)` | Fully keyword-only: `build_probe_split(*, h5_path, model_dir=DEFAULT_MODEL_DIR, seed, participants=2, episodes_per_participant=8, validation_episodes=4)`; **`seed` required**. | `lewm_tworooms_probe_pairs.py:74` |
| K7 | cost via `model.goal_cost` | Cost = **accumulated L1** to goal (`(latent−goal).abs().sum(-1)`, `mpc.py:175`). `model.goal_cost` is squared-L2 **terminal-only** — wrong on two axes. The instrumented planner keeps a **dual ring buffer** (latent + action-embedding, each ≤3; `predict` raises `ContractViolation` if T>3). | `mpc.py:160-176`; `lewm_tworooms.py:479,597` |
| K8 | ablation `naive-fedavg` at `ablation.py:120` | It is a `RungSpec('naive-fedavg', lambda_sig=0.0, lambda_anc=0.0, …)` **config entry at line 121**, not a collapse generator. | `eval/ablation.py:121` |
| K9 | `onnxruntime` available | **Absent** from the uv venv (torch 2.12.0, h5py 3.16.0 present). CART-1/2/3 run on the **torch** model (`load_tworooms_model`), so both normalizations are applied in Python (K1/K2). Re-bake needs `uv run --with onnxruntime …`. | venv probe |

---

## 0. Model facts (verified)

`lensemble/model/lewm_tworooms.py` — `LeWMTwoRooms`, config `LeWMTwoRoomsConfig`:

| Field | Value |
|---|---|
| `hidden_dim` (latent dim **D**) | **192** (CLS token only) |
| `image_size` / `patch_size` / `num_patches` | 224 / 14 / 256 |
| `pred_num_frames` (history window) | **3** |
| `action_input_dim` | **10** (5 env-steps × 2-DOF, flattened) |
| `action_emb_dim` | 192 |

Forward surface (all verified, `lewm_tworooms.py`):
- `encode_frames(pixels) -> (B,T,192)` — pixels `(B,T,3,224,224)`; uses CLS token `[:,0]` then `projector` (BN-MLP). (line ~557)
- `encode_actions(actions) -> (B,T,192)` — actions `(B,T,10)`. (line ~565)
- `predict(emb, act_emb) -> (B,T,192)` — teacher-forced, `T ≤ 3`. (line ~569)
- `rollout(emb_init, act_emb) -> (B, H+n_steps+1, 192)` — `@torch.no_grad()`, sliding 3-window. (line ~575-592)
- `goal_cost(predicted, goal_emb) -> (S,)` — **terminal MSE** (note: planner uses L1; pick one — Decision D4 says L1). (line ~594)

Load: `lensemble/model/lewm_checkpoint.py` →
`resolve_checkpoint(local_dir=None, revision=TWOROOMS_PINNED_REVISION, repo_id=TWOROOMS_REPO_ID, claim_grade=True)` (line ~75) then
`load_tworooms_model(ckpt) -> (model.eval(), upstream_config)` (line ~133). Pinned revision `77adaae0bc31deab21c93740d1f8bb947cd0bdec`, weights sha `566f2236…0f7dd`.

> ⚠️ The pinned upstream checkpoint uses the OLD transformers ViT key schema; in-tree reconstruction `lewm_tworooms.py` is parity ≤2e-5. Don't "upgrade" transformers in the bake env.

---

## 1. WS1 — Latent harvest (`CART-1`)

**Goal:** real episodes → 192-d CLS latents + actions + per-step metadata, dumped to an intermediate artifact the projection/MPC stages consume.

**Reuse:** `lensemble/eval/lewm_tworooms_probe_pairs.py`:
- `build_probe_split(h5_path, *, participants, episodes_per_participant=8, validation_episodes=4, ...)` (line ~74) — reads `pixels`, `action`, `ep_offset`, `ep_len` from the H5 with `h5py`+`hdf5plugin`. It builds *probe pairs* through the **ONNX** graphs; we want raw **latents per step**, so factor out / mirror its episode-reading loop.

**[BUILD]** `lensemble/eval/manifold_harvest.py`:
```python
def harvest_latents(
    h5_path: Path, *, model: LeWMTwoRooms, num_episodes: int, max_steps_per_episode: int,
    seed: int,
) -> HarvestResult:
    """Encode real episode frames to (N,192) latents with frozen encoder; keep actions and
    (episode_id, step) provenance. Returns latents float32 (N,192), actions (N,10),
    episode_ids (N,), step_idx (N,), and the true (x,y) agent state per step if available
    in the H5 (for state_probe_r2)."""
```
Notes:
- One NaN action row per episode (terminal step) — drop it (memory: dataset quirk).
- **Normalization (RESOLVED — K1/K2, load-bearing):** the torch `encode_frames`/`encode_actions` do **no** normalization (it lives in the ONNX graphs only). So the harvest must, in Python: pixels uint8 → `/255` → HWC→CHW → ImageNet `(·−mean)/std` (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`) before `encode_frames`; and actions → tile the 2-DOF z-score (`mean=[0.00309341,−0.05298233]`, `std=[0.86747581,0.86776555]` from `docs/evidence/lewm_tworooms_action_stats.json`) before `encode_actions`. Skipping the pixel norm shifts each latent by L2≈30.28 and breaks the eff_rank≈9.86 / latent_std≈0.904 sanity. `state_probe_r2` labels = `pos_agent[idx]` (the H5 has it — resolves the "if available" ambiguity).
- Harvest **a few thousand** latents (e.g. 40 episodes × ~50 steps ≈ 2k points) — enough for a dense cloud, small enough for instant PCA and a small JSON.

**Artifact:** `runs/cartographer/harvest-<hash>.npz` (latents, actions, episode_ids, step_idx, states). `safetensors`/`npz` only — never pickle.

**Acceptance:** deterministic given seed; `effective_rank(latents)` ≈ 9–10 (sanity vs the certified 9.86); shape asserts.

---

## 2. WS2 — Instrumented MPC (`CART-2`)

**Goal:** expose the planning *search*, not just the winning plan, over LeWMTwoRooms dynamics.

**Two problems with stock `lensemble/eval/mpc.py` (verified):**
1. `Planner.plan(dynamics, initial_latent, goal_latent) -> PlanResult` returns only `actions,(horizon,A)`, `cost`, `planner`, `num_samples`, `num_iters`, `wall_time_s` (dataclass line ~34). The `(N,H,d)` candidate latent trajectories computed in `_rollout_costs` (line ~160-176) are **summed into scalar costs and discarded**.
2. The `dynamics` contract is `(latents (N,d), actions (N,A)) -> next (N,d)` for the harness's generic `Predictor` (spatial tokens, single-step). **LeWMTwoRooms is temporal**: `predict` needs a `(N, T≤3, 192)` history. A single-step CEM call must maintain a **3-frame ring buffer per candidate**.

**[BUILD]** `lensemble/eval/mpc_instrumented.py`:
```python
@dataclass(frozen=True)
class InstrumentedPlanResult(PlanResult):     # extend, keep base fields
    candidate_latents: Tensor   # (num_iters, N, H, d)  predicted latent trajectories
    candidate_costs: Tensor     # (num_iters, N)
    elite_mask: Tensor          # (num_iters, N) bool
    chosen_latents: Tensor      # (H, d) the winning rollout's latent trajectory

class InstrumentedPlanner(Planner):
    def plan_with_candidates(self, dynamics, initial_latent, goal_latent) -> InstrumentedPlanResult: ...

def lewm_dynamics(model: LeWMTwoRooms) -> Callable:
    """Return a single-step dynamics callable that internally keeps a 3-frame latent history
    ring buffer per candidate, calls model.predict on the truncated window, and returns the
    next latent (N,192)."""
```
- Cost: accumulated **L1** to `goal_latent` over the horizon (mirror `_rollout_costs` line ~175: `(latent-goal).abs().sum(-1)`), Decision D4.
- Goal latent: encode a goal frame (an agent rendered at a target position — pull a successful episode's terminal frame from the harvest, or the dataset). `(192,)`.
- `family="icem"`, modest `horizon` (e.g. 8–12), `num_samples` 64–256, `num_iters` 4. All CPU-cheap (predictor 0.8 ms/step from the spike).

**Artifact:** a **plan trace** = a small set (1–3) of plans, each with: chosen trajectory latents, a subsample of candidate trajectories per iteration (cap N rendered, e.g. 24/iter to keep JSON small — **log the cap**, Decision D-noSilentCaps), costs, elite flags, goal latent, `wall_time_s`.

**Acceptance:** `plan_with_candidates` reaches a goal latent with monotone-improving best-cost across iterations on a held-out start/goal; candidate tensor shapes correct; a unit test on a toy linear dynamics confirms branching capture.

---

## 3. WS3 — Federated before/after (`CART-3`)

**Goal:** one demo-path adapter round → an offset → pre/post-federation predicted latents, with the certified improvement number.

**Reuse (verified):**
- `lensemble/demo/system_probe.run_system_composed_probe(participants, validation, checkpoint, manifest, rounds=3, steps_per_round=20, batch_size=32, seed=20260612, dim=192)` drives the **real** node-trained adapter through `FederatedDemoService.submit_update` → `_close_round_lewm` and scores the server-produced revision. `write_evidence(path, evidence)` (system_probe.py ~273).
- Or the lower-level `FederatedDemoService` sequence (federated.py): `create_run({mode: REAL_LEWM_MODE, rounds, quorum})` → `join_run` ×N → `start_run` → per participant `update_progress` + `submit_update(artifact)` → `model_revision(run_id, rev)["adapterState"]` (list[float], len 12512).
- Adapter math (`web/federated-demo/lewm_adapter.mjs`): `y = z_hat + W2·tanh(W1·z_hat + b1) + b2`; **W2,b2 zero-init, W1 = Gaussian × 1/√inputDim (LeCun fan-in, NOT Xavier), via mulberry32(initSeed=42)**. Server stores cumulative **offset** from that shared init. `adapterFromInitAndOffset({inputDim:192, hiddenDim:32, initSeed:42, offset})` (line 208, object arg) reconstructs `init + offset`. ⚠️ The adapter operates on **predictor outputs** `z_hat` (= `pairs.x`), not the harvested encoder-CLS latents — keep the pre/post clouds (predictor-output pairs) separate from the point cloud (encoder CLS), per K3.

**[BUILD]** `lensemble/eval/manifold_federation.py`:
```python
def apply_adapter_offset(z_hat: Tensor, offset: np.ndarray) -> Tensor:
    """Python port of lewm_adapter.mjs adapterForward, including the mulberry32 seed-42 W1 init.
    Unpack offset (12512,) -> w1(32,192),b1(32),w2(192,32),b2(192); add to init; forward."""
```
> ⚠️ **W1 has a non-zero init** (seed-42 Xavier via mulberry32). To apply the post-federation adapter in Python you must replicate mulberry32 **exactly** (small, deterministic) OR drive `web/federated-demo/lewm_system_round.mjs op=probe` via a node bridge (already used by `system_probe.py`). The node bridge is the lower-risk path; the Python port is cleaner for the bake. Pick per CART-3; default = **node bridge** (proven), Python port = stretch.

**Pre vs post latents:** take harvested latents `z_hat` (frozen predictor outputs on held-out pairs), produce `z_pre = z_hat` (identity adapter) and `z_post = apply_adapter_offset(z_hat, offset)`. These two clouds feed projection (WS4).

**Number:** **do not recompute** — read `relativeImprovement` from the system probe evidence (committed +0.1227; seed-mean +0.168 from `lewm_tworooms_probe_seedsweep.json`). Decision D3.

**Acceptance:** offset length 12512; `apply_adapter_offset` parity vs the JS `probe` op within 1e-3 on a fixture (the repo already verifies l2Norm to 1e-3); post-federation held-out MSE < pre (sign check).

---

## 4. WS4 — Projection + gauge alignment + metrics (`CART-4`)

**Goal:** stable, comparable 3D coordinates for pre/post/collapsed clouds + structure metrics.

**Reuse (verified):**
- `lensemble/gauge/procrustes.procrustes_align(source, target, *, singular_floor=1e-6) -> (Q (d,d) in SO(d), residual float)` (line ~28). Requires **same n** and a shared landmark set. Raises `DegenerateProcrustes`.
- `lensemble/eval/jepa_metrics.effective_rank(emb) -> float` (line ~59, entropy of covariance spectrum).
- `lensemble/eval/metrics.effective_dim(emb) -> float` (line ~96, participation ratio), `covariance_eigenvalues(centered) -> Tensor` (line ~36), `state_probe_r2(train_x,train_y,test_x,test_y) -> float` (line ~172).
- `lensemble/model/sigreg.sigreg_statistic(emb, sketch, *, ep_knots=17)` (line ~40) + `build_sketch(seed, d, sketch_dim=64)` (line ~27).

**[BUILD]** `lensemble/eval/manifold_projection.py`:
```python
def project_3d(latents: np.ndarray, *, basis: np.ndarray | None = None) -> tuple[coords3d, basis, variance_explained]:
    """Deterministic PCA. If basis given (the post-federation basis), reuse it so pre/post
    share axes. Center with the SAME mean. Return (N,3) coords + the 3-component basis + per-axis
    variance ratio."""

def align_clouds(pre, post, landmarks_idx) -> Q:
    """procrustes_align(pre[landmarks], post[landmarks]) -> Q; apply pre @ Q before projection."""

def structure_metrics(latents, *, seed) -> dict:  # effective_rank, effective_dim, sigreg_statistic, latent_std_mean
```
**Projection policy (Decision D5):** fit PCA basis on the **post-federation** cloud; project pre (after Procrustes `Q`) and collapsed into the **same** basis with the **same** center, so the toggles are geometrically comparable. Record `variance_explained` (PCA to 3D on a ~10-eff-rank cloud should capture a healthy fraction — display it honestly).

**Acceptance:** projecting a cloud into its own basis reproduces top-3 PCA; Procrustes residual reported; metrics match the certified ranges (healthy eff_rank ≈ 9.86, eff_rank_ratio ≈ 0.051).

---

## 5. WS5 — Collapse counterfactual (`CART-5`)

**Goal:** an honest "healthy vs collapsed" contrast. No collapse generator exists (verified).

**[BUILD]** in `manifold_projection.py` (or `manifold_collapse.py`):
```python
def synth_collapsed(reference: np.ndarray, *, mode: Literal["rank1","magnitude"], seed) -> np.ndarray:
    # rank1:      scales(n,1) * direction(d)         -> effective_dim ≈ 1.0
    # magnitude:  1e-6 * randn(reference.shape)      -> eff_rank high, magnitude ≈ 0 (the eff_rank blind spot)
```
Both patterns are taken from existing repo code (the spike `run_spike.py:383` magnitude pattern; test files' rank-1 pattern). **Label it "synthetic illustration"** in the JSON and on screen (Decision D6, claim discipline). Compute `effective_rank`/`effective_dim` on both real and collapsed so the contrast is a real measurement (healthy ≈ 9.86 vs rank-1 ≈ 1.0).

Stretch: use the ablation ladder's `naive-fedavg` rung (`eval/ablation.py:120`) for an *organically* collapsed model — more honest, heavier; only if time.

**Acceptance:** `effective_dim(rank1) ∈ [1.0, 1.5]`; magnitude cloud has near-zero std with non-trivial rank; both projected into the healthy basis for the toggle.

---

## 6. WS6 — Evidence JSON (`CART-6`)

**Goal:** back every on-screen number with a generated, tested evidence file (AGENTS.md "Claim Discipline" — every public number traces to a committed evidence artifact; AGENTS.md has no numbered sections).

**Schema `lewm-manifold/1`** (full shape in doc `02`). Producer **[BUILD]** `scripts/lewm_manifold_check.py` → `docs/evidence/lewm_tworooms_manifold.json`, writing via the existing `system_probe.write_evidence` helper or `Path.write_text(json.dumps(..., indent=2)+"\n")`.

Required fields: `schema`, `seed`, `checkpoint{repoId,revision,weightsSha256}`, `result{ planningLatencyMsPerStep, plannerWallTimeS, goalReachSuccessRate, federatedRelativeImprovement, effectiveRankHealthy, effectiveRankCollapsed, effectiveDimHealthy, effectiveDimCollapsed, pcaVarianceExplained3d }`, `passes: bool`, `nonClaims: [str]`, `provenance{ exportGraphHashes }`.

`nonClaims` MUST include (claim discipline, doc `05`):
- "Federated result is adapter continuation on a frozen checkpoint, not federated world-model training."
- "Collapsed cloud is a synthetic illustration, not a trained model."
- "3D layout is a PCA projection of a ~192-dim latent space; distances are approximate."
- "No latent-vs-pixel compute comparison is claimed."

**Test [BUILD]** `tests/ml/test_lewm_manifold.py`: schema string, presence + types of all result fields, `nonClaims` contains the required negations, `passes` truthiness on a small synthetic fixture. Wire into gate 4 (`pytest tests/ml`).

**Acceptance:** producer is deterministic; test green; `check_docs_links.py` passes after referencing the file from `docs/roadmap` + this plan.

---

## 7. WS7 — Bake orchestrator (`CART-7`)

**Goal:** one command → the viewer's `manifold.json`, deterministic + hash-stamped.

**[BUILD]** `scripts/cartographer/bake.py` (or `lensemble/eval/manifold_bake.py` + a thin script):
```
load frozen model
  → harvest_latents (WS1)
  → run/load federation offset (WS3)  → pre/post predicted latents
  → instrumented plans (WS2)          → plan traces (goal latents from harvest)
  → synth_collapsed (WS5)
  → align + project all clouds into the shared post-fed PCA basis (WS4)
  → assemble manifold.json per doc 02 (downsample point counts to the contract caps; LOG caps)
  → stamp: checkpoint revision/sha, export-graph hashes, bake seed, git SHA, schema version
  → write web/latent-manifold-viewer/data/manifold.json
  → also write the evidence JSON (WS6) so numbers stay in sync
```
Determinism: fixed seeds throughout; no `Date.now()`/`random` without seed. Size budget: target `manifold.json` ≤ ~3–5 MB (downsample points/candidates; the contract caps are in doc `02`).

**Acceptance:** two runs with the same seed produce byte-identical `manifold.json` (modulo the git-SHA stamp); size within budget; validates against the contract.

---

## 8. WS8 — WebGPU viewer (`CART-8`)

**Goal:** a self-contained page that renders the manifold beautifully and never hangs.

**Template:** copy `web/dynamic-env-demo/` structure (self-contained, inline CSS, WebGPU EP already wired via the CDN `ort.webgpu.min.js` tag — though Cartographer's default needs **no ONNX**, only Three.js).

**[BUILD]** `web/latent-manifold-viewer/`:
- `index.html` — inline CSS; `<script type="module" src="./app.mjs">`.
- `vendor/three.module.min.js` — **vendor manually** (nothing is vendored in-repo except `qrcode.mjs`); import `import * as THREE from "./vendor/three.module.min.js"`.
- `app.mjs` — `fetch("./data/manifold.json")` → build scene:
  - **Point cloud**: `THREE.Points` from `clouds.post.coords3d`, colored by `episode_id` or local density; slow auto-rotate.
  - **MPC plan trails**: `THREE.Line` per chosen plan toward the goal node; candidate trajectories as faint lines, elites brighter; optional animated "ignition" along the chosen path.
  - **Toggles**: healthy↔collapsed (swap point positions/colors), pre↔post-federation (swap to `clouds.pre.coords3d`), with a short tween.
  - **Metrics HUD**: eff_rank healthy vs collapsed, federated improvement, planning latency, PCA variance-explained — all read from `manifold.json.metrics`, each labelled, each tagged with its evidence provenance.
  - **Provenance card**: checkpoint revision (short), export-graph hash (short), bake seed, schema — from `manifold.json.provenance`.
  - **nonClaims footer**: render `manifold.json.nonClaims` verbatim (claim discipline on the *artifact itself*).

Served at `http://127.0.0.1:8765/web/latent-manifold-viewer/` by the existing static server (`lensemble/demo/server.py` `_static`, serves any path under `web/`) — **zero server changes**. Launch via the existing `uv run lensemble demo federated --port 8765` (it serves the whole `web/` tree), or any static server.

**Acceptance:** renders `manifold.json` on the demo laptop in Chrome (WebGPU) at smooth framerate; all toggles work; HUD numbers equal the evidence JSON; degrades to WASM/Canvas if WebGPU absent (Three.js WebGLRenderer is the safe default — only use WebGPURenderer if it's robust on the demo machine, else WebGL).

> Decision: use Three.js **WebGLRenderer** by default (universally supported, "WebGPU allowed" ≠ "WebGPU required"); WebGPURenderer is a stretch toggle. A few-thousand-point cloud + a few hundred lines is trivial for WebGL.

---

## 9. WS9 — Rehearsal + fallback + capture (`CART-9`)

- **Rehearsal gate [BUILD]** `scripts/cartographer/rehearsal.py` (mirror `scripts/hackathon_demo_rehearsal.py` style): asserts `manifold.json` validates, the viewer's required fields exist, the evidence JSON passes its test, and the live federation round (Act 1) completes headlessly. Deterministic, CI-shaped.
- **Fallback bundle:** a committed `manifold.json` baked pre-event so the viewer always works even if the day's bake fails (Decision D8).
- **Capture:** a ≤20 s screen recording (rotate → play a plan → toggle collapse → toggle pre/post) + a result card image. Producer notes in runsheet doc `04`.

---

## 10. External APIs touched (quick index)

| Need | Symbol | File:line |
|---|---|---|
| Load checkpoint | `resolve_checkpoint`, `load_tworooms_model` | `model/lewm_checkpoint.py:75,133` |
| Encode/predict/rollout | `encode_frames`,`encode_actions`,`predict`,`rollout`,`goal_cost` | `model/lewm_tworooms.py:557,565,569,575,594` |
| Episode reader | `build_probe_split` | `eval/lewm_tworooms_probe_pairs.py:74` |
| Planner (to extend) | `Planner.plan`, `_rollout_costs`, `PlanResult` | `eval/mpc.py:97,160,34` |
| Federated round | `run_system_composed_probe`, `write_evidence` | `demo/system_probe.py` |
| Demo service | `FederatedDemoService.{create_run,join_run,start_run,submit_update,model_revision,export_evidence}` | `demo/federated.py` |
| Adapter math (JS) | `adapterForward`, `adapterFromInitAndOffset` | `web/federated-demo/lewm_adapter.mjs:231,208` |
| Procrustes | `procrustes_align` | `gauge/procrustes.py:28` |
| Metrics | `effective_rank` / `effective_dim` / `covariance_eigenvalues` / `state_probe_r2` | `eval/jepa_metrics.py:59`, `eval/metrics.py:96,36,172` |
| SIGReg | `sigreg_statistic`, `build_sketch` | `model/sigreg.py:40,27` |
| Static serving | `_static`, `WEB_ROOT` | `demo/server.py:560,27` |
| Evidence audit (real-mode only) | `audit_real_lewm_evidence` | `demo/evidence_audit.py:61` |
