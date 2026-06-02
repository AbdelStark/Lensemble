# RFC-0002 — The Latent Gauge & Frame-Anchored Aggregation

| | |
|---|---|
| **RFC** | 0002 |
| **Title** | The Latent Gauge & Frame-Anchored Aggregation |
| **Status** | Draft |
| **Track** | Standards |
| **Author** | Abdelhamid Bakhta (@AbdelStark) |
| **Requires** | RFC-0001 |
| **Date** | June 2026 |

> This is the scientific core of Lensemble. It states the one problem that makes federated end-to-end JEPA genuinely hard, proves why the standard fix does not apply, and specifies the solution.

## 1. The problem in one sentence

The SIGReg-JEPA objective is invariant under rotations of the latent space, so independently-updated participants converge to mutually-rotated coordinate frames, and averaging their weights in parameter space is meaningless.

## 2. Why JEPA federation ≠ LLM federation

Federated/averaged training of supervised models and LLMs is approximately well-posed because the loss depends on a **fixed output basis** — class labels, or a softmax over a fixed vocabulary. That basis breaks the symmetry of the representation and pins a shared coordinate frame, so client weight matrices are comparable and `FedAvg` / DiLoCo outer-averaging behaves.

End-to-end JEPA has **no fixed output basis**: the target is another embedding produced by the (same) network, and the regularizer's target — an isotropic Gaussian — is itself basis-free. Nothing pins the frame.

## 3. Background: the objective

Per LeJEPA / LeWM, each participant minimizes

$$\mathcal{L} = \underbrace{\mathbb{E}\,\lVert g_\phi(f_\theta(x_t),\,a_t) - \text{sg}[f_\theta(x_{t+1})]\rVert^2}_{\text{next-embedding prediction}} \;+\; \lambda_{\text{sig}}\,\mathrm{SIGReg}\big(f_\theta(x)\big),$$

where **SIGReg** pushes the embedding marginal toward $\mathcal{N}(0,I_d)$ by projecting embeddings onto random 1-D directions and matching each univariate marginal to a standard Gaussian via the Epps–Pulley characteristic-function statistic (Cramér–Wold). SIGReg is what removes the EMA target, stop-gradient, and teacher–student machinery — which is exactly why it is federation-friendly (no momentum-encoder state to reconcile). It is also the source of the gauge.

## 4. The $O(d)$ gauge, formally

Let $Q \in O(d)$. Transform the encoder and conjugate the predictor:

$$f_\theta \;\mapsto\; Q f_\theta, \qquad g_\phi \;\mapsto\; Q\, g_\phi\, Q^{\top}.$$

Then:

- **Prediction loss is unchanged** — it depends only on inner products / Euclidean distances, which $Q$ preserves: $\lVert Q\hat{z} - Q z\rVert = \lVert \hat z - z\rVert$.
- **SIGReg is unchanged** — $\mathcal{N}(0,I)$ is invariant under $O(d)$; every projected marginal of $Qz$ is still standard normal.

So the **entire objective is invariant under a global $O(d)$ rotation**. Independently-updated participants therefore converge to frames related by arbitrary $Q_c$, and

$$\bar\theta=\tfrac{1}{C}\textstyle\sum_c \theta_c$$

averages weight matrices expressing features in mutually-rotated bases — producing neither participant's representation. This is the irreducible, JEPA-specific obstruction. (Neuron-permutation symmetries exist in any network and are handled by the same alignment machinery; the $O(d)$ latent rotation is the *extra* one that anchored models never see.)

### 4.0 Two failures that compound it

- **Non-IID marginals** — each silo's empirical embedding distribution differs, so SIGReg constrains a different marginal per participant.
- **Collapse re-entry** — averaging across frames can re-introduce the low-rank solutions SIGReg prevented locally.
- **DiLoCo drift** — the longer the inner horizon $H$, the further frames rotate apart before the outer step.

## 4.1 Layer 1 — shared sketch matrix (objective consistency, **not** the gauge fix)

All participants use the **same** random projection matrix $A$ each round (broadcast a seed). This ensures everyone minimizes the *identical* regularizer rather than $C$ different empirical objectives, and it reduces SIGReg's small-batch variance (which falls as batch grows; reduce the projection statistics *within* a participant's inner-parallel group freely — same trust domain).

**Important:** a shared sketch does **not** close the gauge. Matching an isotropic target along shared directions is still rotation-invariant. Layer 1 fixes consistency; the frame needs Layer 2.

## 4.2 Layer 2 — frame anchoring on a public probe (the gauge fix)

Two ingredients manufacture the anchor the objective otherwise lacks. The probe is **public** data, so nothing private leaves a boundary.

**(a) Shared warm-start.** Initialize every participant from the same released encoder ⇒ the gauge is **closed at $t{=}0$**; it can only re-open *during* federated fine-tuning. This is most of the battle.

**(b) A public probe with fixed targets.** Maintain a small fixed public probe set $\mathcal{P}$ (RFC-0004 §3). Fix reference embeddings $E_{\text{ref}}=f_{\text{ref}}(\mathcal{P})$ from the round-0 encoder. Two anchoring variants:

- **Variant A — landmark anchoring (recommended to start; simple, no differentiable SVD).** Choose $k \ge d$ generic landmark points with fixed absolute targets $t_i = f_{\text{ref}}(p_i)$. Add
  $$\mathcal{L}_{\text{anchor}} = \lambda_{\text{anc}}\,\tfrac{1}{k}\textstyle\sum_{i=1}^{k}\lVert f_\theta(p_i)-t_i\rVert^2 .$$
  Because $k\ge d$ generic absolute constraints admit only $Q=I$ as the satisfying orthogonal map, this **pins the frame** while leaving the representation of all other (probe and private) points free — *pin the frame, not the content*.

- **Variant B — rotational-drift penalty (principled; fuller frame control).** Decompose the probe drift into a *frame* part and a *content* part. Compute the optimal Procrustes rotation $Q^\star=\arg\min_{Q\in O(d)}\lVert f_\theta(\mathcal{P})Q-E_{\text{ref}}\rVert_F$ (closed form: $Q^\star=VU^\top$ from the SVD $E_{\text{ref}}^\top f_\theta(\mathcal{P})=U\Sigma V^\top$). Penalize **only the rotation**:
  $$\mathcal{L}_{\text{anchor}} = \lambda_{\text{anc}}\,\lVert Q^\star - I\rVert_F^2 .$$
  Gradients flow through the (differentiable) SVD into $f_\theta$, pushing the frame toward the reference while the content drift (the post-alignment residual) stays unpenalized. Mind near-degenerate singular values; clamp/condition the SVD.

Variant A is the safe default; Variant B is the cleaner statement of "pin frame, not content" when fuller control is wanted.

## 4.3 Layer 3 — Procrustes re-alignment at aggregation (backstop)

Immediately before each outer step, recompute the hard alignment $Q_c^\star$ on $\mathcal{P}$ for each participant and fold it into the encoder's terminal linear map (and conjugate the predictor I/O) before averaging. With Layer 2 active this should rarely bind. **Verifiability bonus:** alignment is a deterministic function of *public-probe data + committed weights*, so it is publicly recomputable — anyone can recompute and check it, and it needs **no** ZK proof (RFC-0006 §3).

## 4.4 Layer 4 — function-space distillation (fallback / heterogeneity)

If weight-space averaging is unstable at scale, or to admit participants with *different* encoder sizes, aggregate **behaviors** instead of weights: each participant emits predictions on $\mathcal{P}$; the coordinator forms a consensus (after frame alignment); a global student distills it. Gauge-invariant by construction (compares functions on shared inputs). Higher per-round cost; held in reserve.

## 4.5 The central hyperparameter

$\lambda_{\text{anc}}$ trades frame stability against representational freedom:

- too high → the encoder is clamped to the reference frame *and* its quality;
- too low → the frame drifts and averaging degrades.

Characterize it in Stage B (RFC-0005) by sweeping $\lambda_{\text{anc}}$ against (i) the frame-drift diagnostic and (ii) downstream MPC success. Hypothesis: **warm-start + a small $\lambda_{\text{anc}}$ keeps frames pinned cheaply**, so Layers 3–4 rarely fire.

## 5. The training algorithm (per outer round)

```
Given: global θ, φ; participants c = 1..C; public probe P, landmark targets {t_i};
       sketch seed s_t (→ projection matrix A); inner horizon H.
1. Broadcast θ, φ, s_t to all participants. Action heads h^(c) stay local.
2. Each participant c, in parallel (inner FSDP/TP):
     for H steps:  minimize  L_pred + λ_sig·SIGReg_A(f) + λ_anc·L_anchor(f; P, t_i)
                   on local data via AdamW.   # raw data never leaves
     form pseudo-gradient Δ_c = (θ_c, φ_c) − (θ, φ)
     DP: clip ‖Δ_c‖, add Gaussian noise               # RFC-0003 §4
3. Secure-aggregate Σ_c Δ_c  (individual Δ_c hidden).  # RFC-0003 §3
4. (Backstop) Procrustes-align participants on P if drift exceeds threshold.
5. Outer Nesterov step:  (θ, φ) ← (θ, φ) − η_out · OuterOpt( mean_c Δ_c )
6. Hash-commit θ^{global}_{t+1}.                        # RFC-0006 §4
```

## 6. What success looks like (pointer)

The headline empirical artifact is the **frame-drift diagnostic**: inter-participant Procrustes residual (or mean rotation angle) on $\mathcal{P}$ over training — naive `FedAvg` visibly diverges; the anchored design holds it pinned. This is itself novel (first measurement of latent frame-drift under federated JEPA). Full eval in RFC-0005.

## 7. Fork A fallback

If Fork-B gauge control proves unstable at foundation scale, freeze the warm-started encoder and federate only the predictor (Fork A, the V-JEPA-2-AC structure). The frozen shared encoder *is* a shared frame, so the gauge problem dissolves and Layers 2–4 become unnecessary. This sacrifices the end-to-end novelty but preserves the sovereignty story and is a safe degrade.
