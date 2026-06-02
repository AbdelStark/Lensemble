# RFC-0003 — Federated Training Protocol

| | |
|---|---|
| **RFC** | 0003 |
| **Title** | Federated Training Protocol |
| **Status** | Draft |
| **Track** | Standards |
| **Author** | Abdelhamid Bakhta (@AbdelStark) |
| **Requires** | RFC-0001, RFC-0002 |
| **Date** | June 2026 |

## 1. Scope

The operational protocol: how a round runs, what crosses a boundary, and the privacy/fault-tolerance machinery. The aggregation *semantics* (frame alignment, anchoring) are in RFC-0002; this RFC covers the *mechanics*.

## 2. Roles

- **Participant** $c$ — holds sovereign data; runs local training (intra-participant parallelism per RFC-0001 §4); emits pseudo-gradients.
- **Coordinator** — orchestrates rounds, holds the canonical global model, runs the outer optimizer. Untrusted with respect to raw data; treated as honest-but-curious for Phase 1 and as a proving target in Phase 2 (RFC-0006).
- **Secure aggregator** — computes the masked sum of deltas; may be the coordinator under a secure-aggregation protocol or a separate party.

## 3. Round structure (DiLoCo outer loop)

Per round $t$:

1. **Broadcast** global parameters $(\theta_t,\phi_t)$, the round sketch seed $s_t$, and the probe/landmark hashes. Per-embodiment action heads $h^{(c)}$ remain local and are never broadcast or aggregated.
2. **Local optimization** — each participant runs $H$ inner steps with **AdamW** on the objective of RFC-0002 §3, over local data only.
3. **Pseudo-gradient** — $\Delta_c = (\theta_c^{\text{local}},\phi_c^{\text{local}}) - (\theta_t,\phi_t)$. (DiLoCo treats the $H$-step local update as a single "gradient.")
4. **Privatize** — clip and noise $\Delta_c$ (§4).
5. **Secure-aggregate** — compute $\sum_c \Delta_c$ without revealing any individual $\Delta_c$ (§5).
6. **(Backstop) align** — Procrustes re-alignment on the public probe if drift exceeds threshold (RFC-0002 §4.3).
7. **Outer step** — **Nesterov momentum** on the averaged pseudo-gradient: $(\theta_{t+1},\phi_{t+1}) = (\theta_t,\phi_t) - \eta_{\text{out}}\,\mathrm{Nesterov}\big(\tfrac1C\sum_c\Delta_c\big)$.
8. **Commit** — hash-commit $(\theta_{t+1},\phi_{t+1})$ (RFC-0006 §4).

Sync frequency: communicate every $H$ inner steps (DiLoCo/INTELLECT use $H\!\approx\!500$ for LLMs). $H$ is tuned against frame drift (RFC-0002 §4.0): larger $H$ = cheaper communication but more drift to anchor.

## 4. Differential privacy

Per-participant, before release:

- **Clip**: $\Delta_c \leftarrow \Delta_c \cdot \min(1, C_{\text{clip}}/\lVert\Delta_c\rVert)$.
- **Noise**: add $\mathcal{N}(0,\sigma^2 C_{\text{clip}}^2 I)$, calibrated to a target $(\varepsilon,\delta)$ over the planned number of rounds.

**Interaction to tune (open):** DP noise on small predictor deltas interacts with SIGReg's variance and with the anchor term. Joint calibration of $(\sigma, \lambda_{\text{sig}}, \lambda_{\text{anc}}, C_{\text{clip}})$ is a Stage-B experiment, not a default.

## 5. Secure aggregation

The coordinator must learn only $\sum_c \Delta_c$, never an individual $\Delta_c$ (an individual update leaks more about a silo's data than the sum). Use pairwise-mask secure aggregation (Bonawitz-style) or a TEE-based aggregator. Dropout-robustness is required because participants may vanish mid-round (§6).

## 6. Heterogeneity & fault tolerance

- **Embodiment heterogeneity** — handled at the model level: shared encoder + predictor core federate; per-embodiment action encoders/heads stay local (RFC-0001 §3). The shared latent interface is the WMCP contract (RFC-0004 §6).
- **Compute heterogeneity & churn** — adopt INTELLECT-1/PRIME-style elasticity: an outer step proceeds with whatever participants are present; late/dropped participants are reconciled at the next round; live checkpoint recovery for rejoining. The outer optimizer is robust to a varying participant count (a known DiLoCo property).
- **Communication compression** — optional int8 quantization of pseudo-gradients (per INTELLECT-1's int8 all-reduce) to cut outer-step bandwidth; orthogonal to the gauge machinery.

## 7. Message summary

| Message | Direction | Contents | Protection |
|---|---|---|---|
| `RoundOpen` | coord → participant | $(\theta_t,\phi_t)$ ref/hash, $s_t$, probe hash, $H$ | integrity (hash) |
| `Update` | participant → aggregator | $\Delta_c$ | DP + secure-agg mask |
| `Commitment` | participant → coord | dataset Merkle root $R_c$ (RFC-0004 §4) | binding |
| `RoundClose` | coord → all | $(\theta_{t+1},\phi_{t+1})$ hash | integrity (hash) |

Raw observations, actions, and embeddings of private data appear in **no** message.

## 8. Reference parameters (starting points, to be tuned)

Inner optimizer AdamW · outer optimizer Nesterov SGD · $H \in [50,500]$ (smaller while characterizing drift) · sketch dim 64, ~17 Epps–Pulley integration knots (LeJEPA defaults) · $\lambda_{\text{anc}}$ small (RFC-0002 §4.5) · DP $(\varepsilon,\delta)$ per deployment policy.
