# Sovereign Economy - Risks And Claim Discipline

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Narrative over-names a company or incident | Credibility/legal risk | Use placeholders unless separately verified and approved. |
| Payment integration leaks API key to browser | Severe security failure | Server-side SDK only; secret scan; `.env` ignored. |
| Mollie test amount rejects EUR 1,000,000 | Demo interruption | Separate simulated sale amount from smaller test checkout amount. |
| Reward split sounds like real payout | Regulatory/credibility risk | Label as simulation; no legal payout or securities language. |
| Economics distracts from model proof | Weak demo | Keep surprise-meter beside economics panel; buyer pays for a measured revision. |
| Ledger math does not balance | Trust failure | Unit-test rounding and totals. |
| Live Mollie network fails | Stage risk | Mock checkout is a first-class rung. |
| Full-model training language reappears | Scientific overclaim | Keep adapter-continuation wording on every public surface. |

## Mandatory Non-Claims

- This is not a legal payout, wage, security, or investment contract.
- This is not Mollie Connect/submerchant routing in the first version.
- This is not full federated world-model training.
- This demo path does not wire secure aggregation or differential privacy.
- A humanoid robotics buyer is a placeholder.
- Frontier model access restrictions are discussed as a general risk, not as a
  claim about a named company or incident.

## Approved Phrases

- "A community federated run for a robotics world-model revision."
- "Participants keep raw trajectories local and submit bounded adapter deltas."
- "A simulated buyer pays through a Mollie test checkout."
- "The reward split is a demo ledger, not a legal payout."
- "The surprise meter is the model-quality proof: held-out prediction error
  drops after adapter continuation."

## Forbidden Phrases

- "We federated-trained the full world model."
- "Private by differential privacy" or "cryptographically guaranteed."
- "Participants are paid real revenue."
- "This proves humanoid robot performance."
- Named-company incident claims without verification.
- "Per-pixel surprise heatmap."
