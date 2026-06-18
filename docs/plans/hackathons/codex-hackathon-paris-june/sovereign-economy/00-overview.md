# Sovereign Economy - Overview

## One-Sentence Pitch

L'Ensemble Labs coordinates a community federated run so people can improve a
robotics world model with local data and compute, prove the improvement with a
surprise meter, sell the resulting revision to a simulated humanoid robotics
buyer, and split the test-mode economics by auditable contribution.

## Narrative Arc

1. **Problem:** intelligence infrastructure is concentrated. Frontier model
   access restrictions are a sovereignty risk for individuals, Europe, and
   nation-states.
2. **Mechanism:** federated runs make model improvement more accessible:
   participants keep local trajectory data resident while contributing compute
   and bounded updates.
3. **Proof:** the TwoRooms surprise meter shows the model is less surprised
   after federated adapter continuation on a frozen checkpoint.
4. **Economy:** a humanoid robotics buyer pays for the improved world-model
   revision through Mollie test checkout; L'Ensemble Labs keeps an orchestrator
   share; participants receive simulated rewards by ledger weight.
5. **Close:** world models are a credible robotics "ChatGPT moment" substrate
   because they predict dynamics and make future-state error inspectable; the
   opportunity is to build them with sovereignty and shared upside.

## Scope

In scope:

- New parent tracker [#359](https://github.com/AbdelStark/Lensemble/issues/359).
- Existing surprise-meter track [#338](https://github.com/AbdelStark/Lensemble/issues/338)
  and children [#349](https://github.com/AbdelStark/Lensemble/issues/349)-[#354](https://github.com/AbdelStark/Lensemble/issues/354).
- Deterministic contribution ledger and reward split.
- Mollie test checkout/payment-link creation from the server side.
- Integrated buyer/orchestrator/participant dashboard.
- Demo script, fallback ladder, and autonomous validation matrix.

Out of scope for the first shippable version:

- Real legal payouts or payout onboarding.
- Mollie Connect/submerchant routing.
- Payment webhooks as a hard dependency.
- Secure aggregation or differential privacy in the demo coordinator path.
- Full federated world-model training; the demo is adapter continuation on a
  frozen checkpoint.

## Priority Ladder

| Priority | Work | Issue | Demo value |
|---|---|---|---|
| 0 | Narrative and claim-safe deck arc | [#364](https://github.com/AbdelStark/Lensemble/issues/364) | Makes the new story coherent. |
| 1 | Contribution ledger and reward split | [#361](https://github.com/AbdelStark/Lensemble/issues/361) | Makes the economy auditable without Mollie. |
| 2 | Mollie test checkout/payment links | [#360](https://github.com/AbdelStark/Lensemble/issues/360) | Makes monetization tangible. |
| 3 | Surprise-meter technical proof | [#338](https://github.com/AbdelStark/Lensemble/issues/338) | Shows model-quality progress. |
| 4 | Economics dashboard | [#363](https://github.com/AbdelStark/Lensemble/issues/363) | Joins buyer, orchestrator, and contributors. |
| 5 | Rehearsal and fallback pack | [#362](https://github.com/AbdelStark/Lensemble/issues/362) | Makes Demo Night survivable. |

## Default Demo Scenario

- Buyer offer: EUR 1,000,000.
- Orchestrator share: 20%.
- Community pool: 80%.
- Participants: 4 demo participants with accepted rounds, sample counts, local
  steps/runtime proxy, and update health.
- Reward split: proportional by normalized contribution weight.
- Payment: Mollie test checkout URL if `MOLLIE_API_KEY` is present; deterministic
  mock checkout URL otherwise.

## Completion Bar

The parent is hackathon-ready when a fresh agent can start from `TODAY.md`,
understand the narrative, pick a child issue, know the exact validation gate,
and avoid public overclaims without asking for missing context.
