# Sovereign Economy - Data Contracts

## `sovereign-sale/1`

```json
{
  "schema": "sovereign-sale/1",
  "saleId": "sale_demo_001",
  "runId": "run_...",
  "modelRevisionId": "rev_...",
  "buyer": {
    "kind": "humanoid-robotics-buyer",
    "displayName": "Humanoid robotics buyer"
  },
  "saleAmount": { "currency": "EUR", "value": "1000000.00" },
  "orchestratorShare": "0.20",
  "communityPool": { "currency": "EUR", "value": "800000.00" },
  "payment": {
    "provider": "mollie",
    "mode": "mock",
    "paymentId": "mock_tr_sale_demo_001",
    "status": "open",
    "checkoutUrl": "http://127.0.0.1:8765/web/surprise-meter/?mockPaid=sale_demo_001"
  },
  "ledger": { "schema": "sovereign-contribution-ledger/1" },
  "nonClaims": [
    "simulation-not-legal-payout",
    "adapter-continuation-not-full-model-training",
    "single-coordinator-no-secure-agg-or-dp",
    "buyer-is-placeholder"
  ]
}
```

## `sovereign-contribution-ledger/1`

```json
{
  "schema": "sovereign-contribution-ledger/1",
  "currency": "EUR",
  "orchestratorReward": { "value": "200000.00" },
  "participantRewards": [
    {
      "participantId": "p_alice",
      "displayName": "Alice",
      "acceptedRounds": 3,
      "sampleCount": 640,
      "localSteps": 180,
      "runtimeMs": 42100,
      "updateHealth": 0.98,
      "capFactor": 1.0,
      "contributionWeight": 627.2,
      "reward": { "value": "254912.34" }
    }
  ],
  "roundingRemainder": { "value": "0.02" },
  "provenance": {
    "evidenceFiles": [
      "docs/evidence/lewm_tworooms_system_probe.json",
      "docs/evidence/lewm_tworooms_probe_seedsweep.json"
    ],
    "parentIssue": 359
  }
}
```

## Reward Formula

```text
community_pool = sale_amount * (1 - orchestrator_share)
raw_weight_i = accepted_rounds_i * sample_count_i * quality_i * cap_factor_i
reward_i = community_pool * raw_weight_i / sum(raw_weight)
```

The exact implementation may tune `raw_weight_i`, but it must be deterministic,
documented, and tested. The first shippable path can use fixture participants
when no live run exists.

## Payment Modes

| Mode | When | Required env | Behavior |
|---|---|---|---|
| `mock` | Default, no credentials | none | Deterministic checkout URL and status transitions. |
| `mollie-test` | Credentialed smoke/demo | `MOLLIE_API_KEY` | Server creates a Mollie test payment and returns checkout URL. |
| `mollie-webhook` | Bonus | public HTTPS URL + webhook route | Reconciles status by webhook. |

## Validation Rules

- `saleAmount.value` and reward values are decimal strings with two cents.
- `0 <= orchestratorShare <= 1`.
- At least one participant has positive contribution weight.
- Sum of participant rewards plus orchestrator reward plus/minus documented
  rounding remainder equals sale amount.
- `nonClaims` contains the four mandatory negations above.
- Payment metadata never includes raw data, local samples, participant tokens,
  Mollie API key, or base model weights.
