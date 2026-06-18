# Sovereign Robotics World-Model Economy (#359)

Parent tracker: [#359](https://github.com/AbdelStark/Lensemble/issues/359).
Technical surprise-meter child: [#338](https://github.com/AbdelStark/Lensemble/issues/338).

This is the active Codex Hackathon plan. It keeps the LeWM TwoRooms
surprise-meter foundation, but reframes the demo around sovereignty and a
test-mode economics loop: community participants improve a robotics world model
through a federated adapter-continuation round, a simulated humanoid robotics
buyer pays through a Mollie test checkout, and rewards are split by an auditable
contribution ledger.

## Read In Order

| Doc | Purpose |
|---|---|
| [`00-overview.md`](00-overview.md) | Narrative, scope, priority ladder, demo shape. |
| [`01-architecture.md`](01-architecture.md) | Backend, Mollie boundary, routes, UI integration, env contract. |
| [`02-data-contracts.md`](02-data-contracts.md) | Ledger, sale, payment, reward, and evidence schemas. |
| [`03-workstreams-and-issues.md`](03-workstreams-and-issues.md) | Issue graph, dependencies, autonomous gates. |
| [`04-demo-runsheet.md`](04-demo-runsheet.md) | Stage flow, fallback ladder, rehearsal checklist. |
| [`05-risks-and-claim-discipline.md`](05-risks-and-claim-discipline.md) | Risks, non-claims, forbidden phrasings. |

## Issue Graph

- [#364](https://github.com/AbdelStark/Lensemble/issues/364) - SE-1 narrative, deck arc, claim-safe copy.
- [#361](https://github.com/AbdelStark/Lensemble/issues/361) - SE-2 contribution ledger and reward contract.
- [#360](https://github.com/AbdelStark/Lensemble/issues/360) - SE-3 Mollie test checkout/payment links.
- [#338](https://github.com/AbdelStark/Lensemble/issues/338) - SM surprise-meter technical track.
- [#363](https://github.com/AbdelStark/Lensemble/issues/363) - SE-4 economics dashboard.
- [#362](https://github.com/AbdelStark/Lensemble/issues/362) - SE-5 integrated rehearsal and validation.

Critical path: **#364 -> #361 -> #360 -> #363 -> #362**, with **#338 /
#349-#354** running in parallel as the model-quality proof.

## Hard Boundaries

- Public narrative uses placeholders: "frontier model access restrictions" and
  "a humanoid robotics buyer." Do not name specific companies or incidents
  unless separately verified and approved.
- The first shippable economics path is simulated ledger plus Mollie test
  checkout/payment links. Webhooks and payout simulation are bonus.
- Mollie API keys stay server-side in ignored `.env` or process env. Browser
  code never imports the SDK and never sees `MOLLIE_API_KEY`.
- Rewards are demo allocations, not legal payouts, wages, securities, or real
  revenue-share promises.
