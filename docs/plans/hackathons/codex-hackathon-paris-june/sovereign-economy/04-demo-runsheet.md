# Sovereign Economy - Demo Runsheet

## 90-Second Script

1. **Sovereignty hook (0:00-0:15).** "Frontier intelligence is increasingly
   gated by a few infrastructure owners. For robotics, that means the world
   model that understands your factory, lab, or city may be controlled
   elsewhere."
2. **Federated run (0:15-0:35).** "Here, people contribute local data and
   compute to improve a shared robotics world model. Their raw trajectories stay
   on their device; only bounded adapter deltas are aggregated."
3. **Surprise proof (0:35-0:55).** Show the TwoRooms surprise meter. Perturb the
   world; surprise spikes. Toggle post-federation; held-out prediction error is
   lower. Show +12.3% this run, +16.8% mean, +5.4% worst seed.
4. **Economy (0:55-1:15).** "A humanoid robotics buyer wants the improved
   revision." Open the Mollie test checkout/payment link. Mark paid in test
   mode or mock mode.
5. **Reward split (1:15-1:30).** Show L'Ensemble Labs' orchestrator share and
   participant rewards split by contribution ledger. Close: "sovereign model
   improvement with shared upside, not data extraction."

## Live Flow

1. Start server: `uv run lensemble demo federated --port 8765`.
2. Open `http://127.0.0.1:8765/web/federated-demo/`.
3. Run or replay the federated adapter-continuation result.
4. Use the host dashboard's **Sovereign economy** panel to create the buyer sale.
5. Create Mollie test checkout if `MOLLIE_API_KEY` exists; otherwise use mock.
6. Open checkout/payment link or mark mock paid.
7. Show reward split, then open the surprise-meter view for the model-quality proof.

## Fallback Ladder

| Rung | What | Use when |
|---|---|---|
| A | Live run + live surprise-meter + Mollie test checkout + reward split | Everything rehearsed. |
| B | Live surprise-meter + mock checkout + deterministic ledger | Payment credentials/network fragile. |
| C | Recorded surprise trajectory + mock checkout + deterministic ledger | Runtime fragile. |
| D | <=20 s capture clip + result/economics card | Projector/browser fragile. |

## Autonomous Gate

Default deterministic rehearsal:

```bash
uv run python scripts/hackathon_demo_rehearsal.py
```

This emits `sovereign-economy-rehearsal/1` and executes rung C: synthetic
federated run, surprise-meter rehearsal, recorded fallback assets, mock checkout,
deterministic ledger, reward split, public non-claim string checks, and a JSON
demo card with the required economics and surprise fields.

Stage-machine rehearsal after local capture:

```bash
uv run python scripts/hackathon_demo_rehearsal.py --require-capture
```

Credentialed payment rehearsal, only when a server-side `.env` or shell already
contains a Mollie test key:

```bash
uv run python scripts/hackathon_demo_rehearsal.py --payment-mode auto
```

## Preflight

- [ ] `.env` or shell has `MOLLIE_API_KEY` for credentialed path, or mock mode is selected.
- [ ] `LENSEMBLE_DEMO_CHECKOUT_AMOUNT_EUR` is a small Mollie test amount; the
      EUR 1,000,000 figure remains the simulated sale amount.
- [ ] `.env` is ignored and no real key is committed.
- [ ] Ledger fixture has at least four participants and cent-balanced rewards.
- [ ] Surprise-meter fallback assets exist.
- [ ] Claim-safe copy uses placeholders, not named incidents or named buyers.
- [ ] Result card shows both economics and model-quality numbers.

## Result Card

Required fields:

- Simulated sale amount.
- Orchestrator share.
- Community pool.
- Top participant reward rows.
- +12.3% this run, +16.8% mean, +5.4% worst seed.
- Non-claims: simulation-only, no legal payout, adapter continuation only, no
  DP/secure aggregation in demo path.
