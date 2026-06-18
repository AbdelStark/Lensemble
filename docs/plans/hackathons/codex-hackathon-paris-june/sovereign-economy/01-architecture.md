# Sovereign Economy - Technical Architecture

## Existing Foundation

The economics layer extends the existing stdlib demo server:

- `uv run lensemble demo federated --port 8765`
- `lensemble/demo/server.py`
- `lensemble/demo/federated.py`
- `web/federated-demo/`
- `web/surprise-meter/`

The server already owns run creation, participants, update submission, model
revision publication, events, and evidence export. The new layer should stay
beside that service rather than replacing it.

## Proposed Modules

| Module | Responsibility |
|---|---|
| `lensemble/demo/economy.py` | Pure deterministic ledger, sale, reward, and mock-payment logic. No network. |
| `lensemble/demo/mollie.py` | Thin server-side Mollie client adapter. Imports the official Python SDK/API client only when used. |
| `tests/ml/test_demo_economy.py` | Ledger math, rounding, bad-input rejection, mock payment contract. |
| `tests/ml/test_mollie_env_contract.py` | Secret/env and browser-boundary checks. |
| `web/surprise-meter/economy_panel.mjs` | Buyer checkout card, ledger rows, reward split UI. |

Implementation can choose different filenames, but these ownership boundaries
should hold.

## Server Routes

Add routes under the existing local demo API:

| Route | Method | Purpose |
|---|---|---|
| `/api/economy/sales` | `POST` | Create a sale scenario from a run/model revision and participant contribution rows. |
| `/api/economy/sales/<sale_id>` | `GET` | Return sale, payment status, ledger, reward rows, and provenance. |
| `/api/economy/sales/<sale_id>/mollie-payment` | `POST` | Create a Mollie test payment or mock checkout link. |
| `/api/economy/sales/<sale_id>/status` | `POST` | Refresh/poll payment status; mock mode can mark paid. |

First shippable behavior may use in-memory state like `FederatedDemoService`.
Durable persistence is not required for the hackathon demo.

## Mollie Boundary

Official references checked on 2026-06-18:

- API endpoint: `POST https://api.mollie.com/v2/payments`.
- Required payment fields include `description`, `amount`, and `redirectUrl`.
- The repo server is Python, so the default integration should use Mollie's
  official Python SDK/API client (`mollie-api-python` or the newer
  `mollie-api-py`) from server code.
- Mollie's Node client documentation gives the same security boundary: API keys
  belong on the server, not in browser code.

Spec links:

- <https://docs.mollie.com/reference/create-payment>
- <https://github.com/mollie/mollie-api-python>
- <https://github.com/mollie/mollie-api-py>
- <https://github.com/mollie/mollie-api-node>

Server-side adapter contract:

```python
payment = mollie_client.payments.create(
    {
        "amount": {"value": "1000000.00", "currency": "EUR"},
        "description": "L'Ensemble sovereign world-model demo sale",
        "redirectUrl": f"{base_url}/web/surprise-meter/?sale={sale_id}",
        "webhookUrl": f"{base_url}/api/economy/mollie/webhook",
        "metadata": {
            "saleId": sale_id,
            "runId": run_id,
            "modelRevisionId": model_revision_id,
        },
    }
)
checkout_url = payment.checkout_url
```

If the Mollie test environment rejects the large scenario amount, the payment
adapter should fall back to a smaller **payment amount** while the ledger still
displays the configurable scenario amount. The UI must label that split clearly:
"test checkout amount" vs "simulated sale amount."

## Environment Contract

Tracked template: `.env.example`.

Ignored/local values:

- `MOLLIE_API_KEY` - Mollie test API key. Required only for credentialed smoke.
- `LENSEMBLE_PUBLIC_BASE_URL` - redirect/webhook base URL, default
  `http://127.0.0.1:8765`.
- `LENSEMBLE_DEMO_SALE_AMOUNT_EUR` - scenario amount, default `1000000.00`.
- `LENSEMBLE_DEMO_ORCHESTRATOR_SHARE` - decimal share, default `0.20`.

No frontend file may read or contain `MOLLIE_API_KEY`, `test_...`, or `live_...`.

## UI Integration

The integrated view should put three proof surfaces in one flow:

1. **Run proof:** participants and contribution rows.
2. **Model proof:** surprise meter and certified held-out improvement.
3. **Economy proof:** buyer checkout, orchestrator share, participant rewards.

The default implementation can add an economy panel to `web/surprise-meter/`.
It may use mock economy JSON until the backend routes are ready.

## Bonus Expansion

Only after the first shippable path works:

- Mollie webhook route with payment status reconciliation.
- Refund/failed payment states.
- Payout simulation export.
- Mollie Connect/submerchant research spike.
