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

## Implemented Modules

| Module | Responsibility |
|---|---|
| `lensemble/demo/economy.py` | Pure deterministic ledger, sale, reward, and mock-payment logic. No network. |
| `lensemble/demo/mollie.py` | Thin server-side adapter over the official `mollie-api-python` SDK. |
| `tests/ml/test_demo_economy.py` | Ledger math, run-derived weights, fake-Mollie checkout, env loading, browser secret-boundary checks. |
| `web/federated-demo/app.mjs` | Host-dashboard economy panel: buyer sale, checkout link, payment status, reward split. |

## Server Routes

Routes under the existing local demo API:

| Route | Method | Purpose |
|---|---|---|
| `/api/economy/config` | `GET` | Return public economy config: key presence/prefix, default amounts, no secret. |
| `/api/economy/sales` | `POST` | Create a sale scenario from a run/model revision and participant contribution rows. |
| `/api/economy/sales/<sale_id>` | `GET` | Return sale, payment status, ledger, reward rows, and provenance. |
| `/api/economy/sales/<sale_id>/payment` | `POST` | Create a Mollie test payment or mock checkout link. |
| `/api/economy/sales/<sale_id>/status` | `POST` | Refresh/poll payment status; mock mode marks paid. |

The first shippable behavior uses in-memory state like `FederatedDemoService`.
Durable persistence is not required for the hackathon demo. When a sale is
created from an existing run, contribution rows are derived from the run's
redacted `updateMetadata`; when the run is missing or has no accepted updates,
the server falls back to deterministic fixture participants.

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
payment = MolliePaymentClient(api_key).create_payment(
    MolliePaymentRequest(
        amount_value=checkout_amount,
        currency="EUR",
        description="L'Ensemble sovereign world-model demo sale",
        redirect_url=f"{base_url}/web/federated-demo/#/host/{run_id}?sale={sale_id}",
        webhook_url=f"{base_url}/api/economy/mollie/webhook",
        metadata={
            "saleId": sale_id,
            "runId": run_id,
            "modelRevisionId": model_revision_id,
        },
    )
)
checkout_url = payment.checkout_url
```

The default **simulated sale amount** is EUR 1,000,000. The default **Mollie
test checkout amount** is EUR 10.00 so the credentialed checkout is reliable
while the ledger still displays the high-value scenario. The dashboard labels
those as "simulated sale" and "checkout."

Localhost demos omit `webhookUrl` because Mollie rejects unreachable local
webhook URLs. Webhook reconciliation starts only when `LENSEMBLE_PUBLIC_BASE_URL`
is an HTTPS non-local origin.

## Environment Contract

Tracked template: `.env.example`.

Ignored/local values:

- `MOLLIE_API_KEY` - Mollie test API key. Required only for credentialed smoke.
- `LENSEMBLE_PUBLIC_BASE_URL` - redirect/webhook base URL, default
  `http://127.0.0.1:8765`.
- `LENSEMBLE_DEMO_SALE_AMOUNT_EUR` - scenario amount, default `1000000.00`.
- `LENSEMBLE_DEMO_CHECKOUT_AMOUNT_EUR` - test checkout amount, default `10.00`.
- `LENSEMBLE_DEMO_ORCHESTRATOR_SHARE` - decimal share, default `0.20`.
- `LENSEMBLE_DEMO_MOCK_PAYMENTS` - force the mock payment path, default `false`.

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
