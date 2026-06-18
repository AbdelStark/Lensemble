"""Server-side Mollie payment adapter for the hackathon economy demo.

The browser never sees the API key. This module is intentionally small: it
wraps Mollie's official Python client at the boundary where the local demo
server creates a test checkout URL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class MolliePaymentRequest:
    amount_value: str
    currency: str
    description: str
    redirect_url: str
    webhook_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MolliePaymentResult:
    payment_id: str
    checkout_url: str
    status: str
    amount_value: str
    currency: str
    sdk: str = "mollie-api-python"


class MolliePaymentClient:
    """Thin adapter over the official `mollie-api-python` SDK."""

    def __init__(self, api_key: str) -> None:
        if not api_key.startswith(("test_", "live_")):
            raise ValueError("Mollie API key must start with test_ or live_")
        self._api_key = api_key

    def create_payment(self, request: MolliePaymentRequest) -> MolliePaymentResult:
        try:
            from mollie.api.client import Client
        except ImportError as exc:  # pragma: no cover - exercised by env contract
            raise RuntimeError(
                "mollie-api-python is not installed; install the project dependencies"
            ) from exc

        client = Client()
        client.set_api_key(self._api_key)
        payload: dict[str, Any] = {
            "amount": {
                "currency": request.currency,
                "value": request.amount_value,
            },
            "description": request.description,
            "redirectUrl": request.redirect_url,
            "metadata": dict(request.metadata),
        }
        if request.webhook_url:
            payload["webhookUrl"] = request.webhook_url
        payment = client.payments.create(payload)
        checkout_url = getattr(payment, "checkout_url", None)
        payment_id = getattr(payment, "id", None)
        if not checkout_url or not payment_id:
            raise RuntimeError(
                "Mollie payment response did not include id/checkout_url"
            )
        return MolliePaymentResult(
            payment_id=str(payment_id),
            checkout_url=str(checkout_url),
            status=str(getattr(payment, "status", "open") or "open"),
            amount_value=request.amount_value,
            currency=request.currency,
        )

    def get_payment_status(self, payment_id: str) -> str:
        try:
            from mollie.api.client import Client
        except ImportError as exc:  # pragma: no cover - exercised by env contract
            raise RuntimeError(
                "mollie-api-python is not installed; install the project dependencies"
            ) from exc

        client = Client()
        client.set_api_key(self._api_key)
        payment = client.payments.get(payment_id)
        is_paid = getattr(payment, "is_paid", None)
        if callable(is_paid) and is_paid():
            return "paid"
        return str(getattr(payment, "status", "open") or "open")
