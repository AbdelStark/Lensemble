from __future__ import annotations

import json
import re
import threading
import urllib.request
from decimal import Decimal
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from lensemble.demo.economy import EconomyConfig, EconomyDemoError, EconomyDemoService
from lensemble.demo.federated import FederatedDemoService
from lensemble.demo.mollie import MolliePaymentRequest, MolliePaymentResult
from lensemble.demo.server import make_handler


class FakeMollieClient:
    def __init__(self) -> None:
        self.created: list[MolliePaymentRequest] = []

    def create_payment(self, request: MolliePaymentRequest) -> MolliePaymentResult:
        self.created.append(request)
        return MolliePaymentResult(
            payment_id="tr_demo_payment",
            checkout_url="https://www.mollie.com/checkout/select-method/demo",
            status="open",
            amount_value=request.amount_value,
            currency=request.currency,
        )

    def get_payment_status(self, payment_id: str) -> str:
        assert payment_id == "tr_demo_payment"
        return "paid"


def test_economy_ledger_balances_to_the_cent() -> None:
    service = EconomyDemoService(
        config=EconomyConfig(api_key=None, sale_amount=Decimal("1000000.00"))
    )
    sale = service.create_sale({"saleId": "sale_demo_balance"})

    orchestrator = _money(sale["ledger"]["orchestratorReward"])
    participant_total = sum(
        _money(row["reward"]) for row in sale["ledger"]["participantRewards"]
    )
    assert sale["schema"] == "sovereign-sale/1"
    assert sale["ledger"]["schema"] == "sovereign-contribution-ledger/1"
    assert orchestrator == Decimal("200000.00")
    assert participant_total == Decimal("800000.00")
    assert orchestrator + participant_total == _money(sale["saleAmount"])
    assert set(sale["nonClaims"]) >= {
        "simulation-not-legal-payout",
        "adapter-continuation-not-full-model-training",
        "single-coordinator-no-secure-agg-or-dp",
        "buyer-is-placeholder",
    }


def test_economy_uses_run_snapshot_when_updates_exist() -> None:
    run_snapshot = {
        "id": "run_live",
        "currentModelRevisionId": "rev-live",
        "participants": [
            {
                "id": "browser-a",
                "displayName": "Alice",
                "updateMetadata": {
                    "1": {
                        "schema": "browser-update/1",
                        "sampleCount": 20,
                        "localSteps": 5,
                        "runtimeMs": 1000,
                        "collapseRisk": "low",
                    }
                },
            },
            {
                "id": "browser-b",
                "displayName": "Bob",
                "updateMetadata": {
                    "1": {
                        "schema": "browser-update/1",
                        "sampleCount": 10,
                        "localSteps": 2,
                        "runtimeMs": 500,
                        "collapseRisk": "watch",
                    }
                },
            },
        ],
    }
    service = EconomyDemoService()
    sale = service.create_sale({"saleId": "sale_run"}, run_snapshot=run_snapshot)
    rows = sale["ledger"]["participantRewards"]

    assert sale["runId"] == "run_live"
    assert sale["modelRevisionId"] == "rev-live"
    assert [row["participantId"] for row in rows] == ["browser-a", "browser-b"]
    assert rows[0]["reward"]["value"] > rows[1]["reward"]["value"]


def test_economy_rejects_bad_inputs() -> None:
    service = EconomyDemoService()

    with pytest.raises(EconomyDemoError, match="orchestratorShare"):
        service.create_sale({"orchestratorShare": "-0.1"})
    with pytest.raises(EconomyDemoError, match="only EUR"):
        service.create_sale({"saleAmount": {"currency": "USD", "value": "100.00"}})
    with pytest.raises(EconomyDemoError, match="cannot be empty"):
        service.create_sale({"participants": []})
    with pytest.raises(EconomyDemoError, match="positive weight"):
        service.create_sale(
            {
                "participants": [
                    {
                        "participantId": "zero",
                        "acceptedRounds": 0,
                        "sampleCount": 0,
                        "localSteps": 0,
                        "runtimeMs": 0,
                        "updateHealth": 1,
                    }
                ]
            }
        )


def test_http_economy_routes_create_mollie_checkout_with_fake_client() -> None:
    fake = FakeMollieClient()
    demo = FederatedDemoService(public_base_url="http://127.0.0.1:0/web/federated-demo")
    economy = EconomyDemoService(
        config=EconomyConfig(
            api_key="test_short",
            public_base_url="http://127.0.0.1:0",
            checkout_amount=Decimal("10.00"),
        ),
        payment_client=fake,
    )
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), make_handler(demo, economy_service=economy)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        health = json.loads(urllib.request.urlopen(f"{base}/api/health").read())
        encoded_health = json.dumps(health)
        assert health["economy"]["mollieAvailable"] is True
        assert "test_short" not in encoded_health

        sale = _post_json(
            f"{base}/api/economy/sales",
            {"saleId": "sale_http", "runId": "run_http", "modelRevisionId": "rev_1"},
        )
        assert sale["payment"]["status"] == "not_created"

        paid_open = _post_json(f"{base}/api/economy/sales/sale_http/payment", {})
        assert paid_open["payment"]["mode"] == "mollie-test"
        assert paid_open["payment"]["checkoutUrl"].startswith("https://www.mollie.com/")
        assert fake.created[0].metadata == {
            "saleId": "sale_http",
            "runId": "run_http",
            "modelRevisionId": "rev_1",
        }
        assert fake.created[0].amount_value == "10.00"
        assert fake.created[0].webhook_url == ""

        paid = _post_json(f"{base}/api/economy/sales/sale_http/status", {})
        assert paid["payment"]["status"] == "paid"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_env_file_contract_loads_key_without_public_exposure(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "MOLLIE_API_KEY=test_short",
                "LENSEMBLE_PUBLIC_BASE_URL=http://demo.local",
                "LENSEMBLE_DEMO_SALE_AMOUNT_EUR=1234.56",
                "LENSEMBLE_DEMO_CHECKOUT_AMOUNT_EUR=7.89",
                "LENSEMBLE_DEMO_ORCHESTRATOR_SHARE=0.25",
            ]
        ),
        encoding="utf-8",
    )
    config = EconomyConfig.from_env(env_path=env)
    payload = config.public_payload()

    assert config.mollie_available is True
    assert payload["apiKeyPresent"] is True
    assert payload["apiKeyPrefix"] == "test"
    assert "test_short" not in json.dumps(payload)
    assert payload["defaultSaleAmount"] == "1234.56"
    assert payload["defaultCheckoutAmount"] == "7.89"


def test_browser_files_do_not_expose_mollie_secrets() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("web").rglob("*")
        if path.suffix in {".mjs", ".js", ".html", ".css"}
    )
    assert "MOLLIE_API_KEY" not in text
    assert re.search(r"(?:test|live)_[A-Za-z0-9]{24,}", text) is None


def _money(amount: dict[str, Any]) -> Decimal:
    return Decimal(str(amount["value"]))


def _post_json(url: str, payload: dict[str, object]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))
