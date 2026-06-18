"""Deterministic contribution ledger and sale simulation for the demo economy."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from lensemble.demo.mollie import (
    MolliePaymentClient,
    MolliePaymentRequest,
    MolliePaymentResult,
)

SALE_SCHEMA = "sovereign-sale/1"
LEDGER_SCHEMA = "sovereign-contribution-ledger/1"
DEFAULT_CURRENCY = "EUR"
DEFAULT_SALE_AMOUNT = Decimal("1000000.00")
DEFAULT_CHECKOUT_AMOUNT = Decimal("10.00")
DEFAULT_ORCHESTRATOR_SHARE = Decimal("0.20")
CENT = Decimal("0.01")
MANDATORY_NON_CLAIMS = [
    "simulation-not-legal-payout",
    "adapter-continuation-not-full-model-training",
    "single-coordinator-no-secure-agg-or-dp",
    "buyer-is-placeholder",
]


class EconomyDemoError(ValueError):
    def __init__(self, code: str, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass(frozen=True, slots=True)
class EconomyConfig:
    api_key: str | None = None
    public_base_url: str = "http://127.0.0.1:8765"
    sale_amount: Decimal = DEFAULT_SALE_AMOUNT
    checkout_amount: Decimal = DEFAULT_CHECKOUT_AMOUNT
    orchestrator_share: Decimal = DEFAULT_ORCHESTRATOR_SHARE
    force_mock: bool = False

    @property
    def mollie_available(self) -> bool:
        return (
            not self.force_mock
            and self.api_key is not None
            and self.api_key.startswith(("test_", "live_"))
        )

    @property
    def api_key_prefix(self) -> str | None:
        if not self.api_key:
            return None
        if self.api_key.startswith("test_"):
            return "test"
        if self.api_key.startswith("live_"):
            return "live"
        return "unknown"

    def public_payload(self) -> dict[str, Any]:
        return {
            "schema": "sovereign-economy-config/1",
            "provider": "mollie" if self.mollie_available else "mock",
            "mollieAvailable": self.mollie_available,
            "apiKeyPresent": self.api_key is not None,
            "apiKeyPrefix": self.api_key_prefix,
            "publicBaseUrl": self.public_base_url,
            "defaultSaleAmount": _money(self.sale_amount),
            "defaultCheckoutAmount": _money(self.checkout_amount),
            "defaultOrchestratorShare": _share(self.orchestrator_share),
        }

    @classmethod
    def from_env(
        cls,
        *,
        env_path: str | Path = ".env",
        public_base_url: str | None = None,
    ) -> "EconomyConfig":
        env = _load_env_file(Path(env_path))
        merged = {**env, **os.environ}
        base = public_base_url or merged.get("LENSEMBLE_PUBLIC_BASE_URL")
        return cls(
            api_key=_blank_to_none(merged.get("MOLLIE_API_KEY")),
            public_base_url=(base or "http://127.0.0.1:8765").rstrip("/"),
            sale_amount=_decimal_env(
                merged.get("LENSEMBLE_DEMO_SALE_AMOUNT_EUR"),
                DEFAULT_SALE_AMOUNT,
            ),
            checkout_amount=_decimal_env(
                merged.get("LENSEMBLE_DEMO_CHECKOUT_AMOUNT_EUR"),
                DEFAULT_CHECKOUT_AMOUNT,
            ),
            orchestrator_share=_decimal_env_raw(
                merged.get("LENSEMBLE_DEMO_ORCHESTRATOR_SHARE"),
                DEFAULT_ORCHESTRATOR_SHARE,
            ),
            force_mock=str(merged.get("LENSEMBLE_DEMO_MOCK_PAYMENTS", "")).lower()
            in {"1", "true", "yes"},
        )


class EconomyDemoService:
    """In-memory sale, payment, and reward ledger service for the local demo."""

    def __init__(
        self,
        *,
        config: EconomyConfig | None = None,
        payment_client: Any | None = None,
    ) -> None:
        self.config = config or EconomyConfig.from_env()
        self._payment_client = payment_client
        self._sales: dict[str, dict[str, Any]] = {}

    def config_payload(self) -> dict[str, Any]:
        return self.config.public_payload()

    def create_sale(
        self,
        payload: dict[str, Any] | None = None,
        *,
        run_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = dict(payload or {})
        sale_id = str(payload.get("saleId") or self._new_sale_id())
        if sale_id in self._sales:
            raise EconomyDemoError(
                "sale_exists", f"sale {sale_id!r} already exists", status=409
            )

        sale_amount = _decimal_payload(
            payload.get("saleAmount"),
            self.config.sale_amount,
        )
        sale_currency = _currency_payload(payload.get("saleAmount"))
        checkout_amount = _decimal_payload(
            payload.get("checkoutAmount"),
            self.config.checkout_amount,
        )
        checkout_currency = _currency_payload(payload.get("checkoutAmount"))
        if sale_currency != DEFAULT_CURRENCY or checkout_currency != DEFAULT_CURRENCY:
            raise EconomyDemoError(
                "unsupported_currency", "only EUR is supported in the demo economy"
            )
        orchestrator_share = _decimal_payload(
            payload.get("orchestratorShare"),
            self.config.orchestrator_share,
        )
        if orchestrator_share < Decimal("0") or orchestrator_share > Decimal("1"):
            raise EconomyDemoError(
                "invalid_share", "orchestratorShare must be between 0 and 1"
            )

        rows = _contribution_rows(payload.get("participants"), run_snapshot)
        ledger = build_contribution_ledger(
            rows,
            sale_amount=sale_amount,
            orchestrator_share=orchestrator_share,
        )
        run_id = str(
            payload.get("runId") or (run_snapshot or {}).get("id") or "demo-run"
        )
        revision_id = str(
            payload.get("modelRevisionId")
            or (run_snapshot or {}).get("currentModelRevisionId")
            or "latest"
        )
        sale = {
            "schema": SALE_SCHEMA,
            "saleId": sale_id,
            "runId": run_id,
            "modelRevisionId": revision_id,
            "buyer": {
                "kind": "humanoid-robotics-buyer",
                "displayName": "Humanoid robotics buyer",
            },
            "saleAmount": {"currency": DEFAULT_CURRENCY, "value": _money(sale_amount)},
            "checkoutAmount": {
                "currency": DEFAULT_CURRENCY,
                "value": _money(checkout_amount),
                "label": "Mollie test checkout amount",
            },
            "orchestratorShare": _share(orchestrator_share),
            "communityPool": ledger["communityPool"],
            "payment": {
                "provider": "mollie" if self.config.mollie_available else "mock",
                "mode": "not-created",
                "paymentId": None,
                "status": "not_created",
                "checkoutUrl": None,
            },
            "ledger": ledger,
            "nonClaims": list(MANDATORY_NON_CLAIMS),
            "createdAt": _now_ms(),
        }
        self._sales[sale_id] = sale
        return _copy_sale(sale)

    def get_sale(self, sale_id: str) -> dict[str, Any]:
        sale = self._sales.get(sale_id)
        if sale is None:
            raise EconomyDemoError("not_found", f"unknown sale {sale_id}", status=404)
        return _copy_sale(sale)

    def create_payment(
        self,
        sale_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = dict(payload or {})
        sale = self._sale(sale_id)
        preferred_mode = str(payload.get("mode") or "auto")
        allow_mock_fallback = bool(payload.get("allowMockFallback", True))
        if preferred_mode == "mock" or not self.config.mollie_available:
            sale["payment"] = self._mock_payment(sale, mode="mock")
            return _copy_sale(sale)

        try:
            payment = self._mollie_client().create_payment(
                MolliePaymentRequest(
                    amount_value=sale["checkoutAmount"]["value"],
                    currency=sale["checkoutAmount"]["currency"],
                    description="L'Ensemble sovereign world-model demo sale",
                    redirect_url=self._redirect_url(sale),
                    webhook_url=self._webhook_url(),
                    metadata={
                        "saleId": sale["saleId"],
                        "runId": sale["runId"],
                        "modelRevisionId": sale["modelRevisionId"],
                    },
                )
            )
            sale["payment"] = self._mollie_payment(payment)
        except Exception as exc:
            if not allow_mock_fallback:
                raise EconomyDemoError(
                    "mollie_payment_failed",
                    _sanitize_error(exc),
                    status=502,
                ) from exc
            sale["payment"] = self._mock_payment(
                sale,
                mode="mock-fallback",
                provider="mock",
                note=_sanitize_error(exc),
            )
        return _copy_sale(sale)

    def refresh_status(
        self,
        sale_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = dict(payload or {})
        sale = self._sale(sale_id)
        payment = sale["payment"]
        if payment.get("mode") in {"mock", "mock-fallback"}:
            if payload.get("markPaid", True):
                payment["status"] = "paid"
                payment["paidAt"] = _now_ms()
            return _copy_sale(sale)
        payment_id = payment.get("paymentId")
        if payment_id and self.config.mollie_available:
            try:
                payment["status"] = self._mollie_client().get_payment_status(payment_id)
                payment["checkedAt"] = _now_ms()
            except Exception as exc:
                payment["lastStatusError"] = _sanitize_error(exc)
        return _copy_sale(sale)

    def _sale(self, sale_id: str) -> dict[str, Any]:
        sale = self._sales.get(sale_id)
        if sale is None:
            raise EconomyDemoError("not_found", f"unknown sale {sale_id}", status=404)
        return sale

    def _new_sale_id(self) -> str:
        return f"sale_{int(time.time() * 1000)}_{len(self._sales) + 1}"

    def _mollie_client(self) -> Any:
        if self._payment_client is not None:
            return self._payment_client
        if not self.config.api_key:
            raise EconomyDemoError(
                "mollie_key_missing",
                "MOLLIE_API_KEY is required for Mollie test payments",
                status=409,
            )
        self._payment_client = MolliePaymentClient(self.config.api_key)
        return self._payment_client

    def _redirect_url(self, sale: dict[str, Any]) -> str:
        return (
            f"{self.config.public_base_url}/web/federated-demo/"
            f"#/economy/{sale['runId']}?sale={sale['saleId']}"
        )

    def _webhook_url(self) -> str:
        base = self.config.public_base_url
        if not base.startswith("https://"):
            return ""
        if "127.0.0.1" in base or "localhost" in base:
            return ""
        return f"{base}/api/economy/mollie/webhook"

    def _mock_payment(
        self,
        sale: dict[str, Any],
        *,
        mode: str,
        provider: str = "mock",
        note: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "provider": provider,
            "mode": mode,
            "paymentId": f"mock_tr_{sale['saleId']}",
            "status": "open",
            "checkoutUrl": self._redirect_url(sale) + "&mockCheckout=1",
            "amount": sale["checkoutAmount"],
            "scenarioAmount": sale["saleAmount"],
        }
        if note:
            payload["fallbackReason"] = note
        return payload

    @staticmethod
    def _mollie_payment(payment: MolliePaymentResult) -> dict[str, Any]:
        return {
            "provider": "mollie",
            "mode": "mollie-test",
            "sdk": payment.sdk,
            "paymentId": payment.payment_id,
            "status": payment.status,
            "checkoutUrl": payment.checkout_url,
            "amount": {
                "currency": payment.currency,
                "value": payment.amount_value,
                "label": "Mollie test checkout amount",
            },
        }


def build_contribution_ledger(
    rows: list[dict[str, Any]],
    *,
    sale_amount: Decimal,
    orchestrator_share: Decimal,
) -> dict[str, Any]:
    if not rows:
        raise EconomyDemoError("invalid_ledger", "at least one participant is required")
    sale_amount = _quantize_money(sale_amount)
    orchestrator_reward = _quantize_money(sale_amount * orchestrator_share)
    community_pool = _quantize_money(sale_amount - orchestrator_reward)
    weighted = [_with_weight(row) for row in rows]
    total_weight = sum((row["_weight"] for row in weighted), Decimal("0"))
    if total_weight <= 0:
        raise EconomyDemoError(
            "invalid_ledger", "at least one participant needs positive weight"
        )

    reward_cents: list[tuple[int, Decimal, dict[str, Any]]] = []
    pool_cents = int((community_pool / CENT).to_integral_value(rounding=ROUND_DOWN))
    assigned = 0
    for row in weighted:
        exact = (Decimal(pool_cents) * row["_weight"]) / total_weight
        cents = int(exact.to_integral_value(rounding=ROUND_DOWN))
        assigned += cents
        reward_cents.append((cents, exact - Decimal(cents), row))
    remainder = pool_cents - assigned
    for index, (_, _, row) in enumerate(
        sorted(reward_cents, key=lambda item: (-item[1], str(item[2]["participantId"])))
    ):
        if index >= remainder:
            break
        for pos, current in enumerate(reward_cents):
            if current[2] is row:
                reward_cents[pos] = (current[0] + 1, current[1], current[2])
                break

    participant_rewards = []
    for cents, _, row in sorted(
        reward_cents, key=lambda item: str(item[2]["participantId"])
    ):
        public = {key: value for key, value in row.items() if not key.startswith("_")}
        public["contributionWeight"] = float(row["_weight"].quantize(Decimal("0.0001")))
        public["reward"] = {
            "currency": DEFAULT_CURRENCY,
            "value": _money(Decimal(cents) * CENT),
        }
        participant_rewards.append(public)

    return {
        "schema": LEDGER_SCHEMA,
        "currency": DEFAULT_CURRENCY,
        "saleAmount": {"currency": DEFAULT_CURRENCY, "value": _money(sale_amount)},
        "orchestratorShare": _share(orchestrator_share),
        "orchestratorReward": {
            "currency": DEFAULT_CURRENCY,
            "value": _money(orchestrator_reward),
        },
        "communityPool": {
            "currency": DEFAULT_CURRENCY,
            "value": _money(community_pool),
        },
        "participantRewards": participant_rewards,
        "roundingRemainder": {"currency": DEFAULT_CURRENCY, "value": "0.00"},
        "formula": (
            "weight=(sampleCount + 4*localSteps + runtimeMs/1000) "
            "* max(acceptedRounds,1) * updateHealth * capFactor"
        ),
        "provenance": {
            "parentIssue": 359,
            "evidenceFiles": [
                "docs/evidence/lewm_tworooms_system_probe.json",
                "docs/evidence/lewm_tworooms_probe_seedsweep.json",
            ],
        },
    }


def _contribution_rows(
    explicit_rows: Any,
    run_snapshot: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if isinstance(explicit_rows, list):
        if not explicit_rows:
            raise EconomyDemoError(
                "invalid_ledger", "explicit participant rows cannot be empty"
            )
        rows = [_normalize_row(row) for row in explicit_rows if isinstance(row, dict)]
        if not rows:
            raise EconomyDemoError(
                "invalid_ledger", "explicit participant rows must be objects"
            )
        return rows
    if run_snapshot:
        rows = _rows_from_run(run_snapshot)
        if any(_row_weight(row) > 0 for row in rows):
            return rows
    return _fixture_rows()


def _rows_from_run(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for participant in run.get("participants") or []:
        updates = participant.get("updateMetadata") or {}
        accepted_rounds = len(updates)
        sample_count = 0
        local_steps = 0
        runtime_ms = Decimal("0")
        health_values: list[Decimal] = []
        for metadata in updates.values():
            sample_count += int(metadata.get("sampleCount") or 0)
            local_steps += int(metadata.get("localSteps") or 0)
            runtime_ms += _decimal_payload(metadata.get("runtimeMs"), Decimal("0"))
            if metadata.get("schema") == "lewm-adapter-delta/1":
                metrics = metadata.get("metrics") or {}
                health_values.append(
                    Decimal("1.0") if metrics.get("lossDecreased") else Decimal("0.86")
                )
            else:
                risk = str(metadata.get("collapseRisk") or "watch")
                health_values.append(
                    {"low": Decimal("1.0"), "watch": Decimal("0.88")}.get(
                        risk, Decimal("0.72")
                    )
                )
        health = (
            sum(health_values, Decimal("0")) / Decimal(len(health_values))
            if health_values
            else Decimal("0")
        )
        rows.append(
            _normalize_row(
                {
                    "participantId": participant.get("id"),
                    "displayName": participant.get("displayName")
                    or participant.get("id"),
                    "acceptedRounds": accepted_rounds,
                    "sampleCount": sample_count,
                    "localSteps": local_steps,
                    "runtimeMs": runtime_ms,
                    "updateHealth": health,
                    "capFactor": Decimal("1.0"),
                }
            )
        )
    return rows


def _fixture_rows() -> list[dict[str, Any]]:
    return [
        _normalize_row(
            {
                "participantId": "p_ada",
                "displayName": "Ada",
                "acceptedRounds": 3,
                "sampleCount": 640,
                "localSteps": 180,
                "runtimeMs": 42100,
                "updateHealth": Decimal("0.98"),
                "capFactor": Decimal("1.0"),
            }
        ),
        _normalize_row(
            {
                "participantId": "p_ben",
                "displayName": "Ben",
                "acceptedRounds": 3,
                "sampleCount": 480,
                "localSteps": 144,
                "runtimeMs": 39800,
                "updateHealth": Decimal("0.95"),
                "capFactor": Decimal("1.0"),
            }
        ),
        _normalize_row(
            {
                "participantId": "p_chloe",
                "displayName": "Chloe",
                "acceptedRounds": 2,
                "sampleCount": 360,
                "localSteps": 96,
                "runtimeMs": 31400,
                "updateHealth": Decimal("0.92"),
                "capFactor": Decimal("1.0"),
            }
        ),
        _normalize_row(
            {
                "participantId": "p_dan",
                "displayName": "Dan",
                "acceptedRounds": 1,
                "sampleCount": 180,
                "localSteps": 64,
                "runtimeMs": 18800,
                "updateHealth": Decimal("0.90"),
                "capFactor": Decimal("1.0"),
            }
        ),
    ]


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "participantId": str(row.get("participantId") or "participant"),
        "displayName": str(
            row.get("displayName") or row.get("participantId") or "Participant"
        ),
        "acceptedRounds": max(0, int(row.get("acceptedRounds") or 0)),
        "sampleCount": max(0, int(row.get("sampleCount") or 0)),
        "localSteps": max(0, int(row.get("localSteps") or 0)),
        "runtimeMs": int(_decimal_payload(row.get("runtimeMs"), Decimal("0"))),
        "updateHealth": float(
            _bounded_decimal(row.get("updateHealth"), Decimal("1.0"))
        ),
        "capFactor": float(_bounded_decimal(row.get("capFactor"), Decimal("1.0"))),
    }


def _with_weight(row: dict[str, Any]) -> dict[str, Any]:
    copied = dict(row)
    copied["_weight"] = _row_weight(row)
    return copied


def _row_weight(row: dict[str, Any]) -> Decimal:
    accepted = Decimal(max(1, int(row.get("acceptedRounds") or 0)))
    data = Decimal(int(row.get("sampleCount") or 0))
    compute = Decimal(int(row.get("localSteps") or 0)) * Decimal("4")
    runtime = Decimal(int(row.get("runtimeMs") or 0)) / Decimal("1000")
    health = _bounded_decimal(row.get("updateHealth"), Decimal("1.0"))
    cap = _bounded_decimal(row.get("capFactor"), Decimal("1.0"))
    return (data + compute + runtime) * accepted * health * cap


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def _blank_to_none(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value.strip()


def _decimal_env(value: str | None, default: Decimal) -> Decimal:
    if not value:
        return default
    return _quantize_money(Decimal(value))


def _decimal_env_raw(value: str | None, default: Decimal) -> Decimal:
    if not value:
        return default
    return Decimal(value)


def _decimal_payload(value: Any, default: Decimal) -> Decimal:
    if value is None:
        return default
    if isinstance(value, dict):
        value = value.get("value")
    return Decimal(str(value))


def _currency_payload(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("currency") or DEFAULT_CURRENCY)
    return DEFAULT_CURRENCY


def _bounded_decimal(value: Any, default: Decimal) -> Decimal:
    raw = _decimal_payload(value, default)
    if raw < Decimal("0"):
        return Decimal("0")
    if raw > Decimal("1"):
        return Decimal("1")
    return raw


def _quantize_money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def _money(value: Decimal) -> str:
    return format(_quantize_money(value), ".2f")


def _share(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP), "f")


def _copy_sale(sale: dict[str, Any]) -> dict[str, Any]:
    import copy

    return copy.deepcopy(sale)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _sanitize_error(exc: Exception) -> str:
    text = str(exc) or exc.__class__.__name__
    return text.replace("test_", "test_[redacted]").replace("live_", "live_[redacted]")
