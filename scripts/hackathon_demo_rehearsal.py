"""Integrated rehearsal gate for the #359 sovereign-economy demo.

The command keeps the older browser-federation rehearsal checks, then joins the
new Codex-Paris surfaces: surprise-meter fallback evidence, buyer sale,
mock/optional checkout, deterministic contribution ledger, reward split, public
non-claims, and local capture-asset status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

from lensemble.demo import FederatedDemoService
from lensemble.demo.economy import (
    MANDATORY_NON_CLAIMS,
    EconomyConfig,
    EconomyDemoService,
)

SURPRISE_EVIDENCE = Path("docs/evidence/lewm_tworooms_surprise.json")
SURPRISE_RESULT_CARD = Path("web/surprise-meter/data/result_card.json")
SURPRISE_TRAJECTORY = Path("web/surprise-meter/data/surprise_trajectory.json")
SURPRISE_OFFSET = Path("web/surprise-meter/fixtures/adapter_offset.json")
FEDERATED_APP = Path("web/federated-demo/app.mjs")
SURPRISE_INDEX = Path("web/surprise-meter/index.html")
RUNSHEET = Path(
    "docs/plans/hackathons/codex-hackathon-paris-june/"
    "sovereign-economy/04-demo-runsheet.md"
)
CAPTURE_CLIP = Path("runs/surprise/surprise-meter-capture.mp4")
CAPTURE_CARD = Path("runs/surprise/surprise-result-card.png")


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_needles(path: Path, needles: tuple[str, ...]) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise SystemExit(f"{path} is missing public rehearsal strings: {missing}")
    return {"path": str(path), "needles": list(needles)}


def _ffprobe_duration_seconds(path: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def _artifact(
    *,
    run_id: str,
    participant_id: str,
    round_index: int,
    model_revision_id: str,
    seed: int,
) -> dict[str, Any]:
    vector = [
        round(0.04 + seed * 0.003, 8),
        round(0.02 + seed * 0.002, 8),
        round(0.01 * ((seed % 3) - 1), 8),
        round(-0.015 + seed * 0.001, 8),
    ]
    l2_norm = sum(value * value for value in vector) ** 0.5
    hash_payload = {
        "runId": run_id,
        "participantId": participant_id,
        "round": round_index,
        "modelRevisionId": model_revision_id,
        "vector": vector,
        "seed": seed,
    }
    return {
        "schema": "browser-update/1",
        "source": "browser-local-surrogate",
        "runtime": "js-worker-tiny-jepa-v1",
        "runId": run_id,
        "participantId": participant_id,
        "round": round_index,
        "roundId": f"{run_id}:round-{round_index}",
        "modelRevisionId": model_revision_id,
        "shape": [len(vector)],
        "parameterCount": len(vector),
        "vector": vector,
        "sampleCount": 24,
        "localSteps": 8,
        "hash": _hash_payload(hash_payload),
        "l2Norm": round(l2_norm, 8),
        "clipNorm": 1.0,
        "loss": round(0.2 - seed * 0.005, 8),
        "probe": round(0.8 + seed * 0.005, 8),
        "runtimeMs": 10 + seed,
        "seed": seed,
        "simulated": False,
    }


def _join_four(
    service: FederatedDemoService, run_id: str, join_token: str
) -> list[dict[str, Any]]:
    return [
        service.join_run(
            run_id,
            join_token=join_token,
            display_name=f"phone-{index + 1}",
            session_id=f"rehearsal-session-{index + 1}",
        )
        for index in range(4)
    ]


def _submit(
    service: FederatedDemoService,
    run_id: str,
    participant: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    snapshot = service.snapshot(run_id)
    service.update_progress(
        run_id,
        participant["participantId"],
        participant_token=participant["participantToken"],
        progress=1.0,
    )
    return service.submit_update(
        run_id,
        participant["participantId"],
        participant_token=participant["participantToken"],
        artifact=_artifact(
            run_id=run_id,
            participant_id=participant["participantId"],
            round_index=snapshot["round"],
            model_revision_id=snapshot["currentModelRevisionId"],
            seed=seed,
        ),
    )


def _happy_path(service: FederatedDemoService) -> dict[str, Any]:
    run = service.create_run({"maxParticipants": 4, "quorum": 4, "rounds": 1})
    participants = _join_four(service, run["id"], run["joinToken"])
    service.start_run(run["id"])
    result: dict[str, Any] = {}
    for index, participant in enumerate(participants):
        result = _submit(service, run["id"], participant, seed=index + 1)
    evidence = service.export_evidence(run["id"])
    assert result["run"]["state"] == "completed"
    assert evidence["modelRevisionRefs"]
    assert any(
        artifact["kind"] == "inference-model" for artifact in evidence["artifacts"]
    )
    return {
        "runId": run["id"],
        "state": evidence["state"],
        "revision": evidence["modelRevisionRefs"][-1]["modelRevisionId"],
        "participants": len(evidence["participants"]),
        "events": len(evidence["eventTrace"]),
    }


def _dropout_path(service: FederatedDemoService) -> dict[str, Any]:
    run = service.create_run({"maxParticipants": 4, "quorum": 3, "rounds": 1})
    participants = _join_four(service, run["id"], run["joinToken"])
    service.start_run(run["id"])
    service.drop_participant(
        run["id"], participants[-1]["participantId"], reason="rehearsal dropout"
    )
    result: dict[str, Any] = {}
    for index, participant in enumerate(participants[:3]):
        result = _submit(service, run["id"], participant, seed=index + 10)
    evidence = service.export_evidence(run["id"])
    assert result["run"]["state"] == "completed"
    assert any(
        event["kind"] == "participant.dropped" for event in evidence["eventTrace"]
    )
    return {
        "runId": run["id"],
        "state": evidence["state"],
        "revision": evidence["modelRevisionRefs"][-1]["modelRevisionId"],
        "participants": len(evidence["participants"]),
        "dropped": [
            participant["id"]
            for participant in evidence["participants"]
            if participant["state"] == "dropped"
        ],
    }


def _run_surprise_gate(offset_out: Path) -> dict[str, Any]:
    offset_out.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/surprise/rehearsal.py",
            "--offset-out",
            str(offset_out),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(result.stdout + result.stderr)
    report = json.loads(result.stdout)
    if report.get("ok") is not True:
        raise SystemExit(f"surprise rehearsal failed: {result.stdout}")
    return report


def _validate_surprise_card() -> dict[str, Any]:
    evidence = _load_json(SURPRISE_EVIDENCE)
    result_card = _load_json(SURPRISE_RESULT_CARD)
    trajectory = _load_json(SURPRISE_TRAJECTORY)
    offset = json.loads(SURPRISE_OFFSET.read_text(encoding="utf-8"))

    if evidence.get("schema") != "lewm-surprise/1" or evidence.get("passes") is not True:
        raise SystemExit(f"{SURPRISE_EVIDENCE} is not a passing lewm-surprise/1 card")
    if result_card.get("schema") != "lewm-surprise-result-card/1":
        raise SystemExit(f"{SURPRISE_RESULT_CARD} has the wrong schema")
    if trajectory.get("schema") != "lewm-surprise-traj/1":
        raise SystemExit(f"{SURPRISE_TRAJECTORY} has the wrong schema")
    if not isinstance(offset, list) or len(offset) != 12512:
        raise SystemExit(f"{SURPRISE_OFFSET} must contain the 12,512-value adapter offset")

    display = result_card.get("display", {})
    for key, expected in {
        "thisRun": "+12.3%",
        "seedMean": "+16.8%",
        "seedWorst": "+5.4%",
    }.items():
        if display.get(key) != expected:
            raise SystemExit(f"{SURPRISE_RESULT_CARD} display.{key} must be {expected}")

    return {
        "evidence": str(SURPRISE_EVIDENCE),
        "resultCard": str(SURPRISE_RESULT_CARD),
        "trajectory": str(SURPRISE_TRAJECTORY),
        "offset": str(SURPRISE_OFFSET),
        "offsetLength": len(offset),
        "trajectorySteps": len(trajectory.get("steps", [])),
        "display": display,
        "nonClaims": result_card.get("nonClaims", []),
    }


def _money_cents(amount: dict[str, Any]) -> int:
    return int((Decimal(str(amount["value"])) * Decimal("100")).to_integral_value())


def _economy_path(
    service: FederatedDemoService,
    *,
    run_id: str,
    revision: str,
    public_root_url: str,
    payment_mode: str,
) -> dict[str, Any]:
    config = (
        EconomyConfig.from_env(public_base_url=public_root_url)
        if payment_mode == "auto"
        else EconomyConfig(public_base_url=public_root_url, force_mock=True)
    )
    economy = EconomyDemoService(config=config)
    sale = economy.create_sale(
        {
            "saleId": f"sale_{run_id}",
            "runId": run_id,
            "modelRevisionId": revision,
        },
        run_snapshot=service.snapshot(run_id),
    )
    sale = economy.create_payment(
        sale["saleId"],
        {
            "mode": "auto" if payment_mode == "auto" else "mock",
            "allowMockFallback": True,
        },
    )
    sale = economy.refresh_status(
        sale["saleId"], {"markPaid": sale["payment"]["mode"] != "mollie-test"}
    )

    participant_total = sum(
        _money_cents(row["reward"]) for row in sale["ledger"]["participantRewards"]
    )
    orchestrator = _money_cents(sale["ledger"]["orchestratorReward"])
    if orchestrator + participant_total != _money_cents(sale["saleAmount"]):
        raise SystemExit("economy rehearsal ledger does not balance to sale amount")
    if set(sale["nonClaims"]) < set(MANDATORY_NON_CLAIMS):
        raise SystemExit("economy sale is missing mandatory non-claims")

    return {
        "schema": sale["schema"],
        "saleId": sale["saleId"],
        "runId": sale["runId"],
        "modelRevisionId": sale["modelRevisionId"],
        "paymentMode": sale["payment"]["mode"],
        "paymentStatus": sale["payment"]["status"],
        "saleAmount": sale["saleAmount"],
        "checkoutAmount": sale["checkoutAmount"],
        "orchestratorShare": sale["orchestratorShare"],
        "orchestratorReward": sale["ledger"]["orchestratorReward"],
        "communityPool": sale["communityPool"],
        "participantRewards": sale["ledger"]["participantRewards"],
        "nonClaims": sale["nonClaims"],
    }


def _capture_assets(*, require_capture: bool) -> dict[str, Any]:
    assets: dict[str, Any] = {
        "clip": {"path": str(CAPTURE_CLIP), "present": CAPTURE_CLIP.is_file()},
        "resultCardImage": {"path": str(CAPTURE_CARD), "present": CAPTURE_CARD.is_file()},
    }
    if CAPTURE_CLIP.is_file():
        duration = _ffprobe_duration_seconds(CAPTURE_CLIP)
        assets["clip"].update(
            {
                "sha256": _hash_file(CAPTURE_CLIP),
                "durationSeconds": duration,
                "maxSeconds": 20,
            }
        )
        if duration is not None and duration > 20:
            raise SystemExit(f"{CAPTURE_CLIP} is longer than 20 seconds")
    elif require_capture:
        raise SystemExit(f"required capture clip is missing: {CAPTURE_CLIP}")

    if CAPTURE_CARD.is_file():
        assets["resultCardImage"]["sha256"] = _hash_file(CAPTURE_CARD)
    elif require_capture:
        raise SystemExit(f"required result card image is missing: {CAPTURE_CARD}")
    return assets


def _claim_surfaces() -> dict[str, Any]:
    return {
        "economyDashboard": _require_needles(
            FEDERATED_APP,
            (
                "Sovereign economy",
                "Model-quality proof",
                "+12.3%",
                "+16.8%",
                "+5.4%",
                "Simulation only: no legal payout",
                "no raw data or Mollie secret leaves the server",
            ),
        ),
        "surpriseMeter": _require_needles(
            SURPRISE_INDEX,
            (
                "Held-out error drop from federated adapter continuation on a frozen checkpoint",
            ),
        ),
        "runsheet": _require_needles(
            RUNSHEET,
            (
                "federated adapter-continuation",
                "+12.3% this run, +16.8% mean, +5.4% worst seed",
                "simulation-only, no legal payout",
                "no\n  DP/secure aggregation in demo path",
            ),
        ),
    }


def _demo_card(
    *,
    economy: dict[str, Any],
    surprise: dict[str, Any],
    captures: dict[str, Any],
) -> dict[str, Any]:
    card = {
        "schema": "sovereign-economy-demo-card/1",
        "salePrice": economy["saleAmount"],
        "checkoutAmount": economy["checkoutAmount"],
        "orchestratorShare": economy["orchestratorShare"],
        "orchestratorReward": economy["orchestratorReward"],
        "communityPool": economy["communityPool"],
        "participantRewards": economy["participantRewards"],
        "surprise": {
            "thisRun": surprise["display"]["thisRun"],
            "seedMean": surprise["display"]["seedMean"],
            "seedWorst": surprise["display"]["seedWorst"],
        },
        "nonClaims": {
            "economy": economy["nonClaims"],
            "surprise": surprise["nonClaims"],
        },
        "fallbackAssets": {
            "offset": surprise["offset"],
            "trajectory": surprise["trajectory"],
            "resultCard": surprise["resultCard"],
        },
        "captureAssets": captures,
    }
    for needle in ("+12.3%", "+16.8%", "+5.4%"):
        if needle not in json.dumps(card):
            raise SystemExit(f"demo card missing surprise display {needle}")
    for needle in ("1000000.00", "200000.00", "800000.00"):
        if needle not in json.dumps(card):
            raise SystemExit(f"demo card missing economy value {needle}")
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the #359 demo rehearsal gate")
    parser.add_argument(
        "--public-base-url",
        default="https://demo.example/web/federated-demo",
        help="base URL recorded in generated join/WSS URLs",
    )
    parser.add_argument(
        "--public-root-url",
        default="https://demo.example",
        help="root URL used for checkout redirects in the economy card",
    )
    parser.add_argument(
        "--payment-mode",
        choices=("mock", "auto"),
        default="mock",
        help="mock is deterministic; auto uses server-side Mollie env when present",
    )
    parser.add_argument(
        "--surprise-offset-out",
        type=Path,
        default=Path("runs/surprise/integrated_rehearsal_offset.json"),
        help="throwaway sidecar written by the synthetic surprise rehearsal",
    )
    parser.add_argument(
        "--require-capture",
        action="store_true",
        help="fail unless local <=20 s clip and result-card image are present",
    )
    args = parser.parse_args()
    service = FederatedDemoService(
        public_base_url=args.public_base_url,
        public_demo=True,
        deployment_target="rehearsal",
        transport_mode="websocket-primary",
    )
    happy = _happy_path(service)
    dropout = _dropout_path(service)
    surprise_gate = _run_surprise_gate(args.surprise_offset_out)
    surprise_card = _validate_surprise_card()
    economy = _economy_path(
        service,
        run_id=happy["runId"],
        revision=happy["revision"],
        public_root_url=args.public_root_url,
        payment_mode=args.payment_mode,
    )
    captures = _capture_assets(require_capture=args.require_capture)
    report = {
        "ok": True,
        "schema": "sovereign-economy-rehearsal/1",
        "autonomousRung": "C",
        "rungReason": (
            "recorded surprise trajectory plus mock checkout and deterministic ledger; "
            "use the runsheet for live browser and credentialed Mollie rungs"
        ),
        "happyPath": happy,
        "dropoutPath": dropout,
        "surpriseGate": surprise_gate,
        "surpriseCard": surprise_card,
        "economy": economy,
        "claimSurfaces": _claim_surfaces(),
        "demoCard": _demo_card(
            economy=economy,
            surprise=surprise_card,
            captures=captures,
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
