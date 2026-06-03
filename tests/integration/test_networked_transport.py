"""Networked control-plane transport — the four ControlMessages over a MessageChannel (#45).

Covers the RFC-0013 §5 acceptance criteria for the *networked* realization of the operation-oriented
``Transport`` Protocol (``lensemble.federation.transport.Transport`` — the SAME seam #42/#43 consume).
``NetworkedTransport`` drops into ``Coordinator(cfg, transport=...)`` unchanged, realizing each operation
(``broadcast_round_open`` / ``collect_updates`` / ``fetch_params`` / ...) by exchanging the four
RFC-0013 §5 ``ControlMessage`` s (``RoundOpen`` / ``Commitment`` / ``Update`` / ``RoundClose``) over a
pluggable low-level ``MessageChannel`` (``send`` / ``recv`` / ``broadcast`` / ``peers``). ``LoopbackChannel``
is the in-process, in-memory realization of that wire layer (the Stage-C network's testable stand-in).

Acceptance criteria exercised here:

- **Interchangeability + identical aggregation hash** — a ``Coordinator`` round driven over a
  ``NetworkedTransport`` + ``LoopbackChannel`` commits a ``global_state_hash()`` BIT-IDENTICAL to the SAME
  round driven over ``InProcessTransport`` on the same seed/updates (the outer step is deterministic, so
  the equality is exact). This proves both interchangeability and the "identical aggregation hash" line.
- **Malformed payload rejected at ingress** — a ``Commitment``/``Update`` with a missing/extra field or a
  too-new ``schema_version`` raises the typed error (``ValidationError`` / ``SchemaVersionMismatch``) and
  the round state does NOT advance (the bad update is not counted).
- **Unbound ``Δ_c`` → ``CommitmentMismatch``** — an ``Update`` whose ``dataset_root`` does not match the
  participant's committed ``R_c`` raises ``CommitmentMismatch`` (``COMMITMENT_MISMATCH``, never swallowed).
- **Residency (``INV-RESIDENCY``)** — none of the four messages can carry a raw observation/action/
  embedding: building an ``Update`` from a non-``PseudoGradient`` raw-tensor payload raises
  ``ResidencyViolation``, and a serialized ``Update`` carries only JSON-native scalars/lists (no tensor).
- **``recv`` timeout returns ``None``** — a ``recv`` on an empty channel past the budget returns ``None``.
- **``peers()`` / ``broadcast``** — a broadcast reaches every connected peer.

NOTE: this lives in tests/integration (NOT tests/federation): the conventions §8 CI gate scans
tests/{unit,property,integration,ml,e2e,regression}; the issue's `tests/federation/test_transport.py`
path would NOT be collected, so the suite is placed in a scanned directory (mirrors the same note in
tests/ml/test_coordinator.py). Dims/participants/rounds are kept tiny.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pytest
import torch
from pydantic import ValidationError

from lensemble.config import LensembleConfig
from lensemble.contracts import WMCP_VERSION
from lensemble.errors import (
    CommitmentMismatch,
    LensembleErrorCode,
    ResidencyViolation,
    SchemaVersionMismatch,
)
from lensemble.federation import (
    Coordinator,
    InProcessTransport,
    LoopbackChannel,
    NetworkedTransport,
    RoundState,
    build_pseudogradient,
)
from lensemble.federation.messages import (
    CONTROL_MESSAGE_SCHEMA_VERSION,
    Commitment,
    RoundClose,
    RoundOpen,
    Update,
    from_pseudogradient,
    parse_control_message,
    to_delta_tensor,
)
from lensemble.federation.pseudogradient import PseudoGradient
from lensemble.model import build_encoder, build_predictor

_COORD = "coordinator"

# --- the tiny-CPU coordinator config + toy-update helpers (mirrors tests/ml/test_coordinator.py: the
# cross-test path is not importable — the `tests` tree has no package __init__, so the config bridge is
# duplicated here, the same way test_coordinator.py duplicates _ModelConfig from test_participant.py). The
# interchangeability test must stage the SAME updates the InProcessTransport run stages, byte-for-byte. ---

_D = 8
_NUM_TOKENS = (
    4  # (num_frames//tubelet) * (image_size//patch_size)**2 = (2//2)*(4//2)**2 = 4
)
_T, _C, _H, _W = 2, 3, 4, 4
_ROOT = b"\x2a" * 32  # a fixed 32-byte dataset root R_c (INV-COMMIT-BINDING)


@dataclass(frozen=True)
class _ModelConfig:
    # real ModelConfig fields (keep config_hash / manifest well-formed)
    encoder: str = "vjepa2-vit-l"
    warm_start_release: str = "vjepa2-2.0"
    latent_dim: int = _D
    num_tokens: int = _NUM_TOKENS
    predictor_depth: int = 1
    predictor_width: int = _D
    wmcp_version: str = WMCP_VERSION
    encoder_frozen: bool = False
    # V-JEPA shape fields the build_* functions read
    d: int = _D
    in_channels: int = _C
    num_frames: int = _T
    image_size: int = _H
    patch_size: int = 2
    tubelet: int = 2
    depth: int = 1
    num_heads: int = 2
    cond_dim: int = _D


def _cfg(**overrides: object) -> LensembleConfig:
    """A coordinator-mode config with a tiny model and a low quorum so a 2-participant round runs."""
    base = LensembleConfig()
    fed = dataclasses.replace(
        base.federation,
        num_rounds=2,
        outer_lr=0.7,
        outer_nesterov_momentum=0.9,
        fault_tolerance_min_participants=2,
    )
    model = _ModelConfig()  # type: ignore[arg-type]
    cfg = dataclasses.replace(base, model=model, federation=fed, run_mode="coordinator")
    return dataclasses.replace(cfg, **overrides)  # type: ignore[arg-type]


def _toy_update(
    cfg: LensembleConfig,
    *,
    round_index: int,
    seed: int,
    scale: float = 1e-2,
) -> PseudoGradient:
    """A tiny θ⊕φ-sized PseudoGradient from per-group toy deltas (canonical encoder.*/predictor.*)."""
    torch.manual_seed(seed)
    enc = build_encoder(cfg)
    pred = build_predictor(cfg)
    gen = torch.Generator().manual_seed(seed)
    param_deltas: dict[str, torch.Tensor] = {}
    for name, t in enc.state_dict().items():
        param_deltas[f"encoder.{name}"] = scale * torch.randn(
            t.shape, generator=gen, dtype=torch.float32
        )
    for name, t in pred.state_dict().items():
        param_deltas[f"predictor.{name}"] = scale * torch.randn(
            t.shape, generator=gen, dtype=torch.float32
        )
    return build_pseudogradient(
        param_deltas, dataset_root=_ROOT, round_index=round_index, clipped=True
    )


# --------------------------------------------------------------------------------------------------
# the message layer (RFC-0013 §5): the four ControlMessages + parse_control_message
# --------------------------------------------------------------------------------------------------


def test_round_open_is_frozen_and_schema_versioned() -> None:
    msg = RoundOpen(
        theta_ref_hash="a" * 64,
        theta_ref_locator="artifact://round-00000/encoder",
        phi_ref_hash="b" * 64,
        phi_ref_locator="artifact://round-00000/predictor",
        round_index=0,
        sketch_seed=7,
        probe_hash="c" * 64,
        landmark_hashes=("d" * 64,),
        inner_horizon=2,
    )
    assert msg.schema_version == CONTROL_MESSAGE_SCHEMA_VERSION
    with pytest.raises(ValidationError):
        msg.round_index = 1  # type: ignore[misc]  # frozen


def test_parse_control_message_round_trips_each_kind() -> None:
    open_msg = RoundOpen(
        theta_ref_hash="a" * 64,
        theta_ref_locator="artifact://round-00000/encoder",
        phi_ref_hash="b" * 64,
        phi_ref_locator="artifact://round-00000/predictor",
        round_index=0,
        sketch_seed=7,
        probe_hash="c" * 64,
        landmark_hashes=(),
        inner_horizon=2,
    )
    commit_msg = Commitment(
        participant_id="p0", round_index=0, dataset_root=_ROOT.hex()
    )
    close_msg = RoundClose(round_index=0, global_model_hash="e" * 64)
    for original in (open_msg, commit_msg, close_msg):
        parsed = parse_control_message(original.model_dump())
        assert parsed == original
        assert type(parsed) is type(original)


def test_parse_control_message_gates_schema_version_first() -> None:
    # A too-new schema_version raises SchemaVersionMismatch BEFORE field validation (fail-closed loader).
    raw = Commitment(
        participant_id="p0", round_index=0, dataset_root=_ROOT.hex()
    ).model_dump()
    raw["schema_version"] = CONTROL_MESSAGE_SCHEMA_VERSION + 1
    with pytest.raises(SchemaVersionMismatch) as exc:
        parse_control_message(raw)
    assert exc.value.code == LensembleErrorCode.SCHEMA_VERSION_MISMATCH


def test_parse_control_message_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        parse_control_message({"schema_version": 1, "kind": "not-a-message"})


def test_malformed_commitment_missing_field_raises() -> None:
    raw = {"schema_version": 1, "kind": "commitment", "participant_id": "p0"}  # no root
    with pytest.raises(ValidationError):
        parse_control_message(raw)


def test_malformed_update_extra_field_raises() -> None:
    cfg = _cfg()
    pg = _toy_update(cfg, round_index=0, seed=100)
    raw = from_pseudogradient(pg, participant_id="p0").model_dump()
    raw["surprise"] = 1  # extra="forbid"
    with pytest.raises(ValidationError):
        parse_control_message(raw)


def test_update_round_trips_through_pseudogradient() -> None:
    cfg = _cfg()
    pg = _toy_update(cfg, round_index=0, seed=100)
    update = from_pseudogradient(pg, participant_id="p0")
    assert update.participant_id == "p0"
    assert update.round_index == 0
    assert update.dataset_root == _ROOT.hex()
    # the masked Δ_c crosses as a JSON-native finite list of floats — never a tensor
    assert isinstance(update.delta, tuple)
    assert all(isinstance(x, float) for x in update.delta)
    recovered = to_delta_tensor(update)
    assert torch.allclose(recovered, pg.delta, atol=1e-6)


def test_update_delta_must_be_finite() -> None:
    with pytest.raises(ValidationError):
        Update(
            participant_id="p0",
            round_index=0,
            dataset_root=_ROOT.hex(),
            delta=(float("nan"), 0.0),
            l2_norm=0.0,
        )


# --------------------------------------------------------------------------------------------------
# residency (INV-RESIDENCY): no raw tensor / observation can enter a message
# --------------------------------------------------------------------------------------------------


def test_from_pseudogradient_rejects_non_pseudogradient_payload() -> None:
    # A bare tensor is NOT a PseudoGradient carrier; the residency guard fails closed.
    class _FakeRawPayload:
        delta = torch.randn(4)
        l2_norm = 1.0
        dataset_root = _ROOT
        round_index = 0

    with pytest.raises(ResidencyViolation) as exc:
        from_pseudogradient(_FakeRawPayload(), participant_id="p0")  # type: ignore[arg-type]
    assert exc.value.code == LensembleErrorCode.RESIDENCY_VIOLATION


def test_serialized_update_carries_no_tensor() -> None:
    import json

    cfg = _cfg()
    pg = _toy_update(cfg, round_index=0, seed=100)
    update = from_pseudogradient(pg, participant_id="p0")
    # model_dump_json must succeed (pure JSON-native scalars/lists) and carry no tensor.
    payload = json.loads(update.model_dump_json())
    assert isinstance(payload["delta"], list)
    assert "Tensor" not in update.model_dump_json()


# --------------------------------------------------------------------------------------------------
# the channel layer (RFC-0013 §5): LoopbackChannel send/recv/broadcast/peers
# --------------------------------------------------------------------------------------------------


def test_recv_timeout_returns_none_on_empty_channel() -> None:
    # The §5 contract: recv on an empty inbox past the budget returns None (never blocks forever).
    pair = LoopbackChannel.connected_pair(_COORD, "p0")
    coord_channel = pair[_COORD]
    assert coord_channel.recv(timeout_s=0.0) is None


def test_send_and_recv_round_trip_between_peers() -> None:
    pair = LoopbackChannel.connected_pair(_COORD, "p0")
    msg = Commitment(participant_id="p0", round_index=0, dataset_root=_ROOT.hex())
    pair["p0"].send(_COORD, msg)
    received = pair[_COORD].recv(timeout_s=0.1)
    assert received == msg
    # the coordinator's inbox is now drained
    assert pair[_COORD].recv(timeout_s=0.0) is None


def test_peers_lists_connected_peers() -> None:
    mesh = LoopbackChannel.connected_mesh(_COORD, "p0", "p1")
    assert set(mesh[_COORD].peers()) == {"p0", "p1"}
    assert set(mesh["p0"].peers()) == {_COORD, "p1"}


def test_broadcast_reaches_all_peers() -> None:
    mesh = LoopbackChannel.connected_mesh(_COORD, "p0", "p1")
    close_msg = RoundClose(round_index=0, global_model_hash="e" * 64)
    mesh[_COORD].broadcast(close_msg)
    assert mesh["p0"].recv(timeout_s=0.1) == close_msg
    assert mesh["p1"].recv(timeout_s=0.1) == close_msg
    # broadcast does not echo to the sender's own inbox
    assert mesh[_COORD].recv(timeout_s=0.0) is None


# --------------------------------------------------------------------------------------------------
# NetworkedTransport satisfies the operation-oriented Transport over a channel
# --------------------------------------------------------------------------------------------------


def _participant_submits(
    transport: NetworkedTransport,
    cfg: LensembleConfig,
    *,
    participant_ids: list[str],
    round_index: int,
    seed_base: int = 100,
    roots: dict[str, bytes] | None = None,
) -> dict[str, PseudoGradient]:
    """Each participant commits its R_c then submits its toy Update over the channel (the wire path)."""
    staged: dict[str, PseudoGradient] = {}
    for i, pid in enumerate(participant_ids):
        root = (roots or {}).get(pid, _ROOT)
        pg = dataclasses.replace(
            _toy_update(cfg, round_index=round_index, seed=seed_base + i),
            dataset_root=root,
        )
        transport.commit_root(participant_id=pid, round_index=round_index, root=root)
        transport.submit_update(participant_id=pid, round_index=round_index, update=pg)
        staged[pid] = pg
    return staged


def test_networked_transport_is_a_transport_instance() -> None:
    from lensemble.federation.transport import Transport

    channel = LoopbackChannel.connected_mesh(_COORD, "p0")[_COORD]
    transport = NetworkedTransport(channel=channel, coordinator_id=_COORD)
    assert isinstance(transport, Transport)


def test_fetch_params_round_trips_and_hash_verifies() -> None:
    from lensemble.errors import CheckpointIntegrityError

    cfg = _cfg()
    mesh = LoopbackChannel.connected_mesh(_COORD, "p0")
    transport = NetworkedTransport(channel=mesh[_COORD], coordinator_id=_COORD)
    coord = Coordinator(cfg, transport=transport)
    gs = coord.global_state()
    theta = transport.fetch_params(gs.theta_ref)
    phi = transport.fetch_params(gs.phi_ref)
    assert sum(t.numel() for t in theta.values()) > 0
    assert sum(t.numel() for t in phi.values()) > 0
    # recover_global_state returns the last broadcast GlobalState
    assert transport.recover_global_state(participant_id="p0") == gs
    # a ref with an unknown content hash fails closed
    bogus = dataclasses.replace(gs.theta_ref, locator="artifact://nope/encoder")
    bogus = dataclasses.replace(bogus, content_hash="f" * 64)
    with pytest.raises(CheckpointIntegrityError):
        transport.fetch_params(bogus)


# --- ACCEPTANCE: interchangeability + identical aggregation hash on the same seed/updates ---


def test_networked_round_commits_identical_hash_to_inprocess() -> None:
    cfg = _cfg()
    pids = ["c0", "c1"]

    # in-process baseline: stage the toy updates, run one round, read the committed hash.
    in_transport = InProcessTransport()
    in_coord = Coordinator(cfg, transport=in_transport)
    for i, pid in enumerate(pids):
        pg = _toy_update(cfg, round_index=0, seed=100 + i)
        in_transport.submit_update(participant_id=pid, round_index=0, update=pg)
    in_coord.run(1)
    in_hash = in_coord.global_state_hash()

    # networked: drive the SAME round over NetworkedTransport + LoopbackChannel.
    mesh = LoopbackChannel.connected_mesh(_COORD, *pids)
    net_transport = NetworkedTransport(channel=mesh[_COORD], coordinator_id=_COORD)
    net_coord = Coordinator(cfg, transport=net_transport)
    _participant_submits(net_transport, cfg, participant_ids=pids, round_index=0)
    net_coord.run(1)
    net_hash = net_coord.global_state_hash()

    assert net_coord.round_state() == RoundState.CLOSED
    assert net_hash == in_hash  # bit-identical committed (θ_{t+1}, φ_{t+1}) hash


# --- ACCEPTANCE: malformed ingress payload → typed error, state does not advance ---


def test_malformed_update_at_ingress_does_not_advance_state() -> None:
    cfg = _cfg()
    pids = ["c0", "c1"]
    mesh = LoopbackChannel.connected_mesh(_COORD, *pids)
    net_transport = NetworkedTransport(channel=mesh[_COORD], coordinator_id=_COORD)
    coord = Coordinator(cfg, transport=net_transport)
    h_before = coord.global_state_hash()

    # one good update; one malformed raw payload pushed directly onto the coordinator inbox
    pg = _toy_update(cfg, round_index=0, seed=100)
    net_transport.commit_root(participant_id="c0", round_index=0, root=_ROOT)
    net_transport.submit_update(participant_id="c0", round_index=0, update=pg)
    # _RawDictMessage models a hostile peer that puts an unvalidated dict on the wire (it is intentionally
    # NOT a ControlMessage; the ingress validator re-parses it and rejects it — that is the point).
    mesh["c1"].send(_COORD, _RawDictMessage({"schema_version": 1, "kind": "update"}))  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        coord.run(1)
    # the round did not advance: the committed global hash is unchanged.
    assert coord.global_state_hash() == h_before


def test_too_new_schema_version_at_ingress_raises_and_does_not_advance() -> None:
    cfg = _cfg()
    pids = ["c0", "c1"]
    mesh = LoopbackChannel.connected_mesh(_COORD, *pids)
    net_transport = NetworkedTransport(channel=mesh[_COORD], coordinator_id=_COORD)
    coord = Coordinator(cfg, transport=net_transport)
    h_before = coord.global_state_hash()

    pg = _toy_update(cfg, round_index=0, seed=100)
    net_transport.commit_root(participant_id="c0", round_index=0, root=_ROOT)
    net_transport.submit_update(participant_id="c0", round_index=0, update=pg)
    raw = from_pseudogradient(pg, participant_id="c1").model_dump()
    raw["schema_version"] = CONTROL_MESSAGE_SCHEMA_VERSION + 1
    mesh["c1"].send(_COORD, _RawDictMessage(raw))  # type: ignore[arg-type]  # hostile raw dict on the wire

    with pytest.raises(SchemaVersionMismatch):
        coord.run(1)
    assert coord.global_state_hash() == h_before


# --- ACCEPTANCE: Δ_c not bound to a valid R_c → CommitmentMismatch (never swallowed) ---


def test_unbound_delta_raises_commitment_mismatch() -> None:
    cfg = _cfg()
    pids = ["c0", "c1"]
    mesh = LoopbackChannel.connected_mesh(_COORD, *pids)
    net_transport = NetworkedTransport(channel=mesh[_COORD], coordinator_id=_COORD)
    coord = Coordinator(cfg, transport=net_transport)
    h_before = coord.global_state_hash()

    # c0 commits R_c = _ROOT but submits a delta bound to a DIFFERENT root → binding mismatch.
    wrong_root = b"\x99" * 32
    net_transport.commit_root(participant_id="c0", round_index=0, root=_ROOT)
    pg = dataclasses.replace(
        _toy_update(cfg, round_index=0, seed=100), dataset_root=wrong_root
    )
    net_transport.submit_update(participant_id="c0", round_index=0, update=pg)

    with pytest.raises(CommitmentMismatch) as exc:
        coord.run(1)
    assert exc.value.code == LensembleErrorCode.COMMITMENT_MISMATCH
    assert coord.global_state_hash() == h_before


def test_uncommitted_participant_update_raises_commitment_mismatch() -> None:
    cfg = _cfg()
    pids = ["c0", "c1"]
    mesh = LoopbackChannel.connected_mesh(_COORD, *pids)
    net_transport = NetworkedTransport(channel=mesh[_COORD], coordinator_id=_COORD)
    coord = Coordinator(cfg, transport=net_transport)

    # c0 submits an Update WITHOUT ever sending a Commitment — no committed R_c to bind against.
    pg = _toy_update(cfg, round_index=0, seed=100)
    net_transport.submit_update(participant_id="c0", round_index=0, update=pg)
    with pytest.raises(CommitmentMismatch):
        coord.run(1)


# --- collect_updates only returns the matching round's updates ---


def test_collect_updates_only_returns_matching_round() -> None:
    cfg = _cfg()
    pids = ["c0", "c1"]
    mesh = LoopbackChannel.connected_mesh(_COORD, *pids)
    net_transport = NetworkedTransport(channel=mesh[_COORD], coordinator_id=_COORD)
    staged = _participant_submits(
        net_transport, cfg, participant_ids=pids, round_index=0
    )
    collected = net_transport.collect_updates(0)
    assert set(collected) == set(pids)
    for pid in pids:
        assert torch.allclose(collected[pid].delta, staged[pid].delta, atol=1e-6)
    # a round with no submitted updates collects nothing
    assert dict(net_transport.collect_updates(5)) == {}


def test_full_commitment_then_update_wire_handshake() -> None:
    # The two-step §5 handshake over the wire: a participant SENDS a Commitment, then an Update; the
    # coordinator's collect_updates ingests the Commitment (records R_c) then binds the Update against it.
    cfg = _cfg()
    pids = ["c0", "c1"]
    mesh = LoopbackChannel.connected_mesh(_COORD, *pids)
    net_transport = NetworkedTransport(channel=mesh[_COORD], coordinator_id=_COORD)
    coord = Coordinator(cfg, transport=net_transport)
    h_before = coord.global_state_hash()

    for i, pid in enumerate(pids):
        pg = _toy_update(cfg, round_index=0, seed=100 + i)
        # the participant sends its Commitment (R_c) then its Update over the channel (no commit_root seam)
        mesh[pid].send(
            _COORD,
            Commitment(participant_id=pid, round_index=0, dataset_root=_ROOT.hex()),
        )
        mesh[pid].send(_COORD, from_pseudogradient(pg, participant_id=pid))
    # also drop a RoundClose echo on the coordinator inbox: it must be ignored at collect (not counted)
    mesh["c0"].send(_COORD, RoundClose(round_index=0, global_model_hash="e" * 64))

    coord.run(1)
    assert coord.round_state() == RoundState.CLOSED
    assert coord.global_state_hash() != h_before
    rec = coord.ledger_records()[-1]
    assert rec.participants == ("c0", "c1")


def test_register_records_participant_endpoint() -> None:
    channel = LoopbackChannel.connected_mesh(_COORD, "p0")[_COORD]
    transport = NetworkedTransport(channel=channel, coordinator_id=_COORD)
    # register is the control-plane bookkeeping #43's Participant.join calls; it does not touch the wire.
    transport.register("p0", "tcp://p0:9000")
    assert transport._registered["p0"] == "tcp://p0:9000"  # noqa: SLF001 — inspect the recorded endpoint


def test_collect_skips_update_for_a_different_round() -> None:
    # An Update whose round_index != the collected round is NOT part of THIS round's present set (it is
    # skipped, not bound), even though its participant committed an R_c — the per-round filter (§3/§5).
    cfg = _cfg()
    mesh = LoopbackChannel.connected_mesh(_COORD, "c0")
    net_transport = NetworkedTransport(channel=mesh[_COORD], coordinator_id=_COORD)
    Coordinator(
        cfg, transport=net_transport
    )  # opens round 0 (irrelevant; we collect directly)
    pg = dataclasses.replace(_toy_update(cfg, round_index=1, seed=100), round_index=1)
    mesh["c0"].send(
        _COORD, Commitment(participant_id="c0", round_index=1, dataset_root=_ROOT.hex())
    )
    mesh["c0"].send(_COORD, from_pseudogradient(pg, participant_id="c0"))
    # collecting round 0 sees the Commitment but skips the round-1 Update (different round).
    assert dict(net_transport.collect_updates(0)) == {}


def test_unknown_peer_send_raises_key_error() -> None:
    mesh = LoopbackChannel.connected_mesh(_COORD, "p0")
    msg = RoundClose(round_index=0, global_model_hash="e" * 64)
    with pytest.raises(KeyError):
        mesh[_COORD].send("not-a-peer", msg)


def test_connected_mesh_rejects_duplicate_node_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        LoopbackChannel.connected_mesh("a", "a")


def test_channel_node_id_property() -> None:
    mesh = LoopbackChannel.connected_mesh(_COORD, "p0")
    assert mesh[_COORD].node_id == _COORD


def test_recover_global_state_fails_closed_before_any_round() -> None:
    from lensemble.errors import CheckpointIntegrityError

    channel = LoopbackChannel.connected_mesh(_COORD, "p0")[_COORD]
    transport = NetworkedTransport(channel=channel, coordinator_id=_COORD)
    # no RoundOpen broadcast yet: there is no committed GlobalState to recover.
    with pytest.raises(CheckpointIntegrityError):
        transport.recover_global_state(participant_id="p0")


def test_fetch_params_fails_closed_on_corrupt_store() -> None:
    from lensemble.errors import CheckpointIntegrityError

    cfg = _cfg()
    channel = LoopbackChannel.connected_mesh(_COORD, "p0")[_COORD]
    transport = NetworkedTransport(channel=channel, coordinator_id=_COORD)
    coord = Coordinator(cfg, transport=transport)
    gs = coord.global_state()
    # Overwrite the store under theta_ref's hash with MISMATCHING weights: the recomputed hash differs, so
    # fetch_params fails closed (INV-CHECKPOINT-HASH) — the recomputed-mismatch branch, not the missing one.
    transport._weights[gs.theta_ref.content_hash] = {  # noqa: SLF001 — tamper seam (tests only)
        "tampered": torch.zeros(3)
    }
    with pytest.raises(CheckpointIntegrityError):
        transport.fetch_params(gs.theta_ref)


def test_round_open_rejects_malformed_hash_and_locator() -> None:
    base = dict(
        theta_ref_hash="a" * 64,
        theta_ref_locator="artifact://round-00000/encoder",
        phi_ref_hash="b" * 64,
        phi_ref_locator="artifact://round-00000/predictor",
        round_index=0,
        sketch_seed=7,
        probe_hash="c" * 64,
        landmark_hashes=(),
        inner_horizon=2,
    )
    with pytest.raises(ValidationError):
        RoundOpen(**{**base, "theta_ref_hash": "not-hex"})  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        RoundOpen(**{**base, "theta_ref_locator": ""})  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        RoundOpen(**{**base, "landmark_hashes": ("short",)})  # type: ignore[arg-type]


def test_commitment_and_update_reject_malformed_root() -> None:
    with pytest.raises(ValidationError):
        Commitment(participant_id="p0", round_index=0, dataset_root="zz")
    with pytest.raises(ValidationError):
        Update(
            participant_id="p0",
            round_index=0,
            dataset_root="zz",
            delta=(0.0,),
            l2_norm=0.0,
        )


def test_update_rejects_non_finite_l2_norm() -> None:
    with pytest.raises(ValidationError):
        Update(
            participant_id="p0",
            round_index=0,
            dataset_root=_ROOT.hex(),
            delta=(1.0,),
            l2_norm=float("inf"),
        )


def test_round_close_rejects_malformed_hash() -> None:
    with pytest.raises(ValidationError):
        RoundClose(round_index=0, global_model_hash="short")


class _RawDictMessage:
    """A test-only stand-in: a node that pushes an unvalidated dict onto the wire (a hostile peer).

    ``NetworkedTransport.collect_updates`` calls ``parse_control_message`` on each received message's
    ``model_dump()``; this object yields the raw dict directly so a malformed/too-new payload reaches the
    ingress validator unchanged (the wire is untrusted — every payload is re-validated at ingress).
    """

    def __init__(self, raw: dict[str, object]) -> None:
        self._raw = raw

    def model_dump(self) -> dict[str, object]:
        return dict(self._raw)
