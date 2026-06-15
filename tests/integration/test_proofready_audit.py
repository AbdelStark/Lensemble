"""Proof-readiness audit — the v1.0 "proof-ready guarantees verified end-to-end" gate (RFC-0006). Issue #63.

This is the single integration-test audit that exercises ALL FIVE Phase-1 proof-ready disciplines of
[RFC-0006 §3](../../docs/rfcs/RFC-0006-verifiable-contribution.md) together over a tiny synthetic
federated round. It COMPOSES the already-shipped per-discipline primitives (it adds no public Python
symbol and re-implements none of them); each test asserts a POSITIVE path that the discipline holds and a
NEGATIVE path that the documented typed error / report fires and fails closed (04 §Error Model: the three
security-critical errors are never swallowed). It is referenced from the v1.0 reproducibility-package
release checklist ([09 §5.2](../../docs/spec/09-release-and-versioning.md#52-release-checklist-release-blocking-gates)).

The five RFC-0006 §3 disciplines audited here, by ``INV-*`` id (the audit entry point
``test_proofready_audit_covers_all_five_disciplines`` re-enumerates them so the audit is discoverable):

1. ``INV-AGG-DETERMINISM`` — the outer step is bitwise-reproducible. POSITIVE: a real ``Coordinator``
   round commits a bitwise-identical ``(θ_{t+1}, φ_{t+1})`` hash on a re-run with the same seed/inputs
   (backed by ``Coordinator`` + ``aggregation.determinism.assert_outer_step_deterministic``). NEGATIVE:
   an injected nondeterministic ``OuterOptimizer.average_deltas`` raises ``NonDeterministicAggregation``
   and the round ABORTs with the global hash unchanged (mirrors
   ``test_coordinator.py::test_corrupt_reduction_aborts_with_global_hash_unchanged``).
2. ``INV-CHECKPOINT-HASH`` — every committed ``(θ_t, φ_t)`` artifact's recomputed content hash matches
   its header with a valid ``parent_hash`` chain. POSITIVE: two artifacts committed in a parent chain
   load+verify and round ``t+1``'s ``parent_hash`` equals round ``t``'s ``content_hash`` (backed by
   ``artifacts.checkpoint.save_checkpoint`` / ``load_checkpoint`` / ``verify``). NEGATIVE: a tampered
   ``weights.safetensors`` byte makes ``load_checkpoint`` / ``verify`` raise ``CheckpointIntegrityError``.
3. ``INV-COMMIT-BINDING`` — each released ``Δ_c`` carries exactly one 32-byte ``dataset_root`` ``R_c``.
   POSITIVE: ``build_pseudogradient`` releases a single ``R_c`` and ``provenance.verify_binding`` accepts
   it against the committed root. NEGATIVE: a ``Δ_c`` declaring a wrong/foreign ``R_c`` raises
   ``CommitmentMismatch`` (security-critical, excluded from the sum, never swallowed).
4. ``INV-PROBE-PIN`` — the probe content hash matches the ``RoundOpen`` / ``GlobalState.probe_hash``
   commitment and landmark targets derive ONLY from ``f_ref``. POSITIVE: a ``build_probe`` probe's
   content hash equals the coordinator's broadcast ``probe_hash`` and its ``landmark_targets`` equal
   ``f_ref(landmarks)`` (and NOT a later-mutated encoder's). NEGATIVE: a probe whose recomputed content
   hash differs from its pin raises ``ProbeError`` (via ``verify_probe_pin``).
5. ``recompute_alignment`` (the free Phase-2 mechanism, RFC-0006 §4) — public recomputation reproduces
   the coordinator's alignment from public inputs alone. POSITIVE:
   ``recompute_alignment_claim(committed_weights, probe, expected=<honest claim>)`` returns
   ``matches_expected=True`` from the committed checkpoint + the pinned probe alone (the 32-token config
   so the honest LayerNorm-terminated encoder recovers ``Q* = I``). NEGATIVE: a perturbed expected claim
   (wrong ``procrustes_q_hash``) returns ``matches_expected=False``.

Out of scope (RFC-0006 §3 / conventions §12): implementing any discipline, and any Phase-2 proof itself
(Stwo circuit correctness, TEE attestation, Poseidon2 equivalence) — Stage D, not the v0.1–v1.0 suite.

NOTE on the tiny round. The audit reuses two independent toy fixtures, each as the matching per-discipline
``tests/ml`` test builds them (the ``tests/`` tree has no package ``__init__`` so cross-import fails — the
small config helpers are duplicated, as the other ml tests do, not imported):

- a coordinator-mode ``LensembleConfig`` with a 4-token ``_CoordModelConfig`` (mirrors
  ``test_coordinator.py``) drives the real ``Coordinator`` round for disciplines 1 + 4 (the broadcast
  ``GlobalState`` carries the live ``probe_hash``);
- a 32-token ``_EvalModelCfg`` ``SimpleNamespace`` (mirrors ``test_recompute_alignment.py``) commits a
  self-describing checkpoint + a pinned probe for disciplines 2 + 5, so the honest public recomputation
  recovers ``Q* = I`` (≈0 rotation angle).

Discipline 3 (``INV-COMMIT-BINDING``) is exercised at the primitive level over a real
``build_pseudogradient`` ``Δ_c`` + ``provenance.verify_binding`` (the same surface the ``Coordinator``
binds each ``Δ_c`` through in its ``ContributionRecord``): the binding check is a pure function of
``(committed_root, declared_root)`` with no round state, so the audit asserts it directly on the released
carrier's ``dataset_root`` rather than threading a wrong-root delta through a full coordinator commit.

Placed in tests/integration (the issue's named audit-entry-point path) — collected by the §8 CI gate
(``tests/{unit,property,integration,ml,e2e,regression}``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import Tensor

from lensemble.aggregation.determinism import assert_outer_step_deterministic
from lensemble.artifacts import load_checkpoint, save_checkpoint, verify
from lensemble.artifacts.checkpoint import model_arch_from_config
from lensemble.config import LensembleConfig
from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
from lensemble.data.dataset import EpisodeDataset
from lensemble.data.episode import Episode, Transition
from lensemble.data.probe import build_probe, save_probe, verify_probe_pin
from lensemble.errors import (
    CheckpointIntegrityError,
    CommitmentMismatch,
    LensembleErrorCode,
    NonDeterministicAggregation,
    ProbeError,
)
from lensemble.federation import (
    Coordinator,
    InProcessTransport,
    RoundState,
    build_pseudogradient,
)
from lensemble.federation.outer_optimizer import OuterOptimizer
from lensemble.federation.pseudogradient import PseudoGradient
from lensemble.model import build_encoder, build_predictor
from lensemble.model.encoder import snapshot_reference
from lensemble.provenance import commit_dataset, verify_binding
from lensemble.provenance.merkle import DIGEST_SIZE, CommitmentScheme
from lensemble.verify import recompute_alignment_claim

# The five RFC-0006 §3 proof-ready disciplines audited here, by INV-* id (re-enumerated by the audit
# entry-point test so the audit is a single discoverable surface).
_AUDITED_INVARIANTS = (
    "INV-AGG-DETERMINISM",  # 1. bitwise-reproducible outer step
    "INV-CHECKPOINT-HASH",  # 2. committed (θ, φ) content hash + parent_hash chain
    "INV-COMMIT-BINDING",  # 3. each Δ_c bound to exactly one R_c
    "INV-PROBE-PIN",  # 4. probe content hash == RoundOpen commitment; targets from f_ref only
    "INV-AGG-DETERMINISM (recompute_alignment)",  # 5. public recomputation from public inputs alone
)

_SCHEME = CommitmentScheme()


# =====================================================================================================
# Fixtures A — the coordinator-mode round (disciplines 1 + 4). Mirrors tests/ml/test_coordinator.py:
# a 4-token tiny CPU V-JEPA config + a low quorum so a 2-participant round runs.
# =====================================================================================================

_CD = 8
_CNUM_TOKENS = (
    4  # (num_frames//tubelet)*(image_size//patch_size)**2 = (2//2)*(4//2)**2 = 4
)
_CT, _CC, _CH, _CW = 2, 3, 4, 4
_ROOT = b"\x2a" * 32  # a fixed 32-byte dataset root R_c (INV-COMMIT-BINDING)


@dataclass(frozen=True)
class _CoordModelConfig:
    # real ModelConfig fields (keep config_hash / model_arch_from_config well-formed)
    encoder: str = "vjepa2-vit-l"
    warm_start_release: str = "vjepa2-2.0"
    latent_dim: int = _CD
    num_tokens: int = _CNUM_TOKENS
    predictor_depth: int = 1
    predictor_width: int = _CD
    wmcp_version: str = WMCP_VERSION
    encoder_frozen: bool = False
    # V-JEPA shape fields the build_* functions read
    d: int = _CD
    in_channels: int = _CC
    num_frames: int = _CT
    image_size: int = _CH
    patch_size: int = 2
    tubelet: int = 2
    depth: int = 1
    num_heads: int = 2
    cond_dim: int = _CD


def _coord_cfg(**overrides: object) -> LensembleConfig:
    """A coordinator-mode config with a tiny model and a low quorum so a 2-participant round runs."""
    base = LensembleConfig()
    fed = dataclasses.replace(
        base.federation,
        num_rounds=2,
        outer_lr=0.7,
        outer_nesterov_momentum=0.9,
        fault_tolerance_min_participants=2,
    )
    model = _CoordModelConfig()  # type: ignore[arg-type]
    cfg = dataclasses.replace(base, model=model, federation=fed, run_mode="coordinator")
    return dataclasses.replace(cfg, **overrides)  # type: ignore[arg-type]


def _toy_update(cfg: LensembleConfig, *, round_index: int, seed: int) -> PseudoGradient:
    """A tiny θ⊕φ-sized PseudoGradient built from per-group toy deltas (canonical encoder.*/predictor.*).

    Reuses ``build_pseudogradient`` so the flat delta is in the SAME canonical order the coordinator
    flattens the global params in — element-wise aligned (the reduction precondition).
    """
    torch.manual_seed(seed)
    enc = build_encoder(cfg)
    pred = build_predictor(cfg)
    gen = torch.Generator().manual_seed(seed)
    param_deltas: dict[str, Tensor] = {}
    for name, t in enc.state_dict().items():
        param_deltas[f"encoder.{name}"] = 1e-2 * torch.randn(
            t.shape, generator=gen, dtype=torch.float32
        )
    for name, t in pred.state_dict().items():
        param_deltas[f"predictor.{name}"] = 1e-2 * torch.randn(
            t.shape, generator=gen, dtype=torch.float32
        )
    return build_pseudogradient(
        param_deltas, dataset_root=_ROOT, round_index=round_index, clipped=True
    )


def _stage(
    transport: InProcessTransport,
    cfg: LensembleConfig,
    *,
    round_index: int,
    participant_ids: list[str],
) -> dict[str, PseudoGradient]:
    """Stage one PseudoGradient per participant for ``round_index``; return the staged updates."""
    staged: dict[str, PseudoGradient] = {}
    for i, pid in enumerate(participant_ids):
        pg = _toy_update(cfg, round_index=round_index, seed=100 + i)
        transport.submit_update(participant_id=pid, round_index=round_index, update=pg)
        staged[pid] = pg
    return staged


# =====================================================================================================
# Fixtures B — the self-describing checkpoint + pinned probe (disciplines 2 + 5). Mirrors
# tests/ml/test_recompute_alignment.py: a 32-token config so the honest LayerNorm-terminated encoder
# recovers Q* = I (the real Encoder ends in a LayerNorm; <32 tokens give a rank-(d-1) embedding and a
# degenerate Procrustes even in the honest f_θ == f_ref case — verified upstream).
# =====================================================================================================

_ED = 8
_ET, _EC, _EH, _EW = 4, 3, 8, 8
_EPATCH, _ETUBELET = 2, 2
_ENUM_TOKENS = (_ET // _ETUBELET) * (_EH // _EPATCH) ** 2  # = 2 * 16 = 32


@dataclass(frozen=True)
class _EvalModelCfg:
    encoder: str = "vjepa2-vit-l"
    warm_start_release: str = "vjepa2-2.0"
    latent_dim: int = _ED
    num_tokens: int = _ENUM_TOKENS
    predictor_depth: int = 1
    predictor_width: int = _ED
    wmcp_version: str = WMCP_VERSION
    encoder_frozen: bool = False
    d: int = _ED
    in_channels: int = _EC
    num_frames: int = _ET
    image_size: int = _EH
    patch_size: int = _EPATCH
    tubelet: int = _ETUBELET
    depth: int = 1
    num_heads: int = 2
    cond_dim: int = _ED


def _eval_cfg() -> SimpleNamespace:
    return SimpleNamespace(model=_EvalModelCfg())  # type: ignore[arg-type]


def _eval_points(p: int = 16) -> Tensor:
    gen = torch.Generator().manual_seed(0)
    return torch.randn(p, _ET, _EC, _EH, _EW, generator=gen)


def _commit_self_describing_encoder(
    ckpt_dir: Path, cfg: SimpleNamespace, *, round_index: int, parent_hash: str | None
) -> tuple[str, object]:
    """Build a fresh encoder (+ predictor), snapshot f_ref, and commit a self-describing checkpoint.

    Returns ``(content_hash, f_ref)``. The committed encoder IS ``f_ref`` (the honest case), and the
    checkpoint carries a ``ModelArchDescriptor`` (#171) so ``Encoder.from_header`` can reconstruct ``f_θ``.
    """
    torch.manual_seed(0)
    encoder = build_encoder(cfg)
    predictor = build_predictor(cfg)
    f_ref = snapshot_reference(encoder)  # the committed encoder IS f_ref (honest case)
    weights: dict[str, Tensor] = {}
    for name, tensor in encoder.state_dict().items():
        weights[f"encoder.{name}"] = tensor
    for name, tensor in predictor.state_dict().items():
        weights[f"predictor.{name}"] = tensor
    content_hash = save_checkpoint(
        ckpt_dir,
        weights,
        wmcp_version=WMCP_VERSION,
        round_index=round_index,
        config_hash="b" * 64,
        parent_hash=parent_hash,
        model_arch=model_arch_from_config(cfg),
    )
    return content_hash, f_ref


def _honest_setup(tmp_path: Path) -> tuple[Path, Path]:
    """Commit a self-describing checkpoint + save the matching pinned probe; return (ckpt_dir, probe_path).

    The probe's ``landmark_targets = f_ref(landmarks)`` are the pinned reference targets ``E_ref``.
    """
    cfg = _eval_cfg()
    ckpt_dir = tmp_path / "ckpt"
    _, f_ref = _commit_self_describing_encoder(
        ckpt_dir, cfg, round_index=7, parent_hash=None
    )
    probe = build_probe(_eval_points(), torch.arange(_ED), f_ref)  # type: ignore[arg-type]
    probe_path = tmp_path / "probe.safetensors"
    save_probe(probe, probe_path)
    return ckpt_dir, probe_path


# =====================================================================================================
# Fixtures C — a deterministic in-memory dataset for the commitment-binding discipline (discipline 3).
# Duplicated (not cross-imported) from tests/ml/test_provenance_commit.py: the tests/ tree has no package
# __init__, so an `import tests.ml.test_provenance_commit` is not portable under pytest's import modes —
# the other ml tests duplicate their small helpers for the same reason.
# =====================================================================================================


def _commit_spec(
    embodiment_id: str = "so101-arm-7dof", wmcp: str = WMCP_VERSION
) -> ActionSpec:
    return ActionSpec(
        embodiment_id=embodiment_id,
        kind=ActionKind.CONTINUOUS,
        dim=3,
        low=(-1.0, -1.0, -1.0),
        high=(1.0, 1.0, 1.0),
        num_classes=None,
        units=("rad", "rad", "rad"),
        wmcp_version=wmcp,
    )


def _commit_episode(seed: int) -> Episode:
    gen = torch.Generator().manual_seed(seed)
    transitions = [
        Transition(
            obs_t=torch.randint(0, 256, (3, 2, 4, 4), dtype=torch.uint8, generator=gen),
            action_t=torch.randn(3, generator=gen),
            obs_tp1=torch.randint(
                0, 256, (3, 2, 4, 4), dtype=torch.uint8, generator=gen
            ),
        )
        for _ in range(2)
    ]
    return Episode(
        episode_id=f"ep-{seed}",
        transitions=transitions,
        embodiment_id="so101-arm-7dof",
        modality="rgb-video",
        action_spec=_commit_spec(),
        collection_meta={"site": "lab-a"},
    )


def _build_dataset(seeds: tuple[int, ...]) -> EpisodeDataset:
    """A deterministic in-memory dataset; its Merkle root is reproducible byte-for-byte from the seeds."""
    return EpisodeDataset([_commit_episode(s) for s in seeds])


# =====================================================================================================
# Discipline 1 — INV-AGG-DETERMINISM: the outer step is bitwise-reproducible.
# =====================================================================================================


def test_discipline_1_agg_determinism_positive_commits_reproducible_hash() -> None:
    """POSITIVE: the same contributing set committed twice commits a bitwise-identical (θ_{t+1}, φ_{t+1}).

    Backed by ``Coordinator`` (whose AGGREGATING state runs ``assert_outer_step_deterministic``): the
    committed ``global_state_hash`` and the flat params agree byte-for-byte across two independent runs
    with the same seed/inputs.
    """
    cfg = _coord_cfg()

    def _commit_once() -> tuple[str, Tensor]:
        transport = InProcessTransport()
        coord = Coordinator(cfg, transport=transport)
        _stage(transport, cfg, round_index=0, participant_ids=["c0", "c1"])
        coord.run(1)
        assert coord.round_state() == RoundState.CLOSED
        return coord.global_state_hash(), coord.global_params().clone()

    h_a, params_a = _commit_once()
    h_b, params_b = _commit_once()

    assert h_a == h_b  # bitwise-identical committed (θ_{t+1}, φ_{t+1}) content hash
    assert torch.equal(params_a, params_b)

    # And the underlying primitive is itself reproducible on a staged contributing set: the same pure
    # reduction thunk passes assert_outer_step_deterministic (returns the verified flat result).
    prior = Coordinator(cfg, transport=InProcessTransport()).global_params()
    updates = {
        "c0": _toy_update(cfg, round_index=0, seed=100),
        "c1": _toy_update(cfg, round_index=0, seed=101),
    }
    lr, momentum = cfg.federation.outer_lr, cfg.federation.outer_nesterov_momentum
    verified = assert_outer_step_deterministic(
        lambda: OuterOptimizer(lr=lr, momentum=momentum).step(prior, updates),
        round_index=0,
    )
    assert verified.numel() == prior.numel()


def test_discipline_1_agg_determinism_negative_nondeterministic_reduction_aborts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NEGATIVE: an injected nondeterministic reduction raises ``NonDeterministicAggregation`` and ABORTs.

    Forcing ``OuterOptimizer.average_deltas`` to return a DIFFERENT tensor each call makes the AGGREGATING
    determinism self-check's two recomputations disagree; the security-critical
    ``NonDeterministicAggregation`` is raised (never swallowed) and the round → ``ABORTED`` with the
    committed global hash UNCHANGED (no partial commit). Mirrors
    ``test_coordinator.py::test_corrupt_reduction_aborts_with_global_hash_unchanged``.
    """
    cfg = _coord_cfg()
    transport = InProcessTransport()
    coord = Coordinator(cfg, transport=transport)
    h_before = coord.global_state_hash()
    _stage(transport, cfg, round_index=0, participant_ids=["c0", "c1"])

    counter = {"n": 0}

    def _nondeterministic_average(self: OuterOptimizer, deltas: object) -> Tensor:
        counter["n"] += 1
        return torch.full((coord.global_params().numel(),), float(counter["n"]))

    monkeypatch.setattr(OuterOptimizer, "average_deltas", _nondeterministic_average)

    with pytest.raises(NonDeterministicAggregation) as exc:
        coord.run(1)
    assert exc.value.code == LensembleErrorCode.AGG_NONDETERMINISTIC
    assert coord.round_state() == RoundState.ABORTED
    assert coord.global_state_hash() == h_before  # no partial commit (hash unchanged)


# =====================================================================================================
# Discipline 2 — INV-CHECKPOINT-HASH: committed (θ, φ) content hash + a valid parent_hash chain.
# =====================================================================================================


def test_discipline_2_checkpoint_hash_positive_content_hash_and_parent_chain(
    tmp_path: Path,
) -> None:
    """POSITIVE: each committed artifact's recomputed content hash equals its header's, and round t+1's
    ``parent_hash`` equals round t's ``content_hash`` (a valid chain).

    Two artifacts are committed in a parent chain (round 7 → round 8); ``load_checkpoint`` / ``verify``
    recompute and match each header's hash, and the round-8 header's ``parent_hash`` IS the round-7
    ``content_hash``.
    """
    cfg = _eval_cfg()
    ckpt_t = tmp_path / "round-00007"
    h_t, _ = _commit_self_describing_encoder(
        ckpt_t, cfg, round_index=7, parent_hash=None
    )
    ckpt_t1 = tmp_path / "round-00008"
    h_t1, _ = _commit_self_describing_encoder(
        ckpt_t1, cfg, round_index=8, parent_hash=h_t
    )

    # The recomputed content hash equals the header's (load_checkpoint verifies it before returning).
    _, header_t = load_checkpoint(ckpt_t)
    _, header_t1 = load_checkpoint(ckpt_t1)
    assert header_t.content_hash == h_t
    assert header_t1.content_hash == h_t1
    assert verify(ckpt_t).content_hash == h_t
    assert verify(ckpt_t1).content_hash == h_t1

    # The chain link: round t+1's parent_hash == round t's content_hash (INV-CHECKPOINT-HASH).
    assert header_t.parent_hash is None  # the chain root
    assert header_t1.parent_hash == h_t


def test_discipline_2_checkpoint_hash_negative_tampered_weights_raises(
    tmp_path: Path,
) -> None:
    """NEGATIVE: a tampered ``weights.safetensors`` byte makes ``load_checkpoint`` / ``verify`` raise.

    Overwriting the weight payload with different (valid safetensors) bytes diverges the recomputed
    content hash from the committed header's, so the integrity check raises ``CheckpointIntegrityError``
    (fail-closed, before the tensors are used downstream).
    """
    cfg = _eval_cfg()
    ckpt_dir = tmp_path / "ckpt"
    _commit_self_describing_encoder(ckpt_dir, cfg, round_index=7, parent_hash=None)
    save_file(
        {"encoder.norm.weight": torch.ones(_ED)},
        str(ckpt_dir / "weights.safetensors"),
    )
    with pytest.raises(CheckpointIntegrityError):
        load_checkpoint(ckpt_dir)
    with pytest.raises(CheckpointIntegrityError):
        verify(ckpt_dir)


# =====================================================================================================
# Discipline 3 — INV-COMMIT-BINDING: each released Δ_c is bound to exactly one 32-byte R_c.
# =====================================================================================================


def test_discipline_3_commit_binding_positive_single_root_accepted() -> None:
    """POSITIVE: a released ``Δ_c`` carries exactly one 32-byte ``R_c`` and ``verify_binding`` accepts it.

    The participant commits its local dataset to ``R_c = commit_dataset(...).merkle_root``; the released
    ``PseudoGradient`` carries that single 32-byte root and the pure binding check accepts it (returns
    ``None``) against the committed root.
    """
    committed = commit_dataset(_build_dataset((1, 2, 3)))
    r_c = bytes.fromhex(committed.merkle_root)
    assert len(r_c) == DIGEST_SIZE == 32

    delta = build_pseudogradient(
        {"encoder.w": torch.zeros(4, dtype=torch.float32)},
        dataset_root=r_c,
        round_index=0,
        clipped=True,
    )
    assert delta.dataset_root == r_c and len(delta.dataset_root) == 32
    # The bound root is accepted against the participant's committed R_c.
    assert verify_binding(r_c, delta.dataset_root, _SCHEME) is None


def test_discipline_3_commit_binding_negative_wrong_root_raises() -> None:
    """NEGATIVE: a ``Δ_c`` declaring a wrong/foreign ``R_c`` raises ``CommitmentMismatch``.

    A delta whose ``dataset_root`` is a DIFFERENT participant's committed root (or any non-matching
    32-byte digest) is rejected by ``verify_binding`` with the security-critical ``CommitmentMismatch``
    (excluded from the sum, never swallowed).
    """
    committed = bytes.fromhex(commit_dataset(_build_dataset((1, 2))).merkle_root)
    foreign = bytes.fromhex(commit_dataset(_build_dataset((3, 4))).merkle_root)
    assert committed != foreign

    foreign_delta = build_pseudogradient(
        {"encoder.w": torch.zeros(4, dtype=torch.float32)},
        dataset_root=foreign,  # a foreign R_c, not the participant's committed one
        round_index=0,
        clipped=True,
    )
    with pytest.raises(CommitmentMismatch) as exc:
        verify_binding(committed, foreign_delta.dataset_root, _SCHEME)
    assert exc.value.code == LensembleErrorCode.COMMITMENT_MISMATCH
    assert (
        exc.value.remediation
    )  # security-critical: carries a remediation, never swallowed


# =====================================================================================================
# Discipline 4 — INV-PROBE-PIN: probe content hash == RoundOpen commitment; targets from f_ref only.
# =====================================================================================================


def test_discipline_4_probe_pin_positive_hash_equals_commitment_and_targets_from_f_ref(
    tmp_path: Path,
) -> None:
    """POSITIVE: a built probe's content hash equals the coordinator's broadcast ``GlobalState.probe_hash``
    and its ``landmark_targets`` derive only from ``f_ref`` (not a later encoder).

    The pinned probe is built from a fresh ``f_ref`` and saved; pointing ``cfg.data.probe_path`` at it
    makes the ``Coordinator`` broadcast its content hash as ``GlobalState.probe_hash`` (the ``RoundOpen``
    commitment). ``verify_probe_pin`` accepts the probe against that broadcast hash, and the probe's
    targets equal ``f_ref(landmarks)`` while a later-mutated encoder would give different targets.
    """
    # Build f_ref + the pinned probe (landmark_targets = f_ref(landmarks)).
    cfg_eval = _eval_cfg()
    torch.manual_seed(0)
    encoder = build_encoder(cfg_eval)
    f_ref = snapshot_reference(encoder)
    points = _eval_points()
    landmark_idx = torch.arange(_ED)
    probe = build_probe(points, landmark_idx, f_ref)  # type: ignore[arg-type]
    probe_path = tmp_path / "probe.safetensors"
    save_probe(probe, probe_path)

    # The coordinator broadcasts the pinned probe's content hash as GlobalState.probe_hash (RoundOpen).
    cfg = _coord_cfg()
    cfg = dataclasses.replace(
        cfg, data=dataclasses.replace(cfg.data, probe_path=str(probe_path))
    )
    coord = Coordinator(cfg, transport=InProcessTransport())
    broadcast_probe_hash = coord.global_state().probe_hash
    assert broadcast_probe_hash == probe.content_hash  # pin == RoundOpen commitment
    assert broadcast_probe_hash != b"\x00" * 32  # not the no-probe placeholder
    # verify_probe_pin accepts the probe against the broadcast commitment (no-op return).
    assert verify_probe_pin(probe, broadcast_probe_hash) is None

    # Targets derive ONLY from f_ref: they equal f_ref(landmarks), not a later-mutated encoder's.
    expected_targets = f_ref(points[landmark_idx]).tokens
    assert torch.equal(probe.landmark_targets, expected_targets)  # INV-PROBE-PIN
    with torch.no_grad():
        encoder.pos_embed.add_(1.0)  # a later "training step" mutates the live encoder
    later_targets = snapshot_reference(encoder)(points[landmark_idx]).tokens
    assert not torch.allclose(later_targets, probe.landmark_targets)  # later ≠ pinned


def test_discipline_4_probe_pin_negative_mismatched_hash_raises(
    tmp_path: Path,
) -> None:
    """NEGATIVE: a probe whose content hash differs from the broadcast commitment raises ``ProbeError``.

    A probe pinned to one content hash, checked against a DIFFERENT broadcast ``probe_hash`` (a
    re-anchoring event the federation has not agreed to), is refused by ``verify_probe_pin`` with the
    fail-closed ``ProbeError`` (``PROBE_INVALID``).
    """
    cfg_eval = _eval_cfg()
    torch.manual_seed(0)
    f_ref = snapshot_reference(build_encoder(cfg_eval))
    probe = build_probe(_eval_points(), torch.arange(_ED), f_ref)  # type: ignore[arg-type]
    save_probe(probe, tmp_path / "probe.safetensors")  # round-trips the pin to disk

    # A broadcast hash that does not match the probe's pin (a foreign / re-anchored probe).
    with pytest.raises(ProbeError) as exc:
        verify_probe_pin(probe, b"\x00" * 32)
    assert exc.value.code == LensembleErrorCode.PROBE_INVALID
    assert exc.value.expected_hash == probe.content_hash  # type: ignore[attr-defined]
    assert exc.value.remediation


# =====================================================================================================
# Discipline 5 — recompute_alignment: public recomputation reproduces the alignment from public inputs.
# =====================================================================================================


def test_discipline_5_public_recompute_positive_matches_honest_claim(
    tmp_path: Path,
) -> None:
    """POSITIVE: ``recompute_alignment_claim(committed_weights, probe, expected=<honest claim>)`` returns
    ``matches_expected=True`` from the committed checkpoint + the pinned probe alone.

    The honest claim is exactly what an honest recomputation produces; re-checking it against the same
    public inputs (committed self-describing checkpoint + pinned probe, 32-token config so the honest
    encoder recovers ``Q* = I``) matches the ``procrustes_q_hash`` EXACTLY and the residual within
    tolerance — public verifiability with no secret inputs (RFC-0006 §4).
    """
    ckpt_dir, probe_path = _honest_setup(tmp_path)

    honest = recompute_alignment_claim(ckpt_dir, probe_path).recomputed
    checked = recompute_alignment_claim(ckpt_dir, probe_path, expected=honest)

    assert checked.matches_expected is True
    assert checked.recomputed.round_index == 7  # from the committed header
    assert checked.probe_hash == honest.probe_hash
    assert checked.max_abs_residual_delta is not None
    assert checked.max_abs_residual_delta <= 1e-6


def test_discipline_5_public_recompute_negative_perturbed_claim_does_not_match(
    tmp_path: Path,
) -> None:
    """NEGATIVE: a perturbed expected claim (wrong ``procrustes_q_hash``) returns ``matches_expected=False``.

    The recomputed record is still the honest one (the claim is what is rejected, not the recomputation):
    a fabricated ``procrustes_q_hash`` cannot match the bitwise-deterministic recomputed ``Q*`` hash, so
    the public check fails closed.
    """
    ckpt_dir, probe_path = _honest_setup(tmp_path)
    honest = recompute_alignment_claim(ckpt_dir, probe_path).recomputed
    perturbed = honest.model_copy(update={"procrustes_q_hash": "f" * 64})

    checked = recompute_alignment_claim(ckpt_dir, probe_path, expected=perturbed)
    assert checked.matches_expected is False
    # the recomputed Q* hash is the honest one — only the fabricated claim is rejected
    assert checked.recomputed.procrustes_q_hash == honest.procrustes_q_hash


# =====================================================================================================
# The audit entry point — a single discoverable test enumerating the five disciplines it covers.
# =====================================================================================================


def test_proofready_audit_covers_all_five_disciplines() -> None:
    """The v1.0 proof-ready audit entry point: assert this module covers the five RFC-0006 §3 disciplines.

    Enumerates the five ``INV-*`` ids audited here (INV-AGG-DETERMINISM, INV-CHECKPOINT-HASH,
    INV-COMMIT-BINDING, INV-PROBE-PIN, and the recompute_alignment public-recomputation guarantee) and
    pins that each has BOTH a positive and a negative test function in this module — so the audit is a
    single, discoverable, regression-resistant surface (the issue's "audit test entry point").
    """
    assert len(_AUDITED_INVARIANTS) == 5
    assert _AUDITED_INVARIANTS[0] == "INV-AGG-DETERMINISM"
    assert "INV-CHECKPOINT-HASH" in _AUDITED_INVARIANTS
    assert "INV-COMMIT-BINDING" in _AUDITED_INVARIANTS
    assert "INV-PROBE-PIN" in _AUDITED_INVARIANTS
    assert "recompute_alignment" in _AUDITED_INVARIANTS[4]

    # Each discipline has a positive AND a negative test function defined in this module (1..5).
    module = globals()
    for n in range(1, 6):
        positives = [
            name
            for name in module
            if name.startswith(f"test_discipline_{n}_") and "_positive" in name
        ]
        negatives = [
            name
            for name in module
            if name.startswith(f"test_discipline_{n}_") and "_negative" in name
        ]
        assert positives, f"discipline {n} is missing a positive-path test"
        assert negatives, f"discipline {n} is missing a negative-path test"
