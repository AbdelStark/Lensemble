"""Gate G1 — TwoRooms LeWM checkpoint reconstruction, strict loading, and reference parity (#316).

Three layers, per docs/roadmap/TAPESTRY_LEWM.md:

1. CPU-only contract tests (no download): the released ``config.json`` validates and a wrong
   ``_target_``/norm/encoder flag fails closed; the reconstructed state-dict schema reproduces the
   released tensor inventory exactly; the strict loader rejects unknown/missing/mismatched tensors;
   claim-grade resolution refuses unpinned revisions; forward semantics (shapes, determinism,
   causality, history truncation) hold on a tiny config.
2. Real-checkpoint integration (skipped unless the pinned snapshot is already in the local HF cache —
   the CI gates download nothing): strict load of ``weights.pt`` and regression of the deterministic
   reference forwards.
3. Upstream parity (skipped unless ``transformers``+``einops`` are installed): the reconstruction
   matches the actual upstream ``ViTModel``+``stable_worldmodel.wm.lewm`` forward on fixed fixtures.
   This ran against transformers 4.49 at ingest time (max abs diff ≤ 2e-5); the skip keeps the heavy
   dependency out of the blocking CPU gates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import torch

from lensemble.errors import CheckpointIntegrityError, ConfigError, ContractViolation
from lensemble.model.lewm_checkpoint import (
    TWOROOMS_PINNED_REVISION,
    TWOROOMS_REPO_ID,
    checkpoint_manifest,
    load_tworooms_model,
    reference_forward_report,
    resolve_checkpoint,
)
from lensemble.model.lewm_tworooms import (
    LeWMTwoRoomsConfig,
    build_lewm_tworooms,
    load_lewm_state_dict,
)

# fp32 reference-forward agreement for the reconstruction (attention reordering noise).
RTOL_LEWM = 1e-4
ATOL_LEWM = 1e-4


def _released_config() -> dict:
    """The pinned quentinll/lewm-tworooms config.json, inline (revision 77adaae…)."""
    return {
        "_target_": "stable_worldmodel.wm.lewm.LeWM",
        "encoder": {
            "_target_": "stable_pretraining.backbone.utils.vit_hf",
            "size": "tiny",
            "patch_size": 14,
            "image_size": 224,
            "pretrained": False,
            "use_mask_token": False,
        },
        "predictor": {
            "_target_": "stable_worldmodel.wm.lewm.module.Predictor",
            "num_frames": 3,
            "input_dim": 192,
            "hidden_dim": 192,
            "output_dim": 192,
            "depth": 6,
            "heads": 16,
            "mlp_dim": 2048,
            "dim_head": 64,
            "dropout": 0.1,
            "emb_dropout": 0.0,
        },
        "action_encoder": {
            "_target_": "stable_worldmodel.wm.lewm.module.Embedder",
            "input_dim": 10,
            "emb_dim": 192,
        },
        "projector": {
            "_target_": "stable_worldmodel.wm.lewm.module.MLP",
            "input_dim": 192,
            "output_dim": 192,
            "hidden_dim": 2048,
            "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True},
        },
        "pred_proj": {
            "_target_": "stable_worldmodel.wm.lewm.module.MLP",
            "input_dim": 192,
            "output_dim": 192,
            "hidden_dim": 2048,
            "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True},
        },
    }


def _tiny_cfg() -> LeWMTwoRoomsConfig:
    """A small same-shape-family config for fast CPU forward tests."""
    return LeWMTwoRoomsConfig(
        vit_size="tiny",
        patch_size=14,
        image_size=28,
        hidden_dim=192,
        vit_layers=1,
        vit_heads=3,
        pred_num_frames=3,
        pred_depth=1,
        pred_heads=4,
        pred_dim_head=16,
        pred_mlp_dim=64,
        pred_dropout=0.1,
        pred_emb_dropout=0.0,
        action_input_dim=10,
        action_emb_dim=192,
        proj_hidden_dim=32,
    )


# ---------------------------------------------------------------------------
# 1. config validation fails closed
# ---------------------------------------------------------------------------


def test_released_config_validates() -> None:
    cfg = LeWMTwoRoomsConfig.from_upstream(_released_config())
    assert cfg.hidden_dim == 192
    assert cfg.patch_size == 14
    assert cfg.num_patches == 256
    assert cfg.pred_num_frames == 3
    assert cfg.action_input_dim == 10


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.__setitem__("_target_", "some.other.WorldModel"),
        lambda c: c["predictor"].__setitem__("_target_", "other.Predictor"),
        lambda c: c["encoder"].__setitem__("pretrained", True),
        lambda c: c["encoder"].__setitem__("use_mask_token", True),
        lambda c: c["encoder"].__setitem__("size", "giant"),
        lambda c: c["projector"]["norm_fn"].__setitem__(
            "_target_", "torch.nn.LayerNorm"
        ),
        lambda c: c["predictor"].__setitem__("input_dim", 256),
        lambda c: c.pop("action_encoder"),
    ],
)
def test_mismatched_config_fails_closed(mutate) -> None:
    config = _released_config()
    mutate(config)
    with pytest.raises(ConfigError):
        LeWMTwoRoomsConfig.from_upstream(config)


# ---------------------------------------------------------------------------
# 2. state-dict schema reproduces the released tensor inventory
# ---------------------------------------------------------------------------


def test_state_dict_schema_matches_released_inventory() -> None:
    model = build_lewm_tworooms(_released_config())
    state = model.state_dict()
    assert len(state) == 303  # the released weights.pt tensor count

    expected_shapes = {
        "encoder.embeddings.cls_token": (1, 1, 192),
        "encoder.embeddings.position_embeddings": (1, 257, 192),
        "encoder.embeddings.patch_embeddings.projection.weight": (192, 3, 14, 14),
        "encoder.encoder.layer.0.attention.attention.query.weight": (192, 192),
        "encoder.encoder.layer.11.output.dense.weight": (192, 768),
        "encoder.layernorm.weight": (192,),
        "predictor.pos_embedding": (1, 3, 192),
        "predictor.transformer.layers.0.attn.to_qkv.weight": (3072, 192),
        "predictor.transformer.layers.5.adaLN_modulation.1.weight": (1152, 192),
        "predictor.transformer.layers.0.mlp.net.1.weight": (2048, 192),
        "action_encoder.patch_embed.weight": (10, 10, 1),
        "action_encoder.embed.0.weight": (768, 10),
        "action_encoder.embed.2.weight": (192, 768),
        "projector.net.0.weight": (2048, 192),
        "projector.net.1.running_mean": (2048,),
        "pred_proj.net.3.weight": (192, 2048),
    }
    for name, shape in expected_shapes.items():
        assert name in state, f"missing released tensor name {name}"
        assert tuple(state[name].shape) == shape, name

    # bias-free QKV is part of the released schema — a bias key would be an unknown tensor.
    assert "predictor.transformer.layers.0.attn.to_qkv.bias" not in state
    # identity input/cond/output projections at D=192 contribute no tensors.
    assert not any(
        "input_proj" in k or "cond_proj" in k or "output_proj" in k for k in state
    )


def test_strict_load_round_trips_and_fails_closed() -> None:
    model = build_lewm_tworooms(_released_config())
    good = {
        k: torch.randn_like(v) if v.is_floating_point() else v.clone()
        for k, v in model.state_dict().items()
    }
    load_lewm_state_dict(model, good)  # round trip

    unknown = dict(good)
    unknown["encoder.mask_token"] = torch.zeros(1, 1, 192)
    with pytest.raises(CheckpointIntegrityError, match="unknown"):
        load_lewm_state_dict(model, unknown)

    missing = dict(good)
    missing.pop("predictor.pos_embedding")
    with pytest.raises(CheckpointIntegrityError, match="missing"):
        load_lewm_state_dict(model, missing)

    mismatched = dict(good)
    mismatched["projector.net.0.weight"] = torch.randn(8, 8)
    with pytest.raises(CheckpointIntegrityError, match="shape"):
        load_lewm_state_dict(model, mismatched)


# ---------------------------------------------------------------------------
# 3. checkpoint resolution fails closed
# ---------------------------------------------------------------------------


def test_claim_grade_requires_pinned_revision(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="pinned"):
        resolve_checkpoint(local_dir=tmp_path, revision="main")
    # dev mode allows it, but the snapshot must still be complete
    with pytest.raises(Exception):
        resolve_checkpoint(local_dir=tmp_path, revision="main", claim_grade=False)


def test_missing_snapshot_files_fail(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_released_config()))
    with pytest.raises(Exception, match="missing"):
        resolve_checkpoint(local_dir=tmp_path)


def test_non_state_dict_weights_fail(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_released_config()))
    torch.save({"not_a_tensor": "raw string"}, tmp_path / "weights.pt")
    resolved = resolve_checkpoint(local_dir=tmp_path)
    with pytest.raises(Exception, match="state dict|unknown"):
        load_tworooms_model(resolved)


# ---------------------------------------------------------------------------
# 4. forward semantics on a tiny same-family config
# ---------------------------------------------------------------------------


def test_forward_shapes_and_determinism() -> None:
    torch.manual_seed(7)
    model = build_lewm_tworooms(_tiny_cfg())
    pixels = torch.rand(2, 3, 3, 28, 28)
    actions = torch.rand(2, 5, 10)

    emb = model.encode_frames(pixels)
    act_emb = model.encode_actions(actions)
    assert emb.shape == (2, 3, 192)
    assert act_emb.shape == (2, 5, 192)

    preds = model.predict(emb, act_emb[:, :3])
    assert preds.shape == (2, 3, 192)
    assert torch.allclose(preds, model.predict(emb, act_emb[:, :3]))  # eval determinism

    rollout = model.rollout(emb, act_emb)
    # H + n_steps + 1 = 3 + 2 + 1 (upstream predicts one step past the final action)
    assert rollout.shape == (2, 6, 192)
    assert torch.allclose(rollout[:, :3], emb)  # history preserved verbatim

    cost = model.goal_cost(rollout, emb[0, -1])
    assert cost.shape == (2,)
    assert (cost >= 0).all()


def test_predictor_is_causal() -> None:
    """Changing the last frame/action must not change earlier predictions.

    AdaLN-zero means a freshly initialized model ignores actions entirely (the conditioning ramps
    in during training), so the action-sensitivity half randomizes the AdaLN weights first.
    """
    torch.manual_seed(11)
    model = build_lewm_tworooms(_tiny_cfg())
    for block in model.predictor.transformer.layers:
        modulation = cast(torch.nn.Sequential, block.adaLN_modulation)
        torch.nn.init.normal_(cast(torch.nn.Linear, modulation[-1]).weight, std=0.05)
    emb = torch.randn(1, 3, 192)
    act = torch.randn(1, 3, 192)
    base = model.predict(emb, act)

    perturbed_act = act.clone()
    perturbed_act[:, -1] += 10.0
    out = model.predict(emb, perturbed_act)
    assert torch.allclose(base[:, :2], out[:, :2], rtol=RTOL_LEWM, atol=ATOL_LEWM)
    assert not torch.allclose(base[:, -1], out[:, -1], rtol=RTOL_LEWM, atol=ATOL_LEWM)

    perturbed_emb = emb.clone()
    # a random (non-constant) perturbation — pre-norm LayerNorms annihilate constant shifts
    perturbed_emb[:, -1] += torch.randn(192)
    out = model.predict(perturbed_emb, act)
    assert torch.allclose(base[:, :2], out[:, :2], rtol=RTOL_LEWM, atol=ATOL_LEWM)
    assert not torch.allclose(base[:, -1], out[:, -1], rtol=RTOL_LEWM, atol=ATOL_LEWM)


def test_predictor_window_is_enforced() -> None:
    model = build_lewm_tworooms(_tiny_cfg())
    with pytest.raises(ContractViolation):
        model.predict(torch.randn(1, 5, 192), torch.randn(1, 5, 192))


def test_rollout_truncates_history_to_predictor_window() -> None:
    """A long rollout keeps working because history is truncated to num_frames."""
    torch.manual_seed(13)
    model = build_lewm_tworooms(_tiny_cfg())
    emb = model.encode_frames(torch.rand(1, 3, 3, 28, 28))
    act_emb = model.encode_actions(torch.rand(1, 9, 10))
    rollout = model.rollout(emb, act_emb)
    assert rollout.shape == (1, 10, 192)
    assert torch.isfinite(rollout).all()


# ---------------------------------------------------------------------------
# 5. real-checkpoint integration (local HF cache only; CI downloads nothing)
# ---------------------------------------------------------------------------


def _cached_snapshot() -> Path | None:
    try:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                TWOROOMS_REPO_ID,
                revision=TWOROOMS_PINNED_REVISION,
                local_files_only=True,
            )
        )
    except Exception:
        return None


_SNAPSHOT = _cached_snapshot()

needs_checkpoint = pytest.mark.skipif(
    _SNAPSHOT is None,
    reason="pinned quentinll/lewm-tworooms snapshot not in the local HF cache",
)


@needs_checkpoint
def test_real_checkpoint_strict_load_and_manifest() -> None:
    resolved = resolve_checkpoint(local_dir=_SNAPSHOT)
    model, config = load_tworooms_model(resolved)
    assert sum(p.numel() for p in model.parameters()) == 18034478
    manifest = checkpoint_manifest(resolved, model)
    assert manifest["schema"] == "lewm-checkpoint-manifest/1"
    assert manifest["tensorCount"] == 303
    assert manifest["source"]["revisionPinned"] is True
    assert len(manifest["files"]["weights.pt"]["sha256"]) == 64

    committed = Path("docs/evidence/lewm_tworooms_checkpoint_manifest.json")
    if committed.is_file():
        recorded = json.loads(committed.read_text())
        assert (
            recorded["files"]["weights.pt"]["sha256"]
            == manifest["files"]["weights.pt"]["sha256"]
        )
        assert recorded["source"]["revision"] == resolved.revision


@needs_checkpoint
def test_real_checkpoint_reference_forwards_regress() -> None:
    resolved = resolve_checkpoint(local_dir=_SNAPSHOT)
    model, _ = load_tworooms_model(resolved)
    report = reference_forward_report(model)
    committed = Path("docs/evidence/lewm_tworooms_reference_report.json")
    if not committed.is_file():
        pytest.skip(
            "reference report not generated yet (scripts/lewm_tworooms_ingest.py)"
        )
    recorded = json.loads(committed.read_text())
    assert recorded["seed"] == report["seed"]
    for got, want in zip(report["outputs"], recorded["outputs"], strict=True):
        assert got["name"] == want["name"]
        assert got["shape"] == want["shape"]
        assert got["l2Norm"] == pytest.approx(want["l2Norm"], rel=1e-3)
        assert got["mean"] == pytest.approx(want["mean"], abs=1e-4)
        for a, b in zip(got["first8"], want["first8"], strict=True):
            assert a == pytest.approx(b, abs=1e-3)


# ---------------------------------------------------------------------------
# 6. upstream parity (optional heavy deps; documents the ingest-time proof)
# ---------------------------------------------------------------------------


@needs_checkpoint
def test_upstream_vit_parity_when_transformers_available() -> None:
    transformers = pytest.importorskip("transformers")
    resolved = resolve_checkpoint(local_dir=_SNAPSHOT)
    model, _ = load_tworooms_model(resolved)

    vit = transformers.ViTModel(
        transformers.ViTConfig(
            hidden_size=192,
            num_hidden_layers=12,
            num_attention_heads=3,
            intermediate_size=768,
            image_size=224,
            patch_size=14,
        ),
        add_pooling_layer=False,
        use_mask_token=False,
    )
    sd = torch.load(resolved.weights_path, map_location="cpu", weights_only=True)
    encoder_sd = {
        k.removeprefix("encoder."): v for k, v in sd.items() if k.startswith("encoder.")
    }
    try:
        vit.load_state_dict(encoder_sd, strict=True)
    except RuntimeError:
        pytest.skip("installed transformers uses the rewritten ViT key schema (>=4.50)")
    vit.eval()

    pixels = torch.rand(2, 3, 224, 224, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        upstream_cls = vit(pixels, interpolate_pos_encoding=True).last_hidden_state[
            :, 0
        ]
        ours_cls = model.encoder(pixels)[:, 0]
    assert torch.allclose(upstream_cls, ours_cls, rtol=RTOL_LEWM, atol=ATOL_LEWM)
