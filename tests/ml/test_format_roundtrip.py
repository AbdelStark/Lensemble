"""Format round-trip across the lance / hdf5 / lerobot data adapters (RFC-0004 §1). Issue #22.

The contract under test (RFC-0004 §Testing "Format round-trip"): ``lance`` is only the *default*,
never the canonical encoding — a run is reproducible regardless of which backend produced its windows
([RFC-0009](docs/rfcs/RFC-0009-configuration-reproducibility.md)). So an ``EpisodeDataset`` written by
each backend and read back must yield byte/tensor-identical ``Window``s, and the round-tripped
``Episode`` metadata (id / embodiment / modality / ``ActionSpec`` / collection conditions) must be
preserved exactly.

NOTE: placed in tests/ml (NOT tests/data): the §8 CI gate scans tests/{unit,property,integration,
ml,e2e,regression}; a tests/data directory would not run. The adapters materialize raw, model-bearing
tensors round-tripped through the loader, so tests/ml is their home (mirrors tests/ml/test_probe.py).
The ``lance``/``h5py`` backends require their (pinned runtime) libraries; the ``lerobot://`` adapter's
network path is the only optional extra and is exercised only via its factored conformance check.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lensemble.contracts import WMCP_VERSION, ActionKind, ActionSpec
from lensemble.data import (
    Episode,
    EpisodeDataset,
    Transition,
    load_episodes,
    register_adapter,
    save_episodes,
)
from lensemble.data.adapters._serialize import (
    dtype_label,
    tensor_from_bytes,
    tensor_to_bytes,
)
from lensemble.data.adapters.lerobot_adapter import (
    _validate_episode_conformance,
    load_lerobot,
)
from lensemble.errors import ContractViolation, LensembleErrorCode

# --- a tiny toy dataset: 2 episodes, (C,T,H,W) obs + (action_dim,) actions, a continuous ActionSpec ---

_C, _T, _H, _W = 2, 2, 3, 3
_ACTION_DIM = 4
_NUM_STEPS = 2


def _spec() -> ActionSpec:
    return ActionSpec(
        embodiment_id="toy-arm-4dof",
        kind=ActionKind.CONTINUOUS,
        dim=_ACTION_DIM,
        low=(-1.0,) * _ACTION_DIM,
        high=(1.0,) * _ACTION_DIM,
        num_classes=None,
        units=("rad",) * _ACTION_DIM,
        wmcp_version=WMCP_VERSION,
    )


def _episode(episode_id: str, n_transitions: int, seed: int) -> Episode:
    gen = torch.Generator().manual_seed(seed)
    transitions = []
    for _ in range(n_transitions):
        obs_t = torch.randn(_C, _T, _H, _W, generator=gen)
        obs_tp1 = torch.randn(_C, _T, _H, _W, generator=gen)
        action_t = torch.randn(_ACTION_DIM, generator=gen)
        transitions.append(Transition(obs_t=obs_t, action_t=action_t, obs_tp1=obs_tp1))
    return Episode(
        episode_id=episode_id,
        transitions=transitions,
        embodiment_id="toy-arm-4dof",
        modality="rgb-video",
        action_spec=_spec(),
        collection_meta={"site": "lab-a", "fps": "30"},
    )


def _toy_dataset() -> EpisodeDataset:
    return EpisodeDataset(
        [_episode("ep-0", 4, seed=1), _episode("ep-1", 3, seed=2)],
        fmt="lance",
    )


def _materialize(ds: EpisodeDataset) -> list[tuple]:
    """A stable, comparable list of (obs, actions, num_steps, embodiment_id) per window."""
    return [
        (w.obs, w.actions, w.num_steps, w.embodiment_id) for w in ds.windows(_NUM_STEPS)
    ]


def _assert_windows_equal(a: list[tuple], b: list[tuple]) -> None:
    assert len(a) == len(b)
    for (obs_a, act_a, n_a, emb_a), (obs_b, act_b, n_b, emb_b) in zip(
        a, b, strict=True
    ):
        assert torch.equal(obs_a, obs_b)
        assert torch.equal(act_a, act_b)
        assert n_a == n_b
        assert emb_a == emb_b


# --- the RFC-0004 "Format round-trip" test: lance == hdf5 == in-memory windows ---


def test_lance_hdf5_window_roundtrip_matches_in_memory(tmp_path: Path) -> None:
    original = _toy_dataset()
    in_mem = _materialize(original)

    lance_path = tmp_path / "ds.lance"
    hdf5_path = tmp_path / "ds.h5"
    save_episodes(original, lance_path, fmt="lance")
    save_episodes(original, hdf5_path, fmt="hdf5")

    ds_lance = load_episodes(lance_path, fmt="lance")
    ds_hdf5 = load_episodes(hdf5_path, fmt="hdf5")

    assert ds_lance.fmt == "lance"
    assert ds_hdf5.fmt == "hdf5"

    win_lance = _materialize(ds_lance)
    win_hdf5 = _materialize(ds_hdf5)

    # windows equal across the two formats AND against the original in-memory dataset
    _assert_windows_equal(win_lance, win_hdf5)
    _assert_windows_equal(win_lance, in_mem)
    _assert_windows_equal(win_hdf5, in_mem)


@pytest.mark.parametrize("fmt", ["lance", "hdf5"])
def test_episode_metadata_preserved(tmp_path: Path, fmt: str) -> None:
    original = _toy_dataset()
    path = tmp_path / f"ds.{fmt if fmt == 'lance' else 'h5'}"
    save_episodes(original, path, fmt=fmt)  # type: ignore[arg-type]
    read = load_episodes(path, fmt=fmt)  # type: ignore[arg-type]

    assert len(read) == len(original)
    for got, want in zip(read.episodes, original.episodes, strict=True):
        assert got.episode_id == want.episode_id
        assert got.embodiment_id == want.embodiment_id
        assert got.modality == want.modality
        assert got.collection_meta == want.collection_meta
        assert got.action_spec == want.action_spec
        assert len(got.transitions) == len(want.transitions)
        for t_got, t_want in zip(got.transitions, want.transitions, strict=True):
            assert torch.equal(t_got.obs_t, t_want.obs_t)
            assert torch.equal(t_got.action_t, t_want.action_t)
            assert torch.equal(t_got.obs_tp1, t_want.obs_tp1)


@pytest.mark.parametrize("fmt", ["lance", "hdf5"])
def test_roundtrip_preserves_tensor_dtype(tmp_path: Path, fmt: str) -> None:
    """A non-float32 obs dtype is recovered exactly (the recorded dtype+shape drives the reshape)."""
    spec = _spec()
    obs = torch.arange(_C * _T * _H * _W, dtype=torch.float64).reshape(_C, _T, _H, _W)
    transitions = [
        Transition(
            obs_t=obs.clone(),
            action_t=torch.ones(_ACTION_DIM, dtype=torch.float64),
            obs_tp1=obs.clone() + 1,
        )
        for _ in range(3)
    ]
    ds = EpisodeDataset(
        [
            Episode(
                episode_id="dt",
                transitions=transitions,
                embodiment_id="toy-arm-4dof",
                modality="rgb-video",
                action_spec=spec,
                collection_meta={},
            )
        ]
    )
    path = tmp_path / f"dt.{fmt if fmt == 'lance' else 'h5'}"
    save_episodes(ds, path, fmt=fmt)  # type: ignore[arg-type]
    read = load_episodes(path, fmt=fmt)  # type: ignore[arg-type]
    t = read.episodes[0].transitions[0]
    assert t.obs_t.dtype == torch.float64
    assert t.action_t.dtype == torch.float64
    assert torch.equal(t.obs_t, obs)


# --- lerobot adapter: on-load conformance is unit-testable WITHOUT lerobot installed ---


def test_lerobot_conformance_rejects_action_dim_mismatch() -> None:
    """An action tensor whose trailing dim != ActionSpec.dim raises ContractViolation."""
    spec = _spec()  # dim == 4
    bad = Episode(
        episode_id="bad",
        transitions=[
            Transition(
                obs_t=torch.zeros(_C, _T, _H, _W),
                action_t=torch.zeros(_ACTION_DIM + 1),  # trailing dim 5 != spec.dim 4
                obs_tp1=torch.zeros(_C, _T, _H, _W),
            )
        ],
        embodiment_id="toy-arm-4dof",
        modality="rgb-video",
        action_spec=spec,
        collection_meta={},
    )
    with pytest.raises(ContractViolation) as exc:
        _validate_episode_conformance(bad, spec)
    assert exc.value.code == LensembleErrorCode.WMCP_CONTRACT_VIOLATION


def test_lerobot_conformance_rejects_kind_mismatch() -> None:
    """A discrete ActionSpec whose num_classes is malformed (invalid spec) is rejected."""
    bad_spec = ActionSpec(
        embodiment_id="toy-arm-4dof",
        kind=ActionKind.DISCRETE,
        dim=_ACTION_DIM,
        low=None,
        high=None,
        num_classes=(1, 1, 1, 1),  # invalid: a discrete dim needs >= 2 classes
        units=("idx",) * _ACTION_DIM,
        wmcp_version=WMCP_VERSION,
    )
    ep = _episode("ep", 2, seed=3)
    with pytest.raises(ContractViolation):
        _validate_episode_conformance(ep, bad_spec)


def test_lerobot_conformance_rejects_embodiment_mismatch() -> None:
    """An episode whose embodiment_id disagrees with the ActionSpec is latent-incompatible."""
    spec = _spec()
    ep = Episode(
        episode_id="ep",
        transitions=[
            Transition(
                obs_t=torch.zeros(_C, _T, _H, _W),
                action_t=torch.zeros(_ACTION_DIM),
                obs_tp1=torch.zeros(_C, _T, _H, _W),
            )
        ],
        embodiment_id="some-other-arm",  # != spec.embodiment_id
        modality="rgb-video",
        action_spec=spec,
        collection_meta={},
    )
    with pytest.raises(ContractViolation):
        _validate_episode_conformance(ep, spec)


def test_lerobot_conformance_rejects_latent_incompatible_modality() -> None:
    """An episode whose modality is not latent-compatible raises ContractViolation."""
    spec = _spec()
    ep = Episode(
        episode_id="ep",
        transitions=[
            Transition(
                obs_t=torch.zeros(_C, _T, _H, _W),
                action_t=torch.zeros(_ACTION_DIM),
                obs_tp1=torch.zeros(_C, _T, _H, _W),
            )
        ],
        embodiment_id="toy-arm-4dof",  # matches the spec, so the modality clause is reached
        modality="point-cloud",  # not in the latent-compatible set
        action_spec=spec,
        collection_meta={},
    )
    with pytest.raises(ContractViolation) as exc:
        _validate_episode_conformance(ep, spec)
    assert "latent-incompatible" in str(exc.value)


def test_lerobot_conformance_rejects_discrete_index_out_of_range() -> None:
    """A discrete action index outside [0, num_classes[j]) raises ContractViolation."""
    spec = ActionSpec(
        embodiment_id="toy-arm-4dof",
        kind=ActionKind.DISCRETE,
        dim=2,
        low=None,
        high=None,
        num_classes=(3, 3),
        units=("idx", "idx"),
        wmcp_version=WMCP_VERSION,
    )
    ep = Episode(
        episode_id="ep",
        transitions=[
            Transition(
                obs_t=torch.zeros(_C, _T, _H, _W),
                action_t=torch.tensor([0.0, 5.0]),  # index 5 >= num_classes 3
                obs_tp1=torch.zeros(_C, _T, _H, _W),
            )
        ],
        embodiment_id="toy-arm-4dof",
        modality="rgb-video",
        action_spec=spec,
        collection_meta={},
    )
    with pytest.raises(ContractViolation) as exc:
        _validate_episode_conformance(ep, spec)
    assert "out of range" in str(exc.value)


def test_lerobot_conformance_accepts_valid_discrete_episode() -> None:
    """A discrete spec with in-range indices validates with no exception (covers the in-range path)."""
    spec = ActionSpec(
        embodiment_id="toy-arm-4dof",
        kind=ActionKind.DISCRETE,
        dim=2,
        low=None,
        high=None,
        num_classes=(3, 3),
        units=("idx", "idx"),
        wmcp_version=WMCP_VERSION,
    )
    ep = Episode(
        episode_id="ep",
        transitions=[
            Transition(
                obs_t=torch.zeros(_C, _T, _H, _W),
                action_t=torch.tensor([0.0, 2.0]),  # both in [0, 3)
                obs_tp1=torch.zeros(_C, _T, _H, _W),
            )
        ],
        embodiment_id="toy-arm-4dof",
        modality="rgb-video",
        action_spec=spec,
        collection_meta={},
    )
    _validate_episode_conformance(ep, spec)  # must not raise


def test_lerobot_conformance_accepts_conforming_episode() -> None:
    """A well-formed episode validates with no exception (no-op return)."""
    ep = _episode("ep", 3, seed=7)
    _validate_episode_conformance(ep, ep.action_spec)  # must not raise


def test_load_lerobot_without_scheme_prefix_raises_when_absent() -> None:
    """``load_lerobot`` accepts a bare repo id (no scheme) and still gates on the absent library.

    Covers the no-prefix branch of the scheme stripping. Skipped if ``lerobot`` is importable.
    """
    try:
        import lerobot  # type: ignore  # noqa: F401
    except ImportError:
        with pytest.raises(ContractViolation) as exc:
            load_lerobot("bare-repo-id")
        assert "lerobot" in exc.value.remediation
    else:  # pragma: no cover - the suite runs without the optional lerobot extra
        pytest.skip("lerobot is installed; the absent-lib path is unreachable")


def test_lerobot_uri_raises_when_lib_absent() -> None:
    """``lerobot://<repo>`` raises a clear error when the optional ``lerobot`` extra is absent.

    If ``lerobot`` happens to be importable, skip — the absent-lib branch is unreachable.
    """
    try:
        import lerobot  # type: ignore  # noqa: F401
    except ImportError:
        with pytest.raises((ContractViolation, RuntimeError)) as exc:
            load_episodes("lerobot://toy")
        err = exc.value
        # the built-in adapter raises ContractViolation, which carries a .remediation; a custom
        # read-only adapter MAY raise a bare RuntimeError, so fall back to the message text.
        remediation = getattr(err, "remediation", str(err))
        assert "lerobot" in remediation
    else:  # pragma: no cover - the suite runs without the optional lerobot extra
        pytest.skip("lerobot is installed; the absent-lib path is unreachable")


# --- edge cases: dispatch errors, read-only save, custom adapter registration ---


def test_save_with_unknown_fmt_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError) as exc:
        save_episodes(_toy_dataset(), tmp_path / "x", fmt="parquet")  # type: ignore[arg-type]
    assert "parquet" in str(exc.value)


def test_load_with_unknown_fmt_raises(tmp_path: Path) -> None:
    p = tmp_path / "x.unknownext"
    p.write_text("")
    with pytest.raises(ValueError):
        load_episodes(p, fmt="parquet")  # type: ignore[arg-type]


def test_load_infers_fmt_from_suffix(tmp_path: Path) -> None:
    original = _toy_dataset()
    lance_path = tmp_path / "ds.lance"
    hdf5_path = tmp_path / "ds.h5"
    save_episodes(original, lance_path, fmt="lance")
    save_episodes(original, hdf5_path, fmt="hdf5")
    # no explicit fmt -> inferred from the suffix
    assert load_episodes(lance_path).fmt == "lance"
    assert load_episodes(hdf5_path).fmt == "hdf5"


def test_load_unknown_suffix_raises(tmp_path: Path) -> None:
    p = tmp_path / "ds.bin"
    p.write_text("")
    with pytest.raises(ValueError):
        load_episodes(p)


def test_save_lerobot_is_read_only(tmp_path: Path) -> None:
    """The lerobot adapter is read-only: saving through it raises (ValueError/ContractViolation)."""
    with pytest.raises((ValueError, ContractViolation)) as exc:
        save_episodes(_toy_dataset(), tmp_path / "x", fmt="lerobot")
    assert (
        "read-only" in str(exc.value).lower() or "read only" in str(exc.value).lower()
    )


def test_register_custom_adapter_roundtrips(tmp_path: Path) -> None:
    """A user-registered adapter plugs in through register_adapter and round-trips (02 §5.2)."""
    store: dict[str, EpisodeDataset] = {}

    def _saver(dataset: EpisodeDataset, path: Path) -> None:
        store[str(path)] = dataset

    def _loader(source: str | Path) -> EpisodeDataset:
        ds = store[str(source)]
        return EpisodeDataset(ds.episodes, path=Path(source), fmt="memstore")  # type: ignore[arg-type]

    register_adapter("memstore", loader=_loader, saver=_saver)  # type: ignore[arg-type]

    original = _toy_dataset()
    path = tmp_path / "ds.memstore"
    save_episodes(original, path, fmt="memstore")  # type: ignore[arg-type]
    read = load_episodes(path, fmt="memstore")  # type: ignore[arg-type]
    assert read.fmt == "memstore"
    _assert_windows_equal(_materialize(read), _materialize(original))


# --- the shared (de)serialization helpers: exact byte round-trip incl. dtypes ---


def test_tensor_bytes_roundtrip_float32() -> None:
    t = torch.randn(2, 3, 4)
    raw, label, shape = tensor_to_bytes(t)
    assert label == "float32"
    assert torch.equal(tensor_from_bytes(raw, label, shape), t)


def test_tensor_bytes_roundtrip_bfloat16() -> None:
    """bfloat16 has no numpy dtype; it is bit-cast through uint16 and restored exactly."""
    t = torch.randn(2, 3).to(torch.bfloat16)
    raw, label, shape = tensor_to_bytes(t)
    assert label == "bfloat16"
    back = tensor_from_bytes(raw, label, shape)
    assert back.dtype == torch.bfloat16
    assert torch.equal(back, t)


def test_dtype_label_rejects_unsupported_dtype() -> None:
    with pytest.raises(ValueError) as exc:
        dtype_label(torch.complex64)
    assert "unsupported tensor dtype" in str(exc.value)


def test_tensor_from_bytes_rejects_unknown_label() -> None:
    with pytest.raises(ValueError):
        tensor_from_bytes(b"\x00\x00\x00\x00", "complex64", (1,))


@pytest.mark.parametrize("fmt", ["lance", "hdf5"])
def test_bfloat16_obs_roundtrips_through_backends(tmp_path: Path, fmt: str) -> None:
    """A bfloat16 observation survives both on-disk backends byte-identically (covers the bf16 branch)."""
    spec = _spec()
    obs = torch.randn(_C, _T, _H, _W).to(torch.bfloat16)
    transitions = [
        Transition(
            obs_t=obs.clone(),
            action_t=torch.ones(_ACTION_DIM),
            obs_tp1=obs.clone(),
        )
        for _ in range(2)
    ]
    ds = EpisodeDataset(
        [
            Episode(
                episode_id="bf16",
                transitions=transitions,
                embodiment_id="toy-arm-4dof",
                modality="rgb-video",
                action_spec=spec,
                collection_meta={},
            )
        ]
    )
    path = tmp_path / f"bf16.{fmt if fmt == 'lance' else 'h5'}"
    save_episodes(ds, path, fmt=fmt)  # type: ignore[arg-type]
    read = load_episodes(path, fmt=fmt)  # type: ignore[arg-type]
    got = read.episodes[0].transitions[0].obs_t
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, obs)
