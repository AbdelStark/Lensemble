"""Packaging metadata: pinned deps, runtime, version agreement (09-release 5.1/8). Issue #71."""

from __future__ import annotations

import tomllib
from importlib import metadata
from pathlib import Path

import lensemble

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _project() -> dict:
    with _PYPROJECT.open("rb") as f:
        return tomllib.load(f)["project"]


def test_requires_python_is_311() -> None:
    assert _project()["requires-python"] == ">=3.11"


def test_runtime_dependency_constraints() -> None:
    deps = set(_project()["dependencies"])
    for required in (
        "torch>=2.4,<3",
        "numpy>=1.26",
        "safetensors>=0.4",
        "lance>=0.10",
        "h5py>=3.10",
        "hydra-core>=1.3",
        "omegaconf>=2.3",
        "pydantic>=2,<3",
    ):
        assert required in deps, f"missing pinned dependency: {required}"
    # vendored / pinned-by-content deps are NOT PyPI runtime deps (RFC-0016)
    joined = " ".join(deps)
    assert "stable-worldmodel" not in joined and "stable-pretraining" not in joined


def test_optional_groups_present() -> None:
    extras = _project()["optional-dependencies"]
    assert "opacus" in extras["dp"]
    assert set(extras["observability"]) >= {"tensorboard", "wandb"}
    assert {"pytest", "hypothesis", "ruff", "pyright"} <= set(extras["dev"])
    assert "verify" in extras  # Phase-2 extra declared


def test_cli_entry_point() -> None:
    assert _project()["scripts"]["lensemble"] == "lensemble.cli:main"


def test_dynamic_version_is_single_sourced() -> None:
    assert "version" in _project()["dynamic"]
    # the installed distribution version equals the one source of truth, lensemble.__version__
    assert metadata.version("lensemble") == lensemble.__version__


def test_no_pickle_dependency_surface() -> None:
    # tensors serialize via safetensors only (06-security 7); safetensors is a pinned runtime dep
    assert any(d.startswith("safetensors") for d in _project()["dependencies"])
