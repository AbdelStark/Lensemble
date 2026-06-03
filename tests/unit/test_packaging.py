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
        "pylance>=0.10",  # the real Lance lib (import `lance`); the bare `lance` PyPI name is a typosquat
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


# --- the code/docs/data license split (09 §7, #72) ---

_ROOT = Path(__file__).resolve().parents[2]


def _toml() -> dict:
    with _PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def test_three_license_files_exist_at_repo_root() -> None:
    for name in ("LICENSE", "LICENSE-docs", "LICENSE-data"):
        assert (_ROOT / name).is_file(), f"missing license file: {name}"


def test_license_file_contents_are_the_right_licenses() -> None:
    code = (_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in code and "Version 2.0" in code
    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in code
    docs = (_ROOT / "LICENSE-docs").read_text(encoding="utf-8")
    assert "Creative Commons Attribution 4.0 International" in docs
    data = (_ROOT / "LICENSE-data").read_text(encoding="utf-8")
    assert "Community Data License Agreement - Permissive - Version 2.0" in data


def test_pyproject_declares_apache_code_license() -> None:
    project = _project()
    license_field = project["license"]
    # Accept either the table form {text = "Apache-2.0"} or the PEP 639 SPDX string.
    spdx = (
        license_field["text"] if isinstance(license_field, dict) else str(license_field)
    )
    assert spdx == "Apache-2.0"
    assert (
        "License :: OSI Approved :: Apache Software License" in project["classifiers"]
    )


def test_pyproject_bundles_all_three_license_files() -> None:
    license_files = _toml()["tool"]["setuptools"]["license-files"]
    assert set(license_files) == {"LICENSE", "LICENSE-docs", "LICENSE-data"}


def test_readme_links_each_license_file() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    for target in ("(LICENSE)", "(LICENSE-docs)", "(LICENSE-data)"):
        assert target in readme, f"README does not link {target}"
