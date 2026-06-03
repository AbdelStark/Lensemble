"""Vendoring scaffold + deployment-stub discipline (RFC-0016 §2 / §4; issue #96).

This is the *scaffolding* guard for the `third_party/` vendoring backbone of
[RFC-0016 §2](../../docs/rfcs/RFC-0016-deployment-vendoring-topology.md). The real vendoring — cloning
each upstream repo at a *confirmed* commit SHA and committing its source — is a maintainer-gated step
that is **deferred** (the candidate SHAs in the `UPSTREAM.md` manifests are UNCONFIRMED research leads).
Until then these tests assert that:

- the per-project `UPSTREAM.md` VENDORING manifests exist and carry every RFC-0016 §2 field;
- no `stable_worldmodel`/`stable_pretraining` symbol leaks into `lensemble`'s public surface, checked
  **statically** (the subtrees are not yet present, so we never `import` them) — complements the
  no-PyPI-dep assertion of `tests/unit/test_packaging.py`;
- the SHA-drift guard is wired but *skips* until the SHAs are confirmed at vendor time;
- the `deploy/` IaC stubs (Compose, Helm chart, Kustomize base) exist and parse as YAML.

All checks are CPU-only, read tracked files, and download nothing (conventions §9).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THIRD_PARTY = _REPO_ROOT / "third_party"
_DEPLOY = _REPO_ROOT / "deploy"

# The vendored projects (RFC-0016 §2). Directory name == importable package name.
_VENDORED = ("stable_worldmodel", "stable_pretraining")


def _manifest_text(project: str) -> str:
    path = _THIRD_PARTY / project / "UPSTREAM.md"
    assert path.is_file(), f"missing VENDORING manifest: {path}"
    return path.read_text(encoding="utf-8")


def test_upstream_manifests_present_and_complete() -> None:
    """Each `third_party/<project>/UPSTREAM.md` carries every RFC-0016 §2 manifest field."""
    for project in _VENDORED:
        text = _manifest_text(project)
        lower = text.lower()
        # source URL (RFC-0016 §2)
        assert "source url" in lower, f"{project}: manifest missing 'source URL' field"
        assert "https://" in text, f"{project}: manifest missing an upstream URL"
        # vendored commit SHA field + the candidate SHA marked UNCONFIRMED
        assert "commit sha" in lower or "vendored sha" in lower, (
            f"{project}: manifest missing the commit-SHA field"
        )
        assert "UNCONFIRMED" in text, (
            f"{project}: candidate SHA must be marked UNCONFIRMED (maintainer-gated #96)"
        )
        # vendored date (left as not-yet-vendored)
        assert "vendored date" in lower, f"{project}: manifest missing 'vendored date'"
        # license: SPDX marker + MIT + the in-tree LICENSE path
        assert "spdx" in lower, f"{project}: manifest missing the SPDX license marker"
        assert "MIT" in text, f"{project}: manifest missing the MIT license id"
        assert "./LICENSE" in text, f"{project}: manifest missing in-tree LICENSE path"
        # local-modification log (empty for now)
        assert "modification" in lower, (
            f"{project}: manifest missing the local-modification log"
        )
        # upstream-sync procedure
        assert "sync" in lower, (
            f"{project}: manifest missing the upstream-sync procedure"
        )
        # the candidate SHA is actually recorded
        shas = {
            "stable_worldmodel": "40dff37fc983c5276ada65eb1c7873cefbcccd8a",
            "stable_pretraining": "d83f1426bb34049403642e82c1ce9fed3aa06435",
        }
        assert shas[project] in text, (
            f"{project}: candidate SHA not recorded in manifest"
        )


def test_patches_dir_is_tracked() -> None:
    """Each vendored project has a git-tracked (via `.gitkeep`) empty `patches/` dir."""
    for project in _VENDORED:
        keep = _THIRD_PARTY / project / "patches" / ".gitkeep"
        assert keep.is_file(), f"{project}: missing tracked patches/.gitkeep"


def _lensemble_init_source() -> str:
    return (_REPO_ROOT / "lensemble" / "__init__.py").read_text(encoding="utf-8")


def test_third_party_not_in_lensemble_public_surface() -> None:
    """No `stable_worldmodel`/`stable_pretraining` symbol leaks into `lensemble.__init__`.

    Checked STATICALLY (the subtrees are not yet vendored, so we must not `import` them): parse
    `lensemble/__init__.py` with `ast` and assert neither vendored package is imported there, and that
    `__all__` / the `_EXPORTS` lazy table name no vendored package. This preserves the RFC-0001 §3
    no-cycle DAG (third_party stays outside it) and complements
    `tests/unit/test_packaging.py::test_runtime_dependency_constraints` (the no-PyPI-dep assertion).
    """
    source = _lensemble_init_source()
    tree = ast.parse(source, "lensemble/__init__.py")

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    for project in _VENDORED:
        assert project not in imported, (
            f"lensemble.__init__ must not import vendored package {project!r}"
        )

    # __all__ is a literal list of strings; no vendored name may appear in the public surface.
    all_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            if isinstance(node.value, ast.List):
                all_names = {
                    elt.value
                    for elt in node.value.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                }
    assert all_names, "could not locate a literal __all__ in lensemble/__init__.py"
    for project in _VENDORED:
        assert not any(project in name for name in all_names), (
            f"vendored package {project!r} appears in lensemble.__all__"
        )

    # Belt-and-suspenders: no vendored package name appears anywhere in the module source.
    for project in _VENDORED:
        assert project not in source, (
            f"vendored package {project!r} referenced in lensemble/__init__.py source"
        )


def test_vendored_sha_is_confirmed() -> None:
    """SHA-drift guard — wired but skipping until the real vendor lands at a confirmed SHA.

    Once the maintainer confirms each candidate SHA against upstream and clones the source into
    `third_party/<project>/` (the maintainer-gated real-vendor step of #96), this becomes a real check:
    assert the vendored tree equals its recorded SHA with `patches/*.patch` applied (RFC-0016 Testing
    Strategy — the vendoring-drift guard). It is left skipping so the wiring exists without asserting on
    source that is not yet present.
    """
    pytest.skip(
        "stable-worldmodel/stable_pretraining not yet vendored at a confirmed SHA — "
        "#96 real-vendor step is maintainer-gated"
    )


# --- deploy/ IaC stubs (RFC-0016 §4) ---

_DEPLOY_FILES = (
    _DEPLOY / "compose.yaml",
    _DEPLOY / "helm" / "Chart.yaml",
    _DEPLOY / "kustomize" / "base" / "kustomization.yaml",
)


def test_deploy_stubs_are_valid() -> None:
    """The Compose / Helm / Kustomize stubs exist and parse as YAML (`yaml.safe_load`).

    PyYAML is available in-tree (it is pulled transitively by `omegaconf`, the Hydra config backend),
    so the structural check is a real parse rather than a text heuristic. Each file is asserted to be a
    mapping with the substrate's load-bearing top-level keys.
    """
    for path in _DEPLOY_FILES:
        assert path.is_file(), f"missing deploy stub: {path}"
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(doc, dict), f"{path} did not parse to a YAML mapping"

    compose = yaml.safe_load((_DEPLOY / "compose.yaml").read_text(encoding="utf-8"))
    assert "services" in compose, "compose.yaml missing top-level 'services'"
    assert {"coordinator", "participant"} <= set(compose["services"]), (
        "compose.yaml must define coordinator + participant service skeletons"
    )

    chart = yaml.safe_load(
        (_DEPLOY / "helm" / "Chart.yaml").read_text(encoding="utf-8")
    )
    assert chart.get("apiVersion") == "v2", "Helm Chart.yaml must be apiVersion v2"
    assert chart.get("name") == "lensemble", "Helm chart name must be 'lensemble'"
    assert "version" in chart, "Helm Chart.yaml missing a chart version"

    kustomization = yaml.safe_load(
        (_DEPLOY / "kustomize" / "base" / "kustomization.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert kustomization.get("kind") == "Kustomization", (
        "kustomization.yaml must declare kind: Kustomization"
    )
    assert "resources" in kustomization, "kustomization.yaml missing 'resources'"
