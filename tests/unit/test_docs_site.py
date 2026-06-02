"""Documentation site build + API-doc coverage (09 §1, v1.0; #77).

Offline checks (run everywhere): every `mkdocs.yml` nav target exists, and the API reference documents
every public symbol re-exported from `lensemble`. The zero-warning `mkdocs build --strict` runs both in the
`docs` CI job and here when mkdocs is installed (skipped otherwise).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

import lensemble

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _REPO_ROOT / "docs"
_MKDOCS = _REPO_ROOT / "mkdocs.yml"
_REFERENCE = _DOCS / "reference.md"

_NAV_MD_RE = re.compile(r":\s*([\w./-]+\.md)\s*$", re.M)
_PUBLIC_SYMBOLS = [s for s in lensemble.__all__ if s != "__version__"]


def test_mkdocs_config_exists() -> None:
    assert _MKDOCS.is_file()
    assert "site_name: Lensemble" in _MKDOCS.read_text(encoding="utf-8")


def test_every_nav_target_exists() -> None:
    targets = _NAV_MD_RE.findall(_MKDOCS.read_text(encoding="utf-8"))
    assert targets, "no nav .md targets parsed from mkdocs.yml"
    missing = [t for t in targets if not (_DOCS / t).is_file()]
    assert not missing, f"nav targets missing under docs/: {missing}"


def test_nav_covers_the_spec_and_rfc_corpus() -> None:
    nav = _NAV_MD_RE.findall(_MKDOCS.read_text(encoding="utf-8"))
    spec = {p.name for p in (_DOCS / "spec").glob("*.md")}
    rfcs = {p.name for p in (_DOCS / "rfcs").glob("*.md")}
    nav_names = {Path(t).name for t in nav}
    assert spec <= nav_names, (
        f"spec sections missing from nav: {sorted(spec - nav_names)}"
    )
    assert rfcs <= nav_names, f"RFCs missing from nav: {sorted(rfcs - nav_names)}"


def test_api_reference_documents_every_public_symbol() -> None:
    ref = _REFERENCE.read_text(encoding="utf-8")
    missing = [s for s in _PUBLIC_SYMBOLS if s not in ref]
    assert not missing, f"public symbols absent from the API reference: {missing}"
    # the page invokes mkdocstrings on the documented subpackages
    assert "::: lensemble.federation" in ref and "::: lensemble.provenance" in ref


def test_site_builds_strict_with_zero_warnings(tmp_path) -> None:
    pytest.importorskip("mkdocs")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
            "-d",
            str(tmp_path / "site"),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"mkdocs --strict failed:\n{result.stderr}"
    rendered = (tmp_path / "site" / "reference" / "index.html").read_text(
        encoding="utf-8"
    )
    absent = [s for s in _PUBLIC_SYMBOLS if s not in rendered]
    assert not absent, f"public symbols not rendered in the API reference: {absent}"
