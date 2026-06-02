"""CHANGELOG.md discipline (Keep a Changelog; #76 / 09 §3).

Validates the changelog is well-formed: exactly one `## [Unreleased]` block, every released version block
titled `## [X.Y.Z] - YYYY-MM-DD`, and every category heading drawn from the six permitted values. The
file is the human-readable counterpart of the RunManifest / artifact headers, so it is machine-checked
rather than aspirational.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHANGELOG = _REPO_ROOT / "CHANGELOG.md"

_PERMITTED_CATEGORIES = {
    "Added",
    "Changed",
    "Deprecated",
    "Removed",
    "Fixed",
    "Security",
}
_VERSION_RE = re.compile(r"^## \[(\d+\.\d+\.\d+)\] - \d{4}-\d{2}-\d{2}$")
_H2_RE = re.compile(r"^## (.+)$")
_H3_RE = re.compile(r"^### (.+)$")


def _lines() -> list[str]:
    return _CHANGELOG.read_text(encoding="utf-8").splitlines()


def test_changelog_exists_at_repo_root() -> None:
    assert _CHANGELOG.is_file()


def test_exactly_one_unreleased_block() -> None:
    h2 = [m.group(1) for line in _lines() if (m := _H2_RE.match(line))]
    assert h2.count("[Unreleased]") == 1
    # Unreleased is the first version-level (##) section.
    assert h2[0] == "[Unreleased]"


def test_released_version_blocks_are_well_titled() -> None:
    # Every `## [...]` that is not Unreleased must be a `## [X.Y.Z] - YYYY-MM-DD` title.
    for line in _lines():
        m = _H2_RE.match(line)
        if m and m.group(1) != "[Unreleased]" and m.group(1).startswith("["):
            assert _VERSION_RE.match(line), f"malformed version heading: {line!r}"


def test_category_headings_are_permitted() -> None:
    headings = [m.group(1).strip() for line in _lines() if (m := _H3_RE.match(line))]
    assert headings, "expected at least one category heading"
    unknown = set(headings) - _PERMITTED_CATEGORIES
    assert not unknown, f"non-permitted changelog categories: {sorted(unknown)}"


def test_keep_a_changelog_header_present() -> None:
    text = _CHANGELOG.read_text(encoding="utf-8")
    assert text.startswith("# Changelog")
    assert "Keep a Changelog" in text
