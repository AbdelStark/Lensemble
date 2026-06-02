"""Documentation link-check gate (07 §8 gate 7; #70).

The committed corpus must resolve cleanly; a missing file or a missing ``#anchor`` must fail; and the
GitHub heading-slug rule must map punctuation/spaced headings to the anchors the corpus actually uses.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "check_docs_links.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_docs_links", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_checker = _load_checker()


def test_committed_corpus_resolves() -> None:
    problems = _checker.check([_REPO_ROOT / "docs", _REPO_ROOT / "SPEC.md"])
    assert problems == [], f"corpus has unresolved references: {problems}"


def test_missing_file_is_a_failure(tmp_path) -> None:
    (tmp_path / "a.md").write_text("see [b](b.md) — does not exist\n", encoding="utf-8")
    problems = _checker.check([tmp_path])
    assert any("missing file" in p for p in problems)


def test_missing_anchor_is_a_failure(tmp_path) -> None:
    (tmp_path / "target.md").write_text("# Real Heading\n", encoding="utf-8")
    (tmp_path / "src.md").write_text(
        "see [x](target.md#nope-not-here)\n", encoding="utf-8"
    )
    problems = _checker.check([tmp_path])
    assert any("missing anchor" in p for p in problems)


def test_valid_anchor_resolves(tmp_path) -> None:
    (tmp_path / "target.md").write_text("## 9. Foo Bar\n", encoding="utf-8")
    (tmp_path / "src.md").write_text("[x](target.md#9-foo-bar)\n", encoding="utf-8")
    assert _checker.check([tmp_path]) == []


def test_external_and_fenced_links_are_ignored(tmp_path) -> None:
    (tmp_path / "a.md").write_text(
        "[ext](https://example.com/missing) is skipped\n\n"
        "```\n[code](nonexistent.md) is in a fence\n```\n",
        encoding="utf-8",
    )
    assert _checker.check([tmp_path]) == []


def test_github_slug_rule_matches_corpus_anchors() -> None:
    # Punctuation dropped, spaces -> hyphens, an em dash leaves a double hyphen (matching the corpus).
    assert (
        _checker.github_slug("9. Determinism, dtype, device")
        == "9-determinism-dtype-device"
    )
    assert (
        _checker.github_slug("4. `Transition` and `Episode` — the data layer")
        == "4-transition-and-episode--the-data-layer"
    )
    assert _checker.github_slug("Added") == "added"


def test_duplicate_headings_get_suffixes() -> None:
    anchors = _checker.heading_anchors("# Foo\n\n## Foo\n\n## Foo\n")
    assert {"foo", "foo-1", "foo-2"} <= anchors
