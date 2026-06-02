#!/usr/bin/env python3
"""Intra-corpus documentation link checker (07 §8 gate 7; #70).

Walks every Markdown file under the given paths, extracts each relative link ``[label](relpath#anchor)``,
and asserts that the target file exists relative to the linking file and — when an ``#anchor`` is present
and the target is Markdown — that the anchor resolves to a real heading under GitHub's heading-slug rule.
External (``http``/``https``/``mailto``) links and in-code-fence links are out of scope; the gate checks
intra-corpus relative references only (conventions §3).

Usage:
    python scripts/check_docs_links.py docs/ SPEC.md

Exits 0 if every relative reference resolves; otherwise prints each unresolved ``(file:line) -> target``
triple and exits 1. The slug rule mirrors GitHub's (github-slugger): lowercase, drop every character that
is not a letter, number, underscore, hyphen, or whitespace, then replace whitespace with hyphens, with a
``-1``/``-2`` suffix disambiguating duplicate headings.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# `[label](target)` but not an image `![label](target)`; target is everything up to the first ')'.
_LINK_RE = re.compile(r"(?<!\!)\[(?P<label>[^\]]*)\]\((?P<target>[^)]+)\)")
# A markdown link inside a heading: collapse to its label before slugifying.
_INLINE_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(?P<text>.*?)\s*#*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_SLUG_STRIP_RE = re.compile(r"[^\w\s-]", re.UNICODE)
_EXTERNAL = ("http://", "https://", "mailto:", "tel:")


def github_slug(heading_text: str) -> str:
    """The GitHub heading-slug of a heading's rendered text (no duplicate suffix)."""
    text = _INLINE_LINK_RE.sub(r"\1", heading_text)  # [label](url) -> label
    text = text.strip().lower()
    text = _SLUG_STRIP_RE.sub("", text)  # drop punctuation, keep word/space/hyphen
    return re.sub(r"\s", "-", text)  # each whitespace char -> a hyphen


def heading_anchors(text: str) -> set[str]:
    """Every heading slug in a Markdown document, with GitHub's duplicate ``-1``/``-2`` disambiguation."""
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    in_fence = False
    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING_RE.match(line)
        if not m:
            continue
        base = github_slug(m.group("text"))
        n = counts.get(base, 0)
        anchors.add(base if n == 0 else f"{base}-{n}")
        counts[base] = n + 1
    return anchors


def iter_links(text: str):
    """Yield ``(line_number, target)`` for every non-image link outside a fenced code block."""
    in_fence = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in _LINK_RE.finditer(line):
            yield lineno, m.group("target").strip()


def collect_markdown(paths: list[Path]) -> list[Path]:
    """Resolve the CLI paths to a sorted, de-duplicated list of Markdown files."""
    files: set[Path] = set()
    for p in paths:
        if p.is_dir():
            files.update(q.resolve() for q in p.rglob("*.md"))
        elif p.suffix == ".md":
            files.add(p.resolve())
    return sorted(files)


def check(paths: list[Path]) -> list[str]:
    """Return a list of human-readable problem strings (empty == every reference resolves)."""
    md_files = collect_markdown(paths)
    anchor_cache: dict[Path, set[str]] = {}

    def anchors_of(path: Path) -> set[str]:
        if path not in anchor_cache:
            anchor_cache[path] = heading_anchors(path.read_text(encoding="utf-8"))
        return anchor_cache[path]

    problems: list[str] = []
    for md in md_files:
        text = md.read_text(encoding="utf-8")
        for lineno, target in iter_links(text):
            if target.startswith(_EXTERNAL):
                continue
            rel, _, anchor = target.partition("#")
            if rel == "":  # same-document anchor
                dest = md
            else:
                dest = (md.parent / rel).resolve()
                if not dest.exists():
                    problems.append(f"{md}:{lineno} -> {target} (missing file)")
                    continue
            if anchor and dest.suffix == ".md":
                if anchor not in anchors_of(dest):
                    problems.append(
                        f"{md}:{lineno} -> {target} (missing anchor #{anchor})"
                    )
    return problems


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: check_docs_links.py <path> [<path> ...]", file=sys.stderr)
        return 2
    problems = check([Path(a) for a in argv])
    if problems:
        print(
            f"docs link-check: {len(problems)} unresolved reference(s):",
            file=sys.stderr,
        )
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print("docs link-check: all intra-corpus references resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
