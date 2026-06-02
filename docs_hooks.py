"""MkDocs build hooks (#77).

The spec corpus links to repo-root files (e.g. `../../SPEC.md`, `../../README.md`) that sit above the
site root and so are not site pages. Rewrite those above-root relative links to absolute GitHub URLs at
render time — the corpus files are left untouched (this issue renders, it does not edit the corpus), and
the rendered site has no broken links. Intra-corpus references (spec<->rfcs) resolve via the
GitHub-faithful toc slugify (docs_slug.slugify).
"""

from __future__ import annotations

import re

_BLOB = "https://github.com/AbdelStark/Lensemble/blob/main/"
# `](../../<path><#anchor>)` — a link from a docs/<area>/ file up to a repo-root file.
_ABOVE_ROOT = re.compile(r"\]\(\.\./\.\./([^)#]+)((?:#[^)]*)?)\)")


def on_page_markdown(markdown: str, **kwargs: object) -> str:
    return _ABOVE_ROOT.sub(lambda m: f"]({_BLOB}{m.group(1)}{m.group(2)})", markdown)
