"""GitHub-faithful heading slugify for the MkDocs `toc` extension (#77).

Makes the rendered site's anchor IDs match GitHub's heading slugs, so the spec corpus' GitHub-style
cross-references resolve on the site. Mirrors `scripts/check_docs_links.py::github_slug` (the algorithm
the docs link-check gate validates): lowercase; drop every character that is not a letter, number,
underscore, hyphen, or whitespace; then replace each whitespace character with `separator` (so an em
dash's two surrounding spaces become a double hyphen, as GitHub renders).
"""

from __future__ import annotations

import re

_STRIP = re.compile(r"[^\w\s-]", re.UNICODE)


def slugify(value: str, separator: str) -> str:
    text = _STRIP.sub("", value.strip().lower())
    return re.sub(r"\s", separator, text)
