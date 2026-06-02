#!/usr/bin/env python3
"""Release-blocking gate runner and the version-agreement gate (09 §5.2; #78).

A tagged release runs the eight release-blocking gates of 09 §5.2 **in order** and fails closed on any
security-critical gate (``ResidencyViolation`` / ``CommitmentMismatch`` / ``NonDeterministicAggregation``)
with **no waiver path** — PyPI publication is irreversible, so a red gate stops the release before upload.

This module owns the gate *ordering* and the concrete **version-agreement** gate (``pyproject.toml``
``[project].version`` == ``lensemble.__version__`` == the ``CHANGELOG.md`` release version, with a
non-empty release block). The other gates are existing CI commands the release workflow invokes; here they
are declared so the ordering and the fail-closed contract are machine-checked.

Usage:
    python scripts/release_gates.py version-agreement [<repo_root>]   # the concrete gate
    python scripts/release_gates.py plan                              # print the ordered gate plan
"""

from __future__ import annotations

import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

_VERSION_RE = re.compile(r"^## \[(\d+\.\d+\.\d+)\] - \d{4}-\d{2}-\d{2}\s*$")
_INIT_VERSION_RE = re.compile(r'^__version__\s*=\s*["\'](\d+\.\d+\.\d+)["\']', re.M)


@dataclass(frozen=True)
class Gate:
    """One release-blocking gate (09 §5.2). ``security_critical`` gates are fail-closed, never waived."""

    name: str
    description: str
    security_critical: bool = False


# The eight release-blocking gates in release order (09 §5.2). Order is load-bearing: lint/type before
# tests, the security-critical determinism/recompute gates before the version/manifest gates, and the
# version-agreement gate before any build.
RELEASE_GATES: tuple[Gate, ...] = (
    Gate("lint-types", "ruff check + ruff format --check + pyright are clean"),
    Gate("unit-property", "the unit + property suite is green on the CPU fallback"),
    Gate(
        "aggregation-determinism",
        "the outer-step determinism self-check (INV-AGG-DETERMINISM)",
        security_critical=True,
    ),
    Gate(
        "reproducibility",
        "same-seed => same RunManifest config_hash (the reproducibility gate)",
    ),
    Gate(
        "public-recomputation",
        "recompute_alignment reproduces the coordinator's frame alignment (RFC-0006 §4)",
    ),
    Gate("coverage-docs", "coverage thresholds + the docs link-check pass"),
    Gate(
        "version-agreement",
        "pyproject version == lensemble.__version__ == CHANGELOG release version (non-empty block)",
    ),
    Gate("manifest-roundtrip", "a RunManifest schema-validates and round-trips JSON"),
)

# Security-critical gates whose failure aborts the release with no waiver (09 §5.2; 04 §1 principle 3).
SECURITY_CRITICAL_GATES: frozenset[str] = frozenset(
    g.name for g in RELEASE_GATES if g.security_critical
)


class ReleaseGateError(RuntimeError):
    """A release-blocking gate failed; the release must not proceed (no waiver)."""


def _pyproject_version(root: Path) -> str:
    """The package version resolved for ``[project].version`` — following the dynamic ``attr`` source."""
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    if "version" in project:
        return str(project["version"])
    # dynamic version via [tool.setuptools.dynamic].version = { attr = "lensemble.__version__" }
    if "version" in project.get("dynamic", []):
        return _init_version(root)
    raise ReleaseGateError(
        "pyproject [project].version is neither static nor a dynamic attr"
    )


def _init_version(root: Path) -> str:
    text = (root / "lensemble" / "__init__.py").read_text(encoding="utf-8")
    match = _INIT_VERSION_RE.search(text)
    if not match:
        raise ReleaseGateError("no __version__ = 'X.Y.Z' in lensemble/__init__.py")
    return match.group(1)


def _changelog_release_version(root: Path) -> str:
    """The newest released ``## [X.Y.Z] - date`` version in CHANGELOG.md, asserting its block is non-empty.

    Skips the ``## [Unreleased]`` block. Raises if there is no released version block or the block has no
    entries (a release must ship notes).
    """
    lines = (root / "CHANGELOG.md").read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        m = _VERSION_RE.match(line)
        if not m:
            continue
        body = []
        for nxt in lines[i + 1 :]:
            if nxt.startswith("## "):
                break
            body.append(nxt.strip())
        if not any(body):
            raise ReleaseGateError(
                f"CHANGELOG release block [{m.group(1)}] is empty; a release must ship notes"
            )
        return m.group(1)
    raise ReleaseGateError(
        "no released ## [X.Y.Z] - date block in CHANGELOG.md (only [Unreleased]); cut a release block"
    )


def check_version_agreement(root: Path) -> str:
    """The version-agreement gate (09 §5.2): the three version sources agree; the changelog block is non-empty.

    Returns the agreed version. Raises :class:`ReleaseGateError` if ``pyproject`` version,
    ``lensemble.__version__``, and the newest ``CHANGELOG.md`` release version are not all identical.
    """
    pyproject = _pyproject_version(root)
    init = _init_version(root)
    changelog = _changelog_release_version(root)
    if not (pyproject == init == changelog):
        raise ReleaseGateError(
            "version disagreement: pyproject="
            f"{pyproject!r}, __version__={init!r}, CHANGELOG={changelog!r}; "
            "align all three before releasing"
        )
    return pyproject


def run_gates(
    gates: tuple[Gate, ...], runner, *, stop_on_first: bool = True
) -> list[tuple[str, bool]]:
    """Run ``gates`` in order via ``runner(gate) -> bool``; fail closed on a security-critical failure.

    A security-critical gate that returns ``False`` immediately raises :class:`ReleaseGateError` (no
    waiver, 09 §5.2). A non-security gate failure also stops the run when ``stop_on_first``. Returns the
    ``(name, passed)`` results up to the stopping point.
    """
    results: list[tuple[str, bool]] = []
    for gate in gates:
        passed = bool(runner(gate))
        results.append((gate.name, passed))
        if not passed:
            if gate.security_critical:
                raise ReleaseGateError(
                    f"security-critical release gate {gate.name!r} failed; the release is blocked "
                    "(no waiver path): " + gate.description
                )
            if stop_on_first:
                break
    return results


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "plan"
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[1]
    if cmd == "version-agreement":
        try:
            version = check_version_agreement(root)
        except ReleaseGateError as exc:
            print(f"version-agreement: FAIL — {exc}", file=sys.stderr)
            return 1
        print(f"version-agreement: ok — {version}")
        return 0
    if cmd == "plan":
        for i, gate in enumerate(RELEASE_GATES, start=1):
            mark = " [security-critical, no waiver]" if gate.security_critical else ""
            print(f"{i}. {gate.name}{mark}: {gate.description}")
        return 0
    print(
        f"unknown command {cmd!r}; use 'version-agreement' or 'plan'", file=sys.stderr
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
