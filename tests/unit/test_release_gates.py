"""Release-blocking gate runner and the version-agreement gate (09 §5.2; #78).

Pins the gate ordering, the fail-closed (no-waiver) semantics for security-critical gates, and the
version-agreement gate over fixture trees (agreement passes; a deliberate mismatch fails). The actual
gate commands (ruff/pytest/coverage/...) run in CI; this checks the orchestration contract.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "release_gates.py"
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release.yml"
_WORKFLOW_DIR = _REPO_ROOT / ".github" / "workflows"
_USES_DIRECTIVE = re.compile(r"(?m)^\s*(?:-\s+)?uses:\s+['\"]?([^'\"\s#]+)")


def _load():
    spec = importlib.util.spec_from_file_location("release_gates", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = (
        mod  # the @dataclass needs its module registered for type resolution
    )
    spec.loader.exec_module(mod)
    return mod


rg = _load()


def _write_tree(
    tmp_path: Path, *, pyproject_v: str, init_v: str, changelog_v: str | None
) -> Path:
    (tmp_path / "lensemble").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "lensemble"\nversion = "{pyproject_v}"\n', encoding="utf-8"
    )
    (tmp_path / "lensemble" / "__init__.py").write_text(
        f'__version__ = "{init_v}"\n', encoding="utf-8"
    )
    block = (
        f"## [{changelog_v}] - 2026-06-02\n\n### Added\n- a thing\n"
        if changelog_v
        else ""
    )
    (tmp_path / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## [Unreleased]\n\n{block}", encoding="utf-8"
    )
    return tmp_path


# --- the ordered, fail-closed gate plan (09 §5.2) ---


def test_release_gate_plan_is_ordered_and_complete() -> None:
    names = [g.name for g in rg.RELEASE_GATES]
    assert names == [
        "lint-types",
        "unit-property",
        "aggregation-determinism",
        "reproducibility",
        "public-recomputation",
        "coverage-docs",
        "version-agreement",
        "manifest-roundtrip",
    ]
    assert "aggregation-determinism" in rg.SECURITY_CRITICAL_GATES


def test_tag_workflow_materializes_all_gates_before_build() -> None:
    text = _WORKFLOW.read_text(encoding="utf-8")
    for number in range(1, 9):
        assert f"Gate {number} —" in text
    build_job = re.search(
        r"(?ms)^  build-and-smoke:\n(?P<body>.*?)(?=^  publish:)", text
    )
    assert build_job is not None
    assert "needs: release-gates" in build_job.group("body")
    assert "uv lock --check" in text


def test_external_workflow_actions_are_pinned_to_full_commit_shas() -> None:
    failures: list[str] = []
    matched = 0
    workflow_paths = sorted(
        [*_WORKFLOW_DIR.glob("*.yml"), *_WORKFLOW_DIR.glob("*.yaml")]
    )

    for path in workflow_paths:
        for target in _USES_DIRECTIVE.findall(path.read_text(encoding="utf-8")):
            if target.startswith(("./", "docker://")):
                continue
            matched += 1
            _, separator, ref = target.rpartition("@")
            if separator != "@" or re.fullmatch(r"[0-9a-f]{40}", ref) is None:
                failures.append(f"{path.relative_to(_REPO_ROOT)}: {target}")

    assert matched > 0
    assert not failures, "unpinned external workflow actions:\n" + "\n".join(failures)


def test_manual_dispatch_cannot_publish_and_secret_is_upload_scoped() -> None:
    text = _WORKFLOW.read_text(encoding="utf-8")
    tag_and_auth = (
        "if: startsWith(github.ref, 'refs/tags/v') && "
        "steps.pypi_auth.outputs.available == 'true'"
    )
    assert text.count(tag_and_auth) == 3
    assert "TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}" in text
    assert '-p "$PYPI_API_TOKEN"' not in text


def test_release_write_permission_is_scoped_to_publish_job() -> None:
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"(?m)^permissions:\n  contents: read$", text)
    publish_job = re.search(r"(?ms)^  publish:\n(?P<body>.*)$", text)
    assert publish_job is not None
    assert re.search(
        r"(?m)^    permissions:\n(?:      #.*\n)*      contents: write$",
        publish_job.group("body"),
    )


def test_security_critical_failure_is_fail_closed_no_waiver() -> None:
    failing = "aggregation-determinism"

    def runner(gate):
        return gate.name != failing  # the security-critical gate fails

    with pytest.raises(rg.ReleaseGateError):
        rg.run_gates(rg.RELEASE_GATES, runner)


def test_non_security_failure_stops_the_run_without_raising() -> None:
    def runner(gate):
        return gate.name != "manifest-roundtrip"  # a non-security gate fails last

    results = rg.run_gates(rg.RELEASE_GATES, runner)
    assert results[-1] == ("manifest-roundtrip", False)
    assert all(passed for _, passed in results[:-1])


def test_all_gates_pass_runs_to_completion() -> None:
    results = rg.run_gates(rg.RELEASE_GATES, lambda gate: True)
    assert len(results) == len(rg.RELEASE_GATES)
    assert all(passed for _, passed in results)


# --- the version-agreement gate over fixture trees ---


def test_version_agreement_passes_when_all_three_match(tmp_path) -> None:
    root = _write_tree(
        tmp_path, pyproject_v="0.2.0", init_v="0.2.0", changelog_v="0.2.0"
    )
    assert rg.check_version_agreement(root) == "0.2.0"


def test_version_agreement_fails_on_changelog_mismatch(tmp_path) -> None:
    root = _write_tree(
        tmp_path, pyproject_v="0.2.0", init_v="0.2.0", changelog_v="0.1.0"
    )
    with pytest.raises(rg.ReleaseGateError):
        rg.check_version_agreement(root)


def test_version_agreement_fails_on_init_mismatch(tmp_path) -> None:
    root = _write_tree(
        tmp_path, pyproject_v="0.2.0", init_v="0.3.0", changelog_v="0.2.0"
    )
    with pytest.raises(rg.ReleaseGateError):
        rg.check_version_agreement(root)


def test_version_agreement_fails_without_a_release_block(tmp_path) -> None:
    root = _write_tree(tmp_path, pyproject_v="0.2.0", init_v="0.2.0", changelog_v=None)
    with pytest.raises(rg.ReleaseGateError):  # only [Unreleased]; no cut release block
        rg.check_version_agreement(root)


def test_version_agreement_fails_on_empty_release_block(tmp_path) -> None:
    (tmp_path / "lensemble").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "lensemble"\nversion = "0.2.0"\n', encoding="utf-8"
    )
    (tmp_path / "lensemble" / "__init__.py").write_text(
        '__version__ = "0.2.0"\n', encoding="utf-8"
    )
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [0.2.0] - 2026-06-02\n\n",
        encoding="utf-8",
    )
    with pytest.raises(rg.ReleaseGateError):  # a release must ship non-empty notes
        rg.check_version_agreement(tmp_path)


def test_dynamic_pyproject_version_follows_init(tmp_path) -> None:
    # the live pyproject uses dynamic = ["version"] with attr = lensemble.__version__
    (tmp_path / "lensemble").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "lensemble"\ndynamic = ["version"]\n', encoding="utf-8"
    )
    (tmp_path / "lensemble" / "__init__.py").write_text(
        '__version__ = "0.2.0"\n', encoding="utf-8"
    )
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [0.2.0] - 2026-06-02\n\n### Added\n- x\n",
        encoding="utf-8",
    )
    assert rg.check_version_agreement(tmp_path) == "0.2.0"
