"""Dynamic-env seeded silo registry generator (#284)."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib import util
from pathlib import Path

from lensemble.data import load_episodes, validate_phase3_registry_against_manifest

_SCRIPT_PATH = Path("scripts/dynamic_env_silos.py")
_SPEC = util.spec_from_file_location("dynamic_env_silos_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
build_dynamic_env_manifest = _MODULE.build_dynamic_env_manifest
build_dynamic_env_registry = _MODULE.build_dynamic_env_registry


def test_dynamic_env_manifest_and_placeholder_registry_validate() -> None:
    manifest, heldout = build_dynamic_env_manifest(num_silos=4, seed=42)
    registry = build_dynamic_env_registry(manifest)
    validate_phase3_registry_against_manifest(registry, manifest)
    assert heldout not in {
        p.data.data_ref for p in manifest.participants if p.data is not None
    }
    assert {p.format for p in registry.participants} == {"synthetic-dynamic"}
    assert {p.publication_status for p in registry.participants} == {"placeholder"}
    assert all(
        "placeholder" in (p.publication_blocker or "") for p in registry.participants
    )


def test_dynamic_env_silos_are_deterministic_disjoint_and_long_enough() -> None:
    manifest, heldout = build_dynamic_env_manifest(
        num_silos=2, seed=55, steps=8, window_steps=4
    )
    refs = [p.data.data_ref for p in manifest.participants if p.data is not None]
    assert len(set(refs + [heldout])) == 3
    for ref in refs + [heldout]:
        dataset = load_episodes(ref, fmt="synthetic-dynamic")
        assert list(dataset.windows(4))
        again = load_episodes(ref, fmt="synthetic-dynamic")
        assert dataset.episodes[0].episode_id == again.episodes[0].episode_id


def test_dynamic_env_silos_script_writes_valid_outputs(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    registry_path = tmp_path / "registry.json"
    plan_path = tmp_path / "plan.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/dynamic_env_silos.py",
            "--num-silos",
            "2",
            "--steps",
            "8",
            "--window-steps",
            "4",
            "--manifest-output",
            str(manifest_path),
            "--registry-output",
            str(registry_path),
            "--plan-output",
            str(plan_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "heldout_source" in result.stdout
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["manifest"] == str(manifest_path)
    assert plan["registry"] == str(registry_path)
    assert len(plan["silo_sources"]) == 2


def test_dynamic_env_silos_push_requires_repo_and_records_missing_token(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    registry_path = tmp_path / "registry.json"
    plan_path = tmp_path / "plan.json"
    monkeypatch.delenv("HF_TOKEN", raising=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/dynamic_env_silos.py",
            "--num-silos",
            "2",
            "--steps",
            "8",
            "--window-steps",
            "4",
            "--manifest-output",
            str(manifest_path),
            "--registry-output",
            str(registry_path),
            "--plan-output",
            str(plan_path),
            "--push",
            "--out-repo",
            "example/dynamic-env-silos",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "skipped dynamic-env metadata push" in result.stdout
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["publication"]["pushed"] is False
    assert "HF_TOKEN" in plan["publication"]["blocker"]

    missing_repo = subprocess.run(
        [
            sys.executable,
            "scripts/dynamic_env_silos.py",
            "--manifest-output",
            str(tmp_path / "missing-manifest.json"),
            "--registry-output",
            str(tmp_path / "missing-registry.json"),
            "--plan-output",
            str(tmp_path / "missing-plan.json"),
            "--push",
        ],
        capture_output=True,
        text=True,
    )
    assert missing_repo.returncode != 0
    assert "--out-repo is required" in missing_repo.stderr
