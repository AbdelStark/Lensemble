"""Every module in the canonical layout imports on the CPU path (conventions 1, issue #2)."""

from __future__ import annotations

import importlib
import subprocess
import sys

SUBPACKAGES = [
    "contracts",
    "model",
    "gauge",
    "federation",
    "aggregation",
    "privacy",
    "data",
    "provenance",
    "eval",
    "config",
    "artifacts",
    "observability",
    "verify",
]

MODULES = [
    "lensemble.errors",
    "lensemble.cli",
    "lensemble.model.encoder",
    "lensemble.model.predictor",
    "lensemble.model.action_head",
    "lensemble.model.objective",
    "lensemble.model.sigreg",
    "lensemble.gauge.anchor",
    "lensemble.gauge.procrustes",
    "lensemble.gauge.drift",
    "lensemble.federation.coordinator",
    "lensemble.federation.participant",
    "lensemble.federation.round",
    "lensemble.federation.outer_optimizer",
    "lensemble.aggregation.secure_agg",
    "lensemble.aggregation.masking",
    "lensemble.privacy.dp",
    "lensemble.privacy.accountant",
    "lensemble.data.dataset",
    "lensemble.data.loaders",
    "lensemble.data.residency",
    "lensemble.data.probe",
    "lensemble.data.adapters",
    "lensemble.provenance.merkle",
    "lensemble.provenance.commit",
    "lensemble.provenance.ledger",
    "lensemble.eval.mpc",
    "lensemble.eval.harness",
    "lensemble.eval.metrics",
    "lensemble.config.schema",
    "lensemble.config.manifest",
    "lensemble.config.seed",
    "lensemble.artifacts.checkpoint",
    "lensemble.artifacts.schema",
    "lensemble.artifacts.hashing",
    "lensemble.observability.logging",
    "lensemble.observability.metrics",
    "lensemble.observability.redaction",
    "lensemble.verify.recompute",
    "lensemble.verify.stark",
]


def test_subpackages_exist() -> None:
    for sp in SUBPACKAGES:
        importlib.import_module(f"lensemble.{sp}")


def test_all_layout_modules_import() -> None:
    for module in MODULES:
        importlib.import_module(module)


def test_import_lensemble_is_lazy() -> None:
    """`import lensemble` must not eager-load subpackages (issue #2 Notes: import-light)."""
    code = (
        "import sys, lensemble; "
        "assert 'lensemble.model' not in sys.modules, 'model eager-loaded'; "
        "assert 'lensemble.federation' not in sys.modules, 'federation eager-loaded'; "
        "assert 'torch' not in sys.modules and 'lance' not in sys.modules; "
        "assert isinstance(lensemble.__version__, str)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
