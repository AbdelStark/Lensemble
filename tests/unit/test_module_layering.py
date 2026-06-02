"""Module-boundary import test: the ``lensemble`` import graph is an acyclic, banded DAG.

Enforces the L0–L9 dependency layering of
[RFC-0001 §3](../../docs/rfcs/RFC-0001-architecture.md#3-dependency-layering-no-cycles) /
[01 — Architecture §2](../../docs/spec/01-architecture.md#2-dependency-layering): a module never
imports a strictly-higher band, and there are no import cycles
([07 §5 module-boundary and import tests](../../docs/spec/07-testing-strategy.md#5-module-boundary-and-import-tests)).
Issue #4.

The band-to-module mapping enforced here:

| Band | Modules |
|------|---------|
| L0 | ``errors`` |
| L1 | ``config``, ``observability`` |
| L2 | ``contracts`` |
| L3 | ``data``, ``artifacts``, ``provenance`` |
| L4 | ``model``, ``gauge`` |
| L5 | ``aggregation``, ``privacy`` |
| L6 | ``eval`` |
| L7 | ``federation`` |
| L8 | ``verify`` |
| L9 | ``cli`` |

The graph is built statically with ``ast`` so no heavy optional dependency is imported. Edges inside
an ``if TYPE_CHECKING:`` block are excluded: they are type-checker-only annotations that impose no
runtime coupling and cannot form a runtime import cycle, so they do not constrain the banding (this is
how a lower band may name a higher band's *type* — e.g. ``data.probe`` annotating ``ReferenceEncoder``).
Cross-cutting edges into Band-0/1 (``errors``, ``config``, ``observability``) are always allowed.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parents[2] / "lensemble"

# L0–L9 band assignment (RFC-0001 §3). Lower index = lower in the stack.
_BAND: dict[str, int] = {
    "errors": 0,
    "config": 1,
    "observability": 1,
    "contracts": 2,
    "data": 3,
    "artifacts": 3,
    "provenance": 3,
    "model": 4,
    "gauge": 4,
    "aggregation": 5,
    "privacy": 5,
    "eval": 6,
    "federation": 7,
    "verify": 8,
    "cli": 9,
}

# An edge into the cross-cutting band is never "upward" (RFC-0001 §3 note).
_CROSS_CUTTING = frozenset({"errors", "config", "observability"})


def _is_type_checking_test(test: ast.expr) -> bool:
    """``TYPE_CHECKING`` or ``typing.TYPE_CHECKING`` as an ``if`` guard."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _runtime_import_nodes(tree: ast.Module) -> Iterator[ast.Import | ast.ImportFrom]:
    """Import nodes that execute at runtime — bodies of ``if TYPE_CHECKING:`` are skipped."""
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking_test(node.test):
            for stmt in node.body:
                for descendant in ast.walk(stmt):
                    skip.add(id(descendant))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and id(node) not in skip:
            yield node


def _component_of(path: Path) -> str | None:
    """The top-level ``lensemble`` subpackage a file belongs to; ``None`` for the package facade."""
    rel = path.relative_to(_PKG_DIR).parts
    if (
        len(rel) == 1
    ):  # a file directly under lensemble/ (errors.py, cli.py, __init__.py)
        stem = rel[0].removesuffix(".py")
        return None if stem == "__init__" else stem
    return rel[0]


def _target_components(
    node: ast.Import | ast.ImportFrom, pkg_parts: list[str]
) -> set[str]:
    """The ``lensemble`` components a single import statement references."""
    candidates: list[str] = []
    if isinstance(node, ast.Import):
        candidates = [alias.name for alias in node.names]
    else:  # ImportFrom — resolve relative levels against the importing package
        if node.level:
            base = pkg_parts[: len(pkg_parts) - (node.level - 1)]
            if node.module:
                base = [*base, *node.module.split(".")]
        else:
            base = node.module.split(".") if node.module else []
        prefix = ".".join(base)
        if prefix:
            candidates.append(prefix)
        for alias in node.names:
            candidates.append(f"{prefix}.{alias.name}" if prefix else alias.name)

    targets: set[str] = set()
    for dotted in candidates:
        parts = dotted.split(".")
        if len(parts) >= 2 and parts[0] == "lensemble" and parts[1] in _BAND:
            targets.add(parts[1])
    return targets


def build_import_graph() -> set[tuple[str, str]]:
    """Static intra-``lensemble`` import graph at component granularity (runtime edges only)."""
    edges: set[tuple[str, str]] = set()
    for file in _PKG_DIR.rglob("*.py"):
        src = _component_of(file)
        if src is None:
            continue
        module_parts = list(file.relative_to(_PKG_DIR.parent).with_suffix("").parts)
        pkg_parts = module_parts if file.name == "__init__.py" else module_parts[:-1]
        tree = ast.parse(file.read_text(encoding="utf-8"), str(file))
        for node in _runtime_import_nodes(tree):
            for tgt in _target_components(node, pkg_parts):
                if tgt != src:
                    edges.add((src, tgt))
    return edges


def upward_edges(edges: set[tuple[str, str]]) -> list[str]:
    """Edges that point to a strictly-higher band (excluding the cross-cutting band), named."""
    return sorted(
        f"{src} -> {tgt}"
        for src, tgt in edges
        if tgt not in _CROSS_CUTTING and _BAND[tgt] > _BAND[src]
    )


def find_cycle(edges: set[tuple[str, str]]) -> list[str] | None:
    """A cycle in the directed graph as a node path, or ``None`` if the graph is a DAG."""
    adjacency: dict[str, set[str]] = {}
    for src, tgt in edges:
        adjacency.setdefault(src, set()).add(tgt)

    visiting, done = 1, 2
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        state[node] = visiting
        stack.append(node)
        for nxt in sorted(adjacency.get(node, set())):
            if state.get(nxt, 0) == 0:
                cycle = visit(nxt)
                if cycle is not None:
                    return cycle
            elif state.get(nxt) == visiting:
                return stack[stack.index(nxt) :] + [nxt]
        stack.pop()
        state[node] = done
        return None

    for node in sorted(adjacency):
        if state.get(node, 0) == 0:
            cycle = visit(node)
            if cycle is not None:
                return cycle
    return None


_GRAPH = build_import_graph()


def test_every_subpackage_has_a_band() -> None:
    # A new top-level lensemble package must be assigned a band, or the layering test goes blind to it.
    discovered = {
        c for c in (_component_of(f) for f in _PKG_DIR.rglob("*.py")) if c is not None
    }
    unbanded = discovered - set(_BAND)
    assert not unbanded, (
        f"top-level packages with no band assignment: {sorted(unbanded)}"
    )


def test_import_graph_is_acyclic() -> None:
    cycle = find_cycle(_GRAPH)
    assert cycle is None, f"import cycle: {' -> '.join(cycle) if cycle else ''}"


def test_no_module_imports_a_higher_band() -> None:
    violations = upward_edges(_GRAPH)
    assert not violations, f"upward (back-edge) imports: {violations}"


def test_named_back_edge_prohibitions_hold() -> None:
    # The specific prohibitions called out in 01 — Architecture §2.
    forbidden = {
        ("contracts", "model"),
        ("contracts", "data"),
        ("contracts", "federation"),
        ("model", "federation"),
        ("model", "eval"),
        ("eval", "federation"),
    }
    present = forbidden & _GRAPH
    assert not present, f"prohibited back-edges present: {sorted(present)}"


def test_detector_names_a_synthetic_upward_edge() -> None:
    # A fixture upward edge is detected and named, so a real violation is a build failure.
    assert upward_edges({("contracts", "federation")}) == ["contracts -> federation"]
    # A downward edge and a cross-cutting edge are not flagged.
    assert upward_edges({("federation", "contracts"), ("model", "errors")}) == []


def test_detector_finds_a_synthetic_cycle() -> None:
    cycle = find_cycle({("model", "gauge"), ("gauge", "model")})
    assert cycle is not None
    assert set(cycle) == {"model", "gauge"}
    assert find_cycle({("model", "gauge"), ("gauge", "errors")}) is None
