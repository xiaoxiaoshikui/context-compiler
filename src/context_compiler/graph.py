"""Real (if shallow) cross-file dependency graph.

`ingest.py`'s `extract_dependencies()` only ever produced opaque import
strings (module names, relative-import dots, bare paths) -- useful as text
for `scoring.py`'s term-overlap dependency signal, but not an actual
*graph*: there was no way to ask "what does this file depend on" as
resolved `ContextItem` ids, and no reverse direction ("what depends on
this file") at all.

`build_dependency_graph()` resolves those strings against the *same
store* -- e.g. `.gateway_client` (a Python relative import, one directory
up) or `./cache` (a relative JS/TS import) get matched to the actual
ingested item at that path, when one exists. Unresolvable strings
(standard-library imports, third-party packages, ambiguous basenames)
stay unresolved and are dropped from the graph, not guessed at.

This is intentionally a best-effort *within-repo* resolver, not a full
import system: no `sys.path`/package-root modeling, no `node_modules`
resolution, and Go/Java-style module paths only get a weak basename
fallback. See README.md limitation #3.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .models import ContextItem
from .store import ContextStore


def _strip_ext(path: str) -> str:
    p = PurePosixPath(path)
    return str(p.with_suffix("")) if p.suffix else str(p)


def _dirname(source: str) -> str:
    d = str(PurePosixPath(source).parent)
    return "" if d == "." else d


def _index_items(items: list[ContextItem]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Returns (by_full_path, by_basename), both mapping to item ids.

    `by_full_path` keys are extension-less, forward-slash paths relative
    to the ingest root (e.g. "payments/charge"). `__init__`/`index` files
    additionally register their containing directory, so package-style
    imports (`from . import x` resolving to a package, `import pkg`)
    still match.
    """
    by_full_path: dict[str, str] = {}
    by_basename: dict[str, list[str]] = {}
    for item in items:
        if not item.source:
            continue
        source = item.source.replace("\\", "/")
        stem = _strip_ext(source)
        by_full_path.setdefault(stem, item.id)
        name = PurePosixPath(stem).name
        if name in ("__init__", "index"):
            parent = _dirname(stem)
            if parent:
                by_full_path.setdefault(parent, item.id)
        by_basename.setdefault(name, []).append(item.id)
    return by_full_path, by_basename


def _go_up(path: str, n: int) -> str:
    for _ in range(n):
        path = posixpath.dirname(path)
    return path


def _resolve_python_relative(dep: str, origin_dir: str, by_full_path: dict[str, str]) -> str | None:
    level = len(dep) - len(dep.lstrip("."))
    rest = dep[level:]
    base = _go_up(origin_dir, max(0, level - 1))
    candidate = posixpath.join(base, rest.replace(".", "/")) if rest else base
    return by_full_path.get(candidate.strip("/"))


def _resolve_relative_path(dep: str, origin_dir: str, by_full_path: dict[str, str]) -> str | None:
    candidate = posixpath.normpath(posixpath.join(origin_dir, dep))
    return by_full_path.get(_strip_ext(candidate))


def _resolve_absolute(
    dep: str, by_full_path: dict[str, str], by_basename: dict[str, list[str]]
) -> str | None:
    as_path = dep.replace(".", "/")
    if as_path in by_full_path:
        return by_full_path[as_path]
    basename = dep.rsplit(".", 1)[-1].rsplit("/", 1)[-1]
    candidates = by_basename.get(basename, [])
    # An ambiguous basename (two files with the same name in different
    # directories) is left unresolved rather than guessed -- a wrong edge
    # is worse than a missing one.
    return candidates[0] if len(candidates) == 1 else None


def resolve_dependency(
    dep: str,
    origin_source: str,
    by_full_path: dict[str, str],
    by_basename: dict[str, list[str]],
) -> str | None:
    origin_dir = _dirname(origin_source)
    if dep.startswith("./") or dep.startswith("../"):
        return _resolve_relative_path(dep, origin_dir, by_full_path)
    if dep.startswith("."):
        return _resolve_python_relative(dep, origin_dir, by_full_path)
    return _resolve_absolute(dep, by_full_path, by_basename)


@dataclass(slots=True)
class DependencyGraph:
    forward: dict[str, set[str]] = field(default_factory=dict)  # item_id -> ids it depends on
    reverse: dict[str, set[str]] = field(default_factory=dict)  # item_id -> ids that depend on it
    unresolved: dict[str, set[str]] = field(default_factory=dict)  # item_id -> raw strings, no match

    def dependencies_of(self, item_id: str) -> set[str]:
        return self.forward.get(item_id, set())

    def dependents_of(self, item_id: str) -> set[str]:
        return self.reverse.get(item_id, set())

    def related(self, item_id: str, *, depth: int = 1) -> set[str]:
        """BFS over both directions, up to `depth` hops. Excludes `item_id` itself."""
        frontier = {item_id}
        seen = {item_id}
        for _ in range(max(0, depth)):
            next_frontier: set[str] = set()
            for node in frontier:
                next_frontier |= self.forward.get(node, set()) | self.reverse.get(node, set())
            next_frontier -= seen
            if not next_frontier:
                break
            seen |= next_frontier
            frontier = next_frontier
        seen.discard(item_id)
        return seen


def build_dependency_graph(store: ContextStore, *, limit: int = 5000) -> DependencyGraph:
    items = store.list(limit=limit)
    by_full_path, by_basename = _index_items(items)
    graph = DependencyGraph()
    for item in items:
        if not item.dependencies or not item.source:
            continue
        for dep in item.dependencies:
            target_id = resolve_dependency(dep, item.source, by_full_path, by_basename)
            if target_id is None or target_id == item.id:
                graph.unresolved.setdefault(item.id, set()).add(dep)
                continue
            graph.forward.setdefault(item.id, set()).add(target_id)
            graph.reverse.setdefault(target_id, set()).add(item.id)
    return graph
