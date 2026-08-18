import tempfile
import unittest
from pathlib import Path

from context_compiler import ContextStore
from context_compiler.graph import (
    _index_items,
    build_dependency_graph,
    resolve_dependency,
)


class ResolveDependencyTests(unittest.TestCase):
    def setUp(self):
        self.by_full_path = {
            "gateway_client": "id-gateway",
            "payments/charge": "id-charge",
            "payments/shared/utils": "id-utils",
            "payments": "id-payments-pkg",  # __init__.py alias
            "cache": "id-cache",
        }
        self.by_basename = {
            "gateway_client": ["id-gateway"],
            "charge": ["id-charge"],
            "utils": ["id-utils"],
            "cache": ["id-cache"],
        }

    def test_python_relative_same_directory(self):
        # "from .gateway_client import X" in a root-level charge.py
        got = resolve_dependency(".gateway_client", "charge.py", self.by_full_path, self.by_basename)
        self.assertEqual(got, "id-gateway")

    def test_python_relative_within_subpackage(self):
        # "from .gateway_client import X" inside payments/charge.py
        by_full_path = {**self.by_full_path, "payments/gateway_client": "id-nested-gw"}
        got = resolve_dependency(".gateway_client", "payments/charge.py", by_full_path, self.by_basename)
        self.assertEqual(got, "id-nested-gw")

    def test_python_relative_parent_package(self):
        # "from ..shared.utils import Y" inside payments/sub/foo.py
        got = resolve_dependency(
            "..shared.utils", "payments/sub/foo.py", self.by_full_path, self.by_basename
        )
        self.assertEqual(got, "id-utils")

    def test_python_from_import_bare_name(self):
        # "from . import gateway_client" was rewritten by ingest.py to ".gateway_client"
        got = resolve_dependency(".gateway_client", "charge.py", self.by_full_path, self.by_basename)
        self.assertEqual(got, "id-gateway")

    def test_python_absolute_dotted_path(self):
        got = resolve_dependency("payments.charge", "unrelated.py", self.by_full_path, self.by_basename)
        self.assertEqual(got, "id-charge")

    def test_python_absolute_unresolvable_stdlib(self):
        got = resolve_dependency("os", "charge.py", self.by_full_path, self.by_basename)
        self.assertIsNone(got)

    def test_js_relative_import(self):
        got = resolve_dependency("./cache", "handlers.py", self.by_full_path, self.by_basename)
        self.assertEqual(got, "id-cache")

    def test_ambiguous_basename_is_left_unresolved(self):
        by_basename = {**self.by_basename, "utils": ["id-utils", "id-other-utils"]}
        by_full_path = {k: v for k, v in self.by_full_path.items() if k != "payments/shared/utils"}
        got = resolve_dependency("utils", "somewhere.py", by_full_path, by_basename)
        self.assertIsNone(got)

    def test_package_init_alias_resolves_bare_package_import(self):
        got = resolve_dependency("payments", "unrelated.py", self.by_full_path, self.by_basename)
        self.assertEqual(got, "id-payments-pkg")


class IndexItemsTests(unittest.TestCase):
    def test_init_py_registers_directory_alias(self):
        from context_compiler.models import ContextItem

        items = [
            ContextItem(id="pkg", title="payments/__init__.py", content="", source="payments/__init__.py"),
            ContextItem(id="mod", title="payments/charge.py", content="", source="payments/charge.py"),
        ]
        by_full_path, by_basename = _index_items(items)
        self.assertEqual(by_full_path["payments"], "pkg")
        self.assertEqual(by_full_path["payments/charge"], "mod")


class BuildDependencyGraphTests(unittest.TestCase):
    def test_resolves_edges_within_a_flat_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            handler = store.add(
                title="handler.py",
                source="handler.py",
                kind="code",
                content="x",
                dependencies=[".validators", ".processor"],
            )
            validators = store.add(
                title="validators.py", source="validators.py", kind="code", content="x"
            )
            store.add(title="processor.py", source="processor.py", kind="code", content="x")
            store.add(title="unrelated.py", source="unrelated.py", kind="code", content="x")

            graph = build_dependency_graph(store)
            self.assertIn(validators.id, graph.dependencies_of(handler.id))
            self.assertIn(handler.id, graph.dependents_of(validators.id))
            # A leaf with no outgoing deps of its own is still reachable via
            # the reverse edge -- this is exactly what makes the graph boost
            # apply to validators.py even though it imports nothing itself.
            self.assertEqual(graph.dependencies_of(validators.id), set())
            self.assertIn(handler.id, graph.dependents_of(validators.id))

    def test_related_bfs_respects_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            a = store.add(title="a.py", source="a.py", kind="code", content="x", dependencies=[".b"])
            b = store.add(title="b.py", source="b.py", kind="code", content="x", dependencies=[".c"])
            c = store.add(title="c.py", source="c.py", kind="code", content="x")

            graph = build_dependency_graph(store)
            self.assertEqual(graph.related(a.id, depth=1), {b.id})
            self.assertEqual(graph.related(a.id, depth=2), {b.id, c.id})
            self.assertNotIn(a.id, graph.related(a.id, depth=5))

    def test_unresolvable_dependency_is_recorded_not_dropped_silently(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            item = store.add(
                title="a.py", source="a.py", kind="code", content="x", dependencies=["os", "sys"]
            )
            graph = build_dependency_graph(store)
            self.assertEqual(graph.dependencies_of(item.id), set())
            self.assertEqual(graph.unresolved[item.id], {"os", "sys"})


if __name__ == "__main__":
    unittest.main()
