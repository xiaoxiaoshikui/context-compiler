import tempfile
import unittest
from pathlib import Path

from context_compiler import ContextStore
from context_compiler.ingest import RepositoryIngestor


class IngestTests(unittest.TestCase):
    def test_ingest_repository_and_python_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            (root / "a.py").write_text("import json\nfrom app.auth import login\n\ndef f():\n    pass\n", encoding="utf-8")
            (root / "README.md").write_text("hello", encoding="utf-8")
            (root / "blob.bin").write_bytes(b"\x00\x01\x02")
            store = ContextStore(Path(tmp) / "ctx.db")
            report = RepositoryIngestor(store).ingest(root)
            self.assertEqual(report.added, 2)
            item = store.get_by_source("a.py")
            self.assertIsNotNone(item)
            self.assertIn("json", item.dependencies)
            self.assertIn("app.auth", item.dependencies)


if __name__ == "__main__":
    unittest.main()
