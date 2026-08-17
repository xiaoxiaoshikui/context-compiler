import tempfile
import unittest
from pathlib import Path

from context_compiler import ContextStore


class StoreTests(unittest.TestCase):
    def test_roundtrip_and_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            item = store.add(
                title="Invariant",
                content="Never replay authorization codes.",
                kind="constraint",
                pinned=True,
                tags=["oauth", "security"],
                importance=1.0,
                omission_risk=1.0,
            )
            loaded = store.get(item.id)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.content, item.content)
            self.assertTrue(loaded.pinned)
            self.assertIn("oauth", loaded.tags)

            item.content = "Never replay OAuth authorization codes."
            store.upsert(item)
            loaded2 = store.get(item.id)
            self.assertEqual(loaded2.content, item.content)
            self.assertEqual(store.stats()["items"], 1)


if __name__ == "__main__":
    unittest.main()
