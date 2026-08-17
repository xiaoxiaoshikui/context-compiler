import tempfile
import unittest
from pathlib import Path

from context_compiler import ContextStore
from context_compiler.baselines import FullContextBaseline, RandomContextBaseline


def _seed_store(store: ContextStore) -> None:
    store.add(title="a.py", source="a.py", kind="code", content="print('a')\n" * 40)
    store.add(title="b.py", source="b.py", kind="code", content="print('b')\n" * 40)
    store.add(title="POLICY.md", source="POLICY.md", kind="constraint", content="Never do the bad thing.")


class BaselineTests(unittest.TestCase):
    def test_full_context_baseline_respects_budget_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            _seed_store(store)
            baseline = FullContextBaseline(store)
            first = baseline.compile("task", budget=60)
            second = baseline.compile("task", budget=60)
            self.assertLessEqual(first.used_tokens, 60)
            self.assertEqual(first.text, second.text)
            # Path-sorted order: "POLICY.md" sorts before "a.py"/"b.py".
            if first.selections:
                self.assertEqual(first.selections[0].source, "POLICY.md")

    def test_random_context_baseline_respects_budget_and_varies(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            _seed_store(store)
            baseline = RandomContextBaseline(store, seed=7)
            compiled = baseline.compile("task", budget=60)
            self.assertLessEqual(compiled.used_tokens, 60)
            orders = {
                tuple(s.item_id for s in baseline.compile("task", budget=1000).selections)
                for _ in range(8)
            }
            self.assertGreater(len(orders), 1, "expected shuffled order to vary across calls")


if __name__ == "__main__":
    unittest.main()
