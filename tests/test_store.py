import re
import tempfile
import unittest
from pathlib import Path

from context_compiler import ContextStore

_SINGLE_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class StoreTests(unittest.TestCase):
    def test_generated_ids_always_tokenize_as_a_single_token(self):
        # Every rendered item embeds its id in a `[CTX:...]` header (see
        # render.py), so the id's own token count is part of every budget
        # decision. A bare hex id can start with a digit, which
        # HeuristicTokenCounter's regex splits into two tokens (a leading
        # number plus a trailing identifier) instead of one -- making the
        # *same* content cost a different number of tokens purely by chance
        # across separate ingests. Sample many ids since the failure mode
        # is probabilistic (roughly 5/8 of raw hex ids start with a digit).
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            for i in range(50):
                item = store.add(title=f"item-{i}", content="x")
                self.assertRegex(
                    item.id,
                    _SINGLE_TOKEN_RE,
                    f"id {item.id!r} would not tokenize as a single token",
                )

    def test_list_order_is_deterministic_across_repeated_ingests(self):
        # Items ingested in the same batch commonly share the same
        # `updated_at` down to the stored precision. Without a stable
        # tiebreak in ContextStore.list()'s ORDER BY, candidate ranking --
        # and therefore compile() output -- was nondeterministic across
        # separate ingests of the identical repository content.
        def source_order(tmp_path: Path) -> list[str]:
            store = ContextStore(tmp_path)
            for name in ("b.py", "a.py", "d.py", "c.py"):
                store.add(title=name, source=name, kind="code", content="x")
            return [i.source for i in store.list(limit=10)]

        with tempfile.TemporaryDirectory() as tmp:
            orders = {tuple(source_order(Path(tmp) / f"ctx{i}.db")) for i in range(5)}
        self.assertEqual(len(orders), 1, f"list() order varied across ingests: {orders}")

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
