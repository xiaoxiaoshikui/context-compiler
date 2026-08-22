import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from context_compiler import ContextStore
from context_compiler.models import ContextItem, ContextKind
from context_compiler.retrieval import (
    EmbeddingRetriever,
    FTSRetriever,
    TfidfRetriever,
    augment_with_critical_items,
)


def _item(item_id: str, title: str, content: str, **kwargs) -> ContextItem:
    return ContextItem(id=item_id, title=title, content=content, **kwargs)


class TfidfRetrieverTests(unittest.TestCase):
    def test_never_drops_a_zero_overlap_item_within_limit(self):
        # This is the exact failure mode benchmarks/README.md documents for
        # the old FTS-based retrieval: a task worded "cached prices" never
        # retrieving cache.py, whose content only ever says "cache" -- zero
        # exact term overlap. TfidfRetriever must still return it.
        items = [
            _item("relevant", "pricing_service.py", "get_price uses the price cache"),
            _item("zero-overlap", "cache.py", "class Cache: def __init__(self, ttl_seconds=30)"),
        ]
        ranked = TfidfRetriever().rank("cached prices", items, limit=10)
        self.assertEqual({i.id for i in ranked}, {"relevant", "zero-overlap"})

    def test_ranks_overlapping_item_above_zero_overlap_item(self):
        items = [
            _item("zero-overlap", "unrelated.py", "totally different subject matter"),
            _item("relevant", "auth.py", "oauth safari callback handler"),
        ]
        ranked = TfidfRetriever().rank("fix the safari oauth callback", items, limit=10)
        self.assertEqual(ranked[0].id, "relevant")

    def test_respects_limit(self):
        items = [_item(f"i{i}", f"file{i}.py", "some content here") for i in range(10)]
        ranked = TfidfRetriever().rank("some content", items, limit=3)
        self.assertEqual(len(ranked), 3)

    def test_empty_query_returns_items_up_to_limit(self):
        items = [_item(f"i{i}", f"file{i}.py", "content") for i in range(5)]
        ranked = TfidfRetriever().rank("   ", items, limit=3)
        self.assertEqual(len(ranked), 3)

    def test_empty_item_list_returns_empty(self):
        self.assertEqual(TfidfRetriever().rank("anything", [], limit=10), [])

    def test_deterministic_across_repeated_calls(self):
        items = [_item(f"i{i}", f"file{i}.py", "cache pricing service logic") for i in range(8)]
        first = TfidfRetriever().rank("cache pricing", items, limit=5)
        second = TfidfRetriever().rank("cache pricing", items, limit=5)
        self.assertEqual([i.id for i in first], [i.id for i in second])


class AugmentWithCriticalItemsTests(unittest.TestCase):
    def test_adds_missing_pinned_item(self):
        pinned = _item("pinned", "policy.md", "must never do X", pinned=True)
        ranked = [_item("other", "code.py", "code")]
        pool = [pinned, *ranked]
        result = augment_with_critical_items(ranked, pool)
        self.assertIn(pinned.id, {i.id for i in result})
        self.assertEqual(result[0].id, "pinned")  # critical items come first

    def test_adds_missing_constraint_item(self):
        constraint = _item("c1", "CONSTRAINTS.md", "invariant", kind=ContextKind.CONSTRAINT)
        pool = [constraint]
        result = augment_with_critical_items([], pool)
        self.assertEqual({i.id for i in result}, {"c1"})

    def test_does_not_duplicate_item_already_ranked(self):
        constraint = _item("c1", "CONSTRAINTS.md", "invariant", kind=ContextKind.CONSTRAINT)
        pool = [constraint]
        result = augment_with_critical_items([constraint], pool)
        self.assertEqual(len(result), 1)

    def test_leaves_non_critical_pool_items_out(self):
        ranked = [_item("a", "a.py", "a")]
        pool = [ranked[0], _item("b", "b.py", "unrelated code", importance=0.1)]
        result = augment_with_critical_items(ranked, pool)
        self.assertEqual({i.id for i in result}, {"a"})


class _FakeEmbeddingsResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._payload


def _mock_urlopen_returning(vector_for) -> mock.Mock:
    """Builds a urlopen replacement for a fake /embeddings endpoint.

    `vector_for(text) -> list[float]` decides the fake embedding for each
    string in the request's "input" list, so tests can control similarity
    without a real API call.
    """

    def _urlopen(req, timeout=60):
        body = json.loads(req.data.decode("utf-8"))
        data = [{"embedding": vector_for(text)} for text in body["input"]]
        return _FakeEmbeddingsResponse(json.dumps({"data": data}).encode("utf-8"))

    return mock.Mock(side_effect=_urlopen)


class EmbeddingRetrieverTests(unittest.TestCase):
    def _vector_for(self, text: str) -> list[float]:
        if "oauth safari callback handler" in text:
            return [1.0, 0.0]
        if "totally different subject matter" in text:
            return [0.0, 1.0]
        if text == "fix the safari oauth callback":
            return [1.0, 0.0]
        return [0.5, 0.5]

    def test_ranks_by_real_cosine_similarity_not_lexical_overlap(self):
        items = [
            _item("zero-overlap", "unrelated.py", "totally different subject matter"),
            _item("relevant", "auth.py", "oauth safari callback handler"),
        ]
        mock_urlopen = _mock_urlopen_returning(self._vector_for)
        with mock.patch("context_compiler.retrieval.urllib.request.urlopen", mock_urlopen):
            retriever = EmbeddingRetriever(api_key="test-key")
            ranked = retriever.rank("fix the safari oauth callback", items, limit=10)
        self.assertEqual(ranked[0].id, "relevant")

    def test_caches_item_embeddings_across_repeated_calls(self):
        items = [_item("a", "a.py", "oauth safari callback handler")]
        mock_urlopen = _mock_urlopen_returning(self._vector_for)
        with mock.patch("context_compiler.retrieval.urllib.request.urlopen", mock_urlopen):
            retriever = EmbeddingRetriever(api_key="test-key")
            retriever.rank("fix the safari oauth callback", items, limit=10)
            calls_after_first = mock_urlopen.call_count
            retriever.rank("a different query entirely", items, limit=10)
            calls_after_second = mock_urlopen.call_count
        # Second rank() should only embed the (uncached) query, not re-embed
        # the already-cached item -- one more urlopen call, not two.
        self.assertEqual(calls_after_second - calls_after_first, 1)

    def test_missing_api_key_raises_instead_of_silently_calling_out(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            retriever = EmbeddingRetriever(api_key=None)
            with self.assertRaises(RuntimeError):
                retriever.rank("anything", [_item("a", "a.py", "content")], limit=10)

    def test_empty_item_list_returns_empty_without_any_call(self):
        mock_urlopen = _mock_urlopen_returning(self._vector_for)
        with mock.patch("context_compiler.retrieval.urllib.request.urlopen", mock_urlopen):
            retriever = EmbeddingRetriever(api_key="test-key")
            result = retriever.rank("anything", [], limit=10)
        self.assertEqual(result, [])
        mock_urlopen.assert_not_called()


class FTSRetrieverTests(unittest.TestCase):
    def test_delegates_to_store_candidate_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            store.add(title="auth.py", source="auth.py", kind="code", content="oauth safari handler")
            store.add(title="unrelated.py", source="unrelated.py", kind="code", content="totally different")

            retriever = FTSRetriever(store)
            ranked = retriever.rank("safari oauth", [], limit=10)  # items arg is ignored
            self.assertTrue(any(i.source == "auth.py" for i in ranked))


if __name__ == "__main__":
    unittest.main()
