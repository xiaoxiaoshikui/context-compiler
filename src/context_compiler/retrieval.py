"""Pluggable first-stage candidate retrieval.

`ContextStore.candidate_search`'s SQLite FTS5 query is a hard recall
bottleneck: an item with zero exact-token overlap against the query never
becomes a candidate at all, no matter how well every other scoring signal
(importance, kind, dependency, omission risk) would justify including it.
`benchmarks/README.md` ("Real findings" #2) documents a concrete,
reproducible case: a task worded "cached prices" never retrieves
`cache.py`, whose content only ever says "cache," at any budget.

`TfidfRetriever` below fixes exactly that failure mode: it scores every
candidate item the caller hands it, not just the ones an exact term match
found, so a poorly-worded query degrades ranking quality rather than
silently deleting a candidate before scoring ever runs. It is still
classical, deterministic, dependency-free lexical retrieval -- a
meaningfully stronger drop-in for the FTS step, not a replacement for real
embedding-based retrieval (still open; see README.md limitation #1). A
real embedding retriever can implement the same `Retriever` protocol and
be swapped in without touching `ContextCompiler`.
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
from typing import TYPE_CHECKING, Protocol

from .models import ContextItem, ContextKind
from .scoring import term_set

if TYPE_CHECKING:
    from .store import ContextStore


class Retriever(Protocol):
    def rank(self, query: str, items: list[ContextItem], *, limit: int) -> list[ContextItem]: ...


def _item_text(item: ContextItem) -> str:
    return " ".join([item.title, item.content[:120_000], " ".join(item.tags), item.source])


class TfidfRetriever:
    """Cosine similarity over TF-IDF vectors, computed fresh per query.

    No persisted index -- the item counts this project targets (a
    repository's worth of context items, not a web-scale corpus) make
    recomputing per query fast enough, and keeps the implementation
    simple and dependency-free. Always returns up to `limit` items ranked
    by score, including items that score exactly 0.0 against the query --
    it degrades to "everything, unranked" rather than "nothing," so
    downstream scoring in `ContextCompiler` still gets a chance to select
    an item on non-lexical signals (importance, kind, dependency).
    """

    def rank(self, query: str, items: list[ContextItem], *, limit: int) -> list[ContextItem]:
        if not items:
            return []
        query_terms = term_set(query)
        if not query_terms:
            return items[:limit]

        doc_terms = [term_set(_item_text(item)) for item in items]
        n_docs = len(items)
        doc_freq: dict[str, int] = {}
        for terms in doc_terms:
            for t in query_terms & terms:
                doc_freq[t] = doc_freq.get(t, 0) + 1

        # idf is only ever needed over the query vocabulary -- a term the
        # query never asked about can't contribute to the dot product.
        idf = {t: math.log((n_docs + 1) / (doc_freq.get(t, 0) + 1)) + 1.0 for t in query_terms}
        query_vec = {t: idf[t] for t in query_terms}
        query_norm = math.sqrt(sum(v * v for v in query_vec.values())) or 1.0

        scored: list[tuple[float, int, ContextItem]] = []
        for idx, (item, terms) in enumerate(zip(items, doc_terms)):
            overlap = query_terms & terms
            if not overlap:
                scored.append((0.0, idx, item))
                continue
            doc_vec = {t: idf[t] for t in overlap}
            dot = sum(query_vec[t] * doc_vec[t] for t in overlap)
            doc_norm = math.sqrt(sum(v * v for v in doc_vec.values())) or 1.0
            scored.append((dot / (query_norm * doc_norm), idx, item))

        # Stable on ties (original/insertion order) via the idx tiebreak,
        # so results are deterministic regardless of dict/set iteration order.
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [item for _, _, item in scored[:limit]]


class FTSRetriever:
    """Wraps ContextStore's existing SQLite FTS5 `candidate_search`.

    Kept for direct comparison and for callers who want the old behavior
    back (`benchmarks/run.py --retriever fts`); `TfidfRetriever` is the
    compiler's default from here on. Unlike `TfidfRetriever`, this
    retriever queries the store directly rather than ranking the `items`
    it's handed, since FTS is inherently a SQL-side operation -- the
    `items` argument exists only to satisfy the shared `Retriever`
    protocol and is ignored.
    """

    def __init__(self, store: "ContextStore") -> None:
        self.store = store

    def rank(self, query: str, items: list[ContextItem], *, limit: int) -> list[ContextItem]:
        return self.store.candidate_search(query, limit=limit)


class EmbeddingRetriever:
    """Cosine similarity over real embedding vectors from a hosted API,
    instead of `TfidfRetriever`'s lexical term-overlap proxy.

    This is the "real embedding-based retrieval" the module docstring above
    flags as still open -- opt-in only, never the default, since it needs a
    real API key and makes a real (billed, network) call. Implemented over
    stdlib `urllib` against an OpenAI-compatible `/embeddings` endpoint --
    no new required dependency, consistent with this project's
    zero-required-dependency core (the same choice `benchmarks/real_eval.py`
    makes for its DeepSeek/GPT/Gemini calls).

    Item embeddings are cached on the instance by item id, since a
    `ContextCompiler` typically calls `rank()` many times against the same
    repository across a session (a synthetic benchmark's task x budget x
    repeat sweep, or `compile()` followed by `search()`) -- recomputing them
    per call would be both slow and needlessly expensive. The cache is
    intentionally unbounded and never invalidated: if the underlying store's
    content changes, construct a new `EmbeddingRetriever`.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        url: str = "https://api.openai.com/v1/embeddings",
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        self.api_key = api_key or os.environ.get(api_key_env)
        self.model = model
        self.url = url
        self._cache: dict[str, list[float]] = {}

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return [row["embedding"] for row in data["data"]]

    def _embeddings_for(self, items: list[ContextItem]) -> dict[str, list[float]]:
        missing = [item for item in items if item.id not in self._cache]
        # Batched, not one call per item -- the embeddings endpoint accepts
        # a list -- but bounded on both axes: OpenAI's embeddings endpoint
        # caps a single request at 300k input tokens total, confirmed
        # directly (a real repository's item pool blew through that at 256
        # items/batch x 8000 chars each). 2000 chars (~500 tokens) is
        # plenty for a *retrieval* embedding to capture "what is this file
        # broadly about" -- it doesn't need full-fidelity content the way
        # the final compiled context sent to a model does.
        batch_size, chars_per_item = 64, 2000
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            vectors = self._embed_batch([_item_text(item)[:chars_per_item] for item in batch])
            for item, vec in zip(batch, vectors):
                self._cache[item.id] = vec
        return {item.id: self._cache[item.id] for item in items}

    def rank(self, query: str, items: list[ContextItem], *, limit: int) -> list[ContextItem]:
        if not items:
            return []
        if not self.api_key:
            raise RuntimeError(
                "EmbeddingRetriever needs a real API key (pass api_key= or set the env "
                "var named by api_key_env, default OPENAI_API_KEY) -- it is never the "
                "default retriever precisely because it needs one."
            )
        item_vecs = self._embeddings_for(items)
        query_vec = self._embed_batch([query])[0]
        q_norm = math.sqrt(sum(v * v for v in query_vec)) or 1.0

        def cos_sim(vec: list[float]) -> float:
            dot = sum(a * b for a, b in zip(query_vec, vec))
            v_norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            return dot / (q_norm * v_norm)

        scored = [(cos_sim(item_vecs[item.id]), idx, item) for idx, item in enumerate(items)]
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [item for _, _, item in scored[:limit]]


def augment_with_critical_items(
    ranked: list[ContextItem], pool: list[ContextItem]
) -> list[ContextItem]:
    """Union in pinned/constraint/high-risk items regardless of retriever ranking.

    A retriever's job is relevance ranking; never silently dropping a
    safety-critical item is a policy decision `ContextCompiler` makes
    uniformly, independent of which retriever is plugged in.
    """
    ranked_ids = {i.id for i in ranked}
    critical = [
        i
        for i in pool
        if (i.pinned or i.kind is ContextKind.CONSTRAINT or i.omission_risk >= 0.75)
        and i.id not in ranked_ids
    ]
    return critical + ranked
