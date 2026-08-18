from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .models import ContextItem, ContextKind


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_item_id() -> str:
    """A random id that always tokenizes as exactly one token.

    Every rendered representation embeds the item id in its `[CTX:...]`
    header (see render.py), so the id's own token count is part of every
    budget decision. `HeuristicTokenCounter`'s regex matches a leading
    letter plus following alphanumerics as a single token, but a bare hex
    string starting with a digit (e.g. "8cfd6750...") splits into a
    leading number token plus a trailing identifier token -- one token
    more. A plain `uuid.uuid4().hex[:16]` starts with a digit roughly
    5/8 of the time, so the *same* file, re-ingested, could cost a
    different number of tokens purely by chance, occasionally flipping a
    budget-boundary decision between runs. Prefixing with a fixed letter
    makes every id single-token, deterministically.
    """
    return f"c{uuid.uuid4().hex[:15]}"


class ContextStore:
    def __init__(self, path: str | Path = ".context-compiler.db") -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS context_items (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT '',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    dependencies_json TEXT NOT NULL DEFAULT '[]',
                    importance REAL NOT NULL DEFAULT 0.5,
                    omission_risk REAL NOT NULL DEFAULT 0.2,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_context_source ON context_items(source);
                CREATE INDEX IF NOT EXISTS idx_context_kind ON context_items(kind);
                """
            )
            try:
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS context_fts USING fts5("
                    "id UNINDEXED, title, content, tags, source)"
                )
            except sqlite3.OperationalError:
                # Some minimal SQLite builds omit FTS5; all core features still work.
                pass

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()

    def add(
        self,
        *,
        title: str,
        content: str,
        kind: ContextKind | str = ContextKind.NOTE,
        source: str = "",
        tags: tuple[str, ...] | list[str] = (),
        dependencies: tuple[str, ...] | list[str] = (),
        importance: float = 0.5,
        omission_risk: float = 0.2,
        pinned: bool = False,
        metadata: dict | None = None,
        item_id: str | None = None,
    ) -> ContextItem:
        now = utc_now()
        item = ContextItem(
            id=item_id or new_item_id(),
            title=title,
            content=content,
            kind=ContextKind(kind),
            source=source,
            tags=tuple(tags),
            dependencies=tuple(dependencies),
            importance=max(0.0, min(1.0, float(importance))),
            omission_risk=max(0.0, min(1.0, float(omission_risk))),
            pinned=bool(pinned),
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )
        return self.upsert(item)

    def upsert(self, item: ContextItem) -> ContextItem:
        existing = self.get(item.id)
        now = utc_now()
        if existing:
            item = replace(item, created_at=existing.created_at, updated_at=now)
        elif not item.created_at:
            item = replace(item, created_at=now, updated_at=now)
        elif not item.updated_at:
            item = replace(item, updated_at=now)

        payload = (
            item.id,
            item.title,
            item.content,
            item.kind.value,
            item.source,
            json.dumps(list(item.tags), ensure_ascii=False),
            json.dumps(list(item.dependencies), ensure_ascii=False),
            item.importance,
            item.omission_risk,
            int(item.pinned),
            json.dumps(item.metadata, ensure_ascii=False),
            self._hash(item.content),
            item.created_at,
            item.updated_at,
        )
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO context_items (
                    id,title,content,kind,source,tags_json,dependencies_json,
                    importance,omission_risk,pinned,metadata_json,content_hash,
                    created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    content=excluded.content,
                    kind=excluded.kind,
                    source=excluded.source,
                    tags_json=excluded.tags_json,
                    dependencies_json=excluded.dependencies_json,
                    importance=excluded.importance,
                    omission_risk=excluded.omission_risk,
                    pinned=excluded.pinned,
                    metadata_json=excluded.metadata_json,
                    content_hash=excluded.content_hash,
                    updated_at=excluded.updated_at
                """,
                payload,
            )
            try:
                conn.execute("DELETE FROM context_fts WHERE id=?", (item.id,))
                conn.execute(
                    "INSERT INTO context_fts(id,title,content,tags,source) VALUES (?,?,?,?,?)",
                    (item.id, item.title, item.content, " ".join(item.tags), item.source),
                )
            except sqlite3.OperationalError:
                pass
        return item

    def get(self, item_id: str) -> ContextItem | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM context_items WHERE id=?", (item_id,)).fetchone()
        return self._row_to_item(row) if row else None

    def get_by_source(self, source: str) -> ContextItem | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM context_items WHERE source=? ORDER BY updated_at DESC LIMIT 1",
                (source,),
            ).fetchone()
        return self._row_to_item(row) if row else None

    def list(self, *, limit: int = 1000, kind: str | None = None) -> list[ContextItem]:
        query = "SELECT * FROM context_items"
        params: list[object] = []
        if kind:
            query += " WHERE kind=?"
            params.append(kind)
        # `source` is a final, reproducible tiebreaker: items ingested in the
        # same batch commonly share the same `updated_at` down to the stored
        # precision, and without a deterministic tiebreak SQLite's order for
        # those tied rows is not guaranteed to be stable across separate
        # connections/processes -- silently making candidate ranking, and
        # therefore compile() output, nondeterministic for the same task and
        # budget. `id` (a randomly generated UUID prefix) would be internally
        # consistent within one run but *not* reproducible across separate
        # ingests of the same repository, since it's re-randomized every
        # insert -- `source` is stable across runs for the same input.
        query += " ORDER BY pinned DESC, updated_at DESC, source ASC, id ASC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_item(r) for r in rows]

    def candidate_search(self, query: str, *, limit: int = 200) -> list[ContextItem]:
        terms = [t for t in _simple_terms(query) if len(t) > 1]
        if not terms:
            return self.list(limit=limit)
        fts_query = " OR ".join(f'"{t.replace(chr(34), "")}"' for t in terms[:20])
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT id FROM context_fts WHERE context_fts MATCH ? "
                    "ORDER BY bm25(context_fts), source LIMIT ?",
                    (fts_query, limit),
                ).fetchall()
            ids = [r["id"] for r in rows]
            items = [self.get(i) for i in ids]
            result = [i for i in items if i is not None]
            # Always include hard constraints / high omission-risk items even if
            # lexical retrieval misses them. This is an explicit defense against
            # semantic relevance dropping a critical invariant.
            critical = [
                i
                for i in self.list(limit=2000)
                if (i.pinned or i.kind is ContextKind.CONSTRAINT or i.omission_risk >= 0.75)
                and i.id not in ids
            ]
            merged = critical + result
            return merged[: max(limit, len(critical))]
        except sqlite3.OperationalError:
            return self.list(limit=limit)

    def delete(self, item_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM context_items WHERE id=?", (item_id,))
            try:
                conn.execute("DELETE FROM context_fts WHERE id=?", (item_id,))
            except sqlite3.OperationalError:
                pass
        return cur.rowcount > 0

    def stats(self) -> dict[str, object]:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) c FROM context_items").fetchone()["c"]
            kinds = conn.execute(
                "SELECT kind, COUNT(*) c FROM context_items GROUP BY kind ORDER BY c DESC"
            ).fetchall()
            pinned = conn.execute(
                "SELECT COUNT(*) c FROM context_items WHERE pinned=1"
            ).fetchone()["c"]
            chars = conn.execute(
                "SELECT COALESCE(SUM(LENGTH(content)),0) c FROM context_items"
            ).fetchone()["c"]
        return {
            "items": total,
            "pinned": pinned,
            "characters": chars,
            "by_kind": {r["kind"]: r["c"] for r in kinds},
            "db": self.path,
        }

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> ContextItem:
        return ContextItem(
            id=row["id"],
            title=row["title"],
            content=row["content"],
            kind=ContextKind(row["kind"]),
            source=row["source"],
            tags=tuple(json.loads(row["tags_json"])),
            dependencies=tuple(json.loads(row["dependencies_json"])),
            importance=float(row["importance"]),
            omission_risk=float(row["omission_risk"]),
            pinned=bool(row["pinned"]),
            metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


def _simple_terms(text: str) -> list[str]:
    import re

    return re.findall(r"[A-Za-z_][A-Za-z0-9_\-./]*|[\u3400-\u9fff]", text.lower())
