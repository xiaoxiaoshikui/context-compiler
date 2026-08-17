from __future__ import annotations

import argparse
from typing import Any

from ..compiler import ContextCompiler
from ..ingest import RepositoryIngestor
from ..models import ContextKind, RenderLevel
from ..store import ContextStore
from ..tokenizer import make_token_counter


def build_mcp(*, store_path: str = ".context-compiler.db", tokenizer: str = "heuristic"):
    try:
        from mcp.server import MCPServer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "MCP adapter requires the optional dependency. "
            "Install with: pip install -e '.[mcp]'"
        ) from exc

    store = ContextStore(store_path)
    compiler = ContextCompiler(store, counter=make_token_counter(tokenizer))
    mcp = MCPServer(
        "Context Compiler",
        instructions=(
            "Compile task-specific minimum sufficient context. Prefer context_compile "
            "before loading raw resources; call context_expand only when more fidelity is needed."
        ),
    )

    @mcp.tool()
    def context_add(
        title: str,
        content: str,
        kind: str = "note",
        source: str = "",
        tags: list[str] | None = None,
        dependencies: list[str] | None = None,
        importance: float = 0.5,
        omission_risk: float = 0.2,
        pinned: bool = False,
    ) -> dict[str, Any]:
        """Store raw context losslessly and return its CTX id."""
        item = store.add(
            title=title,
            content=content,
            kind=ContextKind(kind),
            source=source,
            tags=tags or [],
            dependencies=dependencies or [],
            importance=importance,
            omission_risk=omission_risk,
            pinned=pinned,
        )
        return item.to_dict()

    @mcp.tool()
    def context_ingest(path: str, max_file_bytes: int = 1_500_000) -> dict[str, Any]:
        """Ingest a local text file or source repository into the context store."""
        return RepositoryIngestor(store, max_file_bytes=max_file_bytes).ingest(path).to_dict()

    @mcp.tool()
    def context_compile(task: str, budget: int = 8000, max_candidates: int = 120) -> dict[str, Any]:
        """Return the best task-specific working set that fits the token budget."""
        return compiler.compile(task, budget, max_candidates=max_candidates).to_dict()

    @mcp.tool()
    def context_search(task: str, limit: int = 10) -> list[dict[str, Any]]:
        """Rank candidate context items for a task without compiling them."""
        return compiler.search(task, limit=limit)

    @mcp.tool()
    def context_expand(item_id: str, level: str = "L4", task: str = "") -> dict[str, Any]:
        """Expand a CTX item reversibly from pointer/summary to excerpts or raw content."""
        return compiler.expand(item_id, level=RenderLevel(level), task=task)

    @mcp.tool()
    def context_stats() -> dict[str, Any]:
        """Return store statistics."""
        return store.stats()

    @mcp.resource("context://item/{item_id}")
    def raw_context_item(item_id: str) -> str:
        """Read the exact raw stored content for a CTX id."""
        item = store.get(item_id)
        if not item:
            return f"Unknown context item: {item_id}"
        return item.content

    return mcp


def run_mcp(*, store_path: str, transport: str = "stdio", tokenizer: str = "heuristic") -> None:
    mcp = build_mcp(store_path=store_path, tokenizer=tokenizer)
    if transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport="streamable-http", stateless_http=True, json_response=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=".context-compiler.db")
    p.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    p.add_argument("--tokenizer", choices=["heuristic", "tiktoken"], default="heuristic")
    args = p.parse_args(argv)
    run_mcp(store_path=args.db, transport=args.transport, tokenizer=args.tokenizer)


if __name__ == "__main__":
    main()
