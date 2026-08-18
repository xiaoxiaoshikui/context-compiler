from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .compiler import ContextCompiler
from .ingest import RepositoryIngestor
from .models import ContextKind, RenderLevel
from .store import ContextStore
from .tokenizer import make_token_counter


def _json(data: object) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _store(args: argparse.Namespace) -> ContextStore:
    return ContextStore(args.db)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ctxc",
        description="Compile minimum sufficient context under a hard token budget.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--db", default=".context-compiler.db", help="SQLite context store")
    p.add_argument(
        "--tokenizer",
        default="heuristic",
        choices=["heuristic", "tiktoken"],
        help="Budget counter. Core default is dependency-free heuristic.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Initialize the SQLite store")

    add = sub.add_parser("add", help="Add a context item")
    add.add_argument("--title", required=True)
    content = add.add_mutually_exclusive_group(required=True)
    content.add_argument("--content")
    content.add_argument("--content-file")
    add.add_argument("--kind", choices=[k.value for k in ContextKind], default="note")
    add.add_argument("--source", default="")
    add.add_argument("--tag", action="append", default=[])
    add.add_argument("--dependency", action="append", default=[])
    add.add_argument("--importance", type=float, default=0.5)
    add.add_argument("--risk", type=float, default=0.2)
    add.add_argument("--pinned", action="store_true")

    ing = sub.add_parser("ingest", help="Ingest a file or repository")
    ing.add_argument("path")
    ing.add_argument("--max-file-bytes", type=int, default=1_500_000)
    ing.add_argument("--ignore", action="append", default=[], help="Glob relative to root")

    comp = sub.add_parser("compile", help="Compile a working set for a task")
    comp.add_argument("task")
    comp.add_argument("--budget", type=int, required=True)
    comp.add_argument("--reserve", type=int, default=0)
    comp.add_argument("--max-candidates", type=int, default=120)
    comp.add_argument("--json", action="store_true", dest="as_json")
    comp.add_argument("--audit", action="store_true")

    search = sub.add_parser("search", help="Rank context candidates")
    search.add_argument("task")
    search.add_argument("--limit", type=int, default=10)

    exp = sub.add_parser("expand", help="Expand a CTX id to a higher-resolution view")
    exp.add_argument("item_id")
    exp.add_argument("--level", choices=[x.value for x in RenderLevel], default="L4")
    exp.add_argument("--task", default="")

    ls = sub.add_parser("list", help="List stored items")
    ls.add_argument("--limit", type=int, default=100)
    ls.add_argument("--kind", choices=[k.value for k in ContextKind])

    sub.add_parser("stats", help="Show store statistics")

    delete = sub.add_parser("delete", help="Delete an item")
    delete.add_argument("item_id")

    serve = sub.add_parser("serve", help="Run local JSON HTTP API (stdlib only)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    mcp = sub.add_parser("mcp", help="Run the optional MCP v2 adapter")
    mcp.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = _store(args)
    try:
        return _dispatch(args, store)
    except FileNotFoundError as exc:
        print(f"error: file not found: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        # Covers, e.g., `--tokenizer tiktoken` or `mcp` without the optional
        # dependency installed -- both already raise a message telling the
        # user what to `pip install`; this just stops it from surfacing as
        # a raw traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace, store: ContextStore) -> int:
    if args.command == "init":
        _json(store.stats())
        return 0

    if args.command == "add":
        text = args.content
        if args.content_file:
            text = Path(args.content_file).read_text(encoding="utf-8")
        item = store.add(
            title=args.title,
            content=text or "",
            kind=args.kind,
            source=args.source,
            tags=args.tag,
            dependencies=args.dependency,
            importance=args.importance,
            omission_risk=args.risk,
            pinned=args.pinned,
        )
        _json(item.to_dict())
        return 0

    if args.command == "ingest":
        report = RepositoryIngestor(
            store,
            max_file_bytes=args.max_file_bytes,
            ignore_globs=args.ignore,
        ).ingest(args.path)
        _json(report.to_dict())
        return 0

    counter = make_token_counter(args.tokenizer)
    compiler = ContextCompiler(store, counter=counter)

    if args.command == "compile":
        result = compiler.compile(
            args.task,
            args.budget,
            max_candidates=args.max_candidates,
            reserve_tokens=args.reserve,
        )
        if args.as_json:
            data = result.to_dict()
            if not args.audit:
                data.pop("audit", None)
            _json(data)
        else:
            print(result.text)
            print(
                f"\n---\nused={result.used_tokens}/{result.budget} "
                f"items={len(result.selections)} candidates={result.candidate_count}",
                file=sys.stderr,
            )
            if args.audit:
                for row in result.audit:
                    print("AUDIT", row, file=sys.stderr)
        return 0

    if args.command == "search":
        _json(compiler.search(args.task, limit=args.limit))
        return 0

    if args.command == "expand":
        try:
            _json(compiler.expand(args.item_id, level=args.level, task=args.task))
        except KeyError:
            print(f"Unknown item id: {args.item_id}", file=sys.stderr)
            return 2
        return 0

    if args.command == "list":
        _json([i.to_dict() for i in store.list(limit=args.limit, kind=args.kind)])
        return 0

    if args.command == "stats":
        _json(store.stats())
        return 0

    if args.command == "delete":
        ok = store.delete(args.item_id)
        _json({"deleted": ok, "item_id": args.item_id})
        return 0 if ok else 2

    if args.command == "serve":
        from .adapters.http_server import run_http

        run_http(store_path=args.db, host=args.host, port=args.port, tokenizer=args.tokenizer)
        return 0

    if args.command == "mcp":
        from .adapters.mcp_server import run_mcp

        run_mcp(store_path=args.db, transport=args.transport, tokenizer=args.tokenizer)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
