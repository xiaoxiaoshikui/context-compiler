from __future__ import annotations

import argparse
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ..compiler import ContextCompiler
from ..models import ContextKind, RenderLevel
from ..store import ContextStore
from ..tokenizer import make_token_counter


def _json_bytes(data: object) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


class ContextAPIHandler(BaseHTTPRequestHandler):
    store: ContextStore
    compiler: ContextCompiler

    server_version = "ContextCompilerHTTP/0.1"

    def _send(self, status: int, data: object) -> None:
        payload = _json_bytes(data)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/health":
                self._send(200, {"ok": True})
                return
            if parsed.path == "/stats":
                self._send(200, self.store.stats())
                return
            if parsed.path == "/items":
                limit = int(qs.get("limit", ["100"])[0])
                kind = qs.get("kind", [None])[0]
                self._send(200, [i.to_dict() for i in self.store.list(limit=limit, kind=kind)])
                return
            if parsed.path.startswith("/items/"):
                item_id = parsed.path.split("/", 2)[2]
                item = self.store.get(item_id)
                if not item:
                    self._send(404, {"error": "not_found", "item_id": item_id})
                    return
                self._send(200, item.to_dict())
                return
            if parsed.path.startswith("/expand/"):
                item_id = parsed.path.split("/", 2)[2]
                level = qs.get("level", ["L4"])[0]
                task = qs.get("task", [""])[0]
                self._send(200, self.compiler.expand(item_id, level=level, task=task))
                return
            self._send(404, {"error": "not_found"})
        except (ValueError, KeyError) as exc:
            self._send(400, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._body()
            if self.path == "/items":
                item = self.store.add(
                    title=body["title"],
                    content=body.get("content", ""),
                    kind=body.get("kind", ContextKind.NOTE.value),
                    source=body.get("source", ""),
                    tags=body.get("tags", []),
                    dependencies=body.get("dependencies", []),
                    importance=float(body.get("importance", 0.5)),
                    omission_risk=float(body.get("omission_risk", 0.2)),
                    pinned=bool(body.get("pinned", False)),
                    metadata=body.get("metadata", {}),
                )
                self._send(201, item.to_dict())
                return
            if self.path == "/compile":
                result = self.compiler.compile(
                    body["task"],
                    int(body["budget"]),
                    max_candidates=int(body.get("max_candidates", 120)),
                    reserve_tokens=int(body.get("reserve_tokens", 0)),
                )
                self._send(200, result.to_dict())
                return
            if self.path == "/search":
                self._send(
                    200,
                    self.compiler.search(body["task"], limit=int(body.get("limit", 10))),
                )
                return
            self._send(404, {"error": "not_found"})
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})

    def log_message(self, fmt: str, *args: object) -> None:
        # Keep the default concise but include client information.
        super().log_message(fmt, *args)


def run_http(*, store_path: str, host: str = "127.0.0.1", port: int = 8765, tokenizer: str = "heuristic") -> None:
    store = ContextStore(store_path)
    compiler = ContextCompiler(store, counter=make_token_counter(tokenizer))
    handler = type("BoundContextAPIHandler", (ContextAPIHandler,), {"store": store, "compiler": compiler})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Context Compiler HTTP API listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=".context-compiler.db")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--tokenizer", choices=["heuristic", "tiktoken"], default="heuristic")
    args = p.parse_args(argv)
    run_http(store_path=args.db, host=args.host, port=args.port, tokenizer=args.tokenizer)


if __name__ == "__main__":
    main()
