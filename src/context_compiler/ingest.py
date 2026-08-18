from __future__ import annotations

import ast
import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .models import ContextKind
from .store import ContextStore


DEFAULT_IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    "target",
    "__pycache__",
    ".next",
    ".cache",
}

TEXT_EXTENSIONS = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".kts",
    ".go", ".rs", ".rb", ".php", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs",
    ".swift", ".scala", ".sh", ".bash", ".zsh", ".fish", ".sql", ".graphql",
    ".md", ".mdx", ".rst", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".conf", ".xml", ".html", ".css", ".scss", ".less", ".vue",
    ".svelte", ".env", ".properties", ".dockerfile", ".tf", ".hcl",
}


@dataclass(slots=True)
class IngestReport:
    root: str
    added: int = 0
    updated: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    item_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "added": self.added,
            "updated": self.updated,
            "skipped": self.skipped,
            "errors": self.errors,
            "item_ids": self.item_ids,
        }


class RepositoryIngestor:
    def __init__(
        self,
        store: ContextStore,
        *,
        max_file_bytes: int = 1_500_000,
        ignore_dirs: set[str] | None = None,
        ignore_globs: list[str] | None = None,
    ) -> None:
        self.store = store
        self.max_file_bytes = max_file_bytes
        self.ignore_dirs = ignore_dirs or DEFAULT_IGNORE_DIRS
        self.ignore_globs = ignore_globs or []

    def ingest(self, path: str | Path) -> IngestReport:
        root = Path(path).expanduser().resolve()
        report = IngestReport(root=str(root))
        if root.is_file():
            self._ingest_file(root, root.parent, report)
            return report
        if not root.exists():
            raise FileNotFoundError(root)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in self.ignore_dirs]
            current = Path(dirpath)
            for filename in filenames:
                file_path = current / filename
                rel = file_path.relative_to(root).as_posix()
                if any(fnmatch.fnmatch(rel, pat) for pat in self.ignore_globs):
                    report.skipped += 1
                    continue
                self._ingest_file(file_path, root, report)
        return report

    def _ingest_file(self, path: Path, root: Path, report: IngestReport) -> None:
        try:
            if path.stat().st_size > self.max_file_bytes:
                report.skipped += 1
                return
            if not self._looks_textual(path):
                report.skipped += 1
                return
            raw = path.read_bytes()
            if b"\x00" in raw[:4096]:
                report.skipped += 1
                return
            content = raw.decode("utf-8", errors="replace")
            source = path.relative_to(root).as_posix() if path != root else path.name
            previous = self.store.get_by_source(source)
            kind = classify_kind(source)
            deps = extract_dependencies(path, content)
            tags = tags_for(source, kind)
            importance, risk, pinned = default_prior(source, kind, content)
            metadata = {
                "bytes": len(raw),
                "extension": path.suffix.lower(),
                "ingested_from": str(root),
            }
            if previous:
                item = previous
                item.title = source
                item.content = content
                item.kind = kind
                item.tags = tuple(tags)
                item.dependencies = tuple(deps)
                item.importance = max(previous.importance, importance)
                item.omission_risk = max(previous.omission_risk, risk)
                item.pinned = previous.pinned or pinned
                item.metadata = {**previous.metadata, **metadata}
                self.store.upsert(item)
                report.updated += 1
                report.item_ids.append(item.id)
            else:
                item = self.store.add(
                    title=source,
                    content=content,
                    kind=kind,
                    source=source,
                    tags=tags,
                    dependencies=deps,
                    importance=importance,
                    omission_risk=risk,
                    pinned=pinned,
                    metadata=metadata,
                )
                report.added += 1
                report.item_ids.append(item.id)
        except (OSError, UnicodeError, SyntaxError) as exc:
            report.errors.append(f"{path}: {exc}")

    @staticmethod
    def _looks_textual(path: Path) -> bool:
        name = path.name.lower()
        if name in {"dockerfile", "makefile", "license", "readme", "procfile"}:
            return True
        if name.startswith(".env"):
            return True
        return path.suffix.lower() in TEXT_EXTENSIONS


def classify_kind(source: str) -> ContextKind:
    low = source.lower()
    name = Path(low).name
    suffix = Path(low).suffix
    if "test" in name or "/tests/" in "/" + low:
        return ContextKind.TEST
    if suffix in {".md", ".mdx", ".rst", ".txt"} or name.startswith("readme"):
        if any(x in low for x in ("constraint", "policy", "guardrail", "safety")):
            return ContextKind.CONSTRAINT
        if any(x in low for x in ("decision", "adr", "architecture-decision")):
            return ContextKind.DECISION
        return ContextKind.DOC
    if suffix in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env", ".tf", ".hcl"}:
        return ContextKind.CONFIG
    return ContextKind.CODE


def tags_for(source: str, kind: ContextKind) -> list[str]:
    parts = [p for p in source.replace("\\", "/").split("/") if p]
    tags = {kind.value, Path(source).suffix.lower().lstrip(".")}
    tags.update(p.lower() for p in parts[-4:])
    return sorted(t for t in tags if t)


def default_prior(source: str, kind: ContextKind, content: str) -> tuple[float, float, bool]:
    low = (source + "\n" + content[:5000]).lower()
    importance = {
        ContextKind.CODE: 0.58,
        ContextKind.TEST: 0.60,
        ContextKind.DOC: 0.46,
        ContextKind.DECISION: 0.78,
        ContextKind.CONFIG: 0.55,
    }.get(kind, 0.50)
    risk = 0.20
    pinned = False
    if any(k in low for k in ("security", "payment", "billing", "production", "migration", "auth", "oauth")):
        risk += 0.18
    if any(k in source.lower() for k in ("policy", "constraints", "guardrail", "safety")):
        risk = max(risk, 0.75)
        importance = max(importance, 0.75)
    if Path(source).name.lower().startswith("readme"):
        importance = max(importance, 0.62)
    return min(1.0, importance), min(1.0, risk), pinned


def extract_dependencies(path: Path, content: str) -> list[str]:
    suffix = path.suffix.lower()
    deps: set[str] = set()
    if suffix == ".py":
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    deps.update(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    dots = "." * node.level
                    if node.module:
                        # e.g. "from .gateway_client import X" -> ".gateway_client";
                        # dots encode how many directories up to resolve from
                        # (see graph.py) without changing the stored string schema.
                        deps.add(f"{dots}{node.module}")
                    elif dots:
                        # "from . import sibling_a, sibling_b" has no module --
                        # each imported name is itself a resolvable relative module.
                        deps.update(f"{dots}{alias.name}" for alias in node.names)
        except SyntaxError:
            pass
    elif suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        patterns = [
            r"\bfrom\s+['\"]([^'\"]+)['\"]",
            r"\brequire\(\s*['\"]([^'\"]+)['\"]\s*\)",
            r"\bimport\(\s*['\"]([^'\"]+)['\"]\s*\)",
        ]
        for pat in patterns:
            deps.update(re.findall(pat, content))
    elif suffix == ".go":
        deps.update(re.findall(r'"([^"\n]+)"', content[:20_000]))
    return sorted(deps)[:100]
