from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from .models import ContextItem, ContextKind, RenderLevel, RenderedVariant
from .scoring import term_set
from .tokenizer import TokenCounter


FIDELITY: dict[RenderLevel, float] = {
    RenderLevel.L0: 0.12,
    RenderLevel.L1: 0.27,
    RenderLevel.L2: 0.52,
    RenderLevel.L3: 0.82,
    RenderLevel.L4: 1.00,
}


@dataclass(slots=True)
class RendererConfig:
    l2_max_tokens: int = 320
    l3_max_tokens: int = 1400
    excerpt_window_lines: int = 3
    max_excerpt_blocks: int = 8


class ContextRenderer:
    def __init__(self, counter: TokenCounter, config: RendererConfig | None = None) -> None:
        self.counter = counter
        self.config = config or RendererConfig()

    def render(self, item: ContextItem, task: str, level: RenderLevel) -> RenderedVariant:
        if level is RenderLevel.L0:
            text = self._l0(item)
        elif level is RenderLevel.L1:
            text = self._l1(item)
        elif level is RenderLevel.L2:
            text = self._l2(item, task)
        elif level is RenderLevel.L3:
            text = self._l3(item, task)
        else:
            text = self._l4(item)
        return RenderedVariant(
            item_id=item.id,
            level=level,
            text=text,
            token_count=self.counter.count(text),
            fidelity=FIDELITY[level],
        )

    def variants(self, item: ContextItem, task: str) -> list[RenderedVariant]:
        levels = [RenderLevel.L0, RenderLevel.L1, RenderLevel.L2, RenderLevel.L3, RenderLevel.L4]
        variants = [self.render(item, task, lvl) for lvl in levels]
        # Ensure costs are monotonically non-decreasing. If two representations
        # collapse to the same text size, keep the higher-fidelity one only.
        out: list[RenderedVariant] = []
        max_cost = -1
        for variant in variants:
            if variant.token_count > max_cost:
                out.append(variant)
                max_cost = variant.token_count
            elif out:
                out[-1] = variant
                max_cost = max(max_cost, variant.token_count)
        return out

    def _header(self, item: ContextItem, level: str) -> str:
        source = f" source={item.source}" if item.source else ""
        return f"[CTX:{item.id} {level} kind={item.kind.value}{source}]\n"

    def _l0(self, item: ContextItem) -> str:
        return self._header(item, "L0") + item.title

    def _l1(self, item: ContextItem) -> str:
        bits = [self._header(item, "L1") + item.title]
        if item.tags:
            bits.append("tags: " + ", ".join(item.tags[:12]))
        if item.dependencies:
            bits.append("dependencies: " + ", ".join(item.dependencies[:12]))
        if item.kind in {ContextKind.CODE, ContextKind.TEST}:
            outline = self._code_outline(item)
            if outline:
                bits.append("outline:\n" + outline)
        elif item.content:
            first = self._first_meaningful_line(item.content)
            if first:
                bits.append("lead: " + first[:280])
        return "\n".join(bits)

    def _l2(self, item: ContextItem, task: str) -> str:
        base = [self._header(item, "L2") + item.title]
        if item.kind in {ContextKind.CODE, ContextKind.TEST}:
            outline = self._code_outline(item)
            if outline:
                base.append("symbols:\n" + outline)
            excerpts = self._relevant_excerpts(item.content, task, window=1, max_blocks=3)
            if excerpts:
                base.append("task-relevant snippets:\n" + excerpts)
        else:
            summary = self._extractive_summary(item.content, task, max_sentences=6)
            if summary:
                base.append(summary)
        return self.counter.truncate("\n".join(base), self.config.l2_max_tokens)

    def _l3(self, item: ContextItem, task: str) -> str:
        base = [self._header(item, "L3") + item.title]
        if item.dependencies:
            base.append("dependencies: " + ", ".join(item.dependencies[:20]))
        excerpt = self._relevant_excerpts(
            item.content,
            task,
            window=self.config.excerpt_window_lines,
            max_blocks=self.config.max_excerpt_blocks,
        )
        if not excerpt:
            excerpt = item.content
        base.append(excerpt)
        return self.counter.truncate("\n".join(base), self.config.l3_max_tokens)

    def _l4(self, item: ContextItem) -> str:
        return self._header(item, "L4") + item.title + "\n" + item.content

    @staticmethod
    def _first_meaningful_line(text: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return ""

    def _code_outline(self, item: ContextItem) -> str:
        source = item.source.lower()
        if source.endswith(".py"):
            try:
                tree = ast.parse(item.content)
            except SyntaxError:
                tree = None
            if tree:
                rows: list[str] = []
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        args = [a.arg for a in node.args.args]
                        rows.append(f"def {node.name}({', '.join(args)}) @L{node.lineno}")
                    elif isinstance(node, ast.ClassDef):
                        rows.append(f"class {node.name} @L{node.lineno}")
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        mod = getattr(node, "module", None) or ",".join(
                            a.name for a in getattr(node, "names", [])
                        )
                        rows.append(f"import {mod} @L{node.lineno}")
                return "\n".join(rows[:40])

        pattern = re.compile(
            r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|interface|type|def)\s+([A-Za-z_][A-Za-z0-9_]*)",
            re.MULTILINE,
        )
        rows = []
        lines = item.content.splitlines()
        for idx, line in enumerate(lines, 1):
            m = pattern.match(line)
            if m:
                rows.append(f"{m.group(1)} @L{idx}")
        return "\n".join(rows[:40])

    def _extractive_summary(self, text: str, task: str, max_sentences: int) -> str:
        if not text.strip():
            return ""
        q = term_set(task)
        sentences = re.split(r"(?<=[.!?。！？])\s+|\n{2,}", text)
        ranked: list[tuple[float, int, str]] = []
        for idx, sentence in enumerate(sentences):
            s = sentence.strip()
            if not s:
                continue
            st = term_set(s)
            overlap = len(q & st) / max(1, len(q))
            lead_bonus = max(0.0, 0.15 - idx * 0.005)
            score = overlap + lead_bonus
            ranked.append((score, idx, s))
        chosen = sorted(sorted(ranked, reverse=True)[:max_sentences], key=lambda x: x[1])
        return "\n".join(s for _, _, s in chosen)

    def _relevant_excerpts(self, text: str, task: str, *, window: int, max_blocks: int) -> str:
        lines = text.splitlines()
        if not lines:
            return ""
        q = term_set(task)
        if not q:
            return "\n".join(f"{i+1:>5}: {line}" for i, line in enumerate(lines[:80]))

        scored: list[tuple[float, int]] = []
        for idx, line in enumerate(lines):
            lt = term_set(line)
            if not lt:
                continue
            coverage = len(q & lt) / max(1, len(q))
            exact = 0.25 if task.strip().lower() in line.lower() else 0.0
            symbol_bonus = 0.08 if re.search(r"\b(class|def|function|error|exception|TODO|FIXME)\b", line, re.I) else 0.0
            score = coverage + exact + symbol_bonus
            if score > 0:
                scored.append((score, idx))
        if not scored:
            return ""

        centers = [idx for _, idx in sorted(scored, reverse=True)[:max_blocks]]
        ranges: list[tuple[int, int]] = []
        for c in sorted(centers):
            start, end = max(0, c - window), min(len(lines), c + window + 1)
            if ranges and start <= ranges[-1][1] + 1:
                ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
            else:
                ranges.append((start, end))

        blocks = []
        for start, end in ranges:
            body = "\n".join(f"{i+1:>5}: {lines[i]}" for i in range(start, end))
            blocks.append(body)
        return "\n...\n".join(blocks)
