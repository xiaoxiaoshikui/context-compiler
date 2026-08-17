"""Naive baselines for measuring the value of the scored/allocated compiler.

Each baseline exposes the same ``compile(task, budget) -> CompiledContext``
shape as :class:`context_compiler.compiler.ContextCompiler`, so an
:class:`~context_compiler.experiments.ExperimentRunner` can drive any of
them interchangeably. These are intentionally dumb: they exist to give the
budget-sweep / B95 experiments in ``benchmarks/`` a "what if we didn't
rank/allocate at all" reference point (see README.md, "Recommended next
research iterations", item 2).
"""

from __future__ import annotations

import random

from .models import CompiledContext, RenderLevel, ScoreBreakdown, SelectedContext
from .render import ContextRenderer
from .store import ContextStore
from .tokenizer import HeuristicTokenCounter, TokenCounter

_ZERO_BREAKDOWN = ScoreBreakdown(
    relevance=0.0, importance=0.0, risk=0.0, recency=0.0,
    dependency=0.0, kind_prior=0.0, pin_bonus=0.0, total=0.0,
)


def _header(tag: str, task: str) -> str:
    return f"[CONTEXT-COMPILER:{tag}]\nTASK: {task.strip()}"


class FullContextBaseline:
    """"Just paste everything" baseline.

    Concatenates every stored item at full fidelity (L4) in deterministic
    path order (like ``cat $(find . -type f | sort)``), stopping at the
    first item that would not fit. No relevance ranking, no resolution
    selection, no protection for pinned/constraint items.
    """

    name = "full"

    def __init__(self, store: ContextStore, *, counter: TokenCounter | None = None) -> None:
        self.store = store
        self.counter = counter or HeuristicTokenCounter()
        self.renderer = ContextRenderer(self.counter)

    def compile(self, task: str, budget: int) -> CompiledContext:
        header = _header("FULL", task)
        used = self.counter.count(header)
        items = sorted(self.store.list(limit=1_000_000), key=lambda i: i.source or i.title)
        selections: list[SelectedContext] = []
        for item in items:
            variant = self.renderer.render(item, task, RenderLevel.L4)
            if used + variant.token_count > budget:
                break
            used += variant.token_count
            selections.append(_as_selection(item, variant))
        return _finish(task, budget, header, used, selections, items, self.counter,
                        audit=["full-context baseline: path-sorted, no ranking, stop at first overflow"])


class RandomContextBaseline:
    """Randomly shuffled selection baseline.

    Shuffles all stored items and greedily takes full-fidelity (L4)
    representations that still fit the budget, skipping over items that
    are individually too large. This models "grab a random subset that
    fits" rather than any relevance- or risk-aware policy.
    """

    name = "random"

    def __init__(
        self,
        store: ContextStore,
        *,
        counter: TokenCounter | None = None,
        seed: int | None = None,
    ) -> None:
        self.store = store
        self.counter = counter or HeuristicTokenCounter()
        self.renderer = ContextRenderer(self.counter)
        self._rng = random.Random(seed)

    def compile(self, task: str, budget: int) -> CompiledContext:
        header = _header("RANDOM", task)
        used = self.counter.count(header)
        items = list(self.store.list(limit=1_000_000))
        self._rng.shuffle(items)
        selections: list[SelectedContext] = []
        for item in items:
            variant = self.renderer.render(item, task, RenderLevel.L4)
            if used + variant.token_count > budget:
                continue
            used += variant.token_count
            selections.append(_as_selection(item, variant))
        return _finish(task, budget, header, used, selections, items, self.counter,
                        audit=["random baseline: shuffled item order, greedy fit"])


def _as_selection(item, variant) -> SelectedContext:
    return SelectedContext(
        item_id=item.id,
        title=item.title,
        kind=item.kind.value,
        level=variant.level,
        text=variant.text,
        token_count=variant.token_count,
        score=0.0,
        utility=0.0,
        source=item.source,
        breakdown=_ZERO_BREAKDOWN,
    )


def _finish(task, budget, header, used, selections, items, counter, *, audit) -> CompiledContext:
    body = "\n\n".join(s.text for s in selections)
    text = header + ("\n\n" + body if body else "")
    used_tokens = counter.count(text)
    if used_tokens > budget:
        text = counter.truncate(text, budget)
        used_tokens = counter.count(text)
        audit = audit + ["final text was hard-truncated to enforce budget"]
    selected_ids = {s.item_id for s in selections}
    return CompiledContext(
        task=task,
        budget=budget,
        used_tokens=used_tokens,
        body_tokens=used,
        selections=selections,
        text=text,
        candidate_count=len(items),
        omitted_ids=[i.id for i in items if i.id not in selected_ids],
        audit=audit,
    )
