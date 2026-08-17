from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import (
    CompiledContext,
    ContextItem,
    ContextKind,
    RenderLevel,
    RenderedVariant,
    ScoreBreakdown,
    SelectedContext,
)
from .render import ContextRenderer
from .scoring import ScoringWeights, activated_dependency_terms, score_item
from .store import ContextStore
from .tokenizer import HeuristicTokenCounter, TokenCounter


@dataclass(slots=True)
class CompilerConfig:
    max_candidates: int = 120
    min_candidate_score: float = 0.055
    pinned_min_level: RenderLevel = RenderLevel.L2
    # L1 only ever renders a fixed lead line/outline regardless of the task, so an
    # ingested (non-pinned) constraint's forced-minimum guarantee can miss the
    # actual operative sentence if it isn't the file's first line (e.g. a markdown
    # doc with "# Title" followed by the real invariant in the next paragraph). L2
    # runs task-aware extractive summarization instead, which is far more likely to
    # surface the sentence that actually matters. Found via benchmarks/run.py.
    constraint_min_level: RenderLevel = RenderLevel.L2
    reserve_tokens: int = 0
    # Penalize tiny low-value pointers so the compiler does not fill the
    # context with a directory listing of everything it knows.
    admission_penalty: float = 0.018


_LEVEL_ORDER = [
    RenderLevel.L0,
    RenderLevel.L1,
    RenderLevel.L2,
    RenderLevel.L3,
    RenderLevel.L4,
]


class ContextCompiler:
    """Compile a task-specific working set under a hard token budget.

    The allocator models each context item as an ordered set of representation
    upgrades. Each upgrade has marginal utility and marginal token cost. It
    forces safety-critical/pinned minimums first, then greedily buys the highest
    utility-per-token upgrades that fit the remaining budget.
    """

    def __init__(
        self,
        store: ContextStore,
        *,
        counter: TokenCounter | None = None,
        config: CompilerConfig | None = None,
        weights: ScoringWeights | None = None,
    ) -> None:
        self.store = store
        self.counter = counter or HeuristicTokenCounter()
        self.config = config or CompilerConfig()
        self.weights = weights or ScoringWeights()
        self.renderer = ContextRenderer(self.counter)

    def compile(
        self,
        task: str,
        budget: int,
        *,
        max_candidates: int | None = None,
        reserve_tokens: int | None = None,
    ) -> CompiledContext:
        if budget <= 0:
            raise ValueError("budget must be > 0")
        candidate_limit = max_candidates or self.config.max_candidates
        reserve = self.config.reserve_tokens if reserve_tokens is None else reserve_tokens
        reserve = max(0, min(reserve, max(0, budget - 1)))

        header = self._task_header(task)
        header_tokens = self.counter.count(header)
        available = max(0, budget - header_tokens - reserve)

        candidates = self.store.candidate_search(task, limit=candidate_limit)
        first_pass = [(i, score_item(task, i, weights=self.weights)) for i in candidates]
        active = activated_dependency_terms(first_pass)
        scored = [(i, score_item(task, i, weights=self.weights, activated_terms=active)) for i in candidates]
        scored.sort(key=lambda x: x[1].total, reverse=True)
        scored = [
            pair
            for pair in scored
            if pair[1].total >= self.config.min_candidate_score or pair[0].pinned
        ]

        variants: dict[str, list[RenderedVariant]] = {
            item.id: self.renderer.variants(item, task) for item, _ in scored
        }
        breakdowns: dict[str, ScoreBreakdown] = {item.id: b for item, b in scored}
        items: dict[str, ContextItem] = {item.id: item for item, _ in scored}

        # selected index in each item's variant list; -1 means absent.
        selected_idx: dict[str, int] = {item.id: -1 for item, _ in scored}
        used = 0
        audit: list[str] = []

        # Force minimum representations for pinned items and high-risk constraints.
        forced = sorted(
            [pair for pair in scored if pair[0].pinned or pair[0].kind is ContextKind.CONSTRAINT],
            key=lambda x: (not x[0].pinned, -x[1].total),
        )
        for item, b in forced:
            min_level = self.config.pinned_min_level if item.pinned else self.config.constraint_min_level
            idx = self._best_idx_at_or_below(variants[item.id], min_level)
            if idx < 0:
                continue
            cost = variants[item.id][idx].token_count
            if used + cost <= available:
                selected_idx[item.id] = idx
                used += cost
                audit.append(
                    f"forced {item.id} {variants[item.id][idx].level.value}: "
                    f"pinned={item.pinned} kind={item.kind.value} score={b.total:.3f} cost={cost}"
                )
            else:
                # Degrade forced items to the largest representation that fits.
                fit = self._largest_fitting_idx(variants[item.id], available - used)
                if fit >= 0:
                    selected_idx[item.id] = fit
                    used += variants[item.id][fit].token_count
                    audit.append(
                        f"degraded forced {item.id} to {variants[item.id][fit].level.value} "
                        f"because budget was tight"
                    )

        # Iteratively buy the highest marginal utility / marginal token upgrade.
        while True:
            best: tuple[float, str, int, int, float] | None = None
            # (ratio, item_id, new_idx, delta_cost, delta_utility)
            for item, b in scored:
                current = selected_idx[item.id]
                new_idx = current + 1
                if new_idx >= len(variants[item.id]):
                    continue
                prev_cost = 0 if current < 0 else variants[item.id][current].token_count
                prev_fidelity = 0.0 if current < 0 else variants[item.id][current].fidelity
                new = variants[item.id][new_idx]
                delta_cost = max(1, new.token_count - prev_cost)
                if used + delta_cost > available:
                    continue
                delta_fidelity = max(0.001, new.fidelity - prev_fidelity)
                delta_utility = b.total * delta_fidelity
                if current < 0:
                    delta_utility -= self.config.admission_penalty
                # High omission risk increases the value of fidelity upgrades.
                delta_utility *= 1.0 + 0.30 * item.omission_risk
                ratio = delta_utility / delta_cost
                if delta_utility <= 0:
                    continue
                if best is None or ratio > best[0]:
                    best = (ratio, item.id, new_idx, delta_cost, delta_utility)
            if best is None:
                break
            _, item_id, new_idx, delta_cost, delta_utility = best
            selected_idx[item_id] = new_idx
            used += delta_cost
            audit.append(
                f"upgrade {item_id} -> {variants[item_id][new_idx].level.value}: "
                f"+{delta_cost} tokens, marginal_utility={delta_utility:.4f}"
            )

        selections: list[SelectedContext] = []
        for item, b in scored:
            idx = selected_idx[item.id]
            if idx < 0:
                continue
            variant = variants[item.id][idx]
            selections.append(
                SelectedContext(
                    item_id=item.id,
                    title=item.title,
                    kind=item.kind.value,
                    level=variant.level,
                    text=variant.text,
                    token_count=variant.token_count,
                    score=b.total,
                    utility=b.total * variant.fidelity,
                    source=item.source,
                    breakdown=b,
                )
            )

        # Put constraints/decisions first, then by utility. This makes the emitted
        # prompt stable and keeps high-risk invariants early in the context.
        order = {
            ContextKind.CONSTRAINT.value: 0,
            ContextKind.DECISION.value: 1,
            ContextKind.CONFIG.value: 2,
            ContextKind.TEST.value: 3,
            ContextKind.CODE.value: 4,
            ContextKind.DOC.value: 5,
            ContextKind.TOOL.value: 6,
            ContextKind.CONVERSATION.value: 7,
            ContextKind.NOTE.value: 8,
        }
        selections.sort(key=lambda s: (order.get(s.kind, 9), -s.utility, s.item_id))
        body = "\n\n".join(s.text for s in selections)
        text = header + ("\n\n" + body if body else "")
        used_tokens = self.counter.count(text)

        # Defensive hard cap. The allocator uses additive token costs but
        # tokenizers can behave non-additively at boundaries.
        if used_tokens > budget:
            text = self.counter.truncate(text, budget)
            used_tokens = self.counter.count(text)
            audit.append("final text was hard-truncated to enforce budget")

        selected_ids = {s.item_id for s in selections}
        omitted_ids = [item.id for item, _ in scored if item.id not in selected_ids]
        return CompiledContext(
            task=task,
            budget=budget,
            used_tokens=used_tokens,
            body_tokens=used,
            selections=selections,
            text=text,
            candidate_count=len(scored),
            omitted_ids=omitted_ids,
            audit=audit,
        )

    def search(self, task: str, *, limit: int = 10) -> list[dict[str, object]]:
        items = self.store.candidate_search(task, limit=max(limit * 4, 40))
        first = [(i, score_item(task, i, weights=self.weights)) for i in items]
        active = activated_dependency_terms(first)
        rescored = [
            (i, score_item(task, i, weights=self.weights, activated_terms=active))
            for i in items
        ]
        rescored.sort(key=lambda x: x[1].total, reverse=True)
        return [
            {
                "item": item.to_dict(),
                "score": breakdown.to_dict(),
            }
            for item, breakdown in rescored[:limit]
        ]

    def expand(self, item_id: str, *, level: RenderLevel | str = RenderLevel.L4, task: str = "") -> dict:
        item = self.store.get(item_id)
        if not item:
            raise KeyError(item_id)
        level = RenderLevel(level)
        variant = self.renderer.render(item, task, level)
        return {
            "item": item.to_dict(),
            "level": level.value,
            "token_count": variant.token_count,
            "text": variant.text,
        }

    @staticmethod
    def _task_header(task: str) -> str:
        return (
            "[CONTEXT-COMPILER]\n"
            "Use the following task-specific working set. Context may be summarized; "
            "items can be expanded by CTX id if your host exposes the expansion tool.\n"
            f"TASK: {task.strip()}"
        )

    @staticmethod
    def _best_idx_at_or_below(variants: list[RenderedVariant], level: RenderLevel) -> int:
        target = _LEVEL_ORDER.index(level)
        idx = -1
        for i, v in enumerate(variants):
            if _LEVEL_ORDER.index(v.level) <= target:
                idx = i
        return idx

    @staticmethod
    def _largest_fitting_idx(variants: Iterable[RenderedVariant], remaining: int) -> int:
        idx = -1
        for i, v in enumerate(variants):
            if v.token_count <= remaining:
                idx = i
        return idx
