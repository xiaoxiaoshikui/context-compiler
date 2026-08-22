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
from .graph import build_dependency_graph
from .render import ContextRenderer
from .retrieval import Retriever, TfidfRetriever, augment_with_critical_items
from .scoring import ScoringWeights, activated_dependency_terms, score_item
from .store import ContextStore
from .tokenizer import HeuristicTokenCounter, TokenCounter


@dataclass(slots=True)
class CompilerConfig:
    max_candidates: int = 120
    # Upper bound on how many stored items the retriever ever ranks. Items
    # beyond this are invisible to a compile() call regardless of relevance --
    # generous enough to be "effectively everything" for a repository-sized
    # store; raise it if you have a genuinely huge context store.
    max_pool_size: int = 5000
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
    # A real task string (e.g. a GitHub issue with a pasted stack trace or
    # CLI log) can run to thousands of tokens. The header embeds it
    # verbatim and isn't subject to the L0-L4 compression items get, so
    # without a cap it can consume an entire small/medium budget on its
    # own, leaving nothing for actual context -- found via a real
    # SWE-bench instance (pylint-7080), never triggered by the short
    # one-line synthetic benchmark tasks. Cap it to a fraction of budget
    # so most of the budget is always reserved for the working set itself.
    max_header_fraction: float = 0.3
    # Divisor used to scale the default candidate-retrieval limit with
    # budget (see `compile()`). ~41-49 tokens is the measured cost of an
    # L0 (cheapest) rendering of a typical stored item, so this
    # approximates "how many of the cheapest possible items could this
    # budget even afford" as a floor on how deep retrieval should look.
    candidate_tokens_estimate: int = 40
    # How many of the highest-scoring (non-pinned, non-constraint)
    # candidates are forced to full text (L4) before the greedy allocator
    # spends anything on breadth. Found via a real SWE-bench instance: the
    # allocator will happily spend budget on many cheap L0/L1 pointers
    # (e.g. a dozen near-empty test-file stubs) while the one file that
    # actually needs editing sits at L2 -- a symbol list plus scattered
    # snippets, not the file's real current text. Asking a model to rewrite
    # a whole function from that L2 view (see benchmarks/real_eval.py's
    # patch mechanism) silently drops code it never saw, because it's
    # reconstructing from a lossy summary rather than editing real text.
    # Forcing full text on a small number of top candidates trades some of
    # that breadth for the precision an actual edit needs. 0 disables this
    # and restores pre-2026-08-22 behavior.
    #
    # Raised from 3 to 7 after checking, across 6 real SWE-bench instances,
    # where each one's *actual* fix file (from the gold patch, not a guess)
    # landed in the relevance ranking: 4 of 6 already ranked inside the top
    # 3 (no change from raising this), but astropy-12907's fix file ranked
    # 7th -- edged out by an unrelated top-3 slot -- and stayed invisible
    # (L0) at k=3, reaching full text only at k>=7. Confirmed zero change
    # in the synthetic benchmark's output (byte-identical report at k=3 vs
    # k=7) and the full unit suite still passes, so the extra breadth this
    # costs isn't visible at this benchmark's scale; it only showed up on
    # a real, large repository. One real instance (pylint-7080) saw its
    # already-unselected-in-any-useful-sense fix file (L0, i.e. a bare
    # pointer) get pushed out of selection entirely at k=7 -- a real
    # crowding cost, but not a change to whether it was actually usable.
    top_k_full_text: int = 7
    # Ceiling on how much any single top_k_full_text item may claim: the
    # larger of this fraction of the whole budget, or top_k_max_item_floor
    # tokens. Found via the same real instance: without a cap, a single
    # large file ranked in the top-K can consume the entire budget on its
    # own and crowd out everything else. A fraction alone breaks small
    # budgets, though -- confirmed on the synthetic benchmark: several
    # 150-250 token tasks regressed because 25% of a 150-token budget
    # (~37 tokens) is smaller than files that would otherwise fit at L4
    # outright, capping fidelity for no reason. The floor keeps the cap a
    # no-op until it's actually needed; the real remaining-budget check
    # right after this still degrades anything that doesn't truly fit.
    top_k_max_item_fraction: float = 0.25
    top_k_max_item_floor: int = 2000
    # How many graph-adjacent test files to force to at least L3 per top-K
    # item (see the graph_test_ids block in `compile`). 1 sounds like the
    # obvious choice ("the" test for this file) but is fragile in practice:
    # a widely-imported module has many importing tests, and the correct
    # one frequently loses a near-tie to an unrelated one on lexical score
    # alone -- confirmed on a real instance, 0.4396 vs 0.4433, an 0.8% gap
    # separating the right test from the wrong one. 2 catches that specific
    # failure mode cheaply; it does not help when the correct test is
    # buried much further down a long list of neighbors (confirmed on
    # another real instance with 17 candidates, correct one ranked 15th) --
    # that is a distinct, harder problem this knob is not meant to solve.
    graph_test_ids_per_item: int = 2


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
        retriever: Retriever | None = None,
        use_graph: bool = True,
    ) -> None:
        self.store = store
        self.counter = counter or HeuristicTokenCounter()
        self.config = config or CompilerConfig()
        self.weights = weights or ScoringWeights()
        self.retriever = retriever or TfidfRetriever()
        self.use_graph = use_graph
        self.renderer = ContextRenderer(self.counter)

    def _candidates(self, task: str, limit: int) -> list[ContextItem]:
        pool = self.store.list(limit=self.config.max_pool_size)
        ranked = self.retriever.rank(task, pool, limit=limit)
        ranked = self._augment_with_graph_adjacent_tests(ranked, pool)
        return augment_with_critical_items(ranked, pool)

    def _augment_with_graph_adjacent_tests(
        self, ranked: list[ContextItem], pool: list[ContextItem], *, top_n: int = 12
    ) -> list[ContextItem]:
        """Union in the test file one graph-hop from each top-ranked candidate,
        even when the lexical retriever ranked it too low (or not at all) to
        make the candidate cut on its own.

        A bug report's vocabulary rarely overlaps with its own regression
        test's vocabulary, so TF-IDF/FTS retrieval alone frequently drops the
        one test that would actually validate a fix from the candidate pool
        entirely -- confirmed on real SWE-bench instances (django, astropy):
        the test file scored far outside the retriever's top ~200 even
        though a real import edge connects it to the fix file one hop away.
        `compile`'s later graph_test_ids step can only force a candidate's
        *render fidelity* up; it can't rescue an item that was never a
        candidate in the first place, so that rescue has to happen here,
        before the retriever's cutoff is applied.
        """
        if not self.use_graph:
            return ranked
        ranked_ids = {i.id for i in ranked}
        graph = build_dependency_graph(self.store, limit=self.config.max_pool_size)
        pool_by_id = {i.id: i for i in pool}
        extra: list[ContextItem] = []
        seen: set[str] = set()
        for item in ranked[:top_n]:
            for neighbor_id in graph.related(item.id, depth=1):
                if neighbor_id in ranked_ids or neighbor_id in seen:
                    continue
                neighbor = pool_by_id.get(neighbor_id)
                if neighbor is not None and neighbor.kind is ContextKind.TEST:
                    extra.append(neighbor)
                    seen.add(neighbor_id)
        return ranked + extra

    def _graph_related_ids(
        self, first_pass: list[tuple[ContextItem, ScoreBreakdown]], *, top_n: int = 12
    ) -> set[str]:
        """Item ids within one resolved dependency-graph hop of a top-scoring
        candidate -- see graph.py. A real import edge is a stronger, more
        trustworthy relatedness signal than the term-overlap proxy in
        `activated_dependency_terms`, so a graph-adjacent item's dependency
        score is set to its ceiling in the second scoring pass (see
        `scoring.dependency_score`), rather than blended with it.
        """
        related, _ = self._graph_related_ids_by_item(first_pass, top_n=top_n)
        return related

    def _graph_related_ids_by_item(
        self, first_pass: list[tuple[ContextItem, ScoreBreakdown]], *, top_n: int = 12
    ) -> tuple[set[str], dict[str, set[str]]]:
        """Like `_graph_related_ids`, but also keyed per top-scoring item.

        The union alone is enough for the scoring boost (any graph adjacency
        to *something* highly-ranked is a real signal), but it's not enough
        to answer "what does *this specific* top-K candidate's test file
        neighbor to at least L3" -- picking the highest-scoring item out of
        the union can and does pick a test file adjacent to a *different*
        top-scoring item instead of the one actually being asked about.
        Confirmed on a real SWE-bench instance: the union's top-scoring test
        file scored higher than, and was picked over, the one actually
        importing the file that needed editing.
        """
        if not self.use_graph:
            return set(), {}
        graph = build_dependency_graph(self.store, limit=self.config.max_pool_size)
        top_ids = [
            item.id for item, _ in sorted(first_pass, key=lambda x: x[1].total, reverse=True)[:top_n]
        ]
        per_item: dict[str, set[str]] = {}
        related: set[str] = set()
        for item_id in top_ids:
            neighbors = graph.related(item_id, depth=1)
            per_item[item_id] = neighbors
            related |= neighbors
        return related, per_item

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
        audit: list[str] = []
        if max_candidates is not None:
            candidate_limit = max_candidates
        else:
            candidate_limit = max(
                self.config.max_candidates, budget // self.config.candidate_tokens_estimate
            )
        candidate_limit = min(candidate_limit, self.config.max_pool_size)
        reserve = self.config.reserve_tokens if reserve_tokens is None else reserve_tokens
        reserve = max(0, min(reserve, max(0, budget - 1)))

        header, header_tokens = self._bounded_task_header(task, budget, audit)
        available = max(0, budget - header_tokens - reserve)

        candidates = self._candidates(task, candidate_limit)
        first_pass = [(i, score_item(task, i, weights=self.weights)) for i in candidates]
        active = activated_dependency_terms(first_pass)
        graph_related_ids, graph_related_by_item = self._graph_related_ids_by_item(first_pass)
        scored = [
            (
                i,
                score_item(
                    task,
                    i,
                    weights=self.weights,
                    activated_terms=active,
                    graph_related=i.id in graph_related_ids,
                ),
            )
            for i in candidates
        ]
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

        # Force minimum representations for pinned items, high-risk
        # constraints, and the top-K highest-scoring candidates overall
        # (full text -- see CompilerConfig.top_k_full_text).
        top_k_ids: set[str] = {
            item.id
            for item, _ in sorted(scored, key=lambda x: -x[1].total)[: self.config.top_k_full_text]
        }
        # The test that actually validates a fix is frequently *not* lexically
        # close to the task text (a bug report rarely reuses its test's
        # vocabulary), so plain relevance ranking alone often leaves it far
        # outside top_k_full_text even when it's the single most useful thing
        # to show in full -- confirmed on real SWE-bench instances: 3 of 4
        # checked cases had their real fix-validating test ranked well outside
        # the top few candidates by score alone. But a real code-imports-test
        # edge from the dependency graph reliably exists in most of those
        # cases even when the score doesn't reflect it (graph_related_ids
        # already carries this signal into scoring -- see
        # `_graph_related_ids` -- just not always far enough to win a
        # ranking-only cutoff). Force the top `graph_test_ids_per_item`
        # highest-scoring graph-related test items *specific to each* top-K
        # candidate (not the highest-scoring among all top-K candidates'
        # neighbors combined -- see `_graph_related_ids_by_item`) to at
        # least L3 (a real excerpt, not just a pointer) so each gets a
        # chance to actually
        # inform the fix, regardless of how it scored on relevance alone.
        graph_test_ids: set[str] = set()
        for tid in top_k_ids:
            tid_neighbor_ids = graph_related_by_item.get(tid, set())
            test_neighbors = sorted(
                (
                    (item, b) for item, b in scored
                    if item.id in tid_neighbor_ids
                    and item.kind is ContextKind.TEST
                    and item.id not in top_k_ids
                ),
                key=lambda pair: -pair[1].total,
            )
            for item, _ in test_neighbors[: self.config.graph_test_ids_per_item]:
                graph_test_ids.add(item.id)
        forced = sorted(
            [
                pair
                for pair in scored
                if pair[0].pinned
                or pair[0].kind is ContextKind.CONSTRAINT
                or pair[0].id in top_k_ids
                or pair[0].id in graph_test_ids
            ],
            key=lambda x: (not x[0].pinned, x[0].id not in top_k_ids, x[0].id not in graph_test_ids, -x[1].total),
        )
        # Cap how much of the *whole* budget any single top-K item's forced
        # full text may claim. Without this, one large file among the top-K
        # (e.g. a 16k-token module against an 8k budget) silently consumes
        # the entire budget by itself, crowding out every other candidate --
        # including, concretely, the file a real SWE-bench instance actually
        # needed edited, which was ranked just below the top 3. Confirmed via
        # audit log: "forced ... L4 ... cost=7160" against an 8000 budget
        # left only two other items degraded to L3 and L0.
        top_k_item_cap = max(
            self.config.top_k_max_item_floor, int(budget * self.config.top_k_max_item_fraction)
        )
        for item, b in forced:
            if item.pinned:
                idx = self._best_idx_at_or_below(variants[item.id], self.config.pinned_min_level)
            elif item.kind is ContextKind.CONSTRAINT:
                idx = self._best_idx_at_or_below(variants[item.id], self.config.constraint_min_level)
            elif item.id in graph_test_ids:
                # L3 (a real excerpt), not L4 -- a test file can be huge
                # (hundreds of cases in one file is common), and the point
                # is giving the model a concrete look at real assertions,
                # not necessarily the entire file.
                cap_idx = self._largest_fitting_idx(variants[item.id], top_k_item_cap)
                l3_idx = self._best_idx_at_or_below(variants[item.id], RenderLevel.L3)
                idx = min(cap_idx, l3_idx) if cap_idx >= 0 and l3_idx >= 0 else max(cap_idx, l3_idx)
            else:
                idx = self._largest_fitting_idx(variants[item.id], top_k_item_cap)
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
        items = self._candidates(task, max(limit * 4, 40))
        first = [(i, score_item(task, i, weights=self.weights)) for i in items]
        active = activated_dependency_terms(first)
        graph_related_ids = self._graph_related_ids(first)
        rescored = [
            (
                i,
                score_item(
                    task,
                    i,
                    weights=self.weights,
                    activated_terms=active,
                    graph_related=i.id in graph_related_ids,
                ),
            )
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

    def _bounded_task_header(self, task: str, budget: int, audit: list[str]) -> tuple[str, int]:
        """Build the task header, capped to `max_header_fraction` of budget.

        The header embeds the task string verbatim -- it isn't one of the
        stored items, so it never goes through the L0-L4 compression those
        get. A verbose real task (a GitHub issue with a pasted stack trace,
        say) can otherwise cost more tokens than the whole budget, leaving
        nothing for the actual working set. Truncating the task text itself
        (via the same TokenCounter.truncate used for the final defensive
        hard-cap) keeps a hard floor of budget available for real context,
        and truncate() already appends a visible "…" marker so the
        omission isn't silent.
        """
        header = self._task_header(task)
        header_tokens = self.counter.count(header)
        max_header_tokens = max(1, int(budget * self.config.max_header_fraction))
        if header_tokens <= max_header_tokens:
            return header, header_tokens

        preamble_tokens = self.counter.count(self._task_header(""))
        task_budget = max(0, max_header_tokens - preamble_tokens)
        truncated_task = self.counter.truncate(task.strip(), task_budget)
        header = self._task_header(truncated_task)
        header_tokens = self.counter.count(header)
        audit.append(
            f"task text truncated to fit within {max_header_tokens} tokens "
            f"({self.config.max_header_fraction:.0%} of budget) -- "
            f"header alone would otherwise have cost {self.counter.count(self._task_header(task))}"
        )
        return header, header_tokens

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
