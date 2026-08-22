import tempfile
import unittest
from pathlib import Path

from context_compiler import ContextCompiler, ContextStore
from context_compiler.compiler import CompilerConfig
from context_compiler.models import RenderLevel
from context_compiler.scoring import LEARNED_WEIGHTS_V1, ScoringWeights


class CompilerTests(unittest.TestCase):
    def test_budget_and_pinned_constraint(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            constraint = store.add(
                title="OAuth replay invariant",
                content=(
                    "OAuth authorization codes are single use. Never automatically replay "
                    "the exchange after an ambiguous failure."
                ),
                kind="constraint",
                pinned=True,
                importance=1.0,
                omission_risk=1.0,
            )
            store.add(
                title="auth.py",
                source="auth.py",
                kind="code",
                content="def callback(browser, code):\n    return exchange(code)\n",
                tags=["oauth", "safari"],
                importance=0.7,
            )
            for i in range(20):
                store.add(
                    title=f"noise-{i}.txt",
                    content="unrelated rendering cache image css analytics text " * 20,
                    kind="doc",
                    importance=0.1,
                )

            compiler = ContextCompiler(store)
            result = compiler.compile("Fix Safari OAuth callback", budget=260)
            self.assertLessEqual(result.used_tokens, 260)
            self.assertIn(constraint.id, {s.item_id for s in result.selections})
            self.assertIn("OAuth", result.text)

    def test_learned_weights_v1_is_a_usable_alternative_preset(self):
        # Not the default -- must be opted into explicitly. See
        # benchmarks/README.md "Phase 2" for how it was fit and its caveats.
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            constraint = store.add(
                title="OAuth replay invariant",
                content="Never automatically replay the exchange after an ambiguous failure.",
                kind="constraint",
                pinned=True,
                importance=1.0,
                omission_risk=1.0,
            )
            compiler = ContextCompiler(store, weights=LEARNED_WEIGHTS_V1)
            result = compiler.compile("Fix Safari OAuth callback", budget=260)
            self.assertIn(constraint.id, {s.item_id for s in result.selections})
            # recency/pin_bonus were never testable by the fitting benchmark
            # (see scoring.py) and must be untouched from the hand-tuned default.
            default = ScoringWeights()
            self.assertEqual(LEARNED_WEIGHTS_V1.recency, default.recency)
            self.assertEqual(LEARNED_WEIGHTS_V1.pin_bonus, default.pin_bonus)

    def test_compile_is_deterministic_across_separate_ingests(self):
        # Regression test for a real bug found via benchmarks/run.py: a
        # budget-boundary compile() result flipped between success and
        # failure across repeated runs against *identical* repo content,
        # traced to two compounding nondeterminism sources now fixed --
        # ContextStore.list()'s ORDER BY lacked a stable tiebreak for items
        # sharing the same `updated_at`, and randomly-generated item ids
        # (embedded in every rendered [CTX:...] header) tokenized as either
        # one or two tokens depending on whether the random hex happened to
        # start with a digit, shifting reported token counts by +-1 at
        # random. Re-ingesting the same content into a fresh store must now
        # produce byte-identical compile() output at a tight, boundary-prone
        # budget.
        # Note: raw output *text* legitimately differs run to run -- it embeds
        # each item's freshly-generated random id -- so compare used_tokens
        # and the (source, level) selection set instead, which must not.
        def compile_once() -> tuple[int, tuple[tuple[str, str], ...]]:
            with tempfile.TemporaryDirectory() as tmp:
                store = ContextStore(Path(tmp) / "ctx.db")
                store.add(
                    title="POLICY.md",
                    source="POLICY.md",
                    kind="constraint",
                    content=(
                        "Payment retry policy.\n\n"
                        "Never retry a charge without a confirmed idempotency key; "
                        "a blind retry after a gateway timeout can double-bill the customer."
                    ),
                )
                store.add(title="charge.py", source="charge.py", kind="code", content="def charge(): ...")
                store.add(title="gateway.py", source="gateway.py", kind="code", content="def call(): ...")
                store.add(title="README.md", source="README.md", kind="doc", content="Payments service.")
                compiler = ContextCompiler(store)
                result = compiler.compile("Fix duplicate charges on gateway timeout", budget=150)
                selection_shape = tuple(sorted((s.source, s.level.value) for s in result.selections))
                return result.used_tokens, selection_shape

        outputs = {compile_once() for _ in range(15)}
        self.assertEqual(len(outputs), 1, f"compile() shape varied across ingests: {outputs}")

    def test_verbose_task_text_does_not_starve_the_working_set(self):
        # Regression test for a real bug found by running a real SWE-bench
        # Verified instance (pylint-dev/pylint-7080) through the pipeline:
        # a genuine GitHub issue can paste a long CLI/log dump into its body
        # (thousands of tokens). The header embeds the task verbatim and
        # isn't compressed the way stored items are, so at a small-to-medium
        # budget the header alone consumed the *entire* budget, leaving zero
        # tokens -- and zero selections -- for the actual working set. The
        # 15 short, one-line synthetic benchmark tasks never exercised this
        # path. compile() must now always reserve most of the budget for
        # real context regardless of how verbose the task text is.
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            store.add(
                title="cache.py",
                source="cache.py",
                kind="code",
                content="TTL_SECONDS = 30\ndef get(key): ...",
                importance=0.8,
            )
            huge_task = "Investigate stale cached prices.\n" + (
                "irrelevant pasted log line with unique noise words " * 400
            )
            compiler = ContextCompiler(store)
            result = compiler.compile(huge_task, budget=500)
            self.assertLessEqual(result.used_tokens, 500)
            self.assertGreater(
                len(result.selections), 0, "verbose task text left no room for any context item"
            )
            self.assertTrue(
                any("truncated" in a for a in result.audit),
                "expected an audit entry noting the task text was truncated",
            )

    def test_default_candidate_limit_scales_with_budget(self):
        # Regression test for a second bug found via the same real SWE-bench
        # instance: retrieval hard-truncated to the top `max_candidates`
        # (120) *before* scoring, independent of how large the budget was.
        # A real fix file that TF-IDF ranked 181st out of ~3000 files was
        # therefore permanently unreachable at *any* budget, not just small
        # ones. The default candidate limit must now grow with budget so a
        # generous budget gets a deeper look, while small-budget behavior
        # (where 120 candidates was already more than could ever be
        # afforded) stays exactly as before.
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            task = "investigate the timeout retry bug in the payment gateway"
            for i in range(130):
                store.add(
                    title=f"filler-{i:03d}.py",
                    source=f"aaa_filler_{i:03d}.py",
                    kind="code",
                    content="timeout retry bug payment gateway " * 10,
                    importance=0.5,
                )
            target = store.add(
                title="unrelated-but-important.py",
                source="zzz_target.py",
                kind="code",
                content="widget rendering color palette layout spacing",
                importance=0.9,
                omission_risk=0.6,
            )
            compiler = ContextCompiler(store)

            small = compiler.compile(task, budget=2000)  # 2000 // 40 == 50 < 120, unchanged floor
            small_considered = {s.item_id for s in small.selections} | set(small.omitted_ids)
            self.assertNotIn(
                target.id,
                small_considered,
                "target should be past the default candidate cutoff at a small budget",
            )

            large = compiler.compile(task, budget=6000)  # 6000 // 40 == 150 > 130
            large_considered = {s.item_id for s in large.selections} | set(large.omitted_ids)
            self.assertIn(
                target.id,
                large_considered,
                "a larger budget should widen the candidate pool enough to reach the target",
            )

    def test_top_scoring_candidate_gets_full_text_not_just_a_summary(self):
        # Regression test for a real finding from benchmarks/real_eval.py:
        # given a real SWE-bench task, the compiler correctly selected the
        # file the fix actually belonged in, but the greedy allocator spent
        # the budget on breadth -- many cheap L0 pointers to barely-related
        # files -- leaving the one file that needed editing at L2 (a symbol
        # list plus scattered snippets, not its real current text). A model
        # asked to rewrite a whole function from that L2 view silently
        # dropped code it never saw. The fix: force the top-K highest
        # -scoring candidates to full text (L4) before the allocator is
        # free to spend the rest of the budget on breadth.
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            task = "fix the blueprint name validation bug"
            target = store.add(
                title="blueprints.py",
                source="blueprints.py",
                kind="code",
                content=(
                    "class Blueprint:\n"
                    "    def __init__(self, name, import_name):\n"
                    "        super().__init__(import_name)\n"
                    "        if \".\" in name:\n"
                    "            raise ValueError(\"name may not contain a dot\")\n"
                    "        self.name = name\n"
                    "        self._blueprints = []\n"
                ),
                importance=0.9,
            )
            # A pile of cheap, barely-relevant filler the allocator could
            # otherwise spend the whole budget spreading across as L0/L1
            # pointers instead of upgrading the one file that matters.
            for i in range(30):
                store.add(
                    title=f"unrelated-{i}.py",
                    source=f"unrelated-{i}.py",
                    kind="code",
                    content="blueprint name validation " * 5,
                    importance=0.3,
                )
            compiler = ContextCompiler(store)
            result = compiler.compile(task, budget=600)
            target_selection = next(s for s in result.selections if s.item_id == target.id)
            self.assertEqual(
                target_selection.level,
                RenderLevel.L4,
                "the top-scoring candidate should be forced to full text, not left at a summary level",
            )
            self.assertIn("self._blueprints = []", target_selection.text)

        # top_k_full_text=0 restores the old behavior, for comparison/rollback.
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            target = store.add(
                title="blueprints.py", source="blueprints.py", kind="code",
                content=("class Blueprint:\n" + "    # padding line\n" * 40 +
                          "    def __init__(self, name):\n        self.name = name\n"),
                importance=0.9,
            )
            for i in range(30):
                store.add(title=f"unrelated-{i}.py", source=f"unrelated-{i}.py", kind="code",
                           content="blueprint name validation " * 5, importance=0.3)
            compiler = ContextCompiler(store, config=CompilerConfig(top_k_full_text=0))
            result = compiler.compile("fix the blueprint name validation bug", budget=600)
            target_selection = next(s for s in result.selections if s.item_id == target.id)
            self.assertNotEqual(
                target_selection.level,
                RenderLevel.L4,
                "top_k_full_text=0 should restore the pre-fix behavior on this same setup",
            )

    def test_top_k_full_text_does_not_let_one_huge_file_starve_everything_else(self):
        # Regression test for a real second-order bug the previous fix
        # introduced: forcing top-K candidates to full text unconditionally
        # means a single *huge* top-ranked file can claim the entire budget
        # by itself. Found on the same real flask-5014 instance: a 16.6k
        # -token top-3 file consumed essentially all of an 8000-token
        # budget (audit log: "forced ... L4 ... cost=7160"), leaving only
        # two other items degraded to L3/L0 and dropping every other
        # candidate -- including, concretely, the file the fix actually
        # belonged in, ranked just outside the top 3. A per-item cap
        # (top_k_max_item_fraction, floored by top_k_max_item_floor so it
        # doesn't also break small budgets -- see the next test) fixes
        # this by capping any single forced item's size, leaving room for
        # the rest of the candidate pool.
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            # A huge, highly relevant file that would naturally rank #1.
            store.add(
                title="huge.py", source="huge.py", kind="code",
                content="fix the widget bug here\n" + ("padding line\n" * 4000),
                importance=0.9,
            )
            # Plenty of other candidates that should still get *some* room.
            for i in range(40):
                store.add(
                    title=f"other-{i}.py", source=f"other-{i}.py", kind="code",
                    content="fix the widget bug here " * 5, importance=0.5,
                )
            compiler = ContextCompiler(store)
            result = compiler.compile("fix the widget bug", budget=8000)
            self.assertGreater(
                len(result.selections), 5,
                "one huge top-ranked file should not be able to crowd out nearly everything else",
            )

    def test_top_k_item_cap_does_not_shrink_small_budgets(self):
        # Regression test for the fix above overcorrecting: a *fraction*
        # -only cap (e.g. 25% of budget) shrinks to almost nothing at small
        # budgets and can force a file down from full text even when it
        # would otherwise fit outright. Confirmed on the synthetic 15-task
        # benchmark: several 150-250 token tasks regressed when the cap had
        # no floor. top_k_max_item_floor keeps the cap a no-op until a file
        # is actually large enough for it to matter.
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            target = store.add(
                title="small.py", source="small.py", kind="code",
                content="def fix_widget_bug():\n    return 42\n",
                importance=0.9,
            )
            compiler = ContextCompiler(store)
            result = compiler.compile("fix the widget bug", budget=150)
            target_selection = next(s for s in result.selections if s.item_id == target.id)
            self.assertEqual(
                target_selection.level,
                RenderLevel.L4,
                "a small file that easily fits should still reach full text at a small budget",
            )

    def test_graph_adjacent_test_file_is_rescued_into_candidate_pool(self):
        # Regression test: a bug report's vocabulary rarely overlaps with
        # its own regression test's vocabulary, so the lexical retriever
        # can drop a graph-adjacent test file from the candidate pool
        # entirely before scoring or forcing ever gets a chance to run --
        # confirmed on real SWE-bench instances (django, astropy), where
        # the actual fix-validating test ranked outside the retriever's
        # own top ~200. `compile`'s graph_test_ids forcing can only
        # promote a candidate's *render level*; it can't rescue an item
        # that was never a candidate. `_augment_with_graph_adjacent_tests`
        # fixes this by unioning in graph-adjacent test files before the
        # retriever's cutoff is applied.
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            widget = store.add(
                title="widget.py", source="widget.py", kind="code",
                content="fix the widget bug here " * 5, importance=0.9,
            )
            test_widget = store.add(
                title="test_widget.py", source="test_widget.py", kind="test",
                content="totally unrelated vocabulary about zebras and kites " * 5,
                dependencies=[".widget"], importance=0.5,
            )
            # Distractors that beat test_widget.py on pure lexical overlap
            # with the task, padding the pool past a tiny max_candidates so
            # the retriever alone would drop test_widget.py from the cut.
            for i in range(10):
                store.add(
                    title=f"distractor-{i}.py", source=f"distractor-{i}.py", kind="code",
                    content="widget bug widget bug widget " * 3, importance=0.4,
                )
            config = CompilerConfig(max_candidates=3)
            compiler = ContextCompiler(store, config=config)
            candidates = compiler._candidates("fix the widget bug", config.max_candidates)
            candidate_ids = {c.id for c in candidates}
            self.assertIn(widget.id, candidate_ids)
            self.assertIn(
                test_widget.id,
                candidate_ids,
                "a test file one graph-hop from a top-ranked fix file should be pulled "
                "into the candidate pool even when the retriever's own ranking drops it",
            )

    def test_graph_test_forcing_uses_each_top_items_own_neighbors_not_the_union(self):
        # Regression test for the bug the fix above's own first attempt
        # introduced: forcing looped over each top-K item but checked
        # membership against the *union* of all top-K items' graph
        # neighbors combined, not that specific item's own neighbors. With
        # two top items that each have their own dedicated (and disjoint)
        # test file, that meant the higher-scoring test file got forced
        # for *both* items -- and the other item's real test never got
        # forced at all. Confirmed on a real instance (django-16082):
        # the correct test scored 0.4396 vs. a same-union-but-unrelated
        # test's 0.4433, and the union approach always picked the latter.
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            fix_a = store.add(
                title="alpha.py", source="alpha.py", kind="code",
                content="fix the widget bug in alpha here " * 5, importance=0.9,
            )
            fix_b = store.add(
                title="beta.py", source="beta.py", kind="code",
                content="fix the widget bug in beta here " * 5, importance=0.85,
            )
            # test_alpha's content overlaps *less* with the task than
            # test_beta's does, so a union-and-pick-highest-scoring
            # approach would wrongly pick test_beta for alpha.py's slot
            # too, leaving test_alpha unforced despite being the one
            # actually graph-adjacent to alpha.py.
            test_a = store.add(
                title="test_alpha.py", source="test_alpha.py", kind="test",
                content="alpha regression check", dependencies=[".alpha"], importance=0.3,
            )
            test_b = store.add(
                title="test_beta.py", source="test_beta.py", kind="test",
                content="fix the widget bug in beta here regression check " * 3,
                dependencies=[".beta"], importance=0.3,
            )
            config = CompilerConfig(top_k_full_text=2)
            compiler = ContextCompiler(store, config=config)
            result = compiler.compile("fix the widget bug", budget=8000)
            levels = {s.item_id: s.level for s in result.selections}
            self.assertGreaterEqual(
                levels.get(test_a.id, RenderLevel.L0),
                RenderLevel.L3,
                "test_alpha.py is the test actually adjacent to top-item alpha.py and "
                "should be forced to at least L3 regardless of test_beta.py's score",
            )
            self.assertGreaterEqual(
                levels.get(test_b.id, RenderLevel.L0),
                RenderLevel.L3,
                "test_beta.py should still be forced too, via beta.py's own slot",
            )

    def test_reversible_expand(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ContextStore(Path(tmp) / "ctx.db")
            item = store.add(title="doc", content="alpha beta gamma", kind="doc")
            compiler = ContextCompiler(store)
            expanded = compiler.expand(item.id, level="L4")
            self.assertIn("alpha beta gamma", expanded["text"])
            self.assertEqual(expanded["item"]["content"], "alpha beta gamma")


if __name__ == "__main__":
    unittest.main()
